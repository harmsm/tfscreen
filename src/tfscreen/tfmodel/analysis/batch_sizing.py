"""Estimate a safe genotype_batch_size for predict() from tensor shapes and
available device memory, so CLIs don't require hand-tuning after an OOM.
"""

import os

import jax

from tfscreen.tfmodel.model_orchestrator import FLOAT_DTYPE

# Empirically calibrated on CPU (measuring RSS delta of real predict() calls
# at genotype counts of 500/2000/5000 against the raw tensor-byte formula)
# to absorb overhead not captured by the raw tensor-byte count: the per-site
# quantile copy, the host-transfer copy of posterior samples, and (on GPU)
# XLA compilation/workspace buffers. The measured marginal cost per genotype
# was ~6.5x the raw formula; this is rounded up for headroom, since GPU-side
# overhead (not measurable on this CPU-only calibration) is expected to be
# higher. Lower this if a genotype_batch_size sized with it still OOMs on GPU.
_DEFAULT_OVERHEAD_MULTIPLIER = 6.5

# Fraction of total device memory to treat as usable budget, leaving
# headroom for the host process, other tensors already resident, and the
# unmodeled overhead above.
_DEFAULT_SAFETY_FRACTION = 0.6


def _dtype_itemsize():
    """Byte size of the float dtype used for prediction tensors."""
    return jax.numpy.dtype(FLOAT_DTYPE).itemsize


def estimate_bytes_per_genotype(orchestrator,
                                 predict_sites,
                                 num_marginal_samples=None,
                                 overhead_multiplier=_DEFAULT_OVERHEAD_MULTIPLIER):
    """
    Estimate peak bytes of device memory consumed per genotype during
    predict().

    Each genotype's row in the dense TensorManager tensor carries the full
    cross-product of every other tensor dimension (replicate, time,
    condition_pre, condition_sel, titrant_name, titrant_conc). predict()
    additionally holds one such array per requested predict_sites entry,
    stacked over the posterior-sample axis.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Orchestrator whose growth_tm defines the tensor shape. genotype is
        assumed to be the last dimension in growth_tm.tensor_shape.
    predict_sites : list of str
        Sites requested from predict().
    num_marginal_samples : int or None, optional
        Number of posterior draws run through the model for quantile
        computation. If None, the full posterior draw count is unknown here,
        so callers should pass the resolved value (or an upper bound).
    overhead_multiplier : float, optional
        Multiplier absorbing unmodeled overhead (quantile/host copies, XLA
        workspace). Default calibrated value.

    Returns
    -------
    float
        Estimated bytes of device memory per genotype.
    """
    tensor_shape = orchestrator.growth_tm.tensor_shape
    other_axes_product = 1
    for size in tensor_shape[:-1]:
        other_axes_product *= size

    n_samples = num_marginal_samples if num_marginal_samples is not None else 1
    n_sites = len(predict_sites) if predict_sites else 1

    bytes_per_genotype = (
        other_axes_product
        * n_samples
        * n_sites
        * _dtype_itemsize()
        * overhead_multiplier
    )

    return bytes_per_genotype


def get_available_memory_bytes(safety_fraction=_DEFAULT_SAFETY_FRACTION):
    """
    Estimate a usable memory budget in bytes for the current JAX default
    device.

    On GPU/TPU backends, uses jax.devices()[0].memory_stats()['bytes_limit'],
    which reflects total device memory capacity (not current usage) and is
    reliable regardless of XLA's preallocation behavior. On CPU, or when
    memory_stats() is unsupported (returns None, as it does on the CPU
    backend), falls back to total system RAM via os.sysconf.

    Parameters
    ----------
    safety_fraction : float, optional
        Fraction of total memory to treat as usable, leaving headroom for
        other processes and unmodeled overhead. Default 0.7.

    Returns
    -------
    int
        Usable memory budget in bytes.
    """
    device = jax.devices()[0]
    stats = device.memory_stats()

    if stats is not None and "bytes_limit" in stats:
        total_bytes = stats["bytes_limit"]
    else:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    return int(total_bytes * safety_fraction)


def estimate_genotype_batch_size(orchestrator,
                                  predict_sites,
                                  num_marginal_samples=None,
                                  safety_fraction=_DEFAULT_SAFETY_FRACTION,
                                  overhead_multiplier=_DEFAULT_OVERHEAD_MULTIPLIER):
    """
    Estimate a genotype_batch_size that keeps predict() under the available
    memory budget.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Orchestrator used to derive per-genotype memory cost.
    predict_sites : list of str
        Sites requested from predict().
    num_marginal_samples : int or None, optional
        Number of posterior draws run through the model. See
        estimate_bytes_per_genotype.
    safety_fraction : float, optional
        Fraction of total device memory to treat as usable. Default 0.7.
    overhead_multiplier : float, optional
        Multiplier absorbing unmodeled overhead. Default calibrated value.

    Returns
    -------
    int
        Estimated genotype batch size, clamped to at least 1.
    """
    bytes_per_genotype = estimate_bytes_per_genotype(
        orchestrator,
        predict_sites,
        num_marginal_samples=num_marginal_samples,
        overhead_multiplier=overhead_multiplier,
    )
    available_bytes = get_available_memory_bytes(safety_fraction=safety_fraction)

    batch_size = int(available_bytes // bytes_per_genotype)

    return max(batch_size, 1)
