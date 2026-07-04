"""
Per-genotype pinned dk_geno component.

Unlike ``dk_geno.fixed`` (always exactly zero for every genotype) or
``dk_geno.hierarchical_geno`` (learned per genotype from a pooled prior),
this component treats dk_geno as a fully known, deterministic quantity
supplied by the caller for a subset of genotypes, with every other
genotype defaulting to 0.

Typical use: some genotypes have an independently measured pleiotropic
growth effect (e.g. from an orthogonal assay). Pinning those values lets
downstream fits (in particular the linking-function pre-fit calibration
run by ``tfs-prefit-calibration``) use the *real* dk_geno for those
genotypes instead of forcing every genotype to 0, while still not trying
to *learn* dk_geno for the remaining genotypes from data that isn't
suited to it.

Values are supplied via ``ModelPriors.dk_geno_values``, a
``(num_genotype,)`` array indexed the same way as every other per-genotype
tensor in this codebase (i.e. ``data.batch_idx`` indexes into it). See
``build_dk_geno_values`` for how that array is assembled from a
``{genotype: dk_geno}`` mapping, and ``read_dk_geno_pins`` for reading that
mapping from a CSV file. Wiring from a config file happens in
``ModelOrchestrator`` (its ``dk_geno_pins_file`` constructor argument), not
here -- this module only consumes an already-built array.

There are no sampled latent variables:

- ``define_model`` registers zero ``pyro.sample`` sites and only emits a
  ``pyro.deterministic(name, ...)`` site for visibility.
- ``guide`` is a pure no-op (no ``pyro.param``, no ``pyro.sample``),
  keeping the model/guide sample sets symmetric.

Wildtype is always forced to 0 regardless of ``dk_geno_values`` content;
``build_dk_geno_values`` additionally rejects a nonzero pin for wildtype
outright, so this is defense in depth rather than the primary guard.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro as pyro
from flax.struct import dataclass

from tfscreen.tfmodel.data_class import GrowthData


_REQUIRED_COLUMNS = {"genotype", "dk_geno"}


@dataclass(frozen=True)
class ModelPriors:
    """
    JAX Pytree holding the (always-pinned) per-genotype dk_geno values.

    Attributes
    ----------
    dk_geno_values : jnp.ndarray
        Shape ``(num_genotype,)``, ordered to match the model's genotype
        axis (i.e. indexable by ``data.batch_idx``). Genotypes with no
        independently known dk_geno should be 0.0.
    """

    dk_geno_values: jnp.ndarray


def _compute_dk_geno(data: GrowthData, priors: ModelPriors) -> jnp.ndarray:
    """Shared per-genotype dk_geno lookup, with wildtype forced to 0."""
    dk_geno_per_genotype = priors.dk_geno_values[data.batch_idx]
    is_wt_mask = jnp.isin(data.batch_idx, data.wt_indexes)
    return jnp.where(is_wt_mask, 0.0, dk_geno_per_genotype)


def define_model(name: str,
                 data: GrowthData,
                 priors: ModelPriors) -> jnp.ndarray:
    """
    The pleiotropic effect of a genotype on growth rate independent of
    transcription factor occupancy, pinned to caller-supplied per-genotype
    values (0 for any genotype not explicitly pinned). Returns a full
    dk_geno tensor.

    Parameters
    ----------
    name : str
        The prefix for the ``pyro.deterministic`` site.
    data : GrowthData
        A Pytree (Flax dataclass) containing experimental data and
        metadata. This function primarily uses:
        - ``data.batch_idx`` : (jnp.ndarray) Index array mapping the
          current batch onto per-genotype parameters.
        - ``data.wt_indexes`` : (jnp.ndarray) Genotype indices treated as
          wildtype (always forced to dk_geno = 0).
    priors : ModelPriors
        Holds the pinned ``dk_geno_values``.

    Returns
    -------
    jnp.ndarray
        A tensor of pinned dk_geno values, expanded to match the shape of
        the observations via ``data.batch_idx``.
    """
    dk_geno_per_genotype = _compute_dk_geno(data, priors)

    pyro.deterministic(name, dk_geno_per_genotype)

    dk_geno = dk_geno_per_genotype[None, None, None, None, None, None, :]

    return dk_geno


def guide(name: str,
          data: GrowthData,
          priors: ModelPriors) -> jnp.ndarray:
    """
    Guide for the pinned dk_geno model.

    Since all values are fixed and deterministic, this guide does not
    register any learnable parameters or sample sites.
    """
    dk_geno_per_genotype = _compute_dk_geno(data, priors)

    dk_geno = dk_geno_per_genotype[None, None, None, None, None, None, :]

    return dk_geno


def get_hyperparameters():
    """
    Get default values for the model hyperparameters.

    Returns
    -------
    dict[str, Any]
        A placeholder ``dk_geno_values`` of shape ``(1,)``. Real usage
        wires in a properly-sized array via
        ``ModelOrchestrator(dk_geno_pins_file=...)``; this default only
        supports standalone construction (e.g. in tests).
    """
    return {"dk_geno_values": jnp.zeros(1, dtype=float)}


def get_guesses(name, data):
    """
    Get guess values for the model parameters.

    Returns
    -------
    dict[str, Any]
        An empty dictionary, as this model has no latent parameters.
    """
    return {}


def get_priors(dk_geno_values=None):
    """
    Construct a populated ``ModelPriors``.

    Parameters
    ----------
    dk_geno_values : array-like, optional
        Per-genotype dk_geno values, shape ``(num_genotype,)``. If
        omitted, the placeholder default from ``get_hyperparameters`` is
        used (shape ``(1,)`` -- not usable for a real fit).
    """
    hyperparameters = get_hyperparameters()
    if dk_geno_values is not None:
        hyperparameters["dk_geno_values"] = jnp.asarray(dk_geno_values, dtype=float)
    return ModelPriors(**hyperparameters)


def get_extract_specs(ctx):
    return [dict(
        input_df=ctx.growth_tm.df,
        params_to_get=["dk_geno"],
        map_column="map_genotype",
        get_columns=["genotype"],
        in_run_prefix="",
    )]


# ---------------------------------------------------------------------------
# Pins-file I/O -- used by ModelOrchestrator to build ``dk_geno_values``.
# ---------------------------------------------------------------------------

def read_dk_geno_pins(path):
    """
    Read a per-genotype dk_geno pin CSV.

    Parameters
    ----------
    path : str
        Path to a CSV with columns ``genotype`` and ``dk_geno``.

    Returns
    -------
    dict[str, float]
        ``{genotype: dk_geno}``.

    Raises
    ------
    ValueError
        If required columns are missing, unrecognised columns are
        present, or a genotype is listed more than once.
    """
    df = pd.read_csv(path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"'{path}' is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(_REQUIRED_COLUMNS)}"
        )

    unknown = set(df.columns) - _REQUIRED_COLUMNS
    if unknown:
        raise ValueError(
            f"'{path}' has unrecognised column(s): {sorted(unknown)}. "
            f"Expected columns: {sorted(_REQUIRED_COLUMNS)}"
        )

    genotypes = df["genotype"].astype(str)
    dup = genotypes[genotypes.duplicated()]
    if not dup.empty:
        raise ValueError(
            f"'{path}' lists the same genotype more than once: "
            f"{sorted(set(dup))}"
        )

    return dict(zip(genotypes, df["dk_geno"].astype(float)))


def build_dk_geno_values(pins, genotype_labels, wt_label="wt"):
    """
    Build a per-genotype dk_geno array ordered to match ``genotype_labels``.

    Parameters
    ----------
    pins : dict[str, float]
        Output of :func:`read_dk_geno_pins`.
    genotype_labels : Sequence[str]
        Genotype order used by the model's genotype axis (e.g.
        ``growth_tm.tensor_dim_labels[genotype_idx]``).
    wt_label : str, optional
        The label used for wildtype (default ``"wt"``).

    Returns
    -------
    np.ndarray
        Shape ``(len(genotype_labels),)``. Entries not present in
        ``pins`` are 0.0.

    Raises
    ------
    ValueError
        If ``pins`` references a genotype absent from
        ``genotype_labels``, or pins a non-zero value onto ``wt_label``.
    """
    label_to_idx = {str(g): i for i, g in enumerate(genotype_labels)}

    unknown = sorted(set(pins) - set(label_to_idx))
    if unknown:
        raise ValueError(
            f"dk_geno pins reference genotype(s) not present in the "
            f"growth data: {unknown}"
        )

    if wt_label in pins and float(pins[wt_label]) != 0.0:
        raise ValueError(
            f"dk_geno pins specify dk_geno={pins[wt_label]!r} for "
            f"'{wt_label}'; wildtype dk_geno must be exactly 0.0."
        )

    values = np.zeros(len(genotype_labels), dtype=float)
    for g, v in pins.items():
        values[label_to_idx[g]] = v

    return values
