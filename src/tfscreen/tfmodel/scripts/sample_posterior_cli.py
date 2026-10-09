import os

import dill
import numpy as np
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.util.cli.generalized_main import generalized_main

# Per-observation growth sites, dropped by skip_growth_observations.
GROWTH_OBSERVATION_SITES = ("growth_pred", "growth_obs")

LAPLACE_CHOICES = ("auto", "full", "arrowhead", "blocks", "point")

# Above this many MAP parameters, --laplace auto uses the arrowhead Laplace.
# The full Laplace's dense Hessian is O(D^2) memory (20,000 parameters is
# 1.6 GB in float32 and 3.2 GB in float64) and O(D^3) to factor; the
# arrowhead gives the same Gaussian when the genotypes do not couple.
DEFAULT_LAPLACE_MAX_PARAMS = 20000


def resolve_laplace(laplace, num_params, laplace_max_params):
    """
    Pick the Laplace variant for a MAP checkpoint.

    Parameters
    ----------
    laplace : str
        One of ``LAPLACE_CHOICES``.
    num_params : int
        Number of unconstrained MAP parameters.
    laplace_max_params : int
        Largest model for which ``auto`` uses the full Laplace.

    Returns
    -------
    str
        ``full``, ``arrowhead``, ``blocks`` or ``point``.
    """
    if laplace not in LAPLACE_CHOICES:
        raise ValueError(f"laplace must be one of {LAPLACE_CHOICES}; got "
                         f"{laplace!r}.")
    if laplace != "auto":
        return laplace
    return "full" if num_params <= laplace_max_params else "arrowhead"


class _AllSitesExcept:
    """A ``sites_to_save`` that holds every site name except ``skip``."""

    def __init__(self, skip):
        self.skip = frozenset(skip)

    def __contains__(self, site):
        return site not in self.skip


def sample_posterior(config_file,
                     checkpoint_file,
                     out_prefix="tfs_posterior",
                     seed=0,
                     num_posterior_samples=10000,
                     sampling_batch_size=100,
                     forward_batch_size=512,
                     hessian_chunk_size=64,
                     laplace="auto",
                     laplace_max_params=DEFAULT_LAPLACE_MAX_PARAMS,
                     skip_growth_observations=False,
                     genotype_chunk_size=None):
    """
    Draw posterior samples from an existing MAP, SVI, or NUTS checkpoint.

    Three checkpoint types are handled automatically:

    1. NUTS checkpoint: runs the forward model over the saved MCMC samples and
       writes posterior predictives.
    2. MAP checkpoint (AutoDelta guide): forms a Laplace (Gaussian) approximation
       via the Hessian of the log-joint at the MAP point, then draws samples.
    3. SVI checkpoint (component guide): restores the fitted variational state
       and draws posterior samples directly from the guide.

    The .h5 file written by this command can be passed directly to
    tfs-predict-growth, tfs-predict-theta, and tfs-extract-params to obtain
    full posterior uncertainty (quantile columns) on any quantity of interest.
    This is the recommended path for obtaining uncertainty estimates from a
    MAP-fitted model: fit with tfs-fit-model, sample here, then predict.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file used when fitting the model.
    checkpoint_file : str
        Path to the checkpoint .pkl file produced by tfs-fit-model or
        tfs-prefit-calibration.
    out_prefix : str, optional
        Prefix for the posterior output file (default 'tfs_posterior').
        Posterior samples are written to {out_prefix}.h5.
    seed : int, optional
        Random seed used when constructing the RunInference object (default 0).
        For SVI and MAP checkpoints the PRNG state is restored from the
        checkpoint, so this value has no effect on the posterior samples.
    num_posterior_samples : int, optional
        Number of posterior samples to draw (default 10000). Not used for NUTS
        checkpoints (all MCMC samples are used directly).
    sampling_batch_size : int, optional
        Parameter sampling batch size (default 100). Not used for NUTS.
    forward_batch_size : int, optional
        Forward-model batch size for posterior predictives (default 512).
    hessian_chunk_size : int, optional
        Number of Hessian rows computed per device batch for MAP checkpoints
        (default 64). Reduce if the Hessian computation hits device OOM.
    laplace : str, optional
        How a MAP checkpoint becomes a posterior (ignored for SVI and NUTS):

        - ``auto`` (default): ``full`` up to ``laplace_max_params``
          parameters, ``arrowhead`` above.
        - ``full``: the Laplace from the dense Hessian, O(D^2) memory. Out of
          reach on a full library (millions of parameters).
        - ``arrowhead``: the full Laplace computed per genotype. The Hessian
          is an arrowhead (per-genotype blocks, their coupling to the shared
          parameters, the shared block); the shared parameters (growth k and
          m, hyperparameters, tube offsets) are drawn from their marginal and
          each genotype from its conditional. Runs on a full library. A
          negative direction of the shared block's Schur complement is held
          at the MAP; those directions are written to
          ``{out_prefix}_held_directions.csv``, since they have no interval.
        - ``blocks``: per-genotype blocks only, the shared parameters held at
          the MAP. Leaves out the k/m uncertainty.
        - ``point``: the MAP point itself, one sample, no Hessian.
    laplace_max_params : int, optional
        Largest number of MAP parameters for which ``auto`` picks the full
        Laplace (default 20000).
    skip_growth_observations : bool, optional
        Leave the per-observation growth sites (``growth_pred``,
        ``growth_obs``) out of the file (default False). They are most of its
        size (94% on a 300-genotype subset; about 100 GB per site at 500
        samples on a 200,000-genotype library), and ``tfs-predict-growth``
        recomputes growth from the parameter samples rather than reading
        them.
    genotype_chunk_size : int or None, optional
        Genotypes per Hessian-vector-product pass for the arrowhead and
        blocks Laplace (default None, all at once). Lower it on device OOM.
    """
    if laplace not in LAPLACE_CHOICES:
        raise ValueError(f"laplace must be one of {LAPLACE_CHOICES}; got "
                         f"{laplace!r}.")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{checkpoint_file}'. "
            "Run tfs-fit-model first to produce a checkpoint."
        )

    orchestrator, init_params = read_configuration(config_file)
    ri = RunInference(orchestrator, seed)

    with open(checkpoint_file, "rb") as f:
        chk_data = dill.load(f)

    # RunInference methods write {out_prefix}_posterior.h5; rename to {out_prefix}.h5
    # after each call so the output matches the documented convention.
    ri_prefix = f"{out_prefix}_tmp_posterior"
    sites_to_save = (_AllSitesExcept(GROWTH_OBSERVATION_SITES)
                     if skip_growth_observations else None)

    if "mcmc_samples" in chk_data:
        # NUTS checkpoint: regenerate posteriors from saved samples.
        print("Detected NUTS checkpoint. Writing posterior predictives...", flush=True)
        ri.get_nuts_posteriors(chk_data["mcmc_samples"],
                               out_prefix=ri_prefix,
                               forward_batch_size=forward_batch_size,
                               sites_to_save=sites_to_save)
    else:
        # Checkpoints record the guide that wrote them.  Older ones do not;
        # there the only autoguide was AutoDelta (MAP), recognizable by its
        # "{site}_auto_loc" parameter names.
        guide_type = chk_data.get("guide_type")
        if guide_type is None or guide_type == "delta":
            temp_svi = ri.setup_svi(guide_type="delta")
            chk_params = temp_svi.optim.get_params(chk_data["svi_state"].optim_state)
            is_map = (guide_type == "delta"
                      or any("_auto_loc" in k for k in chk_params))
        else:
            is_map = False

        if is_map:
            num_params = sum(int(np.size(v)) for k, v in chk_params.items()
                             if k.endswith("_auto_loc"))
            method = resolve_laplace(laplace, num_params, laplace_max_params)
            print(f"Detected MAP checkpoint ({num_params} parameters); "
                  f"Laplace: {method}"
                  + (f" (auto, threshold {laplace_max_params})"
                     if laplace == "auto" else ""), flush=True)

        if is_map and method == "point":
            ri.get_map_posteriors(map_params=chk_params,
                                  out_prefix=ri_prefix,
                                  forward_batch_size=forward_batch_size,
                                  sites_to_save=sites_to_save)
        elif is_map:
            ri.get_laplace_posteriors(
                map_params=chk_params,
                out_prefix=ri_prefix,
                num_posterior_samples=num_posterior_samples,
                sampling_batch_size=sampling_batch_size,
                forward_batch_size=forward_batch_size,
                hessian_chunk_size=hessian_chunk_size,
                sites_to_save=sites_to_save,
                block_genotypes=method in ("arrowhead", "blocks"),
                genotype_chunk_size=genotype_chunk_size,
                block_shared=method == "arrowhead",
            )
            held = getattr(ri, "held_shared_directions", None)
            if method == "arrowhead" and held is not None:
                held_file = f"{out_prefix}_held_directions.csv"
                held.to_csv(held_file, index=False)
                print(f"Held shared directions written to {held_file}",
                      flush=True)
        else:
            # SVI checkpoint: rebuild the guide object then restore the saved
            # variational state directly — no optimization loop needed.
            print("Detected SVI checkpoint. Drawing variational posterior samples...", flush=True)
            svi_obj, svi_state = ri.restore_svi_from_checkpoint(
                checkpoint_file, init_params=init_params
            )
            ri.get_posteriors(svi=svi_obj,
                              svi_state=svi_state,
                              out_prefix=ri_prefix,
                              num_posterior_samples=num_posterior_samples,
                              sampling_batch_size=sampling_batch_size,
                              forward_batch_size=forward_batch_size,
                              sites_to_save=sites_to_save)

    src = f"{ri_prefix}_posterior.h5"
    dst = f"{out_prefix}.h5"
    os.rename(src, dst)
    print(f"Posterior samples written to {dst}", flush=True)


def main():
    return generalized_main(sample_posterior,
                            manual_arg_types={"seed": int,
                                              "num_posterior_samples": int,
                                              "sampling_batch_size": int,
                                              "forward_batch_size": int,
                                              "hessian_chunk_size": int,
                                              "laplace_max_params": int,
                                              "genotype_chunk_size": int})


if __name__ == "__main__":
    main()
