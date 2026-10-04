import os
import dill
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.util.cli.generalized_main import generalized_main

# Per-observation growth sites, dropped by skip_growth_observations.
GROWTH_OBSERVATION_SITES = ("growth_pred", "growth_obs")


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
                     map_point=False,
                     skip_growth_observations=False,
                     laplace_blocks=False,
                     laplace_shared=False,
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
    map_point : bool, optional
        For a MAP checkpoint, write the MAP point itself (one sample, no
        Hessian) instead of a Laplace posterior (default False). The Laplace
        needs the full Hessian, O(D^2) memory, which is out of reach on a
        full library (millions of parameters); the point estimate is not.
        Ignored for SVI and NUTS checkpoints.
    skip_growth_observations : bool, optional
        Leave the per-observation growth sites (``growth_pred``,
        ``growth_obs``) out of the file (default False). They are most of its
        size (94% on a 300-genotype subset; about 100 GB per site at 500
        samples on a 200,000-genotype library), and ``tfs-predict-growth``
        recomputes growth from the parameter samples rather than reading
        them.
    laplace_blocks : bool, optional
        For a MAP checkpoint, a per-genotype (block-diagonal) Laplace: the
        shared parameters (growth k and m, hyperparameters, tube offsets)
        stay at the MAP and each genotype's own parameters get their
        conditional Laplace (default False). It scales with the library,
        so it runs where the full Hessian cannot, but it leaves out the
        shared parameters' uncertainty unless ``laplace_shared``. Ignored
        for SVI and NUTS checkpoints and with ``map_point``.
    laplace_shared : bool, optional
        With ``laplace_blocks``, keep the shared parameters' uncertainty
        (default False): they are drawn from their Laplace marginal and
        each genotype from its conditional given them, which is the full
        Laplace. One more Hessian-vector product per shared parameter per
        genotype chunk.
    genotype_chunk_size : int or None, optional
        Genotypes per Hessian-vector-product pass with ``laplace_blocks``
        (default None, all at once). Lower it on device OOM.
    """
    if laplace_shared and not laplace_blocks:
        raise ValueError("--laplace_shared needs --laplace_blocks")
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

        if is_map and map_point:
            print("Detected MAP checkpoint. Writing the MAP point "
                  "(no Laplace)...", flush=True)
            ri.get_map_posteriors(map_params=chk_params,
                                  out_prefix=ri_prefix,
                                  forward_batch_size=forward_batch_size,
                                  sites_to_save=sites_to_save)
        elif is_map:
            # MAP checkpoint: Hessian-based Laplace approximation.
            print("Detected MAP checkpoint. Drawing "
                  f"{'per-genotype ' if laplace_blocks else ''}Laplace "
                  "posterior samples...", flush=True)
            ri.get_laplace_posteriors(
                map_params=chk_params,
                out_prefix=ri_prefix,
                num_posterior_samples=num_posterior_samples,
                sampling_batch_size=sampling_batch_size,
                forward_batch_size=forward_batch_size,
                hessian_chunk_size=hessian_chunk_size,
                sites_to_save=sites_to_save,
                block_genotypes=laplace_blocks,
                genotype_chunk_size=genotype_chunk_size,
                block_shared=laplace_shared,
            )
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
    generalized_main(sample_posterior,
                     manual_arg_types={"seed": int,
                                       "num_posterior_samples": int,
                                       "sampling_batch_size": int,
                                       "forward_batch_size": int,
                                       "hessian_chunk_size": int,
                                       "genotype_chunk_size": int})


if __name__ == "__main__":
    main()
