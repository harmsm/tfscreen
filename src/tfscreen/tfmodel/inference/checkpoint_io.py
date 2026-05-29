import os
import dill


def resolve_param_file(param_file, gm, out_prefix):
    """
    Return a posterior .h5 path, converting a MAP checkpoint on the fly if needed.

    If ``param_file`` ends with ``.pkl``, it is treated as a MAP (AutoDelta)
    checkpoint: the guide parameters are extracted, a 1-sample posterior is
    written to ``{out_prefix}_map_posterior.h5`` via
    :meth:`~tfscreen.tfmodel.inference.run_inference.RunInference.get_map_posteriors`,
    and the path to that file is returned.

    NUTS checkpoints (containing ``mcmc_samples``) and SVI checkpoints (guide
    params without ``_auto_loc`` keys) both raise ``ValueError`` with a helpful
    message directing the user to run ``tfs-sample-posterior`` first.

    If ``param_file`` does not end with ``.pkl`` it is returned unchanged.

    Parameters
    ----------
    param_file : str
        Path to a posterior .h5 file or a MAP checkpoint .pkl file.
    gm : ModelOrchestrator
        Already-loaded model instance (from ``read_configuration``).
    out_prefix : str
        Output prefix for the calling script.  The intermediate map posterior
        is written to ``{out_prefix}_map_posterior.h5``.

    Returns
    -------
    str
        Path to a posterior .h5 file ready for use by extraction functions.
    """
    if not param_file.endswith(".pkl"):
        return param_file

    if not os.path.isfile(param_file):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{param_file}'"
        )

    with open(param_file, "rb") as f:
        chk_data = dill.load(f)

    if "mcmc_samples" in chk_data:
        raise ValueError(
            f"'{param_file}' is a NUTS checkpoint. "
            "Run tfs-sample-posterior first to generate a posterior .h5 file."
        )

    from tfscreen.tfmodel.inference.run_inference import RunInference

    ri = RunInference(gm, seed=0)
    temp_svi = ri.setup_svi(guide_type="delta")
    map_params = temp_svi.optim.get_params(chk_data["svi_state"].optim_state)

    if not any(k.endswith("_auto_loc") for k in map_params):
        raise ValueError(
            f"'{param_file}' is an SVI (variational) checkpoint. "
            "Run tfs-sample-posterior first to generate a posterior .h5 file."
        )

    h5_prefix = f"{out_prefix}_map"
    print(f"MAP checkpoint detected. Writing 1-sample posterior to "
          f"{h5_prefix}_posterior.h5...", flush=True)
    ri.get_map_posteriors(map_params, out_prefix=h5_prefix)
    return f"{h5_prefix}_posterior.h5"
