import os
import dill
import numpy as np
from tfscreen.analysis.hierarchical.growth_model.configuration_io import read_configuration
from tfscreen.analysis.hierarchical.run_inference import RunInference
from tfscreen.analysis.hierarchical.growth_model.extraction import extract_parameters
from tfscreen.util.cli import generalized_main, read_lines


def extract_params(config_file, checkpoints_file, out_prefix="tfs_params"):
    """
    Extract MAP point estimates from one or more checkpoint files.

    Reads each checkpoint listed in ``checkpoints_file``, converts the
    optimized guide parameters to constrained natural-parameter space, and
    writes parameter values as CSV files.  The output format mirrors
    ``tfs-param-quantiles``: one CSV per parameter group named
    ``{out_prefix}_{param_name}.csv``.  Instead of quantile columns
    (``median``, ``lower_95``, ...) there is one column per checkpoint named
    ``step_{N}`` where N is the step number recorded in that checkpoint.

    Useful for two purposes:

    1. Inspecting the MAP point estimate directly — pass a single final
       checkpoint.
    2. Tracking parameter stability across training — pass multiple
       checkpoints from the ``checkpoints/`` subdirectory written by
       ``tfs-fit-model``.

    Only MAP (AutoDelta) checkpoints are supported.  SVI and NUTS checkpoints
    will raise a ``ValueError`` with a helpful message.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file used when fitting the model.
    checkpoints_file : str
        Path to a plain-text file listing one checkpoint .pkl path per line
        (blank lines and lines starting with ``#`` are ignored).
    out_prefix : str, optional
        Prefix for output CSV files.  Each parameter group is written to
        ``{out_prefix}_{param_name}.csv``.  Default ``'tfs_params'``.
    """
    gm, _ = read_configuration(config_file)
    ri = RunInference(gm, seed=0)

    checkpoint_paths = read_lines(checkpoints_file)
    if not checkpoint_paths:
        raise ValueError(f"No checkpoint paths found in '{checkpoints_file}'")

    all_dfs = {}  # param_name → list of single-column DataFrames (one per checkpoint)

    for ckpt_path in checkpoint_paths:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")

        print(f"Loading {ckpt_path}...", flush=True)
        with open(ckpt_path, "rb") as f:
            chk_data = dill.load(f)

        if "mcmc_samples" in chk_data:
            raise ValueError(
                f"'{ckpt_path}' is a NUTS checkpoint. "
                "tfs-extract-params only supports MAP (AutoDelta) checkpoints."
            )

        step = chk_data.get("current_step", 0)
        col_name = f"step_{step}"

        temp_svi = ri.setup_svi(guide_type="delta")
        map_params = temp_svi.optim.get_params(chk_data["svi_state"].optim_state)

        if not any(k.endswith("_auto_loc") for k in map_params):
            raise ValueError(
                f"'{ckpt_path}' is an SVI (variational) checkpoint. "
                "tfs-extract-params only supports MAP (AutoDelta) checkpoints."
            )

        constrained = ri._map_params_to_constrained(map_params)
        # Wrap each value with a leading sample dim so extract_parameters sees
        # a 1-sample "posterior".  np.quantile of a 1-sample array at q=0.5
        # returns the single value, which is exactly the MAP point estimate.
        posteriors = {k: np.expand_dims(np.asarray(v), 0)
                      for k, v in constrained.items()}

        param_dfs = extract_parameters(gm, posteriors, q_to_get={col_name: 0.5})

        for p_name, df in param_dfs.items():
            all_dfs.setdefault(p_name, []).append(df)

    # Merge all per-checkpoint DataFrames into one CSV per parameter group.
    print(f"Writing parameter CSVs to {out_prefix}_*.csv...", flush=True)
    for p_name, df_list in all_dfs.items():
        if len(df_list) == 1:
            merged = df_list[0]
        else:
            step_cols = [c for c in df_list[0].columns if c.startswith("step_")]
            id_cols = [c for c in df_list[0].columns if c not in step_cols]
            merged = df_list[0]
            for df in df_list[1:]:
                new_step = [c for c in df.columns if c.startswith("step_")][0]
                merged = merged.merge(df[id_cols + [new_step]], on=id_cols, how="left")

        out_file = f"{out_prefix}_{p_name}.csv"
        merged.to_csv(out_file, index=False)
        print(f"  Wrote {out_file}", flush=True)

    print("Done.", flush=True)


def main():
    generalized_main(extract_params,
                     manual_arg_types={"checkpoints_file": str})


if __name__ == "__main__":
    main()
