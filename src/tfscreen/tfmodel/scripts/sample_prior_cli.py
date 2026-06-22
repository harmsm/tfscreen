import numpy as np
import h5py

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.analysis.prior_predictive import draw_prior, growth_df_from_prior
from tfscreen.util.cli.generalized_main import generalized_main


def _write_h5(path, samples_dict, num_draws):
    """Write a dict of numpy arrays to an HDF5 file."""
    with h5py.File(path, "w") as hf:
        for k, v in samples_dict.items():
            v = np.array(v)
            # Choose a chunk size that keeps each chunk under ~4 MB.
            trailing = int(np.prod(v.shape[1:])) if v.ndim > 1 else 1
            item_bytes = v.dtype.itemsize
            safe_first = max(1, (4 * 1024 * 1024) // max(trailing * item_bytes, 1))
            chunks = (min(v.shape[0], safe_first),) + v.shape[1:]
            hf.create_dataset(k, data=v, chunks=chunks,
                              compression="gzip", compression_opts=4)
        hf.attrs["num_samples"] = num_draws
        hf.flush()


def sample_prior(config_file,
                 out_prefix="tfs_prior",
                 num_datasets=1,
                 noise=True,
                 seed=0):
    """
    Draw synthetic datasets from the model prior.

    For each dataset, two files are written:

    ``<out_prefix>_NNN_growth.csv``
        Synthetic growth DataFrame with ``ln_cfu`` replaced by prior
        predictions.  When ``--noise`` is set (the default), observation
        noise drawn from ``Normal(0, ln_cfu_std)`` is added.  This file
        can be passed directly to ``tfs-fit-model`` as training data.

    ``<out_prefix>_NNN_ground_truth.h5``
        The latent parameters used to generate that dataset, in the same
        HDF5 format as a ``tfs-sample-posterior`` output.  Pass this to
        ``tfs-predict-growth`` or ``tfs-extract-params`` to inspect the
        ground-truth values, or compare it to the fitted posterior for
        parameter recovery checks.

    When ``--num_datasets 1`` (the default) the ``_NNN`` index is still
    included in the filenames for consistency.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file (same one used for
        ``tfs-fit-model``).
    out_prefix : str, optional
        Prefix for all output files.  Default ``'tfs_prior'``.
    num_datasets : int, optional
        Number of independent synthetic datasets to generate.  Default 1.
        Set to a larger value (e.g. 100) for Simulation-Based Calibration.
    noise : bool, optional
        If True (default), add ``Normal(0, ln_cfu_std)`` noise to each
        synthetic observation.  Set to False to get the deterministic
        prior predictions.
    seed : int, optional
        Base random seed.  Dataset ``i`` uses ``seed + i`` to ensure
        independence.  Default 0.
    """
    print(f"Loading configuration from {config_file}...", flush=True)
    orchestrator, _ = read_configuration(config_file)

    width = len(str(num_datasets))

    for i in range(num_datasets):
        tag = str(i).zfill(max(width, 3))
        print(f"Drawing prior sample {i + 1}/{num_datasets}...", flush=True)

        predictions, latent_params = draw_prior(orchestrator, rng_key=seed + i, num_draws=1)

        rng = np.random.default_rng(seed + i) if noise else None
        growth_df = growth_df_from_prior(orchestrator, latent_params, draw_idx=0, noise_rng=rng)

        csv_path = f"{out_prefix}_{tag}_growth.csv"
        growth_df.to_csv(csv_path, index=False)
        print(f"  Wrote synthetic growth data → {csv_path}", flush=True)

        h5_path = f"{out_prefix}_{tag}_ground_truth.h5"
        _write_h5(h5_path, latent_params, num_draws=1)
        print(f"  Wrote ground-truth parameters → {h5_path}", flush=True)

    print("Done.", flush=True)


def main():
    generalized_main(sample_prior,
                     manual_arg_types={"noise": bool,
                                       "num_datasets": int,
                                       "seed": int})


if __name__ == "__main__":
    main()
