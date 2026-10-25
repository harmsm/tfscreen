"""
tfs-summarize-calibration — posterior calibration across a simulation grid.

Collects the ``tfs-summarize-fit`` calibration inputs from every run of a
``tfs-setup-sim-grid`` grid, computes coverage / PIT / interval width /
accuracy per run, quantity and stratum, averages them over replicates within
each arm, and (with ``--baseline``) pairs each run with the baseline run fit
to the same simulated data.  See ``tfscreen.analysis.calibration_grid``.

Example (the guide-calibration grid)::

    tfs-summarize-calibration guide_grid --out_prefix calib/guide \\
        --baseline guide_type=component guide_rank=None \\
        --facet_by batch_size theta_growth_noise_model
"""

from tfscreen.analysis.calibration_grid import summarize_calibration
from tfscreen.util.cli import generalized_main


def main():
    generalized_main(summarize_calibration,
                     manual_arg_types={"grid_dir": str,
                                       "out_prefix": str,
                                       "summary_subdir": str,
                                       "replicate_keys": str,
                                       "baseline": str,
                                       "facet_by": str,
                                       "plot_quantity": str,
                                       "regime_eps": float},
                     manual_arg_nargs={"replicate_keys": "+",
                                       "baseline": "+",
                                       "facet_by": "+"})


if __name__ == "__main__":
    main()
