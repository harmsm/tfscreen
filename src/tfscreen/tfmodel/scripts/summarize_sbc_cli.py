"""
tfs-summarize-sbc — compute SBC rank statistics from a directory of runs.

Scans *sbc_dir* for ``*_ground_truth.h5`` / ``*_posterior.h5`` pairs produced
by ``tfs-sample-prior`` and ``tfs-sample-posterior``, computes posterior rank
statistics, and writes summary outputs.
"""

from tfscreen.tfmodel.analysis.sbc import summarize_sbc
from tfscreen.util.cli import generalized_main


def main():
    generalized_main(
        summarize_sbc,
        manual_arg_types={"sbc_dir": str, "out_prefix": str},
    )


if __name__ == "__main__":
    main()
