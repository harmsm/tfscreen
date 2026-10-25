"""
tfs-calibrate-od600: fit an OD600-to-CFU calibration from a lab's own data.

See ``tfscreen.process_raw.od600`` for the method and the input formats.
"""

import os

import numpy as np

from tfscreen.process_raw import od600 as od600_cal
from tfscreen.util.cli import generalized_main
from tfscreen.util.io import read_dataframe


def _plot(calibration, noise_table, plate_table, pdf_file):
    """Dilution series, fitted curve, and the calibrated CFU's relative SD."""
    from matplotlib import pyplot as plt
    from tfscreen.plot import default_styles  # noqa: F401 (applies rcParams)

    cal = od600_cal.read_calibration(calibration)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].errorbar(noise_table["dilution"], noise_table["mean"],
                   yerr=noise_table["sd"], fmt="o", mfc="none", capsize=3)
    ax[0].axhline(cal["detection_threshold"], ls="--", color="gray")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("dilution")
    ax[0].set_ylabel("OD600 (mean +/- SD)")

    lo, hi = cal["calibrated_od600_range"]
    grid = np.linspace(cal["detection_threshold"], hi * 1.05, 200)
    cfu, sd, _ = od600_cal.od600_to_cfu_per_mL(grid, cal)
    ax[1].fill_between(grid, cfu - sd, cfu + sd, color="lightgray")
    ax[1].plot(grid, cfu, color="black")
    ax[1].errorbar(plate_table["od600"], plate_table["cfu_per_mL"],
                   yerr=plate_table["cfu_per_mL_std"], fmt="o", mfc="none",
                   capsize=3, color="tab:red")
    ax[1].axvline(cal["detection_threshold"], ls="--", color="gray")
    ax[1].set_xlabel("OD600")
    ax[1].set_ylabel("CFU/mL")

    curve_sd, reading_sd = od600_cal.cfu_per_mL_error_components(grid, cal)
    ax[2].plot(grid, curve_sd / cfu, label="curve (shared by all tubes)")
    ax[2].plot(grid, reading_sd / cfu, label="one reading")
    ax[2].plot(grid, sd / cfu, color="black", label="total")
    ax[2].axvspan(lo, hi, color="lightgray", alpha=0.4, lw=0)
    ax[2].set_xlabel("OD600")
    ax[2].set_ylabel("SD / CFU")
    ax[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(pdf_file)
    plt.close(fig)


def calibrate_od600(replicate_file,
                    plate_count_file,
                    out_prefix="tfs_od600",
                    degree=2,
                    pipette_rel_error=0.02,
                    detection_threshold=None):
    """
    Fit an OD600-to-CFU/mL calibration and write it for the pipeline.

    A calibration depends on the plate reader, plate, volume and strain, so
    each lab makes its own from two small experiments done with the same
    handling as production samples.

    Parameters
    ----------
    replicate_file : str
        CSV/TSV/Excel of repeated OD600 readings of a dilution series, one
        row per reading: ``dilution`` (relative to the undiluted culture),
        ``od600``. Gives the reading noise (the largest relative SD across
        dilutions) and the detection threshold (midway between the mean
        readings of the two most dilute samples).
    plate_count_file : str
        CSV/TSV/Excel of plate counts of cultures whose OD600 was read, one
        row per plate: ``od600``, ``colonies``, ``dilution`` (total dilution
        factor before plating), ``plated_volume_mL``, ``num_dilutions`` and
        ``plating_steps`` (for the pipetting error).
    out_prefix : str, optional
        Writes ``{out_prefix}.yaml`` (the calibration, read by the rest of
        the pipeline), ``{out_prefix}_replicates.csv`` (reading noise by
        dilution), ``{out_prefix}_plate_counts.csv`` (CFU/mL, its SD, the
        fitted curve and the standardized residual) and ``{out_prefix}.pdf``.
    degree : int, optional
        Degree of the polynomial of CFU/mL in OD600 (default 2).
    pipette_rel_error : float, optional
        Relative error of one pipetting step (default 0.02).
    detection_threshold : float, optional
        Override the detection threshold estimated from the dilution series.
    """
    replicate_df = read_dataframe(replicate_file)
    plate_df = read_dataframe(plate_count_file)

    calibration, noise_table, plate_table = od600_cal.calibrate(
        replicate_df, plate_df, degree=degree,
        pipette_rel_error=pipette_rel_error, threshold=detection_threshold)

    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_dir, exist_ok=True)

    yaml_file = f"{out_prefix}.yaml"
    od600_cal.write_calibration(
        calibration, yaml_file,
        source={"replicate_file": os.path.basename(replicate_file),
                "plate_count_file": os.path.basename(plate_count_file)})
    noise_table.to_csv(f"{out_prefix}_replicates.csv", index=False)
    plate_table.to_csv(f"{out_prefix}_plate_counts.csv", index=False)
    _plot(calibration, noise_table, plate_table, f"{out_prefix}.pdf")

    coef = ", ".join(f"{c:.4g}" for c in calibration["coefficients"])
    lo, hi = calibration["calibrated_od600_range"]
    print(f"cfu/mL = polynomial in OD600 with coefficients [{coef}] "
          f"(reduced chi2 {calibration['chi2_reduced']:.3g}, "
          f"{calibration['num_plate_counts']} plates)")
    print(f"reading noise {100 * calibration['reading_rel_sd']:.2g}% of the "
          f"reading; detection threshold {calibration['detection_threshold']:.4g}; "
          f"calibrated OD600 range {lo:.3g}-{hi:.3g}")
    print(f"Wrote {yaml_file}")
    return calibration


def main():
    generalized_main(calibrate_od600,
                     manual_arg_types={"detection_threshold": float})


if __name__ == "__main__":
    main()
