
import pytest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tfscreen.plot.plot_theta_fits import plot_theta_fits
from tfscreen.plot import default_styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(genotypes=("wt",), titrant_names=("IPTG",),
             concs=(0, 0.1, 1.0, 10.0),
             with_std_band=False, with_95_band=False):
    """Return a minimal valid DataFrame for plot_theta_fits."""
    rows = []
    rng = np.random.default_rng(0)
    for geno in genotypes:
        for tname in titrant_names:
            for c in concs:
                row = {
                    "genotype": geno,
                    "titrant_name": tname,
                    "titrant_conc": c,
                    "theta_obs": rng.uniform(0.1, 0.9),
                    "theta_std": 0.05,
                    "q0.5": rng.uniform(0.1, 0.9),
                }
                if with_std_band:
                    row["q0.159"] = row["q0.5"] - 0.1
                    row["q0.841"] = row["q0.5"] + 0.1
                if with_95_band:
                    row["q0.025"] = row["q0.5"] - 0.2
                    row["q0.975"] = row["q0.5"] + 0.2
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

def test_returns_axes():
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df)
    assert ax is not None
    plt.close("all")


def test_xscale_is_log():
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df)
    assert ax.get_xscale() == "log"
    plt.close("all")


def test_ylabel_theta():
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df)
    assert "theta" in ax.get_ylabel().lower() or "θ" in ax.get_ylabel()
    plt.close("all")


# ---------------------------------------------------------------------------
# ax passthrough
# ---------------------------------------------------------------------------

def test_custom_ax_returned():
    df = _make_df()
    _, ax = plt.subplots()
    ret = plot_theta_fits(df, ax=ax)
    assert ret is ax
    plt.close("all")


def test_creates_own_ax_when_none():
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df, ax=None)
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# title
# ---------------------------------------------------------------------------

def test_title_set():
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df, title="My Title")
    assert ax.get_title() == "My Title"
    plt.close("all")


def test_no_title_by_default():
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df)
    assert ax.get_title() == ""
    plt.close("all")


# ---------------------------------------------------------------------------
# Zero-concentration floor
# ---------------------------------------------------------------------------

def test_zero_conc_does_not_crash():
    """Zero titrant_conc must not cause errors on the log-scale x-axis."""
    df = _make_df(concs=(0, 1.0, 10.0))
    plt.close("all")
    ax = plot_theta_fits(df)
    assert ax is not None
    plt.close("all")


def test_zero_conc_not_in_output_data():
    """Zero values in titrant_conc should be replaced by the floor, not left as 0."""
    df = _make_df(concs=(0, 1.0))
    plt.close("all")
    ax = plot_theta_fits(df)
    # x-data of the first scatter collection should contain no zeros
    offsets = ax.collections[0].get_offsets()
    x_vals = offsets[:, 0]
    assert (x_vals > 0).all()
    plt.close("all")


def test_original_df_not_mutated():
    """The caller's DataFrame must not be modified."""
    df = _make_df(concs=(0, 1.0))
    original_concs = df["titrant_conc"].copy()
    plt.close("all")
    plot_theta_fits(df)
    pd.testing.assert_series_equal(df["titrant_conc"], original_concs)
    plt.close("all")


# ---------------------------------------------------------------------------
# Series count (one per genotype × titrant_name pair)
# ---------------------------------------------------------------------------

def test_single_series_produces_one_scatter():
    df = _make_df(genotypes=("wt",), titrant_names=("IPTG",))
    plt.close("all")
    ax = plot_theta_fits(df)
    assert len(ax.collections) >= 1
    plt.close("all")


def test_two_genotypes_produce_two_scatter_collections():
    df = _make_df(genotypes=("wt", "A7T"), titrant_names=("IPTG",))
    plt.close("all")
    ax = plot_theta_fits(df)
    assert len(ax.collections) >= 2
    plt.close("all")


def test_two_titrants_produce_two_scatter_collections():
    df = _make_df(genotypes=("wt",), titrant_names=("IPTG", "aTc"))
    plt.close("all")
    ax = plot_theta_fits(df)
    assert len(ax.collections) >= 2
    plt.close("all")


def test_legend_labels_match_series():
    df = _make_df(genotypes=("wt", "A7T"), titrant_names=("IPTG",))
    plt.close("all")
    ax = plot_theta_fits(df)
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("wt" in t for t in legend_texts)
    assert any("A7T" in t for t in legend_texts)
    plt.close("all")


# ---------------------------------------------------------------------------
# Optional uncertainty bands
# ---------------------------------------------------------------------------

def test_no_fill_without_band_columns():
    df = _make_df(with_std_band=False, with_95_band=False)
    plt.close("all")
    ax = plot_theta_fits(df)
    # scatter + errorbar each produce one collection; no fill_between collections
    base_collections = len(ax.collections)
    assert base_collections == 2
    plt.close("all")


def test_std_band_adds_fill():
    df = _make_df(with_std_band=True, with_95_band=False)
    plt.close("all")
    ax = plot_theta_fits(df)
    # scatter + errorbar + one fill_between = at least 3 collections
    assert len(ax.collections) >= 3
    plt.close("all")


def test_95_band_adds_fill():
    df = _make_df(with_std_band=False, with_95_band=True)
    plt.close("all")
    ax = plot_theta_fits(df)
    # scatter + errorbar + one fill_between = at least 3 collections
    assert len(ax.collections) >= 3
    plt.close("all")


def test_both_bands_add_two_fills():
    df = _make_df(with_std_band=True, with_95_band=True)
    plt.close("all")
    ax = plot_theta_fits(df)
    # scatter + errorbar + 2 fill_between = at least 4 collections
    assert len(ax.collections) >= 4
    plt.close("all")


# ---------------------------------------------------------------------------
# scatter_kwargs override
# ---------------------------------------------------------------------------

def test_user_scatter_kwargs_override_defaults():
    """User-supplied scatter_kwargs values must win over defaults."""
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df, scatter_kwargs={"s": 200})
    sizes = ax.collections[0].get_sizes()
    assert (sizes == 200).all()
    plt.close("all")


def test_default_scatter_size_applied():
    """Without override, the default size from DEFAULT_EXPT_SCATTER_KWARGS is used."""
    df = _make_df()
    plt.close("all")
    ax = plot_theta_fits(df)
    sizes = ax.collections[0].get_sizes()
    assert (sizes == default_styles.DEFAULT_EXPT_SCATTER_KWARGS["s"]).all()
    plt.close("all")


# ---------------------------------------------------------------------------
# Custom colors and markers
# ---------------------------------------------------------------------------

def test_custom_colors_accepted():
    df = _make_df(genotypes=("wt", "A7T"))
    plt.close("all")
    ax = plot_theta_fits(df, colors=["#FF0000", "#0000FF"])
    assert ax is not None
    plt.close("all")


def test_custom_markers_accepted():
    df = _make_df(genotypes=("wt", "A7T"))
    plt.close("all")
    ax = plot_theta_fits(df, markers=["o", "s"])
    assert ax is not None
    plt.close("all")


def test_point_est_used_when_q0_5_absent():
    """point_est column is used as centre line when q0.5 is not present."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "genotype": ["wt"] * 4,
        "titrant_name": ["IPTG"] * 4,
        "titrant_conc": [0.1, 1.0, 10.0, 100.0],
        "theta_obs": rng.uniform(0.1, 0.9, 4),
        "theta_std": [0.05] * 4,
        "point_est": rng.uniform(0.1, 0.9, 4),
    })
    plt.close("all")
    ax = plot_theta_fits(df)
    assert ax is not None
    plt.close("all")


def test_q0_5_takes_precedence_over_point_est():
    """q0.5 is used when both q0.5 and point_est columns are present."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "genotype": ["wt"] * 4,
        "titrant_name": ["IPTG"] * 4,
        "titrant_conc": [0.1, 1.0, 10.0, 100.0],
        "theta_obs": rng.uniform(0.1, 0.9, 4),
        "theta_std": [0.05] * 4,
        "q0.5": np.array([0.1, 0.2, 0.3, 0.4]),
        "point_est": np.array([0.9, 0.8, 0.7, 0.6]),
    })
    plt.close("all")
    ax = plot_theta_fits(df)
    # ax.lines includes error-bar lines; find the solid model line (lw=2, no label).
    # It is the line added by ax.plot(), which has linewidth=2.
    model_lines = [l for l in ax.lines if l.get_linewidth() == 2]
    assert len(model_lines) == 1
    np.testing.assert_array_almost_equal(model_lines[0].get_ydata(), [0.1, 0.2, 0.3, 0.4])
    plt.close("all")
