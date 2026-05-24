
import pytest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from tfscreen.plot.hill_fit import hill_fit


def _scatter_collection(ax):
    """Return the first PathCollection (scatter) on an axes."""
    return next(c for c in ax.collections if isinstance(c, PathCollection))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs_pred(genotype="WT", include_other=True):
    rows = [
        {"genotype": genotype, "titrant_conc": 0,    "theta_est": 0.05, "theta_std": 0.01},
        {"genotype": genotype, "titrant_conc": 1.0,  "theta_est": 0.5,  "theta_std": 0.05},
        {"genotype": genotype, "titrant_conc": 10.0, "theta_est": 0.9,  "theta_std": 0.01},
    ]
    if include_other:
        rows.append({"genotype": "OTHER", "titrant_conc": 1.0, "theta_est": 0.3, "theta_std": 0.05})
    obs_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame({
        "genotype": [genotype, genotype],
        "titrant_conc": [1e-6, 10.0],
        "hill_est": [0.05, 0.9],
        "hill_std": [0.01, 0.01],
    })
    return obs_df, pred_df


# ---------------------------------------------------------------------------
# Smoke / basic
# ---------------------------------------------------------------------------

def test_hill_fit_basic():
    obs_df, pred_df = _make_obs_pred()
    plt.close("all")
    ax = hill_fit(obs_df, pred_df, genotype="WT")
    assert ax is not None
    assert ax.get_xscale() == "log"
    assert len(ax.collections) >= 1
    assert len(ax.lines) >= 1
    plt.close("all")


def test_hill_fit_custom_ax():
    obs_df, pred_df = _make_obs_pred()
    _, ax = plt.subplots()
    ret_ax = hill_fit(obs_df, pred_df, genotype="WT", ax=ax)
    assert ret_ax is ax
    plt.close("all")


# ---------------------------------------------------------------------------
# Zero-concentration floor
# ---------------------------------------------------------------------------

def test_zero_conc_replaced():
    """titrant_conc == 0 must be replaced by zero_titrant_value before plotting."""
    obs_df, pred_df = _make_obs_pred()
    plt.close("all")
    ax = hill_fit(obs_df, pred_df, genotype="WT", zero_titrant_value=1e-3)
    offsets = _scatter_collection(ax).get_offsets()
    assert (offsets[:, 0] > 0).all()
    plt.close("all")


def test_zero_conc_obs_df_not_mutated():
    """The caller's obs_df must not be modified."""
    obs_df, pred_df = _make_obs_pred()
    original = obs_df["titrant_conc"].copy()
    plt.close("all")
    hill_fit(obs_df, pred_df, genotype="WT")
    pd.testing.assert_series_equal(obs_df["titrant_conc"], original)
    plt.close("all")


# ---------------------------------------------------------------------------
# Genotype filtering
# ---------------------------------------------------------------------------

def test_only_requested_genotype_plotted():
    """Data for the non-requested genotype must not appear in the scatter."""
    obs_df, pred_df = _make_obs_pred(genotype="WT", include_other=True)
    plt.close("all")
    ax = hill_fit(obs_df, pred_df, genotype="WT")
    # Only 3 rows belong to "WT" (one zero-conc replaced)
    offsets = _scatter_collection(ax).get_offsets()
    assert len(offsets) == 3
    plt.close("all")


# ---------------------------------------------------------------------------
# kwargs forwarding
# ---------------------------------------------------------------------------

def test_scatter_kwargs_override():
    obs_df, pred_df = _make_obs_pred()
    plt.close("all")
    ax = hill_fit(obs_df, pred_df, genotype="WT", scatter_kwargs={"s": 200})
    sizes = _scatter_collection(ax).get_sizes()
    assert (sizes == 200).all()
    plt.close("all")


def test_fit_line_kwargs_override():
    obs_df, pred_df = _make_obs_pred()
    plt.close("all")
    ax = hill_fit(obs_df, pred_df, genotype="WT", fit_line_kwargs={"lw": 5})
    # The fit line is the last ax.plot call; find it among the lines
    # errorbar also adds lines, so look for lw=5
    lws = [l.get_linewidth() for l in ax.lines]
    assert 5 in lws
    plt.close("all")
