"""
Tests for tfscreen.plot.geno_trajectory.

plot_geno_trajectory is a pure plotting function — all tests pass synthetic
DataFrames and inspect the returned Figure without requiring JAX or a real
ModelOrchestrator.

predict_and_plot_geno_trajectory requires a live orchestrator and is covered
by smoke tests; only its .npz loading path is tested here via a mock.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tfscreen.plot.geno_trajectory import plot_geno_trajectory


# ---------------------------------------------------------------------------
# Synthetic DataFrame helpers
# ---------------------------------------------------------------------------

def _make_pred_df(
    genotypes=("geno_A", "geno_B"),
    replicates=("rep1", "rep2"),
    conditions=(("pre", "sel", "drug", 0.0), ("pre", "sel", "drug", 1.0)),
    t_values=(-5.0, 0.0, 5.0, 10.0, 15.0),
    include_ci=True,
    include_obs=True,
):
    """
    Build a minimal synthetic pred_df.

    Observed data (non-NaN ln_cfu) is placed at t_values that are >= 0.
    Fine-grid rows have NaN ln_cfu.  Rows at negative t carry ln_cfu0 quantiles
    (simulating the pre-selection anchor).
    """
    rng = np.random.default_rng(42)
    rows = []
    for cp, cs, tn, tc in conditions:
        for geno in genotypes:
            for rep in replicates:
                base = rng.uniform(8.0, 12.0)
                slope = rng.uniform(0.1, 0.3)
                for t in t_values:
                    med = base + slope * t
                    row = {
                        "replicate": rep,
                        "condition_pre": cp,
                        "condition_sel": cs,
                        "titrant_name": tn,
                        "titrant_conc": tc,
                        "genotype": geno,
                        "t_sel": t,
                        "ln_cfu": (base + slope * t) if (include_obs and t >= 0) else np.nan,
                        "ln_cfu_std": 0.2 if (include_obs and t >= 0) else np.nan,
                        "median": med,
                    }
                    if include_ci:
                        row["q05"] = med - 0.5
                        row["q95"] = med + 0.5
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_figure(self):
        df = _make_pred_df()
        fig = plot_geno_trajectory(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self):
        df = _make_pred_df()
        fig = plot_geno_trajectory(df)
        assert len(fig.axes) > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# Subplot count
# ---------------------------------------------------------------------------

class TestSubplotCount:
    def test_one_subplot_per_condition_combo(self):
        n_conds = 3
        conditions = tuple(
            ("pre", "sel", "drug", float(i)) for i in range(n_conds)
        )
        df = _make_pred_df(conditions=conditions)
        fig = plot_geno_trajectory(df)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == n_conds
        plt.close(fig)

    def test_single_condition_one_subplot(self):
        df = _make_pred_df(conditions=(("pre", "sel", "drug", 0.0),))
        fig = plot_geno_trajectory(df)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 1
        plt.close(fig)

    def test_four_conditions_two_rows(self):
        conditions = tuple(
            ("pre", "sel", "drug", float(i)) for i in range(4)
        )
        df = _make_pred_df(conditions=conditions)
        fig = plot_geno_trajectory(df)
        n_rows = fig.axes[0].get_subplotspec().get_gridspec().nrows
        assert n_rows == 2
        plt.close(fig)

    def test_unused_cells_hidden(self):
        # 4 conditions → 2×3 grid → 2 hidden cells
        conditions = tuple(
            ("pre", "sel", "drug", float(i)) for i in range(4)
        )
        df = _make_pred_df(conditions=conditions)
        fig = plot_geno_trajectory(df)
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(hidden) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Observed data and predicted lines
# ---------------------------------------------------------------------------

class TestObservedAndPredicted:
    def test_no_crash_all_nan_ln_cfu(self):
        df = _make_pred_df()
        df["ln_cfu"] = np.nan
        df["ln_cfu_std"] = np.nan
        fig = plot_geno_trajectory(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_crash_all_nan_median(self):
        df = _make_pred_df()
        df["median"] = np.nan
        fig = plot_geno_trajectory(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axvline_at_zero_present(self):
        df = _make_pred_df(conditions=(("pre", "sel", "drug", 0.0),))
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        # The axvline draws a vertical line; its x-data is (0, 0).
        vlines = [l for l in ax.lines if list(l.get_xdata()) == [0.0, 0.0]]
        assert len(vlines) >= 1
        plt.close(fig)

    def test_predicted_lines_drawn(self):
        """Each (genotype, replicate) pair should produce at least one line."""
        n_geno = 2
        n_rep = 2
        genotypes = tuple(f"g{i}" for i in range(n_geno))
        replicates = tuple(f"r{i}" for i in range(n_rep))
        df = _make_pred_df(
            genotypes=genotypes,
            replicates=replicates,
            conditions=(("pre", "sel", "drug", 0.0),),
            include_ci=False,
        )
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        # axvline + one line per (geno, rep) pair
        assert len(ax.lines) >= n_geno * n_rep + 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Credible interval
# ---------------------------------------------------------------------------

class TestCredibleInterval:
    def test_ci_band_present_when_columns_exist(self):
        df = _make_pred_df(include_ci=True)
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        # fill_between produces PolyCollection objects
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_no_ci_band_without_columns(self):
        df = _make_pred_df(include_ci=False)
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        # fill_between produces PolyCollection; errorbar may add LineCollections.
        # Only PolyCollections indicate a CI band.
        from matplotlib.collections import PolyCollection
        poly_cols = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(poly_cols) == 0
        plt.close(fig)

    def test_ci_spans_negative_t(self):
        """CI band should cover pre-selection (t < 0) anchor rows."""
        df = _make_pred_df(
            conditions=(("pre", "sel", "drug", 0.0),),
            genotypes=("g0",),
            replicates=("r0",),
            t_values=(-5.0, 0.0, 5.0),
            include_ci=True,
        )
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        polys = ax.collections
        assert len(polys) > 0
        # The polygon should include x-values less than zero
        x_vals = np.concatenate([p.get_paths()[0].vertices[:, 0] for p in polys])
        assert np.any(x_vals < 0)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

class TestLegend:
    def test_legend_contains_genotype_replicate(self):
        df = _make_pred_df(
            genotypes=("WT", "mut1"),
            replicates=("rep1",),
            conditions=(("pre", "sel", "drug", 0.0),),
        )
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("WT" in t for t in legend_texts)
        assert any("mut1" in t for t in legend_texts)
        plt.close(fig)

    def test_legend_includes_replicate_label(self):
        df = _make_pred_df(
            genotypes=("WT",),
            replicates=("rep1", "rep2"),
            conditions=(("pre", "sel", "drug", 0.0),),
        )
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("rep1" in t for t in legend_texts)
        assert any("rep2" in t for t in legend_texts)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Styling overrides
# ---------------------------------------------------------------------------

class TestStylingOverrides:
    def test_custom_figsize(self):
        df = _make_pred_df(conditions=(("pre", "sel", "drug", 0.0),))
        fig = plot_geno_trajectory(df, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert pytest.approx(w) == 10
        assert pytest.approx(h) == 8
        plt.close(fig)

    def test_default_figsize_scales_with_columns(self):
        # 3 conditions → n_cols=3 → width=15
        conditions = tuple(
            ("pre", "sel", "drug", float(i)) for i in range(3)
        )
        df = _make_pred_df(conditions=conditions)
        fig = plot_geno_trajectory(df)
        w, _ = fig.get_size_inches()
        assert pytest.approx(w) == 15
        plt.close(fig)

    def test_custom_colors_accepted(self):
        df = _make_pred_df(genotypes=("g0", "g1"))
        fig = plot_geno_trajectory(df, colors=["#ff0000", "#0000ff"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_markers_accepted(self):
        df = _make_pred_df(replicates=("r0", "r1"))
        fig = plot_geno_trajectory(df, markers=["o", "s"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_genotype_replicate_pairs_get_distinct_colors(self):
        """Each (genotype, replicate) pair should get a distinct color."""
        df = _make_pred_df(
            genotypes=("g0", "g1"),
            replicates=("r0", "r1"),
            conditions=(("pre", "sel", "drug", 0.0),),
            include_ci=False,
        )
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        # Predicted lines are drawn with lw=1.8; exclude axvline
        series_lines = [l for l in ax.lines if l.get_linewidth() == pytest.approx(1.8)]
        colors_used = {l.get_color() for l in series_lines}
        # 2 genotypes × 2 replicates = 4 pairs → 4 distinct colors
        assert len(colors_used) == 4
        plt.close(fig)

    def test_same_genotype_different_replicates_different_colors(self):
        """Two replicates of the same genotype must get different colors."""
        df = _make_pred_df(
            genotypes=("g0",),
            replicates=("r0", "r1"),
            conditions=(("pre", "sel", "drug", 0.0),),
            include_ci=False,
        )
        fig = plot_geno_trajectory(df)
        ax = next(ax for ax in fig.axes if ax.get_visible())
        series_lines = [l for l in ax.lines if l.get_linewidth() == pytest.approx(1.8)]
        colors_used = {l.get_color() for l in series_lines}
        assert len(colors_used) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# predict_and_plot_geno_trajectory — .npz loading path (mocked)
# ---------------------------------------------------------------------------

class TestPredictAndPlotNpzLoading:
    def test_npz_path_converted_to_dict(self, tmp_path):
        """Passing a .npz path loads the file before calling predict()."""
        from unittest.mock import MagicMock, patch
        import matplotlib
        matplotlib.use("Agg")

        npz_path = str(tmp_path / "params.npz")
        np.savez(npz_path, x_auto_loc=np.array(1.0))

        # Build a minimal fake orchestrator
        gd = MagicMock()
        gd.good_mask = np.ones((1, 3, 1, 1, 1, 1, 1), dtype=bool)
        gd.t_sel = np.broadcast_to(
            np.array([0.0, 5.0, 10.0])[None, :, None, None, None, None, None],
            (1, 3, 1, 1, 1, 1, 1),
        ).copy()
        data = MagicMock()
        data.growth = gd

        orchestrator = MagicMock()
        orchestrator.data = data
        orchestrator.presplit_df = None

        # Synthetic fine_df / ln_cfu0_df to return from mocked predict()
        fine_df = pd.DataFrame({
            "replicate": ["r0"] * 5,
            "condition_pre": ["pre"] * 5,
            "condition_sel": ["sel"] * 5,
            "titrant_name": ["drug"] * 5,
            "titrant_conc": [0.0] * 5,
            "genotype": ["g0"] * 5,
            "t_sel": np.linspace(0, 10, 5),
            "ln_cfu": [np.nan] * 5,
            "ln_cfu_std": [np.nan] * 5,
            "q05": np.linspace(8, 12, 5) - 0.5,
            "median": np.linspace(8, 12, 5),
            "q95": np.linspace(8, 12, 5) + 0.5,
        })
        orchestrator.growth_df = pd.DataFrame({
            "replicate": ["r0"],
            "condition_pre": ["pre"],
            "condition_sel": ["sel"],
            "titrant_name": ["drug"],
            "titrant_conc": [0.0],
            "genotype": ["g0"],
            "t_pre": [5.0],
            "t_sel": [5.0],
            "ln_cfu": [10.0],
            "ln_cfu_std": [0.2],
        })

        captured_params = {}

        def _mock_predict(orch, params, predict_sites, **kwargs):
            captured_params["value"] = params
            if len(predict_sites) > 1:
                ln0 = pd.DataFrame({
                    "replicate": ["r0"],
                    "condition_pre": ["pre"],
                    "genotype": ["g0"],
                    "q05": [9.5],
                    "median": [10.0],
                    "q95": [10.5],
                })
                return {"growth_pred": fine_df, "ln_cfu0": ln0}
            return fine_df

        from tfscreen.plot import geno_trajectory
        with patch.object(geno_trajectory, "predict_and_plot_geno_trajectory",
                          wraps=geno_trajectory.predict_and_plot_geno_trajectory):
            with patch("tfscreen.tfmodel.analysis.prediction.predict",
                       side_effect=_mock_predict):
                from tfscreen.plot.geno_trajectory import predict_and_plot_geno_trajectory
                fig = predict_and_plot_geno_trajectory(orchestrator, npz_path)

        assert isinstance(captured_params["value"], dict), \
            "npz path should be loaded to a dict before predict() is called"
        assert "x_auto_loc" in captured_params["value"]
        plt.close(fig)


# ---------------------------------------------------------------------------
# presplit_df overlay
# ---------------------------------------------------------------------------

def _make_mock_orchestrator_with_presplit(presplit_df=None):
    """Return a minimal mock orchestrator for predict_and_plot tests."""
    from unittest.mock import MagicMock

    gd = MagicMock()
    gd.good_mask = np.ones((1, 3, 1, 1, 1, 1, 1), dtype=bool)
    gd.t_sel = np.broadcast_to(
        np.array([0.0, 5.0, 10.0])[None, :, None, None, None, None, None],
        (1, 3, 1, 1, 1, 1, 1),
    ).copy()
    data = MagicMock()
    data.growth = gd

    orch = MagicMock()
    orch.data = data
    orch.growth_df = pd.DataFrame({
        "replicate": ["r0"],
        "condition_pre": ["pre"],
        "condition_sel": ["sel"],
        "titrant_name": ["drug"],
        "titrant_conc": [0.0],
        "genotype": ["g0"],
        "t_pre": [5.0],
        "t_sel": [5.0],
        "ln_cfu": [10.0],
        "ln_cfu_std": [0.2],
    })
    orch.presplit_df = presplit_df
    return orch


def _mock_predict_two_sites(orch, params, predict_sites, **kwargs):
    fine_df = pd.DataFrame({
        "replicate": ["r0"] * 5,
        "condition_pre": ["pre"] * 5,
        "condition_sel": ["sel"] * 5,
        "titrant_name": ["drug"] * 5,
        "titrant_conc": [0.0] * 5,
        "genotype": ["g0"] * 5,
        "t_sel": np.linspace(0, 10, 5),
        "ln_cfu": [np.nan] * 5,
        "ln_cfu_std": [np.nan] * 5,
        "q05": np.linspace(8, 12, 5) - 0.5,
        "median": np.linspace(8, 12, 5),
        "q95": np.linspace(8, 12, 5) + 0.5,
    })
    ln0 = pd.DataFrame({
        "replicate": ["r0"],
        "condition_pre": ["pre"],
        "genotype": ["g0"],
        "q05": [9.5],
        "median": [10.0],
        "q95": [10.5],
    })
    return {"growth_pred": fine_df, "ln_cfu0": ln0}


class TestPresplitOverlay:
    def test_no_presplit_anchor_has_nan_ln_cfu(self):
        """Without presplit_df the anchor row has NaN ln_cfu."""
        from unittest.mock import patch
        orch = _make_mock_orchestrator_with_presplit(presplit_df=None)

        with patch("tfscreen.tfmodel.analysis.prediction.predict",
                   side_effect=_mock_predict_two_sites):
            from tfscreen.plot.geno_trajectory import predict_and_plot_geno_trajectory
            fig = predict_and_plot_geno_trajectory(orch, {"dummy": np.array(1.0)})

        ax = next(a for a in fig.axes if a.get_visible())
        # Only observed data plotted via errorbar comes from non-NaN ln_cfu.
        # With no presplit and obs rows at t_sel>=0 only, t<0 should have no errorbar.
        obs_lines_neg = [
            l for l in ax.lines
            if l.get_linewidth() != pytest.approx(1.8)  # exclude predicted lines
            and list(l.get_xdata()) != [0.0, 0.0]        # exclude axvline
            and any(x < 0 for x in l.get_xdata())
        ]
        assert len(obs_lines_neg) == 0
        plt.close(fig)

    def test_presplit_anchor_shows_observed_point(self):
        """With presplit_df the anchor row has observed ln_cfu rendered as an errorbar."""
        from unittest.mock import patch
        presplit = pd.DataFrame({
            "replicate": ["r0"],
            "condition_pre": ["pre"],
            "genotype": ["g0"],
            "ln_cfu": [9.8],
            "ln_cfu_std": [0.15],
        })
        orch = _make_mock_orchestrator_with_presplit(presplit_df=presplit)

        with patch("tfscreen.tfmodel.analysis.prediction.predict",
                   side_effect=_mock_predict_two_sites):
            from tfscreen.plot.geno_trajectory import predict_and_plot_geno_trajectory
            fig = predict_and_plot_geno_trajectory(orch, {"dummy": np.array(1.0)})

        ax = next(a for a in fig.axes if a.get_visible())
        # errorbar lines at negative t should exist now
        obs_lines_neg = [
            l for l in ax.lines
            if l.get_linewidth() != pytest.approx(1.8)
            and list(l.get_xdata()) != [0.0, 0.0]
            and len(l.get_xdata()) > 0
            and any(x < 0 for x in l.get_xdata())
        ]
        assert len(obs_lines_neg) > 0
        plt.close(fig)

    def test_presplit_unmatched_genotype_stays_nan(self):
        """presplit row for a different genotype does not affect anchor ln_cfu."""
        from unittest.mock import patch
        presplit = pd.DataFrame({
            "replicate": ["r0"],
            "condition_pre": ["pre"],
            "genotype": ["other_geno"],  # not "g0"
            "ln_cfu": [9.8],
            "ln_cfu_std": [0.15],
        })
        orch = _make_mock_orchestrator_with_presplit(presplit_df=presplit)

        with patch("tfscreen.tfmodel.analysis.prediction.predict",
                   side_effect=_mock_predict_two_sites):
            from tfscreen.plot.geno_trajectory import predict_and_plot_geno_trajectory
            fig = predict_and_plot_geno_trajectory(orch, {"dummy": np.array(1.0)})

        ax = next(a for a in fig.axes if a.get_visible())
        obs_lines_neg = [
            l for l in ax.lines
            if l.get_linewidth() != pytest.approx(1.8)
            and list(l.get_xdata()) != [0.0, 0.0]
            and len(l.get_xdata()) > 0
            and any(x < 0 for x in l.get_xdata())
        ]
        assert len(obs_lines_neg) == 0
        plt.close(fig)
