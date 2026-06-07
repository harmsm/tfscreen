"""
Tests for tfscreen.plot.geno_trajectory.plot_geno_trajectory.

Strategy
--------
The public function has two expensive internal steps: (a) a JAX forward
pass via _compute_map_predictions / _compute_posterior_predictions, and
(b) matplotlib PDF rendering.  Both are mocked out here so tests run fast
and without GPU/display dependencies.

A helper ``_make_fake_gm`` constructs a minimal mock ModelOrchestrator
whose attributes match what plot_geno_trajectory reads.  The mock returns
deterministic numpy arrays for all data tensors and a small but realistic
dimension structure (2 replicates, 3 timepoints, 1 condition_pre,
1 condition_sel, 1 titrant, 1 concentration, 2 genotypes).

The prediction arrays (growth_pred, growth_pred_fine, ln_cfu0_map) are
returned by mocked helpers so no JAX compilation is triggered.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tfscreen.plot.geno_trajectory import (
    _compute_posterior_predictions,
    plot_geno_trajectory,
)


# ---------------------------------------------------------------------------
# Shared shape constants
# ---------------------------------------------------------------------------

N_REP, N_T, N_CP, N_CS, N_TN, N_TC, N_GENO = 2, 3, 1, 1, 1, 1, 2
SHAPE = (N_REP, N_T, N_CP, N_CS, N_TN, N_TC, N_GENO)
GENO_NAMES  = ["WT", "mut1"]
REP_NAMES   = ["rep1", "rep2"]
CP_NAMES    = ["preinduction"]
CS_NAMES    = ["selection"]
TN_NAMES    = ["IPTG"]
TC_VALS     = [0.0]


# ---------------------------------------------------------------------------
# Fake ModelOrchestrator
# ---------------------------------------------------------------------------

def _make_fake_gm(with_presplit=False):
    """
    Return a MagicMock that quacks like a ModelOrchestrator for the
    attributes accessed by plot_geno_trajectory.
    """
    rng = np.random.default_rng(0)

    # All timepoints are valid except the last one for rep 0 genotype 0.
    good_mask = np.ones(SHAPE, dtype=bool)
    good_mask[0, -1, 0, 0, 0, 0, 0] = False

    t_sel = np.broadcast_to(
        np.linspace(0.0, 10.0, N_T)[None, :, None, None, None, None, None],
        SHAPE,
    ).copy()
    t_pre = np.full(SHAPE, 5.0)
    ln_cfu = rng.normal(10.0, 1.0, SHAPE)
    ln_cfu_std = np.ones(SHAPE) * 0.5

    gd = MagicMock()
    gd.good_mask  = good_mask
    gd.t_sel      = t_sel
    gd.t_pre      = t_pre
    gd.ln_cfu     = ln_cfu
    gd.ln_cfu_std = ln_cfu_std

    # --- presplit ---
    presplit = None
    if with_presplit:
        ps = MagicMock()
        ps.ln_cfu_t0     = rng.normal(10.0, 0.5, (N_REP, N_CP, N_GENO))
        ps.ln_cfu_t0_std = np.ones((N_REP, N_CP, N_GENO)) * 0.3
        ps.good_mask     = np.ones((N_REP, N_CP, N_GENO), dtype=bool)
        presplit = ps

    data = MagicMock()
    data.growth    = gd
    data.presplit  = presplit
    data.num_genotype = N_GENO

    # --- TensorManager ---
    # Build a small DataFrame that mirrors what tm.df looks like after
    # TensorManager.create_tensors().  Only the columns that plot_geno_trajectory
    # accesses are required.
    rows = []
    for r_i in range(N_REP):
        for t_i in range(N_T):
            for g_i in range(N_GENO):
                rows.append({
                    "replicate_idx":     r_i,
                    "condition_pre_idx": 0,
                    "condition_sel_idx": 0,
                    "titrant_name_idx":  0,
                    "titrant_conc_idx":  0,
                    "genotype_idx":      g_i,
                    "t_sel":             t_sel[r_i, t_i, 0, 0, 0, 0, g_i],
                    "ln_cfu":            ln_cfu[r_i, t_i, 0, 0, 0, 0, g_i],
                    "genotype":          GENO_NAMES[g_i],
                    "condition_sel":     "selection",
                    "condition_pre":     "preinduction",
                })
    tm_df = pd.DataFrame(rows)

    tm = MagicMock()
    tm.tensor_dim_names   = [
        "replicate", "time", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "genotype",
    ]
    tm.tensor_dim_labels  = [
        REP_NAMES, [f"t{i}" for i in range(N_T)],
        CP_NAMES, CS_NAMES, TN_NAMES, TC_VALS, GENO_NAMES,
    ]
    tm.df = tm_df

    binding_df = pd.DataFrame({
        "genotype":     GENO_NAMES * 2,
        "titrant_name": ["IPTG"] * 4,
        "titrant_conc": [0.0] * 4,
        "theta_obs":    [0.5, 0.3, 0.5, 0.3],
    })

    gm = MagicMock()
    gm.data       = data
    gm.growth_tm  = tm
    gm.binding_df = binding_df

    return gm


def _fake_map_predictions(shape):
    """Return plausible synthetic MAP prediction tensors."""
    rng = np.random.default_rng(1)
    growth_pred      = rng.normal(10.0, 1.0, shape)
    ln_cfu0_map      = rng.normal(10.0, 0.5, (shape[0], shape[2], shape[6]))
    growth_pred_fine = rng.normal(10.0, 1.0,
                                  (shape[0], 50, shape[2], shape[3],
                                   shape[4], shape[5], shape[6]))
    t_fine_1d        = np.linspace(0.0, 10.0, 50)
    return growth_pred, ln_cfu0_map, growth_pred_fine, t_fine_1d


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestArgumentValidation:
    def test_raises_when_neither_provided(self, tmp_path):
        gm = _make_fake_gm()
        with pytest.raises(ValueError, match="Exactly one"):
            plot_geno_trajectory(gm, str(tmp_path / "out"))

    def test_raises_when_both_provided(self, tmp_path):
        gm = _make_fake_gm()
        with pytest.raises(ValueError, match="Exactly one"):
            plot_geno_trajectory(
                gm,
                str(tmp_path / "out"),
                params={"x_auto_loc": np.array(1.0)},
                posterior_file="some_file.h5",
            )


# ---------------------------------------------------------------------------
# MAP path — basic smoke test with mocked internals
# ---------------------------------------------------------------------------

class TestMapPath:
    @pytest.fixture
    def patched_plot(self, tmp_path):
        """Patch _compute_map_predictions and matplotlib.pyplot.savefig."""
        gm = _make_fake_gm()
        fake_preds = _fake_map_predictions(SHAPE)

        with patch(
            "tfscreen.plot.geno_trajectory._compute_map_predictions",
            return_value=fake_preds,
        ), patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            yield gm, tmp_path

    def test_returns_dataframe(self, patched_plot):
        gm, tmp_path = patched_plot
        result = plot_geno_trajectory(
            gm,
            str(tmp_path / "out"),
            params={"x_auto_loc": np.array(1.0)},
        )
        assert isinstance(result, pd.DataFrame)

    def test_df_has_ln_cfu_pred(self, patched_plot):
        gm, tmp_path = patched_plot
        result = plot_geno_trajectory(
            gm,
            str(tmp_path / "out"),
            params={"x_auto_loc": np.array(1.0)},
        )
        assert "ln_cfu_pred" in result.columns

    def test_csv_written_by_default(self, patched_plot):
        gm, tmp_path = patched_plot
        plot_geno_trajectory(
            gm,
            str(tmp_path / "out"),
            params={"x_auto_loc": np.array(1.0)},
        )
        assert os.path.exists(tmp_path / "out_calib_growth_df.csv")

    def test_csv_not_written_when_disabled(self, patched_plot):
        gm, tmp_path = patched_plot
        plot_geno_trajectory(
            gm,
            str(tmp_path / "out"),
            params={"x_auto_loc": np.array(1.0)},
            write_csv=False,
        )
        assert not os.path.exists(tmp_path / "out_calib_growth_df.csv")

    def test_growth_pred_std_in_csv(self, patched_plot):
        gm, tmp_path = patched_plot
        gps = np.ones(SHAPE) * 0.1
        plot_geno_trajectory(
            gm,
            str(tmp_path / "out"),
            params={"x_auto_loc": np.array(1.0)},
            growth_pred_std=gps,
        )
        df = pd.read_csv(tmp_path / "out_calib_growth_df.csv")
        assert "ln_cfu_pred_std" in df.columns


# ---------------------------------------------------------------------------
# Genotype filtering
# ---------------------------------------------------------------------------

class TestGenotypeFilter:
    @pytest.fixture
    def gm_and_path(self, tmp_path):
        gm = _make_fake_gm()
        return gm, tmp_path

    def _run(self, gm, tmp_path, **kwargs):
        fake_preds = _fake_map_predictions(SHAPE)
        with patch(
            "tfscreen.plot.geno_trajectory._compute_map_predictions",
            return_value=fake_preds,
        ), patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            return plot_geno_trajectory(
                gm,
                str(tmp_path / "out"),
                params={"x_auto_loc": np.array(1.0)},
                **kwargs,
            )

    def test_no_filter_plots_all_genotypes(self, gm_and_path, capsys):
        gm, tmp_path = gm_and_path
        self._run(gm, tmp_path)
        captured = capsys.readouterr()
        for name in GENO_NAMES:
            assert name in captured.out

    def test_genotype_subset(self, gm_and_path, capsys):
        gm, tmp_path = gm_and_path
        self._run(gm, tmp_path, genotypes=["WT"])
        captured = capsys.readouterr()
        assert "WT" in captured.out
        assert "mut1" not in captured.out

    def test_unknown_genotype_warns(self, gm_and_path, capsys):
        gm, tmp_path = gm_and_path
        self._run(gm, tmp_path, genotypes=["WT", "nonexistent"])
        captured = capsys.readouterr()
        assert "nonexistent" in captured.err

    def test_all_unknown_returns_df(self, gm_and_path):
        gm, tmp_path = gm_and_path
        result = self._run(gm, tmp_path, genotypes=["nonexistent"])
        # Should still return a DataFrame (from the full data), just no PDFs.
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Titrant_name filtering
# ---------------------------------------------------------------------------

class TestTitrantNameFilter:
    def _run(self, gm, tmp_path, **kwargs):
        fake_preds = _fake_map_predictions(SHAPE)
        with patch(
            "tfscreen.plot.geno_trajectory._compute_map_predictions",
            return_value=fake_preds,
        ), patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            return plot_geno_trajectory(
                gm,
                str(tmp_path / "out"),
                params={"x_auto_loc": np.array(1.0)},
                **kwargs,
            )

    def test_matching_titrant_produces_output(self, tmp_path, capsys):
        gm = _make_fake_gm()
        self._run(gm, tmp_path, titrant_names=["IPTG"])
        captured = capsys.readouterr()
        # PDFs should still be saved (the filter matches).
        assert "Saved" in captured.out

    def test_nonmatching_titrant_no_pdfs(self, tmp_path, capsys):
        gm = _make_fake_gm()
        self._run(gm, tmp_path, titrant_names=["arabinose"])
        captured = capsys.readouterr()
        # No valid condition combos → no PDF files saved (the CSV may still be).
        assert ".pdf" not in captured.out


# ---------------------------------------------------------------------------
# Presplit overlay
# ---------------------------------------------------------------------------

class TestPresplit:
    def _run(self, gm, tmp_path):
        fake_preds = _fake_map_predictions(SHAPE)
        ax_mock = MagicMock()
        # Capture all errorbar calls to verify presplit is rendered.
        ax_mock.errorbar = MagicMock()
        ax_mock.plot     = MagicMock()
        ax_mock.legend   = MagicMock()
        ax_mock.set_title = MagicMock()
        ax_mock.set_xlabel = MagicMock()
        ax_mock.set_ylabel = MagicMock()
        ax_mock.axvline   = MagicMock()
        ax_mock.set_xlim  = MagicMock()
        ax_mock.set_visible = MagicMock()

        axes_array = np.empty((1, 1), dtype=object)
        axes_array[0, 0] = ax_mock

        with patch(
            "tfscreen.plot.geno_trajectory._compute_map_predictions",
            return_value=fake_preds,
        ), patch(
            "matplotlib.pyplot.subplots",
            return_value=(MagicMock(), axes_array),
        ), patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            plot_geno_trajectory(
                gm,
                str(tmp_path / "out"),
                params={"x_auto_loc": np.array(1.0)},
            )
        return ax_mock

    @staticmethod
    def _is_scalar_negative(x):
        """True iff x is a 0-d value strictly less than zero.

        The presplit errorbar passes a scalar (−t_pre).  Selection-phase
        errorbars pass an array of t_sel values.  Using ndim prevents
        float() from raising on multi-element arrays.
        """
        a = np.asarray(x)
        return a.ndim == 0 and float(a) < 0

    def test_presplit_errorbar_called_when_data_present(self, tmp_path):
        gm = _make_fake_gm(with_presplit=True)
        ax_mock = self._run(gm, tmp_path)
        # The presplit errorbar is called with a scalar negative x (−t_pre);
        # selection-phase errorbars use array x values.
        neg_x_calls = [
            c for c in ax_mock.errorbar.call_args_list
            if c.args and self._is_scalar_negative(c.args[0])
        ]
        assert len(neg_x_calls) > 0, (
            "Expected at least one scalar-negative-x errorbar call (presplit) "
            f"but got: {ax_mock.errorbar.call_args_list}"
        )

    def test_no_presplit_errorbar_at_negative_x(self, tmp_path):
        gm = _make_fake_gm(with_presplit=False)
        ax_mock = self._run(gm, tmp_path)
        neg_x_calls = [
            c for c in ax_mock.errorbar.call_args_list
            if c.args and self._is_scalar_negative(c.args[0])
        ]
        assert len(neg_x_calls) == 0


# ---------------------------------------------------------------------------
# Posterior path
# ---------------------------------------------------------------------------

class TestPosteriorPath:
    def _make_h5(self, tmp_path):
        """Write a minimal synthetic posterior HDF5 file."""
        import h5py

        h5_path = str(tmp_path / "posterior.h5")
        n_samples = 5
        with h5py.File(h5_path, "w") as fh:
            fh.create_dataset(
                "growth_pred",
                data=np.random.default_rng(2).normal(10.0, 1.0,
                                                      (n_samples,) + SHAPE),
            )
            fh.create_dataset(
                "ln_cfu0",
                data=np.random.default_rng(3).normal(10.0, 0.5,
                                                      (n_samples, N_REP,
                                                       N_CP, N_GENO)),
            )
        return h5_path

    def test_compute_posterior_predictions_shapes(self, tmp_path):
        h5_path = self._make_h5(tmp_path)
        gm = _make_fake_gm()
        med, lo, hi, ln_cfu0 = _compute_posterior_predictions(gm, h5_path)
        assert med.shape == SHAPE
        assert lo.shape  == SHAPE
        assert hi.shape  == SHAPE
        assert ln_cfu0.shape == (N_REP, N_CP, N_GENO)

    def test_compute_posterior_predictions_ordering(self, tmp_path):
        """5th percentile ≤ median ≤ 95th percentile everywhere."""
        h5_path = self._make_h5(tmp_path)
        gm = _make_fake_gm()
        med, lo, hi, _ = _compute_posterior_predictions(gm, h5_path)
        assert np.all(lo <= med + 1e-9)
        assert np.all(med <= hi + 1e-9)

    def test_posterior_path_returns_df_with_lo_hi(self, tmp_path):
        h5_path = self._make_h5(tmp_path)
        gm = _make_fake_gm()
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            result = plot_geno_trajectory(
                gm,
                str(tmp_path / "out"),
                posterior_file=h5_path,
            )
        assert isinstance(result, pd.DataFrame)
        assert "ln_cfu_pred_lo" in result.columns
        assert "ln_cfu_pred_hi" in result.columns

    def test_posterior_raises_on_missing_growth_pred(self, tmp_path):
        import h5py

        h5_path = str(tmp_path / "empty.h5")
        with h5py.File(h5_path, "w") as fh:
            fh.create_dataset("something_else", data=np.zeros(3))

        gm = _make_fake_gm()
        with pytest.raises(ValueError, match="growth_pred not found"):
            _compute_posterior_predictions(gm, h5_path)

    def test_params_path_loads_npz(self, tmp_path):
        """Passing a string path to params loads the .npz file."""
        npz_path = str(tmp_path / "params.npz")
        np.savez(npz_path, x_auto_loc=np.array(1.0))

        fake_preds = _fake_map_predictions(SHAPE)
        gm = _make_fake_gm()

        with patch(
            "tfscreen.plot.geno_trajectory._compute_map_predictions",
            return_value=fake_preds,
        ) as mock_fn, patch("matplotlib.pyplot.savefig"), patch(
            "matplotlib.pyplot.close"
        ):
            plot_geno_trajectory(gm, str(tmp_path / "out"), params=npz_path)

        # The helper should have been called with a dict (loaded from npz),
        # not the raw string path.
        called_params = mock_fn.call_args[0][1]
        assert isinstance(called_params, dict)
        assert "x_auto_loc" in called_params
