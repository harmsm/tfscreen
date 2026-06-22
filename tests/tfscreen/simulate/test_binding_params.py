"""
Unit tests for tfscreen.simulate.binding_params.
"""

import io
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from tfscreen.simulate.binding_params import (
    read_binding_genotype_params,
    build_theta_gc_override_hill_geno,
    build_theta_gc_override_hill_mut,
    build_binding_theta_from_params,
    _hill_theta,
    _fill_params_from_wt,
    _logit,
    _to_log_conc,
    _wt_params_from_sim_priors,
    SUPPORTED_COMPONENTS,
    HILL_PARAM_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_csv(tmp_path):
    """CSV with all four Hill parameter columns."""
    content = (
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "wt,0.99,0.01,-4.1,2.0\n"
        "A47V,0.97,0.03,-3.8,1.8\n"
        "K84L,0.98,0.02,-3.5,2.2\n"
    )
    p = tmp_path / "params.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture
def partial_csv(tmp_path):
    """CSV with only two parameter columns; others should NaN."""
    content = (
        "genotype,log_hill_K,hill_n\n"
        "wt,-4.1,2.0\n"
        "A47V,-3.8,1.8\n"
    )
    p = tmp_path / "partial.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture
def nan_csv(tmp_path):
    """CSV with explicit NaN values."""
    content = (
        "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
        "wt,0.99,0.01,-4.1,2.0\n"
        "A47V,,,-3.8,1.8\n"
    )
    p = tmp_path / "nan_params.csv"
    p.write_text(content)
    return str(p)


@pytest.fixture
def wt_params():
    return {
        "theta_low":  0.99,
        "theta_high": 0.01,
        "log_hill_K": -4.1,
        "hill_n":     2.0,
    }


@pytest.fixture
def mock_sim_priors(wt_params):
    sp = MagicMock()
    sp.wt_theta_low  = wt_params["theta_low"]
    sp.wt_theta_high = wt_params["theta_high"]
    sp.wt_log_K      = wt_params["log_hill_K"]
    sp.wt_hill_n     = wt_params["hill_n"]
    sp.sigma_d_logit_low   = 0.3
    sp.sigma_d_logit_delta = 0.5
    sp.sigma_d_log_K       = 0.5
    sp.sigma_d_log_n       = 0.3
    sp.epi_tau_scale  = 0.0   # no epistasis by default in tests
    sp.epi_slab_scale = 2.0
    sp.epi_slab_df    = 4.0
    return sp


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# read_binding_genotype_params
# ---------------------------------------------------------------------------

class TestReadBindingGenotypeParams:

    def test_reads_all_params(self, simple_csv):
        d = read_binding_genotype_params(simple_csv)
        assert set(d.keys()) == {"wt", "A47V", "K84L"}
        assert d["wt"]["theta_low"] == pytest.approx(0.99)
        assert d["A47V"]["log_hill_K"] == pytest.approx(-3.8)

    def test_reads_partial_columns(self, partial_csv):
        d = read_binding_genotype_params(partial_csv)
        assert "theta_low" not in d["wt"]
        assert d["wt"]["log_hill_K"] == pytest.approx(-4.1)

    def test_nan_preserved(self, nan_csv):
        d = read_binding_genotype_params(nan_csv)
        assert np.isnan(d["A47V"]["theta_low"])
        assert np.isnan(d["A47V"]["theta_high"])
        assert d["A47V"]["log_hill_K"] == pytest.approx(-3.8)

    def test_raises_no_genotype_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("theta_low,theta_high\n0.99,0.01\n")
        with pytest.raises(ValueError, match="genotype"):
            read_binding_genotype_params(str(p))

    def test_raises_no_param_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("genotype\nwt\n")
        with pytest.raises(ValueError, match="parameter column"):
            read_binding_genotype_params(str(p))

    def test_raises_unknown_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("genotype,theta_low,extra_col\nwt,0.99,foo\n")
        with pytest.raises(ValueError, match="Unrecognised"):
            read_binding_genotype_params(str(p))

    def test_clips_theta_above_one(self, tmp_path):
        """theta_low > 1 (float rounding) must be clipped and trigger a warning."""
        p = tmp_path / "over.csv"
        p.write_text("genotype,theta_low,theta_high,log_hill_K,hill_n\n"
                     "wt,1.000004,0.01,-4.1,2.0\n")
        with pytest.warns(UserWarning, match="theta_low"):
            d = read_binding_genotype_params(str(p))
        assert d["wt"]["theta_low"] <= 1.0 - 1e-4

    def test_clips_theta_below_zero(self, tmp_path):
        """theta_high < 0 must be clipped and trigger a warning."""
        p = tmp_path / "under.csv"
        p.write_text("genotype,theta_low,theta_high,log_hill_K,hill_n\n"
                     "wt,0.99,-0.0001,-4.1,2.0\n")
        with pytest.warns(UserWarning, match="theta_high"):
            d = read_binding_genotype_params(str(p))
        assert d["wt"]["theta_high"] >= 1e-4

    def test_no_warning_for_valid_values(self, simple_csv):
        """Valid theta values in (0, 1) must not trigger a warning."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            read_binding_genotype_params(simple_csv)  # should not raise

    def test_nan_theta_not_clipped(self, nan_csv):
        """NaN theta values must pass through without triggering a warning."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            d = read_binding_genotype_params(nan_csv)
        assert np.isnan(d["A47V"]["theta_low"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHillTheta:

    def test_wt_curve_shape(self, wt_params):
        log_conc = np.linspace(-10, 0, 20)
        theta = _hill_theta(**wt_params, log_conc=log_conc)
        assert theta.shape == (20,)
        assert np.all(theta >= 0) and np.all(theta <= 1)

    def test_decreasing_for_repressor(self, wt_params):
        """With theta_low > theta_high, theta decreases as ligand increases."""
        log_conc = np.linspace(-10, 0, 10)
        theta = _hill_theta(**wt_params, log_conc=log_conc)
        assert theta[0] > theta[-1]

    def test_zero_concentration_handled(self, wt_params):
        """Zero concentrations are passed as the sentinel; no NaN."""
        log_conc = _to_log_conc([0.0, 0.001, 0.01])
        theta = _hill_theta(**wt_params, log_conc=log_conc)
        assert np.all(np.isfinite(theta))

    def test_endpoints(self, wt_params):
        """Very low conc → theta_low; very high conc → theta_high."""
        theta_low_end  = _hill_theta(**wt_params, log_conc=np.array([-30.0]))
        theta_high_end = _hill_theta(**wt_params, log_conc=np.array([10.0]))
        assert theta_low_end[0]  == pytest.approx(wt_params["theta_low"],  abs=1e-4)
        assert theta_high_end[0] == pytest.approx(wt_params["theta_high"], abs=1e-4)


class TestFillParamsFromWt:

    def test_fills_nan_from_wt(self, wt_params):
        params = {"theta_low": float("nan"), "log_hill_K": -3.8,
                  "theta_high": float("nan"), "hill_n": float("nan")}
        filled = _fill_params_from_wt(params, wt_params)
        assert filled["theta_low"]  == wt_params["theta_low"]
        assert filled["theta_high"] == wt_params["theta_high"]
        assert filled["hill_n"]     == wt_params["hill_n"]
        assert filled["log_hill_K"] == pytest.approx(-3.8)

    def test_does_not_overwrite_present(self, wt_params):
        params = {"theta_low": 0.5, "theta_high": 0.5,
                  "log_hill_K": -3.0, "hill_n": 1.5}
        filled = _fill_params_from_wt(params, wt_params)
        assert filled["theta_low"] == pytest.approx(0.5)

    def test_missing_key_uses_wt(self, wt_params):
        params = {"log_hill_K": -3.8, "hill_n": 1.8}
        filled = _fill_params_from_wt(params, wt_params)
        assert filled["theta_low"]  == wt_params["theta_low"]
        assert filled["theta_high"] == wt_params["theta_high"]


class TestWtParamsFromSimPriors:

    def test_extracts_correct_keys(self, mock_sim_priors, wt_params):
        result = _wt_params_from_sim_priors(mock_sim_priors)
        assert result["theta_low"]  == pytest.approx(wt_params["theta_low"])
        assert result["log_hill_K"] == pytest.approx(wt_params["log_hill_K"])
        assert set(result.keys()) == HILL_PARAM_COLS


# ---------------------------------------------------------------------------
# build_theta_gc_override_hill_geno
# ---------------------------------------------------------------------------

class TestBuildThetaGcOverrideHillGeno:

    def test_returns_correct_genotypes(self, simple_csv, wt_params):
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.linspace(-10, 0, 5)
        override, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        assert set(override.keys()) == {"wt", "A47V", "K84L"}

    def test_theta_shape(self, simple_csv, wt_params):
        params_dict = read_binding_genotype_params(simple_csv)
        C = 8
        log_conc = np.linspace(-10, 0, C)
        override, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        for g, theta in override.items():
            assert theta.shape == (C,), f"Shape mismatch for {g}"

    def test_wt_uses_wt_params(self, simple_csv, wt_params):
        """WT row in CSV should produce the exact WT curve."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.linspace(-10, 0, 20)
        override, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        expected = _hill_theta(**wt_params, log_conc=log_conc)
        np.testing.assert_allclose(override["wt"], expected, rtol=1e-6)

    def test_mutant_uses_measured_params(self, simple_csv, wt_params):
        """Non-WT row should use the measured parameters, not WT."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.linspace(-10, 0, 20)
        override, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        wt_curve = _hill_theta(**wt_params, log_conc=log_conc)
        # A47V has a different log_hill_K, so its curve should differ from WT
        assert not np.allclose(override["A47V"], wt_curve)

    def test_nan_params_filled_from_wt(self, nan_csv, wt_params):
        """NaN theta_low / theta_high should fall back to WT values."""
        params_dict = read_binding_genotype_params(nan_csv)
        log_conc = np.linspace(-10, 0, 5)
        override, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        # A47V has NaN theta_low / theta_high → WT values used
        # its log_hill_K is -3.8 (different from WT -4.1), so not identical to WT
        assert override["A47V"].shape == (5,)
        assert np.all(np.isfinite(override["A47V"]))

    def test_theta_in_unit_interval(self, simple_csv, wt_params):
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.linspace(-15, 5, 30)
        override, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        for theta in override.values():
            assert np.all(theta >= 0) and np.all(theta <= 1)

    def test_effective_params_returned(self, simple_csv, wt_params):
        """effective_params matches the CSV values (NaN filled from wt_params)."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.linspace(-10, 0, 5)
        _, eff = build_theta_gc_override_hill_geno(params_dict, log_conc, wt_params)
        assert set(eff.keys()) == {"wt", "A47V", "K84L"}
        assert eff["wt"]["theta_low"]  == pytest.approx(0.99)
        assert eff["A47V"]["log_hill_K"] == pytest.approx(-3.8)
        assert eff["K84L"]["hill_n"]   == pytest.approx(2.2)


# ---------------------------------------------------------------------------
# build_theta_gc_override_hill_mut
# ---------------------------------------------------------------------------

def _make_sim_data_for_genotypes(genotypes):
    """Build a real SimData for a small genotype list (no thermo data)."""
    import pandas as pd
    from tfscreen.simulate.sim_data_class import build_sim_data

    library_df  = pd.DataFrame({"genotype": genotypes})
    sample_df   = pd.DataFrame({"titrant_conc": [0.0, 0.001, 0.01, 0.1, 1.0]})
    return build_sim_data(library_df, sample_df)


class TestBuildThetaGcOverrideHillMut:

    @pytest.fixture
    def small_genotypes(self):
        return ["wt", "A47V", "K84L", "A47V/K84L"]

    @pytest.fixture
    def small_sim_data(self, small_genotypes):
        return _make_sim_data_for_genotypes(small_genotypes)

    def test_returns_all_genotypes(self, simple_csv, small_genotypes,
                                   small_sim_data, mock_sim_priors, rng):
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        assert set(result.keys()) == set(small_genotypes)

    def test_theta_shape(self, simple_csv, small_genotypes,
                         small_sim_data, mock_sim_priors, rng):
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        C = len(log_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        for g, theta in result.items():
            assert theta.shape == (C,), f"Shape mismatch for {g}"

    def test_theta_in_unit_interval(self, simple_csv, small_genotypes,
                                    small_sim_data, mock_sim_priors, rng):
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        for g, theta in result.items():
            assert np.all(theta >= 0) and np.all(theta <= 1), f"Out of [0,1] for {g}"

    def test_wt_receives_exact_wt_curve(self, simple_csv, small_genotypes,
                                        small_sim_data, mock_sim_priors, rng):
        """WT genotype has no mutations so its assembled curve = WT reference curve."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        # WT from CSV: theta_low=0.99, theta_high=0.01, log_hill_K=-4.1, hill_n=2.0
        expected_wt = _hill_theta(0.99, 0.01, -4.1, 2.0, log_conc)
        np.testing.assert_allclose(result["wt"], expected_wt, rtol=1e-5)

    def test_single_mutant_uses_measured_params(self, simple_csv, small_genotypes,
                                                small_sim_data, mock_sim_priors, rng):
        """A single-mutant entry in the CSV should produce its measured curve exactly."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        expected_a47v = _hill_theta(0.97, 0.03, -3.8, 1.8, log_conc)
        np.testing.assert_allclose(result["A47V"], expected_a47v, rtol=1e-5)

    def test_double_differs_from_wt(self, simple_csv, small_genotypes,
                                    small_sim_data, mock_sim_priors, rng):
        """Double mutant should differ from WT (deltas from both singles applied)."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        expected_wt = _hill_theta(0.99, 0.01, -4.1, 2.0, log_conc)
        assert not np.allclose(result["A47V/K84L"], expected_wt)

    def test_unmeasured_mut_uses_prior(self, wt_params, mock_sim_priors, rng):
        """A mutation not in the CSV gets a random delta from the prior."""
        genotypes = ["wt", "M99A"]  # M99A not in CSV
        sim_data = _make_sim_data_for_genotypes(genotypes)
        params_dict = {"wt": wt_params}  # only WT in CSV
        log_conc = np.array(sim_data.log_titrant_conc)

        r1, _ = build_theta_gc_override_hill_mut(
            params_dict, genotypes, sim_data, mock_sim_priors,
            log_conc, np.random.default_rng(1),
        )
        r2, _ = build_theta_gc_override_hill_mut(
            params_dict, genotypes, sim_data, mock_sim_priors,
            log_conc, np.random.default_rng(2),
        )
        # WT is pinned: same both times
        np.testing.assert_allclose(r1["wt"], r2["wt"])
        # M99A is random: different seeds → different curves
        assert not np.allclose(r1["M99A"], r2["M99A"])

    def test_csv_wt_overrides_sim_priors(self, small_genotypes, small_sim_data,
                                          mock_sim_priors, rng, tmp_path):
        """When CSV has a 'wt' row, that WT reference overrides sim_priors."""
        # Use a WT with very different log_hill_K
        content = (
            "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
            "wt,0.95,0.05,-6.0,1.5\n"  # dramatically different from sim_priors WT
        )
        p = tmp_path / "alt_wt.csv"
        p.write_text(content)
        params_dict = read_binding_genotype_params(str(p))

        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        expected_wt = _hill_theta(0.95, 0.05, -6.0, 1.5, log_conc)
        np.testing.assert_allclose(result["wt"], expected_wt, rtol=1e-5)

    def test_direct_double_in_csv_overrides_assembled(self, small_genotypes,
                                                       small_sim_data, mock_sim_priors,
                                                       rng, tmp_path):
        """A directly-measured double mutant in the CSV overrides the assembled value."""
        content = (
            "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
            "wt,0.99,0.01,-4.1,2.0\n"
            "A47V/K84L,0.90,0.10,-5.0,3.0\n"  # direct measurement for the double
        )
        p = tmp_path / "with_double.csv"
        p.write_text(content)
        params_dict = read_binding_genotype_params(str(p))

        log_conc = np.array(small_sim_data.log_titrant_conc)
        result, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        expected_double = _hill_theta(0.90, 0.10, -5.0, 3.0, log_conc)
        np.testing.assert_allclose(result["A47V/K84L"], expected_double, rtol=1e-5)

    def test_effective_params_all_genotypes(self, simple_csv, small_genotypes,
                                             small_sim_data, mock_sim_priors, rng):
        """effective_params covers all library genotypes."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        _, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        assert set(eff.keys()) == set(small_genotypes)
        for g, p in eff.items():
            assert set(p.keys()) == {"theta_low", "theta_high", "log_hill_K", "hill_n"}

    def test_effective_params_measured_singles(self, simple_csv, small_genotypes,
                                               small_sim_data, mock_sim_priors, rng):
        """effective_params for CSV-measured singles matches their CSV values."""
        params_dict = read_binding_genotype_params(simple_csv)
        log_conc = np.array(small_sim_data.log_titrant_conc)
        _, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        assert eff["wt"]["theta_low"]    == pytest.approx(0.99, abs=1e-4)
        assert eff["wt"]["theta_high"]   == pytest.approx(0.01, abs=1e-4)
        assert eff["A47V"]["log_hill_K"] == pytest.approx(-3.8, abs=1e-4)
        assert eff["K84L"]["hill_n"]     == pytest.approx(2.2,  abs=1e-4)

    def test_effective_params_direct_double(self, small_genotypes, small_sim_data,
                                            mock_sim_priors, rng, tmp_path):
        """effective_params for a directly-measured double uses the CSV values."""
        content = (
            "genotype,theta_low,theta_high,log_hill_K,hill_n\n"
            "wt,0.99,0.01,-4.1,2.0\n"
            "A47V/K84L,0.90,0.10,-5.0,3.0\n"
        )
        p = tmp_path / "with_double.csv"
        p.write_text(content)
        params_dict = read_binding_genotype_params(str(p))
        log_conc = np.array(small_sim_data.log_titrant_conc)
        _, eff = build_theta_gc_override_hill_mut(
            params_dict, small_genotypes, small_sim_data,
            mock_sim_priors, log_conc, rng,
        )
        assert eff["A47V/K84L"]["theta_low"]  == pytest.approx(0.90, abs=1e-4)
        assert eff["A47V/K84L"]["log_hill_K"] == pytest.approx(-5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# build_binding_theta_from_params
# ---------------------------------------------------------------------------

class TestBuildBindingThetaFromParams:

    @pytest.fixture
    def params_dict(self, simple_csv):
        return read_binding_genotype_params(simple_csv)

    def test_output_shape(self, params_dict, wt_params, rng):
        binding_concs = [0.0, 0.001, 0.01, 0.1, 1.0]
        df = build_binding_theta_from_params(
            params_dict, binding_concs, "iptg", noise=0.0, rng=rng, wt_params=wt_params
        )
        assert set(df.columns) == {"genotype", "titrant_name", "titrant_conc", "theta_true"}
        # 3 genotypes × 5 concentrations = 15 rows
        assert len(df) == 3 * len(binding_concs)

    def test_correct_titrant_name(self, params_dict, wt_params, rng):
        df = build_binding_theta_from_params(
            params_dict, [0.01], "iptg", noise=0.0, rng=rng, wt_params=wt_params
        )
        assert (df["titrant_name"] == "iptg").all()

    def test_no_noise(self, params_dict, wt_params, rng):
        """With noise=0, theta_true should exactly match Hill equation output."""
        binding_concs = [0.0, 0.001, 0.01]
        log_concs = _to_log_conc(binding_concs)
        df = build_binding_theta_from_params(
            params_dict, binding_concs, "iptg", noise=0.0, rng=rng, wt_params=wt_params
        )
        wt_rows = df[df["genotype"] == "wt"].sort_values("titrant_conc")
        expected = _hill_theta(0.99, 0.01, -4.1, 2.0, log_concs)
        np.testing.assert_allclose(wt_rows["theta_true"].values, expected, rtol=1e-6)

    def test_noise_changes_values(self, params_dict, wt_params):
        """With noise > 0, two runs with different seeds produce different theta_true."""
        binding_concs = [0.0, 0.001, 0.01, 0.1]
        df1 = build_binding_theta_from_params(
            params_dict, binding_concs, "iptg", noise=0.05,
            rng=np.random.default_rng(1), wt_params=wt_params,
        )
        df2 = build_binding_theta_from_params(
            params_dict, binding_concs, "iptg", noise=0.05,
            rng=np.random.default_rng(2), wt_params=wt_params,
        )
        assert not np.allclose(df1["theta_true"].values, df2["theta_true"].values)

    def test_theta_in_unit_interval(self, params_dict, wt_params):
        """theta_true must remain in [0, 1] even with noise."""
        binding_concs = np.linspace(0.0, 1.0, 20)
        df = build_binding_theta_from_params(
            params_dict, binding_concs, "iptg", noise=0.1,
            rng=np.random.default_rng(99), wt_params=wt_params,
        )
        assert (df["theta_true"] >= 0).all() and (df["theta_true"] <= 1).all()

    def test_nan_filled_from_wt(self, nan_csv, wt_params, rng):
        """Genotypes with NaN params should produce finite theta using WT fallback."""
        params_dict = read_binding_genotype_params(nan_csv)
        df = build_binding_theta_from_params(
            params_dict, [0.0, 0.01, 1.0], "iptg", noise=0.0, rng=rng, wt_params=wt_params,
        )
        assert np.all(np.isfinite(df["theta_true"].values))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_supported_components():
    assert "hill_geno" in SUPPORTED_COMPONENTS
    assert "hill_mut" in SUPPORTED_COMPONENTS

def test_hill_param_cols():
    assert HILL_PARAM_COLS == {"theta_low", "theta_high", "log_hill_K", "hill_n"}
