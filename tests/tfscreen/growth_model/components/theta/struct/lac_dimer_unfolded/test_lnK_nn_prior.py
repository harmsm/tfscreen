"""Tests for struct/lac_dimer_unfolded/lnK_nn_prior.py."""

import numpy as np
import jax.numpy as jnp
import pytest
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from numpyro.handlers import trace, seed

from tfscreen.genetics.build_mut_geno_matrix import build_mut_sparse_indices
from tfscreen.growth_model.components.theta.struct.lac_dimer_unfolded.lnK_nn_prior import (
    STRUCTURE_NAMES,
    ModelPriors,
    _check_struct_names,
    _project_ddG,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    get_extract_specs,
)
from tfscreen.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
    ThetaParam,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_S = 4   # number of structures: H, HD, L, LE2
_M = 2   # number of mutations
_G = 4   # genotypes: wt, mut0, mut1, double

_MUT_GENO = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1]], dtype=np.float32)  # (M, G)
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)

# Single pair: mut0+mut1 present only in genotype 3
_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

_CONC = np.array([0.0, 1e-4, 1e-3], dtype=np.float32)  # (C,)


def _make_features(seed=0):
    """Small (M, S, 60) feature array."""
    rng = np.random.RandomState(seed)
    return rng.randn(_M, _S, 60).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# MockData
# ──────────────────────────────────────────────────────────────────────────────

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "titrant_conc",
    "log_titrant_conc",
    "geno_theta_idx",
    "scatter_theta",
    "num_mutation",
    "num_pair",
    "mut_geno_matrix",
    "mut_nnz_mut_idx",
    "mut_nnz_geno_idx",
    "pair_nnz_pair_idx",
    "pair_nnz_geno_idx",
    # struct fields
    "num_struct",
    "struct_names",
    "struct_features",
    "struct_n_chains",
    "struct_contact_pair_idx",
    "struct_contact_distances",
])

_NO_DISTANCES = object()  # sentinel: auto-fill contact_distances when num_pair > 0


def _make_mock(num_pair=0, features=None, n_chains=None,
               contact_distances=_NO_DISTANCES, struct_names=None):
    if features is None:
        features = _make_features()
    if n_chains is None:
        n_chains = np.array([2, 2, 2, 2], dtype=np.int32)
    if struct_names is None:
        struct_names = STRUCTURE_NAMES

    pair_nnz_pair = _PAIR_NNZ_PAIR if num_pair > 0 else np.zeros(0, dtype=np.int32)
    pair_nnz_geno = _PAIR_NNZ_GENO if num_pair > 0 else np.zeros(0, dtype=np.int32)

    # Sentinel → auto-fill; explicit None → keep as None (no contact distances)
    if contact_distances is _NO_DISTANCES:
        contact_distances = (
            np.full((num_pair, _S), 5.0, dtype=np.float32) if num_pair > 0 else None
        )

    return MockData(
        num_titrant_name=2,
        num_titrant_conc=len(_CONC),
        num_genotype=_G,
        titrant_conc=jnp.array(_CONC),
        log_titrant_conc=jnp.log(jnp.where(jnp.array(_CONC) == 0, 1e-20, jnp.array(_CONC))),
        geno_theta_idx=jnp.arange(_G, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=_M,
        num_pair=num_pair,
        mut_geno_matrix=_MUT_GENO,
        mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=pair_nnz_pair,
        pair_nnz_geno_idx=pair_nnz_geno,
        num_struct=_S,
        struct_names=struct_names,
        struct_features=features,
        struct_n_chains=n_chains,
        struct_contact_pair_idx=None,
        struct_contact_distances=contact_distances,
    )


# ──────────────────────────────────────────────────────────────────────────────
# _check_struct_names
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckStructNames:
    def test_correct_names_pass(self):
        data = _make_mock()
        _check_struct_names(data)  # no exception

    def test_wrong_order_raises(self):
        data = _make_mock(struct_names=('HD', 'H', 'L', 'LE2'))
        with pytest.raises(ValueError, match="struct_names"):
            _check_struct_names(data)

    def test_missing_structure_raises(self):
        data = _make_mock(struct_names=('H', 'HD', 'L'))
        with pytest.raises(ValueError, match="struct_names"):
            _check_struct_names(data)

    def test_structure_names_constant(self):
        assert STRUCTURE_NAMES == ('H', 'HD', 'L', 'LE2')


# ──────────────────────────────────────────────────────────────────────────────
# _project_ddG
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectDdG:
    def test_output_shape(self):
        ddG = jnp.zeros((5, 4))
        out = _project_ddG(ddG)
        assert out.shape == (5, 3)

    def test_zero_ddG_gives_zero_delta_lnK(self):
        ddG = jnp.zeros((3, 4))
        np.testing.assert_allclose(_project_ddG(ddG), 0.0)

    def test_K_op_equals_H_minus_HD(self):
        ddG = jnp.array([[1.0, 2.0, 0.0, 0.0]])   # H=1, HD=2, L=0, LE2=0
        out = _project_ddG(ddG)
        assert out[0, 0] == pytest.approx(1.0 - 2.0)  # K_op = H - HD

    def test_K_HL_equals_H_minus_L(self):
        ddG = jnp.array([[3.0, 0.0, 1.0, 0.0]])
        out = _project_ddG(ddG)
        assert out[0, 1] == pytest.approx(3.0 - 1.0)  # K_HL = H - L

    def test_K_E_equals_L_minus_LE2(self):
        ddG = jnp.array([[0.0, 0.0, 5.0, 2.0]])
        out = _project_ddG(ddG)
        assert out[0, 2] == pytest.approx(5.0 - 2.0)  # K_E = L - LE2

    def test_leading_batch_dim(self):
        ddG = jnp.ones((2, 7, 4))
        out = _project_ddG(ddG)
        assert out.shape == (2, 7, 3)


# ──────────────────────────────────────────────────────────────────────────────
# get_hyperparameters / get_priors / get_guesses
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_hyperparameters_keys(self):
        hp = get_hyperparameters()
        required = [
            "theta_ln_K_op_wt_loc", "theta_ln_K_op_wt_scale",
            "theta_ln_K_HL_wt_loc", "theta_ln_K_HL_wt_scale",
            "theta_ln_K_E_wt_loc",  "theta_ln_K_E_wt_scale",
            "theta_ln_K_U_wt_loc",  "theta_ln_K_U_wt_scale",
            "theta_tf_total_M", "theta_op_total_M",
            "theta_nn_hidden_size",
            "theta_epi_tau_scale", "theta_epi_slab_scale", "theta_epi_slab_df",
            "theta_epi_d0",
        ]
        for k in required:
            assert k in hp, f"Missing key: {k}"

    def test_ln_K_U_wt_defaults(self):
        hp = get_hyperparameters()
        assert hp["theta_ln_K_U_wt_loc"]   == pytest.approx(-12.0)
        assert hp["theta_ln_K_U_wt_scale"] == pytest.approx(3.0)

    def test_get_priors_type(self):
        assert isinstance(get_priors(), ModelPriors)

    def test_priors_has_ln_K_U_wt_fields(self):
        priors = get_priors()
        assert hasattr(priors, "theta_ln_K_U_wt_loc")
        assert hasattr(priors, "theta_ln_K_U_wt_scale")
        assert priors.theta_ln_K_U_wt_loc   == pytest.approx(-12.0)
        assert priors.theta_ln_K_U_wt_scale == pytest.approx(3.0)

    def test_physical_constants(self):
        priors = get_priors()
        assert priors.theta_tf_total_M == pytest.approx(6.5e-7)
        assert priors.theta_op_total_M == pytest.approx(2.5e-8)

    def test_get_guesses_keys_no_epi(self):
        data = _make_mock()
        g = get_guesses("theta", data)
        assert "theta_ln_K_op_wt"  in g
        assert "theta_ln_K_HL_wt"  in g
        assert "theta_ln_K_E_wt"   in g
        assert "theta_ln_K_U_wt"   in g
        assert "theta_ddG_offset"  in g
        assert "theta_epi_tau" not in g

    def test_get_guesses_ln_K_U_wt_value(self):
        data = _make_mock()
        g = get_guesses("theta", data)
        assert float(g["theta_ln_K_U_wt"]) == pytest.approx(-12.0)

    def test_get_guesses_keys_with_epi(self):
        data = _make_mock(num_pair=1)
        g = get_guesses("theta", data)
        assert "theta_epi_tau"    in g
        assert "theta_epi_c2"     in g
        assert "theta_epi_lambda" in g
        assert "theta_epi_offset" in g

    def test_get_guesses_ddG_offset_shape(self):
        data = _make_mock()
        g = get_guesses("theta", data)
        assert g["theta_ddG_offset"].shape == (_S, _M)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for model / guide execution
# ──────────────────────────────────────────────────────────────────────────────

def _run_model(data, name="theta", seed_val=0):
    priors = get_priors()
    def model():
        return define_model(name, data, priors)
    with seed(rng_seed=seed_val):
        tr = trace(model).get_trace()
    with seed(rng_seed=seed_val):
        out = model()
    return out, tr


def _run_guide(data, name="theta", seed_val=0):
    priors = get_priors()
    def _guide():
        return guide(name, data, priors)
    with seed(rng_seed=seed_val):
        tr = trace(_guide).get_trace()
    with seed(rng_seed=seed_val):
        out = _guide()
    return out, tr


# ──────────────────────────────────────────────────────────────────────────────
# define_model
# ──────────────────────────────────────────────────────────────────────────────

class TestDefineModel:
    def test_returns_theta_param(self):
        data = _make_mock()
        out, _ = _run_model(data)
        assert isinstance(out, ThetaParam)

    def test_ln_K_op_shape(self):
        data = _make_mock()
        out, _ = _run_model(data)
        assert out.ln_K_op.shape == (_G,)

    def test_ln_K_HL_shape(self):
        data = _make_mock()
        out, _ = _run_model(data)
        assert out.ln_K_HL.shape == (_G,)

    def test_ln_K_E_shape(self):
        data = _make_mock()
        out, _ = _run_model(data)
        assert out.ln_K_E.shape == (data.num_titrant_name, _G)

    def test_ln_K_U_shape(self):
        data = _make_mock()
        out, _ = _run_model(data)
        assert out.ln_K_U.shape == (_G,)

    def test_ln_K_U_homogeneous(self):
        """All elements of ln_K_U must be equal — K_U is WT-only with no per-mut effects."""
        data = _make_mock()
        out, _ = _run_model(data)
        vals = np.asarray(out.ln_K_U)
        np.testing.assert_allclose(vals, vals[0], atol=1e-6,
                                   err_msg="ln_K_U should be homogeneous across genotypes")

    def test_sample_sites_present(self):
        data = _make_mock()
        _, tr = _run_model(data)
        assert "theta_ln_K_op_wt" in tr
        assert "theta_ln_K_HL_wt" in tr
        assert "theta_ln_K_E_wt"  in tr
        assert "theta_ddG_offset" in tr
        assert "theta_ln_K_U_wt"  in tr

    def test_deterministic_sites_present(self):
        data = _make_mock()
        _, tr = _run_model(data)
        for site in ("theta_ddG", "theta_d_ln_K_op", "theta_d_ln_K_HL", "theta_d_ln_K_E",
                     "theta_ln_K_op", "theta_ln_K_HL", "theta_ln_K_E", "theta_ln_K_U"):
            assert site in tr, f"Missing deterministic site: {site}"

    def test_wrong_struct_names_raises(self):
        data = _make_mock(struct_names=('HD', 'H', 'L', 'LE2'))
        with pytest.raises(ValueError, match="struct_names"):
            _run_model(data)

    def test_no_epi_sites_when_num_pair_zero(self):
        data = _make_mock(num_pair=0)
        _, tr = _run_model(data)
        assert "theta_epi_tau"    not in tr
        assert "theta_epi_lambda" not in tr
        assert "theta_epi_offset" not in tr

    def test_epi_sites_present_when_num_pair_nonzero(self):
        data = _make_mock(num_pair=1)
        _, tr = _run_model(data)
        assert "theta_epi_tau"    in tr
        assert "theta_epi_c2"     in tr
        assert "theta_epi_lambda" in tr
        assert "theta_epi_offset" in tr

    def test_epi_sites_absent_when_no_contact_distances(self):
        """num_pair > 0 but contact_distances=None → no epistasis sampled."""
        data = _make_mock(num_pair=1, contact_distances=None)
        _, tr = _run_model(data)
        assert "theta_epi_tau" not in tr

    def test_zero_nn_weights_zero_offsets_all_genotypes_at_wt(self):
        """With NN weights=0 and ddG_offset=0, all mutations have zero effect."""
        data = _make_mock()
        priors = get_priors()

        import unittest.mock as mock

        def mock_param(name, init, **kwargs):
            return jnp.zeros_like(init)

        with mock.patch("numpyro.param", side_effect=mock_param):
            with seed(rng_seed=0):
                out = define_model("theta", data, priors)

        # All four genotypes should have the same K_op and K_HL when all effects are zero
        np.testing.assert_allclose(
            np.asarray(out.ln_K_op[0]), np.asarray(out.ln_K_op[1]), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(out.ln_K_op[1]), np.asarray(out.ln_K_op[2]), atol=1e-5
        )

    def test_ln_K_U_homogeneous_with_epi(self):
        """K_U homogeneity holds even when epistasis is active."""
        data = _make_mock(num_pair=1)
        out, _ = _run_model(data)
        vals = np.asarray(out.ln_K_U)
        np.testing.assert_allclose(vals, vals[0], atol=1e-6)

    def test_tf_op_totals_preserved(self):
        data = _make_mock()
        out, _ = _run_model(data)
        assert out.tf_total == pytest.approx(6.5e-7)
        assert out.op_total == pytest.approx(2.5e-8)

    def test_population_moment_shapes(self):
        data = _make_mock()
        out, _ = _run_model(data)
        T, C = data.num_titrant_name, len(_CONC)
        assert out.mu.shape    == (T, C, 1)
        assert out.sigma.shape == (T, C, 1)


# ──────────────────────────────────────────────────────────────────────────────
# guide
# ──────────────────────────────────────────────────────────────────────────────

class TestGuide:
    def test_returns_theta_param(self):
        data = _make_mock()
        out, _ = _run_guide(data)
        assert isinstance(out, ThetaParam)

    def test_ln_K_U_shape(self):
        data = _make_mock()
        out, _ = _run_guide(data)
        assert out.ln_K_U.shape == (_G,)

    def test_ln_K_U_homogeneous_in_guide(self):
        data = _make_mock()
        out, _ = _run_guide(data)
        vals = np.asarray(out.ln_K_U)
        np.testing.assert_allclose(vals, vals[0], atol=1e-6)

    def test_sample_sites_match_model(self):
        """Guide must contain exactly the same pyro.sample site names as the model."""
        data = _make_mock()
        _, model_tr = _run_model(data)
        _, guide_tr = _run_guide(data)

        model_samples = {k for k, v in model_tr.items() if v["type"] == "sample"}
        guide_samples = {k for k, v in guide_tr.items() if v["type"] == "sample"}
        assert model_samples == guide_samples

    def test_sample_sites_match_model_with_epi(self):
        data = _make_mock(num_pair=1)
        _, model_tr = _run_model(data)
        _, guide_tr = _run_guide(data)

        model_samples = {k for k, v in model_tr.items() if v["type"] == "sample"}
        guide_samples = {k for k, v in guide_tr.items() if v["type"] == "sample"}
        assert model_samples == guide_samples

    def test_variational_params_registered(self):
        data = _make_mock()
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert "theta_ln_K_op_wt_loc"    in param_names
        assert "theta_ln_K_op_wt_scale"  in param_names
        assert "theta_ln_K_HL_wt_loc"    in param_names
        assert "theta_ln_K_HL_wt_scale"  in param_names
        assert "theta_ddG_offset_locs"   in param_names
        assert "theta_ddG_offset_scales" in param_names

    def test_ln_K_U_wt_variational_params_registered(self):
        """Guide must register variational parameters for ln_K_U_wt."""
        data = _make_mock()
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert "theta_ln_K_U_wt_loc"   in param_names
        assert "theta_ln_K_U_wt_scale" in param_names

    def test_sigma_s_param_registered_in_guide(self):
        data = _make_mock()
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert "theta_ddG_sigma_s" in param_names

    def test_nn_params_registered_in_guide(self):
        data = _make_mock()
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        # Expect per-structure W1 params from compute_nn_predictions
        for sname in STRUCTURE_NAMES:
            assert f"theta_nn_{sname}_W1" in param_names, f"Missing theta_nn_{sname}_W1"

    def test_tf_op_totals_preserved(self):
        data = _make_mock()
        out, _ = _run_guide(data)
        assert out.tf_total == pytest.approx(6.5e-7)
        assert out.op_total == pytest.approx(2.5e-8)

    def test_K_shapes(self):
        data = _make_mock()
        out, _ = _run_guide(data)
        assert out.ln_K_op.shape == (_G,)
        assert out.ln_K_HL.shape == (_G,)
        assert out.ln_K_E.shape  == (data.num_titrant_name, _G)
        assert out.ln_K_U.shape  == (_G,)


# ──────────────────────────────────────────────────────────────────────────────
# Registry smoke test
# ──────────────────────────────────────────────────────────────────────────────

def test_registry_entry():
    from tfscreen.growth_model.registry import model_registry
    assert "lac_dimer_unfolded_lnK_nn_prior" in model_registry["theta"]
    import tfscreen.growth_model.components.theta.struct.lac_dimer_unfolded.lnK_nn_prior as mod
    assert model_registry["theta"]["lac_dimer_unfolded_lnK_nn_prior"] is mod


# ──────────────────────────────────────────────────────────────────────────────
# get_extract_specs
# ──────────────────────────────────────────────────────────────────────────────

class _FakeTM:
    """Minimal TensorManager stand-in for get_extract_specs."""
    def __init__(self, genotypes, titrant_names):
        self.tensor_dim_names  = ["genotype", "titrant_name"]
        self.tensor_dim_labels = [np.array(genotypes), np.array(titrant_names)]
        rows = [
            {"genotype": g, "titrant_name": t,
             "genotype_idx": gi, "titrant_name_idx": ti}
            for gi, g in enumerate(genotypes)
            for ti, t in enumerate(titrant_names)
        ]
        self.df = pd.DataFrame(rows)


@dataclass
class _FakeCtx:
    growth_tm: object
    mut_labels: list
    pair_labels: list
    growth_shares_replicates: bool = False


def _make_ctx(genotypes=None, titrant_names=None, mut_labels=None, pair_labels=None):
    if genotypes is None:
        genotypes = ["wt", "M42I", "K84L"]
    if titrant_names is None:
        titrant_names = ["iptg"]
    if mut_labels is None:
        mut_labels = ["M42I", "K84L"]
    if pair_labels is None:
        pair_labels = []
    return _FakeCtx(
        growth_tm=_FakeTM(genotypes, titrant_names),
        mut_labels=mut_labels,
        pair_labels=pair_labels,
    )


class TestGetExtractSpecs:
    def test_returns_list(self):
        ctx = _make_ctx()
        specs = get_extract_specs(ctx)
        assert isinstance(specs, list)

    def test_has_three_specs_without_epi(self):
        ctx = _make_ctx()
        specs = get_extract_specs(ctx)
        assert len(specs) == 3

    def test_has_four_specs_with_epi(self):
        ctx = _make_ctx(
            genotypes=["wt", "M42I", "K84L", "M42I/K84L"],
            mut_labels=["M42I", "K84L"],
            pair_labels=["K84L/M42I"],
        )
        specs = get_extract_specs(ctx)
        assert len(specs) == 4

    def test_first_spec_has_ln_K_op_and_ln_K_HL(self):
        ctx = _make_ctx()
        specs = get_extract_specs(ctx)
        assert "ln_K_op" in specs[0]["params_to_get"]
        assert "ln_K_HL" in specs[0]["params_to_get"]

    def test_second_spec_has_ln_K_E(self):
        ctx = _make_ctx()
        specs = get_extract_specs(ctx)
        assert "ln_K_E" in specs[1]["params_to_get"]

    def test_third_spec_has_d_ln_K(self):
        ctx = _make_ctx()
        specs = get_extract_specs(ctx)
        d_params = specs[2]["params_to_get"]
        assert "d_ln_K_op" in d_params
        assert "d_ln_K_HL" in d_params
        assert "d_ln_K_E"  in d_params

    def test_fourth_spec_has_epi_terms(self):
        ctx = _make_ctx(
            genotypes=["wt", "M42I", "K84L", "M42I/K84L"],
            mut_labels=["M42I", "K84L"],
            pair_labels=["K84L/M42I"],
        )
        specs = get_extract_specs(ctx)
        epi_params = specs[3]["params_to_get"]
        assert "epi_ln_K_op" in epi_params
        assert "epi_ln_K_HL" in epi_params
        assert "epi_ln_K_E"  in epi_params

    def test_map_geno_covers_all_genotypes(self):
        genos = ["wt", "M42I", "K84L"]
        ctx = _make_ctx(genotypes=genos)
        specs = get_extract_specs(ctx)
        df0 = specs[0]["input_df"]
        assert set(df0["genotype"]) == set(genos)

    def test_map_theta_KE_covers_all_titrant_genotype_combos(self):
        genos    = ["wt", "M42I"]
        titrants = ["iptg", "tmg"]
        ctx = _make_ctx(genotypes=genos, titrant_names=titrants)
        specs = get_extract_specs(ctx)
        df1 = specs[1]["input_df"]
        assert len(df1) == len(genos) * len(titrants)

    def test_in_run_prefix_is_theta(self):
        ctx = _make_ctx()
        specs = get_extract_specs(ctx)
        for spec in specs:
            assert spec["in_run_prefix"] == "theta_"

    def test_third_spec_has_one_row_per_mutation(self):
        ctx = _make_ctx(mut_labels=["M42I", "K84L", "T71I"])
        specs = get_extract_specs(ctx)
        df2 = specs[2]["input_df"]
        assert len(df2) == 3
