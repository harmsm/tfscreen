"""
Tests for struct/mwc_dimer_unfolded/lnK_ddG_prior.py.

Covers: _get_struct_perm, _project_ddG, ModelPriors, get_hyperparameters,
get_priors, get_guesses, define_model, guide, get_extract_specs.

Key differences from mwc_dimer variant:
- ThetaParam gains ln_K_u (G,) and conc_unit_scale fields
- ModelPriors gains theta_ln_K_u_wt_loc and theta_ln_K_u_wt_scale
- define_model samples theta_ln_K_u_wt and broadcasts it homogeneously
- ln_K_u[g] == ln_K_u_wt for all g (no per-mutation unfolding effect)
- guide samples theta_ln_K_u_wt with variational params theta_ln_K_u_wt_loc/scale
- struct_features is (M, S) float32 ddG prior means (no NN, no epistasis)
- get_extract_specs always returns 3 specs (no epistasis spec)
- first spec includes ln_K_u among the scalar K parameters
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from collections import namedtuple
from dataclasses import dataclass
from numpyro.handlers import trace, seed
import unittest.mock as mock

from tfscreen.genetics.build_mut_geno_matrix import build_mut_sparse_indices
from tfscreen.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_ddG_prior import (
    STRUCTURE_NAMES,
    ModelPriors,
    _get_struct_perm,
    _project_ddG,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors,
    get_extract_specs,
)
from tfscreen.growth_model.components.theta.struct.mwc_dimer_unfolded.thermo import (
    ThetaParam,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S = 6    # number of MWC structures (H, HO, L, LO, HE2, LE2)
_M = 3    # mutations
_G = 4    # genotypes: wt, M1, M2, M1+M2

_MUT_GENO = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0]], dtype=np.float32)  # (M, G)
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)

_CONC = np.array([0.0, 1e-5, 1e-4], dtype=np.float32)

assert set(STRUCTURE_NAMES) == {'H', 'HO', 'L', 'LO', 'HE2', 'LE2'}


def _make_ddG_prior(seed_val=0, m=_M, s=_S):
    rng = np.random.RandomState(seed_val)
    return rng.randn(m, s).astype(np.float32)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

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
    "num_struct",
    "struct_names",
    "struct_features",   # shape (M, S) float32 — ddG prior means
    "struct_n_chains",   # None for this model
    "struct_contact_pair_idx",
    "struct_contact_distances",
])


def _make_mock(struct_names=None, ddG_prior=None):
    if struct_names is None:
        struct_names = STRUCTURE_NAMES
    if ddG_prior is None:
        ddG_prior = _make_ddG_prior()

    return MockData(
        num_titrant_name=2,
        num_titrant_conc=len(_CONC),
        num_genotype=_G,
        titrant_conc=jnp.array(_CONC),
        log_titrant_conc=jnp.log(jnp.where(jnp.array(_CONC) == 0, 1e-20,
                                            jnp.array(_CONC))),
        geno_theta_idx=jnp.arange(_G, dtype=jnp.int32),
        scatter_theta=1,
        num_mutation=_M,
        num_pair=0,
        mut_geno_matrix=_MUT_GENO,
        mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=np.zeros(0, dtype=np.int32),
        pair_nnz_geno_idx=np.zeros(0, dtype=np.int32),
        num_struct=_S,
        struct_names=struct_names,
        struct_features=ddG_prior,
        struct_n_chains=None,
        struct_contact_pair_idx=None,
        struct_contact_distances=None,
    )


# ---------------------------------------------------------------------------
# _get_struct_perm
# ---------------------------------------------------------------------------

class TestGetStructPerm:

    def test_canonical_order_gives_identity_perm(self):
        data = _make_mock(struct_names=STRUCTURE_NAMES)
        perm = _get_struct_perm(data)
        assert perm == list(range(_S))

    def test_permuted_order_gives_correct_perm(self):
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        perm = _get_struct_perm(data)
        for i, sname in enumerate(STRUCTURE_NAMES):
            assert shuffled[perm[i]] == sname

    def test_perm_reindexes_features_correctly(self):
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        l_canonical_idx = list(STRUCTURE_NAMES).index('L')   # 2
        l_shuffled_idx  = list(shuffled).index('L')          # 3
        data = _make_mock(struct_names=shuffled)
        perm = _get_struct_perm(data)
        assert perm[l_canonical_idx] == l_shuffled_idx

    def test_wrong_name_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)

    def test_too_few_names_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)


# ---------------------------------------------------------------------------
# _project_ddG
# ---------------------------------------------------------------------------

class TestProjectDdG:
    """
    Δln_K_h_l = ΔΔG_H − ΔΔG_L
    Δln_K_h_o = ΔΔG_H − ΔΔG_HO
    Δln_K_h_e = (ΔΔG_H − ΔΔG_HE2) / 2
    Δln_K_l_o = ΔΔG_L − ΔΔG_LO
    Δln_K_l_e = (ΔΔG_L − ΔΔG_LE2) / 2
    """

    def test_output_shape(self):
        assert _project_ddG(jnp.zeros((_M, _S))).shape == (_M, 5)

    def test_zero_ddG_gives_zero_delta_lnK(self):
        np.testing.assert_allclose(_project_ddG(jnp.zeros((3, _S))), 0.0)

    def test_K_h_l_is_H_minus_L(self):
        ddG = jnp.array([[3.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 0] == pytest.approx(3.0 - 1.0)

    def test_K_h_o_is_H_minus_HO(self):
        ddG = jnp.array([[4.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 1] == pytest.approx(4.0 - 2.0)

    def test_K_h_e_is_H_minus_HE2_halved(self):
        ddG = jnp.array([[6.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        assert _project_ddG(ddG)[0, 2] == pytest.approx((6.0 - 2.0) / 2.0)

    def test_K_l_o_is_L_minus_LO(self):
        ddG = jnp.array([[0.0, 0.0, 5.0, 1.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 3] == pytest.approx(5.0 - 1.0)

    def test_K_l_e_is_L_minus_LE2_halved(self):
        ddG = jnp.array([[0.0, 0.0, 8.0, 0.0, 0.0, 2.0]])
        assert _project_ddG(ddG)[0, 4] == pytest.approx((8.0 - 2.0) / 2.0)

    def test_H_perturbation_does_not_affect_l_o_or_l_e(self):
        ddG_base = jnp.zeros((1, _S))
        ddG_H    = ddG_base.at[0, 0].set(1.0)
        out_base = _project_ddG(ddG_base)
        out_H    = _project_ddG(ddG_H)
        assert out_H[0, 3] == pytest.approx(float(out_base[0, 3]))  # K_l_o
        assert out_H[0, 4] == pytest.approx(float(out_base[0, 4]))  # K_l_e


# ---------------------------------------------------------------------------
# ModelPriors / get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

class TestConfig:

    def test_hyperparameters_has_required_keys(self):
        hp = get_hyperparameters()
        required = [
            "theta_ln_K_h_l_wt_loc", "theta_ln_K_h_l_wt_scale",
            "theta_ln_K_h_o_wt_loc", "theta_ln_K_h_o_wt_scale",
            "theta_ln_K_l_o_wt_loc", "theta_ln_K_l_o_wt_scale",
            "theta_ln_K_h_e_wt_loc", "theta_ln_K_h_e_wt_scale",
            "theta_ln_K_l_e_wt_loc", "theta_ln_K_l_e_wt_scale",
            # unfolding constant — new in mwc_dimer_unfolded
            "theta_ln_K_u_wt_loc", "theta_ln_K_u_wt_scale",
            "theta_tf_total_M", "theta_op_total_M", "theta_conc_unit_scale",
        ]
        for k in required:
            assert k in hp, f"Missing key: {k}"

    def test_ln_K_u_prior_values(self):
        hp = get_hyperparameters()
        assert hp["theta_ln_K_u_wt_loc"]   == pytest.approx(-12.0)
        assert hp["theta_ln_K_u_wt_scale"] == pytest.approx(3.0)

    def test_hyperparameters_has_no_nn_keys(self):
        hp = get_hyperparameters()
        assert "theta_nn_hidden_size"  not in hp

    def test_get_priors_returns_model_priors_instance(self):
        assert isinstance(get_priors(), ModelPriors)

    def test_model_priors_has_ln_K_u_fields(self):
        p = get_priors()
        assert hasattr(p, "theta_ln_K_u_wt_loc")
        assert hasattr(p, "theta_ln_K_u_wt_scale")

    def test_model_priors_has_no_nn_fields(self):
        p = get_priors()
        assert not hasattr(p, "theta_nn_hidden_size")

    def test_physical_constants(self):
        p = get_priors()
        assert p.theta_tf_total_M == pytest.approx(6.5e-7)
        assert p.theta_op_total_M == pytest.approx(2.5e-8)

    def test_get_guesses_required_keys(self):
        g = get_guesses("theta", _make_mock())
        for k in ("theta_ln_K_h_l_wt", "theta_ln_K_h_o_wt", "theta_ln_K_l_o_wt",
                  "theta_ln_K_h_e_wt", "theta_ln_K_l_e_wt",
                  "theta_ln_K_u_wt",   # new in mwc_dimer_unfolded
                  "theta_ddG_offset"):
            assert k in g, f"Missing guess key: {k}"

    def test_get_guesses_ln_K_u_wt_value(self):
        g = get_guesses("theta", _make_mock())
        assert float(g["theta_ln_K_u_wt"]) == pytest.approx(-12.0)

    def test_get_guesses_no_epi_keys(self):
        g = get_guesses("theta", _make_mock())
        assert "theta_epi_tau"    not in g
        assert "theta_epi_lambda" not in g
        assert "theta_epi_offset" not in g

    def test_get_guesses_ddG_offset_shape(self):
        assert get_guesses("theta", _make_mock())["theta_ddG_offset"].shape == (_S, _M)

    def test_get_guesses_wt_K_shapes(self):
        T = 2
        g = get_guesses("theta", _make_mock())
        assert g["theta_ln_K_h_l_wt"].shape == ()
        assert g["theta_ln_K_h_e_wt"].shape == (T,)
        assert g["theta_ln_K_l_e_wt"].shape == (T,)
        assert g["theta_ln_K_u_wt"].shape   == ()


# ---------------------------------------------------------------------------
# Helpers for model / guide execution
# ---------------------------------------------------------------------------

def _run_model(data, name="theta", seed_val=0):
    priors = get_priors()
    def _model():
        return define_model(name, data, priors)
    with seed(rng_seed=seed_val):
        tr = trace(_model).get_trace()
    with seed(rng_seed=seed_val):
        out = _model()
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


# ---------------------------------------------------------------------------
# define_model
# ---------------------------------------------------------------------------

class TestDefineModel:

    def test_returns_theta_param(self):
        out, _ = _run_model(_make_mock())
        assert isinstance(out, ThetaParam)

    def test_K_shapes(self):
        out, _ = _run_model(_make_mock())
        T = 2
        assert out.ln_K_h_l.shape == (_G,)
        assert out.ln_K_h_o.shape == (_G,)
        assert out.ln_K_l_o.shape == (_G,)
        assert out.ln_K_h_e.shape == (T, _G)
        assert out.ln_K_l_e.shape == (T, _G)

    def test_ln_K_u_shape(self):
        out, _ = _run_model(_make_mock())
        assert out.ln_K_u.shape == (_G,)

    def test_ln_K_u_is_homogeneous(self):
        """ln_K_u[g] must be identical for all genotypes (no per-mutation unfolding)."""
        out, _ = _run_model(_make_mock())
        vals = np.asarray(out.ln_K_u)
        np.testing.assert_allclose(vals, vals[0], atol=1e-6,
                                   err_msg="ln_K_u is not homogeneous across genotypes")

    def test_conc_unit_scale_preserved(self):
        out, _ = _run_model(_make_mock())
        assert out.conc_unit_scale == pytest.approx(1e-3)

    def test_population_moment_shapes(self):
        out, _ = _run_model(_make_mock())
        T, C = 2, len(_CONC)
        assert out.mu.shape    == (T, C, 1)
        assert out.sigma.shape == (T, C, 1)

    def test_sample_sites_present(self):
        _, tr = _run_model(_make_mock())
        for site in ("theta_ln_K_h_l_wt", "theta_ln_K_h_o_wt", "theta_ln_K_l_o_wt",
                     "theta_ln_K_h_e_wt", "theta_ln_K_l_e_wt",
                     "theta_ln_K_u_wt",   # new in mwc_dimer_unfolded
                     "theta_ddG_offset"):
            assert site in tr, f"Missing sample site: {site}"

    def test_deterministic_sites_present(self):
        _, tr = _run_model(_make_mock())
        for site in ("theta_ddG", "theta_d_ln_K_h_l", "theta_d_ln_K_h_o",
                     "theta_d_ln_K_h_e", "theta_d_ln_K_l_o", "theta_d_ln_K_l_e",
                     "theta_ln_K_h_l", "theta_ln_K_h_o", "theta_ln_K_h_e",
                     "theta_ln_K_l_o", "theta_ln_K_l_e",
                     "theta_ln_K_u"):   # new deterministic site
            assert site in tr, f"Missing deterministic site: {site}"

    def test_theta_ln_K_u_det_site_is_homogeneous(self):
        """The theta_ln_K_u deterministic site must be constant across all genotypes."""
        _, tr = _run_model(_make_mock())
        ln_K_u_vals = np.asarray(tr["theta_ln_K_u"]["value"])
        np.testing.assert_allclose(ln_K_u_vals, ln_K_u_vals[0], atol=1e-6)

    def test_no_epi_sites(self):
        """lnK_ddG_prior has no epistasis — ever."""
        _, tr = _run_model(_make_mock())
        assert "theta_epi_tau"    not in tr
        assert "theta_epi_lambda" not in tr
        assert "theta_epi_offset" not in tr

    def test_wrong_struct_names_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _run_model(data)

    def test_permuted_struct_names_accepted(self):
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        out, _ = _run_model(data)
        assert isinstance(out, ThetaParam)

    def test_ddG_prior_means_shift_delta_lnK(self):
        """With sigma_s=0, ddG equals the prior means exactly; nonzero prior produces
        nonzero delta_lnK and makes genotypes with that mutation differ from WT."""
        zero_prior = np.zeros((_M, _S), dtype=np.float32)
        data_zero  = _make_mock(ddG_prior=zero_prior)
        priors = get_priors()

        def _zero_param(name, init, **kwargs):
            return jnp.zeros_like(init)

        with mock.patch("numpyro.param", side_effect=_zero_param):
            with seed(rng_seed=0):
                out_zero = define_model("theta", data_zero, priors)

        # All mutations zero → genotype 0 (WT) and genotype 1 (M1) identical K_h_l
        np.testing.assert_allclose(
            np.asarray(out_zero.ln_K_h_l[0]),
            np.asarray(out_zero.ln_K_h_l[1]), atol=1e-5,
        )

        # Nonzero prior on H structure for mutation 0 should separate M1 from WT
        nonzero_prior = np.zeros((_M, _S), dtype=np.float32)
        nonzero_prior[0, 0] = 5.0   # H structure, mutation 0
        data_nonzero = _make_mock(ddG_prior=nonzero_prior)

        with mock.patch("numpyro.param", side_effect=_zero_param):
            with seed(rng_seed=0):
                out_nonzero = define_model("theta", data_nonzero, priors)

        assert not np.allclose(
            np.asarray(out_nonzero.ln_K_h_l[0]),  # WT
            np.asarray(out_nonzero.ln_K_h_l[1]),  # M1
            atol=1e-5,
        )

    def test_permuted_features_give_same_result_as_canonical(self):
        """_get_struct_perm undoes permutation of struct_features columns."""
        canonical_prior = _make_ddG_prior()
        shuffle_order  = [0, 1, 4, 2, 3, 5]   # H HO HE2 L LO LE2
        shuffled_names = tuple(STRUCTURE_NAMES[i] for i in shuffle_order)
        shuffled_prior = canonical_prior[:, shuffle_order]

        canonical_data = _make_mock(struct_names=STRUCTURE_NAMES, ddG_prior=canonical_prior)
        shuffled_data  = _make_mock(struct_names=shuffled_names,   ddG_prior=shuffled_prior)

        priors = get_priors()
        with seed(rng_seed=7):
            out_c = define_model("theta", canonical_data, priors)
        with seed(rng_seed=7):
            out_s = define_model("theta", shuffled_data,  priors)

        np.testing.assert_allclose(
            np.asarray(out_c.ln_K_h_l),
            np.asarray(out_s.ln_K_h_l), atol=1e-5,
        )

    def test_tf_op_totals_preserved(self):
        out, _ = _run_model(_make_mock())
        assert out.tf_total == pytest.approx(6.5e-7)
        assert out.op_total == pytest.approx(2.5e-8)


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_returns_theta_param(self):
        out, _ = _run_guide(_make_mock())
        assert isinstance(out, ThetaParam)

    def test_K_shapes(self):
        out, _ = _run_guide(_make_mock())
        T = 2
        assert out.ln_K_h_l.shape == (_G,)
        assert out.ln_K_h_e.shape == (T, _G)

    def test_ln_K_u_shape(self):
        out, _ = _run_guide(_make_mock())
        assert out.ln_K_u.shape == (_G,)

    def test_ln_K_u_is_homogeneous(self):
        out, _ = _run_guide(_make_mock())
        vals = np.asarray(out.ln_K_u)
        np.testing.assert_allclose(vals, vals[0], atol=1e-6)

    def test_sample_sites_match_model(self):
        data = _make_mock()
        _, model_tr = _run_model(data)
        _, guide_tr = _run_guide(data)
        model_s = {k for k, v in model_tr.items() if v["type"] == "sample"}
        guide_s = {k for k, v in guide_tr.items() if v["type"] == "sample"}
        assert model_s == guide_s

    def test_variational_params_registered(self):
        _, tr = _run_guide(_make_mock())
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        for p in ("theta_ln_K_h_l_wt_loc", "theta_ln_K_h_l_wt_scale",
                  "theta_ln_K_h_o_wt_loc", "theta_ln_K_l_o_wt_loc",
                  "theta_ddG_offset_locs", "theta_ddG_offset_scales",
                  "theta_ddG_sigma_s",
                  # unfolding constant variational params — new in mwc_dimer_unfolded
                  "theta_ln_K_u_wt_loc",  "theta_ln_K_u_wt_scale"):
            assert p in param_names, f"Missing variational param: {p}"

    def test_no_nn_params_registered(self):
        """lnK_ddG_prior uses prior means directly; no per-structure NN weights."""
        _, tr = _run_guide(_make_mock())
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert not any("_nn_" in k for k in param_names), \
            f"Unexpected NN params: {[k for k in param_names if '_nn_' in k]}"

    def test_no_epi_params_registered(self):
        _, tr = _run_guide(_make_mock())
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert not any("epi_lambda" in k for k in param_names)
        assert not any("epi_offset" in k for k in param_names)

    def test_ddG_sigma_s_shape(self):
        _, tr = _run_guide(_make_mock())
        sigma_val = tr["theta_ddG_sigma_s"]["value"]
        assert sigma_val.shape == (_S,)

    def test_ddG_offset_locs_shape(self):
        _, tr = _run_guide(_make_mock())
        locs_val = tr["theta_ddG_offset_locs"]["value"]
        assert locs_val.shape == (_S, _M)

    def test_tf_op_totals_preserved(self):
        out, _ = _run_guide(_make_mock())
        assert out.tf_total == pytest.approx(6.5e-7)
        assert out.op_total == pytest.approx(2.5e-8)

    def test_permuted_features_give_same_output_as_canonical(self):
        canonical_prior = _make_ddG_prior()
        shuffle_order  = [0, 1, 4, 2, 3, 5]
        shuffled_names = tuple(STRUCTURE_NAMES[i] for i in shuffle_order)
        shuffled_prior = canonical_prior[:, shuffle_order]

        canonical_data = _make_mock(struct_names=STRUCTURE_NAMES, ddG_prior=canonical_prior)
        shuffled_data  = _make_mock(struct_names=shuffled_names,   ddG_prior=shuffled_prior)

        priors = get_priors()
        with seed(rng_seed=5):
            out_c = guide("theta", canonical_data, priors)
        with seed(rng_seed=5):
            out_s = guide("theta", shuffled_data,  priors)

        np.testing.assert_allclose(
            np.asarray(out_c.ln_K_h_l),
            np.asarray(out_s.ln_K_h_l), atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------

def test_registry_entry():
    from tfscreen.growth_model.registry import model_registry
    assert "mwc_dimer_unfolded_lnK_ddG_prior" in model_registry["theta"]
    import tfscreen.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_ddG_prior as mod
    assert model_registry["theta"]["mwc_dimer_unfolded_lnK_ddG_prior"] is mod


# ---------------------------------------------------------------------------
# get_extract_specs
# ---------------------------------------------------------------------------

class _FakeTM:
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


def _make_ctx(genotypes=None, titrant_names=None, mut_labels=None, pair_labels=None):
    if genotypes     is None: genotypes     = ["wt", "M1", "M2", "M1M2"]
    if titrant_names is None: titrant_names = ["iptg"]
    if mut_labels    is None: mut_labels    = ["M1", "M2"]
    if pair_labels   is None: pair_labels   = []
    return _FakeCtx(
        growth_tm=_FakeTM(genotypes, titrant_names),
        mut_labels=mut_labels,
        pair_labels=pair_labels,
    )


class TestGetExtractSpecs:

    def test_returns_list(self):
        assert isinstance(get_extract_specs(_make_ctx()), list)

    def test_three_specs(self):
        # scalar K (incl. ln_K_u), T-dim K, per-mutation d_ln_K
        assert len(get_extract_specs(_make_ctx())) == 3

    def test_four_specs_when_pair_labels_present(self):
        ctx = _make_ctx(pair_labels=["M1+M2"])
        assert len(get_extract_specs(ctx)) == 4

    def test_first_spec_has_scalar_K_values_including_ln_K_u(self):
        """ln_K_u is broadcast from WT and should appear in the scalar K spec."""
        params = get_extract_specs(_make_ctx())[0]["params_to_get"]
        assert "ln_K_h_l" in params
        assert "ln_K_h_o" in params
        assert "ln_K_l_o" in params
        assert "ln_K_u"   in params   # new in mwc_dimer_unfolded

    def test_second_spec_has_titrant_K_values(self):
        params = get_extract_specs(_make_ctx())[1]["params_to_get"]
        assert "ln_K_h_e" in params
        assert "ln_K_l_e" in params

    def test_third_spec_has_all_projected_delta_lnK(self):
        params = get_extract_specs(_make_ctx())[2]["params_to_get"]
        for k in ("d_ln_K_h_l", "d_ln_K_h_o", "d_ln_K_h_e",
                  "d_ln_K_l_o", "d_ln_K_l_e"):
            assert k in params

    def test_all_specs_use_theta_prefix(self):
        for spec in get_extract_specs(_make_ctx()):
            assert spec["in_run_prefix"] == "theta_"

    def test_second_spec_covers_all_titrant_genotype_combos(self):
        ctx = _make_ctx(genotypes=["wt", "M1"], titrant_names=["iptg", "tmg"])
        df = get_extract_specs(ctx)[1]["input_df"]
        assert len(df) == 2 * 2

    def test_third_spec_has_one_row_per_mutation(self):
        ctx = _make_ctx(mut_labels=["M1", "M2", "M3"])
        df = get_extract_specs(ctx)[2]["input_df"]
        assert len(df) == 3


# ---------------------------------------------------------------------------
# predict_unmeasured
# ---------------------------------------------------------------------------

from tfscreen.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_ddG_prior import (
    predict_unmeasured as ddG_predict,
)
from tfscreen.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_mut import (
    predict_unmeasured as lnK_mut_predict,
)

_MUT_LABELS_P  = ["M42I", "K84L"]
_PAIR_LABELS_P = ["K84L/M42I"]
_TITRANT_NAMES_P = ["IPTG"]
_TF_TOTAL  = 6.5e-7
_OP_TOTAL  = 2.5e-8
_CONC_SCALE = 1e-3

_WT_LN_K_H_L =  1.84
_WT_LN_K_H_O = 19.86
_WT_LN_K_H_E = 10.93
_WT_LN_K_L_O = -2.30
_WT_LN_K_L_E = 13.54
_WT_LN_K_U   = -5.0    # plausible WT unfolding constant


def _make_titrant_df_p(concs=None):
    if concs is None:
        concs = [0.0, 0.1, 1.0, 10.0]
    return pd.DataFrame({
        "titrant_name": [_TITRANT_NAMES_P[0]] * len(concs),
        "titrant_conc": concs,
    })


def _make_ddG_posteriors(T, M, P=0, S=5, seed=0):
    """Fake posteriors for mwc_dimer_unfolded lnK_ddG_prior.

    KEY DIFFERENCES from lnK_mut:
    - d_ln_K_h_e and d_ln_K_l_e are (S, M) with no T dimension.
    - epi_h_e and epi_l_e are (S, P) with no T dimension.
    - ln_K_u_wt is a scalar (S,) — no per-mutation delta.
    """
    rng = np.random.default_rng(seed)
    post = {}
    post["theta_ln_K_h_l_wt"] = np.full(S, _WT_LN_K_H_L, dtype=np.float32)
    post["theta_ln_K_h_o_wt"] = np.full(S, _WT_LN_K_H_O, dtype=np.float32)
    post["theta_ln_K_l_o_wt"] = np.full(S, _WT_LN_K_L_O, dtype=np.float32)
    post["theta_ln_K_h_e_wt"] = np.full((S, T), _WT_LN_K_H_E, dtype=np.float32)
    post["theta_ln_K_l_e_wt"] = np.full((S, T), _WT_LN_K_L_E, dtype=np.float32)
    post["theta_ln_K_u_wt"]   = np.full(S, _WT_LN_K_U, dtype=np.float32)
    post["theta_d_ln_K_h_l"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_h_o"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_l_o"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.3
    post["theta_d_ln_K_h_e"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.3  # no T
    post["theta_d_ln_K_l_e"]  = rng.standard_normal((S, M)).astype(np.float32) * 0.3  # no T
    if P > 0:
        for key in ["theta_epi_ln_K_h_l", "theta_epi_ln_K_h_o", "theta_epi_ln_K_l_o"]:
            post[key] = np.zeros((S, P), dtype=np.float32)
        for key in ["theta_epi_ln_K_h_e", "theta_epi_ln_K_l_e"]:
            post[key] = np.zeros((S, P), dtype=np.float32)   # no T!
    return post


def _common_kwargs_p(post, pair_labels=None):
    return dict(
        target_genotypes=["wt", "M42I", "K84L", "M42I/K84L"],
        titrant_names=_TITRANT_NAMES_P,
        manual_titrant_df=_make_titrant_df_p(),
        mut_labels=_MUT_LABELS_P,
        pair_labels=pair_labels or [],
        param_posteriors=post,
        q_to_get={"median": 0.5, "lower": 0.025, "upper": 0.975},
        tf_total=_TF_TOTAL,
        op_total=_OP_TOTAL,
        conc_unit_scale=_CONC_SCALE,
    )


class TestDdGPriorPredictUnmeasured:

    def test_output_shape_and_columns(self):
        post = _make_ddG_posteriors(T=1, M=2)
        result = ddG_predict(**_common_kwargs_p(post))
        assert len(result) == 4 * 4   # 4 genotypes × 4 concs
        for col in ["genotype", "titrant_name", "titrant_conc",
                    "median", "lower", "upper"]:
            assert col in result.columns

    def test_wt_theta_in_unit_interval(self):
        post = _make_ddG_posteriors(T=1, M=2)
        result = ddG_predict(**_common_kwargs_p(post))
        wt_rows = result[result["genotype"] == "wt"]
        assert (wt_rows["median"] >= 0).all()
        assert (wt_rows["median"] <= 1).all()
        assert not wt_rows["median"].isna().any()

    def test_unknown_mutation_is_nan(self):
        post = _make_ddG_posteriors(T=1, M=2)
        kw = _common_kwargs_p(post)
        kw["target_genotypes"] = ["wt", "NOVEL"]
        result = ddG_predict(**kw)
        assert result.loc[result["genotype"] == "NOVEL", "median"].isna().all()
        assert not result.loc[result["genotype"] == "wt", "median"].isna().any()

    def test_zero_conc_handled(self):
        post = _make_ddG_posteriors(T=1, M=2)
        kw = _common_kwargs_p(post)
        kw["manual_titrant_df"] = _make_titrant_df_p(concs=[0.0])
        result = ddG_predict(**kw)
        assert np.isfinite(result["median"].values).all()

    def test_upper_ge_median_ge_lower(self):
        post = _make_ddG_posteriors(T=1, M=2, S=40, seed=5)
        result = ddG_predict(**_common_kwargs_p(post))
        assert (result["upper"] >= result["median"]).all()
        assert (result["median"] >= result["lower"]).all()

    def test_ln_K_u_is_homogeneous(self):
        """All target genotypes receive the same ln_K_u (WT only, no per-mutation delta)."""
        S = 10
        post = _make_ddG_posteriors(T=1, M=2, S=S)
        # Zero out all mutation deltas so only WT K values differ
        post["theta_d_ln_K_h_l"][:] = 0
        post["theta_d_ln_K_h_o"][:] = 0
        post["theta_d_ln_K_l_o"][:] = 0
        post["theta_d_ln_K_h_e"][:] = 0
        post["theta_d_ln_K_l_e"][:] = 0

        kw = _common_kwargs_p(post)
        kw["target_genotypes"] = ["wt", "M42I", "K84L", "M42I/K84L"]
        kw["q_to_get"] = {"median": 0.5}
        result = ddG_predict(**kw)

        # With zero mutation deltas, all genotypes get identical predictions
        for conc in [0.0, 0.1, 1.0, 10.0]:
            rows = result[result["titrant_conc"] == conc]["median"].values
            np.testing.assert_allclose(rows, rows[0], atol=1e-5,
                                       err_msg=f"Predictions differ at conc={conc}")

    def test_epistasis_shifts_double_mutant(self):
        T, M, P, S = 1, 2, 1, 30
        post = _make_ddG_posteriors(T, M, P=P, S=S, seed=4)
        post["theta_epi_ln_K_h_l"] = np.full((S, P), 4.0, dtype=np.float32)

        kw_no_epi = _common_kwargs_p(post, pair_labels=[])
        kw_no_epi["target_genotypes"] = ["M42I/K84L"]
        res_no_epi = ddG_predict(**kw_no_epi)

        kw_epi = _common_kwargs_p(post, pair_labels=_PAIR_LABELS_P)
        kw_epi["target_genotypes"] = ["M42I/K84L"]
        res_epi = ddG_predict(**kw_epi)

        assert not np.allclose(res_no_epi["median"].values,
                               res_epi["median"].values, atol=1e-3)

    def test_scalar_d_he_le_matches_lnK_mut_when_replicated(self):
        """ddG_prior with scalar d_h_e should equal lnK_mut when replicated T times."""
        T, M, S = 2, 2, 5
        rng = np.random.default_rng(77)
        d_he_scalar = rng.standard_normal((S, M)).astype(np.float32) * 0.3
        d_le_scalar = rng.standard_normal((S, M)).astype(np.float32) * 0.3

        post_ddG = _make_ddG_posteriors(T, M, S=S)
        post_ddG["theta_d_ln_K_h_e"] = d_he_scalar
        post_ddG["theta_d_ln_K_l_e"] = d_le_scalar

        post_mut = {}
        for k in ["theta_ln_K_h_l_wt", "theta_ln_K_h_o_wt", "theta_ln_K_l_o_wt",
                  "theta_ln_K_h_e_wt", "theta_ln_K_l_e_wt", "theta_ln_K_u_wt",
                  "theta_d_ln_K_h_l", "theta_d_ln_K_h_o", "theta_d_ln_K_l_o"]:
            post_mut[k] = post_ddG[k].copy()
        post_mut["theta_d_ln_K_h_e"] = np.broadcast_to(
            d_he_scalar[:, None, :], (S, T, M)).astype(np.float32)
        post_mut["theta_d_ln_K_l_e"] = np.broadcast_to(
            d_le_scalar[:, None, :], (S, T, M)).astype(np.float32)
        post_mut["theta_d_ln_K_u"]   = np.zeros((S, M), dtype=np.float32)

        kw = dict(
            target_genotypes=["wt", "M42I", "K84L"],
            titrant_names=["IPTG", "TMAIPP"],
            manual_titrant_df=pd.DataFrame({
                "titrant_name": ["IPTG", "TMAIPP"],
                "titrant_conc": [0.1, 1.0],
            }),
            mut_labels=_MUT_LABELS_P,
            pair_labels=[],
            q_to_get={"median": 0.5},
            tf_total=_TF_TOTAL,
            op_total=_OP_TOTAL,
            conc_unit_scale=_CONC_SCALE,
        )
        res_ddG = ddG_predict(param_posteriors=post_ddG, **kw)
        res_mut = lnK_mut_predict(param_posteriors=post_mut, **kw)

        np.testing.assert_allclose(
            res_ddG["median"].values, res_mut["median"].values, atol=1e-5
        )
