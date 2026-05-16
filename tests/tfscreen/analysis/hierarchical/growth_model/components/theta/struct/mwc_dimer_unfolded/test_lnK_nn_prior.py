"""
Tests for struct/mwc_dimer_unfolded/lnK_nn_prior.py.

Covers: _get_struct_perm, _project_ddG, get_hyperparameters, get_priors,
get_guesses, define_model (no epi / with epi), guide, get_extract_specs.

Key differences from mwc_dimer variant:
- ThetaParam gains ln_K_u (G,) and conc_unit_scale fields
- ModelPriors gains theta_ln_K_u_wt_loc and theta_ln_K_u_wt_scale
- define_model samples theta_ln_K_u_wt and broadcasts it homogeneously
- ln_K_u[g] == ln_K_u_wt for all g (no per-mutation unfolding effect)
- guide samples theta_ln_K_u_wt with variational params theta_ln_K_u_wt_loc/scale
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataclasses import dataclass
from collections import namedtuple
from numpyro.handlers import trace, seed

from tfscreen.genetics.build_mut_geno_matrix import build_mut_sparse_indices
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_nn_prior import (
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
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer_unfolded.thermo import (
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

_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

_CONC = np.array([0.0, 1e-5, 1e-4], dtype=np.float32)

# Sanity-check the expected structure set
assert set(STRUCTURE_NAMES) == {'H', 'HO', 'L', 'LO', 'HE2', 'LE2'}


def _make_features(rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    return rng.randn(_M, _S, 60).astype(np.float32)


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
    # struct fields
    "num_struct",
    "struct_names",
    "struct_features",
    "struct_n_chains",
    "struct_contact_pair_idx",
    "struct_contact_distances",
])

_NO_DISTANCES = object()  # sentinel: auto-fill when num_pair > 0


def _make_mock(num_pair=0, features=None, n_chains=None,
               struct_names=None, contact_distances=_NO_DISTANCES):
    if features is None:
        features = _make_features()
    if n_chains is None:
        n_chains = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
    if struct_names is None:
        struct_names = STRUCTURE_NAMES

    pair_nnz_pair = _PAIR_NNZ_PAIR if num_pair > 0 else np.zeros(0, dtype=np.int32)
    pair_nnz_geno = _PAIR_NNZ_GENO if num_pair > 0 else np.zeros(0, dtype=np.int32)

    if contact_distances is _NO_DISTANCES:
        contact_distances = (
            np.full((num_pair, _S), 5.0, dtype=np.float32) if num_pair > 0 else None
        )

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

    def test_canonical_perm_output_reindexes_correctly(self):
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        perm = _get_struct_perm(data)
        l_canonical_idx = list(STRUCTURE_NAMES).index('L')   # 2
        l_shuffled_idx  = list(shuffled).index('L')          # 3
        assert perm[l_canonical_idx] == l_shuffled_idx

    def test_wrong_structure_names_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)

    def test_missing_structure_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)

    def test_duplicate_structure_raises(self):
        data = _make_mock(struct_names=('H', 'H', 'L', 'LO', 'HE2', 'LE2'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)


# ---------------------------------------------------------------------------
# _project_ddG
# ---------------------------------------------------------------------------

class TestProjectDdG:
    """
    _PROJ rows (K order):  K_h_l, K_h_o, K_h_e, K_l_o, K_l_e
    _PROJ cols (S order):  H, HO, L, LO, HE2, LE2

    Δln_K_h_l = ΔΔG_H − ΔΔG_L
    Δln_K_h_o = ΔΔG_H − ΔΔG_HO
    Δln_K_h_e = (ΔΔG_H − ΔΔG_HE2) / 2
    Δln_K_l_o = ΔΔG_L − ΔΔG_LO
    Δln_K_l_e = (ΔΔG_L − ΔΔG_LE2) / 2
    """

    def test_output_shape(self):
        out = _project_ddG(jnp.zeros((_M, _S)))
        assert out.shape == (_M, 5)

    def test_zero_ddG_gives_zero_delta_lnK(self):
        np.testing.assert_allclose(_project_ddG(jnp.zeros((3, _S))), 0.0)

    def test_K_h_l_equals_H_minus_L(self):
        ddG = jnp.array([[3.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 0] == pytest.approx(3.0 - 1.0)

    def test_K_h_o_equals_H_minus_HO(self):
        ddG = jnp.array([[4.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 1] == pytest.approx(4.0 - 2.0)

    def test_K_h_e_equals_H_minus_HE2_over_two(self):
        ddG = jnp.array([[6.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        assert _project_ddG(ddG)[0, 2] == pytest.approx((6.0 - 2.0) / 2.0)

    def test_K_l_o_equals_L_minus_LO(self):
        ddG = jnp.array([[0.0, 0.0, 5.0, 1.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 3] == pytest.approx(5.0 - 1.0)

    def test_K_l_e_equals_L_minus_LE2_over_two(self):
        ddG = jnp.array([[0.0, 0.0, 8.0, 0.0, 0.0, 2.0]])
        assert _project_ddG(ddG)[0, 4] == pytest.approx((8.0 - 2.0) / 2.0)

    def test_leading_batch_dims(self):
        assert _project_ddG(jnp.ones((2, 7, _S))).shape == (2, 7, 5)

    def test_H_perturbation_does_not_affect_l_o_or_l_e(self):
        ddG_base = jnp.zeros((1, _S))
        ddG_H    = ddG_base.at[0, 0].set(1.0)
        out_base = _project_ddG(ddG_base)
        out_H    = _project_ddG(ddG_H)
        assert out_H[0, 3] == pytest.approx(float(out_base[0, 3]))  # K_l_o
        assert out_H[0, 4] == pytest.approx(float(out_base[0, 4]))  # K_l_e


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
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
            # unfolding constant keys — new in mwc_dimer_unfolded
            "theta_ln_K_u_wt_loc", "theta_ln_K_u_wt_scale",
            "theta_tf_total_M", "theta_op_total_M", "theta_conc_unit_scale",
            "theta_nn_hidden_size",
            "theta_epi_tau_scale", "theta_epi_slab_scale",
            "theta_epi_slab_df",   "theta_epi_d0",
        ]
        for k in required:
            assert k in hp, f"Missing hyperparameter key: {k}"

    def test_ln_K_u_prior_values(self):
        hp = get_hyperparameters()
        assert hp["theta_ln_K_u_wt_loc"]   == pytest.approx(-12.0)
        assert hp["theta_ln_K_u_wt_scale"] == pytest.approx(3.0)

    def test_get_priors_type(self):
        assert isinstance(get_priors(), ModelPriors)

    def test_model_priors_has_ln_K_u_fields(self):
        p = get_priors()
        assert hasattr(p, "theta_ln_K_u_wt_loc")
        assert hasattr(p, "theta_ln_K_u_wt_scale")

    def test_physical_constants(self):
        p = get_priors()
        assert p.theta_tf_total_M == pytest.approx(6.5e-7)
        assert p.theta_op_total_M == pytest.approx(2.5e-8)

    def test_get_guesses_no_epi_keys(self):
        data = _make_mock()
        g = get_guesses("theta", data)
        assert "theta_ln_K_h_l_wt" in g
        assert "theta_ln_K_h_o_wt" in g
        assert "theta_ln_K_l_o_wt" in g
        assert "theta_ln_K_h_e_wt" in g
        assert "theta_ln_K_l_e_wt" in g
        assert "theta_ln_K_u_wt"   in g
        assert "theta_ddG_offset"   in g
        assert "theta_epi_tau" not in g

    def test_get_guesses_ln_K_u_wt_value(self):
        g = get_guesses("theta", _make_mock())
        assert float(g["theta_ln_K_u_wt"]) == pytest.approx(-12.0)

    def test_get_guesses_with_epi_keys(self):
        data = _make_mock(num_pair=1)
        g = get_guesses("theta", data)
        assert "theta_epi_tau"    in g
        assert "theta_epi_c2"     in g
        assert "theta_epi_lambda" in g
        assert "theta_epi_offset" in g

    def test_get_guesses_ddG_offset_shape(self):
        assert get_guesses("theta", _make_mock())["theta_ddG_offset"].shape == (_S, _M)

    def test_get_guesses_epi_shapes_with_epi(self):
        data = _make_mock(num_pair=1)
        g = get_guesses("theta", data)
        assert g["theta_epi_lambda"].shape == (_S, 1)
        assert g["theta_epi_offset"].shape == (_S, 1)

    def test_get_guesses_wt_K_shapes(self):
        T = 2
        g = get_guesses("theta", _make_mock())
        assert g["theta_ln_K_h_l_wt"].shape == ()
        assert g["theta_ln_K_h_o_wt"].shape == ()
        assert g["theta_ln_K_l_o_wt"].shape == ()
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
        """The theta_ln_K_u deterministic site must be a constant-valued (G,) array."""
        _, tr = _run_model(_make_mock())
        ln_K_u_vals = np.asarray(tr["theta_ln_K_u"]["value"])
        np.testing.assert_allclose(ln_K_u_vals, ln_K_u_vals[0], atol=1e-6)

    def test_wrong_struct_names_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _run_model(data)

    def test_permuted_struct_names_accepted(self):
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        out, _ = _run_model(data)
        assert isinstance(out, ThetaParam)

    def test_permuted_features_give_same_K_as_canonical(self):
        """_get_struct_perm should undo any permutation of features."""
        canonical_data = _make_mock(struct_names=STRUCTURE_NAMES)

        shuffle_order  = [0, 1, 4, 2, 3, 5]   # H HO HE2 L LO LE2
        shuffled_names = tuple(STRUCTURE_NAMES[i] for i in shuffle_order)
        canonical_features = _make_features()
        shuffled_features  = canonical_features[:, shuffle_order, :]
        shuffled_data = _make_mock(struct_names=shuffled_names,
                                   features=shuffled_features)

        priors = get_priors()
        with seed(rng_seed=42):
            out_c = define_model("theta", canonical_data, priors)
        with seed(rng_seed=42):
            out_s = define_model("theta", shuffled_data, priors)

        np.testing.assert_allclose(
            np.asarray(out_c.ln_K_h_l),
            np.asarray(out_s.ln_K_h_l), atol=1e-5,
        )

    def test_no_epi_sites_when_num_pair_zero(self):
        _, tr = _run_model(_make_mock(num_pair=0))
        assert "theta_epi_tau"    not in tr
        assert "theta_epi_lambda" not in tr
        assert "theta_epi_offset" not in tr

    def test_epi_sites_present_when_num_pair_nonzero(self):
        _, tr = _run_model(_make_mock(num_pair=1))
        assert "theta_epi_tau"    in tr
        assert "theta_epi_c2"     in tr
        assert "theta_epi_lambda" in tr
        assert "theta_epi_offset" in tr

    def test_epi_absent_when_contact_distances_none(self):
        data = _make_mock(num_pair=1, contact_distances=None)
        _, tr = _run_model(data)
        assert "theta_epi_tau" not in tr

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

    def test_sample_sites_match_model_no_epi(self):
        data = _make_mock()
        _, model_tr = _run_model(data)
        _, guide_tr = _run_guide(data)
        model_s = {k for k, v in model_tr.items() if v["type"] == "sample"}
        guide_s = {k for k, v in guide_tr.items() if v["type"] == "sample"}
        assert model_s == guide_s

    def test_sample_sites_match_model_with_epi(self):
        data = _make_mock(num_pair=1)
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
                  "theta_ddG_offset_locs",  "theta_ddG_offset_scales",
                  "theta_ddG_sigma_s",
                  # unfolding constant variational params — new in mwc_dimer_unfolded
                  "theta_ln_K_u_wt_loc",   "theta_ln_K_u_wt_scale"):
            assert p in param_names, f"Missing variational param: {p}"

    def test_nn_params_registered_per_structure(self):
        _, tr = _run_guide(_make_mock())
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        for sname in STRUCTURE_NAMES:
            assert f"theta_nn_{sname}_W1" in param_names, \
                f"Missing NN param theta_nn_{sname}_W1"

    def test_nn_params_use_canonical_names_with_shuffled_input(self):
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        for sname in STRUCTURE_NAMES:
            assert f"theta_nn_{sname}_W1" in param_names

    def test_no_epi_params_when_num_pair_zero(self):
        _, tr = _run_guide(_make_mock(num_pair=0))
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert not any("epi_lambda" in k for k in param_names)

    def test_epi_params_registered_when_num_pair_nonzero(self):
        _, tr = _run_guide(_make_mock(num_pair=1))
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert "theta_epi_lambda_locs"   in param_names
        assert "theta_epi_lambda_scales" in param_names
        assert "theta_epi_offset_locs"   in param_names
        assert "theta_epi_offset_scales" in param_names

    def test_tf_op_totals_preserved(self):
        out, _ = _run_guide(_make_mock())
        assert out.tf_total == pytest.approx(6.5e-7)
        assert out.op_total == pytest.approx(2.5e-8)


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------

def test_registry_entry():
    from tfscreen.analysis.hierarchical.growth_model.registry import model_registry
    assert "mwc_dimer_unfolded_lnK_nn_prior" in model_registry["theta"]
    import tfscreen.analysis.hierarchical.growth_model.components.theta.struct.mwc_dimer_unfolded.lnK_nn_prior as mod
    assert model_registry["theta"]["mwc_dimer_unfolded_lnK_nn_prior"] is mod


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

    def test_three_specs_without_epi(self):
        # scalar K (incl. ln_K_u), T-dim K, per-mutation d_ln_K
        assert len(get_extract_specs(_make_ctx())) == 3

    def test_four_specs_with_epi(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
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

    def test_third_spec_has_projected_delta_lnK(self):
        params = get_extract_specs(_make_ctx())[2]["params_to_get"]
        for k in ("d_ln_K_h_l", "d_ln_K_h_o", "d_ln_K_h_e",
                  "d_ln_K_l_o", "d_ln_K_l_e"):
            assert k in params

    def test_fourth_spec_has_epi_terms(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        params = get_extract_specs(ctx)[3]["params_to_get"]
        for k in ("epi_ln_K_h_l", "epi_ln_K_h_o", "epi_ln_K_h_e",
                  "epi_ln_K_l_o", "epi_ln_K_l_e"):
            assert k in params

    def test_all_specs_use_theta_prefix(self):
        for spec in get_extract_specs(_make_ctx(pair_labels=["M1M2"])):
            assert spec["in_run_prefix"] == "theta_"

    def test_second_spec_covers_all_titrant_genotype_combos(self):
        genos    = ["wt", "M1"]
        titrants = ["iptg", "tmg"]
        ctx = _make_ctx(genotypes=genos, titrant_names=titrants)
        df = get_extract_specs(ctx)[1]["input_df"]
        assert len(df) == len(genos) * len(titrants)

    def test_third_spec_has_one_row_per_mutation(self):
        ctx = _make_ctx(mut_labels=["M1", "M2", "M3"])
        df = get_extract_specs(ctx)[2]["input_df"]
        assert len(df) == 3
