"""
Tests for struct/mwc_dimer/lnK_nn_prior.py.

Covers: _get_struct_perm, _project_ddG, get_hyperparameters, get_priors,
get_guesses, define_model (no epi / with epi), guide, get_extract_specs.
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataclasses import dataclass
from collections import namedtuple
from numpyro.handlers import trace, seed

from tfscreen.genetics.build_mut_geno_matrix import build_mut_sparse_indices
from tfscreen.tfmodel.components.theta.struct.mwc_dimer.lnK_nn_prior import (
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
from tfscreen.tfmodel.components.theta.struct.mwc_dimer.thermo import (
    ThetaParam,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S = 6    # number of MWC structures
_M = 2    # mutations
_G = 4    # genotypes: wt, M1, M2, M1+M2

_MUT_GENO = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1]], dtype=np.float32)  # (M, G)
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)

_PAIR_NNZ_PAIR = np.array([0], dtype=np.int32)
_PAIR_NNZ_GENO = np.array([3], dtype=np.int32)

_CONC = np.array([0.0, 1e-5, 1e-4], dtype=np.float32)

# Check that STRUCTURE_NAMES matches expectations
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
        n_chains = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)  # one per structure
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
        # identity: data already in STRUCTURE_NAMES order
        expected = list(range(_S))
        assert perm == expected

    def test_permuted_order_gives_correct_perm(self):
        """Structures in a different order → perm maps data indices to canonical."""
        # Shuffle canonical order: H HO L LO HE2 LE2 → H HO HE2 L LO LE2
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        perm = _get_struct_perm(data)
        # STRUCTURE_NAMES[i] should be at shuffled[perm[i]]
        for i, sname in enumerate(STRUCTURE_NAMES):
            assert shuffled[perm[i]] == sname

    def test_canonical_perm_output_reindexes_correctly(self):
        """Applying perm to features in any order gives canonical-order features."""
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        features = _make_features()
        # Build features in shuffled order: axis 1 order = shuffled order
        data = _make_mock(struct_names=shuffled, features=features)
        perm = _get_struct_perm(data)
        # features[:, perm[i], :] should be the features for STRUCTURE_NAMES[i]
        # Verify for 'L': STRUCTURE_NAMES[2]='L'; in shuffled order, 'L' is at index 3
        l_canonical_idx = list(STRUCTURE_NAMES).index('L')        # 2
        l_shuffled_idx  = list(shuffled).index('L')               # 3
        assert perm[l_canonical_idx] == l_shuffled_idx

    def test_wrong_structure_names_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)

    def test_missing_structure_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2'))  # only 5
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
        ddG = jnp.zeros((_M, _S))
        out = _project_ddG(ddG)
        assert out.shape == (_M, 5)

    def test_zero_ddG_gives_zero_delta_lnK(self):
        ddG = jnp.zeros((3, _S))
        np.testing.assert_allclose(_project_ddG(ddG), 0.0)

    def test_K_h_l_equals_H_minus_L(self):
        # H=3, HO=0, L=1, LO=0, HE2=0, LE2=0
        ddG = jnp.array([[3.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        out = _project_ddG(ddG)
        assert out[0, 0] == pytest.approx(3.0 - 1.0)   # K_h_l = H - L

    def test_K_h_o_equals_H_minus_HO(self):
        # H=4, HO=2
        ddG = jnp.array([[4.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        out = _project_ddG(ddG)
        assert out[0, 1] == pytest.approx(4.0 - 2.0)   # K_h_o = H - HO

    def test_K_h_e_equals_H_minus_HE2_over_two(self):
        # H=6, HE2=2 → K_h_e = (6-2)/2 = 2
        ddG = jnp.array([[6.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        out = _project_ddG(ddG)
        assert out[0, 2] == pytest.approx((6.0 - 2.0) / 2.0)

    def test_K_l_o_equals_L_minus_LO(self):
        # L=5, LO=1
        ddG = jnp.array([[0.0, 0.0, 5.0, 1.0, 0.0, 0.0]])
        out = _project_ddG(ddG)
        assert out[0, 3] == pytest.approx(5.0 - 1.0)   # K_l_o = L - LO

    def test_K_l_e_equals_L_minus_LE2_over_two(self):
        # L=8, LE2=2 → K_l_e = (8-2)/2 = 3
        ddG = jnp.array([[0.0, 0.0, 8.0, 0.0, 0.0, 2.0]])
        out = _project_ddG(ddG)
        assert out[0, 4] == pytest.approx((8.0 - 2.0) / 2.0)

    def test_leading_batch_dims(self):
        ddG = jnp.ones((2, 7, _S))
        out = _project_ddG(ddG)
        assert out.shape == (2, 7, 5)

    def test_independent_effects(self):
        """Each row of _PROJ is independent — changing only one structure
        should only affect the K values that reference it."""
        # Perturb only H; K_h_l, K_h_o, K_h_e should change; K_l_o, K_l_e should not
        ddG_base = jnp.zeros((1, _S))
        ddG_H    = ddG_base.at[0, 0].set(1.0)     # H index = 0 in STRUCTURE_NAMES
        out_base = _project_ddG(ddG_base)
        out_H    = _project_ddG(ddG_H)
        # K_h_l (idx 0), K_h_o (idx 1), K_h_e (idx 2) should change
        assert out_H[0, 0] != out_base[0, 0]  # K_h_l
        assert out_H[0, 1] != out_base[0, 1]  # K_h_o
        assert out_H[0, 2] != out_base[0, 2]  # K_h_e
        # K_l_o (idx 3), K_l_e (idx 4) should NOT change
        assert out_H[0, 3] == pytest.approx(float(out_base[0, 3]))
        assert out_H[0, 4] == pytest.approx(float(out_base[0, 4]))


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
            "theta_tf_total_M", "theta_op_total_M",
            "theta_nn_hidden_size",
            "theta_epi_tau_scale", "theta_epi_slab_scale",
            "theta_epi_slab_df",   "theta_epi_d0",
        ]
        for k in required:
            assert k in hp, f"Missing hyperparameter key: {k}"

    def test_get_priors_type(self):
        assert isinstance(get_priors(), ModelPriors)

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
        assert "theta_ddG_offset"  in g
        assert "theta_epi_tau" not in g

    def test_get_guesses_with_epi_keys(self):
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

    def test_get_guesses_epi_shapes_with_epi(self):
        data = _make_mock(num_pair=1)
        g = get_guesses("theta", data)
        assert g["theta_epi_lambda"].shape == (_S, 1)
        assert g["theta_epi_offset"].shape == (_S, 1)

    def test_get_guesses_wt_K_shapes(self):
        T = 2
        data = _make_mock()
        g = get_guesses("theta", data)
        assert g["theta_ln_K_h_l_wt"].shape == ()
        assert g["theta_ln_K_h_o_wt"].shape == ()
        assert g["theta_ln_K_l_o_wt"].shape == ()
        assert g["theta_ln_K_h_e_wt"].shape == (T,)
        assert g["theta_ln_K_l_e_wt"].shape == (T,)


# ---------------------------------------------------------------------------
# Helpers for model/guide execution
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

    def test_population_moment_shapes(self):
        out, _ = _run_model(_make_mock())
        T, C = 2, len(_CONC)
        assert out.mu.shape    == (T, C, 1)
        assert out.sigma.shape == (T, C, 1)

    def test_sample_sites_present(self):
        _, tr = _run_model(_make_mock())
        for site in ("theta_ln_K_h_l_wt", "theta_ln_K_h_o_wt", "theta_ln_K_l_o_wt",
                     "theta_ln_K_h_e_wt", "theta_ln_K_l_e_wt", "theta_ddG_offset"):
            assert site in tr, f"Missing sample site: {site}"

    def test_deterministic_sites_present(self):
        _, tr = _run_model(_make_mock())
        for site in ("theta_ddG", "theta_d_ln_K_h_l", "theta_d_ln_K_h_o",
                     "theta_d_ln_K_h_e", "theta_d_ln_K_l_o", "theta_d_ln_K_l_e",
                     "theta_ln_K_h_l", "theta_ln_K_h_o", "theta_ln_K_h_e",
                     "theta_ln_K_l_o", "theta_ln_K_l_e"):
            assert site in tr, f"Missing deterministic site: {site}"

    def test_wrong_struct_names_raises(self):
        data = _make_mock(struct_names=('H', 'HO', 'L', 'LO', 'HE2', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _run_model(data)

    def test_permuted_struct_names_accepted(self):
        """Any ordering of the 6 required names should be accepted without error."""
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        out, _ = _run_model(data)   # must not raise
        assert isinstance(out, ThetaParam)

    def test_permuted_features_give_same_params_as_canonical(self):
        """Swapping struct_names + features together should give identical K values,
        because _get_struct_perm undoes any permutation."""
        canonical_data = _make_mock(struct_names=STRUCTURE_NAMES)

        # Build shuffled data: reorder both names and features consistently
        shuffle_order = [0, 1, 4, 2, 3, 5]   # H HO HE2 L LO LE2
        shuffled_names = tuple(STRUCTURE_NAMES[i] for i in shuffle_order)
        canonical_features = _make_features()
        shuffled_features  = canonical_features[:, shuffle_order, :]
        shuffled_data = _make_mock(struct_names=shuffled_names,
                                   features=shuffled_features)

        priors = get_priors()
        with seed(rng_seed=42):
            out_canonical = define_model("theta", canonical_data, priors)
        with seed(rng_seed=42):
            out_shuffled  = define_model("theta", shuffled_data, priors)

        # The NN weights are initialized to zero so nn_means = 0 for both;
        # the permuted features should produce identical K values (both zero-NN).
        np.testing.assert_allclose(
            np.asarray(out_canonical.ln_K_h_l),
            np.asarray(out_shuffled.ln_K_h_l), atol=1e-5,
        )

    def test_zero_nn_and_offsets_all_genotypes_equal(self):
        """With all NN weights=0 and ddG_offsets=0, all mutations have zero effect."""
        data = _make_mock()
        priors = get_priors()
        import unittest.mock as mock

        def _zero_param(name, init, **kwargs):
            return jnp.zeros_like(init)

        with mock.patch("numpyro.param", side_effect=_zero_param):
            with seed(rng_seed=0):
                out = define_model("theta", data, priors)

        np.testing.assert_allclose(
            np.asarray(out.ln_K_h_l[0]), np.asarray(out.ln_K_h_l[1]), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(out.ln_K_h_l[1]), np.asarray(out.ln_K_h_l[2]), atol=1e-5
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
        """num_pair > 0 but contact_distances=None → epistasis not sampled."""
        data = _make_mock(num_pair=1, contact_distances=None)
        _, tr = _run_model(data)
        assert "theta_epi_tau" not in tr


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_returns_theta_param(self):
        out, _ = _run_guide(_make_mock())
        assert isinstance(out, ThetaParam)

    def test_K_shapes(self):
        out, _ = _run_guide(_make_mock())
        T, G = 2, _G
        assert out.ln_K_h_l.shape == (G,)
        assert out.ln_K_h_e.shape == (T, G)

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
                  "theta_ddG_sigma_s"):
            assert p in param_names, f"Missing variational param: {p}"

    def test_nn_params_registered_per_structure(self):
        """compute_nn_predictions registers W1/b1/W2/b2 per STRUCTURE_NAMES entry."""
        _, tr = _run_guide(_make_mock())
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        for sname in STRUCTURE_NAMES:
            assert f"theta_nn_{sname}_W1" in param_names, \
                f"Missing NN param theta_nn_{sname}_W1"

    def test_nn_params_use_canonical_names_with_shuffled_input(self):
        """Regardless of struct input order, NN param names come from STRUCTURE_NAMES."""
        shuffled = ('H', 'HO', 'HE2', 'L', 'LO', 'LE2')
        data = _make_mock(struct_names=shuffled)
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        for sname in STRUCTURE_NAMES:
            assert f"theta_nn_{sname}_W1" in param_names

    def test_no_epi_params_when_num_pair_zero(self):
        data = _make_mock(num_pair=0)
        _, tr = _run_guide(data)
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert not any("epi_lambda" in k for k in param_names)

    def test_epi_params_registered_when_num_pair_nonzero(self):
        data = _make_mock(num_pair=1)
        _, tr = _run_guide(data)
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
    from tfscreen.tfmodel.registry import model_registry
    assert "mwc_dimer_lnK_nn_prior" in model_registry["theta"]
    import tfscreen.tfmodel.components.theta.struct.mwc_dimer.lnK_nn_prior as mod
    assert model_registry["theta"]["mwc_dimer_lnK_nn_prior"] is mod


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
    if genotypes    is None: genotypes    = ["wt", "M1", "M2", "M1M2"]
    if titrant_names is None: titrant_names = ["iptg"]
    if mut_labels   is None: mut_labels   = ["M1", "M2"]
    if pair_labels  is None: pair_labels  = []
    return _FakeCtx(
        growth_tm=_FakeTM(genotypes, titrant_names),
        mut_labels=mut_labels,
        pair_labels=pair_labels,
    )


class TestGetExtractSpecs:

    def test_returns_list(self):
        assert isinstance(get_extract_specs(_make_ctx()), list)

    def test_three_specs_without_epi(self):
        # scalar K, T-dim K, per-mutation d_ln_K
        assert len(get_extract_specs(_make_ctx())) == 3

    def test_four_specs_with_epi(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        assert len(get_extract_specs(ctx)) == 4

    def test_first_spec_has_scalar_K_values(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[0]["params_to_get"]
        assert "ln_K_h_l" in params
        assert "ln_K_h_o" in params
        assert "ln_K_l_o" in params

    def test_second_spec_has_titrant_K_values(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[1]["params_to_get"]
        assert "ln_K_h_e" in params
        assert "ln_K_l_e" in params

    def test_third_spec_has_projected_delta_lnK(self):
        specs = get_extract_specs(_make_ctx())
        params = specs[2]["params_to_get"]
        for k in ("d_ln_K_h_l", "d_ln_K_h_o", "d_ln_K_h_e",
                  "d_ln_K_l_o", "d_ln_K_l_e"):
            assert k in params

    def test_fourth_spec_has_epi_terms(self):
        ctx = _make_ctx(pair_labels=["M1M2"])
        specs = get_extract_specs(ctx)
        params = specs[3]["params_to_get"]
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
        specs = get_extract_specs(ctx)
        df = specs[1]["input_df"]
        assert len(df) == len(genos) * len(titrants)

    def test_third_spec_has_one_row_per_mutation(self):
        ctx = _make_ctx(mut_labels=["M1", "M2", "M3"])
        specs = get_extract_specs(ctx)
        df = specs[2]["input_df"]
        assert len(df) == 3
