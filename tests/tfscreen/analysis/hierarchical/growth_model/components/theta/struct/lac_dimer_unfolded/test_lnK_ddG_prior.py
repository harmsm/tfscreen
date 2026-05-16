"""
Tests for struct/lac_dimer_unfolded/lnK_ddG_prior.py.

Covers: _get_struct_perm, _project_ddG, ModelPriors, get_hyperparameters,
get_priors, get_guesses, define_model, guide, get_extract_specs.

Key difference from base lac_dimer/lnK_ddG_prior: the unfolded variant adds
a scalar ln_K_U_wt parameter that is sampled for all genotypes but yields a
homogeneous ln_K_U (no per-mutation effects on K_U).
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
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.lac_dimer_unfolded.lnK_ddG_prior import (
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
from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.lac_dimer_unfolded.thermo import (
    ThetaParam,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S = 4    # number of lac_dimer_unfolded structures: H, HD, L, LE2
_M = 2    # mutations
_G = 4    # genotypes: wt, M1, M2, M1+M2

_MUT_GENO = np.array([[0, 1, 0, 1],
                       [0, 0, 1, 1]], dtype=np.float32)  # (M, G)
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)

_CONC = np.array([0.0, 1e-5, 1e-4], dtype=np.float32)

assert set(STRUCTURE_NAMES) == {'H', 'HD', 'L', 'LE2'}


def _make_ddG_prior(seed=0, m=_M, s=_S):
    rng = np.random.RandomState(seed)
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
        # Shuffle: HD, H, LE2, L
        shuffled = ('HD', 'H', 'LE2', 'L')
        data = _make_mock(struct_names=shuffled)
        perm = _get_struct_perm(data)
        # Each perm[i] should index the shuffled list to give STRUCTURE_NAMES[i]
        for i, sname in enumerate(STRUCTURE_NAMES):
            assert shuffled[perm[i]] == sname

    def test_wrong_name_raises(self):
        data = _make_mock(struct_names=('H', 'HD', 'L', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)

    def test_too_few_names_raises(self):
        data = _make_mock(struct_names=('H', 'HD', 'L'))
        with pytest.raises(ValueError, match="struct_names"):
            _get_struct_perm(data)

    def test_structure_names_constant(self):
        assert STRUCTURE_NAMES == ('H', 'HD', 'L', 'LE2')


# ---------------------------------------------------------------------------
# _project_ddG
# ---------------------------------------------------------------------------

class TestProjectDdG:
    """
    Projection: (M, 4) → (M, 3)
    d_ln_K_op = ΔΔG_H − ΔΔG_HD   (col 0 - col 1)
    d_ln_K_HL = ΔΔG_H − ΔΔG_L    (col 0 - col 2)
    d_ln_K_E  = ΔΔG_L − ΔΔG_LE2  (col 2 - col 3)
    """

    def test_output_shape(self):
        assert _project_ddG(jnp.zeros((_M, _S))).shape == (_M, 3)

    def test_zero_ddG_gives_zero_delta_lnK(self):
        np.testing.assert_allclose(_project_ddG(jnp.zeros((3, _S))), 0.0)

    def test_K_op_is_H_minus_HD(self):
        ddG = jnp.array([[1.0, 2.0, 0.0, 0.0]])
        assert _project_ddG(ddG)[0, 0] == pytest.approx(1.0 - 2.0)

    def test_K_HL_is_H_minus_L(self):
        ddG = jnp.array([[3.0, 0.0, 1.0, 0.0]])
        assert _project_ddG(ddG)[0, 1] == pytest.approx(3.0 - 1.0)

    def test_K_E_is_L_minus_LE2(self):
        ddG = jnp.array([[0.0, 0.0, 5.0, 2.0]])
        assert _project_ddG(ddG)[0, 2] == pytest.approx(5.0 - 2.0)

    def test_H_perturbation_does_not_affect_K_E(self):
        """Changing H alone should not change d_ln_K_E (which depends only on L and LE2)."""
        base = jnp.zeros((1, _S))
        perturbed = base.at[0, 0].set(5.0)
        assert _project_ddG(perturbed)[0, 2] == pytest.approx(float(_project_ddG(base)[0, 2]))

    def test_leading_batch_dim(self):
        ddG = jnp.ones((2, 7, 4))
        out = _project_ddG(ddG)
        assert out.shape == (2, 7, 3)


# ---------------------------------------------------------------------------
# ModelPriors / get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

class TestConfig:

    def test_hyperparameters_has_required_keys(self):
        hp = get_hyperparameters()
        required = [
            "theta_ln_K_op_wt_loc", "theta_ln_K_op_wt_scale",
            "theta_ln_K_HL_wt_loc", "theta_ln_K_HL_wt_scale",
            "theta_ln_K_E_wt_loc",  "theta_ln_K_E_wt_scale",
            "theta_ln_K_U_wt_loc",  "theta_ln_K_U_wt_scale",
            "theta_tf_total_M", "theta_op_total_M",
        ]
        for k in required:
            assert k in hp, f"Missing key: {k}"

    def test_ln_K_U_wt_defaults(self):
        hp = get_hyperparameters()
        assert hp["theta_ln_K_U_wt_loc"]   == pytest.approx(-12.0)
        assert hp["theta_ln_K_U_wt_scale"] == pytest.approx(3.0)

    def test_hyperparameters_has_no_nn_keys(self):
        hp = get_hyperparameters()
        assert "theta_nn_hidden_size"  not in hp

    def test_get_priors_returns_model_priors_instance(self):
        assert isinstance(get_priors(), ModelPriors)

    def test_priors_has_ln_K_U_wt_fields(self):
        priors = get_priors()
        assert hasattr(priors, "theta_ln_K_U_wt_loc")
        assert hasattr(priors, "theta_ln_K_U_wt_scale")
        assert priors.theta_ln_K_U_wt_loc   == pytest.approx(-12.0)
        assert priors.theta_ln_K_U_wt_scale == pytest.approx(3.0)

    def test_model_priors_has_no_nn_fields(self):
        priors = get_priors()
        assert not hasattr(priors, "theta_nn_hidden_size")

    def test_physical_constants(self):
        p = get_priors()
        assert p.theta_tf_total_M == pytest.approx(6.5e-7)
        assert p.theta_op_total_M == pytest.approx(2.5e-8)

    def test_get_guesses_required_keys(self):
        g = get_guesses("theta", _make_mock())
        for k in ("theta_ln_K_op_wt", "theta_ln_K_HL_wt", "theta_ln_K_E_wt",
                  "theta_ln_K_U_wt", "theta_ddG_offset"):
            assert k in g, f"Missing guess key: {k}"

    def test_get_guesses_ln_K_U_wt_value(self):
        g = get_guesses("theta", _make_mock())
        assert float(g["theta_ln_K_U_wt"]) == pytest.approx(-12.0)

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
        assert g["theta_ln_K_op_wt"].shape == ()
        assert g["theta_ln_K_HL_wt"].shape == ()
        assert g["theta_ln_K_U_wt"].shape  == ()
        assert g["theta_ln_K_E_wt"].shape  == (T,)


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
        assert out.ln_K_op.shape == (_G,)
        assert out.ln_K_HL.shape == (_G,)
        assert out.ln_K_E.shape  == (T, _G)
        assert out.ln_K_U.shape  == (_G,)

    def test_ln_K_U_homogeneous(self):
        """All elements of ln_K_U must be equal — no per-mutation K_U effects."""
        out, _ = _run_model(_make_mock())
        vals = np.asarray(out.ln_K_U)
        np.testing.assert_allclose(vals, vals[0], atol=1e-6,
                                   err_msg="ln_K_U should be homogeneous across genotypes")

    def test_population_moment_shapes(self):
        out, _ = _run_model(_make_mock())
        T, C = 2, len(_CONC)
        assert out.mu.shape    == (T, C, 1)
        assert out.sigma.shape == (T, C, 1)

    def test_sample_sites_present(self):
        _, tr = _run_model(_make_mock())
        for site in ("theta_ln_K_op_wt", "theta_ln_K_HL_wt", "theta_ln_K_E_wt",
                     "theta_ln_K_U_wt", "theta_ddG_offset"):
            assert site in tr, f"Missing sample site: {site}"

    def test_deterministic_sites_present(self):
        _, tr = _run_model(_make_mock())
        for site in ("theta_ddG", "theta_d_ln_K_op", "theta_d_ln_K_HL", "theta_d_ln_K_E",
                     "theta_ln_K_op", "theta_ln_K_HL", "theta_ln_K_E", "theta_ln_K_U"):
            assert site in tr, f"Missing deterministic site: {site}"

    def test_no_epi_sites(self):
        _, tr = _run_model(_make_mock())
        assert "theta_epi_tau"    not in tr
        assert "theta_epi_lambda" not in tr
        assert "theta_epi_offset" not in tr

    def test_wrong_struct_names_raises(self):
        data = _make_mock(struct_names=('H', 'HD', 'L', 'WRONG'))
        with pytest.raises(ValueError, match="struct_names"):
            _run_model(data)

    def test_permuted_struct_names_accepted(self):
        shuffled = ('HD', 'H', 'LE2', 'L')
        data = _make_mock(struct_names=shuffled)
        out, _ = _run_model(data)
        assert isinstance(out, ThetaParam)

    def test_permuted_features_give_same_result_as_canonical(self):
        """Swapping struct_names + features together should yield identical K values,
        because _get_struct_perm undoes the permutation before projection."""
        canonical_prior = _make_ddG_prior()
        # Permutation: swap H↔HD and L↔LE2
        shuffle_order = [1, 0, 3, 2]
        shuffled_names = tuple(STRUCTURE_NAMES[i] for i in shuffle_order)
        shuffled_prior = canonical_prior[:, shuffle_order]

        canonical_data = _make_mock(struct_names=STRUCTURE_NAMES, ddG_prior=canonical_prior)
        shuffled_data  = _make_mock(struct_names=shuffled_names,  ddG_prior=shuffled_prior)

        priors = get_priors()
        with seed(rng_seed=7):
            out_c = define_model("theta", canonical_data, priors)
        with seed(rng_seed=7):
            out_s = define_model("theta", shuffled_data,  priors)

        np.testing.assert_allclose(
            np.asarray(out_c.ln_K_op),
            np.asarray(out_s.ln_K_op), atol=1e-5,
        )

    def test_ddG_prior_means_shift_delta_lnK(self):
        """With sigma_s=0, ddG equals the prior means exactly; nonzero prior produces
        nonzero delta_lnK for the affected mutation."""
        zero_prior = np.zeros((_M, _S), dtype=np.float32)
        data_zero = _make_mock(ddG_prior=zero_prior)
        priors = get_priors()

        def _zero_param(name, init, **kwargs):
            return jnp.zeros_like(init)

        with mock.patch("numpyro.param", side_effect=_zero_param):
            with seed(rng_seed=0):
                out_zero = define_model("theta", data_zero, priors)

        np.testing.assert_allclose(
            np.asarray(out_zero.ln_K_op[0]),
            np.asarray(out_zero.ln_K_op[1]), atol=1e-5,
        )

        nonzero_prior = np.zeros((_M, _S), dtype=np.float32)
        nonzero_prior[0, 0] = 5.0   # H structure, mutation 0
        data_nonzero = _make_mock(ddG_prior=nonzero_prior)

        with mock.patch("numpyro.param", side_effect=_zero_param):
            with seed(rng_seed=0):
                out_nonzero = define_model("theta", data_nonzero, priors)

        assert not np.allclose(
            np.asarray(out_nonzero.ln_K_op[0]),   # WT
            np.asarray(out_nonzero.ln_K_op[1]),   # M1
            atol=1e-5,
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
        assert out.ln_K_op.shape == (_G,)
        assert out.ln_K_HL.shape == (_G,)
        assert out.ln_K_E.shape  == (T, _G)
        assert out.ln_K_U.shape  == (_G,)

    def test_ln_K_U_homogeneous_in_guide(self):
        out, _ = _run_guide(_make_mock())
        vals = np.asarray(out.ln_K_U)
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
        for p in ("theta_ln_K_op_wt_loc", "theta_ln_K_op_wt_scale",
                  "theta_ln_K_HL_wt_loc", "theta_ln_K_HL_wt_scale",
                  "theta_ddG_offset_locs", "theta_ddG_offset_scales",
                  "theta_ddG_sigma_s"):
            assert p in param_names, f"Missing variational param: {p}"

    def test_ln_K_U_wt_variational_params_registered(self):
        """Guide must register variational parameters for ln_K_U_wt."""
        _, tr = _run_guide(_make_mock())
        param_names = {k for k, v in tr.items() if v["type"] == "param"}
        assert "theta_ln_K_U_wt_loc"   in param_names
        assert "theta_ln_K_U_wt_scale" in param_names

    def test_no_nn_params_registered(self):
        """lnK_ddG_prior has no NN; no per-structure NN weight params should appear."""
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
        shuffle_order   = [1, 0, 3, 2]
        shuffled_names  = tuple(STRUCTURE_NAMES[i] for i in shuffle_order)
        shuffled_prior  = canonical_prior[:, shuffle_order]

        canonical_data = _make_mock(struct_names=STRUCTURE_NAMES, ddG_prior=canonical_prior)
        shuffled_data  = _make_mock(struct_names=shuffled_names,  ddG_prior=shuffled_prior)

        priors = get_priors()
        with seed(rng_seed=5):
            out_c = guide("theta", canonical_data, priors)
        with seed(rng_seed=5):
            out_s = guide("theta", shuffled_data,  priors)

        np.testing.assert_allclose(
            np.asarray(out_c.ln_K_op),
            np.asarray(out_s.ln_K_op), atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------

def test_registry_entry():
    from tfscreen.analysis.hierarchical.growth_model.registry import model_registry
    assert "lac_dimer_unfolded_lnK_ddG_prior" in model_registry["theta"]
    import tfscreen.analysis.hierarchical.growth_model.components.theta.struct.lac_dimer_unfolded.lnK_ddG_prior as mod
    assert model_registry["theta"]["lac_dimer_unfolded_lnK_ddG_prior"] is mod


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
        # scalar K values (ln_K_op, ln_K_HL), titrant K values (ln_K_E),
        # per-mutation d_ln_K values — ln_K_U is WT-only, not extracted per-genotype
        assert len(get_extract_specs(_make_ctx())) == 3

    def test_four_specs_when_pair_labels_present(self):
        ctx = _make_ctx(pair_labels=["M1+M2"])
        assert len(get_extract_specs(ctx)) == 4

    def test_first_spec_has_scalar_K_values(self):
        params = get_extract_specs(_make_ctx())[0]["params_to_get"]
        assert "ln_K_op" in params
        assert "ln_K_HL" in params

    def test_second_spec_has_titrant_K_values(self):
        params = get_extract_specs(_make_ctx())[1]["params_to_get"]
        assert "ln_K_E" in params

    def test_third_spec_has_projected_delta_lnK(self):
        params = get_extract_specs(_make_ctx())[2]["params_to_get"]
        for k in ("d_ln_K_op", "d_ln_K_HL", "d_ln_K_E"):
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

    def test_map_geno_covers_all_genotypes(self):
        genos = ["wt", "M1", "M2"]
        ctx = _make_ctx(genotypes=genos)
        df = get_extract_specs(ctx)[0]["input_df"]
        assert set(df["genotype"]) == set(genos)
