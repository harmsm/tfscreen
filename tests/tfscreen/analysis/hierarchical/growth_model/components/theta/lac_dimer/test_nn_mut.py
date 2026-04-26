import pytest
import numpy as np
import jax.numpy as jnp
from numpyro.handlers import trace, substitute, seed
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.nn_mut import (
    ModelPriors,
    _PROJ,
    _DEFAULT_HIDDEN_SIZE,
    _ALPHABET,
    _mlp_forward,
    STRUCTURE_KEYS,
    build_ligandmpnn_features,
    define_model,
    guide,
    get_hyperparameters,
    get_priors,
    get_guesses,
    run_model,
    get_population_moments,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MockData = namedtuple("MockData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "num_mutation",
    "num_pair",
    "titrant_conc",
    "geno_theta_idx",
    "scatter_theta",
    "mut_geno_matrix",
    "ligandmpnn_features",
])

_CONC = np.array([0.0, 100.0, 1000.0])
_T, _C, _G, _M = 2, 3, 4, 5

# Simple mut_geno_matrix: each mutation present in exactly one genotype
_MUT_GENO = np.zeros((_M, _G), dtype=np.float32)
for _m in range(_M):
    _MUT_GENO[_m, _m % _G] = 1.0

# Random ΔlogP features (M, 4)
rng = np.random.default_rng(0)
_FEATURES = rng.standard_normal((_M, 4)).astype(np.float32)


@pytest.fixture
def mock_data():
    return MockData(
        num_titrant_name=_T,
        num_titrant_conc=_C,
        num_genotype=_G,
        num_mutation=_M,
        num_pair=0,
        titrant_conc=jnp.array(_CONC, dtype=jnp.float32),
        geno_theta_idx=jnp.arange(_G, dtype=jnp.int32),
        scatter_theta=0,
        mut_geno_matrix=_MUT_GENO,
        ligandmpnn_features=_FEATURES,
    )


@pytest.fixture
def priors():
    return get_priors()


# ---------------------------------------------------------------------------
# Helpers for NPZ-based tests
# ---------------------------------------------------------------------------

def _make_npz(tmp_path, L=30, res_nums=None, logP_override=None):
    """Write a minimal valid NPZ with STRUCTURE_KEYS arrays."""
    if res_nums is None:
        res_nums = np.arange(1, L + 1, dtype=np.int32)
    arrays = {}
    for key in STRUCTURE_KEYS:
        lp = logP_override.get(key, np.zeros((L, 20), dtype=np.float32)) \
             if logP_override else np.zeros((L, 20), dtype=np.float32)
        arrays[key] = lp
        arrays[f"{key}_residue_nums"] = res_nums
    path = tmp_path / "features.npz"
    np.savez(path, **arrays)
    return str(path)


# ---------------------------------------------------------------------------
# STRUCTURE_KEYS
# ---------------------------------------------------------------------------

class TestStructureKeys:

    def test_length(self):
        assert len(STRUCTURE_KEYS) == 4

    def test_order_matches_proj(self):
        """Column order must match _PROJ: H=0, HD=1, L=2, LE2=3."""
        assert STRUCTURE_KEYS == ('H', 'HD', 'L', 'LE2')

    def test_alphabet_length(self):
        assert len(_ALPHABET) == 20


# ---------------------------------------------------------------------------
# build_ligandmpnn_features
# ---------------------------------------------------------------------------

class TestBuildLigandmpnnFeatures:

    def test_output_shape(self, tmp_path):
        path = _make_npz(tmp_path, L=40)
        labels = ["A1G", "C2D", "D3E", "E4F", "F5G"]
        feat = build_ligandmpnn_features(path, labels)
        assert feat.shape == (5, len(STRUCTURE_KEYS))

    def test_zero_logP_gives_zero_delta(self, tmp_path):
        """All-zero logP → ΔlogP = 0 for any mutation."""
        path = _make_npz(tmp_path, L=10)
        feat = build_ligandmpnn_features(path, ["A3G"])
        assert np.allclose(feat, 0.0)

    def test_correct_delta_values(self, tmp_path):
        """ΔlogP = logP(mut) - logP(wt) extracted per structure."""
        L = 20
        res_nums = np.arange(1, L + 1, dtype=np.int32)
        # Position 5 (1-indexed → arr_idx 4): logP(A)=log(0.4), logP(G)=log(0.1)
        a_idx = _ALPHABET.index('A')
        g_idx = _ALPHABET.index('G')
        logP_override = {}
        for s_idx, key in enumerate(STRUCTURE_KEYS):
            lp = np.zeros((L, 20), dtype=np.float32)
            lp[4, a_idx] = np.log(0.4) * (s_idx + 1)   # wt, varies by structure
            lp[4, g_idx] = np.log(0.1) * (s_idx + 1)   # mut
            logP_override[key] = lp
        path = _make_npz(tmp_path, L=L, res_nums=res_nums, logP_override=logP_override)
        feat = build_ligandmpnn_features(path, ["A5G"])
        for s_idx in range(len(STRUCTURE_KEYS)):
            expected = (np.log(0.1) - np.log(0.4)) * (s_idx + 1)
            assert np.isclose(feat[0, s_idx], expected, atol=1e-5), \
                f"structure {s_idx}: got {feat[0, s_idx]:.4f}, expected {expected:.4f}"

    def test_missing_structure_key_raises(self, tmp_path):
        L = 10
        # Only H present
        path = tmp_path / "bad.npz"
        np.savez(str(path),
                 H=np.zeros((L, 20)),
                 H_residue_nums=np.arange(1, L + 1, dtype=np.int32))
        with pytest.raises(KeyError, match="missing"):
            build_ligandmpnn_features(str(path), ["A1G"])

    def test_unknown_position_raises(self, tmp_path):
        path = _make_npz(tmp_path, L=10)  # residues 1–10
        with pytest.raises(ValueError, match="not found"):
            build_ligandmpnn_features(path, ["A99G"])

    def test_bad_mutation_label_raises(self, tmp_path):
        path = _make_npz(tmp_path, L=10)
        with pytest.raises(ValueError, match="parse"):
            build_ligandmpnn_features(path, ["bad_label"])

    def test_multiple_mutations(self, tmp_path):
        path = _make_npz(tmp_path, L=20)
        labels = [f"A{i}G" for i in range(1, 6)]
        feat = build_ligandmpnn_features(path, labels)
        assert feat.shape == (5, len(STRUCTURE_KEYS))
        assert np.all(np.isfinite(feat))

    def test_structure_key_column_order(self, tmp_path):
        """The H column (index 0) should use the H logP array."""
        L = 10
        a_idx = _ALPHABET.index('A')
        g_idx = _ALPHABET.index('G')
        # Only H has a nonzero logP at position 5
        logP_override = {}
        for key in STRUCTURE_KEYS:
            lp = np.zeros((L, 20), dtype=np.float32)
            if key == 'H':
                lp[4, g_idx] = 2.0   # mut logP in H only
                lp[4, a_idx] = 1.0   # wt logP in H only
            logP_override[key] = lp
        path = _make_npz(tmp_path, L=L, logP_override=logP_override)
        feat = build_ligandmpnn_features(path, ["A5G"])
        h_col = list(STRUCTURE_KEYS).index('H')
        assert np.isclose(feat[0, h_col], 1.0, atol=1e-5)        # 2.0 - 1.0
        for key in STRUCTURE_KEYS:
            if key != 'H':
                col = list(STRUCTURE_KEYS).index(key)
                assert np.isclose(feat[0, col], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# _PROJ matrix
# ---------------------------------------------------------------------------

class TestProjMatrix:

    def test_shape(self):
        assert _PROJ.shape == (3, 4)

    def test_K_op_row(self):
        """Δln_K_op = ΔΔG_H - ΔΔG_HD → row [+1,-1, 0, 0]."""
        expected = jnp.array([1., -1., 0., 0.])
        assert jnp.allclose(_PROJ[0], expected)

    def test_K_HL_row(self):
        """Δln_K_HL = ΔΔG_H - ΔΔG_L → row [+1, 0,-1, 0]."""
        expected = jnp.array([1., 0., -1., 0.])
        assert jnp.allclose(_PROJ[1], expected)

    def test_K_E_row(self):
        """Δln_K_E = ΔΔG_L - ΔΔG_LE2 → row [0, 0,+1,-1]."""
        expected = jnp.array([0., 0., 1., -1.])
        assert jnp.allclose(_PROJ[2], expected)

    def test_zero_ddG_gives_zero_delta(self):
        ddG = jnp.zeros((6, 4))
        delta = ddG @ _PROJ.T
        assert jnp.allclose(delta, 0.0)

    def test_unit_ddG_H_only(self):
        """If only ΔΔG_H = 1, then Δln_K_op = Δln_K_HL = 1, Δln_K_E = 0."""
        ddG = jnp.array([[1., 0., 0., 0.]])
        delta = (ddG @ _PROJ.T)[0]
        assert jnp.allclose(delta, jnp.array([1., 1., 0.]))


# ---------------------------------------------------------------------------
# _mlp_forward
# ---------------------------------------------------------------------------

class TestMlpForward:

    def test_output_shape(self):
        H = 8
        W1 = jnp.zeros((4, H))
        b1 = jnp.zeros(H)
        W2 = jnp.zeros((H, 4))
        b2 = jnp.zeros(4)
        features = jnp.ones((_M, 4))
        out = _mlp_forward(features, W1, b1, W2, b2)
        assert out.shape == (_M, 4)

    def test_zero_weights_give_zero_output(self):
        H = 8
        W1 = jnp.zeros((4, H))
        b1 = jnp.zeros(H)
        W2 = jnp.zeros((H, 4))
        b2 = jnp.zeros(4)
        features = jnp.array(_FEATURES)
        out = _mlp_forward(features, W1, b1, W2, b2)
        assert jnp.allclose(out, 0.0)

    def test_nonzero_weights_change_output(self):
        H = 8
        W1 = jnp.ones((4, H)) * 0.1
        b1 = jnp.zeros(H)
        W2 = jnp.ones((H, 4)) * 0.1
        b2 = jnp.zeros(4)
        features = jnp.ones((_M, 4))
        out = _mlp_forward(features, W1, b1, W2, b2)
        assert not jnp.allclose(out, 0.0)

    def test_finite_output(self):
        H = 8
        rng2 = np.random.default_rng(1)
        W1 = jnp.array(rng2.standard_normal((4, H)).astype(np.float32))
        b1 = jnp.array(rng2.standard_normal(H).astype(np.float32))
        W2 = jnp.array(rng2.standard_normal((H, 4)).astype(np.float32))
        b2 = jnp.array(rng2.standard_normal(4).astype(np.float32))
        features = jnp.array(_FEATURES)
        out = _mlp_forward(features, W1, b1, W2, b2)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# get_hyperparameters / get_priors / get_guesses
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_hyperparameters_keys_match_modelpriors(self):
        hp = get_hyperparameters()
        priors = ModelPriors(**hp)   # raises if keys don't match
        assert priors is not None

    def test_get_priors_returns_modelpriors(self):
        p = get_priors()
        assert isinstance(p, ModelPriors)

    def test_default_hidden_size_consistent(self):
        p = get_priors()
        assert p.theta_nn_hidden_size == _DEFAULT_HIDDEN_SIZE

    def test_get_guesses_keys(self, mock_data):
        g = get_guesses("theta", mock_data)
        assert "theta_ln_K_op_wt" in g
        assert "theta_ln_K_HL_wt" in g
        assert "theta_ln_K_E_wt"  in g
        assert "theta_nn_W1"      in g
        assert "theta_nn_b1"      in g
        assert "theta_nn_W2"      in g
        assert "theta_nn_b2"      in g

    def test_get_guesses_shapes(self, mock_data):
        H = _DEFAULT_HIDDEN_SIZE
        g = get_guesses("theta", mock_data)
        assert g["theta_nn_W1"].shape == (4, H)
        assert g["theta_nn_b1"].shape == (H,)
        assert g["theta_nn_W2"].shape == (H, 4)
        assert g["theta_nn_b2"].shape == (4,)
        assert g["theta_ln_K_E_wt"].shape == (_T,)

    def test_get_guesses_zero_init_weights(self, mock_data):
        g = get_guesses("theta", mock_data)
        assert jnp.allclose(g["theta_nn_W1"], 0.0)
        assert jnp.allclose(g["theta_nn_b1"], 0.0)
        assert jnp.allclose(g["theta_nn_W2"], 0.0)
        assert jnp.allclose(g["theta_nn_b2"], 0.0)


# ---------------------------------------------------------------------------
# define_model
# ---------------------------------------------------------------------------

def _run_model(fn, mock_data, priors, seed_val=0):
    """Run fn under a fixed seed and return its output."""
    return seed(fn, seed_val)("theta", mock_data, priors)


class TestDefineModel:

    def test_returns_theta_param(self, mock_data, priors):
        from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.thermo import ThetaParam
        result = _run_model(define_model, mock_data, priors)
        assert isinstance(result, ThetaParam)

    def test_ln_K_op_shape(self, mock_data, priors):
        tp = _run_model(define_model, mock_data, priors)
        assert tp.ln_K_op.shape == (_G,)

    def test_ln_K_HL_shape(self, mock_data, priors):
        tp = _run_model(define_model, mock_data, priors)
        assert tp.ln_K_HL.shape == (_G,)

    def test_ln_K_E_shape(self, mock_data, priors):
        tp = _run_model(define_model, mock_data, priors)
        assert tp.ln_K_E.shape == (_T, _G)

    def test_mu_shape(self, mock_data, priors):
        tp = _run_model(define_model, mock_data, priors)
        assert tp.mu.shape == (_T, _C, 1)

    def test_sigma_shape(self, mock_data, priors):
        tp = _run_model(define_model, mock_data, priors)
        assert tp.sigma.shape == (_T, _C, 1)

    def test_all_finite(self, mock_data, priors):
        tp = _run_model(define_model, mock_data, priors)
        assert jnp.all(jnp.isfinite(tp.ln_K_op))
        assert jnp.all(jnp.isfinite(tp.ln_K_HL))
        assert jnp.all(jnp.isfinite(tp.ln_K_E))
        assert jnp.all(jnp.isfinite(tp.mu))
        assert jnp.all(jnp.isfinite(tp.sigma))

    def test_perturbative_start_gives_wt_K(self, mock_data):
        """Zero-init weights → d_lnK = 0 → all genotypes have same K as WT."""
        ln_K_op_wt = 2.3
        ln_K_HL_wt = -9.0
        ln_K_E_wt  = -8.0
        p = get_priors()

        # Patch the model to use zero weights via substitute
        zero_weights = {
            "theta_nn_W1": jnp.zeros((4, _DEFAULT_HIDDEN_SIZE)),
            "theta_nn_b1": jnp.zeros(_DEFAULT_HIDDEN_SIZE),
            "theta_nn_W2": jnp.zeros((_DEFAULT_HIDDEN_SIZE, 4)),
            "theta_nn_b2": jnp.zeros(4),
            "theta_ln_K_op_wt": jnp.array(ln_K_op_wt),
            "theta_ln_K_HL_wt": jnp.array(ln_K_HL_wt),
            "theta_ln_K_E_wt":  jnp.full(_T, ln_K_E_wt),
        }
        tp = substitute(define_model, zero_weights)("theta", mock_data, p)

        assert jnp.allclose(tp.ln_K_op, ln_K_op_wt, atol=1e-5)
        assert jnp.allclose(tp.ln_K_HL, ln_K_HL_wt, atol=1e-5)
        for t in range(_T):
            assert jnp.allclose(tp.ln_K_E[t], ln_K_E_wt, atol=1e-5)

    def test_sample_sites_present(self, mock_data, priors):
        tr = trace(seed(define_model, 0)).get_trace("theta", mock_data, priors)
        assert "theta_ln_K_op_wt" in tr
        assert "theta_ln_K_HL_wt" in tr
        assert "theta_ln_K_E_wt"  in tr
        assert "theta_nn_W1"      in tr
        assert "theta_nn_b1"      in tr
        assert "theta_nn_W2"      in tr
        assert "theta_nn_b2"      in tr

    def test_deterministic_sites_present(self, mock_data, priors):
        tr = trace(seed(define_model, 0)).get_trace("theta", mock_data, priors)
        assert "theta_d_ln_K_op" in tr
        assert "theta_d_ln_K_HL" in tr
        assert "theta_d_ln_K_E"  in tr
        assert "theta_ln_K_op"   in tr
        assert "theta_ln_K_HL"   in tr
        assert "theta_ln_K_E"    in tr


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------

class TestGuide:

    def test_returns_theta_param(self, mock_data, priors):
        from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.thermo import ThetaParam
        result = _run_model(guide, mock_data, priors)
        assert isinstance(result, ThetaParam)

    def test_ln_K_op_shape(self, mock_data, priors):
        tp = _run_model(guide, mock_data, priors)
        assert tp.ln_K_op.shape == (_G,)

    def test_all_finite(self, mock_data, priors):
        tp = _run_model(guide, mock_data, priors)
        assert jnp.all(jnp.isfinite(tp.ln_K_op))
        assert jnp.all(jnp.isfinite(tp.ln_K_E))
        assert jnp.all(jnp.isfinite(tp.mu))


# ---------------------------------------------------------------------------
# run_model / get_population_moments (passthrough from thermo.py)
# ---------------------------------------------------------------------------

def test_run_model_passthrough(mock_data, priors):
    from tfscreen.analysis.hierarchical.growth_model.components.theta.lac_dimer.thermo import ThetaParam
    tp = _run_model(define_model, mock_data, priors)
    result = run_model(tp, mock_data)
    assert result.shape == (_T, _C, _G)
    assert jnp.all(result >= 0)
    assert jnp.all(result <= 1)


def test_get_population_moments_passthrough(mock_data, priors):
    tp = _run_model(define_model, mock_data, priors)
    mu, sigma = get_population_moments(tp, mock_data)
    assert jnp.allclose(mu, tp.mu)
    assert jnp.allclose(sigma, tp.sigma)
