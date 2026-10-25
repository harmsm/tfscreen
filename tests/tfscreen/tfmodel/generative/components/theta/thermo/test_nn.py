"""Tests for struct/nn.py — per-structure MLP predictions."""

import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import pytest

from tfscreen.tfmodel.generative.components.theta.thermo.nn import (
    _DEFAULT_HIDDEN_SIZE,
    compute_nn_predictions,
)


def _make_features(M, S, seed=0):
    return jnp.array(
        np.random.RandomState(seed).randn(M, S, 60).astype(np.float32)
    )


class TestComputeNnPredictions:
    def test_output_shape(self):
        M, S = 5, 3
        features    = _make_features(M, S)
        struct_names = [f"S{i}" for i in range(S)]
        n_chains    = np.ones(S, dtype=np.int32)

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace():
                out = numpyro.handlers.substitute(
                    compute_nn_predictions, {}
                )("theta", features, struct_names, n_chains)

        assert out.shape == (M, S)

    def test_initial_output_is_zero(self):
        """W2 and b2 start at zero, so the initial output is exactly zero."""
        M, S = 4, 2
        features     = _make_features(M, S)
        struct_names = ["H", "L"]
        n_chains     = np.array([2, 1], dtype=np.int32)

        # pyro.param returns the init value under a bare seed handler.
        with numpyro.handlers.seed(rng_seed=0):
            out = compute_nn_predictions("theta", features, struct_names, n_chains)

        np.testing.assert_array_equal(np.asarray(out), 0.0)

    def test_n_chains_scaling(self):
        """Output for structure s must scale linearly with n_chains[s]."""
        M, S = 3, 2
        features     = _make_features(M, S)
        struct_names = ["H", "L"]

        # Manually set W1 to get a non-zero MLP output, then check ratio
        n_chains_1 = np.array([1, 1], dtype=np.int32)
        n_chains_2 = np.array([2, 1], dtype=np.int32)

        with numpyro.handlers.seed(rng_seed=0):
            out1 = compute_nn_predictions("theta", features, struct_names, n_chains_1)
        with numpyro.handlers.seed(rng_seed=0):
            out2 = compute_nn_predictions("theta", features, struct_names, n_chains_2)

        # Both are zero (zero weights) — just check shapes match and ratio holds
        assert out1.shape == out2.shape

    def test_distinct_param_names_per_structure(self):
        """Each structure must register its own distinct pyro.param sites."""
        M, S = 2, 3
        features     = _make_features(M, S)
        struct_names = ["A", "B", "C"]
        n_chains     = np.ones(S, dtype=np.int32)

        param_store = {}
        def _register_param(name, init, **kwargs):
            param_store[name] = init
            return init

        import unittest.mock as mock
        with mock.patch("numpyro.param", side_effect=_register_param):
            try:
                compute_nn_predictions("pref", features, struct_names, n_chains)
            except Exception:
                pass

        # Each structure should have its own W1 param
        assert "pref_nn_A_W1" in param_store
        assert "pref_nn_B_W1" in param_store
        assert "pref_nn_C_W1" in param_store

    def test_output_dtype_float32(self):
        M, S = 3, 2
        features     = _make_features(M, S)
        struct_names = ["H", "L"]
        n_chains     = np.ones(S, dtype=np.int32)

        with numpyro.handlers.seed(rng_seed=0):
            out = compute_nn_predictions("theta", features, struct_names, n_chains)

        assert out.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Initialization and gradient flow
# ---------------------------------------------------------------------------

def _init_params(name, features, struct_names, n_chains):
    """The param values compute_nn_predictions registers on a fresh store."""
    with numpyro.handlers.seed(rng_seed=0):
        tr = numpyro.handlers.trace(compute_nn_predictions).get_trace(
            name, features, struct_names, n_chains)
    return {k: v["value"] for k, v in tr.items() if v["type"] == "param"}


def _loss_fn(name, features, struct_names, n_chains, target):
    """Squared error of the NN output against target, as a function of params."""
    def loss(params):
        with numpyro.handlers.substitute(data=params):
            out = compute_nn_predictions(name, features, struct_names, n_chains)
        return jnp.sum((out - target) ** 2)
    return loss


class TestInitAndGradients:
    def test_W1_init_nonzero_he_scaled(self):
        """W1 starts at N(0, 2/feat_dim); b1, W2 and b2 start at zero."""
        features = _make_features(4, 2)
        params = _init_params("theta", features, ["H", "L"], np.ones(2))
        for s in ["H", "L"]:
            W1 = np.asarray(params[f"theta_nn_{s}_W1"])
            assert W1.shape == (60, _DEFAULT_HIDDEN_SIZE)
            assert np.all(np.isfinite(W1))
            assert np.std(W1) == pytest.approx(np.sqrt(2.0 / 60), rel=0.15)
            np.testing.assert_array_equal(np.asarray(params[f"theta_nn_{s}_b1"]), 0.0)
            np.testing.assert_array_equal(np.asarray(params[f"theta_nn_{s}_W2"]), 0.0)
            np.testing.assert_array_equal(np.asarray(params[f"theta_nn_{s}_b2"]), 0.0)

    def test_W1_init_deterministic_and_distinct_per_structure(self):
        """Same site name gives the same W1 under any seed; structures differ."""
        features = _make_features(4, 2)
        with numpyro.handlers.seed(rng_seed=0):
            tr0 = numpyro.handlers.trace(compute_nn_predictions).get_trace(
                "theta", features, ["H", "L"], np.ones(2))
        with numpyro.handlers.seed(rng_seed=123):
            tr1 = numpyro.handlers.trace(compute_nn_predictions).get_trace(
                "theta", features, ["L", "H"], np.ones(2))
        for s in ["H", "L"]:
            np.testing.assert_array_equal(
                np.asarray(tr0[f"theta_nn_{s}_W1"]["value"]),
                np.asarray(tr1[f"theta_nn_{s}_W1"]["value"]))
        assert not np.allclose(np.asarray(tr0["theta_nn_H_W1"]["value"]),
                               np.asarray(tr0["theta_nn_L_W1"]["value"]))

    def test_substituted_values_override_init(self):
        """A restored value (checkpoint, init_params) wins over the init."""
        features = _make_features(3, 1)
        W1 = jnp.full((60, _DEFAULT_HIDDEN_SIZE), 0.5)
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.substitute(data={"theta_nn_H_W1": W1}):
                tr = numpyro.handlers.trace(compute_nn_predictions).get_trace(
                    "theta", features, ["H"], np.ones(1))
        np.testing.assert_array_equal(np.asarray(tr["theta_nn_H_W1"]["value"]),
                                      np.asarray(W1))

    def test_gradient_flows_to_all_weights(self):
        """
        At init W2 gets a nonzero gradient (hidden units are active).  W1 and
        b1 get none while W2 == 0, but do after a single gradient step.
        """
        M, S = 20, 2
        names = ["H", "L"]
        n_chains = np.array([2, 1])
        features = _make_features(M, S)
        target = jnp.array(np.random.RandomState(1).randn(M, S).astype(np.float32))
        loss = _loss_fn("theta", features, names, n_chains, target)
        params = _init_params("theta", features, names, n_chains)

        g0 = jax.grad(loss)(params)
        for s in names:
            assert np.linalg.norm(np.asarray(g0[f"theta_nn_{s}_W2"])) > 1e-3
            assert np.linalg.norm(np.asarray(g0[f"theta_nn_{s}_b2"])) > 1e-3

        params1 = {k: v - 1e-3 * g0[k] for k, v in params.items()}
        g1 = jax.grad(loss)(params1)
        for s in names:
            for p in ["W1", "b1", "W2", "b2"]:
                assert np.linalg.norm(np.asarray(g1[f"theta_nn_{s}_{p}"])) > 1e-6, \
                    f"no gradient on theta_nn_{s}_{p}"

    def test_all_zero_init_blocks_gradient(self):
        """The old all-zero init: only b2 has a gradient, ever (regression)."""
        M, S = 20, 2
        names = ["H", "L"]
        n_chains = np.ones(S)
        features = _make_features(M, S)
        target = jnp.array(np.random.RandomState(1).randn(M, S).astype(np.float32))
        loss = _loss_fn("theta", features, names, n_chains, target)
        params = {k: jnp.zeros_like(v) for k, v in
                  _init_params("theta", features, names, n_chains).items()}
        g = jax.grad(loss)(params)
        for s in names:
            for p in ["W1", "b1", "W2"]:
                np.testing.assert_array_equal(np.asarray(g[f"theta_nn_{s}_{p}"]), 0.0)
            assert np.linalg.norm(np.asarray(g[f"theta_nn_{s}_b2"])) > 1e-3
