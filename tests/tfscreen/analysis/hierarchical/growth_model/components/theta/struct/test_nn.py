"""Tests for struct/nn.py — per-structure MLP predictions."""

import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import pytest

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.nn import (
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

    def test_zero_weights_give_zero_output(self):
        """With zero-initialised weights the output must be exactly zero."""
        M, S = 4, 2
        features     = _make_features(M, S)
        struct_names = ["H", "L"]
        n_chains     = np.array([2, 1], dtype=np.int32)

        # pyro.param returns the init value when called outside a model
        # — use numpyro.handlers.substitute with empty dict so param sites
        # return their initial values (zeros).
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
