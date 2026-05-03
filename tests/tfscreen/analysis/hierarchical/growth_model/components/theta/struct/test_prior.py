"""Tests for struct/prior.py — NN-informed ΔΔG sampling."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.prior import (
    sample_ddG,
)


def _trace_sample_ddG(name, struct_names, num_mut, nn_means):
    """Run sample_ddG inside numpyro.handlers.trace and return (output, trace)."""
    def model():
        return sample_ddG(name, struct_names, num_mut, nn_means)

    with numpyro.handlers.seed(rng_seed=42):
        trace = numpyro.handlers.trace(model).get_trace()
    with numpyro.handlers.seed(rng_seed=42):
        out = model()
    return out, trace


class TestSampleDdG:
    def test_output_shape(self):
        M, S = 5, 3
        nn_means     = jnp.zeros((M, S))
        struct_names = [f"S{i}" for i in range(S)]
        out, _ = _trace_sample_ddG("theta", struct_names, M, nn_means)
        assert out.shape == (M, S)

    def test_offset_site_in_trace(self):
        M, S = 4, 2
        nn_means     = jnp.zeros((M, S))
        struct_names = ["H", "L"]
        _, trace = _trace_sample_ddG("theta", struct_names, M, nn_means)
        assert "theta_ddG_offset" in trace

    def test_deterministic_site_in_trace(self):
        M, S = 3, 2
        nn_means     = jnp.zeros((M, S))
        struct_names = ["H", "L"]
        _, trace = _trace_sample_ddG("theta", struct_names, M, nn_means)
        assert "theta_ddG" in trace

    def test_zero_sigma_zero_nn_gives_zero(self):
        """With sigma_s → 0 the output should approach nn_means."""
        M, S = 4, 2
        nn_means     = jnp.ones((M, S)) * 3.0
        struct_names = ["H", "L"]

        import unittest.mock as mock

        # Patch pyro.param to return near-zero sigma_s and zero offsets
        sigma_near_zero = jnp.full(S, 1e-9)

        def mock_param(name, init, **kwargs):
            if "sigma_s" in name:
                return sigma_near_zero
            return init

        with mock.patch("numpyro.param", side_effect=mock_param):
            with numpyro.handlers.seed(rng_seed=0):
                out = sample_ddG("theta", struct_names, M, nn_means)

        # With negligible sigma_s, output ≈ nn_means
        np.testing.assert_allclose(np.asarray(out), np.asarray(nn_means), atol=1e-5)

    def test_nn_means_shift_output(self):
        """Non-zero nn_means should shift the mean of sampled ΔΔG values."""
        M, S = 100, 1
        nn_means_zero = jnp.zeros((M, S))
        nn_means_pos  = jnp.full((M, S), 5.0)

        def run(nn_m):
            def model():
                return sample_ddG("theta", ["H"], M, nn_m)
            with numpyro.handlers.seed(rng_seed=1):
                return model()

        out_zero = np.asarray(run(nn_means_zero))
        out_pos  = np.asarray(run(nn_means_pos))
        # Mean should be shifted by 5.0 (within sampling noise for 100 samples)
        assert out_pos.mean() > out_zero.mean()

    def test_sigma_s_param_registered(self):
        """sigma_s must be a 'param' trace entry, not a 'sample' site."""
        M, S = 3, 2
        nn_means     = jnp.zeros((M, S))
        struct_names = ["H", "L"]
        _, trace = _trace_sample_ddG("theta", struct_names, M, nn_means)
        assert "theta_ddG_sigma_s" in trace
        assert trace["theta_ddG_sigma_s"]["type"] == "param"
