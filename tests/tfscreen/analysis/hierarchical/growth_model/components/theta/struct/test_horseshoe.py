"""Tests for struct/horseshoe.py — distance-dependent horseshoe prior."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from tfscreen.analysis.hierarchical.growth_model.components.theta.struct.horseshoe import (
    _DEFAULT_D0,
    sample_pair_ddG,
)


def _trace_horseshoe(name, struct_names, distances, **kwargs):
    """Run sample_pair_ddG inside trace and return (output, trace)."""
    def model():
        return sample_pair_ddG(name, struct_names, jnp.array(distances), **kwargs)

    with numpyro.handlers.seed(rng_seed=7):
        trace = numpyro.handlers.trace(model).get_trace()
    with numpyro.handlers.seed(rng_seed=7):
        out = model()
    return out, trace


class TestSamplePairDdG:
    def test_output_shape(self):
        P, S = 4, 3
        distances    = np.full((P, S), 10.0, dtype=np.float32)
        struct_names = [f"S{i}" for i in range(S)]
        out, _ = _trace_horseshoe("theta", struct_names, distances)
        assert out.shape == (P, S)

    def test_tau_site_in_trace(self):
        distances = np.ones((2, 2), dtype=np.float32)
        _, trace  = _trace_horseshoe("theta", ["H", "L"], distances)
        assert "theta_epi_tau"    in trace
        assert "theta_epi_c2"     in trace

    def test_lambda_and_offset_sites_in_trace(self):
        distances = np.ones((3, 2), dtype=np.float32)
        _, trace  = _trace_horseshoe("theta", ["H", "L"], distances)
        assert "theta_epi_lambda" in trace
        assert "theta_epi_offset" in trace

    def test_deterministic_site_in_trace(self):
        distances = np.ones((2, 2), dtype=np.float32)
        _, trace  = _trace_horseshoe("theta", ["H", "L"], distances)
        assert "theta_epi_ddG" in trace

    def test_large_distance_shrinks_lambda_scale(self):
        """
        HalfCauchy(scale) shrinks toward zero as scale → 0.  Very distant
        pairs (distance ≫ d0) should produce smaller mean |λ| than close ones.
        """
        d0 = _DEFAULT_D0
        P = 200

        close_dists = np.full((P, 1), 0.1, dtype=np.float32)
        far_dists   = np.full((P, 1), 1e4, dtype=np.float32)

        out_close, _ = _trace_horseshoe("c", ["H"], close_dists, d0=d0)
        out_far,   _ = _trace_horseshoe("f", ["H"], far_dists,   d0=d0)

        # Close contacts should have larger mean absolute epistasis
        assert np.abs(np.asarray(out_close)).mean() >= np.abs(np.asarray(out_far)).mean()

    def test_single_pair_single_struct(self):
        distances = np.array([[5.0]], dtype=np.float32)
        out, _    = _trace_horseshoe("theta", ["H"], distances)
        assert out.shape == (1, 1)

    def test_output_dtype_float32(self):
        distances = np.ones((3, 2), dtype=np.float32)
        out, _    = _trace_horseshoe("theta", ["H", "L"], distances)
        assert out.dtype == jnp.float32

    def test_tau_strictly_positive(self):
        """τ must be positive (HalfCauchy prior)."""
        distances = np.ones((2, 2), dtype=np.float32)
        _, trace  = _trace_horseshoe("theta", ["H", "L"], distances)
        tau_val = trace["theta_epi_tau"]["value"]
        assert float(tau_val) > 0.0

    def test_c2_strictly_positive(self):
        """c² must be positive (InverseGamma prior)."""
        distances = np.ones((2, 2), dtype=np.float32)
        _, trace  = _trace_horseshoe("theta", ["H", "L"], distances)
        c2_val = trace["theta_epi_c2"]["value"]
        assert float(c2_val) > 0.0
