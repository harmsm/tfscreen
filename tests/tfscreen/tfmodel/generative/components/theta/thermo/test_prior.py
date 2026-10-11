"""Tests for thermo/prior.py: the centered per-structure ΔΔG prior."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    ddG_guesses,
    guide_ddG,
    sample_ddG,
)


def _trace_sample_ddG(name, struct_names, num_mut, nn_means, **kwargs):
    """Run sample_ddG inside numpyro.handlers.trace and return (output, trace)."""
    def model():
        return sample_ddG(name, struct_names, num_mut, nn_means, **kwargs)

    with numpyro.handlers.seed(rng_seed=42):
        trace = numpyro.handlers.trace(model).get_trace()
    with numpyro.handlers.seed(rng_seed=42):
        out = model()
    return out, trace


def _base(fn):
    """Strip Independent/ExpandedDistribution wrappers."""
    while hasattr(fn, "base_dist"):
        fn = fn.base_dist
    return fn


class TestSampleDdG:
    def test_output_shape(self):
        M, S = 5, 3
        out, _ = _trace_sample_ddG("theta", [f"S{i}" for i in range(S)], M,
                                   jnp.zeros((M, S)))
        assert out.shape == (M, S)

    def test_ddG_is_a_centered_sample_site(self):
        M, S = 4, 2
        nn_means = jnp.arange(M * S, dtype=jnp.float32).reshape(M, S)
        out, trace = _trace_sample_ddG("theta", ["H", "L"], M, nn_means)
        site = trace["theta_ddG"]
        assert site["type"] == "sample"
        assert site["value"].shape == (M, S)
        # Centered: the site's own distribution is Normal(prior mean, σ_s).
        fn = _base(site["fn"])
        assert isinstance(fn, dist.Normal)
        np.testing.assert_allclose(np.broadcast_to(fn.loc, (M, S)), nn_means)
        sigma = trace["theta_ddG_sigma"]["value"]
        np.testing.assert_allclose(np.broadcast_to(fn.scale, (M, S)),
                                   np.broadcast_to(sigma, (M, S)))
        np.testing.assert_allclose(out, site["value"])

    def test_sigma_is_a_lognormal_sample_site_per_structure(self):
        M, S = 3, 2
        _, trace = _trace_sample_ddG("theta", ["H", "L"], M, jnp.zeros((M, S)),
                                     sigma_loc=0.5, sigma_scale=0.7)
        site = trace["theta_ddG_sigma"]
        assert site["type"] == "sample"
        assert site["value"].shape == (S,)
        fn = site["fn"]
        while not isinstance(fn, dist.LogNormal):
            fn = fn.base_dist
        np.testing.assert_allclose(fn.loc, 0.5)
        np.testing.assert_allclose(fn.scale, 0.7)

    def test_no_param_sites(self):
        """σ_s is no longer a model-side pyro.param."""
        _, trace = _trace_sample_ddG("theta", ["H", "L"], 3, jnp.zeros((3, 2)))
        assert not [k for k, v in trace.items() if v["type"] == "param"]
        assert "theta_ddG_offset" not in trace

    def test_prior_mean_recorded(self):
        M, S = 3, 2
        nn_means = jnp.ones((M, S)) * 2.0
        _, trace = _trace_sample_ddG("theta", ["H", "L"], M, nn_means)
        assert trace["theta_ddG_prior_mean"]["type"] == "deterministic"
        np.testing.assert_allclose(trace["theta_ddG_prior_mean"]["value"],
                                   nn_means)

    def test_zero_sigma_gives_prior_means(self):
        """σ_s conditioned to 0 makes ΔΔG the prior means exactly."""
        M, S = 4, 2
        nn_means = jnp.ones((M, S)) * 3.0

        def model():
            return sample_ddG("theta", ["H", "L"], M, nn_means)

        conditioned = numpyro.handlers.condition(
            model, data={"theta_ddG_sigma": jnp.zeros(S)})
        with numpyro.handlers.seed(rng_seed=0):
            out = conditioned()
        np.testing.assert_array_equal(np.asarray(out), np.asarray(nn_means))

    def test_conditioned_sigma_sets_the_spread(self):
        M, S = 4000, 2
        sigma = jnp.array([0.3, 2.0])

        def model():
            return sample_ddG("theta", ["H", "L"], M, jnp.zeros((M, S)))

        conditioned = numpyro.handlers.condition(
            model, data={"theta_ddG_sigma": sigma})
        with numpyro.handlers.seed(rng_seed=0):
            out = np.asarray(conditioned())
        np.testing.assert_allclose(out.std(axis=0), sigma, rtol=0.05)

    def test_log_density_pins_sigma_at_rms_residual(self):
        """For fixed ΔΔG the joint density peaks near σ_s = RMS(ΔΔG − mean)."""
        M = 500
        rng = np.random.default_rng(0)
        means = jnp.asarray(rng.normal(0, 2, (M, 1)), dtype=jnp.float32)
        ddG = means + jnp.asarray(rng.normal(0, 0.7, (M, 1)), dtype=jnp.float32)
        rms = float(jnp.sqrt(jnp.mean((ddG - means) ** 2)))

        def model():
            sample_ddG("theta", ["H"], M, means)

        grid = np.linspace(0.3, 2.0, 171)
        dens = [float(numpyro.infer.util.log_density(
                    model, (), {}, {"theta_ddG": ddG,
                                    "theta_ddG_sigma": jnp.array([s])})[0])
                for s in grid]
        assert grid[int(np.argmax(dens))] == pytest.approx(rms, abs=0.02)

    def test_nn_means_shift_output(self):
        """Non-zero nn_means should shift the mean of sampled ΔΔG values."""
        M, S = 100, 1

        def run(nn_m):
            def model():
                return sample_ddG("theta", ["H"], M, nn_m)
            with numpyro.handlers.seed(rng_seed=1):
                return model()

        out_zero = np.asarray(run(jnp.zeros((M, S))))
        out_pos  = np.asarray(run(jnp.full((M, S), 5.0)))
        assert out_pos.mean() > out_zero.mean()

    def test_wt_only_library_skips_ddG_site(self):
        out, trace = _trace_sample_ddG("theta", ["H", "L"], 0, jnp.zeros((0, 2)))
        assert out.shape == (0, 2)
        assert "theta_ddG" not in trace
        assert "theta_ddG_sigma" in trace

    def test_default_prior_spans_a_tenth_to_ten_kT(self):
        lo = np.exp(DEFAULT_DDG_SIGMA_LOC - 1.96 * DEFAULT_DDG_SIGMA_SCALE)
        hi = np.exp(DEFAULT_DDG_SIGMA_LOC + 1.96 * DEFAULT_DDG_SIGMA_SCALE)
        assert 0.08 < lo < 0.12
        assert 8.0 < hi < 12.0


class TestGuideDdG:
    def _traces(self, M=3, S=2):
        means = jnp.arange(M * S, dtype=jnp.float32).reshape(M, S)

        def model():
            return sample_ddG("theta", ["H", "L"], M, means)

        def guide():
            return guide_ddG("theta", M, S, means)

        with numpyro.handlers.seed(rng_seed=0):
            mtr = numpyro.handlers.trace(model).get_trace()
        with numpyro.handlers.seed(rng_seed=0):
            gtr = numpyro.handlers.trace(guide).get_trace()
        return means, mtr, gtr

    def test_sample_sites_match_model(self):
        _, mtr, gtr = self._traces()
        ms = {k: v["value"].shape for k, v in mtr.items() if v["type"] == "sample"}
        gs = {k: v["value"].shape for k, v in gtr.items() if v["type"] == "sample"}
        assert ms == gs

    def test_params_follow_loc_scale_convention(self):
        means, _, gtr = self._traces()
        params = {k: v["value"] for k, v in gtr.items() if v["type"] == "param"}
        assert set(params) == {"theta_ddG_locs", "theta_ddG_scales",
                               "theta_ddG_sigma_locs", "theta_ddG_sigma_scales"}
        np.testing.assert_allclose(params["theta_ddG_locs"], means)
        np.testing.assert_allclose(params["theta_ddG_sigma_locs"], 0.0)

    def test_wt_only_library(self):
        def guide():
            return guide_ddG("theta", 0, 2, jnp.zeros((0, 2)))

        with numpyro.handlers.seed(rng_seed=0):
            gtr = numpyro.handlers.trace(guide).get_trace()
        assert "theta_ddG" not in gtr
        assert "theta_ddG_sigma" in gtr

    def test_elbo_is_finite(self):
        M, S = 3, 2
        means = jnp.zeros((M, S))

        def model():
            sample_ddG("theta", ["H", "L"], M, means)

        def guide():
            guide_ddG("theta", M, S, means)

        svi = numpyro.infer.SVI(model, guide, numpyro.optim.Adam(1e-3),
                                numpyro.infer.Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(0))
        _, loss = svi.update(state)
        assert np.isfinite(float(loss))


def test_guesses_start_at_prior_means_and_one_kT():
    means = np.arange(6, dtype=np.float32).reshape(3, 2)
    g = ddG_guesses("theta", means)
    np.testing.assert_allclose(g["theta_ddG"], means)
    np.testing.assert_allclose(g["theta_ddG_sigma"], [1.0, 1.0])
