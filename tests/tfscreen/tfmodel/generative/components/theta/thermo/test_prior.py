"""Tests for thermo/prior.py: the structural prior on Δln K."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    DEFAULT_DDG_SIGMA_LOC,
    DEFAULT_DDG_SIGMA_SCALE,
    constant_scale_tril,
    ddG_per_state_identified,
    dlnK_guesses,
    guide_dlnK,
    identifiable_sigma,
    per_state_truth,
    posthoc_ddG,
    projection,
    sample_dlnK,
)

P_C4_U0 = projection(mwc.PROJ_C4)
P_C4_U1 = projection(mwc.PROJ_C4, mwc.PROJ_U_C4)
P_C12_U0 = projection(mwc.PROJ_C12)
P_C12_U1 = projection(mwc.PROJ_C12, mwc.PROJ_U_C12)
FAMILIES = {"C4_U0": P_C4_U0, "C4_U1": P_C4_U1,
            "C12_U0": P_C12_U0, "C12_U1": P_C12_U1}
STATES = {4: ["H", "HD", "L", "LE2"], 6: list(mwc.STRUCTURE_NAMES_C12)}


def _trace(proj, num_mut, means, seed=42, **kwargs):
    """Run sample_dlnK inside a seeded trace; return (output, trace)."""
    S = proj.shape[1]

    def model():
        return sample_dlnK("theta", STATES[S], num_mut, means, proj, **kwargs)

    with numpyro.handlers.seed(rng_seed=seed):
        trace = numpyro.handlers.trace(model).get_trace()
    with numpyro.handlers.seed(rng_seed=seed):
        out = model()
    return out, trace


class TestProjection:
    def test_shapes_and_unfolding_row_last(self):
        assert P_C4_U0.shape == (3, 4)
        assert P_C4_U1.shape == (4, 4)
        assert P_C12_U0.shape == (5, 6)
        assert P_C12_U1.shape == (6, 6)
        np.testing.assert_array_equal(P_C4_U1[-1], mwc.PROJ_U_C4)
        np.testing.assert_array_equal(P_C12_U1[:5], mwc.PROJ_C12)

    def test_u0_has_the_common_shift_null_vector(self):
        for P in (P_C4_U0, P_C12_U0):
            np.testing.assert_allclose(P @ np.ones(P.shape[1]), 0.0)

    @pytest.mark.parametrize("fam", list(FAMILIES))
    def test_per_state_identified_only_for_u1(self, fam):
        assert ddG_per_state_identified(FAMILIES[fam]) == fam.endswith("U1")

    @pytest.mark.parametrize("fam", list(FAMILIES))
    def test_every_sigma_identifiable_from_the_covariance(self, fam):
        assert identifiable_sigma(FAMILIES[fam])

    def test_identifiable_sigma_detects_a_tied_pair(self):
        # Two states that enter every constant identically cannot be told
        # apart through the covariance.
        P = np.array([[1.0, 1.0, -1.0]])
        assert not identifiable_sigma(P)


class TestScaleTril:
    @pytest.mark.parametrize("fam", list(FAMILIES))
    def test_matches_the_induced_covariance(self, fam):
        P = FAMILIES[fam]
        sigma = jnp.linspace(0.2, 3.0, P.shape[1])
        L = np.asarray(constant_scale_tril(jnp.asarray(P, jnp.float32), sigma))
        cov = P @ np.diag(np.asarray(sigma) ** 2) @ P.T
        np.testing.assert_allclose(L @ L.T, cov, atol=1e-5)
        np.testing.assert_allclose(L, np.tril(L))
        assert np.all(np.diag(L) > 0)

    def test_one_sigma_at_zero_keeps_u0_covariance_positive_definite(self):
        for P in (P_C4_U0, P_C12_U0):
            for s in range(P.shape[1]):
                sigma = np.ones(P.shape[1])
                sigma[s] = 0.0
                L = np.asarray(constant_scale_tril(
                    jnp.asarray(P, jnp.float32), jnp.asarray(sigma)))
                assert np.all(np.abs(np.diag(L)) > 1e-3)

    def test_gradient_matches_slogdet(self):
        P = jnp.asarray(P_C12_U0, jnp.float32)
        sigma = jnp.linspace(0.3, 2.0, 6)
        g1 = jax.grad(lambda s: jnp.sum(jnp.log(jnp.diagonal(
            constant_scale_tril(P, s)))))(sigma)
        g2 = jax.grad(lambda s: 0.5 * jnp.linalg.slogdet(
            P @ jnp.diag(s ** 2) @ P.T)[1])(sigma)
        np.testing.assert_allclose(g1, g2, rtol=1e-4)


class TestSampleDlnK:
    def test_sites_and_shapes(self):
        M = 5
        means = jnp.arange(M * 4, dtype=jnp.float32).reshape(M, 4)
        out, trace = _trace(P_C4_U0, M, means)
        assert out.shape == (M, 3)
        assert trace["theta_dlnK"]["type"] == "sample"
        assert trace["theta_dlnK"]["value"].shape == (M, 3)
        assert trace["theta_ddG_sigma"]["type"] == "sample"
        assert trace["theta_ddG_sigma"]["value"].shape == (4,)
        assert trace["theta_ddG"]["type"] == "deterministic"
        assert trace["theta_ddG"]["value"].shape == (M, 4)
        assert trace["theta_ddG_prior_mean"]["type"] == "deterministic"
        np.testing.assert_allclose(trace["theta_ddG_prior_mean"]["value"], means)
        np.testing.assert_allclose(out, trace["theta_dlnK"]["value"])

    def test_dlnK_is_mvn_at_projected_means_with_induced_covariance(self):
        M = 4
        means = jnp.arange(M * 4, dtype=jnp.float32).reshape(M, 4)
        _, trace = _trace(P_C4_U1, M, means)
        fn = trace["theta_dlnK"]["fn"]
        while not isinstance(fn, dist.MultivariateNormal):
            fn = fn.base_dist
        np.testing.assert_allclose(np.broadcast_to(fn.loc, (M, 4)),
                                   np.asarray(means) @ P_C4_U1.T, rtol=1e-6)
        sigma = np.asarray(trace["theta_ddG_sigma"]["value"])
        np.testing.assert_allclose(np.asarray(fn.covariance_matrix).reshape(-1, 4, 4)[0],
                                   P_C4_U1 @ np.diag(sigma ** 2) @ P_C4_U1.T,
                                   rtol=1e-4, atol=1e-5)
        assert trace["theta_dlnK"]["fn"].batch_shape == (M,)
        assert trace["theta_dlnK"]["fn"].event_shape == (4,)

    def test_sigma_is_a_lognormal_sample_site_per_state(self):
        _, trace = _trace(P_C4_U0, 3, jnp.zeros((3, 4)),
                          sigma_loc=0.5, sigma_scale=0.7)
        fn = trace["theta_ddG_sigma"]["fn"]
        while not isinstance(fn, dist.LogNormal):
            fn = fn.base_dist
        np.testing.assert_allclose(fn.loc, 0.5)
        np.testing.assert_allclose(fn.scale, 0.7)

    def test_no_param_sites_and_no_per_state_latent(self):
        _, trace = _trace(P_C4_U0, 3, jnp.zeros((3, 4)))
        assert not [k for k, v in trace.items() if v["type"] == "param"]
        samples = {k for k, v in trace.items() if v["type"] == "sample"}
        assert samples == {"theta_ddG_sigma", "theta_dlnK"}

    def test_conditioned_sigma_sets_the_constant_covariance(self):
        M = 20000
        sigma = jnp.array([0.3, 1.0, 2.0, 0.5])

        def model():
            return sample_dlnK("theta", STATES[4], M, jnp.zeros((M, 4)),
                               P_C4_U0)

        with numpyro.handlers.seed(rng_seed=0):
            out = np.asarray(numpyro.handlers.condition(
                model, data={"theta_ddG_sigma": sigma})())
        cov = P_C4_U0 @ np.diag(np.asarray(sigma) ** 2) @ P_C4_U0.T
        np.testing.assert_allclose(np.cov(out.T), cov, atol=0.08)

    def test_wt_only_library_skips_the_dlnK_site(self):
        out, trace = _trace(P_C4_U1, 0, jnp.zeros((0, 4)))
        assert out.shape == (0, 4)
        assert "theta_dlnK" not in trace
        assert "theta_ddG_sigma" in trace
        assert trace["theta_ddG"]["value"].shape == (0, 4)

    def test_default_prior_spans_a_tenth_to_ten_kT(self):
        lo = np.exp(DEFAULT_DDG_SIGMA_LOC - 1.96 * DEFAULT_DDG_SIGMA_SCALE)
        hi = np.exp(DEFAULT_DDG_SIGMA_LOC + 1.96 * DEFAULT_DDG_SIGMA_SCALE)
        assert 0.08 < lo < 0.12
        assert 8.0 < hi < 12.0

    def test_log_density_is_finite_and_bounded_as_one_u0_sigma_shrinks(self):
        """The old per-state form was unbounded here; the constant form is not."""
        M = 200
        rng = np.random.default_rng(0)
        means = rng.normal(0, 2, (M, 4)).astype(np.float32)
        truth = means + rng.normal(0, 0.3, (M, 4)).astype(np.float32)
        dlnK = jnp.asarray(truth @ P_C4_U0.T)

        def model():
            sample_dlnK("theta", STATES[4], M, jnp.asarray(means), P_C4_U0)

        def logp(s_last):
            sigma = jnp.array([0.3, 0.3, 0.3, s_last])
            return float(numpyro.infer.util.log_density(
                model, (), {}, {"theta_dlnK": dlnK,
                                "theta_ddG_sigma": sigma})[0])

        vals = [logp(s) for s in (0.3, 1e-2, 1e-4, 1e-6)]
        assert np.all(np.isfinite(vals))
        assert vals[0] > max(vals[1:])


class TestPosthocDdG:
    @pytest.mark.parametrize("fam", ["C4_U1", "C12_U1"])
    def test_u1_exact_inversion(self, fam):
        P = FAMILIES[fam]
        rng = np.random.default_rng(1)
        ddG = rng.normal(0, 1, (7, P.shape[1])).astype(np.float32)
        out = posthoc_ddG(jnp.asarray(ddG @ P.T),
                          jnp.asarray(rng.normal(0, 1, ddG.shape)),
                          jnp.ones(P.shape[1]), P)
        np.testing.assert_allclose(out, ddG, atol=1e-5)

    @pytest.mark.parametrize("fam", ["C4_U0", "C12_U0"])
    def test_u0_conditional_mean_formula(self, fam):
        P = FAMILIES[fam]
        S = P.shape[1]
        rng = np.random.default_rng(2)
        mu = rng.normal(0, 1, (6, S))
        sigma = np.linspace(0.3, 2.0, S)
        dlnK = rng.normal(0, 1, (6, P.shape[0]))
        Sig = np.diag(sigma ** 2)
        expect = mu + (Sig @ P.T @ np.linalg.solve(P @ Sig @ P.T,
                                                   (dlnK - mu @ P.T).T)).T
        out = posthoc_ddG(jnp.asarray(dlnK, jnp.float32),
                          jnp.asarray(mu, jnp.float32),
                          jnp.asarray(sigma, jnp.float32), P)
        np.testing.assert_allclose(out, expect, atol=1e-4)
        # It reproduces the constants exactly.
        np.testing.assert_allclose(np.asarray(out) @ P.T, dlnK, atol=1e-4)

    @pytest.mark.parametrize("fam", list(FAMILIES))
    def test_constants_at_the_projected_table_give_the_table(self, fam):
        """σ = 0 truth (Δln K = P·μ) gives the table back, in every family."""
        P = FAMILIES[fam]
        mu = np.random.default_rng(3).normal(0, 1, (5, P.shape[1]))
        out = posthoc_ddG(jnp.asarray(mu @ P.T, jnp.float32),
                          jnp.asarray(mu, jnp.float32),
                          jnp.linspace(0.2, 2.0, P.shape[1]), P)
        np.testing.assert_allclose(out, mu, atol=1e-5)

    def test_u0_common_mode_comes_from_the_prior(self):
        """Shifting every state's prior mean shifts the U0 post-hoc ΔΔG."""
        P = P_C4_U0
        rng = np.random.default_rng(4)
        mu = rng.normal(0, 1, (3, 4))
        dlnK = rng.normal(0, 1, (3, 3))
        sigma = jnp.ones(4)
        a = np.asarray(posthoc_ddG(jnp.asarray(dlnK), jnp.asarray(mu), sigma, P))
        b = np.asarray(posthoc_ddG(jnp.asarray(dlnK), jnp.asarray(mu + 2.0),
                                   sigma, P))
        np.testing.assert_allclose(b - a, 2.0, atol=1e-5)

    def test_site_records_posthoc_value(self):
        M = 6
        means = jnp.asarray(np.random.default_rng(5).normal(0, 1, (M, 4)),
                            jnp.float32)
        _, trace = _trace(P_C4_U0, M, means)
        expect = posthoc_ddG(trace["theta_dlnK"]["value"], means,
                             trace["theta_ddG_sigma"]["value"], P_C4_U0)
        np.testing.assert_allclose(trace["theta_ddG"]["value"], expect,
                                   rtol=1e-5, atol=1e-6)


class TestSigmaIdentifiability:
    """σ_s comes back from Δln K alone (no data noise), in every family."""

    @pytest.mark.parametrize("fam", list(FAMILIES))
    def test_sigma_recovered_from_noise_free_constants(self, fam):
        P = FAMILIES[fam]
        S = P.shape[1]
        M = 4000
        rng = np.random.default_rng(6)
        sigma_true = np.linspace(0.3, 1.5, S)[rng.permutation(S)]
        means = rng.normal(0, 2, (M, S)).astype(np.float32)
        _, dlnK = per_state_truth(means, sigma_true, P, jax.random.PRNGKey(0))

        def model():
            sample_dlnK("theta", STATES[S], M, jnp.asarray(means), P)

        def neg_logp(log_sigma):
            return -numpyro.infer.util.log_density(
                model, (), {}, {"theta_dlnK": jnp.asarray(dlnK),
                                "theta_ddG_sigma": jnp.exp(log_sigma)})[0] / M

        x = jnp.zeros(S)
        step = jax.jit(jax.value_and_grad(neg_logp))
        for _ in range(4000):
            _, g = step(x)
            x = x - 0.5 * g
        np.testing.assert_allclose(np.exp(np.asarray(x)), sigma_true, rtol=0.1)

    def test_c4_u0_covariance_entries_give_sigma_h_and_l(self):
        P = P_C4_U0
        sigma = np.array([0.7, 1.1, 0.4, 1.3])
        cov = P @ np.diag(sigma ** 2) @ P.T
        assert cov[0, 1] == pytest.approx(sigma[0] ** 2)        # σ_H²
        assert -2 * cov[1, 2] == pytest.approx(sigma[2] ** 2)   # σ_L²


class TestGuideDlnK:
    def _traces(self, P=P_C4_U1, M=3):
        S = P.shape[1]
        means = jnp.arange(M * S, dtype=jnp.float32).reshape(M, S)

        def model():
            return sample_dlnK("theta", STATES[S], M, means, P)

        def guide():
            return guide_dlnK("theta", M, S, P, means)

        with numpyro.handlers.seed(rng_seed=0):
            mtr = numpyro.handlers.trace(model).get_trace()
        with numpyro.handlers.seed(rng_seed=0):
            gtr = numpyro.handlers.trace(guide).get_trace()
        return means, mtr, gtr

    @pytest.mark.parametrize("fam", list(FAMILIES))
    def test_sample_sites_match_model(self, fam):
        _, mtr, gtr = self._traces(FAMILIES[fam])

        def sites(tr):
            return {k: (v["value"].shape, v["fn"].event_shape)
                    for k, v in tr.items() if v["type"] == "sample"}
        assert sites(mtr) == sites(gtr)

    def test_params_follow_loc_scale_convention(self):
        means, _, gtr = self._traces()
        params = {k: v["value"] for k, v in gtr.items() if v["type"] == "param"}
        assert set(params) == {"theta_dlnK_locs", "theta_dlnK_scales",
                               "theta_ddG_sigma_locs", "theta_ddG_sigma_scales"}
        np.testing.assert_allclose(params["theta_dlnK_locs"],
                                   np.asarray(means) @ P_C4_U1.T)
        np.testing.assert_allclose(params["theta_ddG_sigma_locs"], 0.0)

    def test_wt_only_library(self):
        def guide():
            return guide_dlnK("theta", 0, 4, P_C4_U0, jnp.zeros((0, 4)))

        with numpyro.handlers.seed(rng_seed=0):
            gtr = numpyro.handlers.trace(guide).get_trace()
        assert "theta_dlnK" not in gtr
        assert "theta_ddG_sigma" in gtr

    def test_elbo_is_finite(self):
        M, S = 3, 4
        means = jnp.zeros((M, S))

        def model():
            sample_dlnK("theta", STATES[S], M, means, P_C4_U0)

        def guide():
            guide_dlnK("theta", M, S, P_C4_U0, means)

        svi = numpyro.infer.SVI(model, guide, numpyro.optim.Adam(1e-3),
                                numpyro.infer.Trace_ELBO())
        state = svi.init(jax.random.PRNGKey(0))
        _, loss = svi.update(state)
        assert np.isfinite(float(loss))


def test_guesses_start_at_projected_prior_means_and_one_kT():
    means = np.arange(8, dtype=np.float32).reshape(2, 4)
    g = dlnK_guesses("theta", means, P_C4_U0)
    np.testing.assert_allclose(g["theta_dlnK"], means @ P_C4_U0.T)
    np.testing.assert_allclose(g["theta_ddG_sigma"], np.ones(4))


class TestPerStateTruth:
    def test_zero_sigma_gives_the_table(self):
        means = np.random.default_rng(0).normal(0, 1, (5, 4))
        ddG, dlnK = per_state_truth(means, np.zeros(4), P_C4_U0,
                                    jax.random.PRNGKey(0))
        np.testing.assert_allclose(ddG, means, atol=1e-6)
        np.testing.assert_allclose(dlnK, means @ P_C4_U0.T, atol=1e-5)

    def test_spread_and_projection(self):
        M = 20000
        sigma = np.array([0.3, 1.0, 2.0, 0.5])
        ddG, dlnK = per_state_truth(np.zeros((M, 4)), sigma, P_C4_U0,
                                    jax.random.PRNGKey(1))
        np.testing.assert_allclose(ddG.std(axis=0), sigma, rtol=0.03)
        np.testing.assert_allclose(dlnK, ddG @ P_C4_U0.T, atol=1e-5)
