"""
The prior SDs behind the Laplace eigenvalue floor
(``RunInference._laplace_prior_precision``) and the structure traces that
read a model's sites.

On the PddG pilot fits (2026-10-10) the prior SDs were taken from a trace
whose latent sites were drawn from the prior. The draws (ln K up to +/-60)
predicted a non-finite ln_cfu, the count likelihood refused to be built
("CountNegativeBinomial ... invalid log_mu"), and the floor fell back to a
flat 1e-3 without saying why it mattered: 16-38 negative eigenvalues became
directions with SD ~30. The same refusal stopped tfs-summarize-fit's
trajectory plots.
"""
import dataclasses
import os
import warnings

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from flax import struct
from numpyro import handlers

from tfscreen.tfmodel.generative.observe.growth_counts import (
    count_distribution,
)
from tfscreen.tfmodel.inference import run_inference as ri_module
from tfscreen.tfmodel.inference.initialization import (
    structure_trace,
    trace_model_sites,
)
from tfscreen.tfmodel.inference.run_inference import (
    LAPLACE_MIN_EIGENVALUE,
    RunInference,
)

_PRIOR_SD = 1000.0


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# A model whose prior draws break its count observation
# ---------------------------------------------------------------------------

@struct.dataclass
class _Data:
    batch_idx: jnp.ndarray
    counts: jnp.ndarray
    num_genotype: int = struct.field(pytree_node=False, default=4)


def _wide_count_model(data, priors):
    """
    A shared ln K with a very wide prior and one effect per genotype. The
    predicted log mean read count is exp(|ln K|) + effect, so any prior draw
    with |ln K| above ~89 overflows float32 and the count distribution refuses
    the non-finite log_mu, as the PddG fits' ln_cfu did. At the MAP (ln K
    near 0) everything is finite.
    """
    ln_K = numpyro.sample("ln_K", dist.Normal(0.0, _PRIOR_SD))
    with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
        effect = numpyro.sample("geno_p", dist.Normal(0.0, 2.0))
        log_mu = jnp.exp(jnp.abs(ln_K)) + effect
        numpyro.sample("growth_obs", count_distribution(log_mu, 5.0, 0.01),
                       obs=data.counts)


class _WideCountModel:
    def __init__(self, num_genotype=4):
        self.data = _Data(num_genotype=num_genotype,
                          batch_idx=jnp.arange(num_genotype),
                          counts=jnp.full(num_genotype, 3.0))
        self.priors = {}
        self.jax_model = _wide_count_model
        self.jax_model_guide = lambda data, priors: None

    def get_batch(self, data, indices):
        return _Data(num_genotype=len(indices), batch_idx=jnp.asarray(indices),
                     counts=data.counts[jnp.asarray(indices)])

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def _map_params(num_genotype=4):
    """A MAP point (unconstrained, AutoDelta names) where log_mu is finite."""
    return {"ln_K_auto_loc": jnp.array(0.3),
            "geno_p_auto_loc": jnp.linspace(-0.5, 0.5, num_genotype)}


def _prior_draw_breaks_observation(model):
    """The precondition: a validated trace with prior draws fails."""
    fn = handlers.seed(model.jax_model, 0)
    try:
        handlers.trace(fn).get_trace(data=model.data, priors=model.priors)
    except ValueError as err:
        return "invalid log_mu" in str(err)
    return False


def test_prior_draw_breaks_the_count_observation():
    assert _prior_draw_breaks_observation(_WideCountModel())


def test_structure_trace_ignores_invalid_observation():
    model = _WideCountModel()
    tr = structure_trace(model.jax_model, data=model.data,
                         priors=model.priors)
    assert {"ln_K", "geno_p", "growth_obs"} <= set(tr)
    sites = trace_model_sites(model.jax_model, model.priors, model.data)
    assert set(sites) == {"ln_K", "geno_p"}


def test_trace_model_sites_substitutes_unconstrained_values():
    model = _WideCountModel()
    unc = {"ln_K": jnp.array(0.3), "geno_p": jnp.arange(4.0)}
    sites = trace_model_sites(model.jax_model, model.priors, model.data,
                              unconstrained=unc)
    np.testing.assert_allclose(sites["ln_K"]["value"], 0.3)
    np.testing.assert_allclose(sites["geno_p"]["value"], np.arange(4.0))


def test_laplace_prior_precision_uses_prior_sds_not_flat_floor():
    model = _WideCountModel()
    ri = RunInference(model, seed=0)
    mp = _map_params()
    unc = {k[:-len("_auto_loc")]: v for k, v in mp.items()}
    prec = ri._laplace_prior_precision(
        unc, {"priors": model.priors, "data": model.data}, 5)
    # ravel order: geno_p (4), ln_K (1)
    np.testing.assert_allclose(prec[:4], 1 / 2.0 ** 2, rtol=1e-6)
    np.testing.assert_allclose(prec[4], 1 / _PRIOR_SD ** 2, rtol=1e-6)


def test_get_laplace_posteriors_floors_at_prior_on_wide_count_model(tmp_path,
                                                                    capsys):
    model = _WideCountModel()
    ri = RunInference(model, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ri.get_laplace_posteriors(_map_params(), str(tmp_path / "lap"),
                                  num_posterior_samples=20,
                                  sampling_batch_size=10)
    out = capsys.readouterr().out
    assert "Could not compute prior SDs" not in out
    assert os.path.exists(tmp_path / "lap_posterior.h5")


def test_prior_sd_failure_is_an_error_unless_allowed(tmp_path, monkeypatch,
                                                     capsys):
    model = _WideCountModel()

    def broken(*args, **kwargs):
        raise ValueError("broken prior")

    monkeypatch.setattr(ri_module, "site_unconstrained_prior_sds", broken)
    ri = RunInference(model, seed=0)
    with pytest.raises(RuntimeError, match="broken prior"):
        ri.get_laplace_posteriors(_map_params(), str(tmp_path / "lap"),
                                  num_posterior_samples=10,
                                  sampling_batch_size=10)
    assert not os.path.exists(tmp_path / "lap_posterior.h5")

    ri = RunInference(model, seed=0)
    with pytest.warns(RuntimeWarning, match="broken prior"):
        ri.get_laplace_posteriors(_map_params(), str(tmp_path / "lap"),
                                  num_posterior_samples=10,
                                  sampling_batch_size=10,
                                  allow_flat_floor=True)
    assert "WARNING" in capsys.readouterr().out
    assert os.path.exists(tmp_path / "lap_posterior.h5")


def test_elements_without_prior_sd_are_named(monkeypatch):
    model = _WideCountModel()
    real = ri_module.site_unconstrained_prior_sds

    def drop_ln_K(sites, *args, **kwargs):
        out = real(sites, *args, **kwargs)
        out.pop("ln_K")
        return out

    monkeypatch.setattr(ri_module, "site_unconstrained_prior_sds", drop_ln_K)
    ri = RunInference(model, seed=0)
    unc = {k[:-len("_auto_loc")]: v for k, v in _map_params().items()}
    with pytest.warns(RuntimeWarning, match=r"1 of 5 MAP elements.*ln_K \(1\)"):
        prec = ri._laplace_prior_precision(
            unc, {"priors": model.priors, "data": model.data}, 5)
    assert prec[4] == LAPLACE_MIN_EIGENVALUE


# ---------------------------------------------------------------------------
# A count-likelihood thermodynamic model with extreme prior draws
# ---------------------------------------------------------------------------

_WT_SD = 10.0

_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "smoke-tests", "test_data")


@pytest.fixture(scope="module")
def wide_thermo_counts(tmp_path_factory):
    """
    The smoke screen fit with counts and thermo.O2_C4_K3_U0_a.PddG, with the
    wt ln K priors moved to 100 +/- 10 so that prior draws of ln K are
    extreme (the pilot's draws, through the learned sigma_s, reached +/-60).
    Linear-space theta overflowed there (K^2 > float32 max above ln K 44)
    and gave NaN ln_cfu.
    """
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

    growth = os.path.join(_SMOKE, "growth-smoke.csv")
    df = pd.read_csv(growth)
    muts = sorted({m for g in df["genotype"].unique() if g != "wt"
                   for m in g.split("/")})
    ddG = tmp_path_factory.mktemp("ddG") / "ddG.csv"
    pd.DataFrame({"mut": muts, "H": 0.5, "HD": -0.5, "L": 0.2,
                  "LE2": 0.1}).to_csv(ddG, index=False)
    o = ModelOrchestrator(growth_df=growth,
                          binding_df=os.path.join(_SMOKE, "binding-smoke.csv"),
                          batch_size=6, growth_likelihood="counts",
                          sample_offset="level",
                          theta="thermo.O2_C4_K3_U0_a.PddG",
                          thermo_data=str(ddG))
    theta = dataclasses.replace(o.priors.theta,
                                theta_ln_K_op_wt_loc=100.0,
                                theta_ln_K_op_wt_scale=_WT_SD,
                                theta_ln_K_HL_wt_loc=100.0,
                                theta_ln_K_HL_wt_scale=_WT_SD,
                                theta_ln_K_E_wt_loc=100.0,
                                theta_ln_K_E_wt_scale=_WT_SD)
    o._priors = dataclasses.replace(o.priors, theta=theta)
    return o


def test_thermo_counts_prior_draw_was_extreme(wide_thermo_counts):
    o = wide_thermo_counts
    tr = structure_trace(o.jax_model, data=o.data, priors=o.priors)
    for name in ("theta_ln_K_op_wt", "theta_ln_K_HL_wt", "theta_ln_K_E_wt"):
        assert float(np.ravel(tr[name]["value"])[0]) > 60


def test_thermo_counts_laplace_prior_precision(wide_thermo_counts):
    o = wide_thermo_counts
    ri = RunInference(o, seed=0)
    svi = ri.setup_svi(guide_type="delta")
    state = svi.init(ri.get_key(), data=o.data, priors=o.priors)
    mp = svi.optim.get_params(state.optim_state)
    unc = {k[:-len("_auto_loc")]: v for k, v in mp.items()
           if k.endswith("_auto_loc")}
    D = sum(int(np.size(v)) for v in unc.values())
    full = ri._unscaled_batch(o.data, jnp.arange(o.data.num_genotype))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prec = ri._laplace_prior_precision(unc, {"priors": o.priors,
                                                 "data": full}, D)
    assert prec.shape == (D,)
    assert np.all(np.isfinite(prec)) and np.all(prec > 0)
    # the moved wt priors give their sites prior precision 1/10^2
    for name in ("theta_ln_K_op_wt", "theta_ln_K_HL_wt", "theta_ln_K_E_wt"):
        assert name in unc
    assert np.sum(np.isclose(prec, 1 / _WT_SD ** 2, rtol=1e-4)) >= 3


def test_prediction_traces_survive_extreme_prior_draws(wide_thermo_counts):
    """
    predict() reads the model's structure from traces with prior draws
    (tfs-summarize-fit's trajectory plots go through it); they must not need
    the count observation to accept those draws.
    """
    from tfscreen.tfmodel.analysis.prediction import predict

    o = wide_thermo_counts
    ri = RunInference(o, seed=0)
    svi = ri.setup_svi(guide_type="delta")
    state = svi.init(ri.get_key(), data=o.data, priors=o.priors)
    params = {k: np.asarray(v) for k, v in svi.get_params(state).items()}
    out = predict(o, params, predict_sites=["growth_pred"], num_samples=None)
    df = out if isinstance(out, pd.DataFrame) else next(iter(out.values()))
    assert len(df) > 0
