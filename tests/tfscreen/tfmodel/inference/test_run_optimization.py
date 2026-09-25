"""
RunInference.run_optimization end to end on a small real model: windows,
step-size cuts, convergence, checkpoints/resume, output files, and the
parameter normalizers for each guide family.
"""

import os

import dill
import numpy as np
import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from flax import struct
from numpyro import handlers

from tfscreen.tfmodel.inference.run_inference import RunInference

NUM_GENOTYPE = 12
NUM_OBS = 6


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@struct.dataclass
class ToyData:
    batch_idx: jnp.ndarray
    y: jnp.ndarray
    num_genotype: int = struct.field(pytree_node=False, default=NUM_GENOTYPE)


def toy_model(priors, data):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("toy_genotype_plate", data.num_genotype):
        theta = numpyro.sample("theta", dist.Normal(mu, 1.0))
    theta = theta[..., data.batch_idx]
    y = data.y[data.batch_idx]
    batch = data.batch_idx.shape[0]
    with handlers.scale(scale=data.num_genotype / batch):
        with numpyro.plate("toy_obs_plate", batch):
            numpyro.sample("y", dist.Normal(theta[:, None], sigma).to_event(1),
                           obs=y)


def toy_guide(priors, data):
    mu_loc = numpyro.param("mu_loc", 0.0)
    mu_scale = numpyro.param("mu_scale", 1.0,
                             constraint=dist.constraints.positive)
    numpyro.sample("mu", dist.Normal(mu_loc, mu_scale))
    sigma_loc = numpyro.param("sigma_loc", 0.0)
    sigma_scale = numpyro.param("sigma_scale", 1.0,
                                constraint=dist.constraints.positive)
    numpyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))
    theta_locs = numpyro.param("theta_locs", jnp.zeros(data.num_genotype))
    theta_scales = numpyro.param("theta_scales", jnp.ones(data.num_genotype),
                                 constraint=dist.constraints.positive)
    with numpyro.plate("toy_genotype_plate", data.num_genotype):
        numpyro.sample("theta", dist.Normal(theta_locs, theta_scales))


class ToyModel:
    """RunInference-compatible wrapper around toy_model/toy_guide."""

    def __init__(self, batch_size=None, seed=0):
        rng = np.random.default_rng(seed)
        theta = 3.0 + rng.normal(size=NUM_GENOTYPE)
        y = theta[:, None] + 0.5 * rng.normal(size=(NUM_GENOTYPE, NUM_OBS))
        self.data = ToyData(batch_idx=jnp.arange(NUM_GENOTYPE),
                            y=jnp.asarray(y))
        self.priors = {}
        self.jax_model = toy_model
        self.jax_model_guide = toy_guide
        self.init_params = {}
        self._batch = batch_size or NUM_GENOTYPE

    def get_batch(self, data, idx):
        return data.replace(batch_idx=jnp.asarray(idx))

    def get_random_idx(self, batch_key=None, num_batches=1):
        if batch_key is not None:
            self._rng = np.random.default_rng(batch_key)
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(0)
        idx = np.stack([self._rng.permutation(NUM_GENOTYPE)[:self._batch]
                        for _ in range(num_batches)])
        return idx[0] if num_batches == 1 else idx


def _optimizer_step_size(svi):
    """Adam's first step from a fresh state moves each parameter by exactly
    the step size, whatever the gradient."""
    params = {"x": jnp.zeros(1)}
    state = svi.optim.init(params)
    state = svi.optim.update({"x": jnp.array([0.3])}, state)
    return abs(float(svi.optim.get_params(state)["x"][0]))


def _fit(ri, svi, out_prefix="toy", **kwargs):
    opts = dict(convergence_window_steps=400, patience=2,
                max_num_epochs=40000, epoch_checkpoint_interval=None,
                checkpoint_interval=100)
    opts.update(kwargs)
    return ri.run_optimization(svi, out_prefix=out_prefix, **opts)


# -----------------------------------------------------------------------------
# Convergence and step-size cuts
# -----------------------------------------------------------------------------

def test_component_svi_converges_through_cuts():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    state, params, converged = _fit(ri, svi, final_step_size=1e-4)

    assert converged
    assert ri._monitor.num_cuts == 2
    assert ri._monitor.step_size == pytest.approx(1e-4)
    # the optimizer really runs at the final step size
    assert _optimizer_step_size(svi) == pytest.approx(1e-4, rel=1e-3)
    # sensible posterior: mu near the mean of the theta estimates
    assert float(params["mu_loc"]) == pytest.approx(
        float(np.mean(params["theta_locs"])), abs=0.5)

    # convergence.csv: a row per window, ending in "converged"
    with open("toy_convergence.csv") as f:
        rows = [line.strip().split(",") for line in f]
    header, rows = rows[0], rows[1:]
    assert header[0] == "step" and "decision" in header
    decisions = [r[header.index("decision")] for r in rows]
    assert decisions.count("cut") == 2
    assert decisions[-1] == "converged"

    with open("toy_losses.txt") as f:
        assert f.readline().strip() == "epoch,loss,step,step_size"


def test_minibatched_svi_converges():
    model = ToyModel(batch_size=4)
    ri = RunInference(model, seed=1)
    assert ri._iterations_per_epoch == 3
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _, _, converged = _fit(ri, svi, final_step_size=1e-4)
    assert converged


def test_map_converges_and_uses_prior_sds():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="delta")
    _, params, converged = _fit(ri, svi, final_step_size=1e-4)
    assert converged
    assert "mu_auto_loc" in params


def test_max_epochs_reports_not_converged(capsys):
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-3, guide_type="component")
    _, _, converged = _fit(ri, svi, final_step_size=1e-6, max_num_epochs=500)
    assert not converged
    assert "without converging" in capsys.readouterr().out


def test_no_final_step_size_keeps_step_size():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _, _, converged = _fit(ri, svi)
    assert converged
    assert ri._monitor.num_cuts == 0


def test_schedule_with_cuts_is_refused():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=optax.constant_schedule(1e-3),
                       guide_type="component")
    with pytest.raises(ValueError, match="constant step"):
        _fit(ri, svi, final_step_size=1e-6)


def test_schedule_without_cuts_runs():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=optax.constant_schedule(1e-2),
                       guide_type="component")
    _fit(ri, svi, max_num_epochs=1000)


# -----------------------------------------------------------------------------
# Resume, step counter, checkpoints
# -----------------------------------------------------------------------------

def test_resume_continues_stage_and_step():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _fit(ri, svi, final_step_size=1e-5, max_num_epochs=6000,
         checkpoint_interval=10)
    saved_step = ri._current_step
    saved_step_size = ri._monitor.step_size
    assert saved_step_size < 1e-2  # at least one cut happened

    with open("toy_checkpoint.pkl", "rb") as f:
        ckpt = dill.load(f)
    assert ckpt["step_size"] == pytest.approx(saved_step_size)
    assert ckpt["convergence"]["step_size"] == pytest.approx(saved_step_size)
    rows_before = sum(1 for _ in open("toy_convergence.csv"))

    ri2 = RunInference(ToyModel(), seed=0)
    svi2 = ri2.setup_svi(adam_step_size=1e-2, guide_type="component")
    _fit(ri2, svi2, svi_state="toy_checkpoint.pkl", final_step_size=1e-5,
         max_num_epochs=400)
    assert ri2._current_step == saved_step + 400
    assert _optimizer_step_size(svi2) <= saved_step_size * (1 + 1e-3)
    # resumed runs append to the convergence record
    assert sum(1 for _ in open("toy_convergence.csv")) > rows_before


def test_resume_old_checkpoint_without_convergence_state():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _fit(ri, svi, max_num_epochs=50)
    with open("toy_checkpoint.pkl", "rb") as f:
        ckpt = dill.load(f)
    del ckpt["convergence"], ckpt["step_size"]
    ckpt["loss_start"], ckpt["loss_best"] = 1.0, 0.5
    with open("old_checkpoint.pkl", "wb") as f:
        dill.dump(ckpt, f)

    ri2 = RunInference(ToyModel(), seed=0)
    svi2 = ri2.setup_svi(adam_step_size=1e-3, guide_type="component")
    _fit(ri2, svi2, svi_state="old_checkpoint.pkl", max_num_epochs=50)
    assert ri2._monitor.step_size == pytest.approx(1e-3)


def test_fresh_run_resets_step_counter():
    """A pre-MAP followed by SVI on the same RunInference: SVI's counters,
    loss file and epoch checkpoints start at zero."""
    model = ToyModel()
    ri = RunInference(model, seed=0)
    ri._current_step = 777
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _fit(ri, svi, max_num_epochs=100)
    assert ri._current_step == 100
    with open("toy_losses.txt") as f:
        lines = f.read().splitlines()
    assert lines[0] == "epoch,loss,step,step_size"
    assert int(lines[1].split(",")[2]) < 100


def test_invalid_checkpoint_path():
    ri = RunInference(ToyModel(), seed=0)
    svi = ri.setup_svi(guide_type="component")
    with pytest.raises(ValueError, match="is not valid"):
        _fit(ri, svi, svi_state="missing.pkl", max_num_epochs=1)


def test_nan_explosion():
    ri = RunInference(ToyModel(), seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    with pytest.raises(RuntimeError, match="model exploded"):
        _fit(ri, svi, init_params={"mu_loc": jnp.array(np.nan)},
             init_param_jitter=0.0, max_num_epochs=50)


def test_epoch_checkpoints(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    prefix = str(sub / "run")
    ri = RunInference(ToyModel(), seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _fit(ri, svi, out_prefix=prefix, max_num_epochs=100,
         epoch_checkpoint_interval=50)
    ckpt_dir = sub / "checkpoints"
    assert ckpt_dir.is_dir()
    assert not (tmp_path / "checkpoints").exists()
    names = sorted(os.listdir(ckpt_dir))
    assert "0000050_checkpoint.pkl" in names

    ri2 = RunInference(ToyModel(), seed=1)
    svi2 = ri2.setup_svi(adam_step_size=1e-2, guide_type="component")
    with pytest.raises(FileExistsError, match="checkpoint.pkl"):
        _fit(ri2, svi2, out_prefix=prefix, max_num_epochs=100,
             epoch_checkpoint_interval=50)


@pytest.mark.parametrize("interval", [None, 0])
def test_epoch_checkpoints_disabled(interval):
    ri = RunInference(ToyModel(), seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="component")
    _fit(ri, svi, max_num_epochs=60, epoch_checkpoint_interval=interval)
    assert not os.path.isdir("checkpoints")


# -----------------------------------------------------------------------------
# Parameter normalizers
# -----------------------------------------------------------------------------

def _specs(guide_type, **guide_kwargs):
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(guide_type=guide_type, guide_kwargs=guide_kwargs)
    idx = model.get_random_idx()
    state = svi.init(ri.get_key(), priors=model.priors,
                     data=model.get_batch(model.data, idx))
    unconstrained = svi.optim.get_params(state.optim_state)
    specs = ri._param_normalizer_specs(svi, state, unconstrained)
    return ri, svi, state, specs


def test_normalizers_component_guide():
    _, _, _, specs = _specs("component")
    assert specs["theta_locs"] == ("scale", "theta_scales")
    assert specs["mu_loc"] == ("scale", "mu_scale")
    # LogNormal location (log space) paired with its log-space scale
    assert specs["sigma_loc"] == ("scale", "sigma_scale")
    assert specs["theta_scales"] == ("unit",)


def test_normalizers_auto_normal():
    _, _, _, specs = _specs("auto_normal")
    assert specs["theta_auto_loc"] == ("scale", "theta_auto_scale")
    assert specs["theta_auto_scale"] == ("unit",)


def test_normalizers_delta_use_prior_sd():
    _, _, _, specs = _specs("delta")
    kind, sd = specs["mu_auto_loc"]
    assert kind == "prior"
    assert float(sd) == pytest.approx(10.0)
    assert specs["theta_auto_loc"][0] == "prior"
    # positive site: tracked in log units
    assert specs["sigma_auto_loc"] == ("unit",)


@pytest.mark.parametrize("guide_type,kwargs", [
    ("auto_diagonal_normal", {}),
    ("auto_multivariate_normal", {}),
    ("auto_low_rank_multivariate_normal", {"rank": 2}),
])
def test_normalizers_auto_continuous(guide_type, kwargs):
    ri, svi, state, specs = _specs(guide_type, **kwargs)
    assert specs["auto_loc"] == ("auto_continuous",)
    assert not any("scale_tril" in k or "cov_factor" in k for k in specs)
    norm = ri._normalizer(specs["auto_loc"], svi.get_params(state))
    assert norm.shape == (NUM_GENOTYPE + 2,)
    assert np.all(np.asarray(norm) > 0)


# -----------------------------------------------------------------------------
# Starting points
# -----------------------------------------------------------------------------

def test_site_values_and_component_guide_start():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    guesses = {"mu": 2.5, "theta": np.full(NUM_GENOTYPE, 1.5),
               "sigma_loc": np.log(0.7)}
    values = ri.site_values(guesses)
    assert float(values["mu"]) == pytest.approx(2.5)
    assert float(values["sigma"]) == pytest.approx(0.7)  # exp of guide loc
    assert values["theta"].shape == (NUM_GENOTYPE,)

    start = ri.component_guide_start(values, guesses=guesses, init_scale=0.1)
    assert float(start["mu_loc"]) == pytest.approx(2.5)
    assert float(start["sigma_loc"]) == pytest.approx(np.log(0.7))
    np.testing.assert_allclose(start["theta_locs"], 1.5)
    np.testing.assert_allclose(start["theta_scales"], 0.1)
    assert float(start["mu_scale"]) == pytest.approx(0.1)


def test_site_values_from_map_result():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    values = ri.site_values({"mu_auto_loc": 4.0, "mu": 1.0,
                             "unknown_auto_loc": 3.0})
    assert float(values["mu"]) == pytest.approx(4.0)  # MAP key wins
    assert "unknown" not in values


def test_delta_init_values_start_the_map():
    model = ToyModel()
    ri = RunInference(model, seed=0)
    svi = ri.setup_svi(guide_type="delta", init_values={"mu": 7.0})
    state = svi.init(ri.get_key(), priors=model.priors,
                     data=model.get_batch(model.data, model.get_random_idx()))
    params = svi.get_params(state)
    assert float(params["mu_auto_loc"]) == pytest.approx(7.0)
    # sites without a value fall back to the prior median (as AutoDelta
    # does), here given the substituted mu
    assert float(params["theta_auto_loc"][0]) == pytest.approx(7.0, abs=1.0)
