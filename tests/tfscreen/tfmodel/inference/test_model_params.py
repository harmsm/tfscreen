"""
The model's own learned ``pyro.param`` sites (thermo PnnC NN weights, the
PddG/PnnC trust scale sigma_s) must reach every post-fit run of the model.

Regression: the posterior paths kept only sample-site values, so the forward
passes ran the model with every model-side param at its init value (NN = 0,
sigma_s = 1). A ``--laplace point`` posterior's theta then differed from the
MAP's own theta.
"""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to

from tfscreen.tfmodel.inference.model_params import (
    MODEL_PARAMS_GROUP,
    hold_model_params,
    read_model_params,
    sample_keys,
)

STRUCTS = ("H", "HD", "L", "LE2")
GENOTYPES = ["wt", "M42I", "K84L", "M42I/K84L"]
SIGMA = "theta_ddG_sigma_s"
THETA_SITES = ("theta_ddG", "theta_ln_K_op", "theta_ln_K_HL", "theta_ln_K_E")


# ---------------------------------------------------------------------------
# Small PnnC and PddG problems
# ---------------------------------------------------------------------------

def _growth_csv(path):
    rng = np.random.default_rng(0)
    rows = []
    for geno in GENOTYPES:
        parts = geno.split("/") if geno != "wt" else []
        cols = {}
        for i in range(3):
            p = parts[i] if i < len(parts) else None
            cols[f"wt_aa_{i + 1}"] = p[0] if p else ""
            cols[f"resid_{i + 1}"] = int(p[1:-1]) if p else ""
            cols[f"mut_aa_{i + 1}"] = p[-1] if p else ""
        for rep in [1.0, 2.0]:
            for conc in [0.0, 100.0]:
                rows.append({
                    "sample": f"s-{geno}-{rep}-{conc}", "genotype": geno,
                    **cols, "num_muts": len(parts), "counts": 1000,
                    "replicate": rep, "condition_pre": "pre-cond",
                    "t_pre": 24.0, "condition_sel": "sel+cond", "t_sel": 48.0,
                    "titrant_name": "iptg", "titrant_conc": conc,
                    "barcode_i7": "AAAA", "barcode_i5": "TTTT", "seq_run": 1,
                    "od600": 0.3, "library": "lib", "obs_file": "dummy.csv",
                    "sample_cfu": 1e7, "sample_cfu_std": 1e5,
                    "sample_cfu_detectable": True, "adjusted_counts": 1001,
                    "frequency": 0.1, "cfu": 1e6, "cfu_var": 1e8,
                    "ln_cfu_var": 0.01,
                    "ln_cfu": 14.0 + 0.1 * rng.standard_normal(),
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _binding_csv(path):
    rows = [{"genotype": g, "titrant_name": "iptg", "titrant_conc": c,
             "theta_obs": 0.5, "theta_std": 0.05, "tx": 0.5, "pred": 0.5}
            for g in GENOTYPES for c in [1e-6, 1e-4]]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _ensemble_h5(path, resnums=(42, 84)):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("structure_names",
                          data=np.array(STRUCTS, dtype=h5py.string_dtype()))
        for i, name in enumerate(STRUCTS):
            rng = np.random.RandomState(i)
            grp = hf.create_group(name)
            dist_mat = np.abs(rng.randn(2, 2)).astype(np.float32)
            np.fill_diagonal(dist_mat, 0.0)
            grp.create_dataset("logP", data=rng.randn(2, 20).astype(np.float32))
            grp.create_dataset("residue_nums",
                               data=np.asarray(resnums, dtype=np.int32))
            grp.create_dataset("dist_matrix", data=dist_mat)
            grp.create_dataset("n_chains_bearing_mut", data=np.int32(2))
    return str(path)


def _ddG_csv(path):
    pd.DataFrame([{"mut": "K84L", "H": 0.5, "HD": -0.3, "L": 0.2, "LE2": 1.0},
                  {"mut": "M42I", "H": -0.4, "HD": 0.8, "L": 0.1, "LE2": -0.6}]
                 ).to_csv(path, index=False)
    return str(path)


def _orchestrator(tmp_path, kind):
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    thermo_data = (_ensemble_h5(tmp_path / "ens.h5") if kind == "PnnC"
                   else _ddG_csv(tmp_path / "ddG.csv"))
    return ModelOrchestrator(_growth_csv(tmp_path / "growth.csv"),
                             _binding_csv(tmp_path / "binding.csv"),
                             theta=f"thermo.O2_C4_K3_U0_a.{kind}",
                             thermo_data=thermo_data)


@pytest.fixture(scope="module", params=["PnnC", "PddG"])
def fit(request, tmp_path_factory):
    """
    A short MAP of a small PnnC or PddG model, with the model-side params
    then moved well away from their init (sigma_s x 2.5; for PnnC, W2 and b2
    set to nonzero values), so a forward pass that drops them is visibly
    wrong.
    """
    from tfscreen.tfmodel.inference.run_inference import RunInference

    # Two models in one process can collide in jax's trace cache (their
    # data carry arrays as static pytree fields); start each one clean.
    jax.clear_caches()
    tmp = tmp_path_factory.mktemp(request.param)
    orch = _orchestrator(tmp, request.param)
    ri = RunInference(orch, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="delta")
    state, _, _ = ri.run_optimization(svi, max_num_epochs=3,
                                      out_prefix=str(tmp / "map"),
                                      checkpoint_interval=100)

    unconstrained = dict(svi.optim.get_params(state.optim_state))
    rng = np.random.default_rng(1)
    unconstrained[SIGMA] = unconstrained[SIGMA] + np.log(2.5)
    for k in list(unconstrained):
        if k.endswith("_W2") or k.endswith("_b2"):
            unconstrained[k] = jnp.asarray(
                0.5 * rng.standard_normal(np.shape(unconstrained[k])),
                dtype=jnp.float32)

    # The MAP's own theta: the model run at every fitted value, sample
    # sites and model-side params alike.
    pos = biject_to(constraints.positive)
    model_side = {k: (pos(v) if k == SIGMA else v)
                  for k, v in unconstrained.items()
                  if not k.endswith("_auto_loc")}
    sites = RunInference(orch, seed=0)._map_params_to_constrained(unconstrained)
    full = orch.get_batch(orch.data, jnp.arange(orch.data.num_genotype))
    tr = handlers.trace(handlers.substitute(
        handlers.seed(orch.jax_model, 0), data={**sites, **model_side})
    ).get_trace(priors=orch.priors, data=full)
    expected = {k: np.asarray(tr[k]["value"])
                for k in THETA_SITES + ("growth_pred",)}
    return dict(kind=request.param, orch=orch, tmp=tmp,
                unconstrained=unconstrained, model_side=model_side,
                expected=expected, sites=sites)


# ---------------------------------------------------------------------------
# Unit behavior
# ---------------------------------------------------------------------------

def test_hold_model_params_beats_outer_substitute():
    import numpyro

    def model():
        return numpyro.param("w", jnp.zeros(2))

    held = hold_model_params(model, {"w": jnp.ones(2)})
    out = handlers.substitute(handlers.seed(held, 0), data={"w": -jnp.ones(2)})()
    np.testing.assert_array_equal(out, [1.0, 1.0])
    tr = handlers.trace(handlers.seed(held, 0)).get_trace()
    assert "w" not in tr                      # hidden from outer handlers
    assert hold_model_params(model, {}) is model


def test_read_model_params_sources():
    group = {MODEL_PARAMS_GROUP: {"a": np.ones(2)}, "x": np.zeros((3, 2))}
    assert set(read_model_params(group)) == {"a"}
    assert sample_keys(group) == ["x"]
    raw = {"a": np.ones(2), "x_auto_loc": np.zeros(2)}
    assert set(read_model_params(raw, names=["a", "b"])) == {"a"}


def test_model_without_params_is_unchanged(tmp_path):
    """A model with no model-side params writes no group and runs as before."""
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    from tfscreen.tfmodel.inference.run_inference import RunInference

    orch = ModelOrchestrator(_growth_csv(tmp_path / "growth.csv"),
                             _binding_csv(tmp_path / "binding.csv"),
                             theta="hill_geno")
    ri = RunInference(orch, seed=0)
    svi = ri.setup_svi(guide_type="delta")
    state = svi.init(jax.random.PRNGKey(0), priors=orch.priors,
                     data=orch.get_batch(orch.data, jnp.arange(4)))
    params = svi.optim.get_params(state.optim_state)
    assert ri.model_params(params) == {}
    ri.get_map_posteriors(params, out_prefix=str(tmp_path / "p"))
    assert ri._fixed_model() is orch.jax_model
    with h5py.File(tmp_path / "p_posterior.h5", "r") as hf:
        assert MODEL_PARAMS_GROUP not in hf


# ---------------------------------------------------------------------------
# Posterior and prediction paths reproduce the MAP
# ---------------------------------------------------------------------------

def test_point_posterior_matches_map_theta(fit):
    """``--laplace point``: the posterior's theta is the MAP's theta."""
    from tfscreen.tfmodel.inference.run_inference import RunInference

    ri = RunInference(fit["orch"], seed=0)
    prefix = str(fit["tmp"] / "point")
    ri.get_map_posteriors(fit["unconstrained"], out_prefix=prefix)
    with h5py.File(f"{prefix}_posterior.h5", "r") as hf:
        for site, want in fit["expected"].items():
            np.testing.assert_allclose(hf[site][0], want, rtol=1e-5,
                                       atol=1e-5, err_msg=site)
        stored = read_model_params(hf)
    assert set(stored) == set(fit["model_side"])
    np.testing.assert_allclose(stored[SIGMA], fit["model_side"][SIGMA],
                               rtol=1e-6)


def test_predict_from_point_posterior_matches_map(fit):
    """tfs-predict-growth (``predict``) reuses the stored model params."""
    from tfscreen.tfmodel.analysis.prediction import predict
    from tfscreen.tfmodel.inference.run_inference import RunInference

    prefix = str(fit["tmp"] / "point_pred")
    RunInference(fit["orch"], seed=0).get_map_posteriors(
        fit["unconstrained"], out_prefix=prefix)
    stored_growth = fit["expected"]["growth_pred"]      # the MAP's own
    df = predict(fit["orch"], f"{prefix}_posterior.h5",
                 predict_sites=["growth_pred"], q_to_get=[0.5],
                 num_samples=None)
    # every predicted growth value is one of the MAP's own values
    pred = df["q0.5"].to_numpy()
    assert pred.size > 0
    nearest = np.min(np.abs(stored_growth.ravel()[None, :] - pred[:, None]),
                     axis=1)
    np.testing.assert_array_less(nearest, 1e-4 * np.maximum(1.0, np.abs(pred)))


def test_laplace_holds_model_params(fit):
    """The full Laplace: model params outside the Hessian, held at the MAP
    in every draw's forward pass."""
    from tfscreen.tfmodel.inference.run_inference import RunInference

    ri = RunInference(fit["orch"], seed=0)
    prefix = str(fit["tmp"] / "laplace")
    ri.get_laplace_posteriors(fit["unconstrained"], out_prefix=prefix,
                              num_posterior_samples=8, sampling_batch_size=4)
    sigma = np.asarray(fit["model_side"][SIGMA])
    with h5py.File(f"{prefix}_posterior.h5", "r") as hf:
        ddG = np.asarray(hf["theta_ddG"])              # (n, M, S)
        offset = np.asarray(hf["theta_ddG_offset"])    # (n, S, M)
        assert set(read_model_params(hf)) == set(fit["model_side"])
    # ddG = prior mean + sigma_s * offset with the fitted sigma_s and a
    # prior mean (NN output or ddG table) that is the same in every draw
    mean = ddG - np.swapaxes(sigma[:, None] * offset, -1, -2)
    np.testing.assert_allclose(mean, np.broadcast_to(mean[0], mean.shape),
                               atol=1e-4)
    point_mean = (fit["expected"]["theta_ddG"]
                  - (sigma[:, None] * np.asarray(fit["sites"]["theta_ddG_offset"])).T)
    np.testing.assert_allclose(mean[0], point_mean, atol=1e-4)


def test_svi_posterior_uses_fitted_model_params(tmp_path):
    """Guide sampling: the forward pass sees the store's sigma_s / NN."""
    from tfscreen.tfmodel.inference.run_inference import RunInference

    orch = _orchestrator(tmp_path, "PnnC")
    ri = RunInference(orch, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-6, guide_type="component")
    S = len(STRUCTS)
    start = {SIGMA: jnp.full(S, 2.5)}
    state, params, _ = ri.run_optimization(svi, init_params=start,
                                           max_num_epochs=1,
                                           out_prefix=str(tmp_path / "svi"),
                                           checkpoint_interval=100)
    sigma = np.asarray(svi.get_params(state)[SIGMA])
    assert np.all(np.abs(sigma - 2.5) < 0.01)
    ri.get_posteriors(svi, state, out_prefix=str(tmp_path / "svi"),
                      num_posterior_samples=6, sampling_batch_size=3)
    with h5py.File(tmp_path / "svi_posterior.h5", "r") as hf:
        ddG = np.asarray(hf["theta_ddG"])
        offset = np.asarray(hf["theta_ddG_offset"])
        np.testing.assert_allclose(read_model_params(hf)[SIGMA], sigma)
    mean = ddG - np.swapaxes(sigma[:, None] * offset, -1, -2)
    np.testing.assert_allclose(mean, np.broadcast_to(mean[0], mean.shape),
                               atol=1e-4)


def test_map_start_carries_model_params(tmp_path):
    """``init_params`` (the --init_from / pre-MAP hand-off) starts the
    model-side params where the earlier fit left them."""
    from tfscreen.tfmodel.inference.run_inference import RunInference

    orch = _orchestrator(tmp_path, "PddG")
    ri = RunInference(orch, seed=0)
    start = ri.model_params({SIGMA: np.full(len(STRUCTS), 3.0),
                             "not_a_param": np.ones(2)})
    assert set(start) == {SIGMA}
    svi = ri.setup_svi(adam_step_size=1e-6, guide_type="delta")
    state, params, _ = ri.run_optimization(svi, init_params=start,
                                           max_num_epochs=1,
                                           out_prefix=str(tmp_path / "m"),
                                           checkpoint_interval=100)
    np.testing.assert_allclose(params[SIGMA], 3.0, rtol=1e-3)


def test_extract_params_writes_model_params(fit):
    """tfs-extract-params shows the learned sigma_s (and NN weights)."""
    from tfscreen.tfmodel.inference.run_inference import RunInference
    from tfscreen.tfmodel.scripts.extract_params_cli import (
        _extract_from_checkpoint,
        _extract_from_posteriors,
    )

    prefix = str(fit["tmp"] / "extract")
    RunInference(fit["orch"], seed=0).get_map_posteriors(
        fit["unconstrained"], out_prefix=prefix)
    _extract_from_posteriors(fit["orch"], f"{prefix}_posterior.h5",
                             f"{prefix}_tfs")
    df = pd.read_csv(f"{prefix}_tfs_model_params.csv")
    sig = df[df["parameter"] == SIGMA].sort_values("index")
    np.testing.assert_allclose(sig["value"], fit["model_side"][SIGMA],
                               rtol=1e-6)
    assert set(df["parameter"]) == set(fit["model_side"])
