"""Tests for the relative Hill theta component (X scale; roadmap step 5)."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers
from scipy.special import expit

from tfscreen.tfmodel.generative.components.theta import hill_relative as hr

LOG_CONC = np.log(np.array([1e-20, 1e-3, 1e-2, 1e-1, 1.0]))
GAUGE = jnp.array([LOG_CONC[0], LOG_CONC[-1]])


def _data(num_genotype=4, wt=1, T=1):
    idx = jnp.arange(num_genotype)
    return SimpleNamespace(num_titrant_name=T, num_genotype=num_genotype,
                           wt_indexes=jnp.array([wt]),
                           theta_gauge_log_conc=GAUGE,
                           log_titrant_conc=jnp.asarray(LOG_CONC),
                           batch_idx=idx, geno_theta_idx=idx,
                           scatter_theta=0)


def _X(theta_param, data):
    return np.asarray(hr.run_model(theta_param, data))      # (T, C, G)


@pytest.mark.parametrize("log_K,n", [(-4.1, 2.0), (-2.0, 0.7), (-6.0, 4.0)])
def test_gauge_baselines_pin_wt(log_K, n):
    low, high = hr.gauge_baselines(jnp.array(log_K), jnp.array(n), GAUGE)
    occ = expit(n * (np.asarray(GAUGE) - log_K))
    X = float(low) + (float(high) - float(low)) * occ
    assert X == pytest.approx([1.0, 0.0], abs=1e-5)


def test_gauge_span_is_floored():
    # a wt curve that does not move across the gauge range stays finite
    low, high = hr.gauge_baselines(jnp.array(10.0), jnp.array(1.0), GAUGE)
    assert np.isfinite(float(low)) and np.isfinite(float(high))


@pytest.mark.parametrize("fn", [hr.define_model, hr.guide])
def test_wt_is_gauged_others_free(fn):
    data = _data()
    with handlers.seed(rng_seed=3):
        tp = fn("theta", data, hr.get_priors())
    X = _X(tp, data)
    assert X.shape == (1, len(LOG_CONC), 4)
    assert X[0, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert X[0, -1, 1] == pytest.approx(0.0, abs=1e-5)
    # other genotypes are not pinned
    assert not np.allclose(X[0, 0, [0, 2, 3]], 1.0)


def test_sites_and_deterministics():
    data = _data()
    tr = handlers.trace(handlers.seed(hr.define_model, 0)).get_trace(
        "theta", data, hr.get_priors())
    for site in ("theta_X_low", "theta_X_high", "theta_log_hill_K",
                 "theta_hill_n"):
        assert tr[site]["type"] == "deterministic"
        assert tr[site]["value"].shape == (1, 4)
    # per-genotype offsets are library-sized
    assert tr["theta_X_low_offset"]["value"].shape == (1, 4)


def test_run_model_follows_batch_idx():
    data = _data()
    with handlers.seed(rng_seed=0):
        tp = hr.define_model("theta", data, hr.get_priors())
    full = _X(tp, data)
    order = jnp.array([2, 0, 3, 1])
    sub = _X(tp, SimpleNamespace(**{**vars(data), "batch_idx": order}))
    assert np.allclose(sub, full[..., np.asarray(order)])


def test_scatter_shape():
    data = _data()
    with handlers.seed(rng_seed=0):
        tp = hr.define_model("theta", data, hr.get_priors())
    data.scatter_theta = 1
    assert hr.run_model(tp, data).shape == (1, 1, 1, 1, 1, len(LOG_CONC), 4)


def test_population_moments_are_none():
    assert hr.get_population_moments(None, None) == (None, None)


def test_priors_and_guesses():
    p = hr.get_priors()
    assert p.theta_X_low_hyper_loc_loc == 1.0
    assert p.theta_X_delta_hyper_loc_loc == -1.0
    g = hr.get_guesses("theta", _data())
    assert g["theta_X_low_offset"].shape == (1, 4)
    assert set(k for k in g if k.endswith("_hyper_loc")) == {
        "theta_X_low_hyper_loc", "theta_X_delta_hyper_loc",
        "theta_log_hill_K_hyper_loc", "theta_log_hill_n_hyper_loc"}


def test_compute_theta_samples_matches_run_model():
    data = _data()
    with handlers.seed(rng_seed=1):
        tp = hr.define_model("theta", data, hr.get_priors())
    post = {"theta_X_low": np.asarray(tp.X_low)[None],
            "theta_X_high": np.asarray(tp.X_high)[None],
            "theta_log_hill_K": np.asarray(tp.log_hill_K)[None],
            "theta_hill_n": np.asarray(tp.hill_n)[None]}
    conc = np.exp(LOG_CONC)
    conc[0] = 0.0
    calc = pd.DataFrame({"titrant_conc": np.tile(conc, 4),
                         "map_theta_group": np.repeat(np.arange(4), len(conc))})
    got = hr.compute_theta_samples(calc, post)[0].reshape(4, len(conc))
    assert np.allclose(got, _X(tp, data)[0].T, atol=1e-5)


def test_predict_unmeasured_uses_population_means():
    post = {"theta_X_low_hyper_loc": np.array([[1.0]]),
            "theta_X_delta_hyper_loc": np.array([[-1.0]]),
            "theta_log_hill_K_hyper_loc": np.array([[np.log(0.01)]]),
            "theta_log_hill_n_hyper_loc": np.array([[np.log(2.0)]])}
    grid = pd.DataFrame({"titrant_name": ["iptg", "iptg"],
                         "titrant_conc": [0.0, 0.01]})
    out = hr.predict_unmeasured(["A1V", "B2C"], ["iptg"], grid, None, None,
                                post, {"q0.5": 0.5})
    at0 = out[out["titrant_conc"] == 0.0]["q0.5"]
    atK = out[out["titrant_conc"] == 0.01]["q0.5"]
    assert np.allclose(at0, 1.0) and np.allclose(atK, 0.5)


def test_x_scale_truth_gauges_wt():
    wt = np.array([0.95, 0.6, 0.05])            # wt theta at c_lo, mid, c_hi
    X = hr.x_scale_truth(wt, wt[0], wt[-1])
    assert X == pytest.approx([1.0, (0.6 - 0.05) / 0.9, 0.0])
    # activity multiplies occupancy
    assert hr.x_scale_truth(0.5, 1.0, 0.0, activity=0.5) == pytest.approx(0.25)


def test_x_scale_growth_truth_preserves_growth():
    rng = np.random.default_rng(0)
    k, m = 0.02, -0.015
    lo, hi = 0.9, 0.1
    theta = rng.uniform(0, 1, 10)
    k_X, m_X = hr.x_scale_growth_truth(k, m, lo, hi)
    X = hr.x_scale_truth(theta, lo, hi)
    assert np.allclose(k + m * theta, k_X + m_X * X)


@pytest.mark.parametrize("fixed", [0.0, 0.5])
def test_log_hill_n_hyper_scale_fixed(fixed):
    """As hill_geno: a held log(n) population SD is a deterministic site with
    no guide parameters, and the gauge still pins wt."""
    data = _data()
    priors = hr.get_priors().replace(theta_log_hill_n_hyper_scale_fixed=fixed)
    assert hr.get_hyperparameters()["theta_log_hill_n_hyper_scale_fixed"] == 0.0
    mtr = handlers.trace(handlers.seed(hr.define_model, 0)).get_trace(
        "theta", data, priors)
    gtr = handlers.trace(handlers.seed(hr.guide, 0)).get_trace(
        "theta", data, priors)
    site = mtr["theta_log_hill_n_hyper_scale"]
    if fixed:
        assert site["type"] == "deterministic"
        assert np.allclose(site["value"], fixed)
        assert "theta_log_hill_n_hyper_scale_loc" not in gtr
    else:
        assert site["type"] == "sample"
    assert ({n for n, s in mtr.items() if s["type"] == "sample"}
            == {n for n, s in gtr.items() if s["type"] == "sample"})
    with handlers.seed(rng_seed=3):
        tp = hr.guide("theta", data, priors)
    X = _X(tp, data)
    assert X[0, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert X[0, -1, 1] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("h", ["X_low", "X_delta", "log_hill_K"])
def test_other_hyper_scales_fixed(h):
    """Every hyperscale can be held, one at a time, without touching the
    others or the gauge."""
    data = _data()
    field = f"theta_{h}_hyper_scale_fixed"
    assert hr.get_hyperparameters()[field] == 0.0
    priors = hr.get_priors().replace(**{field: 0.7})
    mtr = handlers.trace(handlers.seed(hr.define_model, 0)).get_trace(
        "theta", data, priors)
    gtr = handlers.trace(handlers.seed(hr.guide, 0)).get_trace(
        "theta", data, priors)
    assert mtr[f"theta_{h}_hyper_scale"]["type"] == "deterministic"
    assert np.allclose(mtr[f"theta_{h}_hyper_scale"]["value"], 0.7)
    assert f"theta_{h}_hyper_scale_loc" not in gtr
    others = {"X_low", "X_delta", "log_hill_K", "log_hill_n"} - {h}
    for o in others:
        assert mtr[f"theta_{o}_hyper_scale"]["type"] == "sample"
    assert ({n for n, s in mtr.items() if s["type"] == "sample"}
            == {n for n, s in gtr.items() if s["type"] == "sample"})
    with handlers.seed(rng_seed=3):
        tp = hr.guide("theta", data, priors)
    X = _X(tp, data)
    assert X[0, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert X[0, -1, 1] == pytest.approx(0.0, abs=1e-5)


def _as_h5(post):
    """The posteriors as an in-memory HDF5 file: datasets have a shape but
    no reshape, as in a tfs-sample-posterior .h5."""
    import h5py
    f = h5py.File("posterior.h5", "w", driver="core", backing_store=False)
    for k, v in post.items():
        f.create_dataset(k, data=np.asarray(v))
    return f


def test_compute_theta_samples_reads_h5_datasets():
    data = _data()
    with handlers.seed(rng_seed=1):
        tp = hr.define_model("theta", data, hr.get_priors())
    post = {"theta_X_low": np.asarray(tp.X_low)[None],
            "theta_X_high": np.asarray(tp.X_high)[None],
            "theta_log_hill_K": np.asarray(tp.log_hill_K)[None],
            "theta_hill_n": np.asarray(tp.hill_n)[None]}
    conc = np.exp(LOG_CONC)
    calc = pd.DataFrame({"titrant_conc": np.tile(conc, 4),
                         "map_theta_group": np.repeat(np.arange(4), len(conc))})
    expected = hr.compute_theta_samples(calc, post)
    with _as_h5(post) as f:
        got = hr.compute_theta_samples(calc, f)
    assert np.allclose(got, expected)


def test_predict_unmeasured_reads_h5_datasets():
    post = {"theta_X_low_hyper_loc": np.array([[1.0]]),
            "theta_X_delta_hyper_loc": np.array([[-1.0]]),
            "theta_log_hill_K_hyper_loc": np.array([[np.log(0.01)]]),
            "theta_log_hill_n_hyper_loc": np.array([[np.log(2.0)]])}
    grid = pd.DataFrame({"titrant_name": ["iptg", "iptg"],
                         "titrant_conc": [0.0, 0.01]})
    with _as_h5(post) as f:
        out = hr.predict_unmeasured(["A1V"], ["iptg"], grid, None, None,
                                    f, {"q0.5": 0.5})
    assert np.allclose(out[out["titrant_conc"] == 0.0]["q0.5"], 1.0)
    assert np.allclose(out[out["titrant_conc"] == 0.01]["q0.5"], 0.5)


def test_get_extract_specs_reads_the_growth_rows():
    df = pd.DataFrame({"genotype": ["wt"], "titrant_name": ["iptg"],
                       "map_theta_group": [0]})
    specs = hr.get_extract_specs(SimpleNamespace(
        growth_tm=SimpleNamespace(df=df)))
    assert len(specs) == 1
    spec = specs[0]
    assert spec["input_df"] is df
    assert spec["params_to_get"] == ["hill_n", "log_hill_K", "X_high", "X_low"]
    assert spec["map_column"] == "map_theta_group"
    assert spec["get_columns"] == ["genotype", "titrant_name"]
    assert spec["in_run_prefix"] == "theta_"
