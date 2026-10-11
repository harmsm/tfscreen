"""
Held per-state ΔΔG scale σ_s (``theta_ddG_sigma_fixed``; thermo/prior.py).

With ``theta_ddG_sigma_fixed`` > 0 every PddG/PnnC module records
``theta_ddG_sigma`` as a deterministic site of the held value instead of a
sample site, the guide has no σ_s parameters or site, the MultivariateNormal
on Δln K uses the held value, and a fit still extracts a ``ddG_sigma``
table (the held value in every row).  The default 0 samples σ_s as before.
"""

import importlib
import importlib.util
import os
import warnings
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
import yaml
from numpyro.handlers import seed, trace

from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.prior import (
    constant_scale_tril,
    ddG_sigma_fixed,
    guide_dlnK,
    projection,
    sample_dlnK,
)

_HERE = Path(__file__).resolve().parent
_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_FAMILIES = ["O2_C4_K3_U0_a", "O2_C4_K3_U1_a", "O2_C12_K5_U0_a",
             "O2_C12_K5_U1_a"]
_MODULES = [(fam, kind) for fam in _FAMILIES for kind in ("PddG", "PnnC")]
_HELD = 1.69
_SITE = "theta_ddG_sigma"


def _load_test_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_sites(tr):
    return {k for k, v in tr.items() if v["type"] == "sample"}


def _params(tr):
    return {k for k, v in tr.items() if v["type"] == "param"}


# ── prior.py ────────────────────────────────────────────────────────────────

class TestPriorBlock:
    P = projection(mwc.PROJ_C4)
    S, M = 4, 5

    def _means(self):
        return jnp.arange(self.M * self.S, dtype=jnp.float32).reshape(
            self.M, self.S) / 10.0

    def _model_trace(self, sigma_fixed):
        def model():
            return sample_dlnK("theta", ["H", "HD", "L", "LE2"], self.M,
                               self._means(), self.P, sigma_fixed=sigma_fixed)
        with seed(rng_seed=0):
            return trace(model).get_trace()

    def _guide_trace(self, sigma_fixed):
        def g():
            return guide_dlnK("theta", self.M, self.S, self.P, self._means(),
                              sigma_fixed=sigma_fixed)
        with seed(rng_seed=0):
            return trace(g).get_trace()

    def test_held_sigma_is_deterministic(self):
        tr = self._model_trace(_HELD)
        assert tr[_SITE]["type"] == "deterministic"
        np.testing.assert_allclose(tr[_SITE]["value"], np.full(self.S, _HELD),
                                   rtol=1e-6)
        assert _sample_sites(tr) == {"theta_dlnK"}

    def test_mvn_uses_the_held_sigma(self):
        tr = self._model_trace(_HELD)
        fn = tr["theta_dlnK"]["fn"]
        want = constant_scale_tril(jnp.asarray(self.P, dtype=jnp.float32),
                                   jnp.full(self.S, _HELD))
        np.testing.assert_allclose(np.asarray(fn.scale_tril).reshape(-1, 3, 3)[0],
                                   np.asarray(want),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(fn.loc),
                                   np.asarray(self._means()) @ self.P.T,
                                   rtol=1e-6)

    def test_guide_skips_held_sigma(self):
        gtr = self._guide_trace(_HELD)
        assert _SITE not in gtr
        assert _params(gtr) == {"theta_dlnK_locs", "theta_dlnK_scales"}
        assert _sample_sites(gtr) == _sample_sites(self._model_trace(_HELD))

    def test_default_samples_sigma(self):
        tr = self._model_trace(0.0)
        assert tr[_SITE]["type"] == "sample"
        assert isinstance(tr[_SITE]["fn"].base_dist, dist.LogNormal)
        gtr = self._guide_trace(0.0)
        assert {"theta_ddG_sigma_locs", "theta_ddG_sigma_scales"} <= _params(gtr)
        assert _sample_sites(gtr) == _sample_sites(tr)

    def test_wt_only_library_with_held_sigma(self):
        def model():
            return sample_dlnK("theta", ["H", "HD", "L", "LE2"], 0,
                               jnp.zeros((0, 4)), self.P, sigma_fixed=_HELD)
        with seed(rng_seed=0):
            tr = trace(model).get_trace()
        assert _sample_sites(tr) == set()
        assert tr[_SITE]["type"] == "deterministic"

    def test_ddG_sigma_fixed_reads_priors(self):
        class _P:
            theta_ddG_sigma_fixed = 0.5
        assert ddG_sigma_fixed(_P()) == 0.5
        assert ddG_sigma_fixed(object()) == 0.0


# ── all 8 modules ────────────────────────────────────────────────────────────

def _module_and_mock(fam, kind):
    module = importlib.import_module(f"{_PKG}.{fam}.{kind}")
    test_mod = _load_test_module(_HERE / fam / f"test_{kind}.py",
                                 f"_held_sigma_{fam}_{kind}")
    return module, test_mod._make_mock()


def _traces(module, data, priors):
    with seed(rng_seed=0):
        mtr = trace(lambda: module.define_model("theta", data, priors)).get_trace()
    with seed(rng_seed=0):
        gtr = trace(lambda: module.guide("theta", data, priors)).get_trace()
    return mtr, gtr


@pytest.mark.parametrize("fam, kind", _MODULES)
def test_default_is_zero_and_static(fam, kind):
    module = importlib.import_module(f"{_PKG}.{fam}.{kind}")
    assert module.get_hyperparameters()["theta_ddG_sigma_fixed"] == 0.0
    priors = module.get_priors()
    assert priors.theta_ddG_sigma_fixed == 0.0
    # Static (pytree_node=False): not a leaf, so jit sees a Python float.
    held = priors.replace(theta_ddG_sigma_fixed=_HELD)
    import jax
    assert len(jax.tree_util.tree_leaves(held)) == \
        len(jax.tree_util.tree_leaves(priors))


@pytest.mark.parametrize("fam, kind", _MODULES)
def test_module_holds_sigma(fam, kind):
    module, data = _module_and_mock(fam, kind)
    S = len(module.STRUCTURE_NAMES)
    held = module.get_priors().replace(theta_ddG_sigma_fixed=_HELD)
    mtr, gtr = _traces(module, data, held)

    assert mtr[_SITE]["type"] == "deterministic"
    np.testing.assert_allclose(mtr[_SITE]["value"], np.full(S, _HELD),
                               rtol=1e-6)
    assert _SITE not in gtr
    assert not {"theta_ddG_sigma_locs", "theta_ddG_sigma_scales"} & _params(gtr)
    assert _sample_sites(mtr) == _sample_sites(gtr)

    # The MVN on Δln K uses the held σ_s.
    want = constant_scale_tril(jnp.asarray(module.DLNK_PROJ, dtype=jnp.float32),
                               jnp.full(S, _HELD))
    C = want.shape[0]
    got = np.asarray(mtr["theta_dlnK"]["fn"].scale_tril).reshape(-1, C, C)[0]
    np.testing.assert_allclose(got, np.asarray(want), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fam, kind", _MODULES)
def test_module_default_samples_sigma(fam, kind):
    module, data = _module_and_mock(fam, kind)
    mtr, gtr = _traces(module, data, module.get_priors())
    assert mtr[_SITE]["type"] == "sample"
    assert gtr[_SITE]["type"] == "sample"
    assert {"theta_ddG_sigma_locs", "theta_ddG_sigma_scales"} <= _params(gtr)


# ── tfs-configure-model --set_priors ─────────────────────────────────────────

_EXTRACTION_TEST = (_HERE.parents[3] / "analysis" / "test_ddG_extraction.py")


def _simulate_and_configure(tmp_path, set_priors=None):
    """Simulate a tiny C4_U0 PddG screen and configure a fit of it."""
    from tfscreen.simulate.scripts.simulate_cli import run_simulation_from_config
    from tfscreen.tfmodel.scripts.configure_model_cli import configure_model

    helpers = _load_test_module(_EXTRACTION_TEST, "_held_sigma_extraction")
    component = "thermo.O2_C4_K3_U0_a.PddG"
    module = importlib.import_module(f"{_PKG}.O2_C4_K3_U0_a.PddG")
    thermo_data = helpers._reordered_ddG_csv(tmp_path / "ddG.csv",
                                             module.STRUCTURE_NAMES)
    cfg = dict(helpers._SIM_CONFIG, theta_component=component,
               thermo_data=thermo_data)
    with open(tmp_path / "sim.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    run = tmp_path / "run"
    run.mkdir()
    cwd = os.getcwd()
    os.chdir(run)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_simulation_from_config(str(tmp_path / "sim.yaml"),
                                       out_prefix="tfs_sim", num_replicates=1,
                                       write_raw=False, write_growth=True)
            configure_model(growth_df="tfs_sim_growth.csv",
                            presplit_df="tfs_sim_presplit.csv",
                            theta_model=component,
                            library_config="tfs_sim_input-config.yaml",
                            growth_shares_replicates=True,
                            thermo_data=thermo_data,
                            set_priors=set_priors)
    finally:
        os.chdir(cwd)
    return run, module


@pytest.mark.parametrize("name", ["theta.theta_ddG_sigma_fixed",
                                  "theta_ddG_sigma_fixed"])
def test_set_priors_names_resolve(tmp_path, name):
    """Both spellings set the PddG/PnnC row of a priors CSV."""
    from tfscreen.tfmodel.priors_edit import (
        apply_prior_overrides,
        parse_prior_overrides,
    )

    module = importlib.import_module(f"{_PKG}.O2_C4_K3_U0_a.PddG")
    rows = {f"theta.{k}": v for k, v in module.get_hyperparameters().items()}
    rows["growth.sample_offset.sigma_fixed"] = 0.0
    path = tmp_path / "priors.csv"
    pd.DataFrame({"parameter": list(rows), "value": list(rows.values())}
                 ).to_csv(path, index=False)
    out = apply_prior_overrides(str(path),
                                parse_prior_overrides([f"{name}={_HELD}"]))
    assert out == {"theta.theta_ddG_sigma_fixed": _HELD}
    df = pd.read_csv(path).set_index("parameter")["value"]
    assert float(df["theta.theta_ddG_sigma_fixed"]) == _HELD
    assert float(df["growth.sample_offset.sigma_fixed"]) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("name", ["theta.theta_ddG_sigma_fixed",
                                  "theta_ddG_sigma_fixed"])
def test_set_priors_round_trip(tmp_path, name):
    from tfscreen.tfmodel.configuration_io import read_configuration

    run, _ = _simulate_and_configure(tmp_path, [f"{name}={_HELD}"])
    df = pd.read_csv(run / "tfs_configure_priors.csv")
    row = df.loc[df["parameter"] == "theta.theta_ddG_sigma_fixed", "value"]
    assert float(row.item()) == _HELD
    cwd = os.getcwd()
    os.chdir(run)
    try:
        o, _ = read_configuration("tfs_configure_config.yaml")
    finally:
        os.chdir(cwd)
    assert float(o.priors.theta.theta_ddG_sigma_fixed) == _HELD


@pytest.mark.slow
def test_held_sigma_map_laplace_extract(tmp_path):
    """MAP, full Laplace and extraction on a tiny PddG fit with σ_s held."""
    from tfscreen.tfmodel.configuration_io import read_configuration
    from tfscreen.tfmodel.scripts.extract_params_cli import extract_params
    from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
    from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior

    run, module = _simulate_and_configure(
        tmp_path, [f"theta.theta_ddG_sigma_fixed={_HELD}"])
    structs = list(module.STRUCTURE_NAMES)

    cwd = os.getcwd()
    os.chdir(run)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, _ = read_configuration("tfs_configure_config.yaml")
            assert float(o.priors.theta.theta_ddG_sigma_fixed) == _HELD
            fit_model("tfs_configure_config.yaml", seed=1,
                      analysis_method="map", max_num_epochs=20,
                      convergence_window_steps=10)
            sample_posterior("tfs_configure_config.yaml",
                             "tfs_fit_model_checkpoint.pkl",
                             laplace="full", num_posterior_samples=5,
                             skip_growth_observations=True)
            extract_params("tfs_configure_config.yaml", "tfs_posterior.h5")
    finally:
        os.chdir(cwd)

    # No σ_s in the MAP parameters (no sample site, no guide params).
    params = np.load(next(run.glob("*_params.npz")))
    assert not [k for k in params.files if k.startswith(_SITE)]

    # The posterior carries the deterministic site at the held value.
    with h5py.File(run / "tfs_posterior.h5", "r") as hf:
        sig = np.asarray(hf[_SITE][:])
    np.testing.assert_allclose(sig, _HELD, rtol=1e-6)

    fit_sigma = pd.read_csv(run / "tfs_params_ddG_sigma.csv")
    assert list(fit_sigma["structure"]) == structs
    np.testing.assert_allclose(fit_sigma["q0.5"], _HELD, rtol=1e-6)
