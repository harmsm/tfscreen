"""
Concentration units shared by the thermodynamic theta models.

Every ``O2_C4`` and ``O2_C12`` component reads ``titrant_conc`` in mM (the
units of tfscreen growth and binding tables and simulator configs) and
multiplies it by ``theta_conc_unit_scale`` (1e-3) so that the equations run
in M, with every effector and operator constant in M⁻¹ and tf_total in M
(monomer units). The exact nesting of the two families is tested in
``test_mwc.py``.
"""
import dataclasses
import importlib
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest


_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_FAMILIES = ["O2_C4_K3_U0_a", "O2_C4_K3_U1_a",
             "O2_C12_K5_U0_a", "O2_C12_K5_U1_a"]
_KINDS = ["PK", "PddG", "PnnC"]

# mM grid spanning the simulator configs (0 to 1 mM) and beyond.
_CONC_MM = np.array([0.0, 1e-5, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0])

Data = namedtuple("Data", ["titrant_conc", "batch_idx", "geno_theta_idx",
                           "scatter_theta"])


def _module(family, kind):
    return importlib.import_module(f"{_PKG}.{family}.{kind}")


def _thermo(family):
    return importlib.import_module(f"{_PKG}.{family}.thermo")


# ---------------------------------------------------------------------------
# One convention for both families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("family", _FAMILIES)
def test_every_thermo_component_converts_mM_to_M(family, kind):
    module = _module(family, kind)
    hp = module.get_hyperparameters()
    assert hp["theta_conc_unit_scale"] == pytest.approx(1e-3)
    assert "theta_conc_unit_scale" in {f.name for f in
                                       dataclasses.fields(module.ModelPriors)}
    assert hp["theta_tf_total_M"] == pytest.approx(1e-5)
    # Operator total does not enter either family (TF is in large excess).
    assert "theta_op_total_M" not in hp


def _theta_param(family, G=3, T=1, C=len(_CONC_MM), scale=1e-3):
    """A ThetaParam with distinct per-genotype constants."""
    th = _thermo(family)
    g = jnp.linspace(0.0, 1.0, G)
    common = dict(tf_total=1e-5,
                  mu=jnp.zeros((T, C, 1)), sigma=jnp.ones((T, C, 1)),
                  conc_unit_scale=scale)
    if family.startswith("O2_C4"):
        ks = dict(ln_K_op=19.9 + g, ln_K_HL=1.84 + g,
                  ln_K_E=jnp.tile(12.4 - g, (T, 1)))
        if family.endswith("U1_a"):
            ks["ln_K_U"] = -12.0 + g
    else:
        ks = dict(ln_K_h_l=1.84 + g, ln_K_h_o=19.9 + g,
                  ln_K_h_e=jnp.tile(10.9 - g, (T, 1)), ln_K_l_o=-2.3 + g,
                  ln_K_l_e=jnp.tile(13.5 - g, (T, 1)))
        if family.endswith("U1_a"):
            ks["ln_K_u"] = -12.0 + g
    return th.ThetaParam(**ks, **common)


@pytest.mark.parametrize("family", _FAMILIES)
def test_run_model_reads_data_concentrations_as_mM(family):
    """mM data with scale 1e-3 gives the theta of the same data in M."""
    th = _thermo(family)
    G = 3
    data_mM = Data(titrant_conc=jnp.array(_CONC_MM), batch_idx=jnp.arange(G),
                   geno_theta_idx=jnp.arange(G), scatter_theta=0)
    data_M = data_mM._replace(titrant_conc=jnp.array(_CONC_MM * 1e-3))
    got = th.run_model(_theta_param(family, scale=1e-3), data_mM)
    want = th.run_model(_theta_param(family, scale=1.0), data_M)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)


def _posteriors(param):
    """One-sample posterior dict from a ThetaParam's K fields."""
    return {f"theta_{f.name}": np.asarray(getattr(param, f.name))[None]
            for f in dataclasses.fields(param) if f.name.startswith("ln_K")}


@pytest.mark.parametrize("family", _FAMILIES)
def test_numpy_path_matches_run_model(family):
    """compute_theta_samples applies the same mM-to-M conversion."""
    th = _thermo(family)
    G = 3
    param = _theta_param(family)
    data = Data(titrant_conc=jnp.array(_CONC_MM), batch_idx=jnp.arange(G),
                geno_theta_idx=jnp.arange(G), scatter_theta=0)
    want = np.asarray(th.run_model(param, data))[0]          # (C, G)
    calc_df = pd.DataFrame({
        "titrant_conc": np.tile(_CONC_MM, G),
        "genotype_idx": np.repeat(np.arange(G), len(_CONC_MM)),
        "titrant_name_idx": 0})
    got = th.compute_theta_samples(calc_df, _posteriors(param),
                                   tf_total=param.tf_total,
                                   conc_unit_scale=1e-3)[0]
    np.testing.assert_allclose(got, want.T.ravel(), rtol=1e-4, atol=1e-7)
