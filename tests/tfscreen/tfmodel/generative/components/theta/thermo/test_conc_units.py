"""
Concentration units shared by the thermodynamic theta models.

Every ``O2_C4`` and ``O2_C12`` component reads ``titrant_conc`` in mM (the
units of tfscreen growth and binding tables and simulator configs) and
multiplies it by ``theta_conc_unit_scale`` (1e-3) so that the equations run
in M. Until 2026-10-11 the ``O2_C4`` models skipped the conversion. Their
defaults moved so that behaviour on mM data is unchanged:

* ln_K_E: K_E multiplies [E]², so K_E·[E]² is invariant when [E] is scaled by
  s = 1e-3 and K_E by 1/s². ln_K_E shifts by -2·ln(s) = 2·ln(1e3) ≈ 13.8155.
* tf_total enters the free-effector mass balance next to [E], so it scales
  by s as well: 6.5e-7 (effectively mM) became 6.5e-10 M.
* K_op·op_total does not involve the titrant and is unchanged.
"""
import dataclasses
import importlib
import warnings
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from flax.struct import dataclass

from tfscreen.simulate.sample_theta import sample_theta_prior
from tfscreen.simulate.sim_data_class import build_sim_data
from tfscreen.tfmodel.configuration_io import _legacy_conc_unit_scale

_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"
_FAMILIES = ["O2_C4_K3_U0_a", "O2_C4_K3_U1_a",
             "O2_C12_K5_U0_a", "O2_C12_K5_U1_a"]
_KINDS = ["PK", "PddG", "PnnC"]
_C4 = [f for f in _FAMILIES if f.startswith("O2_C4")]

# The old O2_C4 defaults, in the units the old code used (concentrations
# as given, i.e. mM).
_LEGACY_C4 = {"theta_ln_K_E_wt_loc": 33.4,
              "theta_tf_total_M": 6.5e-7,
              "theta_conc_unit_scale": 1.0}

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
    assert "theta_tf_total_M" in hp and "theta_op_total_M" in hp


@pytest.mark.parametrize("family", _C4)
def test_c4_default_shift_is_exact(family):
    th = _thermo(family)
    assert th.DEFAULT_CONC_UNIT_SCALE == 1e-3
    assert th.DEFAULT_LN_K_E_WT_LOC == pytest.approx(33.4 + 2 * np.log(1e3),
                                                     abs=1e-12)
    assert th.DEFAULT_TF_TOTAL_M == pytest.approx(6.5e-10, rel=1e-12)
    for kind in _KINDS:
        hp = _module(family, kind).get_hyperparameters()
        assert hp["theta_ln_K_E_wt_loc"] == th.DEFAULT_LN_K_E_WT_LOC
        assert hp["theta_tf_total_M"] == th.DEFAULT_TF_TOTAL_M


def _theta_param(family, G=3, T=1, C=len(_CONC_MM), scale=1e-3):
    """A ThetaParam with distinct per-genotype constants."""
    th = _thermo(family)
    g = jnp.linspace(0.0, 1.0, G)
    common = dict(tf_total=6.5e-10, op_total=2.5e-8,
                  mu=jnp.zeros((T, C, 1)), sigma=jnp.ones((T, C, 1)),
                  conc_unit_scale=scale)
    if family.startswith("O2_C4"):
        ks = dict(ln_K_op=23.0 + g, ln_K_HL=-9.0 + g,
                  ln_K_E=jnp.tile(47.2 - 2 * g, (T, 1)))
        if family.endswith("U1_a"):
            ks["ln_K_U"] = -12.0 + g
    else:
        ks = dict(ln_K_h_l=1.84 + g, ln_K_h_o=19.9 + g,
                  ln_K_h_e=jnp.tile(10.9 - g, (T, 1)), ln_K_l_o=-2.3 + g,
                  ln_K_l_e=jnp.tile(13.5 - g, (T, 1)))
        common["tf_total"] = 6.5e-7
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
                                   op_total=param.op_total,
                                   conc_unit_scale=1e-3)[0]
    np.testing.assert_allclose(got, want.T.ravel(), rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# Default behaviour on mM data is unchanged by the unit fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", _C4)
def test_default_wt_prior_median_theta_unchanged(family):
    """
    The wt prior median gives the same theta at the same mM concentrations
    in the old units (scale 1, ln_K_E 33.4, tf 6.5e-7) and the new ones.
    """
    th = _thermo(family)
    hp = _module(family, "PK").get_hyperparameters()
    unfold = family.endswith("U1_a")
    conc = _CONC_MM.copy()

    def solve(ln_K_E, tf, scale):
        c = conc * scale
        c[c == 0] = th._ZERO_CONC_VALUE
        args = [np.full((1, conc.size), hp["theta_ln_K_op_wt_loc"]),
                np.full((1, conc.size), hp["theta_ln_K_HL_wt_loc"]),
                np.full((1, conc.size), ln_K_E)]
        if unfold:
            args.append(np.full((1, conc.size), hp["theta_ln_K_U_wt_loc"]))
        return th._solve_theta_np(*args, c, tf, hp["theta_op_total_M"])[0]

    old = solve(33.4, 6.5e-7, 1.0)
    new = solve(hp["theta_ln_K_E_wt_loc"], hp["theta_tf_total_M"],
                hp["theta_conc_unit_scale"])
    np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-15)
    # The wt is inducible across the mM range, as before.
    assert old[0] > 0.9 and old[-1] < 0.1


_SAMPLE_DF = pd.DataFrame({"titrant_name": "iptg",
                           "titrant_conc": [0.0, 1e-4, 1e-3, 0.01, 0.1, 1.0],
                           "condition_pre": "pre", "condition_sel": "sel"})
_GENOTYPES = ["wt", "H29A", "H29C", "L62A", "H29A/L62A", "H29C/L62A"]


def _ddG_csv(path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, (3, 4)), columns=["H", "HD", "L", "LE2"])
    df.insert(0, "mut", ["H29A", "H29C", "L62A"])
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("kind", ["PK", "PddG"])
@pytest.mark.parametrize("family", _C4)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seeded_c4_simulation_theta_unchanged(family, kind, seed, tmp_path):
    """
    A seeded prior-predictive draw gives the same genotype theta under the
    new defaults as under the old ones; ln_K_E moves by exactly 2·ln(1e3).
    """
    td = _ddG_csv(tmp_path / "ddG.csv") if kind == "PddG" else None
    sim_data = build_sim_data(pd.DataFrame({"genotype": _GENOTYPES}),
                              _SAMPLE_DF, thermo_data=td)
    key = jax.random.PRNGKey(seed)
    name = f"thermo.{family}.{kind}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta_new, param_new = sample_theta_prior(name, sim_data, key)
        theta_old, param_old = sample_theta_prior(name, sim_data, key,
                                                  priors_overrides=_LEGACY_C4)
    np.testing.assert_allclose(theta_new, theta_old, rtol=1e-4, atol=2e-6)
    np.testing.assert_allclose(np.asarray(param_new.ln_K_E)
                               - np.asarray(param_old.ln_K_E),
                               2 * np.log(1e3), atol=1e-4)
    np.testing.assert_array_equal(np.asarray(param_new.ln_K_op),
                                  np.asarray(param_old.ln_K_op))


# ---------------------------------------------------------------------------
# C4 is the K_h_l -> 0 limit of C12 when effector depletion is negligible
# ---------------------------------------------------------------------------

def test_c4_theta_is_a_limit_of_c12_on_the_same_mM_data():
    """
    With H unable to bind effector (K_h_e -> 0) and L unable to bind operator
    (K_l_o -> 0), C12 gives

        theta = r·K_h_o / (1 + K_h_l + r·K_h_o + K_h_l·K_l_e·e + K_h_l·K_l_e²·e²)

    and C4 gives

        theta = A / (B + A + C·e²),  A = K_op·op, B = 1 + K_HL, C = K_HL·K_E.

    theta is unchanged when every term is divided by B, so setting
    K_h_l = eps, K_l_e = sqrt(C / (B·eps)) and r·K_h_o = A/B matches every
    term but the singly-liganded L·E term, which is sqrt(eps·C/B)·e and
    vanishes as eps -> 0. The match needs negligible effector depletion,
    because the two mass balances differ (C4 counts operator-bound TF in the
    free-effector balance, C12 does not); tf_total is set near zero here.
    Both families get the same mM concentrations and the same scale.
    """
    c4 = _thermo("O2_C4_K3_U0_a")
    c12 = _thermo("O2_C12_K5_U0_a")
    rng = np.random.default_rng(0)
    S = 200
    hp = _module("O2_C4_K3_U0_a", "PK").get_hyperparameters()
    ln_K_op = rng.normal(hp["theta_ln_K_op_wt_loc"], 2.0, S)
    ln_K_HL = rng.normal(hp["theta_ln_K_HL_wt_loc"], 3.0, S)
    ln_K_E = rng.normal(hp["theta_ln_K_E_wt_loc"], 3.0, S)
    op_total, tf = 2.5e-8, 1e-20
    conc_mM = np.logspace(-6, 1, 60)
    calc_df = pd.DataFrame({"titrant_conc": conc_mM, "genotype_idx": 0,
                            "titrant_name_idx": 0})

    theta_c4 = c4.compute_theta_samples(
        calc_df, {"theta_ln_K_op": ln_K_op[:, None],
                  "theta_ln_K_HL": ln_K_HL[:, None],
                  "theta_ln_K_E": ln_K_E[:, None, None]},
        tf_total=tf, op_total=op_total, conc_unit_scale=1e-3)

    A = np.exp(ln_K_op) * op_total
    B = 1.0 + np.exp(ln_K_HL)
    lnC = ln_K_HL + ln_K_E
    for ln_eps, atol in [(-20.0, 1e-5), (-30.0, 1e-7)]:
        r = tf                    # dimer; C12 takes monomer tf_total = 2r
        post = {"theta_ln_K_h_l": np.full((S, 1), ln_eps),
                "theta_ln_K_h_o": (np.log(A / B) - np.log(r))[:, None],
                "theta_ln_K_h_e": np.full((S, 1, 1), -80.0),
                "theta_ln_K_l_o": np.full((S, 1), -80.0),
                "theta_ln_K_l_e": (0.5 * (lnC - np.log(B) - ln_eps))[:, None, None]}
        theta_c12 = c12.compute_theta_samples(calc_df, post, tf_total=2 * r,
                                              op_total=op_total,
                                              conc_unit_scale=1e-3)
        np.testing.assert_allclose(theta_c12, theta_c4, atol=atol)


# ---------------------------------------------------------------------------
# Saved priors CSVs from before the change
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Priors:
    theta: object


def test_legacy_priors_csv_reads_with_unit_scale_one():
    """A CSV without the theta_conc_unit_scale row predates the change: its
    values are in concentration-as-given units, so read it with scale 1."""
    theta = _module("O2_C4_K3_U0_a", "PK").get_priors()
    priors = _Priors(theta=theta)
    flat = {"theta.theta_ln_K_E_wt_loc": 33.4, "theta.theta_tf_total_M": 6.5e-7}
    with pytest.warns(UserWarning, match="theta_conc_unit_scale"):
        out = _legacy_conc_unit_scale(priors, flat, "old_priors.csv")
    assert out.theta.theta_conc_unit_scale == 1.0


def test_current_priors_csv_keeps_its_unit_scale():
    theta = _module("O2_C4_K3_U0_a", "PK").get_priors()
    priors = _Priors(theta=theta)
    flat = {"theta.theta_conc_unit_scale": 1e-3}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _legacy_conc_unit_scale(priors, flat)
    assert out is priors


def test_non_thermo_priors_are_untouched():
    @dataclass(frozen=True)
    class _Hill:
        theta_log_hill_K_loc: float = 0.0
    priors = _Priors(theta=_Hill())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _legacy_conc_unit_scale(priors, {}) is priors
