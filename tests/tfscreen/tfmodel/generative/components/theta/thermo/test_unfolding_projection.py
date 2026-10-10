"""
Unfolding projection of the structure-based U1 thermodynamic models.

PddG and PnnC in ``O2_C4_K3_U1_a`` and ``O2_C12_K5_U1_a`` shift the free
energy of each folded structure X by ΔΔG_X and leave the unfolded state U
alone (ΔΔG_U = 0).  Since K_U = [U]/[H], Δln K_U = ΔΔG_H − ΔΔG_U = ΔΔG_H.
Until 2026-10-10 these models held ln_K_U at its wt value for every genotype,
which moved U with H and so made every mutation stability-neutral.
"""
import importlib
import types
from collections import namedtuple
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro.handlers import seed, substitute, trace

from tfscreen.genetics.build_mut_geno_matrix import build_mut_sparse_indices

_PKG = "tfscreen.tfmodel.generative.components.theta.thermo"

# genotypes: wt, A1C, D2E, A1C/D2E
_GENOTYPES = ["wt", "A1C", "D2E", "A1C/D2E"]
_MUT_LABELS = ["A1C", "D2E"]
_PAIR_LABELS = ["A1C/D2E"]
_M, _G, _T = 2, 4, 1
_MUT_GENO = np.array([[0, 1, 0, 1],
                      [0, 0, 1, 1]], dtype=np.float32)
_MUT_NNZ_MUT_IDX, _MUT_NNZ_GENO_IDX = build_mut_sparse_indices(_MUT_GENO)

_C4 = dict(
    family="O2_C4_K3_U1_a",
    structs=("H", "HD", "L", "LE2"),
    folded=("ln_K_op", "ln_K_HL", "ln_K_E"),
    unfold="ln_K_U",
    wt={"ln_K_op_wt": 23.0, "ln_K_HL_wt": -9.0, "ln_K_U_wt": 0.0,
        "ln_K_E_wt": np.full(_T, 33.4)},
    conc=np.array([0.0, 1e-5, 1e-4]),
    predict_kw={},
)
_C12 = dict(
    family="O2_C12_K5_U1_a",
    structs=("H", "HO", "L", "LO", "HE2", "LE2"),
    folded=("ln_K_h_l", "ln_K_h_o", "ln_K_h_e", "ln_K_l_o", "ln_K_l_e"),
    unfold="ln_K_u",
    wt={"ln_K_h_l_wt": 1.84, "ln_K_h_o_wt": 19.9, "ln_K_l_o_wt": -2.3,
        "ln_K_u_wt": 0.0, "ln_K_h_e_wt": np.full(_T, 10.9),
        "ln_K_l_e_wt": np.full(_T, 13.5)},
    conc=np.array([0.0, 0.1, 1.0]),
    predict_kw={"conc_unit_scale": 1e-3},
)

CASES = [(_C4, "PddG"), (_C4, "PnnC"), (_C12, "PddG"), (_C12, "PnnC")]
IDS = [f"{f['family']}.{k}" for f, k in CASES]

Data = namedtuple("Data", [
    "num_titrant_name", "num_titrant_conc", "num_genotype", "titrant_conc",
    "geno_theta_idx", "batch_idx", "scatter_theta", "num_mutation", "num_pair",
    "mut_geno_matrix", "mut_nnz_mut_idx", "mut_nnz_geno_idx",
    "pair_nnz_pair_idx", "pair_nnz_geno_idx", "num_struct", "struct_names",
    "struct_features", "struct_n_chains", "struct_contact_pair_idx",
    "struct_contact_distances",
])


def _modules(fam, kind):
    return (importlib.import_module(f"{_PKG}.{fam['family']}.{kind}"),
            importlib.import_module(f"{_PKG}.{fam['family']}.thermo"))


def _data(fam, kind, ddG_means, num_pair=0):
    S = len(fam["structs"])
    features = (np.asarray(ddG_means, dtype=np.float32) if kind == "PddG"
                else np.zeros((_M, S, 60), dtype=np.float32))
    return Data(
        num_titrant_name=_T, num_titrant_conc=len(fam["conc"]),
        num_genotype=_G, titrant_conc=jnp.array(fam["conc"]),
        geno_theta_idx=jnp.arange(_G), batch_idx=jnp.arange(_G),
        scatter_theta=0, num_mutation=_M, num_pair=num_pair,
        mut_geno_matrix=_MUT_GENO, mut_nnz_mut_idx=_MUT_NNZ_MUT_IDX,
        mut_nnz_geno_idx=_MUT_NNZ_GENO_IDX,
        pair_nnz_pair_idx=np.array([0] if num_pair else [], dtype=np.int32),
        pair_nnz_geno_idx=np.array([3] if num_pair else [], dtype=np.int32),
        num_struct=S, struct_names=fam["structs"], struct_features=features,
        struct_n_chains=np.full(S, 2, dtype=np.int32),
        struct_contact_pair_idx=None,
        struct_contact_distances=(np.full((num_pair, S), 5.0, dtype=np.float32)
                                  if num_pair else None),
    )


def _fixed_sites(fam, num_pair=0, epi_H=0.0):
    """Values for every sample site so model and guide see the same draw."""
    S = len(fam["structs"])
    sites = {f"theta_{k}": jnp.asarray(v) for k, v in fam["wt"].items()}
    sites["theta_ddG_offset"] = jnp.zeros((S, _M))   # ddG = prior means
    if num_pair:
        offset = np.full((S, num_pair), 0.3)
        offset[0] = epi_H                            # H column of epi ddG
        sites.update({
            "theta_epi_tau": jnp.array(0.5),
            "theta_epi_c2": jnp.array(4.0),
            "theta_epi_lambda": jnp.ones((S, num_pair)),
            "theta_epi_offset": jnp.asarray(offset),
        })
    return sites


def _run(fam, kind, ddG_means, fn="define_model", num_pair=0, epi_H=0.0):
    module, _ = _modules(fam, kind)
    data = _data(fam, kind, ddG_means, num_pair=num_pair)
    priors = module.get_priors()
    sites = _fixed_sites(fam, num_pair=num_pair, epi_H=epi_H)
    target = getattr(module, fn)

    def _call():
        with trace() as tr, substitute(data=sites), seed(rng_seed=0):
            out = target("theta", data, priors)
        return out, tr

    if kind == "PnnC":
        # Pin the NN prior means to the requested ΔΔG.
        with mock.patch.object(module, "compute_nn_predictions",
                               return_value=jnp.asarray(ddG_means)):
            return _call()
    return _call()


def _theta(fam, param, ln_K_unfold=None):
    """theta (T, C, G) from a ThetaParam, optionally with another ln_K_U."""
    _, thermo = _modules(fam, "PddG")
    if ln_K_unfold is not None:
        param = param.replace(**{fam["unfold"]: ln_K_unfold})
    return np.asarray(thermo.run_model(param, _data(fam, "PddG",
                                                    np.zeros((_M, 1)))))


def _means(fam, rows):
    S = len(fam["structs"])
    out = np.zeros((_M, S), dtype=np.float32)
    for m, row in rows.items():
        out[m] = row
    return out


@pytest.mark.parametrize("fam, kind", CASES, ids=IDS)
def test_H_only_ddG_shifts_unfolding_by_ddG_H(fam, kind):
    S = len(fam["structs"])
    ddG_H = 3.0
    h_only = np.zeros(S)
    h_only[0] = ddG_H
    out, tr = _run(fam, kind, _means(fam, {0: h_only}))
    ln_K_u = np.asarray(getattr(out, fam["unfold"]))

    # Δln K_U = ΔΔG_H for the mutant carrying it, and only for it.
    np.testing.assert_allclose(ln_K_u - ln_K_u[0], [0, ddG_H, 0, ddG_H],
                               atol=1e-5)
    d_site = f"theta_d_{fam['unfold']}"
    np.testing.assert_allclose(tr[d_site]["value"], [ddG_H, 0.0], atol=1e-6)
    np.testing.assert_allclose(tr[f"theta_{fam['unfold']}"]["value"], ln_K_u)

    # The shift changes theta: holding K_U at wt gives a different curve.
    theta = _theta(fam, out)
    pinned = _theta(fam, out, jnp.full(_G, ln_K_u[0]))
    assert not np.allclose(theta[..., 1], pinned[..., 1], atol=1e-4)
    np.testing.assert_allclose(theta[..., 2], pinned[..., 2], atol=1e-6)


@pytest.mark.parametrize("fam, kind", CASES, ids=IDS)
def test_uniform_ddG_is_a_pure_stability_effect(fam, kind):
    S = len(fam["structs"])
    shift = 2.5
    out, _ = _run(fam, kind, _means(fam, {0: np.full(S, shift)}))

    # Every folded-state constant is unchanged ...
    for k in fam["folded"]:
        v = np.asarray(getattr(out, k))
        np.testing.assert_allclose(v[..., 1], v[..., 0], atol=1e-5,
                                   err_msg=k)
    # ... while K_U rises by the shift, so the mutant still loses theta.
    ln_K_u = np.asarray(getattr(out, fam["unfold"]))
    assert ln_K_u[1] - ln_K_u[0] == pytest.approx(shift, abs=1e-5)
    theta = _theta(fam, out)
    assert np.all(theta[..., 1] < theta[..., 0])


@pytest.mark.parametrize("num_pair", [0, 1])
@pytest.mark.parametrize("fam, kind", CASES, ids=IDS)
def test_model_and_guide_assemble_the_same_constants(fam, kind, num_pair):
    rng = np.random.default_rng(1)
    means = rng.normal(0, 1, (_M, len(fam["structs"]))).astype(np.float32)
    model_out, _ = _run(fam, kind, means, num_pair=num_pair, epi_H=0.7)
    guide_out, _ = _run(fam, kind, means, fn="guide", num_pair=num_pair,
                        epi_H=0.7)
    for k in fam["folded"] + (fam["unfold"],):
        np.testing.assert_allclose(np.asarray(getattr(guide_out, k)),
                                   np.asarray(getattr(model_out, k)),
                                   atol=1e-5, err_msg=k)


@pytest.mark.parametrize("fam, kind", CASES, ids=IDS)
def test_pair_epistasis_projects_onto_unfolding(fam, kind):
    rng = np.random.default_rng(2)
    means = rng.normal(0, 1, (_M, len(fam["structs"]))).astype(np.float32)
    out, tr = _run(fam, kind, means, num_pair=1, epi_H=0.7)
    u = fam["unfold"]
    d = np.asarray(tr[f"theta_d_{u}"]["value"])
    epi = np.asarray(tr[f"theta_epi_{u}"]["value"])
    np.testing.assert_allclose(epi, np.asarray(tr["theta_epi_ddG"]["value"])[:, 0],
                               atol=1e-6)
    assert abs(epi[0]) > 1e-3
    ln_K_u = np.asarray(getattr(out, u))
    assert ln_K_u[3] - ln_K_u[0] == pytest.approx(d[0] + d[1] + epi[0],
                                                  abs=1e-5)


@pytest.mark.parametrize("num_pair", [0, 1])
@pytest.mark.parametrize("fam, kind", CASES, ids=IDS)
def test_predict_unmeasured_matches_model(fam, kind, num_pair):
    """predict_unmeasured rebuilds ln_K_U from d_ln_K_U/epi_ln_K_U."""
    module, _ = _modules(fam, kind)
    rng = np.random.default_rng(3)
    means = rng.normal(0, 1.5, (_M, len(fam["structs"]))).astype(np.float32)
    out, tr = _run(fam, kind, means, num_pair=num_pair, epi_H=0.7)

    post = {f"theta_{k}": np.asarray(v)[None] for k, v in fam["wt"].items()}
    for site, rec in tr.items():
        if site.startswith(("theta_d_", "theta_epi_ln")):
            post[site] = np.asarray(rec["value"])[None]

    titrant_df = pd.DataFrame({"titrant_name": "iptg",
                               "titrant_conc": fam["conc"]})
    priors = module.get_priors()
    result = module.predict_unmeasured(
        _GENOTYPES, ["iptg"], titrant_df, _MUT_LABELS,
        _PAIR_LABELS if num_pair else [], post, {"median": 0.5},
        tf_total=priors.theta_tf_total_M, op_total=priors.theta_op_total_M,
        **fam["predict_kw"])

    theta = _theta(fam, out)[0]           # (C, G)
    for g_idx, g in enumerate(_GENOTYPES):
        got = result.loc[result["genotype"] == g, "median"].to_numpy()
        np.testing.assert_allclose(got, theta[:, g_idx], atol=1e-4,
                                   err_msg=g)


@pytest.mark.parametrize("fam, kind", CASES, ids=IDS)
def test_extract_specs_include_unfolding_delta(fam, kind):
    module, _ = _modules(fam, kind)
    growth_tm = types.SimpleNamespace(
        tensor_dim_names=["genotype", "titrant_name"],
        tensor_dim_labels=[_GENOTYPES, ["iptg"]],
        df=pd.DataFrame({"genotype": _GENOTYPES,
                         "genotype_idx": range(_G),
                         "titrant_name": "iptg", "titrant_name_idx": 0}))
    ctx = types.SimpleNamespace(growth_tm=growth_tm, mut_labels=_MUT_LABELS,
                                pair_labels=_PAIR_LABELS)
    names = [p for s in module.get_extract_specs(ctx) for p in s["params_to_get"]]
    assert f"d_{fam['unfold']}" in names
    assert f"epi_{fam['unfold']}" in names
