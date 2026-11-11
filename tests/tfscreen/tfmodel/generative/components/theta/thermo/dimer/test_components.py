"""
Component tests for thermo.dimer.PK, .PddG and .PnnC (model.py).

Traces on a small hand-built library: sites and shapes per variant and
titrant count, model and guide site agreement, the assembly (single =
wt + d, double = wt + d1 + d2 + epi), the effective operator constant,
the structural block, wt-only libraries, predict_unmeasured against
compute_theta_samples, the subset contract of run_model, extraction
specs, and the priors class (pickle and priors-CSV round trip of the
variant flags).
"""

import pickle
import types

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp
from numpyro.handlers import condition, seed, trace

from tfscreen.genetics.build_mut_geno_matrix import (
    build_mut_geno_matrix,
    build_mut_sparse_indices,
)
from tfscreen.tfmodel import configuration_io
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import (
    PK, PddG, PnnC, model as dm, species as sp,
)

GENOTYPES = ["wt", "A1V", "C2T", "D3K", "A1V/C2T", "C2T/D3K"]
CONC = np.array([0.0, 1e-3, 1e-1, 1.0], dtype=np.float32)    # mM
MODULES = {"PK": PK, "PddG": PddG, "PnnC": PnnC}
ALL_ON = {"theta_monomers": True, "theta_unfolded_monomer": True,
          "theta_effector_coupling": True, "theta_operator_coupling": True}


def _data(T=1, epistasis=True, genotypes=GENOTYPES, distances=True):
    mut_labels, pair_labels, mgm, pp, pg = build_mut_geno_matrix(
        genotypes, skip_pairs=not epistasis)
    mi, mg = build_mut_sparse_indices(mgm)
    M, P, G = len(mut_labels), len(pair_labels), len(genotypes)
    rng = np.random.default_rng(0)
    names = tuple(PddG.STRUCTURE_NAMES) + PddG.MONOMER_STRUCTURE_NAMES
    S = len(names)
    return types.SimpleNamespace(
        num_titrant_name=T, num_titrant_conc=len(CONC), num_genotype=G,
        titrant_conc=jnp.asarray(CONC), log_titrant_conc=None,
        batch_idx=jnp.arange(G), geno_theta_idx=jnp.arange(G),
        scatter_theta=0, num_mutation=M, num_pair=P,
        mut_nnz_mut_idx=mi, mut_nnz_geno_idx=mg,
        pair_nnz_pair_idx=pp, pair_nnz_geno_idx=pg,
        struct_names=names + ("extra",),
        struct_features=None, struct_n_chains=(2,) * (S + 1),
        struct_contact_distances=(rng.uniform(3, 30, (P, S + 1))
                                  .astype(np.float32) if distances and P else None),
        mut_labels=mut_labels, pair_labels=pair_labels, rng=rng, S=S)


def _for(key, data):
    """``data`` with the struct_features ``key`` reads."""
    M, S = data.num_mutation, data.S
    if key == "PddG":
        feats = data.rng.normal(0, 1, (M, S + 1)).astype(np.float32)
    elif key == "PnnC":
        feats = data.rng.normal(0, 1, (M, S + 1, 60)).astype(np.float32)
    else:
        feats = None
    return types.SimpleNamespace(**{**vars(data), "struct_features": feats})


def _priors(module, **flags):
    p = module.get_hyperparameters()
    p.update(flags)
    return module.ModelPriors(**p)


def _trace(fn, module, data, priors, seed_val=0, cond=None):
    f = seed(fn, rng_seed=seed_val)
    if cond:
        f = condition(f, data=cond)
    tr = trace(f).get_trace("theta", data, priors)
    return tr


@pytest.mark.parametrize("key", list(MODULES))
@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("flags", [{}, ALL_ON], ids=["default", "all_on"])
def test_model_traces_and_theta_is_valid(key, T, flags):
    module = MODULES[key]
    data = _for(key, _data(T=T))
    priors = _priors(module, **flags)
    tr = _trace(module.define_model, module, data, priors)
    variant = dm.variant_from_priors(priors)
    G, M, P = data.num_genotype, data.num_mutation, data.num_pair
    for t in sp.genotype_terms(variant):
        assert tr[f"theta_{t}"]["value"].shape == (G,)
        assert tr[f"theta_d_{t}"]["value"].shape == (M,)
        assert tr[f"theta_epi_{t}"]["value"].shape == (P,)
    for t in sp.titrant_terms(variant):
        assert tr[f"theta_{t}"]["value"].shape == (T, G)
        assert tr[f"theta_d_{t}"]["value"].shape == (T, M)
    assert ("theta_d_dev" in tr) == (T > 1)
    param = seed(module.define_model, rng_seed=0)("theta", data, priors)
    th = np.asarray(module.run_model(param, data))
    assert th.shape == (T, len(CONC), G)
    assert np.all(np.isfinite(th)) and np.all((th >= 0) & (th <= 1))


@pytest.mark.parametrize("key", list(MODULES))
@pytest.mark.parametrize("flags", [{}, ALL_ON], ids=["default", "all_on"])
def test_guide_samples_the_model_sites(key, flags):
    module = MODULES[key]
    data = _for(key, _data(T=2))
    priors = _priors(module, **flags)
    model_tr = _trace(module.define_model, module, data, priors)
    guide_tr = _trace(module.guide, module, data, priors)

    def samples(tr):
        return {n: s["value"].shape for n, s in tr.items()
                if s["type"] == "sample" and not s.get("is_observed")}
    assert samples(guide_tr) == samples(model_tr)
    param = seed(module.guide, rng_seed=0)("theta", data, priors)
    assert np.all(np.isfinite(np.asarray(module.run_model(param, data))))


@pytest.mark.parametrize("key", list(MODULES))
def test_assembly_is_additive_in_terms(key):
    module = MODULES[key]
    data = _for(key, _data(T=2))
    priors = _priors(module, **ALL_ON)
    tr = _trace(module.define_model, module, data, priors)
    v = lambda n: np.asarray(tr[f"theta_{n}"]["value"])  # noqa: E731
    idx = {g: i for i, g in enumerate(GENOTYPES)}
    mi = {m: i for i, m in enumerate(data.mut_labels)}
    pi = {p: i for i, p in enumerate(data.pair_labels)}
    for t in sp.active_terms(dm.variant_from_priors(priors)):
        val, d, epi = v(t), v(f"d_{t}"), v(f"epi_{t}")
        wt = val[..., idx["wt"]]
        np.testing.assert_allclose(val[..., idx["A1V"]], wt + d[..., mi["A1V"]],
                                   atol=1e-5)
        expected = (wt + d[..., mi["A1V"]] + d[..., mi["C2T"]]
                    + epi[pi["A1V/C2T"]])
        np.testing.assert_allclose(val[..., idx["A1V/C2T"]], expected, atol=1e-5)


def test_wt_operator_prior_is_on_the_effective_constant():
    data = _data()
    for flags in ({}, {"theta_second_engagement": False}):
        priors = _priors(PK, **flags)
        tr = _trace(PK.define_model, PK, data, priors)
        for x in ("h", "l"):
            eff_wt = float(tr[f"theta_ln_K_{x}_o_eff_wt"]["value"])
            assert float(tr[f"theta_ln_K_{x}_o_eff"]["value"][0]) == \
                pytest.approx(eff_wt, abs=1e-5)
        assert ("theta_ln_K_h_oii_wt" in tr) == (not flags)


def test_effector_deviations_tie_titrants_together():
    data = _data(T=2)
    priors = _priors(PK)
    tr = _trace(PK.define_model, PK, data, priors)
    d = np.asarray(tr["theta_d_ln_K_h_e"]["value"])
    dev = np.asarray(tr["theta_d_dev"]["value"])
    # shared part plus deviation: the titrants' difference is the deviations'
    np.testing.assert_allclose(d[0] - d[1], dev[0, 0] - dev[0, 1], atol=1e-5)
    fn = tr["theta_d_dev"]["fn"]
    assert float(np.max(fn.base_dist.scale)) == pytest.approx(0.5)


def test_pddg_structural_block_at_zero_sigma_is_the_projection():
    data = _for("PddG", _data())
    priors = _priors(PddG)
    tr = _trace(PddG.define_model, PddG, data, priors,
                cond={"theta_ddG_sigma": jnp.zeros(6)})
    perm = PddG.struct_perm(data)
    means = np.asarray(data.struct_features)[:, perm]
    dlnK = means @ PddG.DLNK_PROJ.T
    for i, t in enumerate(dm.STRUCT_TERMS):
        got = np.asarray(tr[f"theta_d_{t}"]["value"])
        np.testing.assert_allclose(got.reshape(-1, data.num_mutation)[0],
                                   dlnK[:, i], atol=1e-5)
    # the non-structural terms get the hierarchical prior
    assert tr["theta_d_offset"]["value"].shape[0] == 2   # Oii H and L
    assert "theta_ddG" in tr and "theta_ddG_prior_mean" in tr


MONOMER_FLAGS = {"theta_monomers": True, "theta_unfolded_monomer": True}


def test_monomer_projection_is_square_and_invertible():
    P = PddG.DLNK_PROJ_MONOMER
    assert P.shape == (10, 10)
    assert np.linalg.matrix_rank(P) == 10
    # the dimer rows are the six-state block, with nothing on the monomers
    np.testing.assert_array_equal(P[:6, :6], PddG.DLNK_PROJ)
    np.testing.assert_array_equal(P[:6, 6:], 0.0)


def test_structure_names_follow_the_variant():
    six = tuple(PddG.STRUCTURE_NAMES)
    assert PddG.structure_names_for() == six
    assert PddG.structure_names_for({"theta_monomers": False}) == six
    ten = six + PddG.MONOMER_STRUCTURE_NAMES
    assert PddG.structure_names_for({"theta_monomers": 1}) == ten
    assert PddG.structure_names_for({"theta.theta_monomers": True}) == ten
    assert PddG.structure_names_for(_priors(PddG, **MONOMER_FLAGS)) == ten
    assert PddG.struct_terms_for(_priors(PddG, **MONOMER_FLAGS)) == (
        dm.STRUCT_TERMS + PddG.MONOMER_TERMS)
    assert PddG.dlnk_proj_for(_priors(PddG)) is PddG.DLNK_PROJ


def test_pddg_monomer_block_at_zero_sigma_is_the_projection():
    data = _for("PddG", _data())
    priors = _priors(PddG, **MONOMER_FLAGS)
    tr = _trace(PddG.define_model, PddG, data, priors,
                cond={"theta_ddG_sigma": jnp.zeros(10)})
    names = PddG.structure_names_for(priors)
    perm = PddG.struct_perm(data, names)
    ddG = np.asarray(data.struct_features)[:, perm]
    dlnK = ddG @ PddG.DLNK_PROJ_MONOMER.T
    M = data.num_mutation
    for i, t in enumerate(PddG.struct_terms_for(priors)):
        got = np.asarray(tr[f"theta_d_{t}"]["value"]).reshape(-1, M)[0]
        np.testing.assert_allclose(got, dlnK[:, i], atol=1e-5)
    # Δln K_d_h = 2 ΔΔG_Hm − ΔΔG_H; Δln K_m_u = ΔΔG_Hm (both U at 0)
    Hm = ddG[:, names.index("H_monomer")]
    np.testing.assert_allclose(np.asarray(tr["theta_d_ln_K_d_h"]["value"]),
                               2 * Hm - ddG[:, 0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(tr["theta_d_ln_K_m_u"]["value"]),
                               Hm, atol=1e-5)
    # only the Oii terms take the hierarchical prior
    assert tr["theta_d_offset"]["value"].shape[0] == 2
    assert tr["theta_ddG"]["value"].shape == (M, 10)
    guide_tr = _trace(PddG.guide, PddG, data, priors)
    model_tr = _trace(PddG.define_model, PddG, data, priors)
    assert _samples(guide_tr) == _samples(model_tr)


def test_monomers_without_the_monomer_states_are_an_error():
    data = _for("PddG", _data())
    data.struct_names = tuple(PddG.STRUCTURE_NAMES)
    data.struct_features = data.struct_features[:, :6]
    _trace(PddG.define_model, PddG, data, _priors(PddG))
    with pytest.raises(ValueError, match="monomer states"):
        _trace(PddG.define_model, PddG, data, _priors(PddG, **MONOMER_FLAGS))


def test_pnnc_keeps_six_states_with_monomers():
    data = _for("PnnC", _data())
    priors = _priors(PnnC, **MONOMER_FLAGS)
    tr = _trace(PnnC.define_model, PnnC, data, priors)
    assert tr["theta_ddG_sigma"]["value"].shape == (6,)
    # the four monomer terms and m_u take the hierarchical prior (+ Oii)
    assert tr["theta_d_offset"]["value"].shape[0] == 7


def test_pnnc_registers_mlp_params_and_uses_contact_distances():
    data = _for("PnnC", _data())
    tr = _trace(PnnC.define_model, PnnC, data, _priors(PnnC))
    assert any(n.startswith("theta_nn_") and s["type"] == "param"
               for n, s in tr.items())
    lam = tr["theta_epi_lambda"]["fn"]
    scale = np.broadcast_to(np.asarray(lam.base_dist.scale), lam.batch_shape)
    d = np.asarray(data.struct_contact_distances)[:, PnnC.struct_perm(data)]
    np.testing.assert_allclose(scale[0], np.exp(-d.min(1) / 8.0), rtol=1e-5)


@pytest.mark.parametrize("key", list(MODULES))
def test_wt_only_library(key):
    module = MODULES[key]
    data = _for(key, _data(genotypes=["wt"], epistasis=False))
    priors = _priors(module)
    for fn in (module.define_model, module.guide):
        param = seed(fn, rng_seed=0)("theta", data, priors)
        th = np.asarray(module.run_model(param, data))
        assert th.shape == (1, len(CONC), 1) and np.all(np.isfinite(th))


def test_run_model_follows_the_batch_index():
    """A batch's θ is the full library's, indexed down (subset contract)."""
    data = _data(T=2)
    priors = _priors(PK, **ALL_ON)
    param = seed(PK.define_model, rng_seed=1)("theta", data, priors)
    full = np.asarray(PK.run_model(param, data))
    sub = types.SimpleNamespace(**{**vars(data),
                                   "batch_idx": jnp.array([4, 0, 2]),
                                   "geno_theta_idx": jnp.array([2, 0, 1, 1])})
    got = np.asarray(PK.run_model(param, sub))
    np.testing.assert_allclose(got, full[:, :, [2, 4, 0, 0]], atol=1e-6)


def _fake_posterior(module, data, priors, n=5):
    sites = {}
    for s in range(n):
        tr = _trace(module.define_model, module, data, priors, seed_val=s)
        for name, site in tr.items():
            if site["type"] in ("sample", "deterministic"):
                sites.setdefault(name, []).append(np.asarray(site["value"]))
    return {k: np.stack(v) for k, v in sites.items()}


@pytest.mark.parametrize("flags", [{}, ALL_ON], ids=["default", "all_on"])
def test_predict_unmeasured_matches_compute_theta_samples(flags):
    data = _data(T=2)
    priors = _priors(PK, **flags)
    post = _fake_posterior(PK, data, priors)
    titrants = ["iptg", "lactose"]
    manual = pd.DataFrame({"titrant_name": np.repeat(titrants, len(CONC)),
                           "titrant_conc": np.tile(CONC, 2)})
    q = {"q0.5": 0.5, "q0.1": 0.1}
    pred = dm.predict_unmeasured(GENOTYPES, titrants, manual, data.mut_labels,
                                 data.pair_labels, post, q, tf_total=1e-5,
                                 conc_unit_scale=1e-3, theta_priors=priors)
    calc = pred[["genotype", "titrant_name", "titrant_conc"]].copy()
    calc["genotype_idx"] = calc["genotype"].map(
        {g: i for i, g in enumerate(GENOTYPES)})
    calc["titrant_name_idx"] = calc["titrant_name"].map(
        {t: i for i, t in enumerate(titrants)})
    samples = dm.compute_theta_samples(calc, post, tf_total=1e-5,
                                       conc_unit_scale=1e-3,
                                       variant=dm.variant_from_priors(priors))
    for name, qv in q.items():
        np.testing.assert_allclose(pred[name].values,
                                   np.quantile(samples, qv, axis=0), atol=2e-5)
    # a genotype with an unknown mutation is NaN
    bad = dm.predict_unmeasured(["Q9R"], titrants, manual, data.mut_labels,
                                data.pair_labels, post, q, tf_total=1e-5,
                                conc_unit_scale=1e-3, theta_priors=priors)
    assert bad["q0.5"].isna().all()


def test_extract_specs_follow_the_variant():
    data = _data(T=1)
    growth_tm = types.SimpleNamespace(
        tensor_dim_names=["genotype", "titrant_name"],
        tensor_dim_labels=[GENOTYPES, ["iptg"]],
        df=pd.DataFrame({"genotype": GENOTYPES,
                         "genotype_idx": range(len(GENOTYPES)),
                         "titrant_name": "iptg", "titrant_name_idx": 0}))
    for flags, has in (({}, False), (ALL_ON, True)):
        ctx = types.SimpleNamespace(growth_tm=growth_tm,
                                    mut_labels=data.mut_labels,
                                    pair_labels=data.pair_labels,
                                    theta_priors=_priors(PK, **flags))
        names = [p for s in PK.get_extract_specs(ctx) for p in s["params_to_get"]]
        assert ("d_ln_K_d_h" in names) == has
        assert "ln_K_h_o_eff" in names and "epi_ln_K_h_e" in names
    ctx.theta_priors = _priors(PddG)
    names = [p for s in PddG.get_extract_specs(ctx)
             for p in s.get("out_names", s["params_to_get"])]
    assert "ddG_H" in names and "ddG_sigma" in names


@pytest.mark.parametrize("key", list(MODULES))
def test_priors_pickle_and_csv_round_trip(key):
    module = MODULES[key]
    priors = _priors(module, theta_monomers=True)
    again = pickle.loads(pickle.dumps(priors))
    import jax
    a, ta = jax.tree_util.tree_flatten(again)
    b, tb = jax.tree_util.tree_flatten(priors)
    assert ta == tb
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    assert dm.variant_from_priors(again).monomers

    flat = configuration_io._extract_scalars(module.get_priors())
    df = pd.DataFrame({"parameter": list(flat), "value": list(flat.values())})
    flat = configuration_io._read_priors_flat(df, None)    # as from the CSV
    flat["theta_monomers"] = "1.0"
    flat["theta_operator_coupling"] = "True"
    back = configuration_io._update_dataclass(module.get_priors(), "", flat)
    v = dm.variant_from_priors(back)
    assert v.monomers and v.operator_coupling and not v.effector_coupling


def test_guesses_name_sites_every_variant_has():
    data = _for("PddG", _data(T=2))
    for module in (PK, PddG):
        g = module.get_guesses("theta", data)
        tr = _trace(module.define_model, module, data,
                    _priors(module, theta_second_engagement=False,
                            theta_unfolded_dimer=False))
        missing = [k for k in g if k not in tr]
        assert missing == []


def test_missing_structures_are_an_error():
    data = _for("PddG", _data())
    data.struct_names = ("H", "HO", "L")
    with pytest.raises(ValueError, match="lacks"):
        PddG.struct_perm(data)


def test_pnnc_guesses_and_extract_specs():
    data = _for("PnnC", _data())
    g = PnnC.get_guesses("theta", data)
    tr = _trace(PnnC.define_model, PnnC, data, _priors(PnnC))
    assert [k for k in g if k not in tr] == []
    assert np.allclose(np.asarray(g["theta_dlnK"]), 0.0)
    growth_tm = types.SimpleNamespace(
        tensor_dim_names=["genotype", "titrant_name"],
        tensor_dim_labels=[GENOTYPES, ["iptg"]],
        df=pd.DataFrame({"genotype": GENOTYPES,
                         "genotype_idx": range(len(GENOTYPES)),
                         "titrant_name": "iptg", "titrant_name_idx": 0}))
    ctx = types.SimpleNamespace(growth_tm=growth_tm, mut_labels=data.mut_labels,
                                pair_labels=data.pair_labels,
                                theta_priors=_priors(PnnC))
    names = [p for s in PnnC.get_extract_specs(ctx)
             for p in s.get("out_names", s["params_to_get"])]
    assert "ddG_HO" in names


def test_run_model_scatter_layout_and_moments():
    data = _data(T=2)
    priors = _priors(PK)
    param = seed(PK.define_model, rng_seed=0)("theta", data, priors)
    full = np.asarray(dm.compute_theta(param.terms(),
                                       data.titrant_conc * 1e-3, 1e-5,
                                       param.variant))
    np.testing.assert_allclose(np.asarray(PK.run_model(param, data)), full,
                               atol=1e-6)
    scattered = types.SimpleNamespace(**{**vars(data), "scatter_theta": 1})
    assert PK.run_model(param, scattered).shape == (1, 1, 1, 1) + full.shape
    assert PK.get_population_moments(param, data) == (None, None)


def test_extract_specs_without_mutations_or_pairs():
    growth_tm = types.SimpleNamespace(
        tensor_dim_names=["genotype", "titrant_name"],
        tensor_dim_labels=[["wt"], ["iptg"]],
        df=pd.DataFrame({"genotype": ["wt"], "genotype_idx": [0],
                         "titrant_name": "iptg", "titrant_name_idx": 0}))
    ctx = types.SimpleNamespace(growth_tm=growth_tm, mut_labels=[],
                                pair_labels=[], theta_priors=None)
    specs = PK.get_extract_specs(ctx)
    names = [p for s in specs for p in s["params_to_get"]]
    assert len(specs) == 2 and not any(n.startswith(("d_", "epi_")) for n in names)


def test_build_calc_df_carries_the_variant():
    df = pd.DataFrame({"genotype": GENOTYPES,
                       "genotype_idx": range(len(GENOTYPES)),
                       "titrant_name": "iptg", "titrant_name_idx": 0,
                       "titrant_conc": 0.1})
    model = types.SimpleNamespace(training_tm=types.SimpleNamespace(df=df),
                                  priors=types.SimpleNamespace(
                                      theta=_priors(PK, theta_monomers=True)))
    calc_df, cols, extra = PK.build_calc_df(model, None)
    assert len(calc_df) == len(GENOTYPES)
    assert extra["variant"].monomers and extra["tf_total"] == pytest.approx(1e-5)


class _H5Like:
    """An HDF5 dataset stand-in: has a shape, no reshape, sliced by [:]."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape

    def __getitem__(self, key):
        return self._arr[key]


def test_predict_unmeasured_without_pairs_reads_dataset_like_posteriors():
    data = _data(T=1, epistasis=False)
    priors = _priors(PK)
    post = {k: _H5Like(v) for k, v in _fake_posterior(PK, data, priors).items()}
    manual = pd.DataFrame({"titrant_name": "iptg", "titrant_conc": CONC})
    pred = dm.predict_unmeasured(["wt", "A1V/C2T/D3K"], ["iptg"], manual,
                                 data.mut_labels, [], post, {"q0.5": 0.5},
                                 tf_total=1e-5, conc_unit_scale=1e-3,
                                 theta_priors=priors)
    assert np.isfinite(pred["q0.5"]).all() and len(pred) == 2 * len(CONC)


# ---------------------------------------------------------------------------
# Held scales (the EM loop of planning/studies/dimer-utility holds them)
# ---------------------------------------------------------------------------

def _samples(tr):
    return {n for n, s in tr.items()
            if s["type"] == "sample" and not s.get("is_observed")}


def test_held_sigma_d_scales_its_offsets_and_drops_from_the_sigma_site():
    data = _data(epistasis=False)
    terms = sp.active_terms(dm.variant_from_priors(_priors(PK)))
    held = {f"theta_sigma_d_{t}_fixed": 0.5 + i for i, t in enumerate(terms)}
    priors = _priors(PK, **held)
    tr = _trace(PK.define_model, PK, data, priors)
    assert "theta_sigma_d" not in tr
    off = np.asarray(tr["theta_d_offset"]["value"])
    for i, t in enumerate(terms):
        d = np.asarray(tr[f"theta_d_{t}"]["value"]).reshape(-1)
        np.testing.assert_allclose(d, off[i] * (0.5 + i), rtol=1e-5)
    guide_tr = _trace(PK.guide, PK, data, priors)
    assert _samples(guide_tr) == _samples(tr)
    assert not any(n.startswith("theta_sigma_d_") for n in guide_tr
                   if guide_tr[n]["type"] == "param")


def test_partly_held_sigma_d_samples_the_free_terms_only():
    data = _data(epistasis=False)
    terms = sp.active_terms(dm.variant_from_priors(_priors(PK)))
    priors = _priors(PK, **{f"theta_sigma_d_{terms[1]}_fixed": 2.0})
    tr = _trace(PK.define_model, PK, data, priors)
    sigma = np.asarray(tr["theta_sigma_d"]["value"])
    assert sigma.shape == (len(terms) - 1,)
    off = np.asarray(tr["theta_d_offset"]["value"])
    d1 = np.asarray(tr[f"theta_d_{terms[1]}"]["value"]).reshape(-1)
    np.testing.assert_allclose(d1, off[1] * 2.0, rtol=1e-5)
    d0 = np.asarray(tr[f"theta_d_{terms[0]}"]["value"]).reshape(-1)
    np.testing.assert_allclose(d0, off[0] * sigma[0], rtol=1e-5)
    guide_tr = _trace(PK.guide, PK, data, priors)
    assert guide_tr["theta_sigma_d"]["value"].shape == sigma.shape


@pytest.mark.parametrize("key", list(MODULES))
def test_held_epi_tau_and_c2_are_deterministic_and_set_the_pair_shifts(key):
    module = MODULES[key]
    data = _for(key, _data())
    priors = _priors(module, theta_epi_tau_fixed=0.05, theta_epi_c2_fixed=4.0)
    tr = _trace(module.define_model, module, data, priors)
    assert tr["theta_epi_tau"]["type"] == "deterministic"
    assert tr["theta_epi_c2"]["type"] == "deterministic"
    lam = np.asarray(tr["theta_epi_lambda"]["value"])
    off = np.asarray(tr["theta_epi_offset"]["value"])
    lt = np.sqrt(4.0 * lam ** 2 / (4.0 + 0.05 ** 2 * lam ** 2))
    epi = off * 0.05 * lt
    terms = sp.active_terms(dm.variant_from_priors(priors))
    for i, t in enumerate(terms):
        np.testing.assert_allclose(np.asarray(tr[f"theta_epi_{t}"]["value"]),
                                   epi[i], rtol=1e-4)
    guide_tr = _trace(module.guide, module, data, priors)
    assert _samples(guide_tr) == _samples(tr)
    assert not any(n.startswith(("theta_epi_tau_", "theta_epi_c2_"))
                   for n in guide_tr if guide_tr[n]["type"] == "param")


def test_held_epi_tau_alone_keeps_c2_sampled():
    data = _data()
    priors = _priors(PK, theta_epi_tau_fixed=0.05)
    tr = _trace(PK.define_model, PK, data, priors)
    assert tr["theta_epi_tau"]["type"] == "deterministic"
    assert tr["theta_epi_c2"]["type"] == "sample"
    guide_tr = _trace(PK.guide, PK, data, priors)
    assert "theta_epi_c2_loc" in guide_tr
    assert "theta_epi_tau_loc" not in guide_tr


def test_held_epi_fields_default_off_and_round_trip_as_static():
    for module in MODULES.values():
        p = module.get_hyperparameters()
        assert p["theta_epi_tau_fixed"] == 0.0
        assert p["theta_epi_c2_fixed"] == 0.0
    tr = _trace(PK.define_model, PK, _data(), _priors(PK))
    assert tr["theta_epi_tau"]["type"] == "sample"
    assert tr["theta_epi_c2"]["type"] == "sample"


@pytest.mark.parametrize("key", ["PddG", "PnnC"])
def test_per_state_held_ddG_sigma(key):
    module = MODULES[key]
    data = _for(key, _data(epistasis=False))
    vals = {s: 0.3 * (i + 1) for i, s in enumerate(module.STRUCTURE_NAMES)}
    priors = _priors(module, **{f"theta_ddG_sigma_{s}_fixed": v
                                for s, v in vals.items()})
    tr = _trace(module.define_model, module, data, priors)
    assert tr["theta_ddG_sigma"]["type"] == "deterministic"
    np.testing.assert_allclose(np.asarray(tr["theta_ddG_sigma"]["value"]),
                               list(vals.values()), rtol=1e-6)
    guide_tr = _trace(module.guide, module, data, priors)
    assert _samples(guide_tr) == _samples(tr)
    assert "theta_ddG_sigma" not in _samples(guide_tr)


def test_per_state_held_ddG_sigma_is_all_or_none():
    first = PddG.STRUCTURE_NAMES[0]
    with pytest.raises(ValueError, match="every state or none"):
        PddG.held_ddG_sigma(_priors(PddG, **{f"theta_ddG_sigma_{first}_fixed": 1.0}))
    both = {f"theta_ddG_sigma_{s}_fixed": 1.0 for s in PddG.STRUCTURE_NAMES}
    with pytest.raises(ValueError, match="not both"):
        PddG.held_ddG_sigma(_priors(PddG, theta_ddG_sigma_fixed=1.0, **both))
    assert PddG.held_ddG_sigma(_priors(PddG, theta_ddG_sigma_fixed=1.0)) == 1.0
    assert PddG.held_ddG_sigma(_priors(PddG, **both)) == (1.0,) * 6
