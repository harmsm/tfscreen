"""
The operator and TF-level axis of thermo.dimer (planning/dimer-thermo-model.md
step 3b): each θ condition's effector, operator offset and total TF.
"""
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from numpyro.handlers import seed, substitute, trace

from tfscreen.genetics.build_mut_geno_matrix import (
    build_mut_geno_matrix,
    build_mut_sparse_indices,
)
from tfscreen.tfmodel.analysis.extraction import ExtractionContext
from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.contexts import read_contexts
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import (
    PK,
    model as dm,
    species as sp,
)
from tfscreen.tfmodel.inference.batch_safety import (
    find_orchestrator_batch_dependent_latents,
)
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model


_SMOKE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      *[".."] * 7, "smoke-tests", "test_data")
_GROWTH_CSV = os.path.join(_SMOKE, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE, "binding-smoke.csv")
_LIBRARY_YAML = os.path.join(_SMOKE, "library-smoke.yaml")

GENOTYPES = ["wt", "A1V", "C2T", "A1V/C2T"]
CONC = np.array([0.0, 1e-3, 1e-1, 1.0], dtype=np.float32)    # mM

CONTEXTS = {"units": "kT",
            "operators": {"O1": {"reference": True},
                          "O2": {"shift": 1.5, "shift_sd": 0.0}},
            "tf_levels": {"hi": {"conc_uM": 30.0, "ln_sd": 0.5},
                          "lo": {"conc_uM": 3.0, "ln_sd": 0.0}}}


# ---------------------------------------------------------------------------
# condition_terms on a stub
# ---------------------------------------------------------------------------

def _stub(**fields):
    mut_labels, pair_labels, mgm, pp, pg = build_mut_geno_matrix(
        GENOTYPES, skip_pairs=True)
    mi, mg = build_mut_sparse_indices(mgm)
    G = len(GENOTYPES)
    base = dict(num_titrant_name=1, num_titrant_conc=len(CONC),
                num_genotype=G, titrant_conc=jnp.asarray(CONC),
                batch_idx=jnp.arange(G), geno_theta_idx=jnp.arange(G),
                scatter_theta=0, num_mutation=len(mut_labels), num_pair=0,
                mut_nnz_mut_idx=mi, mut_nnz_geno_idx=mg,
                pair_nnz_pair_idx=pp, pair_nnz_geno_idx=pg,
                theta_effector_names=("iptg", "tmg"),
                theta_operator_names=("O1", "O2"),
                theta_tf_level_names=("hi", "lo"))
    base.update(fields)
    return types.SimpleNamespace(**base)


def _param(rng=0):
    """A drawn ThetaParam over two effectors, two operators, two levels."""
    data = _stub()
    priors = PK.get_priors(data=data, contexts=read_contexts(CONTEXTS))
    with seed(rng_seed=rng):
        return dm.define("theta", data, priors), priors


def test_conditions_match_separate_evaluations():
    # (effector, operator, level / known TF) per θ condition
    conds = [(0, 0, 1, np.nan), (1, 1, 0, np.nan), (0, 1, -1, 2e-6),
             (1, 0, -1, np.nan)]
    data = _stub(num_titrant_name=len(conds), theta_contexts=True,
                 theta_effector_idx=tuple(c[0] for c in conds),
                 theta_operator_idx=tuple(c[1] for c in conds),
                 theta_tf_level_idx=tuple(c[2] for c in conds),
                 theta_tf_total_M=tuple(c[3] for c in conds))
    tp, priors = _param()
    theta = np.asarray(dm.run_model(tp, data))           # (T, C, G)
    tt = set(sp.titrant_terms(tp.variant))
    for t, (e, o, lvl, known) in enumerate(conds):
        terms = {}
        for k, v in tp.terms().items():
            v = v[e] if k in tt else v
            if k in tp.operator_terms:
                v = v + tp.operator_offset[o, tp.operator_terms.index(k)]
            terms[k] = v[None, :] if k in tt else v
        tf = (known if np.isfinite(known) else
              float(np.exp(tp.ln_tf_total[lvl])) if lvl >= 0 else tp.tf_total)
        want = np.asarray(dm.compute_theta(terms, CONC * 1e-3, tf,
                                           tp.variant))[0]
        np.testing.assert_allclose(theta[t], want, rtol=1e-5, atol=1e-7)


def test_held_values_and_reference():
    tp, priors = _param()
    off = np.asarray(tp.operator_offset)
    assert np.all(off[0] == 0.0)                     # O1 is the reference
    # O2: shift held at 1.5 kT on the first-engagement terms, plus a
    # deviation on every operator term
    tr = trace(seed(dm.define, 3)).get_trace("theta", _stub(), priors)
    dev = np.asarray(tr["theta_operator_dev"]["value"])
    shift = np.asarray(tr["theta_operator_shift"]["value"])
    assert shift[0] == 0.0 and shift[1] == pytest.approx(1.5)
    assert np.all(dev[0] == 0.0)
    ln_tf = np.asarray(tr["theta_ln_tf_total"]["value"])
    assert ln_tf[1] == pytest.approx(np.log(3e-6))   # lo held
    assert ln_tf[0] != pytest.approx(np.log(30e-6))  # hi drawn about its loc
    assert tr["theta_operator_shift_z"]["value"].shape == (1,)
    assert tr["theta_operator_dev_z"]["value"].shape == (
        len(tp.operator_terms), 1)


def test_guide_has_the_same_context_sites():
    data = _stub()
    priors = PK.get_priors(data=data, contexts=read_contexts(CONTEXTS))
    m = trace(seed(dm.define, 0)).get_trace("theta", data, priors)
    g = trace(seed(dm.define, 0)).get_trace("theta", data, priors, guide=True)
    want = {"theta_operator_shift_z", "theta_operator_dev_z",
            "theta_tf_level_z"}
    for tr in (m, g):
        assert want <= {k for k, v in tr.items() if v["type"] == "sample"}
    params = {k for k, v in g.items() if v["type"] == "param"}
    assert {"theta_operator_shift_z_loc", "theta_tf_level_z_locs",
            "theta_operator_dev_z_locs"} <= params


def test_effector_terms_are_indexed_by_effector():
    data = _stub(num_titrant_name=5)
    tp, _ = _param()
    assert tp.ln_K_h_e.shape == (2, len(GENOTYPES))
    assert dm.num_effectors(data) == 2
    # data without θ-condition fields: one effector per titrant
    plain = _stub(num_titrant_name=3)
    del plain.theta_effector_names
    assert dm.num_effectors(plain) == 3


def test_without_labels_there_are_no_context_sites():
    data = _stub(theta_operator_names=("default",),
                 theta_tf_level_names=("default",),
                 theta_effector_names=("iptg",))
    priors = PK.get_priors(data=data)
    assert priors.theta_operator_shift_loc.shape == (0,)
    tr = trace(seed(dm.define, 0)).get_trace("theta", data, priors)
    assert not [k for k in tr if "operator" in k or "tf_level" in k]


def test_gradient_through_tf_level_is_finite():
    data = _stub(num_titrant_name=2, theta_contexts=True,
                 theta_effector_idx=(0, 1), theta_operator_idx=(0, 1),
                 theta_tf_level_idx=(0, 1),
                 theta_tf_total_M=(np.nan, np.nan))
    tp, _ = _param()

    def f(ln_tf):
        return jnp.sum(dm.run_model(tp.replace(ln_tf_total=ln_tf), data))

    g = jax.grad(f)(tp.ln_tf_total)
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0)


def test_context_hyperparameters_defaults():
    data = _stub()
    p = dm.context_hyperparameters(data)
    assert p["theta_operator_reference"] == "O1"
    np.testing.assert_allclose(p["theta_operator_shift_scale"],
                               [0.0, dm.DEFAULT_OPERATOR_SHIFT_SD])
    np.testing.assert_allclose(p["theta_tf_level_ln_conc_scale"],
                               [dm.DEFAULT_TF_LEVEL_LN_SD] * 2)
    one = _stub(theta_tf_level_names=("lo",))
    assert float(dm.context_hyperparameters(one)[
        "theta_tf_level_ln_conc_scale"][0]) == 0.0


def test_mismatched_operator_priors_are_refused():
    data = _stub()
    priors = PK.get_priors(data=data, contexts=read_contexts(CONTEXTS))
    other = _stub(theta_operator_names=("O1", "O2", "O3"))
    with pytest.raises(ValueError, match="2 operator"):
        trace(seed(dm.define, 0)).get_trace("theta", other, priors)


# ---------------------------------------------------------------------------
# The orchestrator and configure
# ---------------------------------------------------------------------------

def _context_growth():
    g = pd.read_csv(_GROWTH_CSV)
    parts = []
    for op, lvl in (("O1", "hi"), ("O2", "lo"), ("O1", "lo")):
        d = g.copy()
        d["operator"], d["tf_level"] = op, lvl
        d["sample"] = d["sample"] + f"_{op}_{lvl}"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _context_binding():
    return pd.read_csv(_BINDING_CSV).assign(operator="O1", tf_total_uM=2.0)


@pytest.fixture(scope="module")
def context_model():
    return ModelOrchestrator(growth_df=_context_growth(),
                             binding_df=_context_binding(), batch_size=6,
                             theta="thermo.dimer.PK", contexts=CONTEXTS)


def _model_trace(o):
    data = o.get_batch(o.data, o.get_random_idx(batch_key=0))
    return trace(seed(o.jax_model, 0)).get_trace(data=data, priors=o.priors)


def test_orchestrator_context_model_traces(context_model):
    o = context_model
    assert o.data.growth.num_titrant_name == 3
    assert o.priors.theta.theta_operator_reference == "O1"
    np.testing.assert_allclose(o.priors.theta.theta_tf_level_ln_conc_loc,
                               np.log([30e-6, 3e-6]))
    tr = _model_trace(o)
    for site in ("theta_growth_pred", "theta_binding_pred"):
        assert np.all(np.isfinite(np.asarray(tr[site]["value"])))
    assert o.data.binding.theta_tf_total_M == pytest.approx((2e-6,))


def test_binding_uses_its_known_protein_concentration(context_model):
    o = context_model
    data = o.get_batch(o.data, o.get_random_idx(batch_key=0))
    tr = trace(seed(o.jax_model, 0)).get_trace(data=data, priors=o.priors)
    params = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}
    bump = dict(params)
    bump["theta_tf_level_z"] = params["theta_tf_level_z"] + 3.0
    a = trace(substitute(seed(o.jax_model, 0), params)).get_trace(
        data=data, priors=o.priors)
    b = trace(substitute(seed(o.jax_model, 0), bump)).get_trace(
        data=data, priors=o.priors)
    np.testing.assert_allclose(a["theta_binding_pred"]["value"],
                               b["theta_binding_pred"]["value"])
    assert not np.allclose(a["theta_growth_pred"]["value"],
                           b["theta_growth_pred"]["value"])


def test_context_model_is_batch_safe(context_model):
    assert find_orchestrator_batch_dependent_latents(context_model) == {}


def test_extraction_tables(context_model):
    o = context_model
    ctx = ExtractionContext(growth_tm=o.training_tm, mut_labels=o.mut_labels,
                            pair_labels=o.pair_labels,
                            growth_shares_replicates=False,
                            theta_priors=o.priors.theta,
                            theta_conditions=o.theta_conditions)
    specs = dm.get_extract_specs(ctx)
    by_param = {p: s for s in specs for p in s["params_to_get"]}
    assert list(by_param["operator_shift"]["input_df"]["operator"]) == [
        "O1", "O2"]
    assert list(by_param["ln_tf_total"]["input_df"]["tf_level"]) == [
        "hi", "lo"]
    ke = by_param["ln_K_h_e"]["input_df"]
    assert set(ke["titrant_name"]) == {"iptg"}       # effectors, not conditions


def test_configure_writes_context_priors_and_fit_reads_them(tmp_path):
    growth = tmp_path / "growth.csv"
    _context_growth().to_csv(growth, index=False)
    binding = tmp_path / "binding.csv"
    _context_binding().to_csv(binding, index=False)
    contexts = tmp_path / "contexts.yaml"
    contexts.write_text(yaml.safe_dump(CONTEXTS))
    out_prefix = str(tmp_path / "tfs")
    configure_model(growth_df=str(growth), binding_df=str(binding),
                    library_config=_LIBRARY_YAML, theta_model="thermo.dimer.PK",
                    contexts=str(contexts), out_prefix=out_prefix,
                    skip_model_stats=True)
    priors = pd.read_csv(f"{out_prefix}_priors.csv")
    rows = priors[priors["parameter"] == "theta.theta_operator_shift_loc"]
    assert list(rows["operator"]) == ["O1", "O2"]
    assert [float(v) for v in rows["value"]] == pytest.approx([0.0, 1.5])
    ref = priors[priors["parameter"] == "theta.theta_operator_reference"]
    assert ref["value"].iloc[0] == "O1"
    with open(f"{out_prefix}_config.yaml") as fh:
        assert "contexts" not in fh.read()

    # an edit to the CSV is what the fit reads
    rows_idx = rows.index[rows["operator"] == "O2"]
    priors.loc[rows_idx, "value"] = 2.25
    priors.to_csv(f"{out_prefix}_priors.csv", index=False)
    o, _ = read_configuration(f"{out_prefix}_config.yaml")
    np.testing.assert_allclose(o.priors.theta.theta_operator_shift_loc,
                               [0.0, 2.25])
    assert o.priors.theta.theta_operator_reference == "O1"
    np.testing.assert_allclose(o.priors.theta.theta_tf_level_ln_conc_scale,
                               [0.5, 0.0])


def test_configure_requires_contexts_for_several_operators(tmp_path):
    growth = tmp_path / "growth.csv"
    _context_growth().to_csv(growth, index=False)
    with pytest.raises(ValueError, match="contexts file"):
        configure_model(growth_df=str(growth), library_config=_LIBRARY_YAML,
                        theta_model="thermo.dimer.PK",
                        out_prefix=str(tmp_path / "tfs"),
                        skip_model_stats=True)


def test_stale_priors_rows_are_refused(tmp_path):
    growth = tmp_path / "growth.csv"
    _context_growth().to_csv(growth, index=False)
    contexts = tmp_path / "contexts.yaml"
    contexts.write_text(yaml.safe_dump(CONTEXTS))
    out_prefix = str(tmp_path / "tfs")
    configure_model(growth_df=str(growth), library_config=_LIBRARY_YAML,
                    theta_model="thermo.dimer.PK", contexts=str(contexts),
                    out_prefix=out_prefix, skip_model_stats=True)
    priors = pd.read_csv(f"{out_prefix}_priors.csv")
    priors.loc[priors["operator"] == "O2", "operator"] = "O9"
    priors.to_csv(f"{out_prefix}_priors.csv", index=False)
    with pytest.raises(ValueError, match="O9"):
        read_configuration(f"{out_prefix}_config.yaml")


def test_theta_prediction_with_contexts_waits_for_3c(context_model):
    with pytest.raises(NotImplementedError, match="step 3c"):
        dm.build_calc_df(context_model, None)
