"""
θ conditions with operator and TF-level contexts (tfmodel/contexts.py,
planning/dimer-thermo-model.md step 3a).
"""
import math
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from numpyro.handlers import seed, trace

from tfscreen.tfmodel import contexts as C
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.scripts.configure_model_cli import _check_contexts_file


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")


def _growth(**cols):
    df = pd.DataFrame({"titrant_name": ["iptg", "iptg", "tmg", "iptg"]})
    for k, v in cols.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# theta_conditions
# ---------------------------------------------------------------------------

def test_without_contexts_the_labels_are_the_titrants():
    c = C.theta_conditions(_growth())
    assert not c.composite
    assert c.growth_labels == ("iptg", "tmg")
    assert c.effector_names == ("iptg", "tmg")
    assert c.growth_effector_idx == (0, 1)
    assert c.operator_names == (C.DEFAULT_LABEL,)
    assert c.tf_level_names == (C.DEFAULT_LABEL,)
    assert c.growth_operator_idx == (0, 0)
    assert c.growth_tf_level_idx == (0, 0)


def test_one_operator_and_level_named_is_not_composite():
    c = C.theta_conditions(_growth(operator="O1", tf_level="low"))
    assert not c.composite
    assert c.growth_labels == ("iptg", "tmg")
    assert c.operator_names == ("O1",)


def test_binding_without_contexts_is_aligned_with_growth():
    # A binding titrant gets the index of the growth condition with that
    # titrant, even when binding measures only some of them.
    b = pd.DataFrame({"titrant_name": ["tmg", "tmg"]})
    c = C.theta_conditions(_growth(), b)
    assert c.binding_labels == ("iptg", "tmg")
    codes = C.assign_theta_condition(b, c, "binding")["theta_condition"]
    assert list(codes.cat.codes) == [1, 1]
    assert c.binding_effector_idx == (0, 1)
    assert all(math.isnan(x) for x in c.binding_tf_total_M)


def test_composite_growth_labels_and_indices():
    g = _growth(operator=["O1", "O2", "O1", "O1"],
                tf_level=["lo", "lo", "lo", "hi"])
    c = C.theta_conditions(g)
    assert c.composite
    assert c.growth_labels == ("iptg|O1|hi", "iptg|O1|lo", "iptg|O2|lo",
                               "tmg|O1|lo")
    assert c.operator_names == ("O1", "O2")
    assert c.tf_level_names == ("hi", "lo")
    assert c.growth_effector_idx == (0, 0, 0, 1)
    assert c.growth_operator_idx == (0, 0, 1, 0)
    assert c.growth_tf_level_idx == (0, 1, 1, 1)
    out = C.assign_theta_condition(g, c, "growth")
    assert list(out["theta_condition"].astype(str)) == [
        "iptg|O1|lo", "iptg|O2|lo", "tmg|O1|lo", "iptg|O1|hi"]


def test_binding_protein_concentration_makes_labels_composite():
    b = pd.DataFrame({"titrant_name": ["iptg", "iptg"],
                      "tf_total_uM": [2.0, 0.5]})
    c = C.theta_conditions(_growth(operator="O1"), b)
    assert c.composite
    assert c.growth_labels == ("iptg|O1|default", "tmg|O1|default")
    # the binding operator defaults to the growth data's only operator
    assert c.binding_labels == ("iptg|O1|0.5uM", "iptg|O1|2uM")
    assert c.binding_tf_total_M == pytest.approx((5e-7, 2e-6))
    assert c.binding_operator_idx == (0, 0)
    fields = c.data_fields("binding")
    assert fields["theta_tf_level_idx"] == (-1, -1)
    assert fields["theta_contexts"] is True


def test_binding_operator_not_in_growth_is_a_name():
    b = pd.DataFrame({"titrant_name": ["iptg"], "operator": ["Oid"]})
    c = C.theta_conditions(_growth(operator="O1"), b)
    assert c.operator_names == ("O1", "Oid")
    assert c.binding_operator_idx == (1,)


def test_binding_needs_operator_when_growth_has_several():
    b = pd.DataFrame({"titrant_name": ["iptg"]})
    with pytest.raises(ValueError, match="needs an 'operator' column"):
        C.theta_conditions(_growth(operator=["O1", "O2", "O1", "O1"]), b)


def test_binding_tf_level_column_is_refused():
    b = pd.DataFrame({"titrant_name": ["iptg"], "tf_level": ["lo"]})
    with pytest.raises(ValueError, match="tf_total_uM"):
        C.theta_conditions(_growth(), b)


@pytest.mark.parametrize("values,match", [
    ([None, "O1", "O1", "O1"], "missing"),
    (["", "O1", "O1", "O1"], "empty"),
    (["O|1", "O1", "O1", "O1"], "may not contain"),
])
def test_bad_labels(values, match):
    with pytest.raises(ValueError, match=match):
        C.theta_conditions(_growth(operator=values))


@pytest.mark.parametrize("conc", [0.0, -1.0, np.nan, "x"])
def test_bad_binding_concentration(conc):
    b = pd.DataFrame({"titrant_name": ["iptg"], "tf_total_uM": [conc]})
    with pytest.raises(ValueError, match="positive concentration"):
        C.theta_conditions(_growth(), b)


def test_needs_titrant_name():
    with pytest.raises(ValueError, match="titrant_name"):
        C.theta_conditions(pd.DataFrame({"operator": ["O1"]}))


def test_assign_refuses_an_unknown_condition():
    c = C.theta_conditions(_growth(operator="O1"))
    other = _growth(operator="O2")
    with pytest.raises(ValueError, match="not among"):
        C.assign_theta_condition(other, c, "growth")


def test_split_theta_condition():
    g = _growth(operator=["O1", "O2", "O1", "O1"], tf_level="lo")
    c = C.theta_conditions(g)
    out = C.split_theta_condition(["tmg|O1|lo", "iptg|O2|lo"], c)
    assert list(out["titrant_name"]) == ["tmg", "iptg"]
    assert list(out["operator"]) == ["O1", "O2"]


# ---------------------------------------------------------------------------
# contexts file
# ---------------------------------------------------------------------------

_SPEC = {"units": "kcal/mol", "temperature_K": 310.15,
         "operators": {"O1": {"reference": True},
                       "O2": {"shift": 1.5, "shift_sd": 0.3}},
         "tf_levels": {"lo": {"conc_uM": 3, "ln_sd": 0.5},
                       "hi": {"conc_uM": 30, "ln_sd": 0.0}}}


def test_read_contexts_converts_units(tmp_path):
    path = tmp_path / "contexts.yaml"
    path.write_text(yaml.safe_dump(_SPEC))
    spec = C.read_contexts(str(path))
    kT = 1.0 / (1.987204e-3 * 310.15)
    assert spec["operators"]["O1"] == {"reference": True, "shift_kT": 0.0,
                                       "shift_sd_kT": 0.0}
    assert spec["operators"]["O2"]["shift_kT"] == pytest.approx(1.5 * kT,
                                                                rel=1e-4)
    assert spec["operators"]["O2"]["shift_sd_kT"] == pytest.approx(0.3 * kT,
                                                                   rel=1e-4)
    assert spec["tf_levels"]["lo"] == {"conc_M": 3e-6, "ln_sd": 0.5}
    assert spec["tf_levels"]["hi"]["ln_sd"] == 0.0


def _with(**changes):
    spec = yaml.safe_load(yaml.safe_dump(_SPEC))
    for k, v in changes.items():
        spec[k] = v
    return spec


@pytest.mark.parametrize("spec,match", [
    (_with(bogus=1), "Unknown contexts keys"),
    (_with(operators={"O1": {"reference": True},
                      "O2": {"reference": True}}), "Exactly one operator"),
    (_with(operators={"O1": {"shift": 1, "shift_sd": 1}}),
     "Exactly one operator"),
    (_with(operators={"O1": {"reference": True}, "O2": {"shift": 1}}),
     "needs 'shift_sd'"),
    (_with(operators={"O1": {"reference": True, "shift": 0}}),
     "is the reference"),
    (_with(operators={"O1": {"reference": True},
                      "O2": {"shift": 1, "shift_sd": -1}}), ">= 0"),
    (_with(operators={"O1": {"reference": True, "color": "red"}}),
     "unknown keys"),
    (_with(tf_levels={"lo": {"conc_uM": 0, "ln_sd": 0.5}}), "> 0"),
    (_with(tf_levels={"lo": {"conc_uM": 3}}), "needs 'ln_sd'"),
    (_with(units="furlongs"), "ddG_units"),
])
def test_read_contexts_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        C.read_contexts(spec)


def _conditions():
    return C.theta_conditions(_growth(operator=["O1", "O2", "O1", "O1"],
                                      tf_level=["lo", "lo", "lo", "hi"]))


def test_check_contexts_passes():
    C.check_contexts(C.read_contexts(_SPEC), _conditions())


def test_check_contexts_requires_a_file_for_several_labels():
    with pytest.raises(ValueError, match="contexts file"):
        C.check_contexts(None, _conditions())
    # one operator, one level: no file needed
    C.check_contexts(None, C.theta_conditions(_growth()))


def test_check_contexts_missing_and_unused_labels():
    spec = C.read_contexts(_with(tf_levels={"lo": {"conc_uM": 3,
                                                   "ln_sd": 0.5}}))
    with pytest.raises(ValueError, match=r"no entry for tf_level \['hi'\]"):
        C.check_contexts(spec, _conditions())
    spec = C.read_contexts(_SPEC)
    single = C.theta_conditions(_growth(operator="O1", tf_level="lo"))
    with pytest.raises(ValueError, match=r"names operator \['O2'\]"):
        C.check_contexts(spec, single)


def test_check_contexts_refuses_partly_labeled_rows():
    b = pd.DataFrame({"titrant_name": ["iptg"], "operator": ["O2"]})
    c = C.theta_conditions(_growth(), b)          # growth rows: "default"
    spec = C.read_contexts(_with(tf_levels={}))
    with pytest.raises(ValueError, match="label every row"):
        C.check_contexts(spec, c)


# ---------------------------------------------------------------------------
# ModelOrchestrator
# ---------------------------------------------------------------------------

def _two_operators(df):
    """The smoke growth data twice, once per operator."""
    a, b = df.copy(), df.copy()
    a["operator"], b["operator"] = "O1", "O2"
    b["sample"] = b["sample"] + "_O2"
    return pd.concat([a, b], ignore_index=True)


def _trace(orchestrator):
    data = orchestrator.get_batch(orchestrator.data,
                                  orchestrator.get_random_idx(batch_key=0))
    return trace(seed(orchestrator.jax_model, 0)).get_trace(
        data=data, priors=orchestrator.priors)


def test_orchestrator_without_contexts_is_unchanged():
    o = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                          batch_size=6)
    assert not o.theta_conditions.composite
    i = o.growth_tm.tensor_dim_names.index("titrant_name")
    assert list(o.growth_tm.tensor_dim_labels[i]) == ["iptg"]
    g, b = o.data.growth, o.data.binding
    assert g.theta_effector_names == ("iptg",)
    assert g.theta_effector_idx == (0,)
    assert g.theta_operator_idx == (0,)
    assert b.theta_effector_idx == (0,)
    assert b.theta_tf_level_idx == (-1,)
    assert g.theta_contexts is False


def test_orchestrator_folds_operators_into_the_titrant_axis():
    growth = _two_operators(pd.read_csv(_GROWTH_CSV))
    o = ModelOrchestrator(growth_df=growth, binding_df=None, batch_size=6)
    i = o.growth_tm.tensor_dim_names.index("titrant_name")
    assert list(o.growth_tm.tensor_dim_labels[i]) == ["iptg|O1|default",
                                                      "iptg|O2|default"]
    assert o.data.growth.num_titrant_name == 2
    assert o.data.growth.theta_operator_idx == (0, 1)
    assert o.data.growth.theta_effector_idx == (0, 0)
    # each operator's tubes are distinct tensor cells: twice the data
    assert int(o.data.growth.good_mask.sum()) == 2 * int(
        ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=None,
                          batch_size=6).data.growth.good_mask.sum())
    sites = _trace(o)
    assert sites["theta_growth_pred"]["value"].shape[-3] == 2


def test_orchestrator_aligns_binding_with_a_subset_of_titrants():
    growth = pd.read_csv(_GROWTH_CSV)
    tmg = growth.copy()
    tmg["titrant_name"] = "tmg"
    tmg["sample"] = tmg["sample"] + "_tmg"
    growth = pd.concat([growth, tmg], ignore_index=True)
    binding = pd.read_csv(_BINDING_CSV).assign(titrant_name="tmg")
    o = ModelOrchestrator(growth_df=growth, binding_df=binding, batch_size=6)
    i = o.binding_tm.tensor_dim_names.index("titrant_name")
    assert list(o.binding_tm.tensor_dim_labels[i]) == ["iptg", "tmg"]
    mask = np.asarray(o.data.binding.good_mask)
    assert not mask[0].any() and mask[1].any()


def test_curve_model_refuses_binding_with_contexts():
    growth = _two_operators(pd.read_csv(_GROWTH_CSV))
    binding = pd.read_csv(_BINDING_CSV).assign(operator="O1")
    with pytest.raises(ValueError, match="cannot link binding data"):
        ModelOrchestrator(growth_df=growth, binding_df=binding, batch_size=6)
    binding = pd.read_csv(_BINDING_CSV).assign(tf_total_uM=2.0)
    with pytest.raises(ValueError, match="cannot link binding data"):
        ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=binding,
                          batch_size=6)


def test_thermo_model_without_context_axis_refuses_contexts():
    growth = _two_operators(pd.read_csv(_GROWTH_CSV))
    with pytest.raises(ValueError, match="no operator or TF-level axis"):
        ModelOrchestrator(growth_df=growth, binding_df=None, batch_size=6,
                          theta="thermo.O2_C12_K5_U1_a.PK")


def test_configure_contexts_file_check(capsys):
    growth = _two_operators(pd.read_csv(_GROWTH_CSV))
    o = ModelOrchestrator(growth_df=growth, binding_df=None, batch_size=6)
    spec = _with(tf_levels={})
    assert _check_contexts_file(o, spec, "hill_geno") is not None
    assert "does not read the contexts file" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"no entry for operator \['O2'\]"):
        _check_contexts_file(o, _with(operators={"O1": {"reference": True},
                                                 "O3": {"shift": 1,
                                                        "shift_sd": 1}},
                                      tf_levels={}), "hill_geno")
    # a curve model needs no file
    assert _check_contexts_file(o, None, "hill_geno") is None


# ---------------------------------------------------------------------------
# Outputs (step 3c)
# ---------------------------------------------------------------------------

from tfscreen.tfmodel.analysis.extraction import (  # noqa: E402
    extract_parameters,
    extract_theta_curves,
    titrant_grid,
    training_keys,
)
from tfscreen.tfmodel.analysis.prediction import copy_orchestrator  # noqa: E402


@pytest.fixture(scope="module")
def curve_model():
    """hill_geno, growth only, two operators: each its own curve."""
    return ModelOrchestrator(growth_df=_two_operators(pd.read_csv(_GROWTH_CSV)),
                             binding_df=None, batch_size=None)


def _prior_draws(o, num=2):
    data = o.get_batch(o.data, o.get_random_idx(batch_key=0))
    out = {}
    for i in range(num):
        tr = trace(seed(o.jax_model, i)).get_trace(data=data, priors=o.priors)
        for k, v in tr.items():
            if v["type"] in ("sample", "deterministic"):
                out.setdefault(k, []).append(np.asarray(v["value"]))
    return {k: np.stack(v) for k, v in out.items()}


def test_curve_model_parameter_tables_carry_the_operator(curve_model):
    params = extract_parameters(curve_model, _prior_draws(curve_model))
    hill = params["hill_n"]
    assert list(hill.columns[:3]) == ["genotype", "titrant_name", "operator"]
    assert set(hill["operator"]) == {"O1", "O2"}
    assert not hill.duplicated(["genotype", "titrant_name", "operator"]).any()


def test_curve_model_theta_curves_carry_the_operator(curve_model):
    out = extract_theta_curves(curve_model, _prior_draws(curve_model),
                               q_to_get=[0.5], num_samples=None)
    assert list(out.columns[:3]) == ["genotype", "titrant_name", "operator"]
    assert not out.duplicated(["genotype", "titrant_name", "operator",
                               "titrant_conc"]).any()


def test_curve_model_refuses_a_manual_grid_with_contexts(curve_model):
    grid = pd.DataFrame({"titrant_name": ["iptg"], "titrant_conc": [0.1]})
    with pytest.raises(ValueError, match="training grid"):
        extract_theta_curves(curve_model, _prior_draws(curve_model),
                             manual_titrant_df=grid)


def test_titrant_grid_and_training_keys(curve_model):
    grid = titrant_grid(curve_model)
    assert list(grid.columns) == ["titrant_name", "operator", "titrant_conc"]
    files = pd.DataFrame({"titrant_name": ["iptg"], "titrant_conc": [0.5]})
    only = titrant_grid(curve_model, files, only_files=True)
    assert sorted(only["operator"]) == ["O1", "O2"]
    assert set(only["titrant_conc"]) == {0.5}
    both = titrant_grid(curve_model, files)
    assert len(both) == len(grid) + 2
    cols, keys = training_keys(curve_model)
    assert cols == ["genotype", "titrant_name", "operator", "titrant_conc"]
    assert len(next(iter(keys))) == 4


def test_without_contexts_the_grid_is_unchanged():
    o = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=None,
                          batch_size=6)
    assert list(titrant_grid(o).columns) == ["titrant_name", "titrant_conc"]
    assert training_keys(o)[0] == ["genotype", "titrant_name", "titrant_conc"]


def test_copy_orchestrator_keeps_the_contexts(curve_model):
    genos = list(curve_model.growth_df["genotype"].astype(str).unique()[:3])
    new = copy_orchestrator(curve_model, genotypes=genos)
    assert new.theta_conditions.growth_labels == \
        curve_model.theta_conditions.growth_labels
    assert new.data.growth.theta_operator_idx == \
        curve_model.data.growth.theta_operator_idx


def test_summarize_theta_join_cols(capsys):
    from tfscreen.tfmodel.scripts.summarize_fit_cli import _theta_join_cols
    base = ["genotype", "titrant_name", "titrant_conc"]
    pred = pd.DataFrame(columns=base + ["operator", "tf_level", "q0.5"])
    truth = pd.DataFrame(columns=base + ["operator", "tf_level", "theta"])
    assert _theta_join_cols(pred, truth) == base + ["operator", "tf_level"]
    plain = pd.DataFrame(columns=base + ["theta"])
    assert _theta_join_cols(pred, plain) == base
    binding = pd.DataFrame(columns=base + ["operator", "tf_total_uM"])
    assert _theta_join_cols(pred, binding) is None
    assert "not compared" in capsys.readouterr().out
