"""
Operators, TF levels and binding protein concentrations in the simulator
(planning/dimer-thermo-model.md step 3e): the design's context columns fold
into θ conditions as in the fit, a thermo.dimer draw takes the contexts
file as its prior, and the truth carries the context columns and tables.
"""
import warnings

import jax
import numpy as np
import pandas as pd
import pytest

from tfscreen.simulate.build_sample_dataframes import (
    build_sample_dataframes,
    design_conditions,
    design_keys,
    read_design,
)
from tfscreen.simulate.library_prediction import (
    binding_condition_frame,
    check_sim_contexts,
)
from tfscreen.simulate.sample_theta import theta_columns
from tfscreen.simulate.sim_data_class import (
    binding_sim_data,
    build_sim_data,
    sim_theta_conditions,
)
from tfscreen.simulate.thermo_to_growth import thermo_to_growth
from tfscreen.tfmodel.contexts import assign_theta_condition

GENOTYPES = ["wt", "H29A", "L62A", "H29A/L62A"]
GROWTH = {"pre": {"m": 0.0, "b": 0.01}, "sel": {"m": 0.01, "b": 0.01}}
CONCS = [0.0, 0.01, 1.0]
DIMER = "thermo.dimer.PK"

# Shifts held exactly (SD 0) so the truth is the file's value.
CONTEXTS = {
    "units": "kT",
    "operators": {"O1": {"reference": True},
                  "O2": {"shift": 3.0, "shift_sd": 0.0}},
    "tf_levels": {"low": {"conc_uM": 2.0, "ln_sd": 0.0},
                  "high": {"conc_uM": 20.0, "ln_sd": 0.0}},
}


def _sample_df(operators=("O1", "O2"), levels=("low", "high")):
    rows = [{"titrant_name": "iptg", "operator": o, "tf_level": t,
             "titrant_conc": c, "condition_pre": "pre",
             "condition_sel": "sel"}
            for o in operators for t in levels for c in CONCS]
    return pd.DataFrame(rows)


def _draw(sample_df, component=DIMER, contexts=CONTEXTS, binding=None,
          seed=0, **kw):
    lib = pd.DataFrame({"genotype": GENOTYPES})
    conditions = sim_theta_conditions(sample_df, binding)
    sim_data = build_sim_data(lib, sample_df, conditions=conditions)
    extra = {}
    if binding is not None:
        label = assign_theta_condition(binding, conditions, "binding")
        extra = dict(binding_sim_data=binding_sim_data(
            sim_data, conditions, str(label["theta_condition"].iloc[0])),
            binding_titrant_conc=CONCS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = thermo_to_growth(lib["genotype"], sim_data, sample_df, component,
                               jax.random.PRNGKey(seed), GROWTH,
                               rng=np.random.default_rng(seed),
                               return_mutation_truth=True, contexts=contexts,
                               **extra, **kw)
    return sim_data, out


def _theta(genotype_theta_df, genotype, operator, level):
    df = genotype_theta_df
    sel = ((df["genotype"].astype(str) == genotype)
           & (df["operator"] == operator) & (df["tf_level"] == level))
    return df.loc[sel].sort_values("titrant_conc")["theta"].to_numpy()


# ---------------------------------------------------------------------------
# Design tables
# ---------------------------------------------------------------------------

def _block(**extra):
    return {"library": "kanR", "titrant_name": "iptg", "titrant_conc": [0, 1],
            "condition_pre": "pre", "t_pre": 30, "condition_sel": "sel",
            "t_sel": [60], **extra}


def test_blocks_carry_context_columns():
    df = build_sample_dataframes([_block(operator="O1", tf_level="low"),
                                  _block(operator="O2", tf_level="low")])
    assert set(df["operator"]) == {"O1", "O2"}
    assert (df["tf_level"] == "low").all()
    assert "background" not in df.columns
    plain = build_sample_dataframes([_block()])
    assert "operator" not in plain.columns


def test_blocks_give_a_context_in_all_or_none():
    with pytest.raises(ValueError, match="every block or none"):
        build_sample_dataframes([_block(operator="O1"), _block()])


def test_design_keeps_context_columns(tmp_path):
    rows = []
    for i, op in enumerate(["O1", "O2"]):
        rows.append({"sample": f"s{i}", "library": "kanR", "replicate": 1,
                     "condition_pre": "pre", "t_pre": 30,
                     "condition_sel": "sel", "t_sel": 60,
                     "titrant_name": "iptg", "titrant_conc": 1.0,
                     "operator": op, "tf_level": "low"})
    path = tmp_path / "design.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    design = read_design(str(path))
    # two tubes alike but for the operator are two conditions
    assert design_keys(design)[-2:] == ["operator", "tf_level"]
    assert len(design_conditions(design)) == 2


# ---------------------------------------------------------------------------
# θ conditions
# ---------------------------------------------------------------------------

def test_plain_design_has_one_condition():
    lib = pd.DataFrame({"genotype": GENOTYPES})
    sample_df = _sample_df().drop(columns=["operator", "tf_level"])
    sim_data = build_sim_data(lib, sample_df)
    assert sim_data.num_titrant_name == 1
    assert sim_data.theta_condition_labels == ("iptg",)
    assert sim_data.theta_contexts is False


def test_theta_columns_are_condition_major():
    theta = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)   # (T, C, G)
    gc = theta_columns(theta)
    assert gc.shape == (4, 6)
    np.testing.assert_array_equal(gc[1, 3:], theta[1, :, 1])


def test_contexts_reach_the_draw_and_the_truth():
    sim_data, (pheno, theta_df, params, truth) = _draw(_sample_df())
    assert sim_data.num_titrant_name == 4 and sim_data.theta_contexts
    assert {"operator", "tf_level"} <= set(theta_df.columns)
    assert {"operator", "tf_level"} <= set(pheno.columns)
    assert len(theta_df) == len(GENOTYPES) * 4 * len(CONCS)

    shift = truth["operator_shift"].set_index("operator")["operator_shift"]
    assert shift["O1"] == 0.0 and shift["O2"] == pytest.approx(3.0)
    ln_tf = truth["ln_tf_total"].set_index("tf_level")["ln_tf_total"]
    assert ln_tf["low"] == pytest.approx(np.log(2e-6))
    assert ln_tf["high"] == pytest.approx(np.log(20e-6))
    dev = truth["operator_dev"]
    assert set(dev.columns) == {"operator", "term", "operator_dev"}
    assert (dev.loc[dev["operator"] == "O1", "operator_dev"] == 0).all()

    # a weaker operator and less TF both lower repression without effector
    o1_low = _theta(theta_df, "wt", "O1", "low")
    o2_low = _theta(theta_df, "wt", "O2", "low")
    o1_high = _theta(theta_df, "wt", "O1", "high")
    assert o2_low[0] < o1_low[0] < o1_high[0]

    # each tube's theta is its own condition's
    tube = pheno[(pheno["genotype"].astype(str) == "wt")
                 & (pheno["operator"] == "O2") & (pheno["tf_level"] == "low")]
    np.testing.assert_allclose(tube.sort_values("titrant_conc")["theta"],
                               o2_low)


def test_binding_at_its_own_protein_concentration():
    binding = binding_condition_frame({"titrant_name": "iptg",
                                       "operator": "O1",
                                       "tf_total_uM": 2.0})
    _, out = _draw(_sample_df(), binding=binding)
    theta_df, binding_theta = out[1], out[-1]
    # in vitro at the low level's concentration and the reference operator
    # is the (O1, low) growth curve
    got = (binding_theta[binding_theta["genotype"] == "wt"]
           .sort_values("titrant_conc")["theta"].to_numpy())
    np.testing.assert_allclose(got, _theta(theta_df, "wt", "O1", "low"),
                               rtol=1e-5, atol=1e-6)


def test_o2_model_refuses_contexts():
    with pytest.raises(ValueError, match="no operator or TF-level axis"):
        _draw(_sample_df(), component="thermo.O2_C12_K5_U1_a.PK")


def test_curve_model_draws_every_context():
    _, (_, theta_df, _, _) = _draw(_sample_df(), component="hill_geno",
                                   contexts=None)
    assert len(theta_df) == len(GENOTYPES) * 4 * len(CONCS)


def test_sim_contexts_are_checked_like_the_fit():
    conditions = sim_theta_conditions(_sample_df())
    with pytest.raises(ValueError, match="contexts file"):
        check_sim_contexts({"theta_component": DIMER}, conditions)
    assert check_sim_contexts({"theta_component": DIMER,
                               "contexts": CONTEXTS}, conditions) is not None
    with pytest.raises(ValueError, match="needs an 'operator' column"):
        sim_theta_conditions(_sample_df(), binding_condition_frame(
            {"titrant_name": "iptg", "tf_total_uM": 2.0}))
    binding = binding_condition_frame({"titrant_name": "iptg",
                                       "operator": "O1",
                                       "tf_total_uM": 2.0})
    conditions = sim_theta_conditions(_sample_df(), binding)
    with pytest.raises(ValueError, match="cannot link binding"):
        check_sim_contexts({"theta_component": "hill_geno",
                            "contexts": CONTEXTS}, conditions, binding)
