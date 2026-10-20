"""
tfs-fit-two-stage: the YAML, the step order, skipping and what each step
hands the chained CLI functions (mocked here; the chain itself is checked
end to end in planning/two-stage-fit.md's validation step).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from unittest.mock import patch

from tfscreen.tfmodel.scripts import fit_two_stage_cli as cli

CONDS = ["kanR+kan", "kanR-kan", "pheS+4CP", "pheS-4CP"]


def _growth(path):
    rows = []
    for lib, sel, pre in (("kanR", "kanR+kan", "kanR-kan"),
                          ("kanR", "kanR-kan", "kanR-kan"),
                          ("pheS", "pheS+4CP", "pheS-4CP"),
                          ("pheS", "pheS-4CP", "pheS-4CP")):
        for x in (0.0, 1.0):
            for r in (1, 2):
                for t in (100, 150, 200):
                    s = f"{sel}-{x}-{r}-{t}"
                    for g in ("wt", "A1B", "C2D", "A1B/C2D"):
                        rows.append({"sample": s, "library": lib,
                                     "condition_pre": pre, "condition_sel": sel,
                                     "titrant_name": "iptg", "titrant_conc": x,
                                     "replicate": r, "t_sel": t,
                                     "genotype": g, "counts": 100})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_yaml(tmp_path, **over):
    _growth(tmp_path / "growth.csv")
    (tmp_path / "lib.yaml").write_text("placeholder: 1\n")
    cfg = {"growth_df": "growth.csv", "library_config": "lib.yaml",
           "theta_model": "hill_relative", "growth_shares_replicates": True,
           "set_priors": ["sigma_fixed=0.17"],
           "two_stage": {"transformation_lambda": [0.38, 0.15],
                         "jackknife_subsamples": 3,
                         "num_posterior_samples": 20}}
    for k, v in over.items():
        if k == "two_stage":
            cfg["two_stage"].update(v)
        elif v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    p = tmp_path / "two_stage.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


# --- the YAML ---

def test_read_config_resolves_paths_and_defaults(tmp_path):
    cfg, st = cli.read_two_stage_config(_write_yaml(tmp_path))
    assert cfg["growth_df"] == str(tmp_path / "growth.csv")
    assert st["jackknife_subsamples"] == 3
    assert st["jackknife_drop_frac"] == 0.2
    assert st["consistency"] is True
    assert st["transformation_lambda"] == [0.38, 0.15]


@pytest.mark.parametrize("over,match", [
    ({"two_stage": {"transformation_lambda": None}}, "transformation_lambda"),
    ({"theta_model": "hill_geno"}, "hill_relative"),
    ({"growth_shares_replicates": False}, "growth_shares_replicates"),
    ({"transformation_model": "mixture"}, "set by the two-stage fit"),
    ({"not_an_arg": 1}, "Unknown key"),
    ({"two_stage": {"bogus": 1}}, "Unknown two_stage"),
    ({"set_priors": ["m_pinned=1"]}, "pins k and m"),
    ({"growth_df": None}, "needs growth_df"),
    ({"growth_df": "missing.csv"}, "does not exist"),
    ({"two_stage": {"fit": {"seed": 3}}}, "cannot be set"),
    ({"two_stage": {"jackknife_subsamples": 1}}, "at least 2"),
    ({"binding_df": "growth.csv"}, "growth-only"),
    ({"theta_binding_noise_model": "genotype_offset"}, "experimental"),
    ({"theta_growth_noise_model": "beta"}, "experimental"),
    ({"set_priors": ["theta_growth_log_conc_shift_scale=1"]}, "experimental"),
    ({"set_priors": ["offset_scale_fixed=0.1"]}, "experimental"),
    ({"set_priors": ["warp_log_a_scale=1"]}, "experimental"),
])
def test_read_config_refuses(tmp_path, over, match):
    with pytest.raises(ValueError, match=match):
        cli.read_two_stage_config(_write_yaml(tmp_path, **over))


# --- the chain, with the CLI functions mocked ---

class Recorder:
    """Stand-ins for the chained CLI functions; each writes its outputs."""

    def __init__(self, km=None, x=None):
        self.calls = []
        # X by refit directory; anything else (stage 2, consistency) 0.43
        self.x = x or {"plus": 0.4, "minus": 0.46}
        self.km = km or {c: (0.01 * (i + 1), (-1) ** i * 0.014)
                         for i, c in enumerate(CONDS)}

    def configure_model(self, out_prefix, **kw):
        self.calls.append(("configure", os.getcwd(), kw))
        open(f"{out_prefix}_config.yaml", "w").write("x: 1\n")
        pd.DataFrame({"parameter": ["growth.dk_geno.hyper_loc_loc"],
                      "value": [-3.5]}).to_csv(f"{out_prefix}_priors.csv",
                                               index=False)

    def fit_model(self, config_file, **kw):
        self.calls.append(("fit", os.getcwd(), kw))
        np.savez(cli.PARAMS, dk_geno_hyper_loc_auto_loc=np.array(-2.17),
                 dk_geno_hyper_scale_auto_loc=np.array(1.53),
                 dk_geno_hyper_shift_auto_loc=np.array(0.076))
        open(cli.CHECKPOINT, "w").write("x")

    def extract_params(self, config_file, param_file, out_prefix):
        self.calls.append(("extract", os.getcwd(), param_file))
        rng = np.random.default_rng(len(self.calls))
        k = {c: v[0] + rng.normal(0, 0.001) for c, v in self.km.items()}
        pd.DataFrame({"condition_rep": list(k), "q0.5": list(k.values())}
                     ).to_csv(f"{out_prefix}_growth_k.csv", index=False)
        pd.DataFrame({"condition_rep": list(self.km),
                      "q0.5": [v[1] for v in self.km.values()]}
                     ).to_csv(f"{out_prefix}_growth_m.csv", index=False)

    def sample_posterior(self, config_file, checkpoint_file, **kw):
        self.calls.append(("posterior", os.getcwd(), kw))
        open(cli.POSTERIOR, "w").write("x")

    def predict_theta(self, config_file, param_file, out_prefix, **kw):
        self.calls.append(("predict", os.getcwd(), param_file, kw))
        here = os.path.basename(os.getcwd())
        x = self.x.get(here, 0.43)
        rows = [{"genotype": g, "titrant_name": "iptg", "titrant_conc": c,
                 "q0.025": x - 0.05, "q0.5": x if g != "wt" else 1 - c,
                 "q0.975": x + 0.05}
                for g in ("wt", "A1B", "C2D", "A1B/C2D") for c in (0.0, 1.0)]
        pd.DataFrame(rows).to_csv(f"{out_prefix}.csv", index=False)

    def predict_epistasis(self, config_file, param_file, out_prefix, **kw):
        self.calls.append(("epistasis", os.getcwd(), param_file, kw))
        open(f"{out_prefix}.csv", "w").write("x\n")


@pytest.fixture
def rec():
    r = Recorder()
    lib = pd.DataFrame({"genotype": ["wt", "A1B", "C2D", "A1B/C2D"],
                        "in_spiked_origin": [True, False, False, False]})
    with patch.object(cli, "configure_model", r.configure_model), \
         patch.object(cli, "fit_model", r.fit_model), \
         patch.object(cli, "extract_params", r.extract_params), \
         patch.object(cli, "sample_posterior", r.sample_posterior), \
         patch.object(cli, "predict_theta", r.predict_theta), \
         patch.object(cli, "predict_epistasis", r.predict_epistasis), \
         patch.object(cli, "library_composition_table", return_value=lib):
        yield r


def _calls(rec, kind, where=None):
    return [c for c in rec.calls if c[0] == kind
            and (where is None or c[1].endswith(where))]


def test_full_run_layout_and_arguments(tmp_path, rec):
    out = tmp_path / "run"
    cli.fit_two_stage(_write_yaml(tmp_path, two_stage={"kshift": True}),
                      out_dir=str(out))
    for f in ("stage1/km.csv", "jackknife/jk_01/km.csv", "jackknife/k_step.csv",
              "jackknife/jackknife.json", "stage2/tfs_pred_theta.csv",
              "kshift/shift.csv", "predict/tfs_pred_theta.csv",
              "consistency/consistency.csv", "two_stage_config.yaml",
              "two_stage_provenance.json", "tfs_x.csv",
              "genotype_reads.csv"):
        assert (out / f).exists(), f

    # stage 1: only wt and singles, the congression mixture with max
    s1 = pd.read_csv(out / "stage1" / "tfs_growth.csv")
    assert set(s1["genotype"]) == {"wt", "A1B", "C2D"}
    conf1 = _calls(rec, "configure", "stage1")[0][2]
    assert conf1["transformation_model"] == "mixture"
    assert conf1["congression_theta_rule"] == "max"
    assert conf1["transformation_lambda"] == [0.38, 0.15]

    # stage 2: the full table, single, every condition pinned
    conf2 = _calls(rec, "configure", "stage2")[0][2]
    assert "transformation_model" not in conf2
    assert conf2["growth_df"] == str(tmp_path / "growth.csv")
    pins = pd.read_csv(out / "stage2" / "growth_priors_pinned.csv")
    assert set(pins["condition_rep"]) == set(CONDS)
    assert (pins["k_pinned"] == 1).all() and (pins["m_pinned"] == 1).all()

    # jackknife: subsamples of stage 1 with tubes dropped
    info = json.load(open(out / "jackknife" / "jk_02" / "subsample.json"))
    assert info["d"] > 0 and info["n"] == s1["sample"].nunique()

    # kshift: three refits from stage 2's MAP at the small step size, the
    # dk_geno population held at stage 2's MAP; k moved, m kept
    for v in ("plus", "minus", "zero"):
        fit = _calls(rec, "fit", f"kshift/{v}")[0][2]
        assert fit["init_from"] == str(out / "stage2" / cli.PARAMS)
        assert fit["adam_step_size"] == 1e-5
        assert fit["adam_final_step_size"] == 1e-6
        pri = pd.read_csv(out / "kshift" / v / "tfs_configure_priors.csv")
        held = dict(zip(pri["parameter"], pri["value"].astype(float)))
        assert held["growth.dk_geno.pinned.hyper_loc"] == pytest.approx(-2.17)
        assert held["growth.dk_geno.pinned.hyper_scale"] == pytest.approx(1.53)
        assert held["growth.dk_geno.pinned.hyper_shift"] == pytest.approx(0.076)
    p = pd.read_csv(out / "kshift" / "plus" / "growth_priors_pinned.csv")
    m = pd.read_csv(out / "kshift" / "minus" / "growth_priors_pinned.csv")
    z = pd.read_csv(out / "kshift" / "zero" / "growth_priors_pinned.csv")
    s2 = pd.read_csv(out / "stage2" / "growth_priors_pinned.csv")
    assert (p["k_loc"] != m["k_loc"]).all()
    assert np.allclose(p["m_loc"], m["m_loc"])
    assert np.allclose(z["k_loc"], s2["k_loc"])
    assert (out / "stage2" / cli.PRED_MAP).exists()
    shift = pd.read_csv(out / "kshift" / "shift.csv")
    assert np.allclose(shift.loc[shift.genotype != "wt", "shift"], -0.03)
    drift = pd.read_csv(out / "kshift" / "drift.csv").set_index("refit")
    assert drift.loc["plus", "median_change"] == pytest.approx(-0.03)
    assert drift.loc["minus", "median_change"] == pytest.approx(0.03)
    assert drift.loc["zero", "median_change"] == pytest.approx(0.0)

    # predict: stage 2's posterior with the shared shift
    pred = [c for c in _calls(rec, "predict") if "predict" in c[3].get("out_prefix", "")
            or c[3].get("shared_shift")]
    assert pred and pred[0][3]["shared_shift"] == str(out / "kshift" / "shift.csv")
    assert not _calls(rec, "epistasis")

    # consistency: kanR alone (first library), its conditions only
    cons = pd.read_csv(out / "consistency" / "growth_priors_pinned.csv")
    assert set(cons["condition_rep"]) == {"kanR+kan", "kanR-kan"}
    assert set(pd.read_csv(out / "consistency" / "tfs_growth.csv")["library"]) == {"kanR"}


def test_rerun_skips_finished_steps(tmp_path, rec):
    out = tmp_path / "run"
    y = _write_yaml(tmp_path)
    cli.fit_two_stage(y, out_dir=str(out))
    n = len(rec.calls)
    cli.fit_two_stage(y, out_dir=str(out))
    assert len(rec.calls) == n


def test_no_consistency_and_epistasis(tmp_path, rec):
    out = tmp_path / "run"
    cli.fit_two_stage(_write_yaml(tmp_path, two_stage={"predict_epistasis": True,
                                                       "kshift": True}),
                      out_dir=str(out), consistency=False)
    assert not (out / "consistency").exists()
    epi = _calls(rec, "epistasis")
    assert epi and epi[0][3]["scale"] == "add" and epi[0][3]["shared_shift"]


def test_one_step_and_one_jackknife_task(tmp_path, rec):
    out = tmp_path / "run"
    y = _write_yaml(tmp_path)
    with pytest.raises(RuntimeError, match="stage 1"):
        cli.fit_two_stage(y, out_dir=str(out), jackknife_task=1)
    cli.fit_two_stage(y, out_dir=str(out), step="stage1")
    assert (out / "stage1" / "km.csv").exists()
    assert not (out / "stage2").exists()
    cli.fit_two_stage(y, out_dir=str(out), jackknife_task=2)
    assert (out / "jackknife" / "jk_02" / "km.csv").exists()
    assert not (out / "jackknife" / "jk_01").exists()
    # kshift collects the jackknife and refuses while subsamples are missing
    cli.fit_two_stage(y, out_dir=str(out), step="stage2")
    with pytest.raises(RuntimeError, match="not done"):
        cli.fit_two_stage(y, out_dir=str(out), step="kshift")
    with pytest.raises(ValueError, match="1-3"):
        cli.fit_two_stage(y, out_dir=str(out), jackknife_task=4)
    with pytest.raises(ValueError, match="step must be"):
        cli.fit_two_stage(y, out_dir=str(out), step="bogus")


def test_consistency_with_another_library_and_name(tmp_path, rec):
    out = tmp_path / "run"
    y = _write_yaml(tmp_path)
    cli.fit_two_stage(y, out_dir=str(out))
    cfg, settings = cli.read_two_stage_config(y)
    settings["consistency_library"] = "pheS"
    cli.run_consistency(str(out), cfg, settings, name="consistency_pheS")
    d = out / "consistency_pheS"
    assert (d / "consistency.csv").exists()
    assert set(pd.read_csv(d / "tfs_growth.csv")["library"]) == {"pheS"}
    cons = pd.read_csv(d / "growth_priors_pinned.csv")
    assert set(cons["condition_rep"]) == {"pheS+4CP", "pheS-4CP"}
    # the default run's kanR check is untouched
    assert set(pd.read_csv(out / "consistency" / "tfs_growth.csv")["library"]) == {"kanR"}


def test_kshift_warns_when_the_control_moves(tmp_path, capsys):
    r = Recorder(x={"plus": 0.4, "minus": 0.46, "zero": 0.5})
    lib = pd.DataFrame({"genotype": ["wt", "A1B", "C2D", "A1B/C2D"],
                        "in_spiked_origin": [True, False, False, False]})
    with patch.object(cli, "configure_model", r.configure_model), \
         patch.object(cli, "fit_model", r.fit_model), \
         patch.object(cli, "extract_params", r.extract_params), \
         patch.object(cli, "sample_posterior", r.sample_posterior), \
         patch.object(cli, "predict_theta", r.predict_theta), \
         patch.object(cli, "library_composition_table", return_value=lib):
        y = _write_yaml(tmp_path)
        out = tmp_path / "run"
        for step in ("stage1", "jackknife", "stage2", "kshift"):
            cli.fit_two_stage(y, out_dir=str(out), step=step)
    assert "warning: the control refit" in capsys.readouterr().out


def test_kshift_settings(tmp_path, rec):
    y = _write_yaml(tmp_path, two_stage={"kshift_step_size": 1e-4,
                                         "kshift_control": False,
                                         "fit": {"adam_step_size": 1e-3}})
    out = tmp_path / "run"
    for step in ("stage1", "jackknife", "stage2", "kshift"):
        cli.fit_two_stage(y, out_dir=str(out), step=step)
    assert not (out / "kshift" / "zero").exists()
    fit = _calls(rec, "fit", "kshift/plus")[0][2]
    assert fit["adam_step_size"] == 1e-4          # the refit's own wins
    assert _calls(rec, "fit", "stage2")[0][2]["adam_step_size"] == 1e-3
    assert set(pd.read_csv(out / "kshift" / "drift.csv")["refit"]) == {"plus", "minus"}
    with pytest.raises(ValueError, match="kshift_step_size"):
        cli.read_two_stage_config(_write_yaml(tmp_path, two_stage={"kshift_step_size": 0}))


def test_read_config_refuses_a_warp_in_growth_priors(tmp_path):
    pd.DataFrame({"condition_rep": CONDS, "k_loc": 0.01, "m_warp": [1, 0, 0, 0]}
                 ).to_csv(tmp_path / "gp.csv", index=False)
    with pytest.raises(ValueError, match="m_warp"):
        cli.read_two_stage_config(_write_yaml(tmp_path, growth_priors="gp.csv"))
    pd.DataFrame({"condition_rep": CONDS, "k_loc": 0.01, "m_warp": 0}
                 ).to_csv(tmp_path / "gp.csv", index=False)
    cfg, _ = cli.read_two_stage_config(_write_yaml(tmp_path, growth_priors="gp.csv"))
    assert cfg["growth_priors"] == str(tmp_path / "gp.csv")


def test_summary_table(tmp_path, rec):
    out = tmp_path / "run"
    cli.fit_two_stage(_write_yaml(tmp_path, two_stage={"kshift": True}),
                      out_dir=str(out))
    x = pd.read_csv(out / "tfs_x.csv")
    assert list(x.columns[:4]) == ["genotype", "num_mutations", "titrant_name",
                                   "titrant_conc"]
    assert len(x) == 8                          # 4 genotypes x 2 concentrations
    g = x.drop_duplicates("genotype").set_index("genotype")
    # _growth: 100 reads per genotype in each of 48 tubes, 24 per library
    assert g.loc["A1B", "reads"] == 4800
    assert g.loc["A1B", "reads_kanR"] == 2400 and g.loc["A1B", "reads_pheS"] == 2400
    assert g.loc["A1B/C2D", "num_mutations"] == 2 and g.loc["wt", "num_mutations"] == 0
    assert np.allclose(x.loc[x.genotype != "wt", "k_shift"], -0.03)
    assert x.loc[x.genotype == "wt", "k_shift"].eq(0).all()
    assert x.loc[x.genotype == "wt", "inconsistent"].isna().all()
    assert x.loc[x.genotype == "A1B", "consistency_z"].notna().all()


def test_summary_before_consistency_and_without_predict(tmp_path, rec):
    y = _write_yaml(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="predict"):
        cli.fit_two_stage(y, out_dir=str(out), step="summary")
    for step in ("stage1", "jackknife", "stage2", "kshift", "predict", "summary"):
        cli.fit_two_stage(y, out_dir=str(out), step=step)
    assert pd.read_csv(out / "tfs_x.csv")["inconsistent"].isna().all()
    cli.fit_two_stage(y, out_dir=str(out), step="consistency")
    cli.fit_two_stage(y, out_dir=str(out), step="summary")   # rewritten
    x = pd.read_csv(out / "tfs_x.csv")
    assert x.loc[x.genotype != "wt", "inconsistent"].notna().all()


def test_default_run_has_no_k_shift(tmp_path, rec):
    """Validation run 2: the shift is off by default, and so is the
    jackknife that only feeds it."""
    out = tmp_path / "run"
    y = _write_yaml(tmp_path, two_stage={"predict_epistasis": True})
    cli.fit_two_stage(y, out_dir=str(out))
    assert not (out / "jackknife").exists() and not (out / "kshift").exists()
    pred = [c for c in _calls(rec, "predict") if "shift_seed" in c[3]]
    assert pred and pred[0][3]["shared_shift"] is None
    assert _calls(rec, "epistasis")[0][3]["shared_shift"] is None
    x = pd.read_csv(out / "tfs_x.csv")
    assert x["k_shift"].isna().all()
    assert x.loc[x.genotype != "wt", "inconsistent"].notna().all()
    # a stale shift table is ignored while kshift is off
    for step in ("jackknife", "kshift"):
        cli.fit_two_stage(y, out_dir=str(out), step=step)
    assert (out / "kshift" / "shift.csv").exists()
    cli.fit_two_stage(y, out_dir=str(out), step="summary")
    assert pd.read_csv(out / "tfs_x.csv")["k_shift"].isna().all()
