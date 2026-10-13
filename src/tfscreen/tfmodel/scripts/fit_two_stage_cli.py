"""
tfs-fit-two-stage: the two-stage fit of a growth-only relative model, with
X intervals that carry stage 1's k uncertainty (planning/two-stage-fit.md).
"""

import contextlib
import inspect
import json
import os

import pandas as pd
import yaml

from tfscreen.genetics.library_design import library_composition_table
from tfscreen.tfmodel import two_stage as ts
from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.extract_params_cli import extract_params
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.scripts.predict_epistasis_cli import predict_epistasis
from tfscreen.tfmodel.scripts.predict_theta_cli import predict_theta
from tfscreen.tfmodel.scripts.sample_posterior_cli import sample_posterior
from tfscreen.util.cli.generalized_main import generalized_main
from tfscreen.util.provenance import get_provenance

STEPS = ("stage1", "jackknife", "stage2", "kshift", "predict", "consistency")

# tfs-configure-model arguments that name files (resolved against the YAML's
# directory).
PATH_KEYS = ("growth_df", "library_config", "growth_priors",
             "growth_priors_wt_rates", "base_growth_df", "presplit_df",
             "thermo_data", "binding_df")

# Set by the two-stage fit itself, so not allowed at the top level.
RESERVED_KEYS = ("out_prefix", "transformation_model", "transformation_lambda",
                 "congression_theta_rule")

TWO_STAGE_DEFAULTS = {
    "transformation_lambda": None,
    "seed": 1,
    "jackknife_subsamples": 30,
    "jackknife_drop_frac": 0.2,
    "num_posterior_samples": 500,
    "consistency": True,
    "consistency_library": None,
    "predict_epistasis": False,
    "fit": {},
    "posterior": {},
}

FIT_RESERVED = ("config_file", "seed", "out_prefix", "analysis_method",
                "init_from", "checkpoint_file")
POSTERIOR_RESERVED = ("config_file", "checkpoint_file", "out_prefix",
                      "num_posterior_samples")

# The chained functions' arguments, read once (the YAML is checked against
# them).
CONFIGURE_ARGS = frozenset(inspect.signature(configure_model).parameters)
FIT_ARGS = frozenset(inspect.signature(fit_model).parameters)
POSTERIOR_ARGS = frozenset(inspect.signature(sample_posterior).parameters)

CONFIG = "tfs_configure_config.yaml"
CHECKPOINT = "tfs_fit_model_checkpoint.pkl"
PARAMS = "tfs_fit_model_params.npz"
POSTERIOR = "tfs_posterior.h5"
PRED = "tfs_pred_theta.csv"


# ---------------------------------------------------------------------------
# The YAML
# ---------------------------------------------------------------------------

def read_two_stage_config(config_file):
    """
    The configure arguments and two-stage settings from the YAML.

    Top-level keys are ``tfs-configure-model``'s arguments, under the same
    names; file paths are relative to the YAML's directory. The
    ``two_stage`` block holds the two-stage settings
    (``TWO_STAGE_DEFAULTS``).

    Returns
    -------
    tuple of dict
        ``(configure_kwargs, settings)``.

    Raises
    ------
    ValueError
        On an unknown or reserved key, a missing input, a model the
        two-stage fit does not support, or a missing lambda prior.
    """
    with open(config_file) as fh:
        raw = yaml.safe_load(fh) or {}
    base = os.path.dirname(os.path.abspath(config_file))
    settings = dict(TWO_STAGE_DEFAULTS)
    block = raw.pop("two_stage", None) or {}
    unknown = sorted(set(block) - set(TWO_STAGE_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown two_stage key(s) {unknown}; allowed: "
                         f"{sorted(TWO_STAGE_DEFAULTS)}.")
    settings.update(block)

    allowed = CONFIGURE_ARGS
    reserved = sorted(set(raw) & set(RESERVED_KEYS))
    if reserved:
        raise ValueError(f"{reserved} are set by the two-stage fit (stage 1 "
                         "fits the congression mixture, stage 2 does not); "
                         "put the lambda prior in the two_stage block.")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown}: top-level keys are "
                         "tfs-configure-model's arguments.")
    for key in ("growth_df", "library_config"):
        if not raw.get(key):
            raise ValueError(f"The two-stage fit needs {key}.")
    cfg = dict(raw)
    for key in PATH_KEYS:
        if cfg.get(key):
            path = cfg[key]
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            if not os.path.exists(path):
                raise ValueError(f"{key}: {path} does not exist")
            cfg[key] = path

    if cfg.get("theta_model") != "hill_relative":
        raise ValueError("The two-stage fit is for the growth-only relative "
                         "model: set theta_model: hill_relative.")
    if cfg.get("condition_growth_model", "linear") != "linear":
        raise ValueError("The two-stage fit pins linear growth's k and m: "
                         "condition_growth_model must be linear.")
    if not cfg.get("growth_shares_replicates"):
        raise ValueError("The two-stage fit pins one k and m per condition: "
                         "set growth_shares_replicates: true.")
    for item in cfg.get("set_priors") or []:
        name = str(item).split("=")[0].strip()
        if name.endswith("_pinned"):
            raise ValueError(f"set_priors {item!r}: stage 2 pins k and m per "
                             "condition itself.")

    lam = settings["transformation_lambda"]
    if lam is None or len(lam) != 2:
        raise ValueError("two_stage.transformation_lambda (mean, std) is "
                         "required: stage 1 fits the congression mixture, "
                         "with the measured lambda as its prior.")
    settings["transformation_lambda"] = [float(x) for x in lam]
    if int(settings["jackknife_subsamples"]) < 2:
        raise ValueError("two_stage.jackknife_subsamples must be at least 2")
    for name, res, sig in (("fit", FIT_RESERVED, FIT_ARGS),
                           ("posterior", POSTERIOR_RESERVED, POSTERIOR_ARGS)):
        extra = settings[name] or {}
        bad = sorted(set(extra) - sig) + sorted(set(extra) & set(res))
        if bad:
            raise ValueError(f"two_stage.{name}: {bad} cannot be set here.")
        settings[name] = dict(extra)
    return cfg, settings


# ---------------------------------------------------------------------------
# Shared pieces of a step
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _in_dir(path):
    os.makedirs(path, exist_ok=True)
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _say(msg):
    print(f">>> {msg}", flush=True)


def _stage1_kwargs(cfg, settings):
    out = dict(cfg)
    out.update(transformation_model="mixture",
               transformation_lambda=settings["transformation_lambda"],
               congression_theta_rule="max")
    return out


def _configure(kwargs, growth_df, growth_priors=None, drop_wt_rates=False):
    if os.path.exists(CONFIG):
        return
    kw = dict(kwargs)
    kw["growth_df"] = os.path.abspath(growth_df)
    if growth_priors is not None:
        kw["growth_priors"] = os.path.abspath(growth_priors)
    if drop_wt_rates:
        kw.pop("growth_priors_wt_rates", None)
    configure_model(out_prefix="tfs_configure", **kw)


def _fit(settings, init_from=None):
    if os.path.exists(PARAMS):
        return
    fit_model(CONFIG, seed=int(settings["seed"]), analysis_method="map",
              init_from=init_from, **settings["fit"])


def _km(path="km.csv"):
    if not os.path.exists(path):
        extract_params(CONFIG, CHECKPOINT, out_prefix="tfs_params")
        km = ts.km_table(pd.read_csv("tfs_params_growth_k.csv"),
                         pd.read_csv("tfs_params_growth_m.csv"))
        km.to_csv(path)
    return pd.read_csv(path, index_col="condition_rep")


def _posterior(settings):
    if os.path.exists(POSTERIOR):
        return
    kw = {"skip_growth_observations": True}
    kw.update(settings["posterior"])
    sample_posterior(CONFIG, CHECKPOINT, out_prefix="tfs_posterior",
                     num_posterior_samples=int(settings["num_posterior_samples"]),
                     **kw)


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

def run_stage1(out_dir, cfg, settings):
    """wt, spikes and singles with the congression mixture; writes km.csv."""
    d = os.path.join(out_dir, "stage1")
    if os.path.exists(os.path.join(d, "km.csv")):
        _say("stage1: done, skipped")
        return
    _say("stage1")
    with _in_dir(d):
        if not os.path.exists("tfs_growth.csv"):
            lib = library_composition_table(cfg["library_config"])
            keep = ts.stage1_genotypes(lib, ts.growth_genotypes(cfg["growth_df"]))
            ts.filter_growth_csv(cfg["growth_df"], "tfs_growth.csv", genotypes=keep)
            print(f"stage 1: {len(keep)} genotypes (wt, spikes, singles)",
                  flush=True)
        _configure(_stage1_kwargs(cfg, settings), "tfs_growth.csv")
        _fit(settings)
        _km()


def run_jackknife_task(out_dir, cfg, settings, task):
    """One delete-d jackknife subsample of stage 1 (task numbered from 1)."""
    n_tasks = int(settings["jackknife_subsamples"])
    if not 1 <= task <= n_tasks:
        raise ValueError(f"jackknife_task must be 1-{n_tasks}, got {task}")
    src = os.path.join(out_dir, "stage1", "tfs_growth.csv")
    if not os.path.exists(src):
        raise RuntimeError("the jackknife needs stage 1's genotype table; run "
                           "--step stage1 first")
    d = os.path.join(out_dir, "jackknife", f"jk_{task:02d}")
    if os.path.exists(os.path.join(d, "km.csv")):
        _say(f"jackknife {task}: done, skipped")
        return
    _say(f"jackknife {task}")
    with _in_dir(d):
        if not os.path.exists("tfs_growth.csv"):
            sub, n, nd, dropped = ts.jackknife_subsample(
                pd.read_csv(src), float(settings["jackknife_drop_frac"]), task)
            sub.to_csv("tfs_growth.csv", index=False)
            with open("subsample.json", "w") as fh:
                json.dump({"n": n, "d": nd, "dropped": dropped}, fh, indent=1)
        _configure(_stage1_kwargs(cfg, settings), "tfs_growth.csv")
        _fit(settings)
        _km()


def summarize_jackknife(out_dir, settings):
    """k and m SEs and k's main direction from every subsample."""
    d = os.path.join(out_dir, "jackknife")
    if os.path.exists(os.path.join(d, "k_step.csv")):
        return
    n_tasks = int(settings["jackknife_subsamples"])
    dirs = [os.path.join(d, f"jk_{i:02d}") for i in range(1, n_tasks + 1)]
    missing = [x for x in dirs if not os.path.exists(os.path.join(x, "km.csv"))]
    if missing:
        raise RuntimeError(f"{len(missing)} of {n_tasks} jackknife subsamples "
                           f"are not done, e.g. {missing[:3]}")
    nd = set()
    est = []
    for x in dirs:
        with open(os.path.join(x, "subsample.json")) as fh:
            info = json.load(fh)
        nd.add((info["n"], info["d"]))
        est.append(pd.read_csv(os.path.join(x, "km.csv"), index_col="condition_rep"))
    if len(nd) != 1:
        raise RuntimeError(f"subsamples differ in tubes kept: {sorted(nd)}")
    n, dd = nd.pop()
    out = ts.jackknife_km(est, n, dd)
    out["se"].to_csv(os.path.join(d, "jackknife_se.csv"))
    out["k_cov"].to_csv(os.path.join(d, "k_cov.csv"))
    out["k_step"].rename_axis("condition_rep").to_csv(os.path.join(d, "k_step.csv"))
    with open(os.path.join(d, "jackknife.json"), "w") as fh:
        json.dump({"n": n, "d": dd, "factor": out["factor"],
                   "k_share": out["k_share"], "subsamples": n_tasks}, fh, indent=1)
    print(f"jackknife: {n_tasks} subsamples, {dd} of {n} tubes dropped; k's "
          f"main direction carries {out['k_share']:.3f} of its variance",
          flush=True)


def run_jackknife(out_dir, cfg, settings):
    for task in range(1, int(settings["jackknife_subsamples"]) + 1):
        run_jackknife_task(out_dir, cfg, settings, task)
    summarize_jackknife(out_dir, settings)


def run_stage2(out_dir, cfg, settings):
    """The full library with every condition's k and m pinned at stage 1."""
    d = os.path.join(out_dir, "stage2")
    if os.path.exists(os.path.join(d, PRED)):
        _say("stage2: done, skipped")
        return
    km_path = os.path.join(out_dir, "stage1", "km.csv")
    if not os.path.exists(km_path):
        raise RuntimeError("stage 2 needs stage 1's k and m; run --step stage1")
    _say("stage2")
    km = pd.read_csv(km_path, index_col="condition_rep")
    with _in_dir(d):
        ts.pinned_growth_priors(km).to_csv("growth_priors_pinned.csv", index=False)
        _configure(cfg, cfg["growth_df"], "growth_priors_pinned.csv",
                   drop_wt_rates=True)
        _fit(settings)
        _posterior(settings)
        if not os.path.exists("tfs_params_growth_k.csv"):
            extract_params(CONFIG, POSTERIOR, out_prefix="tfs_params")
        predict_theta(CONFIG, POSTERIOR, out_prefix="tfs_pred_theta")


def run_kshift(out_dir, cfg, settings):
    """Two stage 2 refits at k plus and minus one jackknife SD; the shift."""
    d = os.path.join(out_dir, "kshift")
    if os.path.exists(os.path.join(d, "shift.csv")):
        _say("kshift: done, skipped")
        return
    summarize_jackknife(out_dir, settings)
    init = os.path.abspath(os.path.join(out_dir, "stage2", PARAMS))
    if not os.path.exists(init):
        raise RuntimeError("the k-shift refits start from stage 2's MAP; run "
                           "--step stage2")
    _say("kshift")
    km = pd.read_csv(os.path.join(out_dir, "stage1", "km.csv"),
                     index_col="condition_rep")
    step = pd.read_csv(os.path.join(out_dir, "jackknife", "k_step.csv"),
                       index_col="condition_rep")["k_step"]
    plus, minus = ts.k_shift_priors(km, step)
    preds = {}
    for name, table in (("plus", plus), ("minus", minus)):
        with _in_dir(os.path.join(d, name)):
            table.to_csv("growth_priors_pinned.csv", index=False)
            _configure(cfg, cfg["growth_df"], "growth_priors_pinned.csv",
                       drop_wt_rates=True)
            _fit(settings, init_from=init)
            if not os.path.exists(PRED):
                predict_theta(CONFIG, CHECKPOINT, out_prefix="tfs_pred_theta")
            preds[name] = pd.read_csv(PRED)
    shift = ts.shift_table(preds["plus"], preds["minus"])
    shift.to_csv(os.path.join(d, "shift.csv"), index=False)
    nw = shift["genotype"] != "wt"
    print(f"kshift: median |shift| {shift.loc[nw, 'shift'].abs().median():.4f}",
          flush=True)


def run_predict(out_dir, cfg, settings):
    """Stage 2's X (and epistasis) with the shared k shift."""
    d = os.path.abspath(os.path.join(out_dir, "predict"))
    os.makedirs(d, exist_ok=True)
    shift = os.path.abspath(os.path.join(out_dir, "kshift", "shift.csv"))
    if not os.path.exists(shift):
        raise RuntimeError("prediction needs kshift/shift.csv; run --step kshift")
    epi_done = (not settings["predict_epistasis"]
                or os.path.exists(os.path.join(d, "tfs_pred_epistasis.csv")))
    if os.path.exists(os.path.join(d, PRED)) and epi_done:
        _say("predict: done, skipped")
        return
    _say("predict")
    with _in_dir(os.path.join(out_dir, "stage2")):
        if not os.path.exists(os.path.join(d, PRED)):
            predict_theta(CONFIG, POSTERIOR, out_prefix=os.path.join(d, "tfs_pred_theta"),
                          shared_shift=shift, shift_seed=int(settings["seed"]))
        epi = os.path.join(d, "tfs_pred_epistasis.csv")
        if settings["predict_epistasis"] and not os.path.exists(epi):
            predict_epistasis(CONFIG, POSTERIOR,
                              out_prefix=os.path.join(d, "tfs_pred_epistasis"),
                              scale="add", shared_shift=shift,
                              shift_seed=int(settings["seed"]))


def _library_conditions(table, library):
    rows = table[table["library"].astype(str) == str(library)]
    return sorted(set(rows["condition_pre"].astype(str))
                  | set(rows["condition_sel"].astype(str)))


def run_consistency(out_dir, cfg, settings):
    """Stage 2 on one library's tubes alone, against the full stage 2."""
    d = os.path.join(out_dir, "consistency")
    if os.path.exists(os.path.join(d, "consistency.csv")):
        _say("consistency: done, skipped")
        return
    both = os.path.join(out_dir, "stage2", PRED)
    if not os.path.exists(both):
        raise RuntimeError("the consistency check needs stage 2; run --step stage2")
    s1 = pd.read_csv(os.path.join(out_dir, "stage1", "tfs_growth.csv"),
                     usecols=["library", "condition_pre", "condition_sel"])
    libraries = sorted(s1["library"].astype(str).unique())
    lib = settings["consistency_library"] or libraries[0]
    if lib not in libraries:
        raise ValueError(f"consistency_library {lib!r} is not one of {libraries}")
    if len(libraries) < 2:
        raise ValueError("the consistency check needs two libraries")
    _say(f"consistency ({lib} alone)")
    km = pd.read_csv(os.path.join(out_dir, "stage1", "km.csv"),
                     index_col="condition_rep")
    conds = [c for c in _library_conditions(s1, lib) if c in km.index]
    with _in_dir(d):
        if not os.path.exists("tfs_growth.csv"):
            ts.filter_growth_csv(cfg["growth_df"], "tfs_growth.csv", library=lib)
        ts.pinned_growth_priors(km, conditions=conds).to_csv(
            "growth_priors_pinned.csv", index=False)
        _configure(cfg, "tfs_growth.csv", "growth_priors_pinned.csv",
                   drop_wt_rates=True)
        _fit(settings)
        _posterior(settings)
        if not os.path.exists(PRED):
            predict_theta(CONFIG, POSTERIOR, out_prefix="tfs_pred_theta")
        out = ts.marker_consistency(pd.read_csv(os.path.abspath(
            os.path.join("..", "stage2", PRED))), pd.read_csv(PRED))
        out.to_csv("consistency.csv")
    print(f"consistency: {int(out['inconsistent'].sum())} of {len(out)} "
          f"genotypes beyond |z| 2 ({lib} alone against both libraries)",
          flush=True)


RUNNERS = {"stage1": run_stage1, "jackknife": run_jackknife,
           "stage2": run_stage2, "kshift": run_kshift,
           "predict": run_predict, "consistency": run_consistency}


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def fit_two_stage(config_file,
                  out_dir="two_stage",
                  step="all",
                  jackknife_task=None,
                  consistency=True):
    """
    Fit a growth-only relative model in two stages, with X intervals that
    carry stage 1's k uncertainty.

    The full-library MAP makes the growth slope m too steep (each double
    mutant pulls it a little and the pulls add), and congression flattens a
    marker's m in any stage 1 large enough to pin X's scale
    (planning/studies/m-bias-factorial/). This command runs, in out_dir:
    stage1 (wt, spikes and singles with the congression mixture, max rule,
    lambda fitted with two_stage.transformation_lambda as its prior: every
    condition's k and m); jackknife (two_stage.jackknife_subsamples refits
    of stage 1, each dropping two_stage.jackknife_drop_frac of the tubes,
    stratified by condition and IPTG: k and m standard errors and k's main
    direction); stage2 (the full library with every condition's k and m
    pinned at stage 1; MAP, Laplace, X); kshift (two stage 2 refits at k
    plus and minus one jackknife SD along its main direction, from stage
    2's MAP: X's shift per SD, kshift/shift.csv); predict (stage 2's X, and
    with two_stage.predict_epistasis its epistasis, with the shift added as
    one shared multiple per posterior draw, predict/); consistency (stage 2
    on one library's tubes alone against the full stage 2, per genotype
    z-scores, consistency/consistency.csv). Each step skips work whose
    outputs exist, so a run resumes and a step reruns by deleting its
    directory.

    Parameters
    ----------
    config_file : str
        YAML whose top-level keys are tfs-configure-model's arguments
        (growth_df and library_config required; theta_model hill_relative,
        growth_shares_replicates true; file paths relative to the YAML),
        plus a two_stage block: transformation_lambda (mean, std; required),
        seed (1), jackknife_subsamples (30), jackknife_drop_frac (0.2),
        num_posterior_samples (500), consistency (true),
        consistency_library (default the first library alphabetically),
        predict_epistasis (false), fit (extra tfs-fit-model arguments) and
        posterior (extra tfs-sample-posterior arguments).
    out_dir : str, optional
        Directory for every step's outputs. Default 'two_stage'.
    step : str, optional
        'all' (default) or one of stage1, jackknife, stage2, kshift,
        predict, consistency.
    jackknife_task : int or None, optional
        Run only this jackknife subsample (1 to jackknife_subsamples), for
        a cluster array; needs stage 1 done. The kshift step collects them.
    consistency : bool, optional
        Run the consistency step with 'all' (default True; --no_consistency
        skips it, as does consistency: false in the YAML).
    """
    cfg, settings = read_two_stage_config(config_file)
    if not consistency:
        settings["consistency"] = False
    if step != "all" and step not in STEPS:
        raise ValueError(f"step must be 'all' or one of {STEPS}, got {step!r}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    resolved = os.path.join(out_dir, "two_stage_config.yaml")
    with open(resolved, "w") as fh:
        yaml.safe_dump({**cfg, "two_stage": settings}, fh, sort_keys=False)
    with open(os.path.join(out_dir, "two_stage_provenance.json"), "w") as fh:
        json.dump(get_provenance(), fh, indent=1, default=str)

    if jackknife_task is not None:
        if step not in ("all", "jackknife"):
            raise ValueError("jackknife_task runs one jackknife subsample; "
                             "use it with --step jackknife (or alone)")
        run_jackknife_task(out_dir, cfg, settings, int(jackknife_task))
        return

    todo = STEPS if step == "all" else (step,)
    for name in todo:
        if name == "consistency" and step == "all" and not settings["consistency"]:
            _say("consistency: off")
            continue
        RUNNERS[name](out_dir, cfg, settings)
    _say("done")


def main():
    generalized_main(fit_two_stage,
                     manual_arg_types={"jackknife_task": int})


if __name__ == "__main__":
    main()
