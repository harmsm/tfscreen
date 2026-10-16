"""
Where two stage 1 fits differ in likelihood (README, run 5): the count
log likelihood of every growth observation and the log prior of every
latent site, at each fit's MAP, and their differences summed by marker,
genotype class, IPTG and tube.

    python attribute.py runs/attr_free/stage1 runs/attr_pinned/stage1 --out out/attr

Each folder is a stage 1 (tfs_configure_config.yaml or config_local.yaml,
tfs_fit_model_params.npz, tfs_growth.csv). The difference is the second
minus the first: negative where the second fit explains the data worse.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro import handlers  # noqa: E402

from tfscreen.tfmodel.configuration_io import read_configuration  # noqa: E402
from tfscreen.tfmodel.inference.run_inference import RunInference  # noqa: E402

CLUSTER = "/gpfs/projects/harmslab/harms/studies/two-stage-validation"
IDX = ["replicate_idx", "time_idx", "condition_pre_idx", "condition_sel_idx",
       "titrant_name_idx", "titrant_conc_idx", "genotype_idx"]
KEY = ["genotype", "replicate", "condition_sel", "titrant_conc", "t_sel"]
SPIKES = {"M42I", "H74A", "K84L", "D88A", "M42I/H74A", "M42I/K84L",
          "H74A/K84L", "M42I/H74A/K84L"}


def config_path(stage1):
    local = os.path.join(stage1, "config_local.yaml")
    if not os.path.exists(local):
        text = open(os.path.join(stage1, "tfs_configure_config.yaml")).read()
        run = os.path.basename(os.path.dirname(os.path.abspath(stage1)))
        open(local, "w").write(text.replace(f"{CLUSTER}/{run}/stage1/", ""))
    return "config_local.yaml"


def score(stage1):
    """Per-row count log likelihood and per-site log prior at the MAP."""
    cwd = os.getcwd()
    os.chdir(stage1)
    try:
        orch, _ = read_configuration(config_path("."))
        ri = RunInference(orch, 0)
        with np.load("tfs_fit_model_params.npz") as p:
            values = {k[:-len("_auto_loc")]: jnp.asarray(p[k]) for k in p.files
                      if k.endswith("_auto_loc")}
        G = orch.data.num_genotype
        batch = ri._unscaled_batch(jax.device_put(orch.data), jnp.arange(G))
        model = handlers.substitute(orch.jax_model, data=values)
        tr = handlers.trace(handlers.seed(model, 0)).get_trace(
            data=batch, priors=orch.priors)
        site = tr["growth_obs"]
        lp = np.asarray(site["fn"].log_prob(site["value"]))
        df = orch.growth_df
        ix = tuple(df[c].to_numpy() for c in IDX)
        rows = df[KEY].copy()
        rows["ll"] = lp[ix]
        priors = {n: float(s["fn"].log_prob(s["value"]).sum())
                  for n, s in tr.items() if s["type"] == "sample" and not s["is_observed"]}
        return rows, pd.Series(priors), float(lp[np.asarray(orch.data.growth.good_mask)].sum())
    finally:
        os.chdir(cwd)


def genotype_class(g, reads):
    if g == "wt":
        return "wt"
    if g in SPIKES:
        return "spike"
    r = reads.get(g, 0)
    for edge, name in ((1e4, "single <1e4"), (1e5, "single 1e4-1e5"),
                       (1e6, "single 1e5-1e6")):
        if r < edge:
            return name
    return "single >=1e6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--out", default="out/attr")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    a, pa, tot_a = score(args.first)
    b, pb, tot_b = score(args.second)
    m = a.merge(b, on=KEY, suffixes=("_1", "_2"))
    m["d"] = m["ll_2"] - m["ll_1"]
    reads = pd.read_csv(os.path.join(args.first, "tfs_growth.csv"),
                        usecols=["genotype", "counts"]).groupby("genotype")["counts"].sum()
    m["class"] = m["genotype"].map(lambda g: genotype_class(g, reads))
    m.to_csv(os.path.join(args.out, "per_row.csv"), index=False)

    print(f"count log likelihood: first {tot_a:.1f}, second {tot_b:.1f}, "
          f"second - first {tot_b - tot_a:+.1f} ({len(m)} rows matched)")
    dp = (pb - pa).dropna()
    print(f"log prior: second - first {dp.sum():+.1f}; largest sites:")
    print(dp[dp.abs().sort_values(ascending=False).index].head(8).round(1).to_string())
    for by in (["condition_sel"], ["condition_sel", "class"],
               ["condition_sel", "titrant_conc"],
               ["condition_sel", "replicate", "t_sel"]):
        t = m.groupby(by)["d"].agg(["sum", "size"]).round(1)
        print(f"\n== second - first, by {', '.join(by)}")
        print(t.to_string())
    g = m.groupby("genotype")["d"].sum().sort_values()
    print("\n== genotypes losing most under the second fit")
    print(g.head(10).round(1).to_string())


if __name__ == "__main__":
    main()
