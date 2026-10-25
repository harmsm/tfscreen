"""
Where two stage 1 fits differ in log density (README, run 4): the count
log likelihood of every growth row, the binding log likelihood of every
binding genotype and the log prior of every latent site, at each fit's
MAP; their differences summed by marker, genotype class, IPTG and tube.

    python attribute.py runs/fit_null_s1_plain runs/fit_null_s1_plain_from_plain_pinkm --out out/attr

Each folder is a stage1.py fit downloaded under runs/ (its config,
priors, guesses, library, tfs_growth.csv and MAP params), next to the
simulation folders its config names. The difference is the second minus
the first: negative where the second fit explains the data worse.
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

CLUSTER = "/gpfs/projects/harmslab/harms/studies/joint-fit-sim/"
IDX = ["replicate_idx", "time_idx", "condition_pre_idx", "condition_sel_idx",
       "titrant_name_idx", "titrant_conc_idx", "genotype_idx"]
KEY = ["genotype", "replicate", "condition_sel", "titrant_conc", "t_sel"]


def local_config(fit):
    """The fit's config with cluster paths pointed at the local runs/ copy."""
    runs = os.path.dirname(os.path.abspath(fit))
    text = open(os.path.join(fit, "tfs_configure_config.yaml")).read()
    out = os.path.join(fit, "config_local.yaml")
    open(out, "w").write(text.replace(CLUSTER, runs + "/"))
    return out


def score(fit):
    """Per-row growth and per-genotype binding log likelihood, per-site prior."""
    cwd = os.getcwd()
    cfg = local_config(fit)
    os.chdir(fit)
    try:
        orch, _ = read_configuration(os.path.basename(cfg))
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
        rows = df[KEY].copy()
        rows["ll"] = lp[tuple(df[c].to_numpy() for c in IDX)]
        good = np.asarray(orch.data.growth.good_mask)
        growth_total = float(lp[good].sum())

        b = tr["binding_obs"]
        blp = np.where(np.asarray(batch.binding.good_mask),
                       np.asarray(b["fn"].log_prob(b["value"])), 0.0)
        names = list(orch.binding_tm.tensor_dim_names)
        labels = orch.binding_tm.tensor_dim_labels[names.index("genotype")]
        binding = pd.Series(blp.sum(axis=(0, 1)), index=[str(g) for g in labels])

        priors = {n: float(s["fn"].log_prob(s["value"]).sum())
                  for n, s in tr.items()
                  if s["type"] == "sample" and not s["is_observed"]}
        return rows, binding, pd.Series(priors), growth_total
    finally:
        os.chdir(cwd)


def genotype_class(g, spikes, reads):
    if g == "wt":
        return "wt"
    if g in spikes:
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
    a, ba, pa, ga = score(args.first)
    b, bb, pb, gb = score(args.second)
    m = a.merge(b, on=KEY, suffixes=("_1", "_2"))
    m["d"] = m["ll_2"] - m["ll_1"]
    lib = pd.read_csv(os.path.join(args.first, "tfs_configure_library.csv"))
    spikes = set(lib.loc[lib["in_spiked_origin"].astype(bool), "genotype"].astype(str))
    reads = pd.read_csv(os.path.join(args.first, "tfs_growth.csv"),
                        usecols=["genotype", "counts"]).groupby("genotype")["counts"].sum()
    m["class"] = m["genotype"].astype(str).map(lambda g: genotype_class(g, spikes, reads))
    m.to_csv(os.path.join(args.out, "per_row.csv"), index=False)

    db = (bb - ba)
    dp = (pb - pa).dropna()
    print(f"growth log likelihood: second - first {gb - ga:+.1f} ({len(m)} rows)")
    print(f"binding log likelihood: second - first {db.sum():+.1f}")
    print(f"log prior: second - first {dp.sum():+.1f}")
    print(f"total: {gb - ga + db.sum() + dp.sum():+.1f}")
    print("\n== binding, second - first, by genotype")
    print(db.sort_values().round(1).to_string())
    print("\n== log prior, largest sites")
    print(dp[dp.abs().sort_values(ascending=False).index].head(10).round(1).to_string())
    for by in (["condition_sel"], ["condition_sel", "class"],
               ["condition_sel", "titrant_conc"],
               ["condition_sel", "replicate", "t_sel"]):
        t = m.groupby(by, observed=True)["d"].agg(["sum", "size"]).round(1)
        t = t[t["size"] > 0]
        print(f"\n== growth, second - first, by {', '.join(by)}")
        print(t.to_string())
    g = m.groupby("genotype")["d"].sum().sort_values()
    print("\n== genotypes losing most under the second fit")
    print(g.head(10).round(1).to_string())
    print("\n== genotypes gaining most under the second fit")
    print(g.tail(5).round(1).to_string())


if __name__ == "__main__":
    main()
