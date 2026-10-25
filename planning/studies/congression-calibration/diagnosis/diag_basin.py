"""
Is the mixture fit's solution better than single's solution under the
mixture model (joint log density, priors included)? And where does the
mixture's better binding fit come from? Run from this directory; GRID
selects the grid (default: the unmasked one).
"""
import sys
sys.argv = [sys.argv[0]]
from diag_lam0 import *
import pandas as pd
from numpyro.infer.util import log_density

(d1, o1, s1, st1), (d2, o2, s2, st2) = load("run_0001"), load("run_0002")
lat1, _ = median_latents(o1, s1, st1)
lat2, _ = median_latents(o2, s2, st2)
data = full_batch(o2)


def joint(orch, lat):
    lp, _ = log_density(seed(orch.jax_model, 0), (), dict(data=full_batch(orch), priors=orch.priors), lat)
    return float(lp)


x = dict(lat1); x["transformation_lam"] = lat2["transformation_lam"]
print("mixture model joint log density:")
print("  at mixture fit          %.1f" % joint(o2, lat2))
print("  at single fit (same lam) %.1f" % joint(o2, x))
y = dict(lat2); y.pop("transformation_lam")
print("single model joint log density:")
print("  at single fit           %.1f" % joint(o1, lat1))
print("  at mixture fit          %.1f" % joint(o1, y))

# Binding log-lik per genotype at each fit
def binding_by_geno(orch, lat):
    tr = trace(substitute(seed(orch.jax_model, 0), data=lat)).get_trace(
        data=full_batch(orch), priors=orch.priors)
    site = tr["binding_obs"]
    lp = site["fn"].log_prob(site["value"])
    if site.get("mask") is not None:
        lp = lp * site["mask"]
    return np.asarray(lp)

b1 = binding_by_geno(o1, lat1); b2 = binding_by_geno(o2, lat2)
bt = o2.binding_tm
names = bt.tensor_dim_names
labels = list(bt.tensor_dim_labels[names.index("genotype")])
gain = (b2 - b1).reshape(-1, b1.shape[-1]).sum(axis=0) if b1.shape[-1] == len(labels) else None
if gain is None:
    print("binding tensor layout", names, b1.shape)
else:
    lib = pd.read_csv(glob.glob(d2 + "/*_sim_library.csv")[0])
    spiked = set(lib.loc[lib.library_origin == "spiked", "genotype"].astype(str))
    df = pd.DataFrame({"genotype": [str(g) for g in labels], "binding_gain": gain})
    df["spiked"] = df.genotype.isin(spiked)
    print("\nbinding log-lik gain (mixture fit - single fit), by origin:")
    print(df.groupby("spiked").binding_gain.agg(["count", "sum"]).round(1))
    print(df.sort_values("binding_gain", ascending=False).head(8).round(1).to_string(index=False))
