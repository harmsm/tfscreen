"""Per-site joint log density of the single model at the single and mixture fits."""
import sys
sys.argv = [sys.argv[0]]
from diag_lam0 import *
import pandas as pd
from numpyro.infer.util import log_density

(d1, o1, s1, st1), (d2, o2, s2, st2) = load("run_0001"), load("run_0002")
lat1, _ = median_latents(o1, s1, st1)
lat2, _ = median_latents(o2, s2, st2)
lat2.pop("transformation_lam")
rows = {}
for tag, lat in (("single_fit", lat1), ("mixture_fit", lat2)):
    _, tr = log_density(seed(o1.jax_model, 0), (), dict(data=full_batch(o1), priors=o1.priors), lat)
    for name, site in tr.items():
        if site["type"] != "sample":
            continue
        lp = site["fn"].log_prob(site["value"])
        if site.get("scale") is not None:
            lp = lp * site["scale"]
        if site.get("mask") is not None:
            lp = jnp.where(site["mask"], lp, 0.0)
        rows.setdefault(name, {})[tag] = float(jnp.sum(lp))
df = pd.DataFrame(rows).T
df["diff"] = df.mixture_fit - df.single_fit
pd.set_option("display.width", 200)
print(df.reindex(df["diff"].abs().sort_values(ascending=False).index).head(12).round(1).to_string())
