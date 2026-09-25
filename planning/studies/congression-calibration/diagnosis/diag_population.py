"""
Is the population dk_geno / activity / theta the mixture looks co-residents
up in the same as each genotype's own value? Compare the component's focal
output with population[batch_idx] at the mixture fit's median latents.
Run from this directory; GRID selects the grid.
"""
import sys
sys.argv = [sys.argv[0]]
from diag_lam0 import *

(d2, o2, s2, st2) = load("run_0002")
lat2, _ = median_latents(o2, s2, st2)
data = full_batch(o2)
tr = trace(substitute(seed(o2.jax_model, 0), data=lat2)).get_trace(data=data, priors=o2.priors)
print("deterministic sites:", [k for k, v in tr.items() if v["type"] == "deterministic"][:40])

from tfscreen.tfmodel.generative.registry import model_registry
comp = model_registry["dk_geno"][o2.settings["dk_geno"]]
def run_dk():
    return comp.define_model("dk_geno", data.growth, o2.priors.growth.dk_geno,
                             return_population=True)
focal, pop = substitute(seed(run_dk, 0), data=lat2)()
bi = np.asarray(data.growth.batch_idx)
f = np.ravel(np.asarray(focal)); p = np.asarray(pop)
print("focal dk_geno: shape", np.asarray(focal).shape, " population:", p.shape)
print("focal - population[batch_idx]: max|d| %.3g" % np.max(np.abs(f - p[bi])))
print("focal mean %.4f  population mean %.4f" % (f.mean(), p.mean()))
if "dk_geno" in tr:
    print("traced dk_geno site mean %.4f" % float(np.mean(tr["dk_geno"]["value"])))
