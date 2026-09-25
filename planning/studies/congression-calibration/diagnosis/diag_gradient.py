"""
Is the mixture's SVI gradient biased along the k/dk_geno direction?

Build the mixture fit's guide parameters from the single fit's solution
(keeping the mixture fit's lambda parameters), then average the ELBO
gradient over many draws with no optimization. Also the deterministic
gradient of the mixture's joint log density at single's median latents.
Run from this directory; GRID selects the grid (default: the unmasked one).
"""
import sys
sys.argv = [sys.argv[0]]
from diag_lam0 import *
from numpyro.infer import Trace_ELBO
from numpyro.infer.util import log_density

N_DRAWS = 64

(d1, o1, s1, st1), (d2, o2, s2, st2) = load("run_0001"), load("run_0002")
p1 = s1.get_params(st1)
p2 = s2.get_params(st2)

shared = sorted(set(p1) & set(p2))
only2 = sorted(set(p2) - set(p1))
print("mixture-only params:", only2)
start = {k: (p1[k] if k in p1 else p2[k]) for k in p2}

data = full_batch(o2)
elbo = Trace_ELBO(num_particles=1)


def mean_grad(params, orch, svi, n=N_DRAWS):
    def loss(p, key):
        return elbo.loss(key, p, orch.jax_model, svi.guide,
                         data=full_batch(orch), priors=orch.priors)
    g = jax.jit(jax.grad(loss))
    grads = [g(params, jax.random.PRNGKey(i)) for i in range(n)]
    mean = jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.stack(x), 0), *grads)
    se = jax.tree_util.tree_map(lambda *x: jnp.std(jnp.stack(x), 0) / np.sqrt(n), *grads)
    return mean, se


def show(tag, grads, keys):
    mean, se = grads
    print(f"\n{tag}: mean d(loss)/d(param) +/- SE (SVI moves against the sign)")
    for k in keys:
        m = np.ravel(np.asarray(mean[k])); s = np.ravel(np.asarray(se[k]))
        print(f"  {k:40s} " + "  ".join(f"{a:+.3g}+/-{b:.2g}" for a, b in zip(m, s)))


keys = [k for k in p2 if k.startswith("condition_growth_k")] + \
       [k for k in p2 if k.startswith("dk_geno") and np.size(p2[k]) <= 4] + only2
show("mixture model, guide at SINGLE's solution", mean_grad(start, o2, s2), keys)
show("mixture model, guide at MIXTURE's own solution", mean_grad(p2, o2, s2), keys)
k1 = [k for k in keys if k in p1]
show("single model, guide at SINGLE's solution", mean_grad(p1, o1, s1), k1)

# Deterministic check: gradient of the joint log density wrt the k latent.
lat1, _ = median_latents(o1, s1, st1)
lat = dict(lat1)
lat["transformation_lam"] = jnp.asarray(np.exp(float(p2["transformation_lam_loc"])))


def joint(l, orch):
    lp, _ = log_density(seed(orch.jax_model, 0), (),
                        dict(data=full_batch(orch), priors=orch.priors), l)
    return lp


for tag, orch, l in (("mixture", o2, lat), ("single", o1, lat1)):
    g = jax.grad(joint)(l, orch)
    kname = [k for k in g if k.startswith("condition_growth_k")]
    print(f"\n{tag} joint log density at single's median latents, d/d(latent):")
    for k in kname:
        print(f"  {k:30s}", np.round(np.ravel(np.asarray(g[k])), 1))
    dk = np.asarray(g["dk_geno_offset"]) if "dk_geno_offset" in g else None
    if dk is not None:
        print(f"  dk_geno_offset (sum over genotypes)   {dk.sum():.1f}")
