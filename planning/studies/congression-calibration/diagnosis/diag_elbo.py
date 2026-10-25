"""
Does the ELBO (averaged over guide draws) prefer the mixture fit's endpoint
while the joint density at median latents prefers single's solution? That
would mean guide noise, passed through the mixture's convex logsumexp and
max-theta, is what drives the SVI drift. Run from this directory; GRID
selects the grid.
"""
import sys
sys.argv = [sys.argv[0]]
from diag_lam0 import *
from numpyro.infer import Trace_ELBO

N_DRAWS = 64
(d1, o1, s1, st1), (d2, o2, s2, st2) = load("run_0001"), load("run_0002")
p1, p2 = s1.get_params(st1), s2.get_params(st2)
start = {k: (p1[k] if k in p1 else p2[k]) for k in p2}
elbo = Trace_ELBO(num_particles=1)


def mean_loss(params, orch, svi, n=N_DRAWS):
    f = jax.jit(lambda p, key: elbo.loss(key, p, orch.jax_model, svi.guide,
                                          data=full_batch(orch), priors=orch.priors))
    v = np.array([float(f(params, jax.random.PRNGKey(i))) for i in range(n)])
    return v.mean(), v.std() / np.sqrt(n)


for tag, params in (("single's solution (+ mixture lambda)", start),
                    ("mixture's own endpoint", p2)):
    m, se = mean_loss(params, o2, s2)
    print(f"mixture -ELBO at {tag:38s} {m:12.1f} +/- {se:.1f}")
m, se = mean_loss(p1, o1, s1)
print(f"single  -ELBO at {'single solution':38s} {m:12.1f} +/- {se:.1f}")

# Same comparison with the guide's scales shrunk toward zero (near-point
# evaluation): if the ordering flips, guide noise is what favors the drift.
def shrink(p, f=1e-3):
    return {k: (v * f if k.endswith("_scale") or k.endswith("_scales") else v)
            for k, v in p.items()}
for tag, params in (("single's solution, scales x1e-3", shrink(start)),
                    ("mixture endpoint, scales x1e-3", shrink(p2))):
    m, se = mean_loss(params, o2, s2, n=8)
    print(f"mixture -ELBO at {tag:38s} {m:12.1f} +/- {se:.1f}")
