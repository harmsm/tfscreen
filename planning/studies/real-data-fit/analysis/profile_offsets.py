"""Fair count-likelihood comparison of two fits whose tube offsets are not
both readable: rebuild each from parameter medians (residual_compare.predict),
then refit one offset per tube under N(0, 0.17) with everything else held.
Usage: python resid/profile_offsets.py armA armB"""
import sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
sys.path.insert(0, ".")
from residual_compare import nb_logpmf, predict, dispersion, GROWTH, TUBE
SIG = 0.17
obs = pd.read_csv(GROWTH, usecols=TUBE + ["genotype", "t_pre", "counts", "sample_reads", "sample_ln_cfu"])
obs = obs[obs.genotype != "__unknown__"].reset_index(drop=True)
obs["ln_sample_reads"] = np.log(obs.sample_reads)
groups = obs.groupby(TUBE).indices
for arm in sys.argv[1:]:
    phi, inv_r = dispersion(sys.argv[1])  # first arm's dispersion for every arm
    lm = obs.ln_sample_reads.to_numpy() + predict(arm, obs) - obs.sample_ln_cfu.to_numpy()
    k = obs.counts.to_numpy(float)
    tot, deltas = 0.0, []
    for key, idx in groups.items():
        f = lambda x: -(nb_logpmf(k[idx], lm[idx] + x, phi, inv_r).sum() - 0.5 * (x / SIG) ** 2)
        r = minimize_scalar(f, bounds=(-3, 3), method="bounded", options={"xatol": 1e-4})
        tot += -r.fun; deltas.append(r.x)
    print(f"{arm}: phi {phi:.3g} inv_r {inv_r:.3g}; ll + offset prior, offsets refit: {tot:.6g}; offsets sd {np.std(deltas):.3f}")
