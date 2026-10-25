"""No-offset MAP params + per-tube offsets only (N(0, 0.17) prior), each tube
optimized alone: how much of the offset fit's likelihood gain does that get?"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
import sys
sys.path.insert(0, ".")
from residual_compare import nb_logpmf, TUBE
SIG = 0.17
d = pd.read_parquet("resid/sig_vs_nooff_obs.parquet")
z = np.load("map_nooffset_b/tfs_fit_model_params.npz")
phi, inv_r = float(z["growth_phi_auto_loc"]), float(z["growth_inv_r_auto_loc"])
d["log_mu"] = d.ln_sample_reads + d.pred_b - d.sample_ln_cfu
rows = []
for key, g in d.groupby(TUBE):
    k = g.counts.to_numpy(float); lm = g.log_mu.to_numpy()
    base = nb_logpmf(k, lm, phi, inv_r).sum()
    f = lambda x: -(nb_logpmf(k, lm + x, phi, inv_r).sum() - 0.5 * (x / SIG) ** 2)
    r = minimize_scalar(f, bounds=(-3, 3), method="bounded", options={"xatol": 1e-4})
    gain = -r.fun - base
    rows.append(dict(zip(TUBE, key), delta=r.x, gain=gain))
r = pd.DataFrame(rows)
t = pd.read_csv("resid/sig_vs_nooff_tube.csv")
r = r.merge(t[TUBE + ["dll", "offset_a"]], on=TUBE)
r.to_csv("resid/small_offsets.csv", index=False)
print(f"sum gain (ll + offset prior) {r.gain.sum():.4g}; offset-fit dll {r.dll.sum():.4g}")
print(f"delta sd {r.delta.std():.3f}, range {r.delta.min():.2f} to {r.delta.max():.2f}")
print(f"corr(delta, offset_a) {np.corrcoef(r.delta, r.offset_a)[0,1]:.2f}; "
      f"corr(gain, dll) {np.corrcoef(r.gain, r.dll)[0,1]:.2f}")
pd.set_option("display.width", 250)
print(r.pivot_table(index=["condition_pre","condition_sel"], columns="titrant_conc", values="delta").round(2).to_string())
