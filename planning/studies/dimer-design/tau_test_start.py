"""
Start point for tau_test.sh's flip mode: a MAP params npz with the pair
horseshoe moved to the other regime.

    python tau_test_start.py <fit params.npz> <out.npz>

Every pair offset is set to 0, so every pair shift starts at 0 and the
likelihood starts at the additive model's. A fit whose tau is above 1
(runaway) gets tau 0.01 and c2 8/3, the mode of the slab prior
InverseGamma(2, 8); a fit with tau below 1 gets tau 800 and c2 600, the
runaway fits' typical values. The local scales lambda and every other
site keep the fit's values.
"""

import sys

import numpy as np

SMALL = {"tau": 0.01, "c2": 8.0 / 3.0}
LARGE = {"tau": 800.0, "c2": 600.0}


def main(src, out):
    with np.load(src) as z:
        p = {k: z[k] for k in z.files}
    tau = float(p["theta_epi_tau_auto_loc"])
    new = SMALL if tau > 1 else LARGE
    p["theta_epi_tau_auto_loc"] = np.asarray(new["tau"], dtype=np.float32)
    p["theta_epi_c2_auto_loc"] = np.asarray(new["c2"], dtype=np.float32)
    p["theta_epi_offset_auto_loc"] = np.zeros_like(
        p["theta_epi_offset_auto_loc"])
    np.savez(out, **p)
    print(f"tau {tau:.3g} -> {new['tau']}, c2 "
          f"{float(np.load(src)['theta_epi_c2_auto_loc']):.3g} -> "
          f"{new['c2']:.3g}, pair offsets -> 0; wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
