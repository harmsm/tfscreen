"""
Re-express a relative-fit MAP's Hill offsets for held population SDs, as an
--init_from file.

    python make_init_hs.py <map_run> <out.npz> X_low=0.5 X_delta=0.5 log_hill_K=1.0

hill_relative is non-centered: value = hyper_loc + offset * hyper_scale. A
run that holds a hyperscale at a new value would put every genotype
somewhere else if it started from the old offsets, so each offset is
rescaled by old_scale / new_scale (same per-genotype values at the start)
and the learned hyperscale site, which the new model does not sample, is
dropped. 2026-10-02: rel_off_n05 -> inputs/init_rel_off_hs.npz.
"""
import sys

import numpy as np


def main():
    run, out = sys.argv[1:3]
    held = {k: float(v) for k, v in (a.split("=") for a in sys.argv[3:])}
    p = np.load(f"{run}/tfs_fit_model_params.npz")
    init = {k: np.asarray(p[k]) for k in p.files}
    for h, new in held.items():
        scale_key = f"theta_{h}_hyper_scale_auto_loc"
        off_key = f"theta_{h}_offset_auto_loc"
        old = init.pop(scale_key)
        init[off_key] = init[off_key] * (old[:, None] / new)
        z = init[off_key]
        print(f"{h}: scale {float(old[0]):.3f} -> {new}; offsets now "
              f"median |z| {np.median(np.abs(z)):.2f}, 99th pct {np.percentile(np.abs(z), 99):.1f}")
    np.savez(out, **init)
    print(f"wrote {out} ({len(init)} sites)")


if __name__ == "__main__":
    main()
