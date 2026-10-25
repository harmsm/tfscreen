"""
Combine a MAP's parameters with another MAP's tube offsets into one
--init_from file.

    python make_init_npz.py <params_run> <offsets_run> <out.npz>

params_run:  every {site}_auto_loc of its tfs_fit_model_params.npz
offsets_run: its sample_offset_offset_auto_loc only (same 288-tube layout:
             same tube grid and condition order)
2026-10-01: map_nooffset_n05 (n held, nine spikes) + map_warm (small
offsets, SD 0.19) -> inputs/init_n05_warm.npz, for run.srun INIT_FROM.
"""
import sys

import numpy as np


def main():
    params_run, offsets_run, out = sys.argv[1:4]
    p = np.load(f"{params_run}/tfs_fit_model_params.npz")
    o = np.load(f"{offsets_run}/tfs_fit_model_params.npz")
    init = {k: p[k] for k in p.files}
    assert "sample_offset_offset_auto_loc" not in init
    off = np.asarray(o["sample_offset_offset_auto_loc"])
    assert off.shape == (288,), off.shape
    init["sample_offset_offset_auto_loc"] = off
    np.savez(out, **init)
    nz = off[off != 0]
    print(f"{out}: {len(p.files)} sites from {params_run} + tube offsets from "
          f"{offsets_run} ({nz.size} nonzero, sd {nz.std():.3f})")


if __name__ == "__main__":
    main()
