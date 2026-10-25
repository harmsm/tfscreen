"""
Warm-start a level-offset MAP from the no-offset MAP.

    python make_warm_start.py <offset_run> <nooffset_run> <out_dir>

offset_run:   a finished level-offset run (sigma_fixed 0.17), for its config,
              priors, guesses and library snapshot (e.g. map_sig017b)
nooffset_run: the no-offset MAP whose point is the start (map_nooffset_b)
out_dir:      new run directory, a sibling of the others (config paths are
              ../inputs/...)

Writes <out_dir>/warm_init.npz for tfs-fit-model --init_from: every
{site}_auto_loc array of the no-offset MAP, plus the tube offsets
(sample_offset_offset_auto_loc) at the per-tube values from
resid/small_offsets.py (the best offset per tube with everything else held
at the no-offset MAP). Run it with ARM=map WARM=1 (run.srun skips configure
and prefit, checks the start with check_warm.py, then fits).
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

TUBE = ["replicate", "condition_pre", "condition_sel", "titrant_conc", "t_sel"]
TUBE_SHAPE = (3, 3, 2, 2, 1, 8)
IDX = ["replicate_idx", "time_idx", "condition_pre_idx",
       "condition_sel_idx", "titrant_name_idx", "titrant_conc_idx"]


def main():
    off_run, nooff_run, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=False)
    for f in ("tfs_configure_config.yaml", "tfs_configure_priors.csv",
              "tfs_configure_guesses.csv", "tfs_configure_library.csv"):
        shutil.copy(os.path.join(off_run, f), os.path.join(out, f))

    z = np.load(os.path.join(nooff_run, "tfs_fit_model_params.npz"))
    init = {k: z[k] for k in z.files}

    so = pd.read_csv("resid/small_offsets.csv")
    ti = pd.read_csv("resid/tube_index.csv")
    so = so.merge(ti, on=TUBE, how="left")
    assert so[IDX].notna().all().all() and len(so) == 118
    flat = np.ravel_multi_index(tuple(so[c].astype(int) for c in IDX), TUBE_SHAPE)
    off = np.zeros(int(np.prod(TUBE_SHAPE)), dtype=np.float32)
    off[flat] = so.delta.to_numpy()
    init["sample_offset_offset_auto_loc"] = off

    np.savez(os.path.join(out, "warm_init.npz"), **init)
    print(f"{out}/warm_init.npz: {len(init)} sites from {nooff_run} "
          f"+ tube offsets (118 set, sd {so.delta.std():.3f})")


if __name__ == "__main__":
    main()
