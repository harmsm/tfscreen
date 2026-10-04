"""
Check that a warm-started run starts where it should.

    python ../check_warm.py ../map_nooffset_b warm_init.npz
    python ../check_warm.py warm_init.npz

Run from the run directory. Reads the config and the --init_from npz the
way tfs-fit-model does and turns them into site values
(RunInference.site_values, what the MAP starts from).

With a reference run, every site of its MAP must start at that MAP's value.
With the npz alone, every site in the npz must start at the npz's value;
sites the model does not sample (a hyperscale held fixed by the priors, say)
are listed and skipped. Either way the tube offsets must be 288 values with
118 nonzero. Exits nonzero on any failure, so the job stops before the fit.
"""

import sys

import numpy as np

from tfscreen.tfmodel.configuration_io import read_configuration
from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.scripts.fit_model_cli import _read_init_from


def main():
    if len(sys.argv) == 3:
        ref_path = f"{sys.argv[1]}/tfs_fit_model_params.npz"
        init_path = sys.argv[2]
    else:
        ref_path = init_path = sys.argv[1]
    ref = np.load(ref_path)
    orch, guesses = read_configuration("tfs_configure_config.yaml")
    guesses = {**guesses, **_read_init_from(init_path)}
    ri = RunInference(orch, 0)
    start = ri.site_values(guesses)
    bad, skipped, checked = [], [], 0
    for key in ref.files:
        site = key[:-len("_auto_loc")]
        if site == "sample_offset_offset" and ref_path == init_path:
            continue
        want = np.asarray(ref[key], dtype=float)
        if site not in start:
            if ref_path == init_path:
                skipped.append(site)
            else:
                bad.append(f"{site}: not in start values")
            continue
        checked += 1
        got = np.asarray(start[site], dtype=float)
        if got.shape != want.shape or not np.allclose(got, want, rtol=1e-5, atol=1e-6):
            err = (np.abs(got - want).max() if got.shape == want.shape
                   else f"shape {got.shape} vs {want.shape}")
            bad.append(f"{site}: max diff {err}")
    off = np.asarray(start.get("sample_offset_offset", np.zeros(0)))
    nz = off[off != 0]
    print(f"sample_offset_offset start: {off.size} values, {nz.size} nonzero, "
          f"sd {nz.std() if nz.size else float('nan'):.3f}")
    print(f"{checked - len(bad)} of {checked} sites start at {ref_path}")
    if skipped:
        print(f"not sampled by this model, skipped: {skipped}")
    if bad or off.size != 288 or nz.size != 118:
        print("\n".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
