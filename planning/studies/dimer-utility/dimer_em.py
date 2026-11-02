"""
Set the thermo.dimer scales by EM, with the scales held in every MAP.

    ~/miniconda3/bin/python dimer_em.py <run> <arm>     e.g. A2 dimerPK

Joint MAP does not estimate the hierarchical scales (stage A, README): in
the non-centered PK shifts σ_d runs up while the offsets shrink, or falls
to 0 for a term the data do not see, and the centered structural prior's
σ_s collapses to 0. Here the scales are held (``theta_sigma_d_{term}_fixed``,
``theta_ddG_sigma_{state}_fixed``), and EM with a Laplace E-step updates
them between fits:

    σ_term²  = mean over mutations and Laplace draws of d_term²
    σ_state² = mean over mutations and draws of e_state², e = P⁻¹ (Δln K − P μ)

the second for the six structural constants of PddG (P square, so e is the
per-state ΔΔG residual about the table exactly). This is the EM update for
a Normal prior's scale with the posterior moments taken from the Laplace,
so its fixed point is the Laplace-approximate marginal-likelihood estimate.
A term the data do not see keeps its starting σ (its posterior is its
prior), which is the honest answer for it.

Each iteration, in runs/<run>/em_<arm>/iter<k>/: configure with the held
scales, MAP from the previous iteration's point (k = 0: stage A's MAP; PK
offsets rescaled so every shift d = offset · σ starts where it was), full
Laplace, E-step. Stops when every scale changes by less than TOL
(relative) or after MAX_ITER iterations. ``em_<arm>/trace.csv`` holds the
scales and the exact loss per iteration. Rerunning resumes.

Environment: MAP_CAP (epochs per MAP, default 30000), NUM_DRAWS (500),
MAX_ITER (6), TOL (0.05), SIGMA0 (starting scale in kT, default 1.0).
"""

import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import (
    PddG, model as dm, species as sp,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.expanduser("~/miniconda3/bin")
MAP_CAP = int(os.environ.get("MAP_CAP", 30000))
NUM_DRAWS = int(os.environ.get("NUM_DRAWS", 500))
MAX_ITER = int(os.environ.get("MAX_ITER", 6))
TOL = float(os.environ.get("TOL", 0.05))
SIGMA0 = float(os.environ.get("SIGMA0", 1.0))

THERMO = "../../../../inputs/2026-10-11_laci-state-ddg.csv"
ARMS = {
    "dimerPK": ["--theta_model", "thermo.dimer.PK"],
    "dimerPddG": ["--theta_model", "thermo.dimer.PddG",
                  "--thermo_data", THERMO,
                  "--thermo_ddG_units", "kcal/mol",
                  "--thermo_temperature_K", "310.15"],
}


def _free_terms(arm):
    """The terms with a hierarchical σ_d, in the model's offset order."""
    v = sp.Variant()
    terms = list(sp.genotype_terms(v)) + list(sp.titrant_terms(v))
    if arm == "dimerPddG":
        terms = [t for t in terms if t not in dm.STRUCT_TERMS]
    return terms


def _run(cmd, log):
    with open(log, "a") as f:
        f.write(">>> " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def _set_priors(arm, scales):
    items = [f"theta_sigma_d_{t}_fixed={scales[t]:.6g}"
             for t in _free_terms(arm)]
    if arm == "dimerPddG":
        items += [f"theta_ddG_sigma_{s}_fixed={scales[s]:.6g}"
                  for s in PddG.STRUCTURE_NAMES]
    return ["--set_priors"] + items


def _init_npz(src, dst, arm, old, new, old_free_sigma=None):
    """
    Copy a MAP's params for ``--init_from``, with each PK offset rescaled so
    its shift stays put under the new held σ and the σ sites dropped.
    """
    z = dict(np.load(src))
    terms = _free_terms(arm)
    key = "theta_d_offset_auto_loc"
    if key in z:
        prev = np.asarray(old_free_sigma if old_free_sigma is not None
                          else [old[t] for t in terms])
        ratio = prev / np.asarray([new[t] for t in terms])
        z[key] = z[key] * ratio[:, None]
    for k in ("theta_sigma_d_auto_loc", "theta_ddG_sigma_auto_loc"):
        z.pop(k, None)
    np.savez(dst, **z)


def _e_step(h5, arm):
    """New scales from the Laplace draws."""
    out = {}
    with h5py.File(h5) as f:
        for t in _free_terms(arm):
            d = np.asarray(f[f"theta_d_{t}"])
            out[t] = float(np.sqrt(np.mean(d ** 2)))
        if arm == "dimerPddG":
            dlnK = np.asarray(f["theta_dlnK"])                 # (N, M, 6)
            mu = np.asarray(f["theta_ddG_prior_mean"])         # (N, M, 6)
            P = PddG.DLNK_PROJ
            e = (dlnK - mu @ P.T) @ np.linalg.inv(P).T         # (N, M, S)
            for i, s in enumerate(PddG.STRUCTURE_NAMES):
                out[s] = float(np.sqrt(np.mean(e[..., i] ** 2)))
    return out


def _exact_loss(fit_dir):
    c = pd.read_csv(os.path.join(fit_dir, "tfs_fit_model_convergence.csv"))
    last = c.iloc[-1]
    return float(last["loss_exact"]), last["decision"], int(last["step"])


def main(run, arm):
    base = os.path.join(HERE, "runs", run)
    em = os.path.join(base, f"em_{arm}")
    os.makedirs(em, exist_ok=True)
    names = _free_terms(arm) + (list(PddG.STRUCTURE_NAMES)
                                if arm == "dimerPddG" else [])
    trace_csv = os.path.join(em, "trace.csv")
    rows = pd.read_csv(trace_csv).to_dict("records") \
        if os.path.exists(trace_csv) else []
    scales = {n: SIGMA0 for n in names} if not rows else \
        {n: rows[-1][f"new_{n}"] for n in names}

    for k in range(len(rows), MAX_ITER):
        it = os.path.join(em, f"iter{k}")
        os.makedirs(it, exist_ok=True)
        log = os.path.join(it, "job.log")
        os.chdir(it)
        init = os.path.join(it, "init_params.npz")
        if k == 0:
            src = os.path.join(base, f"fit_{arm}", "tfs_fit_model_params.npz")
            z = np.load(src)
            old_free = (np.asarray(z["theta_sigma_d_auto_loc"])
                        if "theta_sigma_d_auto_loc" in z else None)
            _init_npz(src, init, arm, scales, scales, old_free_sigma=old_free)
        else:
            prev = os.path.join(em, f"iter{k - 1}")
            _init_npz(os.path.join(prev, "tfs_fit_model_params.npz"), init,
                      arm, {n: rows[-1][f"old_{n}"] for n in names}, scales)
        if not os.path.exists("tfs_configure_config.yaml"):
            _run([f"{BIN}/tfs-configure-model",
                  "--growth_df", "../../sim/tfs_sim_growth.csv",
                  "--binding_df", "../../sim/tfs_sim_binding.csv",
                  "--library_config", "../../tfs_sim_config.yaml",
                  "--growth_shares_replicates", "--epistasis",
                  *ARMS[arm], *_set_priors(arm, scales)], log)
        if not os.path.exists("tfs_fit_model_params.npz"):
            _run([f"{BIN}/tfs-fit-model", "tfs_configure_config.yaml",
                  "--analysis_method", "map", "--seed", "1",
                  "--epoch_checkpoint_interval", "0",
                  "--max_num_epochs", str(MAP_CAP),
                  "--adam_step_size", "1e-4",
                  "--init_from", init], log)
        if not os.path.exists("tfs_posterior.h5"):
            _run([f"{BIN}/tfs-sample-posterior", "tfs_configure_config.yaml",
                  "tfs_fit_model_checkpoint.pkl", "--laplace", "full",
                  "--num_posterior_samples", str(NUM_DRAWS),
                  "--skip_growth_observations"], log)
        new = _e_step("tfs_posterior.h5", arm)
        loss, decision, step = _exact_loss(it)
        row = {"iter": k, "loss_exact": loss, "decision": decision,
               "steps": step}
        row.update({f"old_{n}": scales[n] for n in names})
        row.update({f"new_{n}": new[n] for n in names})
        rows.append(row)
        pd.DataFrame(rows).to_csv(trace_csv, index=False)
        change = max(abs(new[n] - scales[n]) / scales[n] for n in names)
        print(f"iter {k}: loss {loss:.1f} ({decision}), max rel change "
              f"{change:.3f}", flush=True)
        print("  " + ", ".join(f"{n} {scales[n]:.2f}->{new[n]:.2f}"
                               for n in names), flush=True)
        scales = new
        if change < TOL:
            break


if __name__ == "__main__":
    main(*sys.argv[1:3])
