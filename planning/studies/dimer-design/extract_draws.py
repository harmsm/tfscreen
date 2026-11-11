"""
Pull each finished fit's posterior draws into a small npz for
predictability.py, run on the cluster where the posteriors live.

    python extract_draws.py [--runs runs] [--out results/draws]

For every runs/<run>/fit_<arm>/tfs_posterior.h5 whose fit has finished
(fit_<arm>/.done/summarize), writes <out>/<run>__fit_<arm>.npz with:

  terms        the fit's per-mutation terms (suffixes after ln_K_), in the
               order of species.active_terms for its variant
  variant      monomers, unfolded_dimer, unfolded_monomer,
               second_engagement (0/1), read off the posterior's wt sites
  d_<t>        (draws, M) per-mutation constant, kT
  wt_<t>       (draws,) wt constant
  mutations    (M,) labels, from tfs_params_d_ln_K_h_l.csv
  epi_<t>      (draws, P) the fit's pair shifts, when it has pairs
  pairs        (P,) labels, from tfs_params_epi_ln_K_h_l.csv
  epi_tau, epi_c2   (draws,) the pair horseshoe's global scale and slab

The label order is the extraction tables' order; each array's median is
checked against its table's q0.5 so a mismatch fails here. An existing
npz newer than its posterior is kept.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd

from tfscreen.tfmodel.generative.components.theta.thermo.dimer import (
    species as sp)

HERE = os.path.dirname(os.path.abspath(__file__))


def _variant(f):
    full = "theta_ln_K_m_u_wt" in f
    return sp.Variant(monomers=full, unfolded_dimer="theta_ln_K_u_wt" in f,
                      unfolded_monomer=full, second_engagement=True)


def _check(arr, table_path, key):
    t = pd.read_csv(table_path)
    if "titrant_name" in t.columns:
        t = t.groupby(key, sort=False, as_index=False).first()
    q = t["q0.5"].to_numpy()
    med = np.median(arr, axis=0)
    scale = max(np.std(q), 1e-6)
    if med.shape != q.shape or np.max(np.abs(med - q)) > 0.05 * scale + 1e-4:
        raise ValueError(f"{table_path}: posterior median does not match "
                         "the table's q0.5 (label order?)")
    return np.array(t[key].astype(str).tolist(), dtype=str)


def extract(fit_dir, out_path):
    with h5py.File(os.path.join(fit_dir, "tfs_posterior.h5"), "r") as f:
        v = _variant(f)
        terms = [t[len("ln_K_"):] for t in sp.active_terms(v)]
        out = {"terms": np.array(terms),
               "variant": np.array([v.monomers, v.unfolded_dimer,
                                    v.unfolded_monomer,
                                    v.second_engagement], dtype=int)}
        for t in terms:
            d = np.asarray(f[f"theta_d_ln_K_{t}"])
            w = np.asarray(f[f"theta_ln_K_{t}_wt"])
            out[f"d_{t}"] = d.reshape(d.shape[0], -1).astype(np.float32)
            out[f"wt_{t}"] = w.reshape(w.shape[0]).astype(np.float32)
        out["mutations"] = _check(out[f"d_{terms[0]}"], os.path.join(
            fit_dir, f"tfs_params_d_ln_K_{terms[0]}.csv"), "mutation")
        if f"theta_epi_ln_K_{terms[0]}" in f:
            for t in terms:
                e = np.asarray(f[f"theta_epi_ln_K_{t}"])
                out[f"epi_{t}"] = e.reshape(e.shape[0], -1).astype(np.float32)
            out["pairs"] = _check(out[f"epi_{terms[0]}"], os.path.join(
                fit_dir, f"tfs_params_epi_ln_K_{terms[0]}.csv"), "pair")
            out["epi_tau"] = np.asarray(f["theta_epi_tau"]).reshape(-1)
            out["epi_c2"] = np.asarray(f["theta_epi_c2"]).reshape(-1)
    np.savez_compressed(out_path, **out)


def main(runs, out):
    os.makedirs(out, exist_ok=True)
    for h5 in sorted(glob.glob(os.path.join(runs, "*", "fit_*",
                                            "tfs_posterior.h5"))):
        fit_dir = os.path.dirname(h5)
        if not os.path.exists(os.path.join(fit_dir, ".done", "summarize")):
            continue
        run = os.path.basename(os.path.dirname(fit_dir))
        arm = os.path.basename(fit_dir)[len("fit_"):]
        path = os.path.join(out, f"{run}__fit_{arm}.npz")
        if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(h5):
            continue
        extract(fit_dir, path)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "draws"))
    a = ap.parse_args()
    main(a.runs, a.out)
