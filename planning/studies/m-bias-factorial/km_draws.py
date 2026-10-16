"""
Draws of k and m from a stage-1 fit's Laplace, one held-k/m arm per draw
(planning/deep-coverage.md step 2).

    python km_draws.py ../m_subsets/no_doubles --num_draws 10 --out_dir km_draws

The stage-1 directory needs tfs_posterior.h5 (tfs-sample-posterior on its
MAP checkpoint) and tfs_configure_priors.csv (whose per-condition rows name
the conditions in the model's order). Each draw is one joint sample of every
condition's k and m, so their correlations carry over. Each draw directory
gets growth_priors_held.csv with that draw as k_loc and m_loc; run_held.srun
pins every condition at it (its default PINS).

The k/m uncertainty is only as good as the Laplace: a k or m direction the
arrowhead Laplace held at zero width (tfs_posterior_held_directions.csv)
has no spread in the draws, so the script stops if one does, unless
--allow_held.
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd

PREFIX = "growth.condition_growth."


def condition_order(priors_csv):
    """condition_rep labels in the model's order, from the indexed k_loc rows."""
    df = pd.read_csv(priors_csv)
    rows = df[df["parameter"] == PREFIX + "k_loc"].dropna(subset=["flat_index"])
    if rows.empty or "condition_rep" not in rows:
        raise ValueError(f"{priors_csv} has no per-condition k_loc rows to "
                         "name the conditions (configure with --growth_priors).")
    if "replicate" in rows and rows["replicate"].notna().any():
        raise ValueError("per-replicate conditions are not supported; configure "
                         "with --growth_shares_replicates.")
    return list(rows.sort_values("flat_index")["condition_rep"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1_dir")
    ap.add_argument("--num_draws", type=int, default=10)
    ap.add_argument("--out_dir", default="km_draws")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--allow_held", action="store_true")
    args = ap.parse_args()
    s1 = args.stage1_dir

    held_file = os.path.join(s1, "tfs_posterior_held_directions.csv")
    if os.path.exists(held_file):
        held = pd.read_csv(held_file)
        km = held[held["parameter"].str.startswith("condition_growth_")
                  & (held["loading"].abs() > 0.1)]
        if not km.empty:
            print("Held directions with k or m loadings above 0.1:")
            print(km.to_string(index=False))
            if not args.allow_held:
                raise SystemExit("k/m directions are held at zero width; the "
                                 "draws would understate their uncertainty "
                                 "(--allow_held to go on anyway).")

    labels = condition_order(os.path.join(s1, "tfs_configure_priors.csv"))
    with h5py.File(os.path.join(s1, "tfs_posterior.h5"), "r") as hf:
        k = np.asarray(hf["condition_growth_k"]).reshape(-1, len(labels))
        m = np.asarray(hf["condition_growth_m"]).reshape(-1, len(labels))

    # The extracted medians must match the posterior's columns by name.
    for name, arr in (("k", k), ("m", m)):
        ext = pd.read_csv(os.path.join(s1, f"tfs_params_growth_{name}.csv"))
        ext = ext.groupby("condition_rep")["q0.5"].mean()
        med = pd.Series(np.median(arr, axis=0), index=labels)
        off = (med - ext.reindex(labels)).abs().max()
        if not off < 1e-3:
            raise ValueError(f"{name}: posterior medians {med.round(5).to_dict()} "
                             f"do not match tfs_params_growth_{name}.csv "
                             f"{ext.round(5).to_dict()}; condition order?")

    summary = pd.DataFrame({"k_sd": k.std(axis=0), "m_sd": m.std(axis=0),
                            "k_median": np.median(k, axis=0),
                            "m_median": np.median(m, axis=0)}, index=labels)
    print(f"{k.shape[0]} posterior draws of k and m:")
    print(summary.to_string(float_format="%.5f"))

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(k.shape[0], size=args.num_draws, replace=False)
    rows = []
    for i, j in enumerate(pick):
        d = os.path.join(args.out_dir, f"draw_{i:02d}")
        os.makedirs(d, exist_ok=True)
        table = pd.DataFrame({"condition_rep": labels, "k_loc": k[j],
                              "k_scale": 0.002, "m_loc": m[j], "m_scale": 0.002})
        table.to_csv(os.path.join(d, "growth_priors_held.csv"), index=False)
        rows.append(table.assign(draw=i, sample=int(j)))
    pd.concat(rows).to_csv(os.path.join(args.out_dir, "km_draws.csv"),
                           index=False)
    print(f"wrote {args.num_draws} draws to {args.out_dir}/draw_*/ "
          f"(table: {args.out_dir}/km_draws.csv)")


if __name__ == "__main__":
    main()
