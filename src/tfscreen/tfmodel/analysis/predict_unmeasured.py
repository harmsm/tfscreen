"""
Shared utilities for predict_unmeasured across theta components.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def _build_genotype_indicators(
    target_genotypes: List[str],
    mut_labels: List[str],
    pair_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build binary indicator matrices and validity flags for a list of genotypes.

    Parameters
    ----------
    target_genotypes : list[str]
        Genotype strings to predict. Format: slash-separated mutations (e.g.,
        "M42I/K84L") or "wt" for wild-type.
    mut_labels : list[str]
        Ordered list of single-mutation labels seen during training.
    pair_labels : list[str]
        Ordered list of mutation-pair labels seen during training, in canonical
        "A/B" form (alphabetically sorted).

    Returns
    -------
    mut_mat : np.ndarray, shape (N, M), float32
        Binary. mut_mat[g, m] = 1 iff mutation m is present in genotype g.
    pair_mat : np.ndarray, shape (N, P), float32
        Binary. pair_mat[g, p] = 1 iff pair p is present in genotype g.
        Only pairs in pair_labels contribute; novel pairs contribute 0,
        meaning pure additivity is assumed for unobserved combinations.
    is_valid : np.ndarray, shape (N,), bool
        True iff every mutation in the genotype string is in mut_labels.
        Genotypes with any unrecognised mutation should be set to NaN downstream.
    """
    mut_to_idx  = {m: i for i, m in enumerate(mut_labels)}
    pair_to_idx = {p: i for i, p in enumerate(pair_labels)}
    N, M, P = len(target_genotypes), len(mut_labels), len(pair_labels)

    mut_mat  = np.zeros((N, M), dtype=np.float32)
    pair_mat = np.zeros((N, P), dtype=np.float32)
    is_valid = np.ones(N, dtype=bool)

    for g_idx, genotype in enumerate(target_genotypes):
        parts = [p for p in genotype.split("/") if p and p.lower() != "wt"]

        for m in parts:
            if m not in mut_to_idx:
                is_valid[g_idx] = False
                break
            mut_mat[g_idx, mut_to_idx[m]] = 1.0

        if not is_valid[g_idx]:
            continue

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                a, b = sorted([parts[i], parts[j]])
                key = f"{a}/{b}"
                if key in pair_to_idx:
                    pair_mat[g_idx, pair_to_idx[key]] = 1.0

    return mut_mat, pair_mat, is_valid


def _build_prediction_grid(
    target_genotypes: List[str],
    titrant_names: List[str],
    manual_titrant_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the full (genotype × titrant_name × titrant_conc) prediction grid.

    Parameters
    ----------
    target_genotypes : list[str]
    titrant_names : list[str]
        Ordered titrant names matching the T dimension in the posterior.
    manual_titrant_df : pd.DataFrame
        Must have 'titrant_name' and 'titrant_conc' columns. All titrant_name
        values must appear in titrant_names.

    Returns
    -------
    calc_df : pd.DataFrame
        Columns: 'genotype', 'titrant_name', 'titrant_conc', '_geno_idx',
        '_titrant_idx'. One row per (genotype × titrant_name × titrant_conc).
    geno_idx : np.ndarray, shape (N_rows,), int
        Index of each row's genotype in target_genotypes.
    titrant_idx : np.ndarray, shape (N_rows,), int
        Index of each row's titrant_name in titrant_names (T dimension order).
    """
    titrant_to_idx = {t: i for i, t in enumerate(titrant_names)}
    geno_to_idx    = {g: i for i, g in enumerate(target_genotypes)}

    dfs = []
    for g in target_genotypes:
        gdf = manual_titrant_df[["titrant_name", "titrant_conc"]].copy()
        gdf["genotype"] = g
        dfs.append(gdf)
    calc_df = pd.concat(dfs, ignore_index=True)

    calc_df["_geno_idx"] = calc_df["genotype"].map(geno_to_idx).astype(int)
    calc_df["_titrant_idx"] = calc_df["titrant_name"].map(titrant_to_idx)

    missing = calc_df["_titrant_idx"].isna()
    if missing.any():
        bad = calc_df.loc[missing, "titrant_name"].unique()
        raise ValueError(
            f"manual_titrant_df contains titrant_name values not in the model: {bad}. "
            f"Valid names are: {titrant_names}"
        )
    calc_df["_titrant_idx"] = calc_df["_titrant_idx"].astype(int)

    geno_idx    = calc_df["_geno_idx"].values
    titrant_idx = calc_df["_titrant_idx"].values
    return calc_df, geno_idx, titrant_idx
