"""
Pieces of the two-stage fit (``planning/two-stage-fit.md``).

The full-library MAP of a growth-only relative fit makes the growth slope
m too steep (each double mutant pulls it a little, and the pulls add), and
congression flattens a marker's m in any stage 1 large enough to pin X's
scale. The two-stage fit estimates every condition's k and m from a
subset (wt, the spikes and the singles) with the congression mixture,
then fits the full library with them pinned
(``planning/studies/m-bias-factorial/``). k's uncertainty comes from a
delete-d jackknife of stage 1 and reaches X as one shift shared by every
non-wt genotype (``analysis/shared_shift.py``).

These functions work on DataFrames; the command that chains them (with
configure, fit, sample and predict) is ``tfs-fit-two-stage``.
"""

import numpy as np
import pandas as pd

TUBE_COLUMNS = ["sample", "library", "condition_sel", "titrant_conc"]
STRATA = ("condition_sel", "titrant_conc")
PRED_KEY = ["genotype", "titrant_name", "titrant_conc"]


# ---------------------------------------------------------------------------
# Stage 1 genotypes
# ---------------------------------------------------------------------------

def is_single(genotype):
    """True for a genotype with exactly one mutation (no '/', not wt)."""
    g = str(genotype)
    return g != "wt" and not g.startswith("__") and "/" not in g


def stage1_genotypes(library_df, genotypes):
    """
    The stage-1 genotypes among those with growth data.

    Stage 1 is wt, every genotype in the spiked origin and every single
    mutant: about 950 genotypes in the dev-data screen, few enough that
    their pull on m is small, and enough to pin X's scale (the spikes alone
    are not).

    Parameters
    ----------
    library_df : pandas.DataFrame
        The library composition table (``genotype``, ``in_spiked_origin``;
        ``genetics.library_composition_table`` or the configure snapshot).
    genotypes : iterable of str
        Genotypes with growth data.

    Returns
    -------
    list of str
        Sorted; always includes wt when it has data.
    """
    present = {str(g) for g in genotypes}
    spiked = set(library_df.loc[library_df["in_spiked_origin"].astype(bool),
                                "genotype"].astype(str))
    keep = {g for g in present if g == "wt" or g in spiked or is_single(g)}
    return sorted(keep)


# ---------------------------------------------------------------------------
# Jackknife subsamples of tubes
# ---------------------------------------------------------------------------

def tube_table(growth_df):
    """One row per tube: sample, library, condition_sel, titrant_conc."""
    missing = [c for c in TUBE_COLUMNS if c not in growth_df.columns]
    if missing:
        raise ValueError(f"growth table lacks tube column(s) {missing}")
    return (growth_df[TUBE_COLUMNS].drop_duplicates("sample")
            .reset_index(drop=True))


def choose_tube_drops(tubes, drop_frac, seed, strata=STRATA):
    """
    Tubes to drop for one delete-d jackknife subsample.

    Drops ``round(drop_frac * n)`` tubes at random from each stratum (by
    default condition and IPTG level), always leaving at least one. Tubes
    cannot be resampled with replacement: a duplicated tube lands on its
    original's tensor cell.

    Parameters
    ----------
    tubes : pandas.DataFrame
        ``tube_table`` output.
    drop_frac : float
        Share of each stratum to drop, in [0, 1).
    seed : int
        Seed for this subsample.
    strata : sequence of str
        Columns defining the strata.

    Returns
    -------
    set of str
        The dropped tubes' ``sample`` names.
    """
    if not 0 <= drop_frac < 1:
        raise ValueError(f"drop_frac must be in [0, 1), got {drop_frac}")
    rng = np.random.default_rng(seed)
    drop = []
    for _, g in tubes.groupby(list(strata), sort=True):
        k = min(int(round(drop_frac * len(g))), len(g) - 1)
        if k > 0:
            drop.extend(rng.choice(g["sample"].to_numpy(), size=k,
                                   replace=False))
    return set(drop)


def jackknife_subsample(growth_df, drop_frac, seed, strata=STRATA):
    """
    A growth table with one jackknife subsample's tubes dropped.

    Returns
    -------
    tuple
        ``(subsample_df, n, d, dropped)``: the kept rows, the number of
        tubes, the number dropped and their sorted names.
    """
    tubes = tube_table(growth_df)
    dropped = choose_tube_drops(tubes, drop_frac, seed, strata)
    keep = ~growth_df["sample"].isin(dropped)
    return (growth_df[keep].reset_index(drop=True), len(tubes), len(dropped),
            sorted(dropped))


# ---------------------------------------------------------------------------
# k and m from the jackknife
# ---------------------------------------------------------------------------

def km_table(k_df, m_df):
    """
    Per-condition k and m from extracted parameter tables.

    Parameters
    ----------
    k_df, m_df : pandas.DataFrame
        ``tfs-extract-params`` output for growth_k and growth_m
        (``condition_rep``, ``q0.5``; one row per condition with
        ``growth_shares_replicates``, otherwise averaged over replicates).

    Returns
    -------
    pandas.DataFrame
        Index ``condition_rep``, columns ``k`` and ``m``.
    """
    k = k_df.groupby("condition_rep")["q0.5"].mean()
    m = m_df.groupby("condition_rep")["q0.5"].mean()
    out = pd.DataFrame({"k": k, "m": m})
    if out.isna().any().any():
        raise ValueError("k and m tables name different conditions")
    return out


def jackknife_km(estimates, n, d):
    """
    Jackknife standard errors of k and m, and k's main direction.

    For n tubes, d dropped per subsample and subsample estimates t_i with
    mean t_bar, the delete-d jackknife variance is
    (n - d) / d * mean((t_i - t_bar)^2).

    Parameters
    ----------
    estimates : sequence of pandas.DataFrame
        ``km_table`` of each subsample.
    n, d : int
        Tubes per full table and tubes dropped per subsample.

    Returns
    -------
    dict
        ``se``: DataFrame (index ``condition_rep``) with ``k_mean``,
        ``k_se``, ``m_mean``, ``m_se``; ``k_cov``: k's covariance across
        conditions (jackknife-scaled); ``k_step``: one standard deviation
        along its first principal component (a Series by condition, sign
        fixed so its largest entry is positive); ``k_share``: that
        component's share of k's variance; ``factor``: (n - d) / d.
    """
    if len(estimates) < 2:
        raise ValueError("the jackknife needs at least two subsamples")
    if not 0 < d < n:
        raise ValueError(f"need 0 < d < n, got n={n}, d={d}")
    conds = estimates[0].index
    for e in estimates[1:]:
        if set(e.index) != set(conds):
            raise ValueError("subsamples name different conditions")
    factor = (n - d) / d
    k = pd.DataFrame([e["k"].reindex(conds) for e in estimates])
    m = pd.DataFrame([e["m"].reindex(conds) for e in estimates])
    se = pd.DataFrame({
        "k_mean": k.mean(), "k_se": np.sqrt(factor * k.var(ddof=0)),
        "m_mean": m.mean(), "m_se": np.sqrt(factor * m.var(ddof=0)),
    })
    se.index.name = "condition_rep"
    dev = (k - k.mean()) * np.sqrt(factor)
    cov = dev.T.to_numpy() @ dev.to_numpy() / len(k)
    w, v = np.linalg.eigh(cov)
    top = int(np.argmax(w))
    vec = v[:, top] * np.sqrt(max(w[top], 0.0))
    if vec[np.argmax(np.abs(vec))] < 0:
        vec = -vec
    share = float(w[top] / w.sum()) if w.sum() > 0 else float("nan")
    return {"se": se,
            "k_cov": pd.DataFrame(cov, index=conds, columns=conds),
            "k_step": pd.Series(vec, index=conds, name="k_step"),
            "k_share": share,
            "factor": factor}


# ---------------------------------------------------------------------------
# Pinned growth priors for stage 2
# ---------------------------------------------------------------------------

def pinned_growth_priors(km, scale=0.002, conditions=None):
    """
    A ``--growth_priors`` table pinning every condition's k and m.

    Parameters
    ----------
    km : pandas.DataFrame
        Index ``condition_rep``, columns ``k`` and ``m`` (``km_table``).
    scale : float
        The prior scales written alongside (unused while pinned).
    conditions : sequence of str or None
        Keep only these conditions (e.g. one library's), else all.

    Returns
    -------
    pandas.DataFrame
        ``condition_rep``, ``k_loc``, ``k_scale``, ``m_loc``, ``m_scale``,
        ``k_pinned`` and ``m_pinned`` (1), for
        ``tfs-configure-model --growth_priors``.
    """
    sub = km if conditions is None else km.loc[list(conditions)]
    return pd.DataFrame({
        "condition_rep": sub.index.astype(str),
        "k_loc": sub["k"].to_numpy(), "k_scale": scale,
        "m_loc": sub["m"].to_numpy(), "m_scale": scale,
        "k_pinned": 1, "m_pinned": 1,
    })


def k_shift_priors(km, k_step, scale=0.002):
    """
    Pinned tables with k one step up and down (m unchanged).

    Returns
    -------
    tuple of pandas.DataFrame
        ``(plus, minus)``, each a ``pinned_growth_priors`` table.
    """
    step = k_step.reindex(km.index)
    if step.isna().any():
        raise ValueError("k_step does not cover every condition")
    plus = km.assign(k=km["k"] + step)
    minus = km.assign(k=km["k"] - step)
    return pinned_growth_priors(plus, scale), pinned_growth_priors(minus, scale)


# ---------------------------------------------------------------------------
# The shared X shift
# ---------------------------------------------------------------------------

def shift_table(plus_pred, minus_pred):
    """
    X's change per standard deviation of k, from the two shifted refits.

    Parameters
    ----------
    plus_pred, minus_pred : pandas.DataFrame
        ``tfs-predict-theta`` output of the stage 2 refits with k one step
        up and down (``genotype``, ``titrant_name``, ``titrant_conc``,
        ``q0.5``).

    Returns
    -------
    pandas.DataFrame
        ``genotype``, ``titrant_name``, ``titrant_conc`` and ``shift`` =
        (X_plus - X_minus) / 2 (signed; wt set to 0), for
        ``--shared_shift``.
    """
    p = plus_pred[PRED_KEY + ["q0.5"]].rename(columns={"q0.5": "x_plus"})
    m = minus_pred[PRED_KEY + ["q0.5"]].rename(columns={"q0.5": "x_minus"})
    df = p.merge(m, on=PRED_KEY, how="inner")
    if len(df) != len(p) or len(df) != len(m):
        raise ValueError("the plus and minus predictions cover different rows")
    df["shift"] = (df["x_plus"] - df["x_minus"]) / 2
    df.loc[df["genotype"].astype(str) == "wt", "shift"] = 0.0
    return df[PRED_KEY + ["shift"]]
