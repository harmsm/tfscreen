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


def _flag(v):
    """A 0/1 pin flag as written in a priors CSV (0.0, 1, True, ...)."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "1.0", "true")
    return bool(float(v))


def km_from_map(params_npz, priors_csv, name="condition_growth"):
    """
    Per-condition k and m from a MAP's parameters and its priors CSV, for
    when ``tfs-extract-params`` cannot give them: a MAP checkpoint holds no
    deterministic sites, and a pinned k or m (all conditions, or some) is
    one. A pinned condition takes its prior location.

    Parameters
    ----------
    params_npz : str
        ``tfs-fit-model``'s ``_params.npz`` (constrained ``{site}_auto_loc``).
    priors_csv : str
        The configuration's priors CSV, with per-condition (indexed) growth
        rows naming ``condition_rep``.
    name : str
        The growth component's site prefix.

    Returns
    -------
    pandas.DataFrame
        Index ``condition_rep``, columns ``k`` and ``m``.
    """
    z = np.load(params_npz)
    p = pd.read_csv(priors_csv)
    prefix = f"growth.{name}."
    labels = None
    for field in ("k_loc", "m_loc", "k_pinned", "m_pinned", "k_scale"):
        r = p[(p["parameter"] == prefix + field) & p["flat_index"].notna()]
        if len(r):
            labels = r.sort_values("flat_index")["condition_rep"].astype(str).tolist()
            break
    if labels is None:
        raise ValueError(f"{priors_csv} has no per-condition growth rows to "
                         "name the conditions")
    n = len(labels)

    def per_condition(field, default):
        r = p[p["parameter"] == prefix + field]
        if r.empty:
            return np.full(n, default, dtype=object)
        if r["flat_index"].notna().any():
            r = r.sort_values("flat_index")
            return r["value"].to_numpy(dtype=object)
        return np.full(n, r["value"].iloc[0], dtype=object)

    out = {}
    for param in ("k", "m"):
        loc = per_condition(f"{param}_loc", 0.0).astype(float)
        pinned = np.array([_flag(v) for v in per_condition(f"{param}_pinned", 0)])
        full, free = f"{name}_{param}_auto_loc", f"{name}_{param}_free_auto_loc"
        value = loc.copy()
        if full in z.files:
            value = np.asarray(z[full], dtype=float).reshape(n)
        elif free in z.files:
            value[~pinned] = np.asarray(z[free], dtype=float).ravel()
        elif not pinned.all():
            raise ValueError(f"{params_npz} has no {param} for the free "
                             "conditions")
        out[param] = value
    return pd.DataFrame(out, index=pd.Index(labels, name="condition_rep"))


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


# ---------------------------------------------------------------------------
# Growth tables too large to load (a full-size table is about 8 GB)
# ---------------------------------------------------------------------------

CHUNK = 5_000_000


def growth_genotypes(path, chunksize=CHUNK):
    """The genotypes in a growth table, read in chunks."""
    seen = set()
    for c in pd.read_csv(path, usecols=["genotype"], chunksize=chunksize):
        seen.update(c["genotype"].astype(str).unique())
    return seen


def filter_growth_csv(src, dst, genotypes=None, library=None,
                      chunksize=CHUNK):
    """
    Write the rows of a growth table for some genotypes or one library.

    Parameters
    ----------
    src, dst : str
        Input and output CSV paths.
    genotypes : iterable of str or None
        Keep only these genotypes.
    library : str or None
        Keep only this library's tubes.

    Returns
    -------
    int
        Rows written.

    Raises
    ------
    ValueError
        If nothing is kept.
    """
    keep_g = None if genotypes is None else {str(g) for g in genotypes}
    n = 0
    header = True
    with open(dst, "w") as fh:
        for c in pd.read_csv(src, chunksize=chunksize):
            mask = pd.Series(True, index=c.index)
            if keep_g is not None:
                mask &= c["genotype"].astype(str).isin(keep_g)
            if library is not None:
                mask &= c["library"].astype(str) == str(library)
            c = c[mask]
            if len(c):
                c.to_csv(fh, index=False, header=header)
                header = False
                n += len(c)
    if n == 0:
        raise ValueError(f"filtering {src} kept no rows "
                         f"(library={library!r})")
    return n


# ---------------------------------------------------------------------------
# Marker consistency
# ---------------------------------------------------------------------------

def marker_consistency(both_pred, one_pred, z_flag=2.0):
    """
    Per genotype, how far X from one library's data alone departs from X
    from both libraries.

    X from one library is less precise and somewhat shifted and stretched
    (on the full-size simulation, dropping pheS stretched X about 1.08 and
    shifted it about 0.13), so the two are compared after the best affine
    map of one onto the other (weighted by the rows' combined variance,
    non-wt rows). A genotype's level offset is its mean residual over
    concentrations (the disagreement is 91-92% level), and its z is that
    offset over the median of its rows' combined SD. On the simulation
    2.5% of deep doubles' rows fell beyond 2 SD, on the real data 8.2%
    (``planning/studies/m-bias-factorial/``).

    Parameters
    ----------
    both_pred, one_pred : pandas.DataFrame
        ``tfs-predict-theta`` output (``genotype``, ``titrant_name``,
        ``titrant_conc``, ``q0.025``, ``q0.5``, ``q0.975``) from the stage 2
        fit and from the one-library refit.
    z_flag : float
        |z| above which a genotype is flagged.

    Returns
    -------
    pandas.DataFrame
        One row per genotype: ``n_rows``, ``level_offset``, ``level_sd``,
        ``z`` and ``inconsistent``.
    """
    cols = PRED_KEY + ["q0.025", "q0.5", "q0.975"]
    m = both_pred[cols].merge(one_pred[cols], on=PRED_KEY,
                              suffixes=("_both", "_one"))
    m = m[m["genotype"].astype(str) != "wt"]
    sd_b = (m["q0.975_both"] - m["q0.025_both"]) / 3.92
    sd_o = (m["q0.975_one"] - m["q0.025_one"]) / 3.92
    m = m.assign(sd=np.sqrt(sd_b ** 2 + sd_o ** 2))
    ok = np.isfinite(m["q0.5_both"]) & np.isfinite(m["q0.5_one"]) & (m["sd"] > 0)
    m = m[ok]
    if m.empty:
        raise ValueError("no non-wt rows in common between the two fits")
    w = 1.0 / m["sd"] ** 2
    slope, intercept = np.polyfit(m["q0.5_both"], m["q0.5_one"], 1,
                                  w=np.sqrt(w))
    m = m.assign(res=m["q0.5_one"] - (intercept + slope * m["q0.5_both"]))
    g = m.groupby("genotype")
    out = pd.DataFrame({"n_rows": g.size(),
                        "level_offset": g["res"].mean(),
                        "level_sd": g["sd"].median()})
    out["z"] = out["level_offset"] / out["level_sd"]
    out["inconsistent"] = out["z"].abs() > z_flag
    out.attrs["affine"] = (float(intercept), float(slope))
    return out
