"""
Edits to a configured model's priors CSV.

``tfs-configure-model`` writes ``{out_prefix}_priors.csv`` with every prior at
its component default. This module changes it in the documented ways, so
nobody edits the file by hand:

- ``apply_prior_overrides``: set scalar priors by name
  (``tfs-configure-model --set_priors name=value ...``), such as a held
  population SD (``theta_log_hill_n_hyper_scale_fixed``) or the tube-offset
  SD (``sigma_fixed``).
- ``growth_prior_updates``: per-condition growth k/m priors from a table
  (``--growth_priors``), for models the pre-fit does not calibrate.
- ``growth_priors_from_wt_rates``: such a table from wt monoculture growth
  rates on the relative-X gauge (``--growth_priors_wt_rates``).
- ``apply_priors_updates``: write scalar or per-condition updates into the
  CSV (shared with ``tfs-prefit-calibration``).
"""

import shutil
import sys

import numpy as np
import pandas as pd

# Columns of a --growth_priors table and the linear-growth prior rows each
# one sets. m_scale sets both of linear's m scales: a condition reads the one
# its +/- flag names, so they carry the same per-condition values.
GROWTH_PRIOR_FIELDS = {
    "k_loc": ("k_loc",),
    "k_scale": ("k_scale",),
    "m_loc": ("m_loc",),
    "m_scale": ("m_scale_plus", "m_scale_minus"),
}

GROWTH_PRIOR_PREFIX = "growth.condition_growth."

WT_RATE_COLUMNS = ("condition_sel", "titrant_conc", "rate_mean", "rate_sd",
                   "num_replicates")


def condition_rep_labels(orchestrator):
    """
    Per-condition labels (``condition_rep`` and, when present, ``replicate``)
    in the model's condition order (``map_condition_rep``); None without a
    growth model.
    """
    growth_tm = getattr(orchestrator, "growth_tm", None)
    if growth_tm is None:
        return None
    crm = growth_tm.map_groups.get("condition_rep")
    if crm is None or getattr(crm, "empty", True):
        return None
    sorted_map = crm.sort_values("map_condition_rep").reset_index(drop=True)
    cols = [c for c in ("replicate", "condition_rep") if c in sorted_map.columns]
    if not cols:
        return None
    return sorted_map[cols].reset_index(drop=True)


def apply_priors_updates(priors_path, prior_updates, cond_rep_labels=None,
                         backup=True):
    """
    Apply prior updates to a priors CSV.

    Rows whose ``parameter`` is not in ``prior_updates`` are preserved.

    * **scalar** (float): overwrites the ``value`` of the matching scalar row.
      A warning is printed if no row matches.
    * **array** (1-D ``np.ndarray``): a per-condition prior (e.g.
      ``growth.condition_growth.k_loc``). Existing rows for that parameter are
      replaced with one indexed row per condition, tagged with the
      ``condition_rep`` (and ``replicate``) labels from ``cond_rep_labels``,
      so the loader can name-join them to the model's condition order.

    Parameters
    ----------
    priors_path : str
        Path to the priors CSV.
    prior_updates : dict[str, float | np.ndarray]
        Update values keyed by dotted parameter name.
    cond_rep_labels : pandas.DataFrame or None
        Per-condition labels in the order of the array updates (see
        ``condition_rep_labels``). Without them array rows carry only
        ``flat_index``.
    backup : bool
        Copy the file to ``{priors_path}.bak`` before writing.

    Returns
    -------
    set of str
        The parameter names that were updated.
    """
    if not prior_updates:
        return set()
    df = pd.read_csv(priors_path)
    if "parameter" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"Priors CSV {priors_path} is missing required 'parameter' / "
            "'value' columns."
        )

    scalar_updates = {}
    array_updates = {}
    for row_name, new_val in prior_updates.items():
        arr = np.asarray(new_val)
        if arr.ndim == 0:
            scalar_updates[row_name] = float(arr)
        else:
            array_updates[row_name] = arr

    matched = set()

    for row_name, new_val in scalar_updates.items():
        mask = df["parameter"] == row_name
        if mask.any():
            df.loc[mask, "value"] = new_val
            matched.add(row_name)

    missing = sorted(set(scalar_updates) - matched)
    if missing:
        print(
            f"  warning: {len(missing)} prior update(s) had no matching row "
            f"in {priors_path}: {missing}",
            file=sys.stderr,
        )

    if array_updates:
        new_frames = []
        for row_name, arr in array_updates.items():
            flat_val = np.asarray(arr).flatten()
            df = df[df["parameter"] != row_name]
            row_df = pd.DataFrame({"parameter": row_name,
                                   "value": flat_val,
                                   "flat_index": range(len(flat_val))})
            if (cond_rep_labels is not None
                    and len(cond_rep_labels) == len(flat_val)):
                for col in ("replicate", "condition_rep"):
                    if col in cond_rep_labels.columns:
                        row_df[col] = cond_rep_labels[col].to_numpy()
            new_frames.append(row_df)
            matched.add(row_name)
        df = pd.concat([df] + new_frames, ignore_index=True)

    if backup:
        shutil.copy2(priors_path, priors_path + ".bak")
    df.to_csv(priors_path, index=False)
    print(f"Updated {len(matched)} priors row(s) in {priors_path}")
    return matched


def parse_prior_overrides(items):
    """
    Parse ``name=value`` strings into a dict of floats.

    Raises
    ------
    ValueError
        On an item without ``=``, a non-numeric value or a repeated name.
    """
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Prior override '{item}' is not of the form "
                             "name=value.")
        name, value = item.split("=", 1)
        name = name.strip()
        try:
            value = float(value)
        except ValueError as e:
            raise ValueError(f"Prior override '{item}': value is not a "
                             "number.") from e
        if name in out:
            raise ValueError(f"Prior override '{name}' is given twice.")
        out[name] = value
    return out


def resolve_prior_name(name, parameters):
    """
    Resolve a prior name to the full dotted row name in a priors CSV.

    ``name`` may be the full row name (``theta.theta_log_hill_n_hyper_scale_fixed``)
    or any dotted suffix of exactly one row (``theta_log_hill_n_hyper_scale_fixed``,
    ``sample_offset.sigma_fixed``, ``sigma_fixed`` when only one component
    has it).

    Raises
    ------
    ValueError
        If no row or more than one row matches.
    """
    parameters = list(dict.fromkeys(parameters))
    if name in parameters:
        return name
    hits = [p for p in parameters if p.endswith("." + name)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        close = [p for p in parameters if name.split(".")[-1] in p][:10]
        hint = f" Similar: {close}." if close else ""
        raise ValueError(f"No prior named '{name}' in this model.{hint}")
    raise ValueError(f"Prior name '{name}' is ambiguous; it matches {hits}. "
                     "Give the full name.")


def apply_prior_overrides(priors_path, overrides):
    """
    Set scalar priors by name (``tfs-configure-model --set_priors``).

    Parameters
    ----------
    priors_path : str
        Path to the priors CSV.
    overrides : dict[str, float]
        Prior name (full or a unique suffix; see ``resolve_prior_name``) to
        value.

    Returns
    -------
    dict[str, float]
        The full row names set, with their values.

    Raises
    ------
    ValueError
        If a name is unknown or ambiguous, or names a per-condition prior
        (set those with ``--growth_priors``).
    """
    if not overrides:
        return {}
    df = pd.read_csv(priors_path)
    if "flat_index" in df.columns:
        indexed = set(df.loc[df["flat_index"].notna(), "parameter"])
    else:
        indexed = set()

    resolved = {}
    for name, value in overrides.items():
        full = resolve_prior_name(name, df["parameter"])
        if full in indexed:
            raise ValueError(f"Prior '{full}' is per-condition (indexed rows); "
                             "set it with --growth_priors, not --set_priors.")
        if full in resolved:
            raise ValueError(f"Prior '{full}' is set twice.")
        resolved[full] = float(value)

    apply_priors_updates(priors_path, resolved, backup=False)
    return resolved


def _labels_table(cond_rep_labels):
    labels = cond_rep_labels.copy()
    labels["condition_rep"] = labels["condition_rep"].astype(str)
    if "replicate" in labels.columns:
        labels["replicate"] = labels["replicate"].astype(str)
    return labels


def growth_prior_updates(table, cond_rep_labels, defaults):
    """
    Per-condition linear-growth prior arrays from a table.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per condition: ``condition_rep``, optionally ``replicate``,
        and any of ``k_loc``, ``k_scale``, ``m_loc``, ``m_scale``. A table
        without ``replicate`` applies each row to every replicate. A missing
        value (NaN) keeps the default.
    cond_rep_labels : pandas.DataFrame
        The model's per-condition labels (``condition_rep_labels``).
    defaults : dict[str, float]
        Scalar default for each prior row (``k_loc``, ``k_scale``, ``m_loc``,
        ``m_scale_plus``, ``m_scale_minus``), used for conditions or values
        the table leaves out.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``growth.condition_growth.<field>`` to a per-condition array in the
        order of ``cond_rep_labels``, for ``apply_priors_updates``.

    Raises
    ------
    ValueError
        On an unknown column, a condition the model does not have, a
        condition given twice, or a non-positive scale.
    """
    if cond_rep_labels is None:
        raise ValueError("Per-condition growth priors need a growth model.")
    table = table.copy()
    if "condition_rep" not in table.columns:
        raise ValueError("A growth-priors table needs a 'condition_rep' column.")
    value_cols = [c for c in table.columns if c not in ("condition_rep", "replicate")]
    unknown = [c for c in value_cols if c not in GROWTH_PRIOR_FIELDS]
    if unknown:
        raise ValueError(f"Unknown growth-priors column(s) {unknown}; allowed: "
                         f"{list(GROWTH_PRIOR_FIELDS)}.")

    labels = _labels_table(cond_rep_labels)
    table["condition_rep"] = table["condition_rep"].astype(str)
    key_cols = ["condition_rep"]
    if "replicate" in table.columns:
        if "replicate" not in labels.columns:
            raise ValueError("The growth-priors table has a 'replicate' column "
                             "but the model shares conditions across replicates.")
        table["replicate"] = table["replicate"].astype(str)
        key_cols = ["replicate", "condition_rep"]

    dup = table.duplicated(subset=key_cols, keep=False)
    if dup.any():
        raise ValueError("Growth-priors table gives a condition more than once: "
                         f"{table.loc[dup, key_cols].drop_duplicates().values.tolist()}")
    known = set(map(tuple, labels[key_cols].drop_duplicates().values.tolist()))
    given = set(map(tuple, table[key_cols].values.tolist()))
    extra = sorted(given - known)
    if extra:
        raise ValueError(f"Growth-priors table names condition(s) {extra} that "
                         f"the model does not have; it has {sorted(known)}.")

    for col in value_cols:
        if col.endswith("_scale"):
            vals = pd.to_numeric(table[col], errors="coerce")
            if (vals.dropna() <= 0).any():
                raise ValueError(f"Growth-priors column '{col}' must be > 0.")

    merged = labels.merge(table, on=key_cols, how="left")
    updates = {}
    for col in value_cols:
        vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        for field in GROWTH_PRIOR_FIELDS[col]:
            arr = np.where(np.isnan(vals), float(defaults[field]), vals)
            updates[GROWTH_PRIOR_PREFIX + field] = arr
    return updates


def growth_priors_from_wt_rates(rates, gauge_conc, sd_floor=0.002):
    """
    A growth-priors table from wt monoculture growth rates (relative-X fit).

    In the X gauge (``theta: hill_relative``) wt's X is 1 at the low gauge
    concentration and 0 at the high one, and growth is ``k + dk + m X`` with
    wt's dk pinned at 0. So wt grows at ``k`` at the high concentration and
    ``k + m`` at the low one:

        k = wt rate at c_hi,    m = wt rate at c_lo - wt rate at c_hi.

    Each SD is the standard error of the replicate mean, floored at
    ``sd_floor`` (day-to-day and library-versus-monoculture differences are
    at least that).

    Parameters
    ----------
    rates : pandas.DataFrame
        Columns ``condition_sel`` (the condition, matched to the model's
        ``condition_rep``), ``titrant_conc``, ``rate_mean``, ``rate_sd`` and
        ``num_replicates``; rows at both gauge concentrations for each
        condition. Only the conditions it lists get priors; give it only the
        conditions where the monoculture represents the library's wt.
    gauge_conc : sequence of float
        ``(c_lo, c_hi)``, the model's ``theta_gauge_conc``.
    sd_floor : float
        Smallest SD for k and m, per minute.

    Returns
    -------
    pandas.DataFrame
        ``condition_rep``, ``k_loc``, ``k_scale``, ``m_loc``, ``m_scale``.
    """
    missing = [c for c in WT_RATE_COLUMNS if c not in rates.columns]
    if missing:
        raise ValueError(f"wt-rates table is missing column(s) {missing}; it "
                         f"needs {list(WT_RATE_COLUMNS)}.")
    c_lo, c_hi = (float(c) for c in gauge_conc)
    rows = []
    for cond, g in rates.groupby("condition_sel", sort=True):
        conc = g["titrant_conc"].astype(float).to_numpy()

        def _at(c):
            hit = g[np.isclose(conc, c)]
            if len(hit) != 1:
                raise ValueError(f"wt-rates table needs exactly one row for "
                                 f"condition '{cond}' at titrant_conc {c:g}; "
                                 f"found {len(hit)}.")
            r = hit.iloc[0]
            return float(r["rate_mean"]), float(r["rate_sd"]) / np.sqrt(float(r["num_replicates"]))

        lo, se_lo = _at(c_lo)
        hi, se_hi = _at(c_hi)
        rows.append({"condition_rep": str(cond),
                     "k_loc": hi,
                     "k_scale": max(se_hi, sd_floor),
                     "m_loc": lo - hi,
                     "m_scale": max(float(np.hypot(se_lo, se_hi)), sd_floor)})
    return pd.DataFrame(rows, columns=["condition_rep", "k_loc", "k_scale",
                                       "m_loc", "m_scale"])
