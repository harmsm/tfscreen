"""
Utilities for injecting measured Hill binding parameters into the simulation.

When real binding data (theta curves from fluorescence, SPR, ITC, etc.) are
available for specific genotypes, supply a CSV of per-genotype Hill parameters
via the ``binding_data.genotype_params_file`` config key.  This module reads
those parameters and computes theta-value overrides for injection into the
simulation via the ``theta_gc_override`` mechanism in ``thermo_to_growth``.

Supported theta components: ``hill_geno``, ``hill_mut``.

CSV format (``genotype_params_file``)
--------------------------------------
Columns: ``genotype``, plus any subset of
  ``theta_low``, ``theta_high``, ``log_hill_K``, ``hill_n``.

One row per genotype.  ``NaN`` in any parameter column → the WT reference
value for that parameter is used (from ``theta_sim_priors`` in the config,
or overridden by the ``wt`` row in the CSV itself).

Example::

    genotype,theta_low,theta_high,log_hill_K,hill_n
    wt,0.99,0.01,-4.1,2.0
    A47V,0.97,0.03,-3.8,1.8
    K84L,,,−3.5,

``hill_mut`` note
-----------------
Single-mutant rows are back-converted to per-mutation deltas (relative to the
WT reference) and injected into a full ``hill_mut``-style simulation.  Deltas
for mutations absent from the CSV are drawn from the ``SimPriors`` as usual.
Pairwise epistasis is sampled from the regularized horseshoe prior and applied
consistently to the whole library.  The function returns theta for every
library genotype — including doubles assembled from measured singles — so
downstream growth rates are fully consistent with the measured binding data.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

_EPS = 1e-7
_THETA_CLIP = 1e-4          # hard bounds for theta_low / theta_high from CSV
_ZERO_CONC_SENTINEL = 1e-20

HILL_PARAM_COLS = frozenset({"theta_low", "theta_high", "log_hill_K", "hill_n"})
SUPPORTED_COMPONENTS = frozenset({"hill_geno", "hill_mut"})


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def read_binding_genotype_params(csv_path: Union[str, Path]) -> dict:
    """
    Read a per-genotype Hill parameter CSV.

    Parameters
    ----------
    csv_path : str or Path

    Returns
    -------
    dict[str, dict[str, float]]
        ``{genotype: {param: value}}``.  ``NaN`` values are preserved as
        ``float("nan")``; the caller fills them from the WT reference.

    Raises
    ------
    ValueError
        If the CSV lacks a ``genotype`` column or has no recognised parameter
        columns.  Extra columns (e.g. the diagnostic columns in a
        ``*_stage1_fits.csv``) are ignored.
    """
    df = pd.read_csv(csv_path)

    if "genotype" not in df.columns:
        raise ValueError(
            f"'{csv_path}' must have a 'genotype' column."
        )

    param_cols_present = [c for c in df.columns if c in HILL_PARAM_COLS]
    if not param_cols_present:
        raise ValueError(
            f"'{csv_path}' must have at least one parameter column: "
            f"{sorted(HILL_PARAM_COLS)}"
        )

    # Any columns beyond genotype + the recognised Hill params are ignored,
    # so a stage1_fits.csv (which also carries dk_geno, *_t, n_obs, ...) can be
    # used directly as a binding genotype-params file.
    result = {}
    for _, row in df.iterrows():
        g = str(row["genotype"])
        params = {}
        for col in param_cols_present:
            v = row[col]
            if pd.isna(v):
                params[col] = float("nan")
            else:
                v = float(v)
                if col in ("theta_low", "theta_high"):
                    clipped = float(np.clip(v, _THETA_CLIP, 1.0 - _THETA_CLIP))
                    if clipped != v:
                        warnings.warn(
                            f"'{csv_path}': {col} for genotype '{g}' "
                            f"({v:.6g}) is outside [{_THETA_CLIP}, "
                            f"{1.0 - _THETA_CLIP}] and has been clipped to "
                            f"{clipped:.6g}.  theta values must lie strictly "
                            f"between 0 and 1.",
                            UserWarning,
                            stacklevel=2,
                        )
                    v = clipped
                params[col] = v
        result[g] = params

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _logit(x):
    xc = np.clip(np.asarray(x, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(xc / (1.0 - xc))


def _to_log_conc(concs):
    c = np.asarray(concs, dtype=float)
    return np.log(np.where(c == 0.0, _ZERO_CONC_SENTINEL, c))


def _hill_theta(theta_low, theta_high, log_hill_K, hill_n, log_conc):
    """Evaluate Hill equation at log concentrations. Returns shape (len(log_conc),)."""
    lc = np.asarray(log_conc, dtype=float)
    occupancy = 1.0 / (1.0 + np.exp(-hill_n * (lc - log_hill_K)))
    return float(theta_low) + (float(theta_high) - float(theta_low)) * occupancy


def _fill_params_from_wt(params: dict, wt_params: dict) -> dict:
    """Return a copy of *params* with NaN entries replaced from *wt_params*."""
    filled = dict(params)
    for k in list(HILL_PARAM_COLS):
        v = filled.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if k in wt_params:
                filled[k] = wt_params[k]
    return filled


def _wt_params_from_sim_priors(sim_priors) -> dict:
    """
    Extract WT reference in the CSV parameter space from a SimPriors object.

    Works with both ``hill_geno.SimPriors`` and ``hill_mut.SimPriors``;
    both expose ``wt_theta_low``, ``wt_theta_high``, ``wt_log_K``, ``wt_hill_n``.
    """
    return {
        "theta_low":  float(sim_priors.wt_theta_low),
        "theta_high": float(sim_priors.wt_theta_high),
        "log_hill_K": float(sim_priors.wt_log_K),
        "hill_n":     float(sim_priors.wt_hill_n),
    }


# ---------------------------------------------------------------------------
# hill_geno override
# ---------------------------------------------------------------------------

def build_theta_gc_override_hill_geno(
    params_dict: dict,
    log_conc_growth: np.ndarray,
    wt_params: dict,
) -> dict:
    """
    Compute theta at growth concentrations for measured genotypes (``hill_geno``).

    Each entry in *params_dict* contributes one key to the returned override
    dict.  ``NaN`` parameter values are filled from *wt_params*.

    Parameters
    ----------
    params_dict : dict[str, dict[str, float]]
        Output of :func:`read_binding_genotype_params`.
    log_conc_growth : np.ndarray, shape (C,)
        Log concentrations at growth conditions (from ``sim_data.log_titrant_conc``).
    wt_params : dict
        WT reference: ``{theta_low, theta_high, log_hill_K, hill_n}``.
        From :func:`_wt_params_from_sim_priors`.

    Returns
    -------
    theta_dict : dict[str, np.ndarray]
        ``{genotype: theta_array}`` where theta_array has shape (C,).
    effective_params : dict[str, dict[str, float]]
        ``{genotype: {theta_low, theta_high, log_hill_K, hill_n}}`` — the
        actual Hill parameters used for each genotype (NaN fields filled from
        *wt_params*).
    """
    theta_dict = {}
    effective_params = {}
    for g, params in params_dict.items():
        filled = _fill_params_from_wt(params, wt_params)
        theta_dict[g] = _hill_theta(
            filled["theta_low"], filled["theta_high"],
            filled["log_hill_K"], filled["hill_n"],
            log_conc_growth,
        )
        effective_params[g] = {k: filled[k] for k in HILL_PARAM_COLS}
    return theta_dict, effective_params


# ---------------------------------------------------------------------------
# hill_mut override
# ---------------------------------------------------------------------------

def build_theta_gc_override_hill_mut(
    params_dict: dict,
    library_genotypes: list,
    sim_data,
    sim_priors,
    log_conc: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Run a ``hill_mut``-style simulation with per-mutation deltas pinned for
    measured genotypes and return theta for every library genotype.

    Single-mutation entries in *params_dict* are back-converted to per-mutation
    deltas relative to the WT reference (CSV ``wt`` row wins; otherwise
    *sim_priors* WT values are used).  Mutations absent from *params_dict* draw
    deltas from the prior as usual.  Pairwise epistasis is sampled from the
    regularized horseshoe prior for all pairs consistently.  Multi-mutation
    entries in *params_dict* override the assembled theta value directly.

    Parameters
    ----------
    params_dict : dict[str, dict[str, float]]
    library_genotypes : list[str]
        Genotype strings in the same order as *sim_data* was built from.
    sim_data : SimData
        Must have ``mut_nnz_mut_idx``, ``mut_nnz_geno_idx``,
        ``pair_nnz_pair_idx``, ``pair_nnz_geno_idx``, ``num_pair``.
    sim_priors : hill_mut.SimPriors
        Used as fallback WT reference and prior widths for unmeasured mutations.
    log_conc : np.ndarray, shape (C,)
        Log concentrations to evaluate theta at.
    rng : np.random.Generator

    Returns
    -------
    theta_dict : dict[str, np.ndarray]
        ``{genotype: theta_array}`` for every genotype in *library_genotypes*.
        Caller should add all entries to ``theta_gc_override``.
    effective_params : dict[str, dict[str, float]]
        ``{genotype: {theta_low, theta_high, log_hill_K, hill_n}}`` — the
        effective Hill parameters used for each genotype in the simulation.
    """
    import jax.numpy as jnp
    from tfscreen.genetics.build_mut_geno_matrix import (
        build_mut_geno_matrix,
        apply_mut_matrix,
        apply_pair_matrix,
    )

    log_conc = np.asarray(log_conc, dtype=float)
    G = len(library_genotypes)

    # Get mutation labels in the same order as sim_data was built
    mut_labels, pair_labels, _, _, _ = build_mut_geno_matrix(library_genotypes)
    M = len(mut_labels)
    P = len(pair_labels)

    # WT reference in logit/log space ---
    # If 'wt' row is in params_dict, it overrides sim_priors WT values.
    wt_params = _wt_params_from_sim_priors(sim_priors)
    if "wt" in params_dict:
        wt_from_csv = _fill_params_from_wt(params_dict["wt"], wt_params)
        wt_params = wt_from_csv  # CSV WT wins

    wt_logit_low   = float(_logit(wt_params["theta_low"]))
    wt_logit_high  = float(_logit(wt_params["theta_high"]))
    wt_logit_delta = wt_logit_high - wt_logit_low
    wt_log_K       = float(wt_params["log_hill_K"])
    wt_log_n       = float(np.log(wt_params["hill_n"]))

    # Build per-mutation delta arrays (NaN = needs to be sampled from prior)
    d_logit_low   = np.full(M, np.nan)
    d_logit_delta = np.full(M, np.nan)
    d_log_K       = np.full(M, np.nan)
    d_log_n       = np.full(M, np.nan)

    def _parse_muts(g):
        return [p for p in g.split("/") if p and p.lower() != "wt"]

    mut_label_to_idx = {m: i for i, m in enumerate(mut_labels)}

    for g, params in params_dict.items():
        muts = _parse_muts(g)
        if len(muts) != 1:
            continue  # skip wt and multi-mutants for delta back-calculation
        m_label = muts[0]
        if m_label not in mut_label_to_idx:
            continue  # this mutation is not in the library

        m_idx = mut_label_to_idx[m_label]
        filled = _fill_params_from_wt(params, wt_params)

        g_logit_low   = float(_logit(filled["theta_low"]))
        g_logit_high  = float(_logit(filled["theta_high"]))
        g_logit_delta = g_logit_high - g_logit_low

        d_logit_low[m_idx]   = g_logit_low   - wt_logit_low
        d_logit_delta[m_idx] = g_logit_delta - wt_logit_delta
        d_log_K[m_idx]       = float(filled["log_hill_K"]) - wt_log_K
        d_log_n[m_idx]       = float(np.log(filled["hill_n"])) - wt_log_n

    # Sample deltas for unmeasured mutations from prior
    unmeasured = np.isnan(d_logit_low)
    n_un = int(unmeasured.sum())
    if n_un > 0:
        d_logit_low[unmeasured]   = rng.normal(0.0, sim_priors.sigma_d_logit_low,   n_un)
        d_logit_delta[unmeasured] = rng.normal(0.0, sim_priors.sigma_d_logit_delta, n_un)
        d_log_K[unmeasured]       = rng.normal(0.0, sim_priors.sigma_d_log_K,       n_un)
        d_log_n[unmeasured]       = rng.normal(0.0, sim_priors.sigma_d_log_n,       n_un)

    # Scatter helpers: (1, M) or (1, P) → (1, G)
    mut_nnz_mut  = jnp.array(sim_data.mut_nnz_mut_idx)
    mut_nnz_geno = jnp.array(sim_data.mut_nnz_geno_idx)

    def _scatter_mut(d_1d):
        return np.array(
            apply_mut_matrix(
                jnp.array(d_1d, dtype=float)[None, :],
                mut_nnz_mut_idx=mut_nnz_mut,
                mut_nnz_geno_idx=mut_nnz_geno,
                num_genotype=G,
            )
        )[0]

    # Assemble per-genotype parameters: (G,)
    logit_low   = wt_logit_low   + _scatter_mut(d_logit_low)
    logit_delta = wt_logit_delta + _scatter_mut(d_logit_delta)
    log_K       = wt_log_K       + _scatter_mut(d_log_K)
    log_n       = wt_log_n       + _scatter_mut(d_log_n)

    # Horseshoe epistasis for pairs (mirrors hill_mut.simulate exactly)
    if P > 0 and sim_data.pair_nnz_pair_idx is not None:
        pair_nnz_pair = jnp.array(sim_data.pair_nnz_pair_idx)
        pair_nnz_geno = jnp.array(sim_data.pair_nnz_geno_idx)

        def _scatter_pair(e_1d):
            return np.array(
                apply_pair_matrix(
                    jnp.array(e_1d, dtype=float)[None, :],
                    pair_nnz_pair_idx=pair_nnz_pair,
                    pair_nnz_geno_idx=pair_nnz_geno,
                    num_genotype=G,
                )
            )[0]

        tau = float(np.abs(rng.standard_cauchy())) * sim_priors.epi_tau_scale

        if tau > 0.0:
            c2 = 1.0 / rng.gamma(
                shape=sim_priors.epi_slab_df / 2.0,
                scale=2.0 / (sim_priors.epi_slab_df * sim_priors.epi_slab_scale ** 2),
            )

            def _horseshoe(size):
                lam = np.abs(rng.standard_cauchy(size))
                lam_tilde = np.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
                return rng.standard_normal(size) * tau * lam_tilde

            logit_low   += _scatter_pair(_horseshoe(P))
            logit_delta += _scatter_pair(_horseshoe(P))
            log_K       += _scatter_pair(_horseshoe(P))
            log_n       += _scatter_pair(_horseshoe(P))

    # Convert to theta: (G, C)
    theta_low_arr  = 1.0 / (1.0 + np.exp(-logit_low))
    theta_high_arr = 1.0 / (1.0 + np.exp(-(logit_low + logit_delta)))
    hill_n_arr     = np.exp(log_n)

    occupancy = 1.0 / (1.0 + np.exp(
        -hill_n_arr[:, None] * (log_conc[None, :] - log_K[:, None])
    ))
    theta_gc = theta_low_arr[:, None] + (theta_high_arr - theta_low_arr)[:, None] * occupancy

    # Build result dict: last occurrence wins for duplicates (matches theta_gc_override semantics)
    result = {}
    effective_params = {}
    for i, g in enumerate(library_genotypes):
        result[g] = theta_gc[i]
        effective_params[g] = {
            "theta_low":  float(theta_low_arr[i]),
            "theta_high": float(theta_high_arr[i]),
            "log_hill_K": float(log_K[i]),
            "hill_n":     float(hill_n_arr[i]),
        }

    # Directly-measured multi-mutant entries override the assembled values
    for g, params in params_dict.items():
        muts = _parse_muts(g)
        if len(muts) > 1 and g in result:
            filled = _fill_params_from_wt(params, wt_params)
            result[g] = _hill_theta(
                filled["theta_low"], filled["theta_high"],
                filled["log_hill_K"], filled["hill_n"],
                log_conc,
            )
            effective_params[g] = {k: filled[k] for k in HILL_PARAM_COLS}

    return result, effective_params


# ---------------------------------------------------------------------------
# Binding theta output
# ---------------------------------------------------------------------------

def build_binding_theta_from_params(
    params_dict: dict,
    binding_concs,
    titrant_name: str,
    noise: float,
    rng: np.random.Generator,
    wt_params: dict,
) -> pd.DataFrame:
    """
    Compute binding theta rows for measured genotypes.

    Parameters
    ----------
    params_dict : dict[str, dict[str, float]]
    binding_concs : array-like
        Concentrations (mM) at which binding theta is reported.
    titrant_name : str
    noise : float
        Gaussian noise sigma applied to theta (in [0, 1] space).
        Set to 0 to disable.
    rng : np.random.Generator
    wt_params : dict
        ``{theta_low, theta_high, log_hill_K, hill_n}`` used to fill NaN params.

    Returns
    -------
    pd.DataFrame
        Columns: ``genotype``, ``titrant_name``, ``titrant_conc``, ``theta_true``.
    """
    binding_concs = np.asarray(binding_concs, dtype=float)
    log_concs = _to_log_conc(binding_concs)

    rows = []
    for g, params in params_dict.items():
        filled = _fill_params_from_wt(params, wt_params)
        theta_vals = _hill_theta(
            filled["theta_low"], filled["theta_high"],
            filled["log_hill_K"], filled["hill_n"],
            log_concs,
        )
        for conc, theta in zip(binding_concs, theta_vals):
            if noise > 0:
                theta_out = float(np.clip(theta + rng.normal(0.0, noise), 0.0, 1.0))
            else:
                theta_out = float(theta)
            rows.append({
                "genotype":     g,
                "titrant_name": titrant_name,
                "titrant_conc": float(conc),
                "theta_true":   theta_out,
            })

    return pd.DataFrame(rows)
