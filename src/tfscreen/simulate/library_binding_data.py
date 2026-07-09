"""
Generate simulated binding-curve data for IN-LIBRARY genotypes.

Unlike the spiked/monoclonal binding genotypes (handled in ``library_prediction``
+ ``binding_data.generate_binding_df``), these genotypes are drawn from the bulk
library, so their growth data carries congressional transformation.  Only a
binding *measurement* is added here -- their phenotype (Hill theta) is whatever
the library simulation produced (a prior-predictive draw for stratified/random
selection, or the injected file params for the file path), so no theta override
is needed and consistency with growth is automatic.

Selection for the stratified/random paths is deferred until AFTER the growth
simulation so it can be restricted to genotypes that actually survived with
growth data, guaranteeing ``num`` usable anchors.  The file path names its
genotypes up front (validated pre-sim); any that fail to survive are warned
about and dropped here.
"""

import warnings

import numpy as np
import pandas as pd

from tfscreen.genetics import standardize_genotypes
from tfscreen.simulate.sample_theta import _greedy_maximin
from tfscreen.simulate.binding_params import (
    read_binding_genotype_params,
    _to_log_conc,
)

_BINDING_COLS = ["genotype", "titrant_name", "titrant_conc", "theta_obs", "theta_std"]


def _is_file_choice(choose_by):
    """``choose_by`` is a file when it is not one of the builtin keywords."""
    return choose_by not in ("stratified", "random")


def _theta_matrix(params_df, log_concs):
    """Vectorised Hill theta for each row of ``params_df`` at each log conc.

    Returns an array of shape ``(len(params_df), len(log_concs))``.
    """
    tl = params_df["theta_low"].to_numpy(dtype=float)[:, None]
    th = params_df["theta_high"].to_numpy(dtype=float)[:, None]
    lk = params_df["log_hill_K"].to_numpy(dtype=float)[:, None]
    hn = params_df["hill_n"].to_numpy(dtype=float)[:, None]
    lc = np.asarray(log_concs, dtype=float)[None, :]
    occupancy = 1.0 / (1.0 + np.exp(-hn * (lc - lk)))
    return tl + (th - tl) * occupancy


def generate_library_binding_df(library_binding_cfg,
                                titrant_name,
                                titrant_conc,
                                noise,
                                parameters_df,
                                growth_df,
                                spiked_genotypes,
                                rng):
    """
    Generate binding-curve data for in-library (bulk) genotypes.

    Parameters
    ----------
    library_binding_cfg : dict
        The ``binding_data.library_binding`` sub-dict.  Keys:
          ``choose_by`` : "stratified" | "random" | <params-file path>
          ``num``       : int, required for stratified/random (forbidden with a file)
    titrant_name : str
        Titrant name (shared with the rest of ``binding_data``).
    titrant_conc : list of float
        Binding-assay concentrations (shared).
    noise : float
        Gaussian sigma on ``theta_obs`` (shared).
    parameters_df : pandas.DataFrame
        Per-genotype ground-truth phenotype; must carry ``genotype`` plus the
        Hill columns ``theta_low, theta_high, log_hill_K, hill_n``.  For
        file-specified genotypes this already reflects the injected file params.
    growth_df : pandas.DataFrame
        The simulated growth data (post selection); its ``genotype`` column is
        the set of survivors used to guarantee coverage.
    spiked_genotypes : iterable of str or None
        Genotypes that are spiked (excluded from the in-library pool).
    rng : numpy.random.Generator

    Returns
    -------
    (binding_df, manifest_df) : tuple[pandas.DataFrame, pandas.DataFrame]
        ``binding_df`` columns: genotype, titrant_name, titrant_conc,
        theta_obs, theta_std.  ``manifest_df`` records the selected genotypes,
        ``binding_class='library'`` and the ``choose_by`` used.
    """
    choose_by = library_binding_cfg["choose_by"]
    num = library_binding_cfg.get("num")

    titrant_conc = [float(c) for c in titrant_conc]
    log_concs = _to_log_conc(titrant_conc)
    noise = float(noise)

    spiked_set = (set(standardize_genotypes(list(spiked_genotypes)))
                  if spiked_genotypes else set())

    # Genotypes that actually have growth data (survived selection).
    survivors = set(growth_df["genotype"].astype(str).unique())

    # Index the ground-truth phenotypes by genotype for theta lookups.
    pdf = parameters_df.copy()
    pdf["genotype"] = pdf["genotype"].astype(str)
    pdf = pdf.drop_duplicates("genotype").set_index("genotype")

    if _is_file_choice(choose_by):
        # The file names the genotypes; their Hill params were injected into the
        # library phenotype pre-sim, so theta comes from parameters_df.
        named = list(standardize_genotypes(
            list(read_binding_genotype_params(choose_by).keys())))
        eligible = survivors & set(pdf.index)
        missing = [g for g in named if g not in eligible]
        if missing:
            warnings.warn(
                f"library_binding: {len(missing)} file-specified genotype(s) did "
                f"not survive with growth data and are dropped: {missing}"
            )
        genotypes = [g for g in named if g in eligible]
    else:
        # Stratified / random: choose `num` from the surviving bulk
        # (non-spiked, non-wt) so every anchor has matching growth data.
        pool = sorted((survivors & set(pdf.index)) - spiked_set - {"wt"})
        if num is None:
            raise ValueError(
                f"library_binding requires 'num' for choose_by '{choose_by}'."
            )
        if len(pool) < int(num):
            raise ValueError(
                f"library_binding: only {len(pool)} eligible surviving genotype(s) "
                f"available, but num={num} requested."
            )
        pool_theta = _theta_matrix(pdf.loc[pool], log_concs)
        if choose_by == "stratified":
            sel_idx = _greedy_maximin(pool_theta, int(num))
        elif choose_by == "random":
            sel_idx = rng.choice(len(pool), size=int(num), replace=False)
        else:
            raise ValueError(f"Unknown library_binding choose_by: '{choose_by}'.")
        genotypes = [pool[int(i)] for i in np.asarray(sel_idx)]

    # Build binding rows: theta from Hill params + Gaussian noise.
    theta_true = (_theta_matrix(pdf.loc[genotypes], log_concs)
                  if genotypes else np.zeros((0, len(titrant_conc))))
    rows = []
    for gi, g in enumerate(genotypes):
        for cj, conc in enumerate(titrant_conc):
            tt = float(theta_true[gi, cj])
            tobs = (float(np.clip(tt + rng.normal(0, noise), 0.0, 1.0))
                    if noise > 0 else tt)
            rows.append({"genotype": g, "titrant_name": titrant_name,
                         "titrant_conc": conc, "theta_obs": tobs,
                         "theta_std": noise})

    binding_df = pd.DataFrame(rows, columns=_BINDING_COLS)
    manifest_df = pd.DataFrame({
        "genotype": genotypes,
        "binding_class": ["library"] * len(genotypes),
        "choose_by": [str(choose_by)] * len(genotypes),
    })
    return binding_df, manifest_df
