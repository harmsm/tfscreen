"""
Generate simulated direct growth-rate calibration data (base_growth_data).

This mirrors the inference-side ``base_growth_df`` input: a direct,
reference-condition growth-rate observation for a subset of genotypes, tied
to a shared reference rate (``k_ref``) and the existing per-genotype
``dk_geno`` value. It exists to give the hierarchical tfmodel inference a
direct anchor for wt's growth rate, resolving an identifiability confound
between condition_growth's k/m and dk_geno's hierarchical hyperparameters
(see model_orchestrator._read_base_growth_df / generative/model.py's
base_growth_obs block).

Every genotype requested here must already be part of the simulated library
(i.e. present in ``parameters_df``) -- dk_geno is looked up from the value
already assigned during ``library_prediction``, never redrawn.
"""

import pandas as pd

from tfscreen.genetics import standardize_genotypes


def generate_base_growth_df(base_growth_cfg, parameters_df, rng):
    """
    Generate a simulated base_growth_df.

    For each requested genotype (wt is always included), the true growth
    rate is:

        rate_true = k_ref + dk_geno[genotype]

    unless overridden by an explicit value in ``base_growth_cfg["rates"]``,
    in which case ``rate_true`` is that override. Gaussian noise (sigma =
    ``noise``) is added to produce the observed ``rate``; ``rate_std`` is
    reported as ``noise`` for every row, matching the
    ``binding_data.noise`` -> ``theta_std`` convention used elsewhere in
    this module.

    Parameters
    ----------
    base_growth_cfg : dict
        The 'base_growth_data' sub-dict from the config. Must contain:
          k_ref : float
              Reference (wt) growth rate, hr^-1.
        May contain:
          genotypes : list of str, default ["wt"]
              Genotypes to include. 'wt' is always force-included even if
              omitted.
          rates : dict[str, float], optional
              Explicit true-rate override for specific genotypes, bypassing
              the k_ref + dk_geno calculation for that genotype. Every key
              must also appear in (the resolved) ``genotypes``.
          noise : float, default 0.0
              Sigma of the Gaussian noise added to the true rate to produce
              the observed rate.
    parameters_df : pandas.DataFrame
        Output of library_prediction; must contain 'genotype' and 'dk_geno'
        columns, one row per unique library genotype.
    rng : numpy.random.Generator
        Shared random-number generator for the noise draws.

    Returns
    -------
    pandas.DataFrame
        Columns: genotype, rate, rate_std, rate_true. ``rate_true`` is the
        noise-free ground truth, included for validation (analogous to
        presplit's ``ln_cfu_0_true``).

    Raises
    ------
    ValueError
        If 'k_ref' is missing from ``base_growth_cfg``, if any requested
        genotype is not present in ``parameters_df``, or if ``rates``
        contains a genotype not in the resolved ``genotypes`` list.
    """
    if "k_ref" not in base_growth_cfg:
        raise ValueError(
            "base_growth_data config must specify 'k_ref' (the reference "
            "wt growth rate, hr^-1)."
        )
    k_ref = float(base_growth_cfg["k_ref"])
    noise = float(base_growth_cfg.get("noise", 0.0))

    raw_genotypes = list(base_growth_cfg.get("genotypes", ["wt"]))
    genotypes = list(standardize_genotypes(raw_genotypes))
    if "wt" not in genotypes:
        genotypes = ["wt"] + genotypes

    raw_rates = base_growth_cfg.get("rates", {})
    if raw_rates:
        override_genotypes = standardize_genotypes(list(raw_rates.keys()))
        rate_overrides = dict(zip(override_genotypes, raw_rates.values()))
    else:
        rate_overrides = {}

    library_genotypes = set(parameters_df["genotype"])
    missing = [g for g in genotypes if g not in library_genotypes]
    if missing:
        raise ValueError(
            f"base_growth_data.genotypes {missing} not found in the "
            "simulated library. Every genotype listed must already be part "
            "of the library (see tile_combos / library_mixture)."
        )

    unrecognized_rate_keys = [g for g in rate_overrides if g not in genotypes]
    if unrecognized_rate_keys:
        raise ValueError(
            f"base_growth_data.rates key(s) {unrecognized_rate_keys} are "
            "not in base_growth_data.genotypes."
        )

    dk_geno_lookup = parameters_df.set_index("genotype")["dk_geno"]

    rows = []
    for g in genotypes:
        if g in rate_overrides:
            rate_true = float(rate_overrides[g])
        else:
            rate_true = k_ref + float(dk_geno_lookup[g])

        if noise > 0:
            rate_obs = float(rate_true + rng.normal(0.0, noise))
        else:
            rate_obs = rate_true

        rows.append({
            "genotype": g,
            "rate": rate_obs,
            "rate_std": noise,
            "rate_true": rate_true,
        })

    return pd.DataFrame(rows)
