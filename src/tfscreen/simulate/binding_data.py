"""
Generate simulated observed binding curve data for the SPIKED binding genotypes.

Noise-injects the pre-computed ``binding_theta_df`` that ``library_prediction``
builds for the ``binding_data.spiked_binding`` block (clean, monoclonal,
congression-free controls) into a "measured" binding curve CSV -- the mirror
image of generate_base_growth_df, which does the same noise-injection for direct
growth-rate calibration data.

The IN-LIBRARY (congression-affected) binding genotypes are handled separately,
post-growth-sim, by ``library_binding_data.generate_library_binding_df``; the
two results are concatenated in ``simulate/scripts/simulate_cli.py``.
"""

import numpy as np
import pandas as pd

from tfscreen.genetics import standardize_genotypes


def generate_binding_df(binding_cfg, rng, binding_theta_df):
    """
    Generate simulated binding curve data for the spiked binding genotypes.

    Uses the pre-computed theta values ``library_prediction`` built for the
    ``spiked_binding`` block and adds Gaussian noise to produce observed theta.

    Parameters
    ----------
    binding_cfg : dict
        The ``binding_data`` config dict.  Reads the top-level (assay-wide)
        keys:
          titrant_name: str, name of the titrant (e.g. 'iptg')
          titrant_conc: list of concentrations (mM)
          noise       : float, sigma for Gaussian noise on theta_obs (default 0)
        An optional ``genotypes`` list restricts the output to those genotypes;
        when absent (the normal case under the spiked_binding schema) every
        genotype in ``binding_theta_df`` is emitted.
    rng : numpy.random.Generator
    binding_theta_df : pandas.DataFrame
        Pre-computed binding theta from library_prediction.  Must contain
        columns ``genotype``, ``titrant_conc``, ``theta_true``.

    Returns
    -------
    pandas.DataFrame
        Columns: genotype, titrant_name, titrant_conc, theta_obs, theta_std
    """
    titrant_name = binding_cfg["titrant_name"]
    titrant_conc = list(binding_cfg["titrant_conc"])
    noise = float(binding_cfg.get("noise", 0.0))

    # Build lookup: (genotype, conc) → theta_true
    theta_lookup = {
        (row["genotype"], float(row["titrant_conc"])): row["theta_true"]
        for _, row in binding_theta_df.iterrows()
    }

    # Genotypes to output: explicit list from config, or every genotype in binding_theta_df.
    raw_genotypes = binding_cfg.get("genotypes")
    if raw_genotypes:
        genotypes = list(standardize_genotypes(raw_genotypes))
    else:
        genotypes = list(binding_theta_df["genotype"].unique())

    rows = []
    for g in genotypes:
        for conc in titrant_conc:
            key = (g, float(conc))
            if key not in theta_lookup:
                raise ValueError(
                    f"No pre-computed theta for genotype '{g}' at conc {conc}. "
                    f"Ensure binding_data genotypes and titrant_conc match the config."
                )
            theta_true = float(theta_lookup[key])
            if noise > 0:
                theta_obs = float(np.clip(theta_true + rng.normal(0, noise), 0, 1))
            else:
                theta_obs = theta_true
            rows.append({
                "genotype": g,
                "titrant_name": titrant_name,
                "titrant_conc": conc,
                "theta_obs": theta_obs,
                "theta_std": noise,
            })

    return pd.DataFrame(rows)
