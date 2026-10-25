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
          clip_theta_obs : bool, clip noisy theta_obs to [0, 1] (default
                        False). The fit's binding likelihood is an unclipped
                        Normal, and clipping biases the anchors near 0 and 1;
                        True reproduces simulations made before 2026-09-29.
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
    clip = bool(binding_cfg.get("clip_theta_obs", False))

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
                theta_obs = theta_true + rng.normal(0, noise)
                if clip:
                    theta_obs = np.clip(theta_obs, 0, 1)
                theta_obs = float(theta_obs)
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


def apply_genotype_offsets(binding_df, offset_sd, rng):
    """
    Add one offset per genotype to every one of its theta_obs.

    The simulated counterpart of the fit's ``theta_binding_noise:
    genotype_offset``: a genotype's binding differs from its in vivo
    occupancy (or carries its day's normalization error) by a shift shared
    across its whole curve, offset_g ~ Normal(0, offset_sd).

    Parameters
    ----------
    binding_df : pandas.DataFrame
        Must have ``genotype`` and ``theta_obs``.
    offset_sd : float
        SD of the offsets (theta units); 0 returns the table unchanged and
        draws nothing.
    rng : numpy.random.Generator

    Returns
    -------
    binding_df : pandas.DataFrame
    truth : pandas.DataFrame or None
        ``genotype``, ``offset`` (None when offset_sd is 0).
    """
    offset_sd = float(offset_sd)
    if not np.isfinite(offset_sd) or offset_sd < 0:
        raise ValueError(f"genotype_offset_sd must be >= 0, got {offset_sd}")
    if offset_sd == 0 or len(binding_df) == 0:
        return binding_df, None
    genotypes = list(pd.unique(binding_df["genotype"].astype(str)))
    offsets = rng.normal(0.0, offset_sd, size=len(genotypes))
    lookup = dict(zip(genotypes, offsets))
    out = binding_df.copy()
    out["theta_obs"] = (out["theta_obs"].astype(float)
                        + out["genotype"].astype(str).map(lookup))
    return out, pd.DataFrame({"genotype": genotypes, "offset": offsets})
