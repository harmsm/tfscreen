"""
Generate simulated pre-split (t = -t_pre) data (presplit_data config block).

Mirrors the inference-side presplit_df input by drawing a synthetic
sequencing sample from the initial library-frequency distribution (encoded
as ln_cfu_0 in the selection-experiment counts output) for each unique
(replicate, condition_pre) pair, then converting counts to ln_cfu /
ln_cfu_std using the same variance-propagation formula as counts_to_lncfu.
"""

import numpy as np
import pandas as pd

from tfscreen.simulate.selection_experiment import _sim_index_hop


def generate_presplit_df(combined_sample_df, combined_counts_df, cf, rng):
    """
    Generate simulated pre-split (t = -t_pre) data from selection-experiment
    outputs.

    For each unique ``(replicate, condition_pre)`` pair, this function draws a
    synthetic sequencing sample from the initial library-frequency distribution
    (encoded as ``ln_cfu_0`` in ``combined_counts_df``).  The resulting counts
    are converted to ``ln_cfu`` and ``ln_cfu_std`` using the same
    variance-propagation formula as ``counts_to_lncfu``, yielding a DataFrame
    that can be passed directly to ``tfs-configure-model --presplit_df``.

    Parameters
    ----------
    combined_sample_df : pandas.DataFrame
        Concatenated sample metadata across all replicates (index = sample ID).
        Must contain columns ``replicate`` and ``condition_pre``.
    combined_counts_df : pandas.DataFrame
        Concatenated genotype counts across all replicates.  Must contain
        columns ``sample``, ``genotype``, and ``ln_cfu_0`` (the ground-truth
        initial log-CFU per genotype computed by the simulation).
    cf : dict
        Full run configuration (already read from YAML).  Used keys:
        ``cfu0``, ``total_num_reads``, ``prob_index_hop``.  The optional ``presplit_data`` sub-dict may
        contain a scalar ``noise`` key (default 0) to add extra Gaussian noise
        to ``ln_cfu`` on the log scale.
    rng : numpy.random.Generator
        Seeded random-number generator shared with the rest of the simulation.

    Returns
    -------
    pandas.DataFrame
        Columns: ``library``, ``replicate``, ``condition_pre``, ``genotype``,
        ``ln_cfu``, ``ln_cfu_std``, ``ln_cfu_0_true``.
        ``ln_cfu_0_true`` records the simulation ground truth for validation.
        Rows with zero initial frequency (genotypes absent from the
        transformation pool) receive ``ln_cfu = NaN``.
    """
    total_cfu0        = float(cf["cfu0"])
    presplit_cfg      = cf.get("presplit_data", {})
    extra_noise       = float(presplit_cfg.get("noise", 0.0)) if presplit_cfg else 0.0
    pseudocount       = 1

    # Reads per presplit sample: use the same budget as the main experiment
    # (total reads / total selection samples).
    total_num_reads    = int(cf["total_num_reads"])
    total_num_samples  = len(combined_sample_df)
    reads_per_sample   = max(1, int(round(total_num_reads / total_num_samples)))

    # Attach library, condition_pre, and replicate to each (genotype, sample)
    # row so we can group by (replicate, library, condition_pre) below.
    sample_meta = (combined_sample_df
                   .reset_index()[["sample", "replicate", "library", "condition_pre"]]
                   .drop_duplicates("sample"))
    counts_meta = pd.merge(combined_counts_df, sample_meta, on="sample", how="left")

    # One row per (replicate, library, condition_pre, genotype), keeping the
    # first occurrence of ln_cfu_0 (same value for all selection samples within
    # a library group).
    source = (counts_meta
              .groupby(["replicate", "library", "condition_pre", "genotype"], observed=True)
              .first()
              .reset_index()[["replicate", "library", "condition_pre", "genotype", "ln_cfu_0"]])

    rows = []
    for (rep, lib, cp), grp in source.groupby(["replicate", "library", "condition_pre"],
                                               observed=True):
        genos    = grp["genotype"].values
        ln_cfu0  = grp["ln_cfu_0"].values

        # Convert ground-truth ln_cfu_0 to initial frequencies.
        # Genotypes with ln_cfu_0 = -inf (absent from transformation pool)
        # get frequency 0.
        cfu0_raw  = np.where(np.isfinite(ln_cfu0), np.exp(ln_cfu0), 0.0)
        total_cfu_group = cfu0_raw.sum()
        if total_cfu_group == 0:
            continue
        freqs = cfu0_raw / total_cfu_group

        # Simulate multinomial sequencing draw from the initial distribution.
        counts = rng.multinomial(reads_per_sample, freqs)

        # Optionally apply index hopping (same as the selection samples).
        counts = _sim_index_hop(counts, cf.get("prob_index_hop"), rng)

        # ---------- counts → ln_cfu (mirrors counts_to_lncfu logic) ----------
        sample_cfu     = total_cfu0

        total_adjusted = counts.sum() + len(counts) * pseudocount
        adj_counts     = counts + pseudocount
        freq_est       = adj_counts / total_adjusted
        cfu_est        = freq_est * sample_cfu

        # Variance from binomial frequency uncertainty only
        var_freq      = freq_est * (1.0 - freq_est) / total_adjusted
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_var_freq  = np.where(freq_est > 0,
                                     var_freq / (freq_est ** 2), 0.0)
        ln_cfu_var        = rel_var_freq

        # Optional additional noise on the log scale
        if extra_noise > 0.0:
            ln_cfu_var = ln_cfu_var + extra_noise ** 2
            ln_cfu_shift = rng.normal(0.0, extra_noise, size=len(genos))
        else:
            ln_cfu_shift = np.zeros(len(genos))

        with np.errstate(divide="ignore"):
            ln_cfu_vals = np.log(cfu_est) + ln_cfu_shift

        # Zero-count entries become NaN
        ln_cfu_vals = np.where(cfu_est > 0, ln_cfu_vals, np.nan)
        ln_cfu_var  = np.where(cfu_est > 0, ln_cfu_var,  np.nan)
        ln_cfu_std_vals = np.sqrt(ln_cfu_var)

        for i, geno in enumerate(genos):
            rows.append({
                "library":       lib,
                "replicate":     rep,
                "condition_pre": cp,
                "genotype":      geno,
                "ln_cfu":        float(ln_cfu_vals[i]),
                "ln_cfu_std":    float(ln_cfu_std_vals[i]),
                "ln_cfu_0_true": float(ln_cfu0[i]),
            })

    return pd.DataFrame(rows)
