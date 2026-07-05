import numpy as np
import pandas as pd

import tfscreen
from tfscreen.simulate import library_prediction, selection_experiment
from tfscreen.util.cli.generalized_main import generalized_main

_CLASS_ORDER = ["wt", "spiked", "single", "double", "other"]


def _classify_genotypes(library_df):
    """
    Map each unique genotype in library_df to a class.

    Classes (in priority order): 'wt' (genotype == "wt"), 'spiked' (any row
    for this genotype has library_origin == "spiked"), 'single'/'double'
    (one/two "/"-separated mutations), 'other' (anything else, e.g. triple
    mutants).

    Parameters
    ----------
    library_df : pandas.DataFrame
        Must have "genotype" and "library_origin" columns. Should be the
        un-expanded library_df returned by library_prediction (one row per
        genotype per library_origin it was actually drawn from).

    Returns
    -------
    dict
        Maps genotype (str) -> class (str).
    """

    origins_by_geno = library_df.groupby("genotype", observed=True)["library_origin"].apply(set)

    def classify(g):
        if g == "wt":
            return "wt"
        if "spiked" in origins_by_geno.get(g, set()):
            return "spiked"
        num_muts = g.count("/") + 1
        if num_muts == 1:
            return "single"
        if num_muts == 2:
            return "double"
        return "other"

    return {g: classify(g) for g in origins_by_geno.index}


def report_cfu0(config_file, num_replicates=5, seed=None):
    """
    Report average ln_cfu0 by genotype class from a simulate config.

    Runs library_prediction once to get ground-truth phenotypes, then runs
    selection_experiment num_replicates times (mirroring tfs-simulate's
    replicate loop, each with an independent seed) to sample independent
    transformation draws. Genotypes are grouped into four classes -- wt,
    spiked (non-wt), single (non-spiked), double (non-spiked) -- and, for
    each class, this prints the number of genotypes present in the library,
    the mean (across replicates) number of genotypes that survive
    transformation (nonzero cfu0), and the mean/std of ln_cfu0 pooled across
    all replicates. Results are printed to stdout.

    Parameters
    ----------
    config_file : str
        Path to the YAML simulate configuration file.
    num_replicates : int
        Number of independent simulated transformation replicates to average
        over. Default 5.
    seed : int, optional
        Random seed. Overrides the seed in the config file when provided.
    """

    cf = tfscreen.util.read_yaml(config_file)
    if seed is not None:
        cf["seed"] = seed

    base_seed = cf.get("seed", None)

    library_df, phenotype_df, genotype_theta_df, parameters_df, binding_theta_df = \
        library_prediction(cf)

    geno_class = _classify_genotypes(library_df)
    n_in_library = pd.Series(geno_class).value_counts()

    all_rows = []
    for rep in range(1, num_replicates + 1):
        print(f"--- Replicate {rep} of {num_replicates} ---", flush=True)

        # Give each replicate a distinct (but reproducible) random seed, same
        # convention as tfs-simulate.
        rep_cf = dict(cf)
        rep_cf["seed"] = (
            base_seed * num_replicates + rep if base_seed is not None else None
        )

        rep_phenotype_df = phenotype_df.copy()
        rep_phenotype_df["replicate"] = rep

        sample_df_rep, counts_df_rep = selection_experiment(
            rep_cf, library_df, rep_phenotype_df
        )

        # ln_cfu_0 is constant across all samples for a given (genotype,
        # replicate, library) -- it is set by transformation, not growth --
        # so recover replicate/library from sample_df and dedupe down to one
        # row per independent transformation draw.
        merged = pd.merge(
            counts_df_rep,
            sample_df_rep[["sample", "replicate", "library"]],
            on="sample",
            how="left",
        )
        deduped = merged.drop_duplicates(subset=["genotype", "replicate", "library"])
        all_rows.append(deduped[["genotype", "replicate", "library", "ln_cfu_0"]])

    combined = pd.concat(all_rows, ignore_index=True)
    combined["genotype"] = combined["genotype"].astype(str)
    combined["class"] = combined["genotype"].map(geno_class)

    print()
    header = (f"{'class':<8} {'n_library':>10} {'mean_n_observed':>16} "
              f"{'mean_ln_cfu0':>14} {'std_ln_cfu0':>12}")
    print(header)
    print("-" * len(header))

    for cls in _CLASS_ORDER:
        cls_rows = combined[combined["class"] == cls]
        if cls_rows.empty and cls not in n_in_library.index:
            continue

        n_lib = int(n_in_library.get(cls, 0))

        finite = cls_rows[np.isfinite(cls_rows["ln_cfu_0"])]

        n_observed_per_rep = finite.groupby("replicate")["genotype"].nunique()
        mean_n_observed = (
            n_observed_per_rep
            .reindex(range(1, num_replicates + 1), fill_value=0)
            .mean()
        )

        mean_ln_cfu0 = finite["ln_cfu_0"].mean() if not finite.empty else float("nan")
        std_ln_cfu0 = finite["ln_cfu_0"].std() if not finite.empty else float("nan")

        print(f"{cls:<8} {n_lib:>10} {mean_n_observed:>16.1f} "
              f"{mean_ln_cfu0:>14.2f} {std_ln_cfu0:>12.2f}")


def main():
    return generalized_main(report_cfu0, manual_arg_types={"seed": int})
