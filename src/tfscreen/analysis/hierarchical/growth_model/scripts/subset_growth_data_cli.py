import numpy as np
import pandas as pd

from tfscreen.genetics.expand_genotype_columns import expand_genotype_columns
from tfscreen.util.cli import generalized_main, read_lines


def _get_num_muts(df):
    """Return a Series with the number of mutations per genotype row."""
    if "num_muts" not in df.columns:
        df = expand_genotype_columns(df)
    return df["num_muts"]


def _mutations_in_genotype(genotype):
    """Return a frozenset of individual mutation strings for a genotype."""
    if genotype == "wt":
        return frozenset()
    return frozenset(genotype.split("/"))


def _reconcile_doubles_and_singles(double_genos, single_genos, blacklist):
    """
    Return (reconciled_doubles, reconciled_singles).

    Keeps only doubles whose constituent singles are both present in
    single_genos and neither is blacklisted.  The reconciled singles set is
    the collection of all singles that participate in at least one kept double.
    """
    reconciled_doubles = []
    reconciled_singles = set()
    for g in sorted(double_genos):
        muts = _mutations_in_genotype(g)
        if muts <= single_genos and not (muts & blacklist):
            reconciled_doubles.append(g)
            reconciled_singles.update(muts)
    return reconciled_doubles, reconciled_singles


def subset_growth_data(
    growth_file,
    n_singles,
    n_steps,
    out_prefix="growth_subset",
    whitelist_file=None,
    blacklist_file=None,
    random_seed=42,
):
    """
    Subset a growth dataset into titrated training/left-out pairs for
    cross-validation of double-mutant predictive power.

    Selects a fixed set of single-mutant genotypes, finds all double mutants
    constructible from those singles that are present in the dataset, then
    generates n_steps output pairs that progressively include more doubles in
    the training set. The complement doubles (not in training) are written to a
    left-out file for evaluating predictive performance.

    Wildtype rows are always included in every training set.

    Parameters
    ----------
    growth_file : str
        Path to the input growth CSV (must contain a 'genotype' column).
    n_singles : int
        Target number of single-mutant genotypes to include. Whitelist
        genotypes count toward this total.
    n_steps : int
        Number of training/left-out pairs to generate (M). The doubles
        included in training are spaced linearly from 0 to all possible,
        with duplicate counts collapsed.
    out_prefix : str, optional
        Prefix for all output files. Default 'growth_subset'.
    whitelist_file : str, optional
        Path to a plain-text file (one genotype per line, # comments allowed)
        of genotypes that must be included in the selected singles.
    blacklist_file : str, optional
        Path to a plain-text file of genotypes that must never be included.
    random_seed : int, optional
        Random seed for reproducible sampling of singles and doubles ordering.
        Default 42.
    """
    n_singles = int(n_singles)
    n_steps = int(n_steps)
    rng = np.random.default_rng(random_seed)

    df = pd.read_csv(growth_file)
    if "genotype" not in df.columns:
        raise ValueError(f"growth_file '{growth_file}' has no 'genotype' column")

    # Classify genotypes by mutation count
    num_muts = _get_num_muts(df)
    all_genotypes = set(df["genotype"].unique())

    wt_genos = set(df.loc[num_muts == 0, "genotype"].unique())
    single_genos = set(df.loc[num_muts == 1, "genotype"].unique())
    double_genos = set(df.loc[num_muts == 2, "genotype"].unique())

    # Load whitelist / blacklist
    whitelist = set(read_lines(whitelist_file)) if whitelist_file else set()
    blacklist = set(read_lines(blacklist_file)) if blacklist_file else set()

    # wt is always included in every training set; drop it from the whitelist
    # silently so users can include it without causing a validation error.
    whitelist_singles = whitelist - wt_genos

    bad_white = whitelist_singles - single_genos
    if bad_white:
        raise ValueError(
            f"Whitelist contains genotypes not present as singles in the growth "
            f"data: {sorted(bad_white)}"
        )
    overlap = whitelist_singles & blacklist
    if overlap:
        raise ValueError(
            f"Genotypes appear in both whitelist and blacklist: {sorted(overlap)}"
        )

    # Reconcile: keep only doubles whose constituent singles both exist in the
    # data and are not blacklisted.
    reconciled_doubles, _ = _reconcile_doubles_and_singles(
        double_genos, single_genos, blacklist
    )

    print(
        f"Reconciled pool: {len(reconciled_doubles)} doubles with complete cycles.",
        flush=True,
    )

    # Effective target: n_singles, but never below the whitelist size.
    effective_target = max(n_singles, len(whitelist_singles))
    if effective_target > n_singles:
        print(
            f"NOTE: whitelist has {len(whitelist_singles)} singles, which exceeds "
            f"n_singles={n_singles}. Using {effective_target} as the target.",
            flush=True,
        )

    # Phase 1 — seed: add whitelist singles and all reconciled doubles that
    # involve any whitelist single (pulling in their partners too).
    selected_singles = set(whitelist_singles)
    for d in reconciled_doubles:
        muts = _mutations_in_genotype(d)
        if muts & whitelist_singles:
            selected_singles.update(muts)

    if whitelist_singles:
        print(
            f"Whitelist seeded {len(whitelist_singles)} singles → "
            f"{len(selected_singles)} after expanding their cycles.",
            flush=True,
        )

    # Phase 2 — expand: iterate shuffled cycles until effective_target reached.
    shuffled_doubles = list(reconciled_doubles)
    rng.shuffle(shuffled_doubles)

    for d in shuffled_doubles:
        if len(selected_singles) >= effective_target:
            break
        selected_singles.update(_mutations_in_genotype(d))

    # Phase 3 — trim: if seeding/expansion overshot, remove non-whitelist
    # singles one at a time (randomly) until effective_target is reached.
    if len(selected_singles) > effective_target:
        trimmable = list(selected_singles - whitelist_singles)
        rng.shuffle(trimmable)
        while len(selected_singles) > effective_target:
            selected_singles.discard(trimmable.pop())

    if len(selected_singles) < n_singles:
        print(
            f"WARNING: only {len(selected_singles)} singles available "
            f"(requested {n_singles}).",
            flush=True,
        )

    print(f"Selected {len(selected_singles)} single-mutant genotypes.", flush=True)

    # Build double universe: reconciled doubles whose constituent singles are
    # both in the selected set
    double_universe = [
        g for g in reconciled_doubles
        if _mutations_in_genotype(g) <= selected_singles
    ]

    rng.shuffle(double_universe)
    n_doubles_total = len(double_universe)

    print(
        f"Double universe: {n_doubles_total} doubles constructible from "
        f"selected singles and present in the dataset.",
        flush=True,
    )

    # Generate n_steps linearly-spaced counts, then deduplicate
    raw_counts = np.linspace(0, n_doubles_total, n_steps)
    step_counts = sorted(set(int(round(c)) for c in raw_counts))
    actual_steps = len(step_counts)

    if actual_steps < n_steps:
        print(
            f"NOTE: deduplicated to {actual_steps} unique steps "
            f"(double universe smaller than n_steps).",
            flush=True,
        )

    # Base training genotype set: wt + selected singles
    base_training = wt_genos | selected_singles
    n_sing = len(selected_singles)
    pad = max(len(str(n_sing)), len(str(n_doubles_total)))

    for i, k in enumerate(step_counts, start=1):
        train_doubles = set(double_universe[:k])
        leftout_doubles = [g for g in double_universe[k:]]

        train_genos = base_training | train_doubles
        train_df = df[df["genotype"].isin(train_genos)].copy()

        n_doub = len(train_doubles)

        stem = f"{out_prefix}_{n_sing:0{pad}d}-singles_{n_doub:0{pad}d}-doubles"
        growth_out = f"{stem}_growth.csv"
        leftout_out = f"{stem}_leftout.txt"

        train_df.to_csv(growth_out, index=False)

        with open(leftout_out, "w") as fh:
            fh.write("# Double mutants constructible from selected singles not in training set\n")
            for g in leftout_doubles:
                fh.write(f"{g}\n")

        print(
            f"  Step {i:02d}/{actual_steps}: {n_doub} doubles in training, "
            f"{len(leftout_doubles)} left out → {growth_out}",
            flush=True,
        )

    print("Done.", flush=True)


def main():
    generalized_main(
        subset_growth_data,
        manual_arg_types={
            "n_singles": int,
            "n_steps": int,
            "whitelist_file": str,
            "blacklist_file": str,
            "random_seed": int,
        },
    )


if __name__ == "__main__":
    main()
