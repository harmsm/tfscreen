import pandas as pd
import numpy as np
import re

_MUT_PAT = re.compile(r'^[A-Z]\d+[A-Z]$')


def count_mutation_backgrounds(genotype_input):
    """
    For each mutation slot in each genotype, count the number of distinct
    genotypes in the dataset that contain that mutation.

    This measures how well-sampled each mutation is: a value of 3 for M42I
    means three different genotypes in the dataset carry M42I (e.g., M42I,
    M42I/H74A, M42I/K84L).

    Parameters
    ----------
    genotype_input : pandas.DataFrame or iterable of str
        A DataFrame with a 'genotype' column, or an iterable of genotype
        strings. Genotypes follow the tfscreen convention: 'wt', 'H74A',
        'H74A/K84L', etc.

    Returns
    -------
    pandas.DataFrame
        A copy of the input data with N new integer columns inserted after the
        'genotype' column, where N is the maximum number of mutations in any
        single genotype. Columns are named ``mut_backgrounds_1``,
        ``mut_backgrounds_2``, etc. Values are 0 for 'wt' or for slots with no
        mutation.

    Raises
    ------
    ValueError
        If the input is a DataFrame without a 'genotype' column.

    Examples
    --------
    >>> import pandas as pd
    >>> genotypes = ["wt", "M42I", "H74A", "M42I/H74A", "M42I/K84L", "H74A/K84L"]
    >>> count_mutation_backgrounds(genotypes)
          genotype  mut_backgrounds_1  mut_backgrounds_2
    0           wt                  0                  0
    1         M42I                  3                  0
    2         H74A                  3                  0
    3    M42I/H74A                  3                  3
    4    M42I/K84L                  3                  2
    5    H74A/K84L                  3                  2
    """

    if isinstance(genotype_input, pd.DataFrame):
        if "genotype" not in genotype_input.columns:
            raise ValueError("Input DataFrame must have a 'genotype' column.")
        df = genotype_input.copy()
    else:
        df = pd.DataFrame({"genotype": np.asarray(genotype_input)})

    # Work on a positionally-indexed copy so row_pos is always 0..n_rows-1
    genotypes = df["genotype"].astype(str).reset_index(drop=True)
    n_rows = len(genotypes)

    if n_rows == 0:
        return df

    # Split "M42I/H74A" into per-position columns: col 0 = "M42I", col 1 = "H74A"
    indiv_muts = genotypes.str.split("/", expand=True)

    # Boolean mask: True only for tokens that look like a valid mutation (e.g. H74A)
    valid_mask = pd.DataFrame(False, index=indiv_muts.index, columns=indiv_muts.columns)
    for col in indiv_muts.columns:
        valid_mask[col] = indiv_muts[col].str.match(_MUT_PAT, na=False)

    max_muts = int(valid_mask.sum(axis=1).max())
    if max_muts == 0:
        return df

    # For each unique mutation, collect the set of row positions that contain it
    mut_to_rows: dict[str, set] = {}
    for col in range(indiv_muts.shape[1]):
        col_valid = valid_mask.iloc[:, col]
        for row_pos in col_valid[col_valid].index:
            token = indiv_muts.iloc[row_pos, col]
            if token not in mut_to_rows:
                mut_to_rows[token] = set()
            mut_to_rows[token].add(row_pos)

    mut_background_count = {mut: len(rows) for mut, rows in mut_to_rows.items()}

    # Build output array: shape (n_rows, max_muts)
    out = np.zeros((n_rows, max_muts), dtype=int)
    for col_i in range(min(max_muts, indiv_muts.shape[1])):
        col_valid = valid_mask.iloc[:, col_i]
        for row_pos in col_valid[col_valid].index:
            token = indiv_muts.iloc[row_pos, col_i]
            out[row_pos, col_i] = mut_background_count.get(token, 0)

    # Insert new columns immediately after 'genotype'
    geno_pos = list(df.columns).index("genotype")
    for i in range(max_muts):
        col_name = f"mut_backgrounds_{i + 1}"
        df.insert(geno_pos + 1 + i, col_name, out[:, i])

    return df
