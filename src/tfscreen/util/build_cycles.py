from tfscreen.util import argsort_genotypes

import pandas as pd
import numpy as np

def build_cycles(genotypes):
    """
    Identify all valid thermodynamic mutant cycles from a list of genotypes.

    A valid cycle consists of (wt, m1, m2, m12) where wt is the wild-type,
    m1 and m2 are single mutants, and m12 is the corresponding double mutant.
    This function finds all combinations where wt, m1, m2, and m12 are all
    present in the input `genotypes`.

    Parameters
    ----------
    genotypes : array-like
        A 1D array or list of unique genotype strings.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_cycles, 4) where each row contains the
        genotype strings for a valid cycle: [wt, m1, m2, m12].
    """
    
    # Sort genotypes in canonical order
    genotypes = pd.unique(pd.Series(genotypes))
    # idx = tfscreen.util.argsort_genotypes(genotypes)
    genotypes = np.array(genotypes, dtype=str)
    idx = argsort_genotypes(genotypes)
    genotypes = genotypes[idx]

    # Make sure wildtype is present, otherwise no cycles can be calculated
    if not np.any(np.char.lower(genotypes) == 'wt'):
        raise ValueError("wt must be in genotypes to build mutant cycles")
    wt_idx = np.where(np.char.lower(genotypes) == 'wt')[0][0]

    # Create a DataFrame to analyze mutations
    mut_df = pd.DataFrame({"genotype": genotypes})

    # Count the number of mutations by looking for "/"
    num_muts = mut_df["genotype"].str.count('/') + 1
    num_muts.iloc[wt_idx] = 0
    mut_df["num_muts"] = num_muts

    # Identify single and double mutants
    singles = mut_df[mut_df["num_muts"] == 1]
    doubles = mut_df[mut_df["num_muts"] == 2]
    if singles.empty or doubles.empty:
        return np.empty((0, 4), dtype=object)

    # Create a simple series to map single mutant genotypes to their index
    single_to_idx = pd.Series(singles.index, index=singles["genotype"])

    # Split doubles into their constituent single mutations
    pairs = doubles["genotype"].str.split('/', expand=True).to_numpy()

    # Only keep pairs where both constituent singles exist
    good_mask = np.isin(pairs, single_to_idx.index).all(axis=1)
    if not np.any(good_mask):
        return np.empty((0, 4), dtype=object)

    pairs = pairs[good_mask]
    doubles = doubles[good_mask]

    # Build integer array of cycles using original genotype indices
    cycles = np.zeros((len(doubles), 4), dtype=int)
    cycles[:, 0] = wt_idx  
    cycles[:, 1] = single_to_idx.loc[pairs[:, 0]].values
    cycles[:, 2] = single_to_idx.loc[pairs[:, 1]].values
    cycles[:, 3] = doubles.index.values

    # Convert index array back to genotype strings
    genotype_cycles = genotypes[cycles]

    return genotype_cycles

