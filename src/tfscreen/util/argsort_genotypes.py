import numpy as np
import pandas as pd

import string

def argsort_genotypes(genotypes):
    """
    Return indexes in an order that would sort a 1D array of genotypes. 
    
    Genotypes are assumed to be wt (case insensitive) or mutations specified 
    like A15V (case *sensitive* to distinguish nucleic and amino acids). 
    Mutation start and end states are assumed to be single characters (A15V,
    not Ala15Val). Multi-mutants are separated by "/" A15V/V72K/Y17S.
    
    + wt is always first
    + It then sorts on number of mutants. (1, 2, 3, etc.)
    + Within mutants, it sorts by site (numeric) then final state
      (alphabetical). For A15V this would be 15, then V. For example:
      [A15V,A15W,T14Y,T14Y/A15W] --> [T14Y,A15V,A15W,T14Y/A15W]. 

    Parameters
    ----------
    genotypes : list-like
        1D list of genoypes as strings

    Returns
    -------
    np.ndarray
        1D integer array that would sort the genotypes passed in.
    """

    aa_to_idx = dict([(a,i) for i, a in enumerate(string.ascii_letters)])

    # Work on a clean numpy array
    genotypes = np.array(genotypes).copy()
    
    # This will hold columns we are going to sort on
    sort_columns = []
    
    # Look for wildtype (case-insensitive)
    wt_mask = pd.Series(genotypes).str.contains('wt',
                                                case=False,
                                                regex=False).values
    wt_sorter = np.array(np.logical_not(wt_mask),dtype=int) # 0 wt, 1 not wt
    sort_columns.append(wt_sorter)
    
    # number of mutants (1, 2, ...) based on "/" split
    num_muts_sorter = np.array([len(g.split("/")) for g in genotypes],dtype=int)
    sort_columns.append(num_muts_sorter)

    # Go through classes of numbers of mutants seen
    num_muts_seen = pd.unique(num_muts_sorter)
    num_muts_seen.sort()

    # Go through 1st mut, 2nd mut, 3rd mut, etc. 
    for num_muts in num_muts_seen:
        match_mask = (wt_sorter > 0) & (num_muts_sorter == num_muts)
        
        for i in range(num_muts):

            # Site (number)
            sites = [int(m.split("/")[i][1:-1]) for m in genotypes[match_mask]]
            this_site = np.full(len(genotypes),np.nan)
            this_site[match_mask] = sites
            sort_columns.append(this_site)

            # mutate to (character, converted to int)
            muts = [aa_to_idx[m.split("/")[i][-1]] for m in genotypes[match_mask]]
            this_aa = np.full(len(genotypes),np.nan)
            this_aa[match_mask] = muts
            sort_columns.append(this_aa)
            
    # Build an array from sort_columns for lex sort. 
    to_sort = (np.array(sort_columns)).T
    
    # Set all nan to lowest possible value. This will make empty sites (no
    # mutation at site) sort above sites with defined numbers
    min_value = np.nanmin(to_sort)         
    to_sort[np.isnan(to_sort)] = min_value - 1 
    
    # convert to int for speed
    to_sort = np.array(to_sort,dtype=int)       

    # Lex sort and return indexes
    keys = tuple(to_sort.T[::-1])
    return np.lexsort(keys)
    