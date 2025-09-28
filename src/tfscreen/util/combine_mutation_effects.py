import pandas as pd
import numpy as np
from typing import Union, Callable, Optional, Iterable

# A dictionary of recognized string aggregators for the combine function
PANDAS_AGGREGATORS = {"sum", "mean", "prod", "min", "max", "std", "var"}

def combine_mutation_effects(
    unique_genotypes: Iterable[str],
    single_mutant_effects: Union[pd.DataFrame, pd.Series],
    mut_combine_fcn: Union[str, Callable] = "sum" 
) -> Union[pd.DataFrame, pd.Series]:
    """
    Combine single-mutant effects into effects for multi-mutation genotypes.

    This utility takes a map of effects for single mutations (A47C, P92E, etc.)
    and combines them for genotypes that contain multiple mutations (A47C/P92E).

    Parameters
    ----------
    unique_genotypes : Iterable[str]
        An iterable of unique genotype strings (e.g., a pandas Index or list).
    single_mutant_effects : pandas.DataFrame or pandas.Series
        A DataFrame or Series indexed by single-mutation strings, where the
        values are the effects to be combined. All single mutations present
        in `unique_genotypes` must be present in this index.
    mut_combine_fcn : str or callable, default "sum"
        The method for combining effects for multi-mutation genotypes.
        - If a **string**, it must be a recognized pandas aggregator
          (e.g., "mean", "prod", "max").
        - If a **callable**, it will be applied to the group of
          single-mutation effects for each multi-mutation genotype. This 
          function must take a pandas dataframe as its only argument and return
          a pandas series with the same number of columns as the input 
          dataframe. 

    Returns
    -------
    pandas.DataFrame or pandas.Series
        An object of the same type as `single_mutant_effects`, indexed by
        genotype, containing the combined effects.

    Raises
    ------
    TypeError
        If `single_mutant_effects` is not a pandas DataFrame or Series, or if
        `mut_combine_fcn` is an invalid type.
    ValueError
        If any single mutations from `unique_genotypes` are not found in the
        index of `single_mutant_effects`, or if `mut_combine_fcn` is an
        unrecognized aggregator string.
    """
    # ------------------ Input Validation ------------------
    if not isinstance(single_mutant_effects, (pd.DataFrame, pd.Series)):
        raise TypeError("`single_mutant_effects` must be a pandas DataFrame or Series.")

    # Handle empty genotype list gracefully
    if not any(unique_genotypes):
        return single_mutant_effects.iloc[0:0]

    # ------------------ Main Logic ------------------
    # Split multi-mutation genotypes into columns of single mutations
    g_lookup = pd.Series(list(unique_genotypes),
                         index=unique_genotypes).str.split("/", expand=True)
    
    # Set "wt" to NaN so it's ignored in lookups (effect is 0 by definition)
    if "wt" in g_lookup.index:
        g_lookup.loc["wt", :] = np.nan

    # Convert from wide (genotypes) to long (single mutations) format
    stacked_muts = g_lookup.stack()
    if stacked_muts.empty: # Case where only 'wt' or no multi-mutants exist
         return single_mutant_effects.iloc[0:0].reindex(g_lookup.index, fill_value=0)

    # Check that all required mutations are present in the effects map
    muts_in_genotypes = set(stacked_muts)
    muts_in_effects_map = set(single_mutant_effects.index)
    if not muts_in_genotypes.issubset(muts_in_effects_map):
        missing = muts_in_genotypes - muts_in_effects_map
        raise ValueError(f"Mutations are missing from the effects map: {sorted(list(missing))}")
    
    # Look up the effects for every single mutation in the long series
    looked_up_effects = single_mutant_effects.loc[stacked_muts.values]
    looked_up_effects.index = stacked_muts.index

    # Group by genotype
    grouped_effects = looked_up_effects.groupby(level=0, observed=True)
    
    # Combine the effects using the specified method
    if isinstance(mut_combine_fcn, str):
        if mut_combine_fcn not in PANDAS_AGGREGATORS:
            allowed = "', '".join(PANDAS_AGGREGATORS)
            raise ValueError(f"'{mut_combine_fcn}' is not a recognized aggregator. Use one of: '{allowed}'.")
        genotype_effects = grouped_effects.agg(mut_combine_fcn)
    elif callable(mut_combine_fcn):
        genotype_effects = grouped_effects.apply(mut_combine_fcn)
    else:
        raise TypeError("`mut_combine_fcn` must be a string or a callable.")

    # Add "wt" back with a zero effect and ensure all genotypes are present
    final_effects = genotype_effects.reindex(g_lookup.index, fill_value=0)
    
    return final_effects