"""
Functions for generating phenotypes from mutant libraries and ensemble data.
"""

from tfscreen.calibration import(
    get_wt_k
)
from tfscreen.util import (
    combine_mutation_effects,
    set_categorical_genotype,
    standardize_genotypes,
    argsort_genotypes,
)
from tfscreen.simulate import (
    setup_observable
)

import pandas as pd
import numpy as np
from numpy.random import Generator
from tqdm.auto import tqdm
from scipy.stats import gamma

from typing import Iterable

def _assign_ddG(unique_genotypes,
                ddG_df,
                mut_combine_fcn="sum"):
    """
    Assign a combined ddG value to each genotype.
    
    This function is a simple wrapper around the combine_mutant_effects utility.
    It takes a DataFrame of ddG values for single mutations and combines them
    to calculate the total ddG for multi-mutation genotypes.

    Parameters
    ----------
    unique_genotypes : Iterable[str]
        An iterable of all unique genotype strings.
    ddG_df : pandas.DataFrame
        A DataFrame indexed by single-mutation strings with columns for the
        energetic effect (ddG) on each species in the ensemble.
    mut_combine_fcn : str or callable, default "sum"
        The method for combining ddG effects for multi-mutation genotypes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by genotype with the combined ddG effects.
    """
    return combine_mutation_effects(
        unique_genotypes=unique_genotypes,
        single_mutant_effects=ddG_df,
        mut_combine_fcn=mut_combine_fcn
    )


def _assign_dk_geno(unique_genotypes,
                    shape_param=3,
                    scale_param=0.002,
                    mut_combine_fcn="sum",
                    rng: Generator | None = None):
    """
    Assign a global fitness cost (dk_geno) to each genotype.

    This function simulates pleiotropic fitness effects by sampling a
    per-mutation cost from a gamma distribution, then combines those costs
    for multi-mutation genotypes.

    Parameters
    ----------
    unique_genotypes : Iterable[str]
        An iterable of all unique genotype strings.
    shape_param : float, default 3
        The shape parameter `a` for the gamma distribution.
    scale_param : float, default 0.002
        The scale parameter for the gamma distribution.
    mut_combine_fcn : str or callable, default "sum"
        The method for combining dk_geno effects for multi-mutation genotypes.
    rng : numpy.random.Generator, optional
        An initialized NumPy random number generator. If None, a new default
        generator is created.

    Returns
    -------
    pandas.Series
        A Series indexed by genotype with the combined dk_geno value.
        
    Notes
    -----
    The fitness cost `dk_geno` is calculated as `offset - k`, where `k` is
    drawn from a gamma distribution. This generates a distribution of fitness
    effects (DFE) that is skewed towards deleterious (negative) values,
    which is consistent with biological observations.
    """

    # Initialize random number generator
    if rng is None:
        rng = np.random.default_rng()

    # Find all unique single mutations present in the genotype list
    g_lookup = pd.Series(list(unique_genotypes), 
                         index=unique_genotypes).str.split("/", expand=True)
    if "wt" in g_lookup.index:
        g_lookup.loc["wt", :] = np.nan
    single_mutations = g_lookup.stack().unique()

    # Sample a fitness cost for each unique single mutation
    dk_geno_values = scale_param / 2 - gamma.rvs(
        a=shape_param, 
        scale=scale_param, 
        size=len(single_mutations),
        random_state=rng
    )
    
    # Create the map of single mutations to their fitness cost
    mut_dk_mapper = pd.Series(data=dk_geno_values, index=single_mutations)

    # Use the general helper to combine the single-mutation costs
    return combine_mutation_effects(
        unique_genotypes=unique_genotypes,
        single_mutant_effects=mut_dk_mapper,
        mut_combine_fcn=mut_combine_fcn
    )

    
from typing import Iterable, Union, Callable, Optional

def thermo_to_growth(
    genotypes: Iterable[str],
    sample_df: pd.DataFrame,
    observable_calculator: str,
    observable_calc_kwargs: dict,
    ddG_df: Union[str, pd.DataFrame],
    calibration_data: dict,
    mut_growth_rate_shape: float = 3,
    mut_growth_rate_scale: float = 0.002,
    ddG_combine_fcn: Optional[Union[str, Callable]] = "sum",
    dk_geno_combine_fcn: Optional[Union[str, Callable]] = "sum",
    rng: Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate phenotypes from genotypes and experimental conditions.

    This function takes a set of genotypes and experimental conditions, then
    calculates the predicted phenotype (growth rates) for every combination
    using a thermodynamic model of transcription factor regulation.

    Parameters
    ----------
    genotypes : Iterable[str]
        An iterable of genotype strings (e.g., ["wt", "A47V", "Q93P/A47V"]).
    sample_df : pandas.DataFrame
        DataFrame defining all experimental conditions (titrations, times, etc.).
    observable_calculator : str
        The name of the biophysical model to use (e.g., "eee", "lac").
    observable_calc_kwargs : dict
        Keyword arguments to pass to the constructor of the chosen model.
    ddG_df : str or pandas.DataFrame
        A DataFrame (or path to one) containing the free energy perturbations
        (ddG) for single mutations on each molecular species in the model.
    calibration_data : dict
        Pre-calculated calibration constants required for growth rate prediction.
    mut_growth_rate_shape : float, default 3
        The shape parameter for the gamma distribution used to assign
        pleiotropic fitness costs.
    mut_growth_rate_scale : float, default 0.002
        The scale parameter for the gamma distribution.
    ddG_combine_fcn : str or callable, default "sum"
        The method for combining ddG effects for multi-mutation genotypes.
    dk_geno_combine_fcn : str or callable, default "sum"
        The method for combining pleiotropic fitness effects (dk_geno).
    rng : numpy.random.Generator, optional
        An initialized NumPy random number generator. If None, a new default
        generator is created.

    Returns
    -------
    phenotype_df : pandas.DataFrame
        A long-form DataFrame containing every combination of genotype and
        sample condition, with calculated `theta`, `dk_geno`, `k_pre`, and `k_sel`.
    genotype_ddG_df : pandas.DataFrame
        A DataFrame indexed by genotype with the combined energetic effect
        of each genotype on all model conformations.
    """

    print("Initializing phenotype calculation... ",end="",flush=True)

    # Initialize random number generator
    if rng is None:
        rng = np.random.default_rng()

    # -------------------------------------------------------------------------
    # Load and then sort genotype and sample dataframes in a stereotyped way

    # Make the genotypes unique and sort them in a stereotyped way
    unique_unsorted = np.unique(standardize_genotypes(genotypes))
    genotype_order = argsort_genotypes(unique_unsorted)
    unique_genotypes = unique_unsorted[genotype_order]

    # Sort on as much of the standard sort order as possible. Drop columns from
    # sort that sample_df does not have. 
    standard_sort_order = ["replicate","library","condition_sel",
                           "titrant_name","titrant_conc","t_sel"]
    sort_on = [s for s in standard_sort_order if s in sample_df.columns]
    if len(sort_on) > 0:
        sample_df = sample_df.sort_values(sort_on).reset_index(drop=True)

    # Set up the observable function. This returns two things. 
    # 
    # 1) theta_fcn: a function that takes a 1D array of ddG values for each 
    #    species in the ensemble as its only argument. theta_fcn returns the 
    #    fractional occupancy of the transcription factor binding site at each
    #    of the titrant concentrations specified in sample_df given the ddG 
    #    array. Passing an array of zeros causes the function to return the  
    #    wildtype titration behavior. 
    #
    # 2) ddG_df: a dataframe indexed by single mutation with the ddG of that 
    #    mutation on each of the species in the ensemble. The number of columns
    #.   in this df corresponds to the length of the array expected by theta_fcn.
    #.   For a genotype with one mutation, we could do theta_fcn(ddG_df[genotype,:])
    #.   and get its predicted operator occupancy vs. titrant. 
    
    theta_fcn, ddG_df = setup_observable(observable_calculator,
                                         observable_calc_kwargs,
                                         ddG_df,
                                         sample_df)

    # ------------------------------------------------------------------------
    # Calculate genotype-level effects that are independent of experimental 
    # conditions and store in genotype_df 

    # Calculate the combined effects of mutations on each genotype. This 
    # dataframe has unique genotypes as its index and columns for each ddG 
    # effect in the order expected by theta_fcn
    genotype_ddG_df = _assign_ddG(unique_genotypes,
                                  ddG_df,
                                  ddG_combine_fcn)
    
    # Calculate theta vs. sample_conditions for each unique ddG (genotype).
    # Right now this manually iterates of unique ddG arrays because
    # theta_fcn is not vectorized--each genotype ddG must be run alone. In the 
    # future, vectorizing the theta_fcn call could speed up the overall loop
    # considerably. 
    
    print("Done.",flush=True)

    tqdm.pandas(desc="calculating theta using thermo model")
    theta_out = genotype_ddG_df.progress_apply(
        lambda row: pd.Series(theta_fcn(row.to_numpy())),
        axis=1
    )

    print("Calculating growth rates and building final dataframe... ",
          flush=True,end="")

    # Merge the results with the sample dataframe, creating a giant dataframe
    # of genotypes with every condition
    theta_long_df = theta_out.stack().reset_index()
    theta_long_df.columns = ["genotype","feature_id","theta"]
    phenotype_df = pd.merge(theta_long_df, 
                            sample_df, 
                            left_on='feature_id', 
                            right_index=True)
    
    # Get global shifts in growth rate induced by mutations. 
    genotype_dk_geno_series = _assign_dk_geno(unique_genotypes,
                                              mut_growth_rate_shape,
                                              mut_growth_rate_scale,
                                              dk_geno_combine_fcn,
                                              rng)
    
    # Load the calcualted dk_geno into the phenotype_df
    phenotype_df["dk_geno"] = phenotype_df["genotype"].map(genotype_dk_geno_series)
    
    
    # Get growth-rate in pre-selection condition given theta
    k_pre = get_wt_k(phenotype_df["condition_pre"].to_numpy(),
                     phenotype_df["titrant_name"].to_numpy(),
                     phenotype_df["titrant_conc"].to_numpy(),
                     calibration_data,
                     theta=phenotype_df["theta"].to_numpy())

    # Record k_pre 
    phenotype_df["k_pre"] = k_pre + phenotype_df["dk_geno"].to_numpy()

    # Get growth-rate in selection condition given theta
    k_sel = get_wt_k(phenotype_df["condition_sel"].to_numpy(),
                     phenotype_df["titrant_name"].to_numpy(),
                     phenotype_df["titrant_conc"].to_numpy(),
                     calibration_data,
                     theta=phenotype_df["theta"].to_numpy())

    # Record k_pre 
    phenotype_df["k_sel"] = k_sel + phenotype_df["dk_geno"].to_numpy()

    # -------------------------------------------------------------------------
    # Do some final clean up of the dataframes

    # Organize phenotype_df columns
    final_columns = list(phenotype_df.columns)
    final_columns.remove("feature_id")
    final_columns.remove("theta")
    final_columns.append("theta")
    phenotype_df = phenotype_df.loc[:,final_columns]

    # Move genotype into column in genotype_ddG_df and make categorical
    genotype_ddG_df = (genotype_ddG_df
                       .reset_index()
                       .rename(columns={"index":"genotype"}))
    genotype_ddG_df = set_categorical_genotype(genotype_ddG_df)

    print("Done.",flush=True)

    return phenotype_df, genotype_ddG_df

