"""
Functions for generating phenotypes from mutant libraries and ensemble data.
"""

from tfscreen.calibration import(
    get_wt_k
)
from tfscreen.util import (
    set_categorical_genotype
)
from tfscreen.simulate import (
    setup_observable
)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import gamma


def _assign_ddG(genotype_list,ddG_df):
    """
    Read ddG values and return a mutation-to-ddG dictionary.

    This function processes a DataFrame of free energy changes for all 
    species in an ensemble, converting it into a dictionary for fast
    lookups. The code assumes there are only single mutants in the dataframe.
    Genotypes with multiple mutations (separated by '/') are assumed to behave
    additively within each species. 

    Parameters
    ----------
    genotype_list : list
        A list of all genotype strings (e.g., ["wt", "A1G", "A1G/C2T"]).
    ddG_df : pandas.DataFrame
        A DataFrame of free energy changes, indexed by single mutation strings,
        with columns corresponding to molecular species. This is typically the
        DataFrame returned by `setup_observable`

    Returns
    -------
    ddG_dict : dict
        A dictionary mapping each mutation string to a NumPy array of its
        ddG values for each species.
    """

    # Build ddG dictionary for all genotypes (singles and higher-order)
    ddG_dict = {}
    for g in list(set(genotype_list)):
        if g == "wt":
            ddG_dict["wt"] = np.zeros(len(ddG_df.columns),dtype=float)
        else:
            ddG_dict[g] = np.sum(ddG_df.loc[g.split("/"),:],axis=0)
            
    return ddG_dict


def _assign_dk_geno(genotype_list,
                    shape_param=3,
                    scale_param=0.002):
    """
    Assign a global fitness cost (dk_geno) to each genotype.

    This function simulates the pleiotropic fitness effects of mutations that
    are unrelated to the modeled thermodynamic change. It assigns a fitness
    cost to each unique individual mutation by drawing from a gamma
    distribution. The total fitness cost for a genotype with multiple
    mutations is assumed to be the sum of the costs of its constituent
    mutations.

    Parameters
    ----------
    genotype_list : list of str
        A list of all genotype strings (e.g., ["wt", "A1G", "A1G/C2T"]).
    shape_param : float, optional
        The shape parameter `a` for the gamma distribution used for sampling.
    scale_param : float, optional
        The scale parameter for the gamma distribution used for sampling.

    Returns
    -------
    dk_geno : dict
        A dictionary mapping each full genotype string to its calculated
        total fitness cost (dk_geno).
    """

    # Get all unique mutations (whether seen singly or doubly)
    all_individual_muts = set()
    for g in genotype_list:
        if g != "wt":
            all_individual_muts.update(g.split('/'))
    
    # List of all singles
    unique_singles = sorted(list(all_individual_muts))

    # Get dk_geno effects by random sampling from a gamma distribution
    dk_geno = scale_param/2 - gamma.rvs(a=shape_param, 
                                       scale=scale_param, 
                                       size=len(unique_singles))
    
    # This dataframe maps individual mutations to their fitness effect
    dk_geno_df = pd.DataFrame({"mut":unique_singles,
                              "dk_geno":dk_geno})
    dk_geno_df = dk_geno_df.set_index("mut")

    # Build dk_geno dictionary for all genotypes (singles and higher-order)
    dk_geno_dict = {}
    for g in genotype_list:
        if g == "wt":
            dk_geno_dict[g] = 0
        else:
            dk_geno_dict[g] = np.sum(dk_geno_df.loc[g.split("/"),"dk_geno"])
        
    return dk_geno_dict

    
def generate_phenotypes(genotype_df,
                        sample_df,
                        observable_calculator,
                        observable_calc_kwargs,
                        ddG_spreadsheet,
                        calibration_data,
                        mut_growth_rate_shape=3,
                        mut_growth_rate_scale=0.002):
    """
    Generate phenotypes (growth rates) from genotypes and conditions.

    This is the main simulation engine. It takes a set of genotypes and a set
    of experimental conditions, then calculates the predicted phenotype for
    every combination. The process involves summing free energy changes for
    mutations, assigning a random fitness cost, calculating a biophysical
    observable (e.g., theta), and finally calculating the growth rates in
    pre-selection and selection conditions based on calibration data.

    Parameters
    ----------
    genotype_df : pandas.DataFrame
        DataFrame containing a "genotype" column with all genotypes to be
        simulated.
    sample_df : pandas.DataFrame
        DataFrame defining all experimental conditions (titrations, times, etc.).
    observable_calculator : str
        The name of the biophysical model to use for the calculation. Must be
        "eee" or "lac".
    observable_calc_kwargs : dict
        A dictionary of keyword arguments to pass to the constructor of the
        chosen model. Must contain the key "e_name", which specifies the
        name of the titrant (effector).
    ddG_spreadsheet : str or pandas.DataFrame
        The path to a spreadsheet file (e.g., CSV, Excel) or a pre-loaded
        pandas DataFrame containing the free energy perturbations (ddG) to each
        molecular species in the model. The DataFrame must contain a "mut"
        column for mutation names and one for each molecular species in the 
        model. 
    calibration_data : dict
        A dictionary containing pre-calculated calibration constants required
        by ``tfscreen.calibration.get_wt_k``.
    mut_growth_rate_shape : float, optional
        The shape parameter for the gamma distribution used to assign
        fitness costs.
    mut_growth_rate_scale : float, optional
        The scale parameter for the gamma distribution used to assign
        fitness costs.

    Returns
    -------
    genotype_df : pandas.DataFrame
        The input `genotype_df` with new columns "ddG" and "dk_geno" added.
    phenotype_df : pandas.DataFrame
        A "long" DataFrame containing every combination of genotype and
        sample condition, with calculated observables (theta) and
        phenotypes (k_pre, k_sel).
    """

    # -------------------------------------------------------------------------
    # Load and then sort genotype and sample dataframes in a stereotyped way

    # Work on copies of genotype_df and sample_df
    genotype_df = genotype_df.copy()
    sample_df = sample_df.copy()

    # Make sure the genotype dataframe is sorted by the genotype in a 
    # stereotyped way
    genotype_df = set_categorical_genotype(genotype_df,sort=True)
    
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
                                         ddG_spreadsheet,
                                         sample_df)

    # ------------------------------------------------------------------------
    # Calculate genotype-level effects that are independent of experimental 
    # conditions and store in genotype_df 

    genotype_list = pd.unique(genotype_df["genotype"])

    # Read ddG spreadsheet into a dictionary
    ddG_dict = _assign_ddG(
        genotype_list=genotype_list,
        ddG_df=ddG_df
    )

    # Get global shifts in growth rate induced by mutations. 
    dk_geno_dict = _assign_dk_geno(
        genotype_list=genotype_list,
        shape_param=mut_growth_rate_shape,
        scale_param=mut_growth_rate_scale
    )

    # update the genotype dataframe with the newly calculated ddG and dk_geno
    # effects. 
    genotype_df["ddG"] = genotype_df['genotype'].astype(str).map(ddG_dict)
    genotype_df['dk_geno'] = genotype_df['genotype'].astype(str).map(dk_geno_dict)

    # -------------------------------------------------------------------------
    # Creates every combination of genotype and sample condition via a pandas
    # cross merge.

    # Add merge key for cross merge
    sample_df['_merge_key'] = 1
    genotype_df['_merge_key'] = 1

    # Create phenotype_df 
    phenotype_df = pd.merge(genotype_df, 
                            sample_df, 
                            on='_merge_key').drop('_merge_key', axis=1)

    # -------------------------------------------------------------------------
    # Calculate the observable for every unique genotype and load into 
    # phenotype_df

    # de-duplicate genotypes
    unique_ddG = genotype_df.groupby("genotype",observed=True)["ddG"].first()

    # Calculate theta vs. sample_conditions for each unique ddG (genotype).
    # Right now this manually iterates of unique ddG arrays because
    # theta_fcn is not vectorized--each genotype ddG must be run alone. In the 
    # future, vectorizing the theta_fcn call could speed up the overall loop
    # considerably. 
    theta_out = []
    desc = "calculating theta using thermo model"
    for ddG in tqdm(unique_ddG,desc=desc, ncols=800):
        theta_out.append(theta_fcn(np.array(ddG)))

    # merge theta results back into genotype_df
    ddG_df = unique_ddG.reset_index()
    ddG_df["theta"] = theta_out
    genotype_df = genotype_df.merge(ddG_df[["genotype","theta"]],
                                    on="genotype",
                                    how="left")

    # # Map the results back to the original genotype_df
    # theta_map = pd.Series(theta_out, index=unique_ddG.index)
    # genotype_df['theta'] = genotype_df['genotype'].map(theta_map)

    # Record theta (every genotype, every condition)
    phenotype_df["theta"] = np.concat(genotype_df["theta"].to_numpy())
    
    # Get growth-rate in pre-selection condition given theta
    k_pre = get_wt_k(phenotype_df["condition_pre"].to_numpy(),
                     phenotype_df["titrant_name"].to_numpy(),
                     phenotype_df["titrant_conc"].to_numpy(),
                     calibration_data,
                     theta=phenotype_df["theta"].to_numpy())

    # Record k_pre 
    phenotype_df["k_pre"] = k_pre + phenotype_df["dk_geno"]

    # Get growth-rate in selection condition given theta
    k_sel = get_wt_k(phenotype_df["condition_sel"].to_numpy(),
                     phenotype_df["titrant_name"].to_numpy(),
                     phenotype_df["titrant_conc"].to_numpy(),
                     calibration_data,
                     theta=phenotype_df["theta"].to_numpy())

    # Record k_pre 
    phenotype_df["k_sel"] = k_sel + phenotype_df["dk_geno"]

    # -------------------------------------------------------------------------
    # Do some final clean up of the dataframes

    genotype_df = genotype_df.drop(columns=["_merge_key","theta"])
    phenotype_df = phenotype_df.drop(columns=["ddG"])
    
    return genotype_df, phenotype_df

