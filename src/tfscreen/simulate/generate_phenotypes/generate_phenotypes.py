"""
Functions for generating phenotypes from mutant libraries and ensemble data.
"""

from tfscreen.calibration import predict_growth_rate
from tfscreen.util import read_dataframe

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import gamma


def _read_ddG(ddG_df):
    """
    Read ddG values from spreadsheet and return a dictionary mapping mutation
    strings to ddG arrays for each species.

    Parameters
    ----------
    ddG_df : pandas.DataFrame or str
        dataframe or path to spreadsheet containing ddG values. dataframe must 
        have column 'mut' (with values formatted like M1Q, V238P, etc.) and 
        a column for each molecular species in the model being used. 

    Returns
    -------
    ddG_dict : dict
        Dictionary mapping mutation string to ddG numpy array.
    """

    ddG_df = read_dataframe(ddG_df)

    # Species are all columns except mut
    species_list = [c for c in ddG_df.columns if c != "mut"]
    
    ddG_dict = {}
    for idx in ddG_df.index:
        mut = str(ddG_df.loc[idx,"mut"])
        ddG_values = np.array(ddG_df.loc[idx,species_list],dtype=float)
        ddG_dict[mut] = ddG_values
    
    # If not ddG values for "C" at site, stick in value for "S"
    update_dict = {}
    for k in ddG_dict:
        if k[-1] == "S":
            new_key = f"{k[:-1]}C"
            if new_key not in ddG_dict:
                update_dict[new_key] = ddG_dict[k]
    for k in update_dict:
        ddG_dict[k] = update_dict[k]


    return ddG_dict


def _assign_growth_rate_perturb(genotype_list,
                                shape_param=3,
                                scale_param=0.002):


    singles = [g for g in genotype_list if len(g.split("/")) == 1]

    peturb = scale_param/2 - gamma.rvs(a=shape_param, 
                                       scale=scale_param, 
                                       size=len(singles))
    
    growth_rate_dict = dict(zip(singles,list(peturb)))

    growth_rate_effect = {}
    for genotype in genotype_list:

        # Assume that multi-mutant genotypes sum their individual effects on 
        # growth rate. 
        if genotype == "wt":
            growth_rate_effect[genotype] = 0
        else:
            genotype_as_list = genotype.split("/")
            growth_rate_effect[genotype] = np.sum([growth_rate_dict[g]
                                                   for g in genotype_as_list])
        
    return growth_rate_effect

    
def generate_phenotypes(genotype_df,
                        sample_df,
                        obs_fcn,
                        ddG_df,
                        calibration_dict,
                        mut_growth_rate_shape=3,
                        mut_growth_rate_scale=0.002):
    """
    Generate phenotypes for all genotypes in genotype_df given ensemble and 
    ddG data.

    Parameters
    ----------
    genotype_df : pandas.DataFrame
        dataframe with genotype information. Must contain a column "genotype"
        with genotype strings.
    sample_df : pandas.DataFrame
        dataframe with samples. Expects columns 'marker', 'select', 'iptg' 
    obs_fcn : function
        function for calculating observable values. This function should take 
        an array of ddG values for all species in the model as its only 
        argument and return a 1D array of the observable across the conditions
        specified in the experiment. 
    ddG_df : pandas.DataFrame or str
        dataframe or path to spreadsheet with dataframe. This dataframe should 
        have a 'mut' column and then one column for each species in the model. 
        The order of the columns should match the order of the species in the 
        ddG_array passed to obs_function. 
    calibration_dict : dict
        calibration dictionary with wildtype growth rates under experimental 
        conditions.
    mut_growth_rate_std : float, default = 0.01
        standard deviation on growth rate perturbations (in units of growth 
        rate). 

    Returns
    -------
    phenotype_df : pandas.DataFrame
        DataFrame with phenotypes for each genotype. Contains columns:
        - "clone": genotype string
        - "ddG": ddG values for the genotype
        - "iptg": iptg concentration in mM
        - "obs": observed phenotype (product of fx_occupied and fx_folded)  
        - "base_growth_rate": growth rate under these conditions without a 
          marker or selection
        - "marker_growth_rate": growth rate under these conditions accounting 
          for marker without selection
        - "overall_growth_rate": growth rate under these conditions accounting 
          for marker and selection. (This is the growth rate used for the 
          simulation of each sample.)
    """
    
    # Prepare output dictionary to hold phenotype data
    phenotype_out = {"genotype":[],
                     "marker":[],
                     "select":[],
                     "iptg":[],
                     "obs": [],
                     "base_growth_rate": [],
                     "marker_growth_rate":[],
                     "overall_growth_rate":[]}
    
    # These will hold ddG and growth rate effects for the genotype_df
    ddG_out = []
    growth_rate_effect_out = []

    ddG_dict = _read_ddG(ddG_df)

    # Get marker, select, and iptg
    marker = np.array(sample_df["marker"])
    select = np.array(sample_df["select"])
    iptg = np.array(sample_df["iptg"])
    
    # Create dummy 
    no_marker = np.array(["none" for _ in range(len(iptg))])
    no_select = np.zeros(len(iptg),dtype=int)

    # Calculate wildtype growth rate as a function of IPTG
    no_marker_no_select, _ = predict_growth_rate(
        marker=no_marker,
        select=no_select,
        iptg=iptg,
        calibration_dict=calibration_dict
    )

    # Figure out number of points to add and number of molecular species
    num_points = len(no_marker_no_select)
    num_species = len(ddG_dict[list(ddG_dict.keys())[0]])

    # create list of genotypes
    genotype_list = list(genotype_df["genotype"])

    growth_rate_effect_dict = _assign_growth_rate_perturb(
        genotype_list=genotype_list,
        shape_param=mut_growth_rate_shape,
        scale_param=mut_growth_rate_scale
    )

    desc = "{}".format("calculating phenotypes")
    for genotype in tqdm(genotype_list,desc=desc,ncols=800):

        # Create list of mutations from genotype string. Assume that multi
        # mutant genotypes sum their individual effects. 
        if genotype == "wt":
            genotype_as_list = []
        else:
            genotype_as_list = genotype.split("/")
    
        # Create array of ddG values for the genotype
        genotype_ddG = np.zeros(num_species,dtype=float)
        for mut in genotype_as_list:
            genotype_ddG += ddG_dict[mut]
    
        # Record genotype information 
        ddG_out.append(genotype_ddG)
        growth_rate_effect_out.append(growth_rate_effect_dict[genotype])
        
        # Get observable given ddG
        obs = obs_fcn(genotype_ddG)

        # Calculate growth rate with marker but no selection
        marker_growth_rate, _ = predict_growth_rate(
            marker=marker,
            select=no_select,
            iptg=iptg,
            theta=obs,
            calibration_dict=calibration_dict
        )
        
        # Predict growth with marker + selection (i.e., the real growth rate)
        overall_growth_rate, _ = predict_growth_rate(
            marker=marker,
            select=select,
            iptg=iptg,
            theta=obs,
            calibration_dict=calibration_dict
        )
        
        # The base, marker, and overall growth rates are all perturbed by the 
        # global effect of the mutation on growth rate. 
        growth_rate_effect = growth_rate_effect_dict[genotype]

        # Record phenotype information
        phenotype_out["genotype"].extend([genotype]*num_points)
        phenotype_out["marker"].extend(sample_df["marker"].values)
        phenotype_out["select"].extend(sample_df["select"].values)
        phenotype_out["iptg"].extend(sample_df["iptg"].values)
        phenotype_out["obs"].extend(obs)
        phenotype_out["base_growth_rate"].extend(no_marker_no_select + growth_rate_effect)
        phenotype_out["marker_growth_rate"].extend(marker_growth_rate + growth_rate_effect)
        phenotype_out["overall_growth_rate"].extend(overall_growth_rate + growth_rate_effect)

    phenotype_df = pd.DataFrame(phenotype_out)

    # update the genotype dataframe
    genotype_df = genotype_df.copy()
    genotype_df["ddG"] = ddG_out
    genotype_df["growth_rate_effect"] = growth_rate_effect_out
    
    return phenotype_df, genotype_df

