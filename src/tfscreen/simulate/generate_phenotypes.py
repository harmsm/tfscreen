"""
Functions for generating phenotypes from mutant libraries and ensemble data.
"""

from tfscreen.calibration import predict_growth_rate

import eee

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def _read_ddG(ddG_spreadsheet,ens):
    """
    Read ddG values from spreadsheet and return a dictionary mapping mutation
    strings to ddG arrays for each species.

    Parameters
    ----------
    ddG_spreadsheet : str or file-like
        Path to spreadsheet containing ddG values.
    ens : eee.Ensemble
        Ensemble object with species information.

    Returns
    -------
    ddG_dict : dict
        Dictionary mapping mutation string to ddG numpy array.
    """
    ddG_table = pd.read_excel(ddG_spreadsheet)
    
    ddG_dict = {}
    for idx in ddG_table.index:
        mut = str(ddG_table.loc[idx,"mut"])
        ddG_values = np.array(ddG_table.loc[idx,ens.species],dtype=float)
        ddG_dict[mut] = ddG_values
    
    ### HACK FOR NOW --> CALL C AS S
    for mut in list(ddG_dict.keys()):
        if mut[-1] == "S":
            ddG_dict[f"{mut[:-1]}C"] = ddG_dict[mut]

    return ddG_dict

    
def generate_phenotypes(genotype_df,
                        sample_df,
                        ensemble_spreadsheet,
                        ddG_spreadsheet,
                        calibration_dict,
                        scale_obs_by=1,
                        mut_growth_rate_std=1,
                        T=310,
                        R=0.001987):
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
    ensemble_spreadsheet : str or file-like
        Path to ensemble spreadsheet.
    ddG_spreadsheet : str or file-like
        Path to ddG spreadsheet.
    calibration_dict : dict
        calibration dictionary with wildtype growth rates under experimental 
        conditions.
    scale_obs_by : float, default = 1
        scale the observable by this factor to account for differences between
        transcription factor and operator concentrations in cells
    mut_growth_rate_std : float, default = 1
        standard deviation on growth rate perturbations
    T : float, optional
        Temperature in Kelvin. Default is 310.
    R : float, optional
        gas constant. Default is 0.001987 (kcal/mol). NOTE: this **must** match
        the units in the ddG and ensemble spreadsheets. 

    Returns
    -------
    phenotype_df : pandas.DataFrame
        DataFrame with phenotypes for each genotype. Contains columns:
        - "clone": genotype string
        - "ddG": ddG values for the genotype
        - "iptg": iptg concentration in mM
        - "fx_occupied": fraction of binding sites occupied
        - "fx_folded": fraction of folded protein
        - "obs": observed phenotype (product of fx_occupied and fx_folded)  
        - "base_growth_rate": growth rate under these conditions without a 
          marker or selection
        - "marker_growth_rate": growth rate under these conditions accounting 
          for marker without selection
        - "select_growth_rate": growth rate under these conditions accounting 
          for marker and selection.
    """
    
    # Prepare output dictionary to hold phenotype data
    phenotype_out = {"genotype":[],
                     "marker":[],
                     "select":[],
                     "iptg":[],
                     "fx_occupied":[],
                     "fx_folded":[],
                     "obs": [],
                     "base_growth_rate": [],
                     "marker_growth_rate":[],
                     "select_growth_rate":[]}
    
    # These will hold ddG and growth rate effects for the genotype_df
    ddG_out = []
    growth_rate_effect_out = []

    # create list of genotypes
    list_of_genotypes = list(genotype_df["genotype"])

    # Create thermodynamic ensemble
    ens = eee.io.read_ensemble(ensemble_spreadsheet,gas_constant=R)

    iptg_chemical_potential = R*T*np.log(sample_df["iptg"])

    ens.read_ligand_dict(ligand_dict={"iptg":iptg_chemical_potential})
    T_array = np.array([T])
    ddG_dict = _read_ddG(ddG_spreadsheet,ens=ens)

    # Get marker, select, and iptg
    marker = np.array(sample_df["marker"])
    select = np.array(sample_df["select"])
    iptg = np.array(sample_df["iptg"])
    
    # Create dummy 
    no_marker = np.array(["none" for _ in range(len(iptg))])
    no_select = np.zeros(len(iptg),dtype=int)

    # Calculate wildtype growth rate as a function of IPTG
    wt_effect, _ = predict_growth_rate(marker=no_marker,
                                       select=no_select,
                                       iptg=iptg,
                                       calibration_dict=calibration_dict)
    
    # Get growth rate with lac repressor but no iptg, selection, or iptg
    wt_base, _ = predict_growth_rate(marker=np.array(["none"]),
                                     select=np.zeros(1,dtype=bool),
                                     iptg=np.zeros(1,dtype=float),
                                     calibration_dict=calibration_dict)
    log_wt_base_growth_rate = np.log(wt_base)

    # Figure out number of points to add    
    num_points = len(wt_effect)

    desc = "{}".format("calculating phenotypes")
    for genotype in tqdm(list_of_genotypes,desc=desc,ncols=800):

        # Create list of mutations from genotype string
        if genotype == "wt":
            genotype_as_list = []
            growth_rate_effect = 0
        else:
            genotype_as_list = genotype.split("/")

            perturb = np.random.normal(0,mut_growth_rate_std,1)[0]
            growth_rate_effect = np.exp(log_wt_base_growth_rate + perturb)
    
        # Create array of ddG values for the genotype
        genotype_ddG = np.zeros(len(ens.species),dtype=float)
        for mut in genotype_as_list:
            genotype_ddG += ddG_dict[mut]
    
        # Record genotype information
        ddG_out.append(genotype_ddG)
        growth_rate_effect_out.append(growth_rate_effect)
        
        # Calculate ensemble features for the genotype
        fx_occupied, fx_folded = ens.get_fx_obs_fast(mut_energy_array=genotype_ddG,
                                                     temperature=T_array)
        obs = fx_occupied * fx_folded * scale_obs_by

        # Calculate growth rate without marker or selection
        base_growth_rate = growth_rate_effect + wt_effect

        # Calculate growth rate with marker but no selection
        marker_growth_rate, _ = predict_growth_rate(marker=marker,
                                                    select=no_select,
                                                    iptg=iptg,
                                                    theta=obs,
                                                    calibration_dict=calibration_dict)
        marker_growth_rate += growth_rate_effect 

        # Predict growth with marker + selection
        select_growth_rate, _ = predict_growth_rate(marker=marker,
                                                    select=select,
                                                    iptg=iptg,
                                                    theta=obs,
                                                    calibration_dict=calibration_dict)
        select_growth_rate += growth_rate_effect 

        # Record phenotype information
        phenotype_out["genotype"].extend([genotype]*num_points)
        phenotype_out["marker"].extend(sample_df["marker"].values)
        phenotype_out["select"].extend(sample_df["select"].values)
        phenotype_out["iptg"].extend(sample_df["iptg"].values)
        phenotype_out["fx_occupied"].extend(fx_occupied)
        phenotype_out["fx_folded"].extend(fx_folded)
        phenotype_out["obs"].extend(obs)
        phenotype_out["base_growth_rate"].extend(base_growth_rate)
        phenotype_out["marker_growth_rate"].extend(marker_growth_rate)
        phenotype_out["select_growth_rate"].extend(select_growth_rate)

    phenotype_df = pd.DataFrame(phenotype_out)

    # update the genotype dataframe
    genotype_df = genotype_df.copy()
    genotype_df["ddG"] = ddG_out
    genotype_df["growth_rate_effect"] = growth_rate_effect_out
    
    return phenotype_df, genotype_df

