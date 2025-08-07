"""
Functions for generating phenotypes from mutant libraries and ensemble data.
"""
from tfscreen import data

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
                        condition_df,
                        ensemble_spreadsheet,
                        ddG_spreadsheet,
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
    ensemble_spreadsheet : str or file-like
        Path to ensemble spreadsheet.
    ddG_spreadsheet : str or file-like
        Path to ddG spreadsheet.
    concs_mM : array-like
        Ligand concentrations in mM.
    T : float, optional
        Temperature in Kelvin. Default is 310.
    R : float, optional
        Gas constant. Default is 0.001987.

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
    ens = eee.io.read_ensemble(ensemble_spreadsheet) 
    ens.read_ligand_dict(ligand_dict={"iptg":condition_df["iptg"]*1e3})
    T_array = np.array([T])
    ddG_dict = _read_ddG(ddG_spreadsheet,ens=ens)

    # Calculate wildtype growth rate as a function of IPTG
    wt_effect = np.array([data.wt_growth(iptg)
                          for iptg in condition_df["iptg"]])
    
    # get base growth rate for wildtype
    log_wt_base_growth_rate = np.log(data.wt_growth(0))
    
    # Assign the appropriate marker and selection polynomials. These will have
    # mutant-specific effects because they depend on fraction of binding sites
    # occupied. We apply to the correct conditions based on the masks we 
    # store in marker_masks and selection_masks.

    unique_markers = pd.unique(condition_df["marker"])
    num_markers = len(unique_markers)

    marker_polys = [data.markers[m] for m in unique_markers]
    marker_masks = [np.array(condition_df["marker"] == m)
                    for m in unique_markers]

    selection_polys = [data.selectors[m] for m in unique_markers]
    selection_masks = [np.logical_and(condition_df["marker"] == m,
                                      condition_df["select"] == 1)
                                      for m in unique_markers]

    # Figure out number of points to add    
    num_points = len(wt_effect)

    print(f"calculating phenotypes",flush=True)
    for genotype in tqdm(list_of_genotypes):

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

        # Calculate the growth rates
        base_growth_rate = wt_effect + growth_rate_effect
        marker_growth_rate = base_growth_rate.copy()
        select_growth_rate = base_growth_rate.copy()
        
        # Go through marker and selection
        for i in range(num_markers):
            marker_growth_rate[marker_masks[i]] += marker_polys[i](obs[marker_masks[i]])

            select_growth_rate[marker_masks[i]] += marker_polys[i](obs[marker_masks[i]])
            select_growth_rate[selection_masks[i]] += selection_polys[i](obs[selection_masks[i]])

        # Record phenotype information
        phenotype_out["genotype"].extend([genotype]*num_points)
        phenotype_out["marker"].extend(condition_df["marker"].values)
        phenotype_out["select"].extend(condition_df["select"].values)
        phenotype_out["iptg"].extend(condition_df["iptg"].values)
        phenotype_out["fx_occupied"].extend(fx_occupied)
        phenotype_out["fx_folded"].extend(fx_folded)
        phenotype_out["obs"].extend(obs)
        phenotype_out["base_growth_rate"].extend(base_growth_rate)
        phenotype_out["marker_growth_rate"].extend(marker_growth_rate)
        phenotype_out["select_growth_rate"].extend(select_growth_rate)

    print("storing phenotypes",flush=True)

    phenotype_df = pd.DataFrame(phenotype_out)

    # update the genotype dataframe
    genotype_df = genotype_df.copy()
    genotype_df["ddG"] = ddG_out
    genotype_df["growth_rate_effect"] = growth_rate_effect_out
    
    return phenotype_df, genotype_df

