"""
Functions for converting phenotype data to growth rates for tfscreen simulations.
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tfscreen import data

def pheno_to_growth(phenotype_df,
                    sel_name,
                    pheno_0_growth,
                    pheno_1_growth,
                    ln_base_growth_rate=-4,
                    ln_base_growth_rate_std=0.5,
                    pheno_name="obs",
                    return_df=False,
                    scale_pheno_by=1):
    """
    Convert phenotype values into growth rates for each clone in a library.

    Parameters
    ----------
    lib_phenotype : dict
        Dictionary mapping library names to lists of clone dicts. Each clone
        dict should contain phenotype information.
    sel_name : str
        Name of the selection condition (used for output keys).
    pheno_0_growth : float
        Growth rate for phenotype value 0.
    pheno_1_growth : float
        Growth rate for phenotype value 1.
    ln_base_growth_rate : float, optional
        Mean of the log base growth rate for clones without a specified base
        growth rate. Default is -4.
    ln_base_growth_rate_std : float, optional
        Standard deviation of the log base growth rate for clones without a
        specified base growth rate. Default is 0.5.
    pheno_name : str, optional
        Key in clone dict for the phenotype value. Default is "obs".
    return_df : str or None, optional
        path to save dataframe. If provided, create a dataframe with all inputs
        and save to this location. Default is None.
    scale_pheno_by : float, optional
        Scaling factor for the phenotype effect on growth rate. Default is 1.

    Returns
    -------
    lib_phenotype : dict
        Updated dictionary with growth rates added to each clone dict.
    df_all : pandas.DataFrame, optional
        Returned only if return_df is True. DataFrame with all clones and parameters.
    """

    for lib in lib_phenotype:

        print(f"Calculating growth rates for library {lib} with {sel_name}", flush=True)

        for clone in tqdm(lib_phenotype[lib]):

            # Assign base growth rate if missing
            if "base_growth_rate" not in clone:
                if len(clone["clone"]) == 0:
                    ln_k = ln_base_growth_rate
                else:
                    if ln_base_growth_rate_std <= 0:
                        ln_k = ln_base_growth_rate
                    else:
                        ln_k = np.random.normal(ln_base_growth_rate, ln_base_growth_rate_std)
                clone["base_growth_rate"] = np.exp(ln_k)

            # Scale phenotype growth rates
            p0g = clone["base_growth_rate"] * pheno_0_growth
            p1g = clone["base_growth_rate"] * pheno_1_growth

            # Compute growth rate under selection
            clone[f"sel_{sel_name}"] = clone[pheno_name]*scale_pheno_by*(p1g - p0g) + p0g
            clone[f"{sel_name}_p0g"] = p0g
            clone[f"{sel_name}_p1g"] = p1g

    if return_df:

        all_clones = []

        params = {
            "sel_name": sel_name,
            "pheno_0_growth": pheno_0_growth,
            "pheno_1_growth": pheno_1_growth,
            "ln_base_growth_rate": ln_base_growth_rate,
            "ln_base_growth_rate_std": ln_base_growth_rate_std,
            "pheno_name": pheno_name,
        }

        for lib_name, clones in lib_phenotype.items():
            for clone in clones:
                clone_copy = clone.copy()
                clone_copy["library"] = lib_name
                clone_copy.update(params)
                all_clones.append(clone_copy)

        df_all = pd.DataFrame(all_clones)
        df_all.to_csv(return_df, index=False)

    return lib_phenotype
