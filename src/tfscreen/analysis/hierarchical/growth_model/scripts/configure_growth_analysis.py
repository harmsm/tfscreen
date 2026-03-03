import os
import yaml
import pandas as pd
import numpy as np
import tfscreen
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.analysis.hierarchical.growth_model.configuration_io import write_configuration
from tfscreen.util.cli.generalized_main import generalized_main

def configure_growth_analysis(growth_df,
                              binding_df,
                              out_root="tfs",
                              condition_growth_model="linear",
                              growth_transition_model="instant",
                              ln_cfu0_model="hierarchical",
                              dk_geno_model="hierarchical",
                              activity_model="horseshoe",
                              theta_model="hill",
                              transformation_model="empirical",
                              theta_growth_noise_model="zero",
                              theta_binding_noise_model="zero",
                              spiked=None):
    """
    Construct the analysis configuration step. This creates a tfs_config.yaml file
    along with tfs_priors.csv and tfs_guesses.csv (if any array parameters exist).

    Parameters
    ----------
    growth_df : pandas.DataFrame or str, optional
        Input data for the growth model (e.g., genotype/cfu measurements).
    binding_df : pandas.DataFrame or str, optional
        Input data for the binding model (e.g., theta vs. titrant measurements).
    out_root : str, optional
        Output file root for the generated configuration files (default 'tfs').
    condition_growth_model: str, optional
        Model to use to describe growth under different conditions (e.g., 
        pheS+4CP). Allowed values are 'linear' (default), 'linear_independent',
        'linear_fixed', 'power', or 'saturation'.
    growth_transition_model : str, optional
        Model to use to describe the transition between the pre-selection
        and selection phases. Allowed values are 'instant' (default), 'memory',
        or 'baranyi'.
    ln_cfu0_model : str, optional
        Model to use to describe ln_cfu0, the initial populations of genotypes
        in each replicate. Only 'hierarchical' is allowed at this point. 
    dk_geno_model : str, optional
        Model to use to describe dk_geno, the pleiotropic effect of a genotype
        on growth, independent of occupancy. Allowed values are 'hierarchical' 
        (default) or 'fixed'.
    activity_model : str, optional
        Model to use to describe activity, a scalar multiplied against 
        occupancy that defines how strongly a genotype alters transcription 
        given its occupancy. Allowed values are 'fixed' (default), 'hierarchical',
        and 'horseshoe'.
    theta_model : str, optional
        Model to use to describe theta, the fractional occupancy of a genotype
        on the transcription factor binding site. Allowed values are 'hill' 
        (default) or 'categorical'. 
    transformation_model : str, optional
        Model for transformation correction. Allowed values are 'single', 
        'empirical', 'congression', or 'logit_norm'. Default 'empirical'.
    theta_growth_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by 
        bacterial growth. Allowed values are 'beta' (default) or 'zero'.
    theta_binding_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by 
        binding. Allowed values are 'beta' (default) or 'zero'.
    spiked : list or str, optional
        Names of genotypes that should be excluded from congression
        correction.

    Returns
    -------
    None
    """

    # Initialize model to build mappings and get guesses
    gm = GrowthModel(growth_df,
                     binding_df,
                     condition_growth=condition_growth_model,
                     growth_transition=growth_transition_model,
                     ln_cfu0=ln_cfu0_model,
                     dk_geno=dk_geno_model,
                     activity=activity_model,
                     theta=theta_model,
                     transformation=transformation_model,
                     theta_growth_noise=theta_growth_noise_model,
                     theta_binding_noise=theta_binding_noise_model,
                     spiked_genotypes=spiked)

    # Write the model configuration to a file. This includes the model component
    # names, the data file paths, and the parameter guesses/priors.
    write_configuration(gm=gm,
                        out_root=out_root,
                        growth_df_path=growth_df if isinstance(growth_df, str) else "growth.csv",
                        binding_df_path=binding_df if isinstance(binding_df, str) else "binding.csv")

def main():
    return generalized_main(configure_growth_analysis,
                            manual_arg_types={"growth_df":str,
                                              "binding_df":str,
                                              "spiked":list},
                            manual_arg_nargs={"spiked":"+"})

if __name__ == "__main__":
    main()
