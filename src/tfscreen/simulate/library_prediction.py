
import tfscreen

from tfscreen.simulate import (
    build_sample_dataframes,
    thermo_to_growth,
)
from tfscreen.genetics import library_manager

from typing import Any, Dict, Union
from pathlib import Path

def library_prediction(cf: Union[Dict[str, Any], str, Path],
                       override_keys: dict=None):
    """
    Predict the "ground-truth" phenotypes for a transcription factor screen
    experiment given a library, thermodynamic model, and map between that model
    and bacterial growth rate. 

    Parameters
    ----------
    cf : dict or str or pathlib.Path
        The configuration for the simulation. Can be a dictionary or a path
        to a YAML file containing the configuration parameters.
    override_keys : dict, default=None
        after reading the configuration file, replace keys in the configuration
        with the key/value pairs in override keys. No error checking is done 
        on these keys; the user is responsible for checking their sanity. 

    Returns
    -------
    library_df : pandas.DataFrame
        dataframe holding genotypes one would get when using the degenerate 
        codons and sequences specified in the configuration
    phenotype_df : pandas.DataFrame
        dataframe with predicted fractional occupancy and growth rates for each
        of the genotypes in each of the conditions specified in the configuration
    genotype_ddG_df : pandas.DataFrame
        predicted effects of each genotype on the free energies of all 
        conformations defined in the thermodynamic model 
    """
    
    # -------------------------------------------------------------------------
    # Read inputs and set up simulation

    cf = tfscreen.util.read_yaml(cf,override_keys=override_keys)
    if cf is None:
        err = "Aborting simulation due to configuration error."
        raise RuntimeError(err)
    
    # -------------------------------------------------------------------------
    # Do main calculation

    # Build library_df
    lm = library_manager.LibraryManager(cf)
    library_df = lm.build_library_df()

    # Build sample_df (holds all conditions for the experiment)
    sample_df = build_sample_dataframes(
        cf['condition_blocks'],
        replicate=1
    )

    # Calculate phenotype for each genotype across all conditions in sample_df
    phenotype_df, genotype_ddG_df = thermo_to_growth(
        genotypes=library_df["genotype"],
        sample_df=sample_df,
        observable_calculator=cf["observable_calculator"],
        observable_calc_kwargs=cf["observable_calc_kwargs"],
        ddG_df=cf["ddG_spreadsheet"],
        calibration_data=cf['calibration_file'],
        mut_growth_rate_shape=cf['mut_growth_rate_shape'],
        mut_growth_rate_scale=cf['mut_growth_rate_scale']
    )

    return library_df, phenotype_df, genotype_ddG_df
