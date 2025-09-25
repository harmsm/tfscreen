
from tfscreen.models.lac_model import LacModel
from tfscreen.models.eee_model import EEEModel

from tfscreen.util import read_dataframe

import numpy as np
import pandas as pd

AVAILABLE_CALCULATORS = {
    "eee":EEEModel,
    "lac":LacModel,
}

def setup_observable(observable_calculator,
                     observable_calc_kwargs,
                     ddG_spreadsheet,
                     sample_df):
    """
    Prepare the observable calculator and energy DataFrame for a simulation.

    This function acts as a factory and validation layer. It selects a
    biophysical model ('calculator'), validates its keyword arguments, and
    ensures consistency between the experimental conditions defined in
    `sample_df` and the energy perturbations in `ddG_spreadsheet`. It returns
    the core components needed to run a simulation: a calculation function
    and a validated DataFrame of free energy changes.

    Parameters
    ----------
    observable_calculator : str
        The name of the biophysical model to use for the calculation. Must be
        a key in the `AVAILABLE_CALCULATORS` dictionary (e.g., "eee", "lac").
    observable_calc_kwargs : dict
        A dictionary of keyword arguments to pass to the constructor of the
        chosen model. Must contain the key "e_name", which specifies the
        name of the titrant (effector).
    ddG_spreadsheet : str or pandas.DataFrame
        The path to a spreadsheet file (e.g., CSV, Excel) or a pre-loaded
        pandas DataFrame containing the free energy perturbations (ddG).
        The DataFrame must contain a "mut" column for mutation names and
        columns for each molecular species required by the model.
    sample_df : pandas.DataFrame
        A DataFrame defining the experimental conditions for the simulation.
        It must contain "titrant_name" and "titrant_conc" columns. The
        simulation currently only supports a single unique titrant name
        across all conditions.

    Returns
    -------
    obs_fcn : callable
        The function from the instantiated model object that calculates the
        observable (e.g., fractional saturation). It is ready to be called
        with ddG values.
    ddG_df : pandas.DataFrame
        The validated and subsetted ddG DataFrame, containing only the "mut"
        column and the species columns required by the chosen model.

    Raises
    ------
    ValueError
        If `observable_calculator` is not recognized.
        If "e_name" is not in `observable_calc_kwargs`.
        If `sample_df` contains more than one unique "titrant_name".
        If the "e_name" from kwargs does not match the "titrant_name".
        If the `ddG_spreadsheet` is missing the "mut" column or any
        species columns required by the model.
    """
    
    if observable_calculator not in AVAILABLE_CALCULATORS:
        err = f"observable_calculator '{observable_calculator}' not recognized.\n"
        err += "Should be one of:\n"
        for c in AVAILABLE_CALCULATORS:
            err += f"    {c}\n"
        err += "\n"

        raise ValueError(err)


    # Get the user-specified e-name
    if "e_name" not in observable_calc_kwargs:
        err = "e_name must be defined in `observable_calc_kwargs`\n"
        raise ValueError(err)

    # Get e name from the inputs    
    e_name = observable_calc_kwargs["e_name"]

    # Make sure the user only specified one titrant name in conditions
    sample_titrants = list(pd.unique(sample_df["titrant_name"]))
    if len(sample_titrants) != 1:
        err = "the simulation currently only supports one titrant_name over all\n"
        err += "conditions. To run more than one titrant_name, run multiple sims\n"
        err += "and concatenate.\n"
        raise ValueError(err)
    
    # Make sure the titrant condition is the same as the e_name specified in 
    # the thermodynamic model
    if sample_titrants[0] != e_name:
        err = f"obs_e_name '{e_name}' does not match the titrant specified\n"
        err += f"in the conditions ('{sample_titrants[0]}')\n"
        raise ValueError(err)

    # Set up final calc_kwargs
    calc_kwargs = {k: v for k, v in observable_calc_kwargs.items() if k != "e_name"}
    
    # Get e_total (mM from conditions to molar for sims). We checked to make 
    # sure the 'titrant_name' column matched; now grab 'titrant_conc'. 
    calc_kwargs["e_total"] = np.array(sample_df["titrant_conc"])*1e-3

    # Set up observable calculator object and its observable function
    calculator = AVAILABLE_CALCULATORS[observable_calculator]
    obs_obj = calculator(**calc_kwargs)
    obs_fcn = obs_obj.get_obs

    # Read ddG spreadsheet
    ddG_df = read_dataframe(ddG_spreadsheet)

    # Get mutations and species ddG from the spreadsheet, checking for errors
    # along the way. 
    if "mut" not in ddG_df.columns:
        err = "ddG_spreadsheet must have a 'mut' column\n"
        raise ValueError(err)
    
    missing_columns = set(obs_obj.species).difference(set(ddG_df.columns))
    
    if len(missing_columns) > 0:
        err = "not all molecular species in ddG file. Missing species:\n"
        missing_columns = list(missing_columns)
        missing_columns.sort()
        for c in missing_columns:
            err += f"    {c}\n"
        err += "\n"
        raise ValueError(err)

    # Create final dataframe with mutations and ddG for species relevant to
    # this model. 
    columns_to_take = ["mut"]
    columns_to_take.extend(list(obs_obj.species))
    ddG_df = ddG_df.loc[:,columns_to_take]
    ddG_df = ddG_df.set_index("mut")
    
    return obs_fcn, ddG_df