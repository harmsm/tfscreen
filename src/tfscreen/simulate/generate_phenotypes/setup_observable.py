
from .lac_model import LacModel
from .eee_model import EEEModel

from tfscreen.util import read_dataframe

import numpy as np

AVAILABLE_CALCULATORS = {
    "eee":EEEModel,
    "lac":LacModel,
}

def setup_observable(observable_calculator,
                     observable_calc_kwargs,
                     ddG_spreadsheet,
                     sample_df):
    
    if observable_calculator not in AVAILABLE_CALCULATORS:
        err = f"observable_calculator '{observable_calculator}' not recognized.\n"
        err += "Should be one of:\n"
        for c in AVAILABLE_CALCULATORS:
            err += f"    {c}\n"
        err += "\n"

        raise ValueError(err)
    
    # This is a hack. Annoyingly, I need iptg concentrations to initialize the
    # observables, but iptg concentrations are defined via the conditions, not 
    # the calculator in the yaml. Without a refactor, I can't easily link that
    # information to the observable initialization. So, for now, assume that the
    # observable calc_kwargs have 'e_total', that this maps to molar iptg, and
    # that we can get the correct iptg concs from 1e-3*sample_df iptg... 
    observable_calc_kwargs["e_total"] = np.array(sample_df["iptg"])*1e-3

    # Set up observable calculator object and its observable function
    calculator = AVAILABLE_CALCULATORS[observable_calculator]
    obs_obj = calculator(**observable_calc_kwargs)
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
    
    return obs_fcn, ddG_df