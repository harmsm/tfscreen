
import pandas as pd
from tqdm.auto import tqdm

def _build_sample_dataframe(library,
                            titrant_name,
                            titrant_conc,
                            pre_condition,
                            pre_time,
                            sel_condition,
                            sel_time,
                            replicate=1,
                            current_df=None):
    """
    Build a dataframe of samples for the simulation.

    Parameters
    ----------
    

    time : list of float or None
        list of sample times in minutes, default=None
    replicate : int, optional
        Replicate number for these samples. Default is 1.
    current_df : pandas.DataFrame, optional
        Existing DataFrame to append to. If provided, the new samples will be
        concatenated to this DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: "replicate", "library", "condition",
        "pre_condition", "pre_time", "titrant_name", "titrant_conc" and 
        possibly "time".
    """
    
    out = {"replicate":[],
           "library":[],
           "titrant_name":[],
           "titrant_conc":[],
           "pre_condition":[],
           "pre_time":[],
           "sel_condition":[],
           "sel_time":[]}
    
    for c in titrant_conc:

        out["replicate"].append(replicate)
        out["library"].append(library)
        out["titrant_name"].append(titrant_name)
        out["titrant_conc"].append(c)
        out["pre_condition"].append(pre_condition)
        out["pre_time"].append(pre_time)
        out["sel_condition"].append(sel_condition)
        
    time_stack = []
    for t in sel_time:
        #time_stack.append(df.copy())
        time_stack[-1]["sel_time"] = t

    df = pd.concat(time_stack,ignore_index=True)

    if current_df is not None:
        df = pd.concat([current_df, df], ignore_index=True)
    
    return df


def build_sample_dataframes(condition_blocks,
                            replicate=1):
    """
    Build dataframes of samples for the simulation. These are built 
    combinatorially in time and titrant concentration for the specified 
    conditions. 

    Parameters
    ----------
    condition_blocks : list-like
        list of dictionaries. each dictionary should have the following keys:
        - library : name of the library for these samples.
        - titrant_name : name of titrant
        - titrant_conc : list of titrant concentrations in mM.
        - pre_condition : condition for pre growth
        - pre_time : pre growth time in minutes
        - sel_condition : condition for selective growth
        - sel_time : list of times to take as timepoints
    replicate : int, optional
        Replicate number for these samples. Default is 1.
    
    Returns
    -------
    sample_df : pandas.DataFrame

    """
    
    # Error checking on condition_blocks
    if not hasattr(condition_blocks,"__iter__"):
        err = "condition_blocks should be a list of dictionaries\n"
        raise ValueError(err)
    
    if len(condition_blocks) < 1:
        err = "condition_blocks must have at least one entry\n"
        raise ValueError(err)
    
    types = set([issubclass(type(c),dict) for c in condition_blocks])
    if len(types) != 1 or list(types)[0] is not True:
        err = "condition_blocks must be a list of dictionaries\n"
        raise ValueError(err)

    # build the full sample_df with titrant and time
    sample_df = None
    desc = "{}".format("setting up conditions")
    for c in tqdm(condition_blocks,desc=desc,ncols=800):
        sample_df = _build_sample_dataframe(**c,
                                            replicate=replicate,
                                            current_df=sample_df)
    
    # Sort in a stereotyped way
    sample_df = sample_df.sort_values(["replicate",
                                       "library",
                                       "titrant_name",
                                       "titrant_conc",
                                       "pre_condition",
                                       "pre_time",
                                       "sel_condition",
                                       "sel_time"]).reset_index()

    return sample_df
    


