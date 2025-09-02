"""
Functions for building dataframes of samples (replicate|marker|selection|iptg)
and timepoints (replicate|marker|selection|iptg|time).
"""

import pandas as pd
from tqdm.auto import tqdm

import copy

def _build_sample_dataframe(marker,
                            select,
                            iptg,
                            time=None,
                            replicate=1,
                            current_df=None):
    """
    Build a dataframe of samples for the simulation.

    Parameters
    ----------
    marker : str
        Name of the marker for these samples.
    select : int or float
        Selection value for the marker (e.g., 0 or 1).
    iptg : list of float
        List of IPTG concentrations in mM.
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
        DataFrame with columns: "replicate", "marker", "select", "iptg", and 
        possibly "time".
    """
    
    out = {"replicate":[],
           "marker":[],
           "select":[],
           "iptg":[]}
    
    for c in iptg:

        out["replicate"].append(replicate)
        out["marker"].append(marker)
        out["select"].append(select)
        out["iptg"].append(c)
    
    df = pd.DataFrame(out)
    if time is not None:

        time_stack = []
        for t in time:
            time_stack.append(df.copy())
            time_stack[-1]["time"] = t

        df = pd.concat(time_stack,ignore_index=True)

    if current_df is not None:
        df = pd.concat([current_df, df], ignore_index=True)
    
    return df



def build_sample_dataframes(condition_blocks,
                            replicate=1):
    """
    Build dataframes of samples for the simulation. These are built 
    combinatorially in time and iptg for the specified marker and selection 
    conditions. One dataframe has all timepoints (samples + time), the other has
    only samples (replicate|marker|select|iptg).

    Parameters
    ----------
    condition_blocks : list-like
        list of dictionaries. each dictionary should have the following keys:
        - marker : name of the marker for these samples.
        - select : selection value for the marker 
        - iptg : list of IPTG concentrations in mM.
        - time : list of times to take timepoints
    replicate : int, optional
        Replicate number for these samples. Default is 1.
    
    Returns
    -------
    sample_df : pandas.DataFrame
        dataFrame with columns: "replicate", "marker", "select", "iptg", and
        "time". The index will be a string '{replicate}|{marker}|{select}|{iptg}|{time}'
    sample_df_no_time : pandas.DataFrame
        dataFrame with columns: "replicate", "marker", "select", and "iptg". The
        index will be a string '{replicate}|{marker}|{select}|{iptg}'
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

    # build the full sample_df with iptg and time
    sample_df_t = None
    desc = "{}".format("setting up conditions")
    for c in tqdm(condition_blocks,desc=desc,ncols=800):
        sample_df_t = _build_sample_dataframe(**c,
                                              replicate=replicate,
                                              current_df=sample_df_t)
    
    # Sort in a stereotyped way
    sample_df_t = sample_df_t.sort_values(["replicate",
                                           "marker",
                                           "select",
                                           "iptg",
                                           "time"]).reset_index()
    sample_df_t = sample_df_t.drop(columns=["index"])

    cols_to_join = ["replicate","marker","select","iptg","time"]
    sample_df_t.index = sample_df_t[cols_to_join].astype(str).agg('|'.join,axis=1)

    # build sample_df that has no time column
    sample_df = None
    for c in condition_blocks:
        c_copy = copy.deepcopy(c)
        c_copy["time"] = None
        sample_df = _build_sample_dataframe(**c_copy,
                                            replicate=replicate,
                                            current_df=sample_df)
    
    # Sort in a stereotyped way
    sample_df = sample_df.sort_values(["replicate",
                                       "marker",
                                       "select",
                                       "iptg"]).reset_index()
    sample_df = sample_df.drop(columns=["index"])

    cols_to_join = ["replicate","marker","select","iptg"]
    sample_df.index = sample_df[cols_to_join].astype(str).agg('|'.join,axis=1)


    return sample_df, sample_df_t
    


