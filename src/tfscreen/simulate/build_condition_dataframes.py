"""
Functions for building dataframes specifying marker/selection/iptg/time 
conditions.
"""

import pandas as pd
import copy

def _build_condition_dataframe(iptg,
                               marker,
                               select,
                               time=None,
                               replicate=1,
                               current_df=None):
    """
    Build a dataframe with the conditions for the simulation.

    Parameters
    ----------
    iptg : list of float
        List of IPTG concentrations in mM.
    marker : str
        Name of the marker for these conditions.
    select : int or float
        Selection value for the marker (e.g., 0 or 1).
    time : list of float or None
        list of condition sample times in minutes, default=None
    replicate : int, optional
        Replicate number for these conditions. Default is 1.
    current_df : pandas.DataFrame, optional
        Existing DataFrame to append to. If provided, the new conditions will be
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



def build_condition_dataframes(condition_blocks,
                               replicate=1):
    """
    Build dataframes with the conditions for the simulation. These are built 
    combinatorially in time and iptg for a specified marker and selection 
    conditions. One dataframe has all conditions + time, the other has only 
    conditions. 

    Parameters
    ----------
    condition_blocks : list-like
        list of dictionaries. each dictionary should have the following keys:
        - iptg : list of IPTG concentrations in mM.
        - marker : name of the marker for these conditions.
        - select : selection value for the marker (e.g., 0 or 1).
        - time : list of condition sample times in minutes
    replicate : int, optional
        Replicate number for these conditions. Default is 1.
    
    Returns
    -------
    condition_df : pandas.DataFrame
        dataFrame with columns: "replicate", "marker", "select", "iptg", and "time".
    condition_df_no_time : pandas.DataFrame
        dataFrame with columns: "replicate", "marker", "select", and "iptg"
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


    # build the full condition_df with iptg and time
    condition_df_t = None
    for c in condition_blocks:
        condition_df_t = _build_condition_dataframe(**c,
                                                    replicate=replicate,
                                                    current_df=condition_df_t)
    
    # Sort in a stereotyped way
    condition_df_t = condition_df_t.sort_values(["replicate",
                                                 "marker",
                                                 "select",
                                                 "iptg",
                                                 "time"]).reset_index()
    condition_df_t = condition_df_t.drop(columns=["index"])

    cols_to_join = ["marker","select","iptg","time"]
    condition_df_t.index = condition_df_t[cols_to_join].astype(str).agg('-'.join,axis=1)

    # build condition_df that has no time column
    condition_df = None
    for c in condition_blocks:
        c_copy = copy.deepcopy(c)
        c_copy["time"] = None
        condition_df = _build_condition_dataframe(**c_copy,
                                                  replicate=replicate,
                                                  current_df=condition_df)
    
    # Sort in a stereotyped way
    condition_df = condition_df.sort_values(["marker",
                                             "select",
                                             "iptg"]).reset_index()
    condition_df = condition_df.drop(columns=["index"])

    cols_to_join = ["replicate","marker","select","iptg"]
    condition_df.index = condition_df[cols_to_join].astype(str).agg('-'.join,axis=1)

    return condition_df, condition_df_t
    


