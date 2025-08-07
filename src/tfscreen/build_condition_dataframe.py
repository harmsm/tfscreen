import pandas as pd

def build_condition_dataframe(iptg,
                              marker,
                              select,
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
    replicate : int, optional
        Replicate number for these conditions. Default is 1.
    current_df : pandas.DataFrame, optional
        Existing DataFrame to append to. If provided, the new conditions will be
        concatenated to this DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: "marker", "select", "iptg", and "replicate".
    """
    
    out = {"marker":[],
           "select":[],
           "iptg":[],
           "replicate":[]}
    
    for c in iptg:
            
        out["iptg"].append(c)
        out["marker"].append(marker)
        out["select"].append(select)
        out["replicate"].append(replicate)
                
    df = pd.DataFrame(out)

    if current_df is not None:
        df = pd.concat([current_df, df], ignore_index=True)
    
    return df
