import pandas as pd
from tqdm.auto import tqdm
import itertools

def build_sample_dataframes(condition_blocks, replicate=1):
    """
    Build a DataFrame of experimental conditions from a list of blocks.

    This function takes a compact, list-based representation of experimental
    conditions and expands it into a "tidy" pandas DataFrame. Each row in the
    output DataFrame represents a single, unique experimental condition. The
    function computes the Cartesian product of any parameters specified as
    lists within a condition block (e.g., `titrant_conc` and `t_sel`).

    Parameters
    ----------
    condition_blocks : list of dict
        A list where each dictionary defines a block of related experimental
        conditions. Each dictionary must contain keys defining the experimental
        parameters.
    replicate : int, optional
        The replicate number to assign to all generated conditions, by default 1.

    Returns
    -------
    pandas.DataFrame
        A long-form DataFrame where each row is a unique sample condition,
        sorted by experimental parameters.

    Raises
    ------
    ValueError
        If `condition_blocks` is not a list, is empty, or if any of its
        elements are not dictionaries.

    Notes
    -----
    Each dictionary in the `condition_blocks` list is expected to have a
    structure similar to the following:

    .. code-block:: yaml

        {
            "library": "pheS",
            "titrant_name": "iptg",
            "titrant_conc": [0, 1],
            "condition_pre": "pheS-4CP",
            "t_pre": 30,
            "condition_sel": "pheS-4CP",
            "t_sel": [80, 95, 110]
        }

    """
    # --- Input Validation ---
    if not isinstance(condition_blocks, list) or not condition_blocks:
        raise ValueError("condition_blocks must be a non-empty list.")
    
    if not all(isinstance(c, dict) for c in condition_blocks):
        raise ValueError("All items in condition_blocks must be dictionaries.")

    # --- DataFrame Construction ---
    all_block_dfs = []
    desc = "Setting up conditions"
    
    for block in tqdm(condition_blocks, desc=desc, ncols=800):
        # Use itertools.product to get the cartesian product of the lists
        variable_params = list(itertools.product(
            block["titrant_conc"],
            block["t_sel"]
        ))
        
        # Create a list of dictionaries, one for each experimental row
        rows = [
            {
                "replicate": replicate,
                "library": block["library"],
                "titrant_name": block["titrant_name"],
                "condition_pre": block["condition_pre"],
                "t_pre": block["t_pre"],
                "condition_sel": block["condition_sel"],
                "titrant_conc": conc,
                "t_sel": t
            }
            for conc, t in variable_params
        ]
        
        all_block_dfs.append(pd.DataFrame(rows))

    # Perform a single, efficient concatenation of all DataFrames
    if not all_block_dfs:
        return pd.DataFrame() # Return empty df if no conditions were generated
        
    sample_df = pd.concat(all_block_dfs, ignore_index=True)
    
    # Sort in a stereotyped way
    sort_columns = [
        "replicate", "library", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "t_sel"
    ]
    sample_df = sample_df.sort_values(sort_columns).reset_index(drop=True)

    return sample_df