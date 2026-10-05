import pandas as pd
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
    for block in condition_blocks:
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

    sample_df = pd.concat(all_block_dfs, ignore_index=True)
    
    # Sort in a stereotyped way
    sort_columns = [
        "replicate", "library", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "t_sel"
    ]
    sample_df = sample_df.sort_values(sort_columns).reset_index(drop=True)

    return sample_df

# Columns of a design file: the tube table tfs-process-counts reads, without
# the tube totals (the simulator makes those).
DESIGN_COLUMNS = ["sample", "library", "replicate", "condition_pre", "t_pre",
                  "condition_sel", "t_sel", "titrant_name", "titrant_conc"]

# The columns that make two tubes the same growth condition.
CONDITION_COLUMNS = ["library", "titrant_name", "condition_pre", "t_pre",
                     "condition_sel", "titrant_conc", "t_sel"]


def read_design(design):
    """
    Read and check a simulation design: one row per sequenced tube.

    The design replaces ``condition_blocks`` when a simulation should follow
    a real experiment's layout exactly (irregular time points, per-replicate
    differences). Its format is the tube table ``tfs-process-counts`` reads;
    extra columns (tube totals, OD600, read counts) are ignored.

    Parameters
    ----------
    design : str or pandas.DataFrame
        Path to the tube table (CSV/TSV/Excel) or the table itself.

    Returns
    -------
    pandas.DataFrame
        The ``DESIGN_COLUMNS``, ``sample`` as str and ``replicate`` as int.

    Raises
    ------
    ValueError
        On a missing column, a repeated sample name, or two tubes of one
        replicate with the same growth condition.
    """
    from tfscreen.util.io import read_dataframe

    df = read_dataframe(design)
    if df.index.name == "sample":
        df = df.reset_index()
    missing = [c for c in DESIGN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"design is missing column(s) {missing}; it needs "
                         f"{DESIGN_COLUMNS} (a tube table).")
    df = df[DESIGN_COLUMNS].copy()
    df["sample"] = df["sample"].astype(str)
    df["replicate"] = df["replicate"].astype(int)
    dups = df["sample"][df["sample"].duplicated()].unique().tolist()
    if dups:
        raise ValueError(f"design repeats sample name(s): {dups[:10]}")
    key = ["replicate"] + CONDITION_COLUMNS
    clash = df[df.duplicated(subset=key, keep=False)]
    if not clash.empty:
        raise ValueError("design has more than one tube with the same "
                         "replicate and growth condition: "
                         f"{clash['sample'].tolist()[:10]}")
    return df.reset_index(drop=True)


def design_conditions(design_df):
    """
    The distinct growth conditions of a design, in ``build_sample_dataframes``
    form (``replicate`` 1): the union over replicates, whose phenotypes the
    simulator computes once before each replicate takes its own tubes.
    """
    conds = (design_df[CONDITION_COLUMNS].drop_duplicates()
             .assign(replicate=1))
    sort_columns = ["replicate", "library", "condition_pre", "condition_sel",
                    "titrant_name", "titrant_conc", "t_sel"]
    cols = ["replicate", "library", "titrant_name", "condition_pre", "t_pre",
            "condition_sel", "titrant_conc", "t_sel"]
    return conds[cols].sort_values(sort_columns).reset_index(drop=True)
