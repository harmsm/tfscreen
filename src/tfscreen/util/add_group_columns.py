import pandas as pd

def add_group_columns(target_df,
                      group_cols,
                      group_name,
                      existing_df=None):
    """
    Add two new columns: `{group_name}` and `{group_name}_tuple` to target_df. 
    
    These hold unique entries from a groupby(group_calls) call, with `group_name` 
    holding the integer indexes and `{group_name}_tuple` holding the values of
    `{group_cols}` in each row as tuples. If `existing_df` is passed in, the 
    group indexes are merged from that dataframes and *only* groups matching 
    what is in that group are maintained in the output dataframe.

    Parameters
    ----------
    target_df : pd.DataFrame
        dataframe to update
    group_cols : list
        list of column names to use for the grouping
    group_name : str
        base name of columns to create
    existing_df : pd.DataFrame
        pandas dataframe with existing groupings. Must have a `group_name` 
        column and all columns in `group_cols`

    Returns
    -------
    pd.DataFrame
        copy of target_df modified with the relevant columns

    Raises
    ------
    ValueError
        if existing_df does not have all of the needed columns
    """

    # Works on a copy
    target_df = target_df.copy()

    # If we have an existing dataframe with the group defined, merge from 
    # that. 
    if existing_df is not None:

        # Deal with columns for merge
        merge_on = list(group_cols) 
        merge_on.append(group_name)
        need_cols = set(merge_on)
        seen_cols = set(existing_df.columns)
        if not need_cols.issubset(seen_cols):
            raise ValueError (
                f"existing_df does not have all required columns. Missing "
                f"columns are: {need_cols - seen_cols}"
            )

        # Do merge
        target_df = target_df.merge(existing_df[merge_on],
                                    how="left",
                                    on=group_cols,
                                    sort=False)

        # This drops groups that were not in the existing dataframe,
        # enforcing its role as the source of truth on the group definition. 
        target_df = target_df[~pd.isna(target_df[group_name])].reset_index(drop=True)
        
        # Cast back to int to ensure type consistency
        target_df[group_name] = target_df[group_name].astype(int)
        
    else:

        # Get all unique combinations of the group columns, then sort and 
        # extract indexes. This creates a map with C-order/row-major sorting.
        unique_groups = target_df[group_cols].drop_duplicates().copy()
        sorted_groups = unique_groups.sort_values(by=group_cols).reset_index(drop=True)
        sorted_groups[group_name] = sorted_groups.index

        # Merge the map back onto the target dataframe.
        target_df = target_df.merge(sorted_groups, 
                                    on=group_cols, 
                                    how="left",
                                    sort=False)

    # Record a tuple version of this grouping.
    # Use .values and map(tuple,...) for a fast, hashable conversion
    tuple_values = list(map(tuple, target_df[group_cols].values))
    target_df[f"{group_name}_tuple"] = pd.Categorical(tuple_values)

    return target_df