import pandas as pd
import tfscreen

def expand_on_conditions(df, conditions):
    """
    Expand a DataFrame to include all combinations of genotypes and conditions.

    Ensures that every unique genotype has a row for every unique combination
    of condition values. Rows that did not exist in the original DataFrame are
    created and their value columns are filled with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Must contain a "genotype" column and any columns
        listed in `conditions`.
    conditions : list of str or None
        A list of column names that define the experimental conditions. If None
        or empty, the original DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with a complete grid of genotype-condition combinations.
    """
    
    # Read dataframe
    # df = tfscreen.util.read_dataframe(df)
    if conditions is None or len(conditions) == 0:
        return df

    # Build array of genotypes in canonically sorted order
    genotypes = pd.unique(df["genotype"])
    idx = tfscreen.util.argsort_genotypes(genotypes)
    genotype_order = genotypes[idx]

    # Build index columns (genotype + conditions)
    if isinstance(conditions, str):
        index_cols = [conditions]
    else:
        index_cols = conditions[:]
    index_cols.insert(0, "genotype")

    # Use a categorical to enforce the canonical genotype sort order
    df["genotype"] = pd.Categorical(df["genotype"],
                                    categories=genotype_order,
                                    ordered=True)

    # Get unique levels for all index columns
    unique_levels = [df[col].unique() for col in index_cols]

    # Create the complete, 'Cartesian product' index of all combinations.
    complete_index = pd.MultiIndex.from_product(unique_levels, names=index_cols)

    # Prepare the original DataFrame and reindex it.
    df_complete = (df.set_index(index_cols)    # Set index for alignment
                     .reindex(complete_index)  # Align with the complete index
                     .sort_index()             # Ensure canonical order
                     .reset_index())           # Convert index back to columns

    return df_complete

