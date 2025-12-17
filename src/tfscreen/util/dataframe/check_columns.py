


def check_columns(df,required_columns):
    """
    Check if a DataFrame contains all required columns.

    This function verifies that a given Pandas DataFrame contains all the
    columns specified in a list of required column names. If any required
    columns are missing, it raises a ValueError with a descriptive error
    message.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.
    required_columns : list of str
        A list of column names that are required to be present in the DataFrame.

    Raises
    ------
    ValueError
        If any of the required columns are not found in the DataFrame, a
        ValueError is raised with a message listing the missing columns.
    """

    required_set = set(required_columns)
    seen_set = set(df.columns)
    if not required_set.issubset(seen_set):
        missing = required_set - seen_set
        err = "Not all required columns seen. Missing columns:\n"
        for c in missing:
            err += f"    {c}\n"
        err += "\n"
        raise ValueError(err)