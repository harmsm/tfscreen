import numpy as np
import pandas as pd

def read_dataframe(input,
                   remove_extra_index=True,
                   index_column=None):
    """
    Read a spreadsheet. Handles .csv, .tsv, .xlsx/.xls. If extension is
    not one of these, attempts to parse text as a spreadsheet using
    `pandas.read_csv(sep=None)`.

    Parameters
    ----------
    input : pandas.DataFrame or str
        either a pandas dataframe OR the filename to read in.
    remove_extra_index : bool, default=True
        look for the 'Unnamed: #' columns that pandas writes out and drop them 
        if they are present and have integer indexes. Non-integer columns that
        match this pattern are left intact. 
    index_column : str, optional
        use this column as the index. if this is not in the dataframe but there
        is a "Unnamed: 0" column, assume that column is this column and assign 
        it as such. 

    Returns
    -------
    pandas.DataFrame
        read in dataframe
    """

    # If this is a string, try to load it as a file
    if type(input) is str:

        filename = input

        ext = filename.split(".")[-1].strip().lower()

        if ext in ["xlsx","xls"]:
            df = pd.read_excel(filename)
        elif ext == "csv":
            df = pd.read_csv(filename,sep=",")
        elif ext == "tsv":
            df = pd.read_csv(filename,sep="\t")
        else:
            # Fall back -- try to guess delimiter
            df = pd.read_csv(filename,sep=None,engine="python")

    # If this is a pandas dataframe, work in a copy of it.
    elif type(input) is pd.DataFrame:
        df = input.copy()

    # Otherwise, fail
    else:
        err = f"\n\n'input' {input} not recognized. Should be the filename of\n"
        err += "spreadsheet or a pandas dataframe.\n"
        raise ValueError(err)

    # Look for extra index column that pandas writes out (in case user wrote out
    # pandas frame manually, then re-read). Looks for first column that is
    # Unnamed and has integer values [0,1,2,...,L]. This gets dropped. 
    if remove_extra_index:
        if df.columns[0].startswith("Unnamed:"):
            possible_index = df.loc[:,df.columns[0]]
            if np.issubdtype(possible_index.dtypes,int):
                if np.array_equal(possible_index,np.arange(len(possible_index),dtype=int)):
                    df = df.drop(columns=[df.columns[0]])

    # If an index column is requested
    if index_column is not None:

        # If this is not already the index...
        if df.index.name != index_column:

            # If the column is not in columns
            if index_column not in df.columns:

                # Look for "Unnamed: 0". If this is here, assume this is the
                # requested index and rename it. 
                if df.columns[0] == "Unnamed: 0":
                    print(f"Renaming column 'Unnamed: 0' -> '{index_column}'")
                    df = df.rename(columns={"Unnamed: 0":index_column})

                # If we get here, we can't even try to guess the column
                else:
                    err = f"df does not have a column or index named '{index_column}'\n"
                    raise ValueError(err)   

            # assign the index and drop the original column
            df.index = df[index_column]
            df = df.drop(columns=[index_column])

    return df