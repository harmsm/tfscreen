import numpy as np
import pandas as pd

def read_dataframe(input,remove_extra_index=True):
    """
    Read a spreadsheet. Handles .csv, .tsv, .xlsx/.xls. If extension is
    not one of these, attempts to parse text as a spreadsheet using
    `pandas.read_csv(sep=None)`.

    Parameters
    ----------
    input : pandas.DataFrame or str
        either a pandas dataframe OR the filename to read in.
    remove_extra_index : bool, default=True
        look for the 'Unnamed: 0' column that pandas writes out for
        pandas.to_csv(index=True) and, if found, drop column.

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
    # Unnamed and has integer values [0,1,2,...,L]
    if remove_extra_index:
        if df.columns[0].startswith("Unnamed:"):
            possible_index = df.loc[:,df.columns[0]]
            if np.issubdtype(possible_index.dtypes,int):
                if np.array_equal(possible_index,np.arange(len(possible_index),dtype=int)):
                    df = df.drop(columns=[df.columns[0]])


    return df