
import numpy as np
import pandas as pd

import re

def parse_patsy(df,
                patsy_param_names,
                model_terms,
                factor_terms):
    """
    Parse Patsy parameter names to extract factor information.

    This function parses Patsy-generated parameter names to extract information
    about the underlying factors and their values. It identifies categorical
    variables encoded in the Patsy formula and extracts the corresponding
    factor columns and values from the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame used in the Patsy model.  This is used to coerce
        the factor values to the correct datatype.
    patsy_param_names : list of str
        A list of Patsy-generated parameter names (e.g., from `design_info.column_names`).
    model_terms : dict
        A dictionary mapping pretty parameter names to unique Patsy gobbledygook.
    factor_terms : dict
        A dictionary mapping parameter classes to factor terms.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing parsed parameter information, with columns:
        - 'param_idx': Integer index of the parameter.
        - 'patsy_name': The original Patsy parameter name.
        - 'param_class': The class of the parameter.
        - [factor_column]: Additional columns for each factor column,
          containing the factor value for each parameter (None if not applicable).

    Raises
    ------
    RuntimeError
        If there is a problem parsing the Patsy parameter names or extracting
        factors, or if there is a problem coercing the factor values.
    """

    # Dictionary keys unique patsy gobblygook to pretty parameter names
    patsy_to_pretty = dict([(value,key)
                            for key, value in model_terms.items()])

    # Some regex we use in parsing
    bracket_search = re.compile("\\[.*?\\]")
    quote_search = re.compile("[\'\"]")

    # Lists will store output for each parameter as we generate it
    raw_param_names = []
    param_classes = []
    factor_columns = []
    factor_values = []
    columns_seen = []
    
    for param in patsy_param_names:

        # If we hit a parameter with [...], it's a real categorical variable
        hit = bracket_search.search(param)
        if hit:

            # Pop out match: red[stuff]blue -> redblue
            key = "".join(bracket_search.split(param))
            
            # Get the class of this parameter
            param_class = patsy_to_pretty[key]
    
            # Get the factors associated with this parameter
            param_factors = factor_terms[param_class]
    
            # Get everything inside [ ... ]: [stuff] -> stuff
            specifier = hit.group()[1:-1].strip()

            # Get rid of T. level indicator
            if specifier.startswith("T."):
                specifier = specifier[2:]

            if isinstance(param_factors,str):
                param_factors = [param_factors]
                specifier = [specifier]
    
            # If this looks like a tuple, split it into individual values, stripping
            # commas and spaces.
            if specifier[0] == "(" and specifier[-1] == ")":
                specifier = [s.strip() for s in specifier[1:-1].split(",")]
                specifier = [quote_search.sub("",s) for s in specifier]
    
            # If this is true, we screwed up parsing somewhere. 
            if len(specifier) != len(param_factors):
                err = "There was problem parsing the patsy paramter names and\n"
                err += "extracting factors.\n"
                raise RuntimeError(err)
    
            # Go through parameter factors
            for i, col in enumerate(param_factors):
    
                # If the column is in the input dataframe, coerce the factor to 
                # the value of the data type. (There won't always be a column
                # from the dataframe, but I think we're safe assuming that, if 
                # the factor shares a name with the column, it came from that
                # column. This means the final dataframe will have the right 
                # types for lookups in the original dataframe. 
                if col in df:
                    
                    try:
                        specifier[i] = df[col].dtype.type(specifier[i])
                    except Exception as e:
                        err = "There was a problem coercing the factor values\n"
                        err += "extracted from the patsy parameter names.\n"
                        raise RuntimeError from e

            # Record information
            raw_param_names.append(param.strip())
            param_classes.append(param_class)
            factor_columns.append(param_factors)
            factor_values.append(specifier)
            columns_seen.extend(param_factors)

    # Generate a dataframe that lets us look up position parameters in params 
    # array (param_idx), the full patsy name of the parameter, and the
    # parameter class. 
    out_dict = {"param_idx":np.arange(len(raw_param_names),dtype=int),
                "patsy_name":raw_param_names,
                "param_class":param_classes}
    
    # Create a unique list of all factor columns seen and append to the dataframe
    columns_seen = list(set(columns_seen))
    columns_seen.sort()
    for c in columns_seen: 
        out_dict[c] = [None for _ in range(len(param_classes))]

    # Populate the factor columns. After this operation, the 
    # param_class + factor_columns will uniquely define each parameter in a 
    # factorized pandas way
    for i in range(len(factor_columns)):
        col = factor_columns[i]
        for j in range(len(col)):
            out_dict[col[j]][i] = factor_values[i][j]
    
    return pd.DataFrame(out_dict)
