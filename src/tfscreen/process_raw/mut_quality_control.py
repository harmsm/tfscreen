import pandas as pd
import numpy as np

from tfscreen.util import (
    read_yaml,
    read_dataframe,
    check_columns,
    set_categorical_genotype
)
from tfscreen.simulate import (
    generate_libraries
)

from pathlib import Path
from typing import Union, Optional, Iterable

def _get_expected_geno(config: Union[dict, str, Path]) -> np.ndarray:
    """
    Generate the set of expected genotypes from a library configuration.

    This is a wrapper function that uses `generate_libraries` to predict all
    possible genotypes given a library design specified in a configuration file
    or dictionary.

    Parameters
    ----------
    config : dict or str or Path
        A pre-loaded configuration dictionary or a path to a configuration
        YAML file specifying the library design (`aa_sequence`,
        `mutated_sites`, etc.). See the `generate_libraries` docstring for 
        details. 

    Returns
    -------
    np.ndarray
        A 1D NumPy array of unique, expected genotype strings.
    """

    # Load configuration (can either be a yaml or pre-loaded dictionary)
    config_dict = read_yaml(config)

    # Get simulation library dataframe (will have a column 'genotype' 
    # with the predicted genotypes possible given the design). 
    # Note, this function does a bunch of validation of inputs, so let it
    # do the validation. 
    sim_lib_df = generate_libraries(aa_sequence=config_dict["aa_sequence"],
                                    mutated_sites=config_dict["mutated_sites"],
                                    degen_codon=config_dict["degen_codon"],
                                    seq_starts_at=config_dict["seq_starts_at"],
                                    lib_keys=list(config_dict["library_mixture"].keys()))
                                    
    return pd.unique(sim_lib_df["genotype"])

def mut_quality_control(
    trans_df: Union[pd.DataFrame, str, Path],
    max_allowed_muts: int = 2,
    lib_config: Optional[Union[dict, str, Path]] = None,
    spiked_expected: Optional[Iterable[str]] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter a dataframe of translated mutations based on quality control criteria.

    This function provides two modes of quality control. The simple mode filters
    out genotypes with more mutations than `max_allowed_muts`. The advanced
    mode (enabled by providing `lib_config`) filters against a set of
    theoretically expected genotypes derived from the library design. The
    function tracks filtering statistics at each step.

    Parameters
    ----------
    trans_df : pandas.DataFrame or str or Path
        The input dataframe (or path to it) of translated mutations. Must
        contain "genotype" and "counts" columns. The genotype column should 
        have genotypes formatted like A47Q/P98D, with wildtype specified as 'wt'
    max_allowed_muts : int, default 2
        The maximum number of mutations a genotype can have to pass the
        initial filter. This value is ignored and recalculated from the
        expected set if `lib_config` is provided.
    lib_config : dict or str or Path, optional
        A library design configuration. If provided, this enables filtering
        against the set of genotypes predicted from the design. This follows
        the exact dictionary/yaml format used to simulate screens. (See the 
        docstring on `simulate.generate_libraries` for more details.)
    spiked_expected : Iterable[str], optional
        A list of extra genotypes to be considered "expected", such as known
        spiked-in controls not described in teh library configuration

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two dataframes:
        1. The filtered dataframe of quality-controlled mutations.
        2. A "funnel" dataframe tracking the number of reads and unique
           genotypes remaining at each filtering step.
    """

    # Load dataframe (reads from file or makes a copy)
    df = read_dataframe(trans_df)

    # make sure there are the correct columns present
    check_columns(df,required_columns=["genotype",
                                                     "counts"])

    # Load expected genotypes if a config is passed in
    if lib_config is not None:
        expected_genotypes = _get_expected_geno(lib_config)

        # Append spiked_expected
        if spiked_expected:
            expected_genotypes = list(expected_genotypes)
            expected_genotypes.extend(spiked_expected)

        # Build series of unique expected
        expected_genotypes = pd.Series(np.unique(expected_genotypes))

        # Set max_allowed_muts to match expected. (Let's us toss 
        # many mutants based on a simple count rather than a more expensive
        # .isin call). 
        max_allowed_muts = np.max(expected_genotypes.str.count("/")) + 1
    
    else:
        expected_genotypes = None

    # Record number of genotypes prior to filter
    funnel = {"step":["input"],
              "counts":[np.sum(df["counts"])],
              "unique_geno":[len(df)]}
    
    # Determine number of mutations for each genotype
    df["num_muts"] = np.where(df["genotype"] == "wt",0, 
                              df["genotype"].str.count("/") + 1)

    # Filter too many mutations and record number of counts
    df = df[df["num_muts"] <= max_allowed_muts]
    funnel["step"].append("few_enough")
    funnel["counts"].append(np.sum(df["counts"]))
    funnel["unique_geno"].append(len(df))

    # Make genotypes standardized, clean, sorted genotypes
    df = set_categorical_genotype(df,sort=True,standardize=True)

    # Filter on expected genotypes
    if expected_genotypes is not None:
        
        df = df[df["genotype"].isin(expected_genotypes)].reset_index(drop=True)
        
        funnel["step"].append("in_expected")
        funnel["counts"].append(np.sum(df["counts"]))
        funnel["unique_geno"].append(len(df))
    
    funnel = pd.DataFrame(funnel)

    return df, funnel

