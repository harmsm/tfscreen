
from __future__ import annotations

from tfscreen.simulate import load_simulation_config
from tfscreen.util import (
    read_dataframe,
    vstack_padded,
    zero_truncated_poisson,
)

import numpy as np
import numpy.ma as ma
from numpy.random import Generator

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from scipy.stats import (
    gmean,
    lognorm,
)
from tqdm.auto import tqdm

# Mostly for type hinting
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
)

# Type Aliases introduced during review
_Numeric = TypeVar("_Numeric", int, float)

MULTI_PLASMID_COMBINE_FCNS = {"gmean":gmean,
                              "mean":ma.mean,
                              "min":ma.min,
                              "max":ma.max,
                              "sum":ma.sum}
  
def _check_dict_number(
    key: str,
    input_dict: dict[str, Any],
    cast_type: Callable[[Any], _Numeric] = float,
    min_allowed: Optional[_Numeric] = None,
    max_allowed: Optional[_Numeric] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    allow_none: bool = False,
) -> dict[str, Any]:
    """
    Validate and cast a numerical value within a dictionary.

    Parameters
    ----------
    key : str
        The dictionary key to check.
    input_dict : dict
        The dictionary to validate.
    cast_type : Callable, default: float
        A function to cast the value to (e.g., `int`, `float`).
    min_allowed : int or float, optional
        The minimum allowed value. If None, no minimum is enforced.
    max_allowed : int or float, optional
        The maximum allowed value. If None, no maximum is enforced.
    inclusive_min : bool, default: True
        Whether the minimum bound is inclusive (`<=`).
    inclusive_max : bool, default: True
        Whether the maximum bound is inclusive (`>=`).
    allow_none : bool, default: False
        If True, a missing key or a value of None is permissible. If the
        key is missing, it will be added to the dict with a value of None.

    Returns
    -------
    dict
        The validated and potentially modified input dictionary.

    Raises
    ------
    ValueError
        If the key is missing (and not `allow_none`), or if the value
        is not a scalar, fails to cast, or falls outside the allowed range.
    
    Notes
    -----
    This function modifies `input_dict` in-place.
    """

    if allow_none:
        if key not in input_dict or input_dict[key] is None:
            input_dict[key] = None
            return input_dict

    if key not in input_dict:
        raise ValueError(f"Required key '{key}' not found in input dictionary.")

    v = input_dict[key]
    try:
        if not np.isscalar(v):
            raise TypeError("Value must be a scalar.")

        # Cast to the desired type and update the dictionary
        v_cast = cast_type(v)
        input_dict[key] = v_cast

        # Perform range checks on the casted value
        if min_allowed is not None:
            if inclusive_min and v_cast < min_allowed:
                raise ValueError(f"Value must be >= {min_allowed}.")
            if not inclusive_min and v_cast <= min_allowed:
                raise ValueError(f"Value must be > {min_allowed}.")
        if max_allowed is not None:
            if inclusive_max and v_cast > max_allowed:
                raise ValueError(f"Value must be <= {max_allowed}.")
            if not inclusive_max and v_cast >= max_allowed:
                raise ValueError(f"Value must be < {max_allowed}.")

    except (ValueError, TypeError) as e:
        err = f"Could not process key '{key}' with value '{v}'.\n"
        err += f"Reason: {e}"
        raise ValueError(err) from e

    return input_dict


def _check_cf(
    cf: dict[str, Any] | str | Path
) -> dict[str, Any]:
    """Validate and set defaults for the simulation configuration.

    Parameters
    ----------
    cf : dict or str or pathlib.Path
        The configuration, which can be a dictionary or a path to a YAML file
        that will be loaded.

    Returns
    -------
    dict
        A validated configuration dictionary with default values set for
        any optional keys that were not provided.

    Raises
    ------
    ValueError
        If any required keys are missing or if any values are invalid
        (e.g., wrong type, out of range).
    
    Notes
    -----
    This function first calls `tfscreen.simulate.load_simulation_config` to
    ensure the configuration is loaded into a dictionary. It then modifies
    this dictionary in-place by validating values and setting defaults.
    """

    # Load from YAML if a path is provided, otherwise assume it's a dict
    cf = load_simulation_config(cf)

    if "final_cfu_pct_err" not in cf:
        cf["final_cfu_pct_err"] = 0.05

    # --- Validate single numerical values ---
    cf = _check_dict_number("prob_index_hop", cf, min_allowed=0, max_allowed=1, allow_none=True)
    cf = _check_dict_number("lib_assembly_skew_sigma", cf, min_allowed=0, allow_none=True)
    cf = _check_dict_number("transformation_poisson_lambda", cf, min_allowed=0, allow_none=True)
    cf = _check_dict_number("growth_rate_noise", cf, min_allowed=0, allow_none=True)
    cf = _check_dict_number("random_seed", cf, cast_type=int, min_allowed=0, allow_none=True)
    cf = _check_dict_number("cfu0", cf, allow_none=False,min_allowed=0)
    cf = _check_dict_number("total_num_reads", cf, cast_type=int, min_allowed=0, inclusive_min=False)
    cf = _check_dict_number("final_cfu_pct_err",cf,min_allowed=0,inclusive_min=False,allow_none=False)

    # --- Validate nested dictionaries ---
    for key in ["transform_sizes", "library_mixture"]:
        try:
            if key not in cf or not isinstance(cf[key], dict):
                raise TypeError("must be a dictionary.")

            # Validate all numerical values within the nested dictionary
            for sub_key in cf[key]:
                cast_type = int if key == "transform_sizes" else float
                cf[key] = _check_dict_number(sub_key, cf[key], cast_type=cast_type, min_allowed=0)

        except (TypeError, ValueError) as e:
            err = f"Configuration key '{key}' is invalid: {e}\n"
            err += "It should be a dictionary mapping library_origin names to positive numbers."
            raise ValueError(err) from e

    # --- Validate and set defaults for selectors ---
    # By default, we specific growth conditions are defined by these columns
    if "condition_selector" not in cf:
        cf["condition_selector"] = [
            "titrant_name", "titrant_conc",
            "condition_pre","t_pre",
            "condition_sel","t_sel"
        ]

    # By default, a unique library in phenotype_df is defined by these columns
    if "library_selector" not in cf:
        cf["library_selector"] = ["replicate", "library"]

    if not isinstance(cf["condition_selector"], list):
        raise ValueError("condition_selector must be a list of column names.")
    if not isinstance(cf["library_selector"], list):
        raise ValueError("library_selector must be a list of column names.")

    return cf

def _check_lib_spec(
    cf: dict[str, Any],
    library_df: pd.DataFrame,
    phenotype_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Validate consistency between config and library/phenotype dataframes.

    This function performs several checks:
    1. All genotypes in `library_df` are present in `phenotype_df`.
    2. `library_df` has the required columns.
    3. Keys in `cf["library_mixture"]` and `cf["transform_sizes"]` match
       and are present in `library_df["library_origin"]`.
    4. Sets defaults for and validates `cf["multi_plasmid_combine_fcn"]`.

    Parameters
    ----------
    cf : dict
        The main configuration dictionary.
    library_df : pandas.DataFrame
        Dataframe defining the composition of the input libraries.
    phenotype_df : pandas.DataFrame
        Dataframe defining the phenotype (e.g., growth rate) of each genotype.

    Returns
    -------
    dict
        The configuration dictionary, potentially updated with default values
        for `multi_plasmid_combine_fcn`.

    Raises
    ------
    ValueError
        If any of the consistency checks fail.

    Notes
    -----
    This function modifies the `cf` dictionary in-place.
    """

    # Make sure all genotypes in library_df are in genotype_df
    lib_genos = set(library_df["genotype"])
    pheno_genos = set(phenotype_df["genotype"])
    if not lib_genos.issubset(pheno_genos):
        missing = lib_genos - pheno_genos
        raise ValueError(f"Genotypes in library_df are missing from phenotype_df: {missing}")

    # Make sure library_df has required columns
    required_cols = {"library_origin", "weight"}
    if not required_cols.issubset(library_df.columns):
        missing = required_cols - set(library_df.columns)
        raise ValueError(f"library_df is missing required columns: {missing}")
        
    # Get library_origin and keys from transform_sizes/library_mixtures
    p_df_set = set(pd.unique(library_df["library_origin"]))
    t_size_set = set(cf["transform_sizes"].keys())
    l_mix_set = set(cf["library_mixture"].keys())
    
    # Check for matches between columns
    if t_size_set != l_mix_set:
        err = "The keys in 'library_mixture' and 'transform_sizes' must be identical.\n"
        raise ValueError(err)

    if not t_size_set.issubset(p_df_set):
        missing = t_size_set - p_df_set
        err = "The keys in 'library_mixture' and 'transform_sizes' must be a subset\n"
        err += f"of the values in library_df['library_origin']. Missing: {missing}\n"
        raise ValueError(err)

    if sum(cf["library_mixture"].values()) == 0:
        raise ValueError("The sum of ratios in 'library_mixture' cannot be zero.")

    # Deal with multi_plasmid_combine_fcn.

    libraries = np.unique(phenotype_df["library"])
    if "multi_plasmid_combine_fcn" not in cf:
        cf["multi_plasmid_combine_fcn"] = {}
        for lib in libraries:
            cf["multi_plasmid_combine_fcn"][lib] = "mean"
    
    for lib in libraries:

        if lib not in cf["multi_plasmid_combine_fcn"]:
            err = f"lib '{lib}' not found in 'multi_plasmid_combine_fcn'\n"
            raise ValueError(err)
        fcn = cf["multi_plasmid_combine_fcn"][lib]

        if fcn not in MULTI_PLASMID_COMBINE_FCNS:
            err = f"The multi_plasmid_combine_fcn '{fcn}' for library '{lib}'\n"
            err += "was not recognized. The function should be one of:\n"
            for k in MULTI_PLASMID_COMBINE_FCNS:
                err += f"    {k}\n"
            raise ValueError(err)

    return cf


def _sim_plasmid_probabilities(
    frequencies: np.ndarray,
    skew_sigma: float | None = None,
    rng: Generator | None = None,
) -> np.ndarray:
    """Simulate plasmid probabilities in a library, applying optional skew.

    This function converts a vector of plasmid frequencies into a probability
    distribution. If `skew_sigma` is provided, it simulates library assembly
    skew by multiplying the probabilities by random variates from a
    log-normal distribution.

    Parameters
    ----------
    frequencies : numpy.ndarray
        A 1D array of expected plasmid frequencies. Must be non-negative
        and sum to a positive value.
    skew_sigma : float, optional
        The shape parameter (`s`) for the `scipy.stats.lognorm` distribution
        used to generate skew. If None or 0, no skew is applied.
    rng : numpy.random.Generator, optional
        An initialized NumPy random number generator. If None, a new default
        generator is created.

    Returns
    -------
    numpy.ndarray
        A 1D array of plasmid probabilities of the same shape as `frequencies`,
        guaranteed to sum to 1.0.

    Raises
    ------
    ValueError
        If `frequencies` contains non-finite or negative values, or if the
        sum of its elements is not positive.
    
    Notes
    -----
    The log-normal distribution is a suitable model for multiplicative noise
    or skew, which is common in biological processes like PCR amplification
    or library assembly, where errors compound multiplicatively.
    """

    # Initialize random number generator
    if rng is None:
        rng = np.random.default_rng()

    if np.any(np.isnan(frequencies)) or np.any(frequencies < 0) or np.sum(frequencies) == 0:
        err = "all entries in frequencies array must be finite and have values\n"
        err += ">= 0. Further, the sum of the total array must be > 0.\n"
        raise ValueError(err)
    
    # plasmid prob is normalized freq
    probabilities = frequencies/np.sum(frequencies)

    # if we are adding skew...
    if (skew_sigma is not None) and (skew_sigma > 0):

        # Sample from a lognormal distribution
        dist = lognorm(s=skew_sigma,scale=1,loc=0)
        raw_counts = dist.rvs(size=len(frequencies),
                              random_state=rng)

        # Skew probabilities and re-normalize
        skewed_prob = probabilities*raw_counts
        probabilities = skewed_prob/np.sum(skewed_prob)

    return probabilities

def _sim_index_hop(
    counts: np.ndarray,
    index_hop_prob: float | None,
    rng: Generator | None = None,
) -> np.ndarray:
    """Simulate sequencing index hopping.

    This function models index hopping by randomly re-distributing a
    specified fraction of the total reads. Reads are chosen for hopping
    proportional to their abundance, and their new identity is chosen
    uniformly at random from all possibilities.

    Parameters
    ----------
    counts : numpy.ndarray
        A 1D integer array of read counts for each genotype/category.
    index_hop_prob : float, optional
        The fraction of total reads that will be mis-assigned (hop).
        If None or 0, a copy of the original counts is returned.
    rng : numpy.random.Generator, optional
        An initialized NumPy random number generator. If None, a new default
        generator is created.

    Returns
    -------
    numpy.ndarray
        A new array of counts with index hopping applied. The total sum of
        counts remains identical to the input.

    Notes
    -----
    The simulation correctly samples individual reads for hopping without
    replacement. This is achieved by creating a temporary array representing
    every single read via `np.repeat`, ensuring that a category cannot
    lose more reads than it possesses.
    """

    if index_hop_prob is None:
        return counts.copy()
    
    # Initialize random number generator
    if rng is None:
        rng = np.random.default_rng()

    # Figure out how many reads will hop
    num_to_hop = int(np.round(index_hop_prob*np.sum(counts),0))

    # no hopping 
    if num_to_hop == 0:
        return counts.copy()

    # index of counts
    idx = np.arange(counts.shape[0],dtype=int)

    # Figure out where we are going to hop from. The repeat + sampling without
    # replacement guarantees we end up with real reads. (If we sampled with
    # replacement from the counts, we could concievably end up subtracting more
    # reads than we had from a given sample.)
    all_reads = np.repeat(np.arange(counts.size), repeats=counts)
    hop_from = rng.choice(all_reads, size=num_to_hop, replace=False)
    
    # choose reads to gain
    hop_to = rng.choice(idx,size=num_to_hop,replace=True)
    
    # Figure out how much to add and subtract to each read
    hop_from_values = np.bincount(hop_from,minlength=len(idx))
    hop_to_values = np.bincount(hop_to,minlength=len(idx))

    # move reads from sub --> add
    counts = counts - hop_from_values + hop_to_values

    return counts


def _sim_transform(
    genotype_probs: np.ndarray,
    num_transformants: int,
    transformation_poisson_lambda: float | None = None,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the transformation of a plasmid library into cells.

    This function models the transformation process, including the possibility
    of a single cell receiving multiple plasmids.

    Parameters
    ----------
    genotype_probs : numpy.ndarray
        A 1D array of plasmid probabilities (must sum to 1.0) used for
        weighted sampling of plasmids.
    num_transformants : int
        The total number of cells (transformants) to simulate.
    transformation_poisson_lambda : float, optional
        The lambda parameter for a zero-truncated Poisson distribution used
        to determine the number of plasmids per cell. If None, each cell
        receives exactly one plasmid.
    rng : numpy.random.Generator, optional
        An initialized NumPy random number generator. If None, a new one
        is created.

    Returns
    -------
    transformants : numpy.ndarray
        A 2D integer array of shape `(num_transformants, max_plasmids)`,
        where each value is a genotype index.
    plasmid_mask : numpy.ndarray
        A 2D boolean array with the same shape as `transformants`. A `True`
        value indicates a masked (invalid) plasmid entry.

    Notes
    -----
    The `transformants` and `plasmid_mask` arrays together represent a
    "ragged array," where each row can have a different number of valid
    entries. This is an efficient way to store the results of a simulation
    where cells can contain a variable number of plasmids.
    """
    
    # No transformants -- return empty outputs
    if num_transformants == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=bool)

    # Initialize random number generator
    if rng is None:
        rng = np.random.default_rng()

    # Get index for plasmids
    num_plasmids = len(genotype_probs)
    plasmids = np.arange(num_plasmids,dtype=int)
    
    # Assign the number of plasmids each transformant will recieve
    if transformation_poisson_lambda is None:
        num_plas_in_cell = np.ones(num_transformants,dtype=int)
    else:
        num_plas_in_cell = zero_truncated_poisson(num_transformants,
                                                  transformation_poisson_lambda,
                                                  rng) 
    max_num_plasmids = np.max(num_plas_in_cell)

    # Transformants all have max_num_plasmids
    transformants = rng.choice(
        plasmids,
        size=(num_transformants,max_num_plasmids),
        p=genotype_probs
    )

    # Mask plasmid indexes above the number of plasmids
    col_indexes = np.arange(0,max_num_plasmids)
    plasmid_mask = col_indexes >= num_plas_in_cell[:,np.newaxis]

    return transformants, plasmid_mask

def _sim_transform_and_mix(
    lib_origin_grouper: DataFrameGroupBy,
    transform_sizes: dict[str, int],
    library_mixture: dict[str, float],
    transformation_poisson_lambda: float | None,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate transformation for multiple source libraries and mix them.

    This function iterates over a grouper of library origins. For each origin,
    it simulates the transformation process and then combines the resulting
    cell populations into a single, mixed pool according to specified mixture
    ratios.

    Parameters
    ----------
    lib_origin_grouper : pandas.DataFrameGroupBy
        A pandas DataFrame grouped by library origin. Each group should contain
        the `probs` for the genotypes in that library.
    transform_sizes : dict
        A dictionary mapping library origin names (str) to the number of
        transformants to simulate (int).
    library_mixture : dict
        A dictionary mapping library origin names (str) to their relative
        weight (float) in the final mixture.
    transformation_poisson_lambda : float, optional
        The lambda parameter for the multi-plasmid transformation simulation,
        passed to `_sim_transform`.
    rng : numpy.random.Generator
        An initialized NumPy random number generator.

    Returns
    -------
    transformants : numpy.ndarray
        A 2D integer array representing the mixed population of transformants.
    trans_mask : numpy.ndarray
        The corresponding boolean mask for the `transformants` array.
    probs : numpy.ndarray
        A 1D float array representing the frequency of each individual cell
        (row) in the final mixed population. Sums to 1.0.
    """

    # Go through all library_origin and transform. 
    all_trans = []
    all_trans_mask = []
    all_weights = []
    for origin_group, origin_sub_df in lib_origin_grouper:

        if isinstance(origin_group,str):
            lib_key = origin_group
        else:
            lib_key = origin_group[0]

        # Together trans and trans_mask define a ragged array of cells with
        # (possibly) multiple plasmids
        trans, tr_mask = _sim_transform(origin_sub_df["probs"].to_numpy(),
                                        transform_sizes[lib_key],
                                        transformation_poisson_lambda,
                                        rng)
        all_trans.append(trans)
        all_trans_mask.append(tr_mask)

        w = library_mixture[lib_key]
        all_weights.append(np.full(trans.shape[0],w,dtype=float))

    # This is vstack that accounts for the fact that these transformations 
    # might have different numbers of plasmids
    transformants = vstack_padded(all_trans,fill_value=0)
    trans_mask = vstack_padded(all_trans_mask,fill_value=True)
    
    # Build probability vector based on mixtures of the libraries. 
    weights = np.concat(all_weights)
    probs = weights/np.sum(weights)

    return transformants, trans_mask, probs


def _sim_growth(
    transformants: np.ndarray,
    trans_mask: np.ndarray,
    trans_freq: np.ndarray,
    genotype_vs_kt: np.ndarray,
    total_cfu0: float,
    multi_plasmid_combine_fcn: str,
) -> np.ndarray:
    """Simulate cell growth for a population under multiple conditions.

    This function calculates the final cell abundance (CFU) for each initial
    transformant across all experimental conditions. It accounts for
    multi-plasmid transformants by combining their `kt` values using a
    specified function.

    Parameters
    ----------
    transformants : numpy.ndarray
        A 2D integer array of shape `(num_cells, max_plasmids)` holding
        genotype indices for each plasmid in each cell.
    trans_mask : numpy.ndarray
        A 2D boolean array of shape `(num_cells, max_plasmids)` masking
        invalid plasmid entries.
    trans_freq : numpy.ndarray
        A 1D float array of shape `(num_cells,)` with the initial frequency
        of each cell in the population.
    genotype_vs_kt : numpy.ndarray
        A 2D float array of shape `(num_genotypes, num_conditions)` holding
        the growth rate multiplied by time (`k*t`) for each genotype.
    total_cfu0 : float
        The total initial colony-forming units (CFU) for the entire
        population.
    multi_plasmid_combine_fcn : str
        The name of the function used to combine the `kt` values from
        multiple plasmids within a single cell. Must be a key in
        `MULTI_PLASMID_COMBINE_FCNS`.

    Returns
    -------
    numpy.ndarray
        A 2D float array of shape `(num_cells, num_conditions)` holding the
        final CFU for each cell after growth in each condition.
    """
    
    if multi_plasmid_combine_fcn not in MULTI_PLASMID_COMBINE_FCNS:
        err = f"multi_plasmid_combine_fcn '{multi_plasmid_combine_fcn}' not recognized. Should be one\n"
        err += f"of: '{list(MULTI_PLASMID_COMBINE_FCNS.keys())}\n"
        raise ValueError(err)

    # Look up kt for every plasmid slot in every cell for every condition.
    # -> all_kt has shape (num_cells, max_plasmids, num_conditions)
    all_kt = genotype_vs_kt[transformants]

    # Expand the 2D mask to 3D to match the shape of all_kt.
    expanded_mask = np.broadcast_to(trans_mask[:, :, np.newaxis], all_kt.shape)
    masked_kt = ma.array(all_kt, mask=expanded_mask)

    # Calculate the kt for each cell by combining effects of plasmids
    # -> trans_kt has shape (num_cells, num_conditions)
    trans_kt = MULTI_PLASMID_COMBINE_FCNS[multi_plasmid_combine_fcn](masked_kt,axis=1)

    # -> trans_cfu0 has shape (num_cells,)
    trans_cfu0 = total_cfu0*trans_freq

    # -> trans_cfu has shape (num_cells, num_conditions)
    trans_cfu = trans_cfu0[:,np.newaxis]*np.exp(trans_kt)

    return trans_cfu


def _sim_sequencing(
    transformants: np.ndarray,
    trans_mask: np.ndarray,
    trans_cfu: np.ndarray,
    num_genotypes: int,
    reads_per_sample: int,
    rng: Generator,
) -> np.ndarray:
    """Simulate the sequencing of plasmids from grown cell populations.

    This function models deep sequencing. For each condition, it assumes that
    individual plasmids are sampled for sequencing with a probability
    proportional to the final abundance (CFU) of their host cell.

    Parameters
    ----------
    transformants : numpy.ndarray
        A 2D integer array `(num_cells, max_plasmids)` of genotype indices.
    trans_mask : numpy.ndarray
        A 2D boolean mask for `transformants`.
    trans_cfu : numpy.ndarray
        A 2D float array `(num_cells, num_conditions)` holding the final
        CFU of each cell in each condition.
    num_genotypes : int
        The total number of unique genotypes in the library.
    reads_per_sample : int
        The number of sequencing reads to simulate for each condition/sample.
    rng : numpy.random.Generator
        An initialized NumPy random number generator.

    Returns
    -------
    numpy.ndarray
        A 2D integer array `(num_genotypes, num_conditions)` containing the
        simulated read counts for each genotype in each condition.
    """

    genotype_idx = np.arange(num_genotypes,dtype=int)
    num_conditions = trans_cfu.shape[1]

    seq_output = np.zeros((num_genotypes, num_conditions), dtype=int)

    # Flatten the ragged array of transformants to get a 1D list of all
    # successfully transformed plasmid indices. This is shared across all 
    # conditions. 
    all_transformed_plas = ma.array(transformants, mask=trans_mask).compressed()

    # Loop over all conditions
    for i in range(num_conditions):

        # Get the frequencies of cells at this condition and then broadcast to 
        # match the shape/alignment of all_transformed_plas
        trans_freq = trans_cfu[:,i]
        trans_freq_2d = np.broadcast_to(trans_freq[:, np.newaxis],
                                        transformants.shape)
        all_transformed_freq = ma.array(trans_freq_2d, mask=trans_mask).compressed()

        # Count the number of times each plasmid index appears, weighted by the 
        # frequency of the cells in the population
        geno_counts = np.bincount(all_transformed_plas,
                                  weights=all_transformed_freq,
                                  minlength=num_genotypes)
        
        # Do the sequencing (only if there are more than 0 counts)
        if np.sum(geno_counts) > 0:

            # Normalize to get relative probability of each drawing each genotype
            geno_probs = geno_counts/np.sum(geno_counts)
        
            # Sample genotypes based on their probabilities.
            sampled_plasmids = rng.choice(
                genotype_idx,
                size=reads_per_sample,
                p=geno_probs,
                replace=True
            )

            # Count the occurrences of each genotype in the sample
            seq_output[:, i] = np.bincount(sampled_plasmids, minlength=num_genotypes)
        
    return seq_output


def _calc_genotype_cfu0(
    transformants: np.ndarray,
    trans_mask: np.ndarray,
    trans_freq: np.ndarray,
    total_cfu0: float,
    num_genotypes: int,
) -> np.ndarray:
    """Calculate the initial CFU for each genotype after transformation.

    This function determines the starting abundance (cfu0) of each distinct
    genotype based on the stochastic outcome of the transformation simulation.
    It aggregates the frequencies of all cells containing a given plasmid
    and partitions the total library cfu0 accordingly.

    Parameters
    ----------
    transformants : numpy.ndarray
        A 2D integer array `(num_cells, max_plasmids)` of genotype indices.
    trans_mask : numpy.ndarray
        A 2D boolean mask for `transformants`.
    trans_freq : numpy.ndarray
        A 1D float array `(num_cells,)` with the initial frequency of each
        cell in the transformed population.
    total_cfu0 : float
        The total initial colony-forming units (CFU) for the entire library.
    num_genotypes : int
        The total number of unique genotypes in the library.

    Returns
    -------
    numpy.ndarray
        A 1D float array `(num_genotypes,)` holding the calculated initial
        CFU for each genotype.
    """

    # Flatten the ragged array of transformants to get a 1D list of all
    # successfully transformed plasmid indices.
    all_transformed_plas = ma.array(transformants, mask=trans_mask).compressed()

    # Create an array that has the frequency of the cell from which each 
    # plasmid came from in the same shape/alignment as all_transformed_plas
    trans_freq_2d = np.broadcast_to(trans_freq[:, np.newaxis], transformants.shape)
    all_transformed_freq = ma.array(trans_freq_2d, mask=trans_mask).compressed()

    # Count the number of times each plasmid index appears, weighted by the 
    # frequency of the cells in the population
    trans_geno_counts = np.bincount(all_transformed_plas,
                                    weights=all_transformed_freq,
                                    minlength=num_genotypes)

    # Calculate the frequency of each genotype in the post-transformation pool.
    total_plasmids = np.sum(trans_geno_counts)
    if total_plasmids == 0:
        # Handle edge case where no successful transformations occurred
        trans_freqs = np.zeros(num_genotypes, dtype=float)
    else:
        trans_freqs = trans_geno_counts / total_plasmids

    # Partition the total cfu0 according to these new frequencies.
    genotype_cfu0 = total_cfu0 * trans_freqs

    return genotype_cfu0 


def _simulate_library_group(
    sub_df: pd.DataFrame,
    index_offset: int,
    lib_origin_grouper: DataFrameGroupBy,
    ordered_genotypes: np.ndarray,
    reads_per_sample: int,
    cf: dict[str, Any],
    rng: Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a complete simulation for a single library group.

    This function handles the core simulation workflow for a subset of data
    corresponding to one library group (e.g., a single library/replicate). It
    initializes output dataframes and then simulates transformation, growth,
    and sequencing for all conditions within the group.

    Parameters
    ----------
    sub_df : pandas.DataFrame
        A subset of the main phenotype dataframe for a single library group.
    index_offset : int
        An integer offset to ensure unique sample indices across all groups.
    lib_origin_grouper : pandas.DataFrameGroupBy
        A DataFrame grouped by library origin, used for transformation.
    ordered_genotypes : numpy.ndarray
        A 1D array of unique genotype names, used for consistent indexing.
    reads_per_sample : int
        The number of sequencing reads to simulate for each sample.
    cf : dict
        The main, validated configuration dictionary for the simulation.
    rng : numpy.random.Generator
        An initialized NumPy random number generator.

    Returns
    -------
    sample_df : pandas.DataFrame
        A dataframe containing simulation results for each sample in the group.
    counts_df : pandas.DataFrame
        A dataframe containing simulated counts for each genotype in the group.
    """

    condition_selector = cf["condition_selector"]
    transformation_poisson_lambda = cf["transformation_poisson_lambda"]
    growth_rate_noise = cf["growth_rate_noise"]
    library_mixture = cf["library_mixture"]
    transform_sizes = cf["transform_sizes"]
    multi_plasmid_combine_fcn = cf["multi_plasmid_combine_fcn"]
    prob_index_hop = cf["prob_index_hop"]
    total_cfu0 = cf["cfu0"]
    final_cfu_pct_err = cf["final_cfu_pct_err"]
    
    num_genotypes = len(ordered_genotypes)

    # -- create output dataframes --

    sub_grouper = sub_df.groupby(condition_selector)

    # Build sample dataframe
    sample_df = sub_grouper.agg("first")
    sample_df["cfu_per_mL"] = 0.0
    sample_df = sample_df.drop(columns="genotype")
    sample_df = sample_df.reset_index()
    sample_df.index += index_offset
    sample_df["sample"] = sample_df.index
    sample_columns = list(sample_df.columns)
    sample_columns.insert(0,"sample")
    sample_df = sample_df.loc[:,sample_columns[:-1]]

    counts_df = sub_df.copy()
    counts_df["sample"] = sub_grouper.ngroup() + index_offset
    counts_df = counts_df[["sample","genotype","dk_geno","k_pre","k_sel","theta"]]
    counts_df["ln_cfu_0"] = -np.inf
    counts_df["counts"] = 0

    # -- simulate transformation and mixing of libraries --

    transformants, trans_mask, trans_freq = _sim_transform_and_mix(
        lib_origin_grouper,
        transform_sizes,
        library_mixture,
        transformation_poisson_lambda,
        rng)

    # -- calculate cfu0 of every genotype --
    
    # This is not used in the calculation, which uses cfu/mL for *cells*. But 
    # ln_cfu_0 is a fit parameter in our analysis model, so record it in the
    # simulation to allow us to validate our analysis methods. 
    genotype_cfu0 = _calc_genotype_cfu0(transformants,
                                        trans_mask,
                                        trans_freq,
                                        total_cfu0,
                                        num_genotypes)
    

    # Get ln_cfu_0
    genotype_ln_cfu_0 = np.full(len(genotype_cfu0),-np.inf)
    non_zero_mask = genotype_cfu0 > 0
    genotype_ln_cfu_0[non_zero_mask] = np.log(genotype_cfu0[non_zero_mask])

    mapper = pd.Series(data=genotype_ln_cfu_0, index=ordered_genotypes)
    counts_df.loc[sub_df.index, "ln_cfu_0"] = sub_df["genotype"].map(mapper).to_numpy()

    # -- simulate growth -- 

    # Create a 2D array of (genotype,conditions) holding kt
    genotype_vs_kt = sub_df.pivot_table(index="genotype",
                                        columns=condition_selector,
                                        values="kt",
                                        observed=True).to_numpy()

    # Add noise to the growth rate
    if growth_rate_noise is not None:
        std = np.mean(genotype_vs_kt) * growth_rate_noise
        genotype_vs_kt += rng.normal(scale=std, size=genotype_vs_kt.shape)
    
    # Name of the library (e.g. kanR, pheS, ... ) for looking up how to 
    # combine multiple plasmid effects
    lib_name = np.unique(sub_df["library"])[0]

    trans_cfu = _sim_growth(transformants,
                            trans_mask,
                            trans_freq,
                            genotype_vs_kt,
                            total_cfu0,
                            multi_plasmid_combine_fcn[lib_name])
    
    # Record cfu/mL over all conditions
    sample_df.loc[:,"cfu_per_mL"] = np.sum(trans_cfu,axis=0)
    sample_df.loc[:,"cfu_per_mL_std"] = sample_df.loc[:,"cfu_per_mL"]*final_cfu_pct_err

    # -- simulate sequencing -- 

    read_counts = _sim_sequencing(transformants,
                                    trans_mask,
                                    trans_cfu,
                                    num_genotypes,
                                    reads_per_sample,
                                    rng)

    # Flatten read counts
    read_counts = read_counts.flatten()

    read_counts = _sim_index_hop(read_counts, prob_index_hop, rng)
    counts_df.loc[sub_df.index, "counts"] = read_counts

    return sample_df, counts_df

def selection_experiment(
    cf: dict[str, Any] | str | Path,
    library_df: pd.DataFrame | str | Path,
    phenotype_df: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a high-throughput selection experiment.

    This function orchestrates a complete simulation of a multiplexed
    selection experiment. It takes a configuration, a library definition,
    and a phenotype landscape, and simulates library mixing, transformation,
    cell growth under various conditions, and final sequencing. It models
    several common experimental artifacts.

    Parameters
    ----------
    cf : dict or str or pathlib.Path
        The configuration for the simulation. Can be a dictionary or a path
        to a YAML file.
    library_df : pandas.DataFrame or str or pathlib.Path
        A dataframe defining the composition of input libraries. Must have
        columns 'library_origin', 'genotype', and 'weight'.
    phenotype_df : pandas.DataFrame or str or pathlib.Path
        A dataframe containing the fitness landscape (growth rates `k_pre`
        and `k_sel`) for each genotype under all possible experimental
        conditions.

    Returns
    -------
    sample_df_final : pandas.DataFrame
        A dataframe describing each unique sample (experimental condition),
        including the final total CFU/mL.
    counts_df_final : pandas.DataFrame
        A dataframe containing the simulated sequencing counts for every
        genotype in every sample, along with the calculated initial
        abundance (`ln_cfu_0`) for each genotype.
    
    Examples
    --------
    >>> # Run simulation from file paths
    >>> sample_df, counts_df = selection_experiment(
    ...     cf="config.yaml",
    ...     library_df="library_composition.csv",
    ...     phenotype_df="growth_rates.csv"
    ... )
    """

    # ------------------------------------------------------------------------
    # Parse and validate inputs
    # ------------------------------------------------------------------------

    cf = _check_cf(cf)
    phenotype_df = read_dataframe(phenotype_df)
    library_df = read_dataframe(library_df)

    # Integrated check and update for library specification
    cf = _check_lib_spec(cf,library_df,phenotype_df)

    # ------------------------------------------------------------------------
    # Set up calculation
    # ------------------------------------------------------------------------
    
    print("Setting up calculation.", flush=True)

    # Initialize random number generator
    rng = np.random.default_rng(cf["random_seed"])

    # Expand the library_df so every genotype is seen in every library_origin 
    # to keep indexing consistent when we mix libraries. The .fillna(0) in this
    # chain sets the weights of added genotypes to 0 so they are not actually
    # transformed. 
    library_df = (library_df
                  .pivot_table(index="genotype",
                               columns=["library_origin"],
                               values="weight",
                               observed=True)
                  .fillna(0)
                  .stack()
                  .reset_index() # break open multi-index
                  .sort_values(["library_origin","genotype"])
                  .rename(columns={0:"weight"})
                  .reset_index(drop=True))
    
    # Get a reference list of genotypes, in order
    ordered_genotypes = pd.unique(library_df["genotype"])
    
    # groupby on library_df that lets us select individual library origins
    lib_origin_grouper = library_df.groupby(["library_origin"])

    # Build base probabilities for library transformation. By putting skew here
    # we are modeling skew that occurs during library construction, not skew 
    # during out growth or transformation. 
    library_df["probs"] = 0.0
    for _, origin_sub_df in lib_origin_grouper:
        p = _sim_plasmid_probabilities(origin_sub_df["weight"],
                                        cf["lib_assembly_skew_sigma"],rng)
        library_df.loc[origin_sub_df.index,"probs"] = p

    # Remove genotypes from phenotype_df that are not in the library
    phenotype_df = phenotype_df[phenotype_df["genotype"].isin(ordered_genotypes)]

    # Calculate growth given k values and times at each condition. Set growth
    # to zero for nan values. (These can arise if the thermodynamic model
    # failed. Interpret as "this bug does not grow"). 
    phenotype_df["kt"] = (phenotype_df["k_pre"] * phenotype_df["t_pre"] +
                          phenotype_df["k_sel"] * phenotype_df["t_sel"])
    phenotype_df["kt"] = phenotype_df["kt"].fillna(0)

    # groupby on phenotype_df that lets us select individual libraries 
    # (generally based on ['replicate','library'])
    lib_grouper = phenotype_df.groupby(cf["library_selector"], observed=True)

    # Get reads per sample
    total_num_samples = 0
    for _, sub_df in lib_grouper:
        total_num_samples += sub_df.groupby(cf["condition_selector"]).ngroups
    
    reads_per_sample = int(np.round(cf["total_num_reads"]/total_num_samples,0))

    # ------------------------------------------------------------------------
    # Main loop over all libraries
    # ------------------------------------------------------------------------
    print(f"Simulating growth and sequencing", flush=True)

    index_offset = 0
    sample_df_batches = []
    counts_df_batches = []
    for _, sub_df in tqdm(lib_grouper):

        # Do a complete transform + grow + sequence for this block of conditions
        sample_df, counts_df = _simulate_library_group(sub_df,
                                                       index_offset,
                                                       lib_origin_grouper,
                                                       ordered_genotypes,
                                                       reads_per_sample,
                                                       cf,
                                                       rng)

        sample_df_batches.append(sample_df)
        counts_df_batches.append(counts_df)

        if not sample_df.empty:
            index_offset = np.max(sample_df.index) + 1

     # ------------------------------------------------------------------------
    # Final aggregation and cleanup
    # ------------------------------------------------------------------------

    print("Generating final dataframe.", flush=True)

    sample_df_final = pd.concat(sample_df_batches)
    counts_df_final = (pd.concat(counts_df_batches,ignore_index=True)
                       .sort_values(["genotype","sample"]))
    
    print("Simulation complete.",flush=True)

    return sample_df_final, counts_df_final

