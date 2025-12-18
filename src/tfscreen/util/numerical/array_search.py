import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple

import numba

def strict_array_search(search_in: np.ndarray, search_for: np.ndarray) -> int:
    """
    Finds the start index of a sequence within a NumPy array.

    Parameters
    ----------
    search_in : numpy.ndarray
        1D numpy array to search IN.
    search_for : numpy.ndarray
        1D numpy array to search FOR.

    Returns
    -------
    int
        The starting index of the first match, or -1 if no match is found.
    """

    # ensure the array is contiguous
    search_in = np.ascontiguousarray(search_in)

    Na, Nseq = search_in.size, search_for.size
    if Nseq == 0 or Na < Nseq:
        return -1

    # Create a view of all possible sub-arrays of length Nseq without
    # creating a large intermediate array. This is a standard, fast technique.
    shape = (Na - Nseq + 1, Nseq)
    strides = (search_in.itemsize, search_in.itemsize)
    windows = as_strided(search_in, shape=shape, strides=strides)

    # Find the first window that is an exact match
    # (windows == search_for) creates a boolean array for each row
    # .all(axis=1) reduces this to a 1D boolean array (True for matching rows)
    matches = (windows == search_for).all(axis=1)
    match_starts = np.flatnonzero(matches)

    if match_starts.size > 0:
        return match_starts[0]

    return -1

@numba.jit(nopython=True)
def fuzzy_array_search(search_in: np.ndarray, search_for: np.ndarray) -> Tuple[int, int]:
    """
    Fuzzy match of a sequence in a numpy array.

    Finds the best possible alignment of `search_for` within `search_in`
    and returns the number of differences (Hamming distance) and the start
    index of that best alignment. In case of tied scores, returns the
    first index found.

    Parameters
    ----------
    search_in : numpy.ndarray
        1D numpy array to search IN.
    search_for : numpy.ndarray
        1D numpy array to search FOR.

    Returns
    -------
    tuple[int, int]
        A tuple containing:
        - The number of differences for the best match (0 for a perfect match).
        - The starting index of the best match.
    """

    search_len = search_for.size
    search_in_len = search_in.size

    if search_len == 0 or search_in_len < search_len:
        return (search_len, -1)

    num_windows = search_in_len - search_len + 1
    
    best_match_idx = -1
    min_diffs = search_len + 1 # Start with a score worse than any possible

    # This simple loop becomes extremely fast when compiled by Numba
    for i in range(num_windows):
        current_diffs = 0
        for j in range(search_len):
            if search_in[i+j] != search_for[j]:
                current_diffs += 1
        
        if current_diffs < min_diffs:
            min_diffs = current_diffs
            best_match_idx = i
            # Early exit for perfect match
            if min_diffs == 0:
                break
    
    return min_diffs, best_match_idx