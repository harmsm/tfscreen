import numpy as np

import numpy as np

def chunk_by_group(arr, max_chunk_size):
    """
    Splits an array into chunks of a maximum size without breaking groups
    of identical, consecutive values.

    This function assumes the input array `arr` is sorted.

    Parameters
    ----------
    arr : np.ndarray or list-like
        A 1D sorted array with potentially repeating values.
    max_chunk_size : int
        The maximum size for any chunk.

    Returns
    -------
    list of np.ndarray
        A list of NumPy arrays, where each array contains the integer indices
        of the elements belonging to that chunk.

    Raises
    ------
    ValueError
        If `max_chunk_size` is less than 1, or if any single group of
        identical values is larger than `max_chunk_size`.
    TypeError
        If `max_chunk_size` cannot be converted to an integer.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # Ensure max_chunk_size is a valid integer
    try:
        max_chunk_size = int(max_chunk_size)
    except (ValueError, TypeError):
        raise TypeError("max_chunk_size must be an integer.")

    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be 1 or greater.")

    # Handle the simple case of an empty array
    if arr.size == 0:
        return [np.array([], dtype=int)]

    # Find where groups of identical values begin
    group_starts = np.where(arr[:-1] != arr[1:])[0] + 1
    boundaries = np.concatenate(([0], group_starts, [len(arr)]))

    # Check if any single group is larger than the max chunk size
    group_sizes = np.diff(boundaries)
    if np.any(group_sizes > max_chunk_size):
        raise ValueError(
            "A group of identical values is larger than max_chunk_size."
        )

    # Identify the indices where the array should be split
    split_indices = []
    # current_chunk_start_idx is an index *into the boundaries array*
    current_chunk_start_idx = 0
    for i in range(1, len(boundaries)):
        chunk_size = boundaries[i] - boundaries[current_chunk_start_idx]
        if chunk_size > max_chunk_size:
            # The chunk has grown too large, so split at the previous boundary
            split_point = boundaries[i - 1]
            split_indices.append(split_point)
            # The next chunk starts from that same previous boundary
            current_chunk_start_idx = i - 1

    indexes = np.arange(len(arr), dtype=int)
    return np.split(indexes, split_indices)