import numpy as np

def chunk_by_group(arr, max_chunk_size):
    """
    Splits an array into chunks of max size N without breaking groups of
    identical values. Returns lists of numpy arrays of indexes for the chunks.
    It assumes arr values are sorted by value, that values can repeat, but
    that the number of each value is different. 
    
    For example: [0,0,0,1,1,2,2,2,2]

    Parameters
    ----------
    arr : np.ndarray
        A 1D sorted array with repeating values.
    max_chunk_size : int
        the maximum size for any chunk.

    Returns
    -------
    chunks : list
        A list of NumPy arrays with indexes representing the chunks.
    """
    
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    max_chunk_size = int(max_chunk_size)
    if max_chunk_size < 1:
        err = "max_chunk_size must be 1 or greater.\n"
        raise ValueError(err)

    # Find the start of each new group of numbers.
    group_starts = np.where(arr[:-1] != arr[1:])[0] + 1

    # Create a full list of boundaries including the start (0) and end of the
    # array. 
    boundaries = np.concatenate(([0], group_starts, [len(arr)]))

    # Iterate through boundaries to find split points.
    split_indices = []
    current_chunk_start = 0  
    for i in range(1, len(boundaries)):
        
        # If the current group's end minus the chunk's start exceeds
        # max_chunk_size...
        if boundaries[i] - boundaries[current_chunk_start] > max_chunk_size:
            
            # ...then we must split at the previous group's end.
            split_point = boundaries[i-1]
            split_indices.append(split_point)
            
            # The new chunk will start from that split point.
            current_chunk_start = i-1
            
    # Return indexes split into chunks
    indexes = np.arange(len(arr),dtype=int)
    return np.split(indexes, split_indices)