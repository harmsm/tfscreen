# tfscreen/util/vstack_padded.py

import numpy as np

def vstack_padded(arrays, fill_value=0):
    """
    Vertically stacks arrays that may differ in their second dimension, padding
    smaller arrays with a fill_value.

    Args:
        arrays (list or tuple): A sequence of numpy arrays to stack.
        fill_value (int, float, etc.): The value to use for padding.

    Returns:
        np.ndarray: The stacked and padded array.

    Raises:
        TypeError: If an element in the input list is not a numpy array.
    """
    if not arrays:
        return np.empty((0, 0))

    # Ensure all inputs are at least 2D numpy arrays
    processed_arrays = []
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError("All elements in the input list must be numpy arrays.")
        processed_arrays.append(np.atleast_2d(arr))
    
    # Check for simple case where all arrays have the same width
    col_dims = [arr.shape[1] for arr in processed_arrays]
    if len(set(col_dims)) == 1:
        return np.vstack(processed_arrays)
    
    # Find the common data type that can safely hold all array values
    common_dtype = np.result_type(*processed_arrays)
    
    # Calculate the shape of the final array
    total_rows = sum(arr.shape[0] for arr in processed_arrays)
    max_cols = max(col_dims)
    
    # Pre-allocate a full-sized output array with the common dtype
    out_array = np.full((total_rows, max_cols), 
                        fill_value, 
                        dtype=common_dtype)
    
    current_row = 0
    for arr in processed_arrays:
        rows, cols = arr.shape
        out_array[current_row : current_row + rows, :cols] = arr
        current_row += rows
        
    return out_array