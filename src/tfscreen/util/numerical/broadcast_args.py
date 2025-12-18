
import numpy as np

def broadcast_args(*args):
    """
    Broadcasts scalar arguments to match the length of array arguments.

    This utility function takes a variable number of arguments, which can be
    scalars or array-like objects. It determines the maximum length among all
    array-like arguments and expands any scalars or single-element arrays to
    match that length.

    Parameters
    ----------
    *args : object or array_like
        A variable number of arguments. Inputs can be scalars (int, float,
        str) or any object that can be converted to a NumPy array (like a
        list, tuple, or another np.ndarray).

    Returns
    -------
    list of np.ndarray
        A list of NumPy arrays, where each array has been broadcast to the
        same length.

    Raises
    ------
    ValueError
        If arrays of incompatible lengths are provided (i.e., an array with
        a length greater than 1 that does not match the target length of the
        longest array).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([10, 20, 30])
    >>> y = 5
    >>> z = "a"
    >>> broadcast_args(x, y, z)
    [array([10, 20, 30]), array([5, 5, 5]), array(['a', 'a', 'a'], dtype='<U1')]

    >>> broadcast_args([1, 2], [3, 4])
    [array([1, 2]), array([3, 4])]

    >>> try:
    ...     broadcast_args([1, 2], [3, 4, 5])
    ... except ValueError as e:
    ...     print(e)
    ...
    Arguments could not be broadcast to the same length. Target length is 3, but found an array of incompatible length 2.
    """
    
    # 1. Convert all inputs to numpy arrays. Scalars become 1-element arrays.
    # np.atleast_1d is a good choice here, but we must handle strings specially
    # to prevent them from being treated as an array of characters.
    processed_args = []
    for arg in args:
        if isinstance(arg, str) or not hasattr(arg, "__iter__"):
            processed_args.append(np.array([arg]))
        else:
            processed_args.append(np.asanyarray(arg))

    # Return early if there are no arguments
    if not processed_args:
        return []

    # 2. Find the target length from the longest array.
    # The generator expression is slightly more memory-efficient than a list comprehension.
    lengths = [len(a) for a in processed_args]
    target_len = np.max(lengths)

    # 3. Validate and broadcast arguments.
    final_args = []
    for arg, length in zip(processed_args, lengths):
        if length == target_len:
            final_args.append(arg)  # Already correct length
        elif length == 1:
            # Broadcast single-element arrays to the target length
            final_args.append(np.broadcast_to(arg, target_len))
        else:
            # Any other length is an error
            err = (
                "Arguments could not be broadcast to the same length. "
                f"Target length is {target_len}, but found an array "
                f"of incompatible length {length}."
            )
            raise ValueError(err)

    return final_args

