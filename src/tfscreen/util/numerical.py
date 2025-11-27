import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List

# A flexible type for inputs that can be scalars or array-like
ArrayLike = Union[float, int, List, Tuple, np.ndarray, pd.Series]
# A flexible type for the output, which will be a scalar or a numpy array
NumericOutput = Union[np.float64, np.ndarray]

def to_log(
    v: ArrayLike,
    v_var: Optional[ArrayLike] = None,
    v_std: Optional[ArrayLike] = None,
) -> Union[NumericOutput, Tuple[NumericOutput, ...]]:
    """Converts a value and its uncertainty into log space.

    This function transforms the input value `v` using the natural logarithm.
    If the variance (`v_var`) or standard deviation (`v_std`) are provided,
    they are transformed using error propagation formulas:
    - log_var = v_var / v²
    - log_std = v_std / v

    Parameters
    ----------
    v : ArrayLike
        Value(s) to transform. Can be a scalar or any array-like object
        (list, tuple, numpy array, pandas Series). Values should be
        positive; non-positive inputs will result in `np.nan` or `-np.inf`.
    v_var : ArrayLike, optional
        Variance(s) of `v`. If provided, its log-transformed equivalent will be
        returned. Must have the same length as `v` if both are array-like.
    v_std : ArrayLike, optional
        Standard deviation(s) of `v`. If provided, its log-transformed
        equivalent will be returned. Must have the same length as `v` if
        both are array-like.

    Returns
    -------
    log_v : NumericOutput
        The log-transformed value(s). Returns a numpy float if the input was a
        scalar, otherwise a numpy array.
    log_v_var : NumericOutput, optional
        The log-transformed variance(s). Only returned if `v_var` is given.
    log_v_std : NumericOutput, optional
        The log-transformed standard deviation(s). Only returned if `v_std`
        is given.

    Raises
    ------
    ValueError
        If array-like inputs have inconsistent lengths.

    Examples
    --------
    >>> to_log(10)
    2.302585092994046

    >>> to_log([10, 100])
    array([2.30258509, 4.60517019])

    >>> v, v_std = to_log(100, v_std=10)
    >>> print(f"log_v = {v:.3f}, log_std = {v_std:.3f}")
    log_v = 4.605, log_std = 0.100

    >>> to_log(0)
    -inf
    """
    was_scalar = np.isscalar(v)
    v_arr = np.atleast_1d(v)

    # Perform main transformation, suppressing warnings for log(<=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_v = np.log(v_arr)

    results = [log_v]

    # Propagate errors for optional arguments
    if v_var is not None:
        v_var_arr = np.atleast_1d(v_var)
        if len(v_arr) != len(v_var_arr):
            raise ValueError("Inconsistent lengths for `v` and `v_var`.")
        with np.errstate(divide="ignore", invalid="ignore"):
            results.append(v_var_arr / (v_arr**2))

    if v_std is not None:
        v_std_arr = np.atleast_1d(v_std)
        if len(v_arr) != len(v_std_arr):
            raise ValueError("Inconsistent lengths for `v` and `v_std`.")
        with np.errstate(divide="ignore", invalid="ignore"):
            results.append(v_std_arr / v_arr)

    # Format output to match input type (scalar or array)
    if was_scalar:
        final_results = [r.item() for r in results]
    else:
        final_results = results

    if len(final_results) == 1:
        return final_results[0]
    else:
        return tuple(final_results)


def from_log(
    v: ArrayLike,
    v_var: Optional[ArrayLike] = None,
    v_std: Optional[ArrayLike] = None,
) -> Union[NumericOutput, Tuple[NumericOutput, ...]]:
    """Converts a value and its uncertainty from log space to linear space.

    This function transforms the input value `v` using the exponential
    function. If the variance (`v_var`) or standard deviation (`v_std`) are
    provided, they are transformed using error propagation formulas:
    - exp_var = v_var * (exp(v))²
    - exp_std = v_std * exp(v)

    Parameters
    ----------
    v : ArrayLike
        Value(s) in log space to transform. Can be a scalar or any
        array-like object (list, tuple, numpy array, pandas Series).
    v_var : ArrayLike, optional
        Variance(s) of `v`. If provided, its linear-space equivalent will be
        returned. Must have the same length as `v` if both are array-like.
    v_std : ArrayLike, optional
        Standard deviation(s) of `v`. If provided, its linear-space
        equivalent will be returned. Must have the same length as `v` if
        both are array-like.

    Returns
    -------
    exp_v : NumericOutput
        The transformed (linear) value(s). Returns a numpy float if the
        input was a scalar, otherwise a numpy array.
    exp_v_var : NumericOutput, optional
        The transformed (linear) variance(s). Only returned if `v_var`
        is given.
    exp_v_std : NumericOutput, optional
        The transformed (linear) standard deviation(s). Only returned if
        `v_std` is given.

    Raises
    ------
    ValueError
        If array-like inputs have inconsistent lengths.

    Examples
    --------
    >>> from_log(2.302585)
    9.99999952513946

    >>> from_log([2.302585, 4.60517])
    array([ 10., 100.])

    >>> v, v_var = from_log(5, v_var=0.1)
    >>> print(f"v = {v:.3f}, v_var = {v_var:.3f}")
    v = 148.413, v_var = 2202.647
    """
    was_scalar = np.isscalar(v)
    v_arr = np.atleast_1d(v)

    # Perform main transformation
    exp_v = np.exp(v_arr)

    results = [exp_v]

    # Propagate errors for optional arguments
    if v_var is not None:
        v_var_arr = np.atleast_1d(v_var)
        if len(v_arr) != len(v_var_arr):
            raise ValueError("Inconsistent lengths for `v` and `v_var`.")
        results.append(v_var_arr * (exp_v**2))

    if v_std is not None:
        v_std_arr = np.atleast_1d(v_std)
        if len(v_arr) != len(v_std_arr):
            raise ValueError("Inconsistent lengths for `v` and `v_std`.")
        results.append(v_std_arr * exp_v)

    # Format output to match input type (scalar or array)
    if was_scalar:
        final_results = [r.item() for r in results]
    else:
        final_results = results

    if len(final_results) == 1:
        return final_results[0]
    else:
        return tuple(final_results)