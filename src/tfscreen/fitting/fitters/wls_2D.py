import numpy as np

def run_wls_2D(x_arrays,
               y_arrays,
               y_err_arrays):
    """
    Compute slopes, intercepts, and their standard errors for n-point datasets
    using weighted least squares (WLS).

    This function uses a vectorized approach based on the analytical solutions
    for WLS regression parameters and their standard errors, assuming the
    provided variances are known.

    Parameters
    ----------
    x_arrays : np.ndarray
        A 2D NumPy array of x-coordinates, with shape (n_datasets, n_points).
    y_arrays : np.ndarray
        A 2D NumPy array of y-coordinates, with shape (n_datasets, n_points).
    y_err_arrays : np.ndarray
        A 2D NumPy array of the variance (sigma^2) of each y measurement,
        with shape (n_datasets, n_points).

    Returns
    -------
    slopes : np.ndarray
        The calculated slope for each dataset.
    intercepts : np.ndarray
        The calculated intercept for each dataset.
    se_slopes : np.ndarray
        The standard error of the slope for each dataset.
    se_intercepts : np.ndarray
        The standard error of the intercept for each dataset.
    residuals : np.ndarray
        Residuals (y_pred - y)/y_err for the dataset
    """
    
    # --- Step 0: Data Preparation ---
    x = np.asarray(x_arrays)
    y = np.asarray(y_arrays)
    y_err = np.asarray(y_err_arrays)
    
    # If a single dataset is passed, add a dimension to make it 2D
    if x.ndim == 1:
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]
        y_err = y_err[np.newaxis, :]

    # Get the number of points per fit (n_points)
    n_points = x.shape[1]
    if n_points < 2:
        nan_array = np.full(x.shape[0], np.nan)
        return nan_array, nan_array, nan_array, nan_array, nan_array

    # --- Step 1: Calculate Weights and Weighted Sums ---
    # The weight of each point is the inverse of its variance.
    # Handle cases where variance is zero to avoid division errors.
    weights = np.divide(1.0,
                        y_err,
                        out=np.zeros_like(y_err, dtype=float),
                        where=y_err!=0)
    
    # Ensure weights are 0 if x or y are NaN. This ensures nansum correctly
    # ignores these points in all sums. 
    weights[np.isnan(x)] = 0
    weights[np.isnan(y)] = 0

    # Calculate the necessary weighted sums for all datasets at once.
    sw = np.nansum(weights, axis=1)
    swx = np.nansum(weights * x, axis=1)
    swy = np.nansum(weights * y, axis=1)
    swxx = np.nansum(weights * x**2, axis=1)
    swxy = np.nansum(weights * x * y, axis=1)

    # --- Step 2: Calculate Slope and Intercept ---
    # This term is the determinant of the design matrix
    delta = sw * swxx - swx**2

    # Calculate slopes (b1) and intercepts (b0) for each dataset.
    # Use np.divide to safely handle cases where delta is zero (e.g., all x
    # are identical).
    slopes = np.divide(sw * swxy - swx * swy,
                       delta,
                       out=np.full_like(delta, np.nan),
                       where=delta!=0)
    intercepts = np.divide(swxx * swy - swx * swxy,
                           delta,
                           out=np.full_like(delta, np.nan),
                           where=delta!=0)

    # --- Step 3: Calculate Standard Errors ---
    # The standard errors are calculated assuming the input variances are known.
    
    # Variance of the slope
    var_slopes = np.divide(sw,
                           delta,
                           out=np.full_like(delta, np.nan),
                           where=delta!=0)
    se_slopes = np.sqrt(var_slopes)

    # Variance of the intercept
    var_intercepts = np.divide(swxx,
                               delta,
                               out=np.full_like(delta, np.nan), 
                               where=delta!=0)
    se_intercepts = np.sqrt(var_intercepts)

    y_pred = intercepts[:, np.newaxis] + slopes[:, np.newaxis] * x
    weighted_residuals = weights*(y_pred - y)

    return slopes, intercepts, se_slopes, se_intercepts, weighted_residuals


