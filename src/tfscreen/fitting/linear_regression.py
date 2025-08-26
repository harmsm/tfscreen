import numpy as np

def fast_linear_regression(x_arrays,
                           y_arrays):
    """
    Compute slopes, intercepts, and their standard errors for n-point datasets.

    This function uses a vectorized approach based on the analytical solutions
    for ordinary least squares (OLS) regression parameters and their standard
    errors.

    Parameters
    ----------
    x_arrays : np.ndarray
        A 2D NumPy array of x-coordinates, with shape (n_datasets, n_points).
    y_arrays : np.ndarray
        A 2D NumPy array of y-coordinates, with shape (n_datasets, n_points).

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
        Residuals (y_pred - y) for the dataset
    """
    
    # Ensure inputs are NumPy arrays
    x = np.asarray(x_arrays)
    y = np.asarray(y_arrays)
    
    # If a single dataset is passed, add a dimension to make it 2D
    if x.ndim == 1:
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]

    # Get the number of points per fit (n_points)
    n_points = x.shape[1]
    if n_points < 2:
        nan_array = np.full(x.shape[0], np.nan)
        return nan_array, nan_array, nan_array, nan_array, nan_array
    
    # Get the degrees of freedom (df)
    df = n_points - 2
    
    # --- Step 1: Calculate Slope and Intercept ---
    x_mean = np.mean(x, axis=1)
    y_mean = np.mean(y, axis=1)

    # Sum of squares for x (Sxx) and sum of products for xy (Sxy)
    ss_xx = np.sum((x - x_mean[:, np.newaxis])**2, axis=1)
    ss_xy = np.sum((y - y_mean[:, np.newaxis]) * (x - x_mean[:, np.newaxis]), axis=1)

    # Calculate slopes (b1)
    # Handle the case where ss_xx is zero to avoid division by zero errors.
    slopes = np.divide(ss_xy, ss_xx, out=np.full_like(ss_xy, np.nan), where=ss_xx!=0)
    
    # Calculate intercepts (b0)
    intercepts = y_mean - slopes * x_mean

    # --- Step 2: Calculate Standard Errors ---
    # Predicted y values and Residual Sum of Squares (RSS)
    y_pred = intercepts[:, np.newaxis] + slopes[:, np.newaxis] * x
    residuals = y - y_pred
    rss = np.sum((residuals)**2, axis=1)

    # Residual Standard Error (RSE)
    if df > 0:
        rse = np.sqrt(rss / df)
    
        # Standard Error of the Slope (SE_b1)
        se_slopes = rse / np.sqrt(ss_xx)
    
        # Standard Error of the Intercept (SE_b0)
        term_in_sqrt = (1/n_points) + (x_mean**2 / ss_xx)
        se_intercepts = rse * np.sqrt(term_in_sqrt)
    else:
        se_slopes = np.nan*np.ones(len(slopes))
        se_intercepts = np.nan*np.ones(len(slopes))

    return slopes, intercepts, se_slopes, se_intercepts, residuals


def fast_weighted_linear_regression(x_arrays,
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
        return nan_array, nan_array, nan_array, nan_array

    # --- Step 1: Calculate Weights and Weighted Sums ---
    # The weight of each point is the inverse of its variance.
    # Handle cases where variance is zero to avoid division errors.
    weights = np.divide(1.0,
                        y_err,
                        out=np.zeros_like(y_err, dtype=float),
                        where=y_err!=0)

    # Calculate the necessary weighted sums for all datasets at once.
    sw = np.sum(weights, axis=1)
    swx = np.sum(weights * x, axis=1)
    swy = np.sum(weights * y, axis=1)
    swxx = np.sum(weights * x**2, axis=1)
    swxy = np.sum(weights * x * y, axis=1)

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


