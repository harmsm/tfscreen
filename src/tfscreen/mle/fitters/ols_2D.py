import numpy as np

def run_ols_2D(x_arrays,y_arrays):
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
        se_slopes = np.divide(rse, np.sqrt(ss_xx), 
                              out=np.full_like(ss_xx, np.nan), where=ss_xx!=0)
    
        # Standard Error of the Intercept (SE_b0)
        # Handle ss_xx=0 to avoid divide by zero
        term2 = np.divide(x_mean**2, ss_xx, 
                          out=np.full_like(ss_xx, np.nan), where=ss_xx!=0)
        term_in_sqrt = (1/n_points) + term2
        se_intercepts = rse * np.sqrt(term_in_sqrt)
    else:
        se_slopes = np.nan*np.ones(len(slopes))
        se_intercepts = np.nan*np.ones(len(slopes))

    return slopes, intercepts, se_slopes, se_intercepts, residuals


