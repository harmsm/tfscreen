import numpy as np
import pandas as pd

def get_growth_rates_kf(times,
                        ln_cfu,
                        ln_cfu_var,
                        growth_rate_guess=0.015,
                        growth_rate_uncertainty=1.0,
                        min_measurement_noise=1e-12,
                        process_noise=1e-5):
    """
    Estimate growth rates using a linear Kalman filter.

    The Kalman filter is applied independently to each genotype's time series
    data to estimate its growth rate. The state vector consists of the
    logarithm of the population size and the growth rate. The measurement
    is the log-transformed CFU/mL.

    Parameters
    ----------
    times : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).
    ln_cfu_var : np.ndarray
        2D array of variance of the estimate of ln_cfu each genotype, 
        shape (num_genotypes, num_times).
    growth_rate_guess : float, optional
        Initial guess for the growth rate 
    growth_rate_uncertainty : float, optional
        Uncertainty (standard deviation) on the initial growth rate. Default: 0.1.
    min_measurement_noise : float, optional
        Minimum value for the measurement noise variance to avoid instability. Default: 1e-12.
    process_noise : float, optional
        The variance of the process noise, which models the uncertainty in the
        state transition. Default: 1e-5.
    
    Returns
    -------
    param_df : pandas.DataFrame
        dataframe with extracted parameters (A0_est, k_est) and their standard
        errors (A0_std, k_std). Note that A0_std is not estimated by this method.
    pred_df : pandas.DataFrame
        dataframe with obs and pred
    """

    ln_cfu_var = np.maximum(ln_cfu_var, min_measurement_noise)

    # Get num_genotypes
    num_genotypes, num_times = ln_cfu.shape

    # State vector: [ln(population), growth_rate]
    # Shape: (num_genotypes, 2, 1)
    x = np.zeros((num_genotypes, 2, 1))
    x[:, 0, 0] = ln_cfu[:, 0]  # Initialize ln(pop) with the first measurement
    x[:, 1, 0] = growth_rate_guess      # Initialize growth rate to growth_rate_guess

    # State covariance matrix (P)
    # Shape: (num_genotypes, 2, 2)
    P = np.zeros((num_genotypes, 2, 2))

    P[:, 0, 0] = ln_cfu_var[:,0]
    P[:, 1, 1] = growth_rate_uncertainty

    # Process noise covariance matrix (Q)
    # Shape: (num_genotypes, 2, 2)
    Q = np.zeros((num_genotypes, 2, 2))
    Q[:, 0, 0] = process_noise / 4
    Q[:, 0, 1] = process_noise / 2
    Q[:, 1, 0] = process_noise / 2
    Q[:, 1, 1] = process_noise

    # Measurement matrix (H)
    # Shape: (num_genotypes, 1, 2)
    H = np.zeros((num_genotypes, 1, 2))
    H[:, 0, 0] = 1.0

    # Identity matrix
    # Shape: (num_genotypes, 2, 2)
    I = np.eye(2)[np.newaxis, :, :].repeat(num_genotypes, axis=0)

    # --- Run the Kalman Filter loop over time (not genotypes) ---
    for i in range(1, num_times):
        
        # --- PREDICTION STEP (Vectorized) ---
        # Calculate dt for each genotype individually
        dt = times[:, i] - times[:, i-1]

        # State transition matrix (F) is now a stack of matrices,
        # one for each genotype, because dt is different for each.
        F = np.zeros((num_genotypes, 2, 2))
        F[:, 0, 0] = 1.0
        F[:, 1, 1] = 1.0
        F[:, 0, 1] = dt

        # Predict the next state for all genotypes at once
        x_pred = F @ x

        # Predict the next state covariance for all genotypes
        P_pred = F @ P @ F.transpose(0, 2, 1) + Q

        # --- UPDATE STEP (Vectorized) ---
        # Get measurements for all genotypes at the current time step
        z = ln_cfu[:, i].reshape(-1, 1, 1)

        # Innovation (residual) for all genotypes
        y = z - H @ x_pred

        # Measurement noise covariance (R) is now a stack of matrices,
        # one for each genotype, using the provided noise for this time step.
        R = ln_cfu_var[:, i].reshape(-1, 1, 1)

        # Innovation covariance for all genotypes
        S = H @ P_pred @ H.transpose(0, 2, 1) + R

        # Kalman Gain for all genotypes. np.linalg.inv works on stacks of matrices.
        K = P_pred @ H.transpose(0, 2, 1) @ np.linalg.inv(S)

        # Update the state estimate for all genotypes
        x = x_pred + K @ y

        # Update the state covariance for all genotypes
        P = (I - K @ H) @ P_pred


    k_est = x[:, 1, 0]
    k_std = np.sqrt(P[:, 1, 1])

    A0_est = ln_cfu[:,-1] - k_est*times[:,-1]
    A0_std = np.repeat(np.nan,k_std.shape[0])

    param_df = pd.DataFrame({"A0_est":A0_est,
                             "A0_std":A0_std,
                             "k_est":k_est,
                             "k_std":k_std})

    pred = times*k_est[:,np.newaxis] + A0_est[:,np.newaxis]
    pred_df = pd.DataFrame({"obs":ln_cfu.flatten(),
                            "pred":pred.flatten()})
        
    return param_df, pred_df


