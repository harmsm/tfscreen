from tfscreen.util import process_counts
import numpy as np

def get_growth_rates_kf(times,
                        sequence_counts,
                        total_counts,
                        total_cfu_ml,
                        pseudocount=1.0,
                        min_measurement_noise=1e-12,
                        process_noise=1e-5,
                        growth_rate_guess=0.015,
                        growth_rate_guess_std=1.0):
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
    sequence_counts : np.ndarray
        2D array of sequence counts for each genotype, shape (num_genotypes, num_times).
    total_counts : np.ndarray
        2D array of total sequence counts for each time point, shape (num_genotypes, num_times).
    total_cfu_ml : np.ndarray
        2D array of total CFU/mL measurements, shape (num_genotypes, num_times).
    pseudocount : float, optional
        Pseudocount added to sequence counts to avoid division by zero. Default: 1.0.
    min_measurement_noise : float, optional
        Minimum value for the measurement noise variance to avoid instability. Default: 1e-12.
    process_noise : float, optional
        The variance of the process noise, which models the uncertainty in the
        state transition. Default: 1e-5.
    growth_rate_guess : float, optional
        Initial guess for the growth rate (usually wildtype under reference
        conditions). Default: 0.015.
    growth_rate_guess_std : float, optional
        Initial uncertainty (standard deviation) for the growth rate. Default: 1.0.

    Returns
    -------
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,)
    growth_rate_std : np.ndarray
        1D array of standard errors on growth rate estimates, shape (num_genotypes,)
    """

    _counted = process_counts(sequence_counts,
                              total_counts,
                              total_cfu_ml,
                              pseudocount=pseudocount)
    
    ln_pop_measured = _counted["ln_cfu"]
    ln_pop_var = _counted["ln_cfu_var"]
    ln_pop_var = np.maximum(ln_pop_var, min_measurement_noise)

    # Get num_genotypes
    num_genotypes, num_times = ln_pop_measured.shape

    # State vector: [ln(population), growth_rate]
    # Shape: (num_genotypes, 2, 1)
    x = np.zeros((num_genotypes, 2, 1))
    x[:, 0, 0] = ln_pop_measured[:, 0]  # Initialize ln(pop) with the first measurement
    x[:, 1, 0] = growth_rate_guess      # Initialize growth rate to growth_rate_guess

    # State covariance matrix (P)
    # Shape: (num_genotypes, 2, 2)
    P = np.zeros((num_genotypes, 2, 2))

    P[:, 0, 0] = ln_pop_var[:,0]
    P[:, 1, 1] = growth_rate_guess_std

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
        z = ln_pop_measured[:, i].reshape(-1, 1, 1)

        # Innovation (residual) for all genotypes
        y = z - H @ x_pred

        # Measurement noise covariance (R) is now a stack of matrices,
        # one for each genotype, using the provided noise for this time step.
        R = ln_pop_var[:, i].reshape(-1, 1, 1)

        # Innovation covariance for all genotypes
        S = H @ P_pred @ H.transpose(0, 2, 1) + R

        # Kalman Gain for all genotypes. np.linalg.inv works on stacks of matrices.
        K = P_pred @ H.transpose(0, 2, 1) @ np.linalg.inv(S)

        # Update the state estimate for all genotypes
        x = x_pred + K @ y

        # Update the state covariance for all genotypes
        P = (I - K @ H) @ P_pred

    growth_rate_est = x[:, 1, 0]
    growth_rate_std = np.sqrt(P[:, 1, 1])
        
    return growth_rate_est, growth_rate_std


