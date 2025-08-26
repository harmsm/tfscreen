import numpy as np
from tqdm.auto import tqdm
from scipy.linalg import cholesky

def get_growth_rates_ukf_lin(times,
                             ln_cfu,
                             ln_cfu_var,
                             growth_rate_guess=0.015,
                             growth_rate_uncertainty=0.1,
                             process_noise_std=[1e-5,1e-5],
                             alpha=1e-3,
                             beta=2.0,
                             kappa=0.0):
    """
    Estimates population and growth rate using a UKF with a linear process model
    and a non-linear measurement model. 

    State: [ln(population), growth_rate]
    Process Model: ln(pop)_k = ln(pop)_{k-1} + growth_rate * dt  (Linear)
    Measurement Model: z = exp(ln(pop)) (Non-linear)

    Parameters
    ----------
    times : np.ndarray
        2D array of time points, shape (num_genotypes, num_times).
    ln_cfu : np.ndarray
        2D array of ln_cfu each genotype, shape (num_genotypes, num_times).
    ln_cfu_var : np.ndarray
        2D array of variance of the estimate of ln_cfu each genotype, 
        shape (num_genotypes, num_times).
    growth_rate_guess : float or np.ndarray
        initial guess for the growth rate (usually wildtype under reference 
        conditions). Can be an array num_gentoypes long. default: 0.015 
    growth_rate_uncertainty : float, optional
        Uncertainty (standard deviation) on the initial growth rate. Default: 0.1.
    process_noise : np.ndarray
        2D array of process noise. (Assumed to be constant over all genotypes).
        Default : [1e-5,1e-5]
    alpha : float, optional
        UKF scaling parameter, determines the spread of the sigma points. Default: 1e-3.
    beta : float, optional
        UKF scaling parameter, incorporates prior knowledge of the distribution. Default: 2.0 (Gaussian).
    kappa : float, optional
        UKF scaling parameter, secondary scaling parameter. Default: 0.0.

    Returns
    -------
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,)
    growth_rate_std : np.ndarray
        1D array of standard deviations on estimated growth rates, shape (num_genotypes,)
    """
    
    cfu = np.exp(ln_cfu)
    cfu_var = ln_cfu_var*(cfu**2)

    num_genotypes, num_times = times.shape
    n_states = 2  # [population, growth_rate]

    process_noise_std = np.asarray(process_noise_std)

    # --- 2. UKF Parameters and Weights ---
    lam = alpha**2 * (n_states + kappa) - n_states
    gamma = np.sqrt(n_states + lam)

    # Weights for sigma points
    wm = np.full(2 * n_states + 1, 1 / (2 * (n_states + lam)))
    wc = np.full(2 * n_states + 1, 1 / (2 * (n_states + lam)))
    wm[0] = lam / (n_states + lam)
    wc[0] = lam / (n_states + lam) + (1 - alpha**2 + beta)

    # --- 3. Initialize UKF variables ---
    # State vector: [population, growth_rate]
    x = np.zeros((num_genotypes, n_states, 1))
    x[:, 0, 0] = ln_cfu[:, 0] # Initial state is log of first measurement
    x[:, 1, 0] = growth_rate_guess

    # State covariance matrix (P)
    P = np.zeros((num_genotypes, n_states, n_states))

    P[:, 0, 0] = ln_cfu_var[:,0]
    P[:, 1, 1] = growth_rate_uncertainty

    # Process noise covariance (Q)
    Q = np.diag(np.array(process_noise_std)**2)

    # --- 4. Run the UKF loop over time ---
    
    for i in tqdm(range(1, num_times)):
        
        # --- PREDICTION STEP ---
        # 1. Generate sigma points from current state (x) and covariance (P)
        # Shape: (num_genotypes, n_states, 2*n_states+1)
        sigma_points = np.zeros((num_genotypes, n_states, 2 * n_states + 1))
        
        # Cholesky decomposition for the matrix square root
        # Add a small identity matrix for numerical stability
        sqrt_P = cholesky(P + np.eye(n_states) * 1e-9)

        sigma_points[:, :, 0] = x[:, :, 0]
        for j in range(n_states):
            sigma_points[:, :, j + 1]            = x[:, :, 0] + gamma * sqrt_P[:, :, j]
            sigma_points[:, :, j + n_states + 1] = x[:, :, 0] - gamma * sqrt_P[:, :, j]

        # 2. Propagate sigma points through the now LINEAR state transition function
        dt = (times[:, i] - times[:, i-1])[:, np.newaxis]
        ln_pop_sigma = sigma_points[:, 0, :]
        gr_sigma = sigma_points[:, 1, :]

        # Linear state transition
        ln_pop_sigma_pred = ln_pop_sigma + gr_sigma * dt
        gr_sigma_pred = gr_sigma

        propagated_sigmas = np.stack([ln_pop_sigma_pred, gr_sigma_pred], axis=1)

        # 3. Calculate predicted state mean (x_pred) and covariance (P_pred)
        x_pred = np.sum(wm * propagated_sigmas, axis=2)[:, :, np.newaxis]
        
        P_pred = np.zeros((num_genotypes, n_states, n_states))
        for j in range(2 * n_states + 1):
            y = (propagated_sigmas[:, :, j] - x_pred[:, :, 0])[:, :, np.newaxis]
            P_pred += wc[j] * (y @ y.transpose(0, 2, 1))
        P_pred += Q

        # --- UPDATE STEP ---
        # Transform predicted sigma points into measurement space via NON-LINEAR function
        # h(x) = exp(ln(pop))
        z_sigma_pred = np.exp(propagated_sigmas[:, 0, :])
        z_pred_mean = np.sum(wm * z_sigma_pred, axis=1)

        # 2. Calculate measurement covariance (S) and cross-covariance (T)
        S = np.zeros((num_genotypes, 1, 1))
        T = np.zeros((num_genotypes, n_states, 1))
        
        # Propagate the variance from counts through the scaling factors (all
        # done in CFU not ln(CFU) space)
        R = cfu_var[:,i]

        for j in range(2 * n_states + 1):
            x_err = (propagated_sigmas[:, :, j] - x_pred[:, :, 0])[:, :, np.newaxis]
            z_err = (z_sigma_pred[:, j] - z_pred_mean)[:, np.newaxis, np.newaxis]
            
            S += wc[j] * (z_err @ z_err.transpose(0, 2, 1))
            T += wc[j] * (x_err @ z_err.transpose(0, 2, 1))
            
        S += R.reshape(-1, 1, 1)

        # 3. Calculate Kalman Gain (K) and update state (x) and covariance (P)
        K = T @ np.linalg.inv(S)
        z_actual = cfu[:, i][:, np.newaxis, np.newaxis]
        y_residual = z_actual - z_pred_mean[:, np.newaxis, np.newaxis]

        x = x_pred + K @ y_residual
        P = P_pred - (K @ S @ K.transpose(0, 2, 1))

    growth_rate_est = x[:, 1, 0]
    growth_rate_std = np.sqrt(np.abs(P[:, 1, 1]))
        
    return growth_rate_est, growth_rate_std