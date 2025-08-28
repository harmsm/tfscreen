import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import issparse
from tqdm.auto import tqdm


def _model(A0, k, t):
    """
    Calculates exponential growth for multiple curves simultaneously.

    Parameters
    ----------
    A0 : numpy.ndarray
        1D array of initial population sizes (at t=0).
    k : numpy.ndarray
        1D array of growth rates.
    t : numpy.ndarray
        1D array of time points.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_curves, n_times) representing the
        calculated population size at each time point for each curve.
    """

    # Broadcasting: (n, 1) * (1, t) -> (n, t)
    return A0[:, np.newaxis] * np.exp(k[:, np.newaxis] * t)


def _model_residuals(params, cfu, cfu_std, t):
    """
    Calculates weighted residuals for a block of exponential growth curves.

    This function is designed to be called by scipy.optimize.least_squares.
    It unpacks the parameter array into initial populations (A0) and growth
    rates (k), calculates the model predictions, and returns the
    standard-deviation-weighted residuals.

    Parameters
    ----------
    params : numpy.ndarray
        A 1D array containing the fitting parameters. The first half of the
        array contains the initial population (A0) guesses, and the second
        half contains the growth rate (k) guesses.
    cfu : numpy.ndarray
        A 2D array (n_curves, n_times) of observed CFU/mL values.
    cfu_std : numpy.ndarray
        A 2D array (n_curves, n_times) of the standard deviations
        associated with each CFU/mL measurement.
    t : numpy.ndarray
        A 1D array of time points at which measurements were taken.

    Returns
    -------
    numpy.ndarray
        A 1D array of the flattened, weighted residuals, suitable for
        minimization.
    """

    n = cfu.shape[0]
    A0_params = params[:n]
    k_params = params[n:]

    calc = _model(A0_params, k_params, t)
    
    # Return the flattened array of weighted residuals
    return ((cfu - calc) / cfu_std).flatten()


def get_growth_rates_nls(times,
                         cfu,
                         cfu_var,
                         growth_rate_guess=0.015,
                         initial_pop_guess=1e7,
                         block_size=100):
    """
    Performs block-wise non-linear least squares fitting.

    This function iterates through all genotypes in blocks,
    performing an independent exponential fit for each one simultaneously
    within the block. It leverages a sparse Jacobian to accelerate the
    computation.

    Parameters
    ----------
    times : numpy.ndarray
        2D array of time points.
    cfu : numpy.ndarray
        2D array of observed CFU/mL values for all genotypes.
    cfu_std : numpy.ndarray
        2D array of standard deviations for the CFU/mL values.
    growth_rate_guesses : numpy.ndarray
        1D array of initial guesses for the growth rate (k) for each genotype.
    initial_pop_guesses : numpy.ndarray
        1D array of initial guesses for the initial population (A0) for each
        genotype.
    block_size : int
        The number of genotypes to fit simultaneously in each block.

    Returns
    -------
    A0_est : np.ndarray
        1D array of estimated initial populations, shape (num_genotypes,)
    A0_std : np.ndarray
        1D array of standard errors on estimated initial populations, shape (num_genotypes,)
    growth_rate_est : np.ndarray
        1D array of estimated growth rates, shape (num_genotypes,)
    growth_rate_std : np.ndarray
        1D array of standard errors on estimated growth rates, shape (num_genotypes,)
    """

    A0_est = []
    A0_std = []
    growth_rate_est = []
    growth_rate_std = []

    num_genotypes = cfu.shape[0]
    num_times = cfu.shape[1]

    if not hasattr(growth_rate_guess,"__iter__"):
        growth_rate_guess = np.ones(num_genotypes)*growth_rate_guess
    
    if not hasattr(initial_pop_guess,"__iter__"):
        initial_pop_guess = np.ones(num_genotypes)*initial_pop_guess

    for i in tqdm(range(0, num_genotypes, block_size), desc="Fitting Growth Rates"):

        # Grab a block of data for fitting
        indices = slice(i, i + block_size)
        times_block = times[indices,:]
        cfu_block = cfu[indices, :]
        cfu_std_block = cfu_var[indices, :]
        rate_guess_block = growth_rate_guess[indices]
        pop_guess_block = initial_pop_guess[indices]

        n = cfu_block.shape[0]
        if n == 0:
            continue

        # Create array of parameter guesses: [A0_1, A0_2..., k_1, k_2,...]
        params = np.concatenate([pop_guess_block, rate_guess_block])

        # Set parameter bounds: A0 >= 0, -1 <= k <= 1
        lower_bounds = np.concatenate([np.zeros(n), -1.0 * np.ones(n)])
        upper_bounds = np.concatenate([np.inf * np.ones(n), np.ones(n)])
        bounds = (lower_bounds, upper_bounds)

        # Create a sparse jacobian because the genotypes are independent
        num_residuals = n * num_times
        num_params = 2 * n
        jac_sparsity = np.zeros((num_residuals, num_params), dtype=int)

        for j in range(n):

            # Rows for genotype j
            row_start = num_times * j
            row_end = num_times * (j + 1)
            
            # Non-zero derivatives are only for A0_j and k_j
            # Column for A0_j parameter
            jac_sparsity[row_start:row_end, j] = 1

            # Column for k_j parameter
            jac_sparsity[row_start:row_end, n + j] = 1

        # Do nonlinear regression
        fit = least_squares(_model_residuals,
                            x0=params,
                            bounds=bounds,
                            jac_sparsity=jac_sparsity,
                            args=(cfu_block, cfu_std_block, times_block))

        # Extract standard error from the covariance matrix
        J = fit.jac
        if issparse(J):
            JTJ = (J.T @ J).toarray()
        else:
            JTJ = np.dot(J.T, J)
        
        # Use pseudo-inverse to avoid "Singular matrix" errors
        # for poorly constrained fits.
        cov = np.linalg.pinv(2 * JTJ)
        with np.errstate(invalid='ignore'): # Ignore sqrt(negative) for bad fits
            std = np.sqrt(np.diagonal(cov))

        # Record results
        A0_est.extend(fit.x[:n])
        A0_std.extend(std[:n])

        growth_rate_est.extend(fit.x[n:])
        growth_rate_std.extend(std[n:])

    A0_est = np.array(A0_est)
    A0_std = np.array(A0_std)
    growth_rate_est = np.array(growth_rate_est)
    growth_rate_std = np.array(growth_rate_std)

    return A0_est, A0_std, growth_rate_est, growth_rate_std
