
from tfscreen.util import process_counts
from tfscreen.fitting import fast_weighted_linear_regression

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


def _do_nls(times,
            cfu,
            cfu_std,
            growth_rate_guesses,
            initial_pop_guesses,
            block_size):
    """
    Performs block-wise non-linear least squares fitting.

    This internal function iterates through all genotypes in blocks,
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
    growth_rate_est : numpy.ndarray
        1D array of the final estimated growth rates (k) for all genotypes.
    growth_rate_std : numpy.ndarray
        1D array of the estimated standard errors for each growth rate.
    """

    growth_rate_est = []
    growth_rate_std = []

    num_genotypes = cfu.shape[0]
    num_times = cfu.shape[1]

    for i in tqdm(range(0, num_genotypes, block_size), desc="Fitting Growth Rates"):

        # Grab a block of data for fitting
        indices = slice(i, i + block_size)
        times_block = times[indices,:]
        cfu_block = cfu[indices, :]
        cfu_std_block = cfu_std[indices, :]
        rate_guess_block = growth_rate_guesses[indices]
        pop_guess_block = initial_pop_guesses[indices]

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
        growth_rate_est.extend(fit.x[n:])
        growth_rate_std.extend(std[n:])

    return np.array(growth_rate_est), np.array(growth_rate_std)


def get_growth_rates_nls(times,
                         sequence_counts,
                         total_counts,
                         total_cfu_ml,
                         pseudocount=1,
                         block_size=100):
    """
    Estimates exponential growth rates from sequencing count data.

    This function orchestrates the entire fitting process. It first
    calculates CFU/mL and its variance from raw counts. Then, it performs a
    fast weighted linear regression on the log-transformed data to obtain
    robust initial guesses for the parameters. Finally, it uses these
    guesses in a non-linear least squares fit to the original data to get
    the final growth rate estimates and their standard errors.

    Parameters
    ----------
    times : numpy.ndarray
        A 1D array of time points at which samples were taken.
    sequence_counts : numpy.ndarray
        A 2D array (n_genotypes, n_times) of the raw sequencing counts for
        each genotype at each time point.
    total_counts : numpy.ndarray
        A 1D array (n_times) of the total sequencing counts at each time point.
    total_cfu_ml : numpy.ndarray
        A 1D array (n_times) of the total measured CFU/mL of the culture
        at each time point.
    pseudocount : int, optional
        A small count to add to all sequence counts to handle zeros before
        log transformation, by default 1.
    block_size : int, optional
        The number of genotypes to fit simultaneously. A larger size can
        be faster but uses more memory, by default 100.

    Returns
    -------
    growth_rate_est : numpy.ndarray
        A 1D array containing the final estimated growth rate (k) for each
        genotype.
    growth_rate_std : numpy.ndarray
        A 1D array containing the estimated standard error for each growth
        rate.
    """
    
    counts = process_counts(sequence_counts,
                            total_counts,
                            total_cfu_ml,
                            pseudocount=pseudocount)
    
    cfu = counts["cfu"]
    cfu_std = np.sqrt(counts["cfu_var"])
    ln_cfu = counts["ln_cfu"]
    ln_cfu_var = counts["ln_cfu_var"]
    
    # Do fast linear regression on ln(cfu) to get initial guesses
    wls_results = fast_weighted_linear_regression(times,
                                                  ln_cfu,
                                                  ln_cfu_var)
    growth_rate_guesses = wls_results[0]
    initial_pop_guesses = np.exp(wls_results[1])

    growth_rate_est, growth_rate_std = _do_nls(
        times=times,
        cfu=cfu,
        cfu_std=cfu_std,
        growth_rate_guesses=growth_rate_guesses,
        initial_pop_guesses=initial_pop_guesses,
        block_size=block_size
    )

    return growth_rate_est, growth_rate_std

