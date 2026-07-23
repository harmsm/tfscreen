import numpy as np

def predict_with_error(some_model,
                       params,
                       cov_matrix,
                       args=None,
                       epsilon=1e-6,
                       full_cov=False):
    """
    Calculate model predictions and their standard errors.

    Uses the linear error propagation method based on a first-order Taylor
    expansion of the model. This requires calculating the Jacobian of the
    model with respect to its parameters.

    Parameters
    ----------
    some_model : callable
        The model function
    params : np.ndarray
        The array of best-fit parameters for the model.
    cov_matrix : np.ndarray
        The covariance matrix of the fitted parameters.
    args : tuple, optional
        A tuple of additional fixed arguments (e.g., x-values) required by
        `some_model`.
    epsilon : float, optional
        The small step size used for numerical differentiation (central
        difference method) to calculate the Jacobian.
    full_cov : bool, optional
        If True, also return the full prediction covariance matrix
        ``J @ Cov(p) @ J.T`` (shape ``(M, M)`` for ``M`` predicted points).
        Default False.

    Returns
    -------
    calc_values : np.ndarray
        The predicted values from the model.
    calc_se : np.ndarray
        The standard error for each predicted value.
    calc_cov : np.ndarray
        Only returned when ``full_cov`` is True: the full ``(M, M)`` prediction
        covariance matrix. Filled with NaN if ``cov_matrix`` is invalid.
    """

    num_params = len(params)
    if args is None:
        args = []

    calc_values = some_model(params,*args)

    # If the covariance matrix is invalid, we can't propagate error.
    if np.any(np.isnan(cov_matrix)):
        calc_se = np.full_like(calc_values, np.nan)
        if full_cov:
            m = calc_values.size
            return calc_values, calc_se, np.full((m, m), np.nan)
        return calc_values, calc_se

    # Calculate the Jacobian of the Model (J_pred), which is the matrix of
    # partial derivatives: ∂(predicted_value) / ∂(parameter).
    # This is calculated numerically via the central difference method.
    J_pred = np.zeros((calc_values.size, num_params))
    for i in range(num_params):

        params_plus = params.copy()
        params_plus[i] += epsilon
        pred_plus = some_model(params_plus,*args)

        params_minus = params.copy()
        params_minus[i] -= epsilon
        pred_minus = some_model(params_minus,*args)

        derivative = (pred_plus - pred_minus) / (2 * epsilon)
        J_pred[:, i] = derivative

    # Propagate error: Cov(y) = J @ Cov(p) @ J.T. The per-point variance is the
    # diagonal, which can be computed efficiently without forming the full
    # matrix.
    calc_var = np.sum((J_pred @ cov_matrix) * J_pred, axis=1)

    with np.errstate(invalid='ignore'):  # Ignore sqrt of potential negative variance
        calc_se = np.sqrt(calc_var)

    if full_cov:
        calc_cov = J_pred @ cov_matrix @ J_pred.T
        return calc_values, calc_se, calc_cov

    return calc_values, calc_se