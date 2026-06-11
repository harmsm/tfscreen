import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Default quantile levels and column-name helper
# ---------------------------------------------------------------------------

_DEFAULT_QUANTILE_LEVELS = np.array([
    0.001, 0.005, 0.01, 0.025, 0.05,
    0.10, 0.159, 0.25, 0.50, 0.75, 0.841, 0.90,
    0.95, 0.975, 0.99, 0.995, 0.999,
])


def quantile_col(level):
    """Return the standard column name for quantile *level*.

    Uses ``f"q{level:g}"`` so trailing zeros are stripped:
    ``quantile_col(0.5)`` → ``"q0.5"``,
    ``quantile_col(0.025)`` → ``"q0.025"``,
    ``quantile_col(0.159)`` → ``"q0.159"``.
    """
    return f"q{level:g}"


def get_posterior_samples(param_posteriors, param_name):
    """
    Get posterior samples for a parameter, handling name fallbacks and HDF5.

    Parameters
    ----------
    param_posteriors : dict
        Dictionary mapping parameter names to posterior samples.
    param_name : str
        Name of the parameter to extract.

    Returns
    -------
    val : numpy.ndarray
        Posterior samples for the requested parameter.

    Raises
    ------
    KeyError
        If the parameter is not found in `param_posteriors`.
    """

    if param_name not in param_posteriors:
        # Try suffixes for MAP/guide keys
        found = False
        for suffix in ["_auto_loc", "_mean"]:
            if f"{param_name}{suffix}" in param_posteriors:
                param_name = f"{param_name}{suffix}"
                found = True
                break

        if not found:
            # Provide more helpful error message if possible
            available_keys = list(param_posteriors.keys())
            if len(available_keys) > 10:
                keys_str = ", ".join(available_keys[:5]) + " ... " + ", ".join(available_keys[-5:])
            else:
                keys_str = ", ".join(available_keys)

            error_msg = f"Parameter '{param_name}' not found in posteriors. Available keys: {keys_str}"
            raise KeyError(error_msg)

    val = param_posteriors[param_name]

    return val

def load_posteriors(posteriors, q_to_get=None):
    """
    Consolidate the reading logic for posterior samples and build quantile mapping.

    Parameters
    ----------
    posteriors : dict or str
        A dictionary of posteriors keying parameters to numpy arrays, a
        ``numpy.lib.npyio.NpzFile`` object, or a path to a ``.npz`` or
        ``.h5``/``.hdf5`` file containing posterior samples for model
        parameters.
    q_to_get : array-like of float, optional
        Quantile levels in [0, 1] to extract from posterior samples.  Column
        names are generated automatically as ``q{level:g}``
        (e.g. ``q0.5`` for the median, ``q0.025`` for the 2.5th percentile).
        If ``None``, a dense default set is used::

            [0.001, 0.005, 0.01, 0.025, 0.05,
             0.10, 0.159, 0.25, 0.50, 0.75, 0.841, 0.90,
             0.95, 0.975, 0.99, 0.995, 0.999]

    Returns
    -------
    tuple
        ``(q_dict, param_posteriors)`` where ``q_dict`` maps column names to
        quantile levels (e.g. ``{"q0.5": 0.5, "q0.025": 0.025, ...}``).

    Raises
    ------
    ValueError
        If ``q_to_get`` cannot be converted to a 1-D float array or any
        value is outside [0, 1].
    """

    # Load the posterior file
    if isinstance(posteriors, (dict, np.lib.npyio.NpzFile, h5py.File, h5py.Group)):
        param_posteriors = posteriors
    elif isinstance(posteriors, str):
        if posteriors.endswith(".h5") or posteriors.endswith(".hdf5"):
            import time
            max_retries = 15
            for attempt in range(max_retries):
                try:
                    param_posteriors = h5py.File(posteriors, 'r')
                    break
                except OSError as e:
                    # Retry on OSError to allow cluster file systems to sync
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise
        else:
            param_posteriors = np.load(posteriors)
    else:
        raise ValueError(
            "posteriors should be a dictionary, an .npz file (loaded or "
            "path), an .h5 file (loaded or path), or an h5py group."
        )

    # Build the quantile level array
    if q_to_get is None:
        q_arr = _DEFAULT_QUANTILE_LEVELS
    else:
        try:
            q_arr = np.atleast_1d(np.asarray(q_to_get, dtype=float)).ravel()
        except (TypeError, ValueError):
            raise ValueError(
                "q_to_get should be a 1-D array-like of quantile levels in [0, 1]; "
                f"got {type(q_to_get).__name__}"
            )
        if not np.all((q_arr >= 0) & (q_arr <= 1)):
            raise ValueError(
                "q_to_get values must all be in [0, 1]; "
                f"got min={float(q_arr.min()):.4g}, max={float(q_arr.max()):.4g}"
            )

    # Build column-name → level dict used by all downstream extraction code
    q_dict = {quantile_col(level): float(level) for level in q_arr}

    return q_dict, param_posteriors
