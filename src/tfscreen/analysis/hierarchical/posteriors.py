import numpy as np
import h5py

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
    Consolidates the reading logic for posterior samples and sets default quantiles.

    Parameters
    ----------
    posteriors : dict or str
        Assumes this is a dictionary of posteriors keying parameters to
        numpy arrays, a numpy.lib.npyio.NpzFile object, or a path to a
        .npz or .h5/.hdf5 file containing posterior samples for model
        parameters.
    q_to_get : dict, optional
        Dictionary mapping output column names to quantile values (between 0 and 1)
        to extract from the posterior samples. If None, a default set of quantiles
        is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
        upper_quartile, upper_95, max).

    Returns
    -------
    tuple
        A tuple containing (q_to_get, param_posteriors).
    """

    # Load the posterior file
    if isinstance(posteriors, (dict, np.lib.npyio.NpzFile, h5py.File, h5py.Group)):
        param_posteriors = posteriors
    elif isinstance(posteriors, str):
        if posteriors.endswith(".h5") or posteriors.endswith(".hdf5"):
            param_posteriors = h5py.File(posteriors, 'r')
        else:
            param_posteriors = np.load(posteriors)
    else:
        raise ValueError(
            "posteriors should be a dictionary, an .npz file (loaded or "
            "path), an .h5 file (loaded or path), or an h5py group."
        )

    # Named quantiles to pull from the posterior distribution
    if q_to_get is None:
        q_to_get = {"min": 0.0,
                    "lower_95": 0.025,
                    "lower_std": 0.159,
                    "lower_quartile": 0.25,
                    "median": 0.5,
                    "upper_quartile": 0.75,
                    "upper_std": 0.841,
                    "upper_95": 0.975,
                    "max": 1.0}

    # make sure q_to_get is a dictionary
    if not isinstance(q_to_get, dict):
        raise ValueError(
            "q_to_get should be a dictionary keying column names to quantiles"
        )
    
    return q_to_get, param_posteriors
