
import inspect
import numpy as np
import torch

def populate_dataclass(target_dataclass, sources):
    """
    Populates a frozen dataclass with data from sources.

    This method inspects the `__init__` signature of `target_dataclass`
    and populates its arguments using keys from the list of dictionaries
    in `sources`.

    It includes validation to ensure no Python lists or tuples are
    passed. It also checks for duplicate keys in source dictionaries and
    respects parameters with default values.

    Parameters
    ----------
    target_dataclass : type
        The dataclass to instantiate.
    sources : dict or list
        A dict or list of dicts holding keys to load.

    Returns
    -------
    object
        An instance of `target_dataclass` populated with data.

    Raises
    ------
    ValueError
        If a key is duplicated across the sources.
    ValueError
        If a required dataclass parameter (one without a default
        value) is not found in `sources`.
    ValueError
        If a value is a Python `list` or `tuple`.
    """

    # Standardize sources as a list of dicts
    if isinstance(sources, dict):
        sources = [sources]

    # Go through all sources and create a flat dictionary of keys, checking
    # for duplicate keys.
    loaded_source_data = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(
                "sources should be a dictionary or list of dictionaries."
            )

        duplicates = loaded_source_data.keys() & source.keys()
        if duplicates:
            raise ValueError(
                f"the keys '{duplicates}' are duplicated in the source data"
            )

        loaded_source_data.update(source)

    # Get required parameters from the target data class
    required_keys = inspect.signature(target_dataclass).parameters

    # Construct dataclass kwargs from data in dataclass.
    dataclass_kwargs = {}
    for k, param in required_keys.items():

        if k in loaded_source_data:
            value = loaded_source_data[k]
        else:
            
            # Check if the parameter has a default value
            if param.default == inspect.Parameter.empty:
                # No default value, so this is a genuine error
                raise ValueError(
                    f"could not find required parameter '{k}' in sources"
                )
            # Parameter has a default, so we skip it.
            # It will not be in dataclass_kwargs, and the
            # dataclass constructor will use its default.
            continue

        # Check for types that are not valid tensor types.
        if isinstance(value, (list, tuple)):
            raise ValueError(
                f"Parameter '{k}' is a '{type(value)}', but must be a "
                f"torch.Tensor, np.ndarray, or scalar. Python lists/tuples are not "
                f"supported."
            )

        # Coerce any numpy arrays into torch tensors
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)

        dataclass_kwargs[k] = value

    return target_dataclass(**dataclass_kwargs)

