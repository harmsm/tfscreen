import jax
import collections
import inspect

def populate_dataclass(target_dataclass,sources):
    """
    Populates a target flax dataclass with data from sources

    This method inspects the `__init__` signature of `target_dataclass`
    and populates its arguments using keys the list of dictionaries
    in `sources`. 

    It includes validation to ensure no Python lists or tuples are
    passed, as these are not valid JAX Pytree nodes. It also checks for 
    duplicate keys in source dictionaries. 

    Parameters
    ----------
    target_dataclass : type
        The (e.g., flax) dataclass to instantiate.
    sources : dict or list
        dict or list of dicts holding keys to load.

    Returns
    -------
    object
        An instance of `target_dataclass` populated with tensor data.

    Raises
    ------
    ValueError
        If a key is duplicated across the sources
    ValueError
        If a required dataclass parameter is not found in `sources`
    ValueError
        If a value is a Python `list` or `tuple`, which are
        not valid JAX Pytree nodes.
    """

    # Standardize sources as a list of dicts
    if isinstance(sources,dict):
        sources = [sources]

    # Go through all sources and create a flat dictionary of keys, checking
    # for duplicate keys.
    loaded_source_data = {}
    for source in sources:
        if not isinstance(source,dict):
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
    for k in required_keys:

        
        if k in loaded_source_data:
            value = loaded_source_data[k]
        else:
            raise ValueError(
                f"could not find required parameter '{k}' in sources"
            )

        # Check if the value is an iterable (like list or tuple) but NOT a
        # string/bytes and NOT a JAX array.
        is_iterable = isinstance(value, collections.abc.Iterable)
        is_string = isinstance(value, (str, bytes))
        is_jax_array = isinstance(value, jax.Array) 

        if is_iterable and not is_string and not is_jax_array:
            raise ValueError(
                f"Parameter '{k}' is a '{type(value)}', but must be a "
                f"jnp.ndarray, scalar, or bool. Python lists/tuples are not "
                f"valid JAX Pytree nodes."
            )        
        
        dataclass_kwargs[k] = value
     
    return target_dataclass(**dataclass_kwargs) 

