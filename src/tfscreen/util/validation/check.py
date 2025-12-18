import numpy as np
from typing import Any, Callable, Optional, TypeVar, Union

# Define a generic Numeric type for hinting
_Numeric = TypeVar("_Numeric", int, float)

def check_number(
    value: Any,
    param_name: Optional[str] = None,
    cast_type: Callable[[Any], _Numeric] = float,
    min_allowed: Optional[_Numeric] = None,
    max_allowed: Optional[_Numeric] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    allow_none: bool = False,
) -> Optional[_Numeric]:
    """
    Validate and cast a scalar numerical value.

    This function checks if a value is a scalar number, optionally casts it
    to a specified type, and verifies that it falls within a given range.

    Parameters
    ----------
    value : Any
        The value to validate.
    param_name : str, optional
        The name of the parameter being checked, used for clearer error messages.
    cast_type : Callable, default: float
        A function to cast the value to (e.g., `int`, `float`).
    min_allowed : int or float, optional
        The minimum allowed value. If None, no minimum is enforced.
    max_allowed : int or float, optional
        The maximum allowed value. If None, no maximum is enforced.
    inclusive_min : bool, default: True
        Whether the minimum bound is inclusive (value >= min_allowed).
    inclusive_max : bool, default: True
        Whether the maximum bound is inclusive (value <= max_allowed).
    allow_none : bool, default: False
        If True, a `value` of None is permissible and will be returned as None.

    Returns
    -------
    _Numeric or None
        The cast and validated value, or None if the input was None and
        `allow_none` was True.

    Raises
    ------
    ValueError
        If the value is None (and not `allow_none`), fails to cast, or falls
        outside the allowed range.
    TypeError
        If the value is not a scalar. This is wrapped in a ValueError.
    """

    if value is None:
        if allow_none:
            return None
        else:
            raise ValueError(f'{param_name} cannot be None')
    
    try:
        if not np.isscalar(value):
            raise TypeError("Value must be a scalar.")

        # Cast to the desired type and update the dictionary
        v_cast = cast_type(value)

        # Perform range checks on the casted value
        if min_allowed is not None:
            if inclusive_min and v_cast < min_allowed:
                raise ValueError(f"Value must be >= {min_allowed}.")
            if not inclusive_min and v_cast <= min_allowed:
                raise ValueError(f"Value must be > {min_allowed}.")
        if max_allowed is not None:
            if inclusive_max and v_cast > max_allowed:
                raise ValueError(f"Value must be <= {max_allowed}.")
            if not inclusive_max and v_cast >= max_allowed:
                raise ValueError(f"Value must be < {max_allowed}.")

    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Could not process parameter '{param_name}' with value '{value}'.\n"
            f"Reason: {e}"
        ) from e

    return v_cast