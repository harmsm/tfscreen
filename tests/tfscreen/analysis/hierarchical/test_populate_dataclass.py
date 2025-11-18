import pytest
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

# Assuming the function is in this module
from tfscreen.analysis.hierarchical import populate_dataclass

# --- Setup: Define a target dataclass for testing ---

@dataclass
class SampleDataClass:
    # --- FIX: Non-default fields must come first ---
    a: int
    c: jnp.ndarray
    d: np.ndarray
    # Default fields
    b: str = "default_b"
    e: float = 0.0
    f: dict = None
    # -----------------------------------------------

# --- Fixtures ---

@pytest.fixture
def source_dict_1():
    return {
        "a": 10,
        "c": jnp.array([1, 2]),
    }

@pytest.fixture
def source_dict_2():
    return {
        "d": np.array([3.0, 4.0]),
        "f": {"nested": "value"}
    }

# --- Test Cases ---

class TestPopulateDataclass:

    def test_success_multiple_sources(self, source_dict_1, source_dict_2):
        """Tests successful population from a list of dicts."""
        sources = [source_dict_1, source_dict_2]
        instance = populate_dataclass(SampleDataClass, sources)

        assert isinstance(instance, SampleDataClass)
        assert instance.a == 10
        assert instance.b == "default_b"  # Correctly uses default
        assert jnp.array_equal(instance.c, jnp.array([1, 2]))
        assert np.array_equal(instance.d, np.array([3.0, 4.0]))
        assert instance.e == 0.0  # Correctly uses default
        assert instance.f == {"nested": "value"}

    def test_success_single_source_dict(self):
        """Tests successful population from a single dict."""
        source = {
            "a": 1,
            "c": jnp.array([1]),
            "d": np.array([2]),
            "b": "override" # Override default
        }
        instance = populate_dataclass(SampleDataClass, source)
        
        assert instance.a == 1
        assert instance.b == "override"
        assert jnp.array_equal(instance.c, jnp.array([1]))
        assert np.array_equal(instance.d, np.array([2]))

    def test_raises_missing_required_param(self, source_dict_2):
        """Tests error if a required param (no default) is missing."""
        # 'a' and 'c' are missing
        with pytest.raises(ValueError, match="could not find required parameter 'a'"):
            populate_dataclass(SampleDataClass, [source_dict_2])

    def test_raises_duplicate_keys(self, source_dict_1):
        """Tests error if keys are duplicated across source dicts."""
        source_duplicate = {"a": 999} # 'a' also in source_dict_1
        sources = [source_dict_1, source_duplicate]
        
        # Use set formatting in match for f-string "{'a'}"
        with pytest.raises(ValueError, match=f"the keys '{{'a'}}' are duplicated"):
            populate_dataclass(SampleDataClass, sources)

    def test_raises_invalid_source_type_list(self):
        """Tests error if sources is a list with a non-dict element."""
        sources = [{"a": 1}, "not_a_dict"]
        with pytest.raises(ValueError, match="sources should be a dictionary or list"):
            populate_dataclass(SampleDataClass, sources)

    def test_raises_invalid_source_type_other(self):
        """Tests error if sources is not a dict or list."""
        with pytest.raises(ValueError, match="sources should be a dictionary or list"):
            populate_dataclass(SampleDataClass, "not_a_list_or_dict")

    def test_raises_invalid_value_type_list(self):
        """Tests error if a value is a Python list."""
        source = {
            "a": 1,
            "c": jnp.array([1]),
            "d": [3.0, 4.0]  # This is a Python list, not np.ndarray
        }
        with pytest.raises(ValueError, match="Parameter 'd' is a '<class 'list'>'"):
            populate_dataclass(SampleDataClass, source)

    def test_raises_invalid_value_type_tuple(self):
        """Tests error if a value is a Python tuple."""
        source = {
            "a": 1,
            "c": (1, 2),  # This is a tuple
            "d": np.array([1])
        }
        with pytest.raises(ValueError, match="Parameter 'c' is a '<class 'tuple'>'"):
            populate_dataclass(SampleDataClass, source)

    def test_success_all_params_provided(self):
        """Tests success when all params, including defaults, are provided."""
        source = {
            "a": 1,
            "b": "custom_b",
            "c": jnp.array([1]),
            "d": np.array([2]),
            "e": 1.5,
            "f": {"key": "val"}
        }
        instance = populate_dataclass(SampleDataClass, source)
        
        assert instance.a == 1
        assert instance.b == "custom_b"
        assert instance.e == 1.5
        assert instance.f == {"key": "val"}