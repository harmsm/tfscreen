import pytest
from tfscreen.util.validation.check import check_number, check_unknown_keys

# ----------------------------------------------------------------------------
# test check_number
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("value, kwargs, expected", [
    # Basic success cases
    (5, {}, 5.0),
    (5.5, {"cast_type": int}, 5),
    (None, {"allow_none": True}, None),
    # Inclusive bounds
    (0, {"min_allowed": 0}, 0.0),
    (10, {"max_allowed": 10}, 10.0),
    (5, {"min_allowed": 0, "max_allowed": 10}, 5.0),
    # Exclusive bounds
    (0.1, {"min_allowed": 0, "inclusive_min": False}, 0.1),
    (9.9, {"max_allowed": 10, "inclusive_max": False}, 9.9),
])
def test_check_number_success(value, kwargs, expected):
    """Tests successful validation and casting of numbers."""
    result = check_number(value, **kwargs)
    assert result == expected

@pytest.mark.parametrize("value, kwargs, error, match", [
    # Nullability errors
    (None, {}, ValueError, "cannot be None"),
    (None, {"param_name": "test_param"}, ValueError, "test_param cannot be None"),
    # Type and casting errors
    ([1, 2], {}, ValueError, "Value must be a scalar"),
    ("abc", {}, ValueError, "could not convert string to float: 'abc'"),
    # Boundary condition errors
    (-1, {"min_allowed": 0}, ValueError, "must be >= 0"),
    (0, {"min_allowed": 0, "inclusive_min": False}, ValueError, "must be > 0"),
    (11, {"max_allowed": 10}, ValueError, "must be <= 10"),
    (10, {"max_allowed": 10, "inclusive_max": False}, ValueError, "must be < 10"),
])
def test_check_number_failures(value, kwargs, error, match):
    """Tests that check_number correctly raises errors for invalid inputs."""
    with pytest.raises(error, match=match):
        check_number(value, **kwargs)


# ----------------------------------------------------------------------------
# test check_unknown_keys
# ----------------------------------------------------------------------------

ALLOWED = frozenset({"alpha", "beta", "gamma"})


def test_check_unknown_keys_all_valid():
    check_unknown_keys({"alpha": 1, "beta": 2}, ALLOWED)


def test_check_unknown_keys_empty_config():
    check_unknown_keys({}, ALLOWED)


def test_check_unknown_keys_single_unknown():
    with pytest.raises(ValueError, match="typo"):
        check_unknown_keys({"alpha": 1, "typo": 99}, ALLOWED)


def test_check_unknown_keys_multiple_unknown():
    with pytest.raises(ValueError) as exc_info:
        check_unknown_keys({"alpha": 1, "bad1": 2, "bad2": 3}, ALLOWED)
    msg = str(exc_info.value)
    assert "bad1" in msg
    assert "bad2" in msg


def test_check_unknown_keys_label_in_message():
    with pytest.raises(ValueError, match="my_label"):
        check_unknown_keys({"unknown": 1}, ALLOWED, label="my_label")