import pytest
import numpy as np
from tfscreen.util.zero_truncated_poisson import zero_truncated_poisson

@pytest.fixture
def rng():
    """Provides a consistent, seeded random number generator for tests."""
    return np.random.default_rng(seed=42)

def test_happy_path(rng):
    """Tests basic functionality with valid inputs."""
    num_samples = 100
    poisson_lambda = 2.5
    
    result = zero_truncated_poisson(num_samples, poisson_lambda, rng=rng)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_samples,)
    assert result.dtype == int
    assert np.min(result) >= 1

def test_reproducibility():
    """Ensures that providing the same seeded rng yields identical results."""
    num_samples = 50
    poisson_lambda = 1.0

    # First run
    rng1 = np.random.default_rng(seed=123)
    result1 = zero_truncated_poisson(num_samples, poisson_lambda, rng=rng1)

    # Second run with a new, but identically seeded, rng
    rng2 = np.random.default_rng(seed=123)
    result2 = zero_truncated_poisson(num_samples, poisson_lambda, rng=rng2)

    assert np.array_equal(result1, result2)

def test_default_rng_creation():
    """Tests that the function runs without a provided rng by creating its own."""
    # This should run without error
    result = zero_truncated_poisson(10, 1.5)
    assert result.shape == (10,)
    assert np.min(result) >= 1

def test_statistical_properties(rng):
    """
    Tests if the mean of a large sample is close to the theoretical mean of
    a zero-truncated Poisson distribution.
    """
    num_samples = 100_000
    poisson_lambda = 2.0
    
    # Theoretical mean of ZTP is lambda / (1 - e^-lambda)
    theoretical_mean = poisson_lambda / (1 - np.exp(-poisson_lambda))
    
    samples = zero_truncated_poisson(num_samples, poisson_lambda, rng=rng)
    sample_mean = np.mean(samples)
    
    # Check if the sample mean is within a reasonable tolerance of the theoretical mean
    assert np.isclose(sample_mean, theoretical_mean, rtol=0.01)

@pytest.mark.parametrize("invalid_samples", [0, -1, 1.5, "a", None])
def test_invalid_num_samples(invalid_samples):
    """Tests that the function rejects invalid 'num_samples' arguments."""
    err_msg = "num_samples must be an integer > 0"
    with pytest.raises(ValueError, match=err_msg):
        zero_truncated_poisson(invalid_samples, 1.0)

@pytest.mark.parametrize("invalid_lambda", [0, -1.5, np.array([1.0]), "a", None])
def test_invalid_poisson_lambda(invalid_lambda):
    """Tests that the function rejects invalid 'poisson_lambda' arguments."""
    err_msg = "poisson_lambda must be a scalar > 0"
    with pytest.raises(ValueError, match=err_msg):
        zero_truncated_poisson(10, invalid_lambda)

def test_edge_case_low_lambda(rng):
    """Tests with a very small lambda, where P(0) is high."""
    num_samples = 100
    poisson_lambda = 1e-6 # Very close to zero
    
    result = zero_truncated_poisson(num_samples, poisson_lambda, rng=rng)
    
    # With such a low lambda, nearly all results should be 1
    assert np.all(result == 1)
