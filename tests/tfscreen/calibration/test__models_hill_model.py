import numpy as np
from tfscreen.calibration._models import hill_model


def test_hill_model_1d_params():
    """Tests the Hill model with a single set of parameters."""
    # baseline=0.1, amplitude=0.9, K=10, n=2
    params = np.array([0.1, 0.9, 10, 2])
    x = np.array([0, 10, 1000000]) # Test below K, at K, and way above K

    result = hill_model(params, x)

    # At x=0, should be close to baseline
    assert np.isclose(result[0], 0.1)
    # At x=K, should be at the midpoint
    assert np.isclose(result[1], 0.1 + 0.9 * 0.5)
    # At x >> K, should be close to baseline + amplitude
    assert np.isclose(result[2], 0.1 + 0.9)

def test_hill_model_2d_params():
    """Tests the Hill model with a vectorized (2D) set of parameters."""
    # Two parameter sets for two points
    # P1: baseline=0, amp=1, K=10, n=1
    # P2: baseline=0.1, amp=0.9, K=20, n=2
    params = np.array([
        [0.0, 0.1],  # baselines
        [1.0, 0.9],  # amplitudes
        [10.0, 20.0], # Ks
        [1.0, 2.0]   # ns
    ])
    x = np.array([10, 20]) # Each x value corresponds to a parameter set

    result = hill_model(params, x)

    # For the first point (x=K=10), result should be the midpoint 0.5
    assert np.isclose(result[0], 0.0 + 1.0 * 0.5)
    # For the second point (x=K=20), result should be the midpoint 0.1 + 0.9*0.5
    assert np.isclose(result[1], 0.1 + 0.9 * 0.5)