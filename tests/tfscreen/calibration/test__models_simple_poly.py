import numpy as np
from tfscreen.calibration._models import simple_poly

def test_simple_poly_1d_params():
    """Tests a single polynomial with a 1D parameter array."""
    # Represents y = 3 + 2*x + 1*x^2
    params = np.array([3, 2, 1])
    x = np.array([0, 1, 2])

    result = simple_poly(params, x)

    # y(0) = 3
    # y(1) = 3 + 2 + 1 = 6
    # y(2) = 3 + 4 + 4 = 11
    expected = np.array([3, 6, 11])
    np.testing.assert_allclose(result, expected)

def test_simple_poly_2d_params():
    """Tests vectorized evaluation with a 2D parameter array."""
    # Poly 1 (for x=2): y = 3 + 2*x + 1*x^2
    # Poly 2 (for x=3): y = 1 + 1*x + 0*x^2
    params = np.array([
        [3, 1], # c0
        [2, 1], # c1
        [1, 0]  # c2
    ])
    x = np.array([2, 3])

    result = simple_poly(params, x)
    
    # y1(2) = 3 + 2*2 + 1*4 = 11
    # y2(3) = 1 + 1*3 + 0*9 = 4
    expected = np.array([11, 4])
    np.testing.assert_allclose(result, expected)