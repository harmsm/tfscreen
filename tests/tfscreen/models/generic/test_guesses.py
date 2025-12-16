
import pytest
import numpy as np

from tfscreen.models.generic import guesses

@pytest.mark.parametrize("guess_func, expected_cols", [
    (guesses.guess_flat, 1),
    (guesses.guess_linear, 2),
    (guesses.guess_poly_2nd, 3),
    (guesses.guess_poly_3rd, 4),
    (guesses.guess_poly_4th, 5),
    (guesses.guess_poly_5th, 6),
])
def test_polynomial_guesses(guess_func, expected_cols):
    x = np.linspace(0, 10, 10)
    y = np.ones_like(x)
    
    design_matrix = guess_func(x, y)
    assert design_matrix.shape == (10, expected_cols)
    assert np.all(design_matrix[:, 0] == 1) # Intercept column is first (reversed vander)

def test_guess_biphasic_dip_edge_case():
    # Dip at the end
    x = np.array([0, 1, 2])
    y = np.array([1, 0.5, 0.0]) # Dip at 2
    guess = guesses.guess_biphasic_dip(x, y)
    assert len(guess) == 4


def test_repressor_and_inducer_guesses():
    x = np.array([0, 1, 10, 100])
    
    # Repressor-like data: high at 0, low at 100
    y_rep = np.array([10, 8, 2, 0])
    guess = guesses.guess_repressor(x, y_rep)
    assert len(guess) == 3
    # Baseline ~ 0 (min y)
    # Amplitude ~ 10 (max - min)
    # lnK: midpoint is 5. closest x is 1 or 10.
    
    # Hill repressor
    guess_h = guesses.guess_hill_repressor(x, y_rep)
    assert len(guess_h) == 4
    assert guess_h[3] == 2.0
    
    # Inducer-like data: low at 0, high at 100
    y_ind = np.array([0, 2, 8, 10])
    guess = guesses.guess_inducer(x, y_ind)
    assert len(guess) == 3
    
    # Hill inducer
    guess_h = guesses.guess_hill_inducer(x, y_ind)
    assert len(guess_h) == 4
    assert guess_h[3] == 2.0

def test_peak_and_dip_guesses():
    x = np.linspace(-5, 5, 20)
    
    # Bell peak
    y_peak = np.exp(-x**2)
    guess = guesses.guess_bell_peak(x, y_peak)
    # [baseline, amplitude, ln_x0, ln_width]
    assert len(guess) == 4
    assert np.isclose(guess[0], 0, atol=0.1) # Baseline
    assert np.isclose(guess[1], 1, atol=0.1) # Amplitude
    
    # Bell dip
    y_dip = -np.exp(-x**2) + 1
    guess = guesses.guess_bell_dip(x, y_dip)
    assert len(guess) == 4
    assert np.isclose(guess[0], 1, atol=0.1) # Baseline
    assert np.isclose(guess[1], -1, atol=0.1) # Amplitude

    # Biphasic peak (Rise then Fall)
    # Just check it runs and produces valid shape
    guess = guesses.guess_biphasic_peak(x, y_peak)
    assert len(guess) == 4
    
    # Biphasic dip
    guess = guesses.guess_biphasic_dip(x, y_dip)
    assert len(guess) == 4
