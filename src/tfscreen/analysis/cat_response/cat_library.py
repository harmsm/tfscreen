"""
A library of empirical mathematical functions for fitting x-y data. The core
data structure defined here is MODEL_LIBRARY, which keys model names to the
function used in the model, the guess function, the names of all parameters, 
and the bounds to use when doing the fit. 
"""

import numpy as np

from .cat_models import (
    model_flat,
    model_linear,
    model_hill_3p,
    model_hill_4p,
    model_bell,
    model_biphasic_peak,
    model_biphasic_dip
)

from .cat_guesses import (
    guess_flat,
    guess_linear,
    guess_repressor,
    guess_inducer,
    guess_hill_repressor,
    guess_hill_inducer,
    guess_bell_peak,
    guess_bell_dip,
    guess_biphasic_peak,
    guess_biphasic_dip
)

# --- Updated MODEL_LIBRARY ---
inf = np.inf
MODEL_LIBRARY = {

    # flat model (no slope)
    "flat": {"model_func":model_flat,
             "guess_func":guess_flat,
             "param_names":['baseline'],
             "bounds":([-inf], [inf])},

    # linear model
    "linear": {"model_func":model_linear,
               "guess_func":guess_linear,
               "param_names":['m', 'b'],
               "bounds":([-inf, -inf],
                         [ inf, inf])},

    # 3 point hill model with negative amplitude
    "repressor": {"model_func":model_hill_3p,
                  "guess_func":guess_repressor,
                  "param_names":['baseline', 'amplitude', 'lnK'],
                  "bounds":([-inf, -inf, -inf],
                            [ inf,    0,  inf])},

    # 3 point hill model with positive amplitude
    "inducer": {"model_func":model_hill_3p,
                "guess_func":guess_inducer,
                "param_names":['baseline', 'amplitude', 'lnK'],
                "bounds":([-inf,   0, -inf],
                          [ inf, inf,  inf])},

    # 4 point hill model with negative amplitude
    "hill_repressor": {"model_func":model_hill_4p,
                       "guess_func":guess_hill_repressor,
                       "param_names":['baseline', 'amplitude', 'lnK', 'n'],
                       "bounds":([-inf, -inf, -inf,   0],
                                 [ inf,    0,  inf, inf])},
    
    # 4 point hill model with positive amplitude
    "hill_inducer": {"model_func":model_hill_4p,
                     "guess_func":guess_hill_inducer,
                     "param_names":['baseline', 'amplitude', 'lnK', 'n'],
                     "bounds":([-inf,   0, -inf,   0],
                               [ inf, inf,  inf, inf])},

    # gaussian with positive amplitude
    "bell_peak": {"model_func":model_bell,
                  "guess_func":guess_bell_peak,
                  "param_names":['baseline', 'amplitude', 'ln_x0', 'ln_width'],
                  "bounds":([-inf,   0, -inf, -inf],
                            [ inf, inf,  inf,  inf])},

    # gaussian with negative amplitude
    "bell_dip": {"model_func":model_bell,
                 "guess_func":guess_bell_dip,
                 "param_names":['baseline', 'amplitude', 'ln_x0', 'ln_width'],
                 "bounds":([-inf, -inf, -inf, -inf],
                           [ inf,    0,  inf,  inf])}, 

    # sequential processes
    "biphasic_peak": {"model_func":model_biphasic_peak,
                      "guess_func":guess_biphasic_peak,
                      "param_names":['baseline', 'amplitude', 'lnK_a', 'lnK_i'],
                      "bounds":([-inf,   0, -inf, -inf],
                                [ inf, inf,  inf,  inf])},

    # parallel competing processes
    "biphasic_dip": {"model_func":model_biphasic_dip,
                     "guess_func":guess_biphasic_dip,
                     "param_names":['baseline', 'amplitude', 'lnK_dip', 'lnK_rise'],
                     "bounds":([  0,   0, -inf, -inf],
                               [inf, inf,  inf,  inf])},
}
