"""
Backward-compatible shim: Stage 1.5 of the empirical-phenotype pipeline.

The congression de-attenuation moved to
:mod:`tfscreen.tfmodel.genotype_fit.congression` (it is reused by the standalone
``tfs-fit-genotypes`` command).  This module re-exports it so existing
``simulate.empirical.congression`` imports keep working.
"""

from tfscreen.tfmodel.genotype_fit.congression import *  # noqa: F401,F403
from tfscreen.tfmodel.genotype_fit.congression import (  # noqa: F401
    correct_theta_matrix,
    deattenuate_congression,
    _theta_from_fit,
    _refit_hill_theta,
    _THETA_PARAM_IDX,
    _THETA_EPS,
)
