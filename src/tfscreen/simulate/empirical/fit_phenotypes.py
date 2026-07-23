"""
Backward-compatible shim: Stage 1 of the empirical-phenotype pipeline.

The per-genotype MLE fitter moved to
:mod:`tfscreen.tfmodel.genotype_fit.fit` (it is a general per-genotype
inference engine, also exposed standalone as ``tfs-fit-genotypes``).  This
module re-exports it so existing ``simulate.empirical.fit_phenotypes`` imports
keep working.
"""

from tfscreen.tfmodel.genotype_fit.fit import *  # noqa: F401,F403
from tfscreen.tfmodel.genotype_fit.fit import (  # noqa: F401
    _hill_theta,
    _natural_from_transformed,
    _build_calib_lookup,
    _lookup_km,
    _initial_guess_transformed,
    _forward,
    _Design,
    _POWER_CLIP,
    _LOGIT_BOUND,
    _EXP_CLIP,
    _ZERO_CONC_SENTINEL,
)
