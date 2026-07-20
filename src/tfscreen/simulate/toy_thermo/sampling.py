"""
Draw random mutation catalogs from per-state Normal distributions.

Instead of specifying every ddG by hand, give a mean/sd per state for main
effects (and optionally epistasis); each mutation's per-state effect is an
independent Normal draw. Produces a MutationEffects for the genotype/DataFrame
machinery in genotypes.py.
"""

from itertools import combinations

import numpy as np

from .genotypes import MutationEffects, STATES


def _draw(rng, spec):
    """spec: {state: sd} or {state: (mean, sd)} -> {state: value}."""
    out = {}
    for s in STATES:
        v = spec.get(s, 0.0) if spec else 0.0
        mean, sd = (0.0, float(v)) if np.isscalar(v) else (float(v[0]), float(v[1]))
        out[s] = rng.normal(mean, sd) if sd > 0 else mean
    return out


def sample_effects(mutations, effect_sd, epistasis_sd=None,
                   pairs=None, rng=None):
    """
    Build a random MutationEffects.

    Parameters
    ----------
    mutations : sequence of str
        Mutation names to draw.
    effect_sd : dict
        Per-state main-effect spec. Each value is an sd (mean 0) or a
        (mean, sd) tuple, e.g. {'HD': 1.0, 'LE': (0.0, 2.0)}.
    epistasis_sd : dict, optional
        Same form, for per-pair in-state epistasis. Omit for no epistasis.
    pairs : iterable of (str, str), optional
        Which pairs get epistasis terms (default: all pairs of ``mutations``).
    rng : numpy.random.Generator, optional

    Returns
    -------
    MutationEffects
    """
    rng = rng if rng is not None else np.random.default_rng()
    eff = MutationEffects()
    for m in mutations:
        eff.add_mutation(m, **_draw(rng, effect_sd))
    if epistasis_sd:
        pairs = combinations(mutations, 2) if pairs is None else pairs
        for a, b in pairs:
            eff.add_epistasis(a, b, **_draw(rng, epistasis_sd))
    return eff
