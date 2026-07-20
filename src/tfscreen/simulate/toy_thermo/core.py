"""
Self-contained equilibrium solver for a four-state monomer TF system.

    HD  <->  H + D  <->  L + D  <->  LE + D

    H, L    : two protein conformations in intrinsic equilibrium
    D        : DNA (operator); H binds D to form HD
    E        : effector; L binds E to form LE
    observable = HD / (HD + D)   (fraction of DNA bound)

Three association constants define the wild-type system:

    K_conf = [H]/[L]              (dimensionless)
    K_dna  = [HD]/([H][D])        (1 / concentration)
    K_eff  = [LE]/([L][E])        (1 / concentration)

Given total protein, DNA, and effector, the free concentrations satisfy mass
balance.  Substituting the closed-form expressions for free DNA and free
effector into protein conservation leaves a single monotonic equation in free
[L] on (0, protein_total], solved here with Brent's method.  All concentrations
must share the same unit; K_dna and K_eff carry its inverse.
"""

import numpy as np
from scipy.optimize import brentq


def _solve_free_L(Kc, Kd, Ke, protein_total, dna_total, effector_total):
    """Solve protein conservation for free [L]. Returns a scalar."""
    KcKd = Kc * Kd

    def conservation(L):
        D_free = dna_total / (1.0 + KcKd * L)
        E_free = effector_total / (1.0 + Ke * L)
        # [L] + [H] + [HD] + [LE] - protein_total
        return L * (1.0 + Kc + KcKd * D_free + Ke * E_free) - protein_total

    # conservation(0) = -protein_total < 0; conservation(protein_total) > 0.
    return brentq(conservation, 0.0, protein_total, xtol=1e-30, rtol=8.9e-16)


def fraction_bound(Kc, Kd, Ke, protein_total, dna_total, effector_total):
    """
    Fraction of DNA bound, HD / (HD + D), for the four-state system.

    Parameters
    ----------
    Kc, Kd, Ke : float
        Association constants K_conf, K_dna, K_eff (see module docstring).
    protein_total, dna_total : float
        Total protein and DNA concentrations (same unit).
    effector_total : float or array_like
        Total effector concentration(s). Scalar in -> scalar out; array in ->
        array out (one observable per effector concentration).

    Returns
    -------
    float or numpy.ndarray
        Observable HD / (HD + D) in [0, 1].
    """
    eff = np.atleast_1d(np.asarray(effector_total, dtype=float))
    theta = np.empty(eff.shape, dtype=float)
    for i, E in enumerate(eff):
        L = _solve_free_L(Kc, Kd, Ke, protein_total, dna_total, E)
        w = Kc * Kd * L                      # [HD]/[D]
        theta[i] = w / (1.0 + w)
    return theta if np.ndim(effector_total) else float(theta[0])


def solve_species(Kc, Kd, Ke, protein_total, dna_total, effector_total):
    """
    Return every free/bound species concentration for one effector value.

    Handy for teaching/plotting the full occupancy breakdown. Returns a dict
    with keys L, H, HD, LE, D, E. Scalar effector only.
    """
    L = _solve_free_L(Kc, Kd, Ke, protein_total, dna_total, effector_total)
    D = dna_total / (1.0 + Kc * Kd * L)
    E = effector_total / (1.0 + Ke * L)
    return {"L": L, "H": Kc * L, "HD": Kc * Kd * L * D, "LE": Ke * L * E,
            "D": D, "E": E}
