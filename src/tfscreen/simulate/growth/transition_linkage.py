"""
Growth-transition components for simulation.

Each class wraps one inference-side growth_transition component from
``tfscreen.tfmodel.generative.components.growth_transition`` and delegates
all arithmetic to that shared implementation.

``BaranyiTransition`` and ``TwoPopTransition`` call the tfmodel deterministic
functions directly (``_baranyi.compute_growth`` and
``two_pop._compute_growth``), ensuring that the simulate and inference paths
use identical maths.  ``InstantTransition`` and ``MemoryTransition`` are
simple enough to implement inline or have no shared tfmodel core.

The growth_transition list in a simulation config selects one of these
classes per condition_pre:

    growth_transition:
      - condition_pre: "M9"
        model: instant
      - condition_pre: "M9+kan"
        model: baranyi
        tau_lag: 120.0
        k_sharp: 0.1
      - condition_pre: "M9+pheS"
        model: two_pop
        k_trans: 0.002
"""

import numpy as np
from tfscreen.tfmodel.generative.components.growth_transition._baranyi import (
    compute_growth as _baranyi_compute_growth,
)
from tfscreen.tfmodel.generative.components.growth_transition.two_pop import (
    _compute_growth as _two_pop_compute_growth,
)


class InstantTransition:
    """
    Instantaneous switch at t_sel=0.

        kt = k_pre * t_pre + k_sel * t_sel

    Matches the 'instant' inference component.
    """
    def compute_kt(self, k_pre, k_sel, t_pre, t_sel, theta=None):
        return k_pre * t_pre + k_sel * t_sel


class MemoryTransition:
    """
    Piecewise transition with theta-dependent lag time:

        tau = tau0 + k1 / (theta + k2)

    For t_sel < tau: bacteria grow at k_pre the entire selection phase.
    For t_sel >= tau: bacteria switch to k_sel after time tau.

    Matches the 'memory' inference component.

    Parameters
    ----------
    tau0 : float
        Baseline lag time (minutes).
    k1 : float
        Coefficient scaling the theta-dependence of tau.
    k2 : float
        Offset preventing division by zero; must be > 0.
    """
    def compute_kt(self, k_pre, k_sel, t_pre, t_sel, theta, tau0, k1, k2):
        tau = tau0 + k1 / (np.asarray(theta) + k2)
        dln_pre = k_pre * t_pre
        dln_sel = np.where(
            t_sel < tau,
            k_pre * t_sel,
            k_pre * tau + k_sel * (t_sel - tau),
        )
        return dln_pre + dln_sel


class BaranyiTransition:
    """
    Smooth sigmoid transition, integrated analytically.

    The instantaneous growth rate during selection is:

        r(t) = k_pre + (k_sel - k_pre) * expit(k_sharp * (t - tau_lag))

    Integrating from 0 to t_sel:

        dln_cfu_sel = k_pre * t_sel
                      + (k_sel - k_pre)
                        * [logaddexp(0, k_sharp*(t_sel - tau_lag))
                           - logaddexp(0, -k_sharp*tau_lag)] / k_sharp

        kt = k_pre * t_pre + dln_cfu_sel

    Matches the 'baranyi' inference component via
    growth_model/components/growth_transition/_baranyi.py.

    Parameters
    ----------
    tau_lag : float
        Midpoint of the sigmoid transition (minutes).
    k_sharp : float
        Sharpness of the transition (must be > 0).
    """
    def compute_kt(self, k_pre, k_sel, t_pre, t_sel, theta=None,
                   tau_lag=100.0, k_sharp=1.0):
        return np.array(_baranyi_compute_growth(k_pre, k_sel, t_pre, t_sel,
                                                tau=tau_lag, k=k_sharp))


class BaranyiKTransition:
    """
    Baranyi integrated-sigmoid transition with sharpness modulated by rate difference.

    The sigmoid sharpness decreases as the growth-rate difference grows, so a
    larger switch demands a slower (more inertial) transition:

        k = k0 / (1 + gamma * |k_sel - k_pre|)

    Total growth:

        dln_cfu_sel = k_pre * t_sel
                      + (k_sel - k_pre)
                        * [logaddexp(0, k*(t_sel - tau)) - logaddexp(0, -k*tau)] / k

        kt = k_pre * t_pre + dln_cfu_sel

    Matches the 'baranyi_k' inference component via
    growth_model/components/growth_transition/baranyi_k.py.

    Parameters
    ----------
    tau : float
        Midpoint of the sigmoid transition (minutes).
    k0 : float
        Base sigmoid sharpness when growth rates are identical (must be > 0).
    gamma : float
        Coefficient scaling the rate-difference suppression of sharpness (must be >= 0).
    """
    def compute_kt(self, k_pre, k_sel, t_pre, t_sel, theta=None,
                   tau=100.0, k0=1.0, gamma=1.0):
        delta_g = np.abs(np.asarray(k_sel) - np.asarray(k_pre))
        k = k0 / (1.0 + gamma * delta_g)
        return np.array(_baranyi_compute_growth(k_pre, k_sel, t_pre, t_sel,
                                                tau=tau, k=k))


class BaranyiTauTransition:
    """
    Baranyi integrated-sigmoid transition with midpoint delayed by rate difference.

    The transition midpoint is pushed later as the growth-rate difference grows,
    so a larger switch takes longer to begin:

        tau = tau_0 + k0 * |k_sel - k_pre|

    Total growth:

        dln_cfu_sel = k_pre * t_sel
                      + (k_sel - k_pre)
                        * [logaddexp(0, k*(t_sel - tau)) - logaddexp(0, -k*tau)] / k

        kt = k_pre * t_pre + dln_cfu_sel

    Matches the 'baranyi_tau' inference component via
    growth_model/components/growth_transition/baranyi_tau.py.

    Parameters
    ----------
    tau_0 : float
        Base transition midpoint (minutes).
    k0 : float
        Coefficient scaling how much the rate difference delays the midpoint.
    k : float
        Fixed sigmoid sharpness (must be > 0).
    """
    def compute_kt(self, k_pre, k_sel, t_pre, t_sel, theta=None,
                   tau_0=100.0, k0=1.0, k=1.0):
        delta_g = np.abs(np.asarray(k_sel) - np.asarray(k_pre))
        tau = tau_0 + k0 * delta_g
        return np.array(_baranyi_compute_growth(k_pre, k_sel, t_pre, t_sel,
                                                tau=tau, k=k))


class TwoPopTransition:
    """
    Two-population ODE transition model.

    Two sub-populations start at t_sel=0 growing at rate k_pre.  Cells
    transition irreversibly to rate k_sel at rate k_trans:

        N_tot(t) = N_0 * [k_trans * exp(k_sel*t)
                           + D * exp((k_pre - k_trans)*t)] / D

    where D = k_pre - k_sel - k_trans.  Valid when D > 0; falls back to
    k_pre growth when D <= 0.

    Matches the 'two_pop' inference component via
    growth_model/components/growth_transition/two_pop.py.

    Parameters
    ----------
    k_trans : float
        Transition rate from pre-selection to selection growth (must be > 0).
    """
    def compute_kt(self, k_pre, k_sel, t_pre, t_sel, theta=None, k_trans=1e-6):
        return np.array(_two_pop_compute_growth(k_pre, k_sel, t_pre, t_sel,
                                                k_trans=k_trans))


TRANSITION_REGISTRY = {
    "instant":    InstantTransition,
    "memory":     MemoryTransition,
    "baranyi":    BaranyiTransition,
    "baranyi_k":  BaranyiKTransition,
    "baranyi_tau": BaranyiTauTransition,
    "two_pop":    TwoPopTransition,
}


def get_transition_model(name):
    """Return an instantiated transition model by name.

    Parameters
    ----------
    name : str
        One of the keys in TRANSITION_REGISTRY.

    Returns
    -------
    InstantTransition | MemoryTransition | BaranyiTransition | TwoPopTransition

    Raises
    ------
    ValueError
        If name is not a known model.
    """
    if name not in TRANSITION_REGISTRY:
        raise ValueError(
            f"Unknown transition model '{name}'. "
            f"Available models: {sorted(TRANSITION_REGISTRY)}"
        )
    return TRANSITION_REGISTRY[name]()
