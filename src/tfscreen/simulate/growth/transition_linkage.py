"""
Numpy implementations of the growth-transition components for simulation.

Each class corresponds to one inference-side growth_transition component in
growth_model/components/growth_transition/ and computes the total log-growth
(kt = Δln_cfu) for a pre-selection + selection experiment.

The Baranyi and two-population formulas are translated directly from
growth_model/components/growth_transition/_baranyi.py and two_pop.py,
replacing jnp with np.

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
        term1 = np.logaddexp(0.0, k_sharp * (t_sel - tau_lag))
        term0 = np.logaddexp(0.0, -k_sharp * tau_lag)
        integrated_sigmoid = (term1 - term0) / k_sharp
        dln_pre = k_pre * t_pre
        dln_sel = k_pre * t_sel + (k_sel - k_pre) * integrated_sigmoid
        return dln_pre + dln_sel


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
        k_pre = np.asarray(k_pre, dtype=float)
        k_sel = np.asarray(k_sel, dtype=float)
        t_pre = np.asarray(t_pre, dtype=float)
        t_sel = np.asarray(t_sel, dtype=float)

        D = k_pre - k_sel - k_trans
        valid = D > 0
        # Replace invalid D with 1.0 before any log so log(safe_D) stays finite
        # even for the invalid branch (those results are discarded by np.where).
        safe_D = np.where(valid, D, np.ones_like(D))

        a = k_sel * t_sel
        b = (k_pre - k_trans) * t_sel
        m = np.maximum(a, b)
        scaled_num = k_trans * np.exp(a - m) + safe_D * np.exp(b - m)

        two_pop_sel = m + np.log(scaled_num) - np.log(safe_D)
        fallback_sel = k_pre * t_sel   # k_trans → 0 limit: no transition

        dln_sel = np.where(valid, two_pop_sel, fallback_sel)
        dln_pre = k_pre * t_pre
        return dln_pre + dln_sel


TRANSITION_REGISTRY = {
    "instant":  InstantTransition,
    "memory":   MemoryTransition,
    "baranyi":  BaranyiTransition,
    "two_pop":  TwoPopTransition,
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
