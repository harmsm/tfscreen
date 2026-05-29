"""
Shared growth computation for the Baranyi family of growth_transition components.

All three components (baranyi, baranyi_k, baranyi_tau) use the same integrated-sigmoid
growth formula.  The instantaneous growth rate during selection is:

    r(t) = g_pre + (g_sel - g_pre) · expit(k · (t - tau))

Integrating from 0 to t_sel gives:

    ∫₀^{t_sel} expit(k·(t - tau)) dt
        = (1/k) · [logaddexp(0, k·(t_sel - tau)) - logaddexp(0, -k·tau)]

So:

    dln_cfu_sel = g_pre·t_sel + (g_sel - g_pre) · integrated_sigmoid
    total       = g_pre·t_pre + dln_cfu_sel

The three components differ only in how they compute the effective (tau, k) pair
from their sampled per-condition parameters and the observed rate difference
|g_sel - g_pre|:

    baranyi   — tau and k sampled directly per condition (no rate-diff coupling)
    baranyi_k — k  modulated: k   = k0 / (1 + gamma · |Δg|)
    baranyi_tau — tau modulated: tau = tau_0 + k0 · |Δg|
"""

import jax.numpy as jnp


def compute_growth(g_pre, g_sel, t_pre, t_sel, tau, k):
    """
    Compute total growth using the Baranyi integrated-sigmoid transition.

    Parameters
    ----------
    g_pre, g_sel : jnp.ndarray
        Pre- and selection-phase growth rate tensors.
    t_pre, t_sel : jnp.ndarray
        Pre- and selection-phase time tensors.
    tau : jnp.ndarray
        Midpoint of the sigmoid transition (same shape as t_sel).
    k : jnp.ndarray
        Sharpness of the sigmoid transition (same shape as t_sel, must be > 0).

    Returns
    -------
    jnp.ndarray
        Total growth: dln_cfu_pre + dln_cfu_sel.
    """
    term1 = jnp.logaddexp(0.0, k * (t_sel - tau))
    term0 = jnp.logaddexp(0.0, -k * tau)
    integrated_sigmoid = (term1 - term0) / k

    dln_cfu_pre = g_pre * t_pre
    dln_cfu_sel = g_pre * t_sel + (g_sel - g_pre) * integrated_sigmoid

    return dln_cfu_pre + dln_cfu_sel
