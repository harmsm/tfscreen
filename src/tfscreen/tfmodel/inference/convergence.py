"""
Convergence detection for the SVI/MAP optimization loop.

Optimization runs in tumbling *windows* of ``window_steps`` optimizer steps.
At the end of each window two tests decide whether the fit is still moving:

**Loss test.**  The window's per-step losses are split into
``loss_blocks`` blocks.  A least-squares line through the block medians gives
the decrease in loss projected over one window (``drop``) and its standard
error (``drop_se``, from the scatter of the block medians about the line).
The loss is *improving* when ``drop`` exceeds both ``z * drop_se`` (the
decrease is larger than the loss's own noise) and ``loss_rtol * |level|`` (a
floor for deterministic losses, whose noise is ~0).  A loss that rises, or
oscillates without net decrease, is not improving: that is the signature of
a step size too large for the current stage, which a cut fixes.  Block
medians keep the right-skewed noise of a stochastic ELBO from biasing the
trend, and blocks of many steps make the standard error robust to
short-range autocorrelation.

**Loss skew.**  The loss being minimized is the *mean* of the per-step
losses, which the block medians can hide: an ELBO estimate that is usually
near its typical value but occasionally carries a huge penalty (a rare guide
draw that violates a razor-sharp likelihood) has a median that ignores the
penalties however much of the mean they make up.  The window's ``skew`` is
``(mean - median) / (1.4826 * MAD)`` of the losses after removing the fitted
trend line -- how many robust standard deviations the mean sits above the
median.  It is roughly the penalized fraction of steps times the penalty, in
per-step noise SDs.  Benign Monte Carlo noise, including the right skew of an
ELBO estimate, stays below ~1; penalties that move the mean by more than
``max_loss_skew`` noise SDs are flagged (on the congression-calibration runs,
5-30% of steps carrying ~200 noise SDs gave 20-40).  The statistic only sees
penalties that are a minority of steps: once most steps carry one, the median
(and the robust spread) follow them, but then the loss level itself shows the
problem.

**Parameter test.**  Each tracked parameter is averaged over ``param_blocks``
blocks of the window.  The same line fit through the block means gives each
element's projected movement over one window and its standard error, both
divided by a *normalizer*: the parameter's posterior SD (its guide scale) for
a variational location, the prior SD for a MAP location, and 1 for a
parameter already in log or logit units (positive or bounded parameters, so
the movement is relative).  A posterior SD is floored at ``PRIOR_SD_FLOOR``
times the site's prior SD (both in the location's own, unconstrained units):
a mean-field guide can collapse a scale far below any honest posterior width
(a hierarchical scale's guide SD reaching 1e-4 of its prior SD), and in those
units a location creeping at a negligible absolute rate reads as moving
indefinitely.  An element's *excess* is the part of its movement
not explained by noise or by the optimizer's resolution,
``max(|drift| - z * se - floor, 0)``, where ``floor`` is
``MIN_STEP_COHERENCE * step_size * window_steps`` (normalized the same way).
Adam moves a parameter by about the step size per step whatever the gradient's
magnitude, so a parameter still descending covers a sizeable fraction of
``step_size * window_steps`` per window, while one jittering about its optimum
covers only a few step sizes; the floor keeps that jitter -- which can be many
posterior SDs when a guide scale is smaller than the step size -- from reading
as drift.  Movement below the floor means a gradient signal-to-noise ratio
under ``MIN_STEP_COHERENCE``, which the current step size cannot resolve.  A
parameter array is summarized by its maximum excess (small arrays) or a high
quantile of it (per-genotype arrays, where a handful of elements always sit in
the noise tail).  The parameters are *moving* when any summary exceeds
``param_tolerance``.

**Decisions.**  The two tests play different roles:

- A window in which the loss is not improving is a *stall*.  After
  ``patience`` consecutive stalls the step size is cut (by ``step_size_cut``,
  floored at ``final_step_size``), whatever the parameters are doing.  This is
  the usual reduce-on-plateau rule: the loss alone decides when the current
  step size has done what it can.  Parameters do not block a cut because some
  directions never settle under a given objective (a MAP of a horseshoe local
  scale drifts toward zero forever, at almost no change in loss), and holding
  the step size high for them helps nothing.
- Once the step size is at its floor, a window that is a stall, has no
  moving parameter *and* has a loss skew within ``max_loss_skew`` is a
  *plateau*, and ``patience`` consecutive plateaus are convergence.  A
  parameter that is still moving therefore keeps the run going (and is named
  in the log) until it stops or ``max_num_epochs`` is reached, so a slow slide
  or a degenerate direction is reported, never mistaken for convergence; so
  does a skewed loss, whose median plateau says nothing about the objective.
  Skew never forces a cut: a smaller step size does not remove rare
  penalties.

Because a parameter's systematic drift and its step-to-step jitter both scale
with the step size, the significance part of the tests does not depend on the
step size; each cut lowers the noise floor so the next stage resolves the
estimates more finely.

Everything here is host-side numpy (``param_drift`` also works on JAX arrays),
so the logic can be tested on synthetic traces without a model.
"""

import numpy as np

# Movement slower than this fraction of the step size per step is below the
# optimizer's resolution at that step size (see the module docstring).
MIN_STEP_COHERENCE = 0.01

# Arrays with at most this many elements are summarized by their maximum
# excess; larger (per-genotype, per-mutation) arrays by a quantile.
SMALL_ARRAY_SIZE = 100

# A posterior SD used as a normalizer is floored at this fraction of the
# site's prior SD (see the module docstring).
PRIOR_SD_FLOOR = 0.01

# Largest loss skew, (mean - median) / robust SD, of a window that can count
# as a plateau at the floor step size.  Clean traces (recorded real-data and
# simulated SVI, MAP, synthetic exponential/lognormal noise) stay below ~1;
# windows with rare large penalties reach 17-150.
MAX_LOSS_SKEW = 3.0

# Convergence decisions returned by ConvergenceMonitor.end_window.
CONTINUE = "continue"
CUT = "cut"
CONVERGED = "converged"


def _line_fit(y, t):
    """
    Least-squares slope of ``y`` against ``t`` along axis 0, and its SE.

    Parameters
    ----------
    y : array, shape (n, ...)
        Values at each of ``n`` time points.
    t : array, shape (n,)
        Time points.

    Returns
    -------
    slope, se : arrays with the trailing shape of ``y``
    """

    n = t.shape[0]
    tc = t - t.mean()
    sxx = float(np.sum(tc * tc))
    tc = tc.reshape((n,) + (1,) * (y.ndim - 1))
    ybar = y.mean(axis=0)
    slope = (tc * (y - ybar)).sum(axis=0) / sxx
    resid = y - ybar - slope * tc
    s2 = (resid * resid).sum(axis=0) / (n - 2)
    se = (s2 / sxx) ** 0.5
    return slope, se


def loss_trend(losses, num_blocks=20):
    """
    Projected change in loss over a window, with its standard error.

    Parameters
    ----------
    losses : 1-D array
        Per-step losses of one window, in step order.
    num_blocks : int, optional
        Number of blocks the window is split into (default 20).  Trailing
        steps that do not fill a block are dropped.

    Returns
    -------
    dict
        ``level`` (median of the last block), ``mean`` (mean of the last
        block), ``drop`` (loss decrease over the window predicted by the
        fitted line; positive when the loss falls), ``drop_se``, ``t``
        (``drop / drop_se``; ``inf`` when the block medians lie exactly on the
        line) and ``skew`` (``loss_skew`` of the window about the line).
    """

    losses = np.asarray(losses, dtype=float).ravel()
    num_blocks = min(num_blocks, losses.size)
    if num_blocks < 3:
        raise ValueError(
            f"loss_trend needs at least 3 losses (got {losses.size})."
        )
    block = losses.size // num_blocks
    used = losses[losses.size - num_blocks * block:]
    blocks = used.reshape(num_blocks, block)
    medians = np.median(blocks, axis=1)
    centers = (np.arange(num_blocks) + 0.5) * block

    slope, se = _line_fit(medians, centers)
    width = float(used.size)
    drop = -float(slope) * width
    drop_se = float(se) * width
    if drop_se > 0:
        t = drop / drop_se
    else:
        t = np.inf if drop != 0 else 0.0

    # Residuals about the fitted line, so a falling loss is not read as skew.
    line = (medians.mean()
            + float(slope) * (np.arange(used.size) + 0.5 - centers.mean()))

    return {"level": float(medians[-1]),
            "mean": float(blocks[-1].mean()),
            "drop": drop,
            "drop_se": drop_se,
            "t": float(t),
            "skew": loss_skew(used - line, level=medians[-1])}


def loss_skew(residuals, level=0.0, rtol=1e-6):
    """
    How far the mean of ``residuals`` sits above their median, in robust SDs.

    Parameters
    ----------
    residuals : 1-D array
        Per-step losses with any trend removed.
    level : float, optional
        Typical loss; ``rtol * |level|`` floors the robust SD so an exactly
        constant (deterministic) loss has skew 0 rather than 0/0.
    rtol : float, optional
        Relative floor on the robust SD (default 1e-6, the default
        ``loss_rtol``: differences below it are not resolved anyway).

    Returns
    -------
    float
        ``(mean - median) / max(1.4826 * MAD, rtol * |level|)``; 0 when the
        spread and the floor are both zero.
    """

    r = np.asarray(residuals, dtype=float).ravel()
    if r.size == 0:
        return 0.0
    med = np.median(r)
    sd = max(1.4826 * float(np.median(np.abs(r - med))),
             rtol * abs(float(level)))
    gap = float(np.mean(r) - med)
    if sd <= 0:
        return 0.0
    return gap / sd


def param_drift(block_means, block_centers, window_steps, normalizer, z,
                floor=0.0):
    """
    Per-element movement of a parameter over one window, in normalizer units.

    Parameters
    ----------
    block_means : array, shape (num_blocks, ...)
        The parameter averaged over each block of the window (unconstrained
        space).
    block_centers : array, shape (num_blocks,)
        Step at the center of each block.
    window_steps : int
        Window length; the fitted slope is projected over this many steps.
    normalizer : array or float
        Divides the movement: posterior SD, prior SD, or 1.  Broadcast against
        the parameter's shape.
    z : float
        Number of standard errors of movement attributed to noise.
    floor : float, optional
        Movement (in parameter units, before normalizing) attributed to the
        optimizer's resolution (default 0).

    Returns
    -------
    drift, excess : arrays with the parameter's shape
        ``drift`` is the projected movement (signed); ``excess`` is
        ``max(|drift| - z * se - floor, 0)``, all in normalizer units.
    """

    # Works for numpy and jax arrays alike.
    xp = np
    if type(block_means).__module__.startswith("jax"):
        import jax.numpy as xp

    t = np.asarray(block_centers, dtype=float)
    slope, se = _line_fit(block_means, t)
    drift = slope * window_steps / normalizer
    se = se * window_steps / normalizer
    excess = xp.maximum(xp.abs(drift) - z * se - floor / normalizer, 0.0)
    return drift, excess


def summarize_excess(excess, quantile=0.99):
    """
    One number per parameter array: max excess for small arrays, else a
    quantile.  Non-finite elements count as infinitely moving.
    """

    excess = np.asarray(excess, dtype=float).ravel()
    if excess.size == 0:
        return 0.0
    excess = np.where(np.isfinite(excess), excess, np.inf)
    if excess.size <= SMALL_ARRAY_SIZE:
        return float(np.max(excess))
    return float(np.quantile(excess, quantile))


class ConvergenceMonitor:
    """
    Decide, window by window, whether to continue, cut the step size, or stop.

    Parameters
    ----------
    step_size : float
        Current (initial) optimizer step size.
    final_step_size : float or None, optional
        Floor for the step size.  None, or a value >= ``step_size``, disables
        cuts: the run is at its floor from the start.
    step_size_cut : float, optional
        Factor applied to the step size at each cut (default 0.1).
    patience : int, optional
        Consecutive windows without loss improvement required for a cut, and
        (at the floor) without loss improvement or parameter movement
        required for a stop (default 3).
    z : float, optional
        Standard errors of change attributed to noise (default 3).
    loss_rtol : float, optional
        Relative floor on a meaningful loss change per window (default 1e-6).
    param_tolerance : float, optional
        Largest allowed parameter excess per window, in normalizer units
        (default 0.05).
    max_loss_skew : float, optional
        Largest loss skew (``loss_skew``) of a window that can be a plateau at
        the floor step size (default ``MAX_LOSS_SKEW``).
    """

    def __init__(self,
                 step_size,
                 final_step_size=None,
                 step_size_cut=0.1,
                 patience=3,
                 z=3.0,
                 loss_rtol=1e-6,
                 param_tolerance=0.05,
                 max_loss_skew=MAX_LOSS_SKEW):

        if not 0 < step_size_cut < 1:
            raise ValueError(
                f"step_size_cut must be between 0 and 1 (got {step_size_cut})."
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1 (got {patience}).")

        self.step_size = float(step_size)
        if final_step_size is None:
            final_step_size = self.step_size
        self.final_step_size = float(final_step_size)
        self.step_size_cut = float(step_size_cut)
        self.patience = int(patience)
        self.z = float(z)
        self.loss_rtol = float(loss_rtol)
        self.param_tolerance = float(param_tolerance)
        self.max_loss_skew = float(max_loss_skew)

        self.plateau_count = 0
        self.num_cuts = 0
        self.converged = False
        self.last = None

    @property
    def at_floor(self):
        """True when no further step-size cut is possible."""
        return self.step_size <= self.final_step_size * (1 + 1e-9)

    def loss_improving(self, stats):
        """Whether a ``loss_trend`` result counts as a real improvement."""
        floor = self.loss_rtol * abs(stats["level"])
        return (stats["drop"] > self.z * stats["drop_se"]
                and stats["drop"] > floor)

    def drift_floor(self, window_steps):
        """Movement per window below the optimizer's resolution."""
        if not np.isfinite(self.step_size):
            return 0.0
        return MIN_STEP_COHERENCE * self.step_size * window_steps

    def end_window(self, step, loss_stats, param_excess=None, param_drift=None):
        """
        Record one finished window and return the decision.

        Parameters
        ----------
        step : int
            Optimizer step at the end of the window.
        loss_stats : dict
            Output of ``loss_trend`` for the window.
        param_excess : dict or None, optional
            ``{parameter name: summarized excess}`` (``summarize_excess``).
        param_drift : dict or None, optional
            ``{parameter name: summarized |drift|}``, reported only.

        Returns
        -------
        str
            ``CONTINUE``, ``CUT`` (the step size has been cut; the caller
            must apply ``self.step_size``) or ``CONVERGED``.
        """

        param_excess = dict(param_excess or {})
        param_drift = dict(param_drift or {})

        loss_improving = self.loss_improving(loss_stats)

        worst_param, worst_excess = None, 0.0
        for name, value in param_excess.items():
            if worst_param is None or value > worst_excess:
                worst_param, worst_excess = name, value
        params_moving = worst_excess > self.param_tolerance

        # A median plateau says nothing about the mean when rare penalties
        # dominate it (see the module docstring).
        loss_skew = float(loss_stats.get("skew", 0.0))
        loss_skewed = not loss_skew <= self.max_loss_skew

        # Cuts follow the loss alone; the stop also needs the parameters and
        # an unskewed loss.
        at_floor = self.at_floor
        if at_floor:
            plateau = not (loss_improving or params_moving or loss_skewed)
        else:
            plateau = not loss_improving
        self.plateau_count = self.plateau_count + 1 if plateau else 0

        decision = CONTINUE
        step_size = self.step_size
        if self.plateau_count >= self.patience:
            if at_floor:
                decision = CONVERGED
                self.converged = True
            else:
                decision = CUT
                self.step_size = max(self.step_size * self.step_size_cut,
                                     self.final_step_size)
                self.num_cuts += 1
                self.plateau_count = 0

        self.last = {
            "step": int(step),
            "step_size": step_size,
            "loss": loss_stats["level"],
            "loss_mean": float(loss_stats.get("mean", np.nan)),
            "loss_drop": loss_stats["drop"],
            "loss_drop_se": loss_stats["drop_se"],
            "loss_t": loss_stats["t"],
            "loss_improving": bool(loss_improving),
            "loss_skew": loss_skew,
            "loss_skewed": bool(loss_skewed),
            "worst_param": worst_param,
            "worst_param_excess": float(worst_excess),
            "worst_param_drift": float(param_drift.get(worst_param, np.nan))
                                 if worst_param is not None else np.nan,
            "params_moving": bool(params_moving),
            "plateau": bool(plateau),
            "plateau_count": self.plateau_count,
            "decision": decision,
        }
        return decision

    def describe(self):
        """One-line summary of the last window, for logs."""
        if self.last is None:
            return "no complete convergence window yet"
        r = self.last
        loss = (f"loss {r['loss']:.6g}, drop/window {r['loss_drop']:.4g} "
                f"+/- {r['loss_drop_se']:.3g} (t={r['loss_t']:.3g})")
        skew = r.get("loss_skew", 0.0)
        if r.get("loss_skewed"):
            loss += (f"; loss skewed ({skew:.3g} robust SDs of mean above "
                     f"median: rare large penalties)")
        if r["worst_param"] is None:
            params = "no parameters tracked"
        else:
            params = (f"largest parameter movement {r['worst_param']} "
                      f"(excess {r['worst_param_excess']:.3g}, "
                      f"drift {r['worst_param_drift']:.3g})")
        return (f"step {r['step']}: {loss}; {params}; step size "
                f"{r['step_size']:.3g}; plateau {r['plateau_count']}/"
                f"{self.patience}; {r['decision']}")

    def state_dict(self):
        """Serializable state for checkpoints."""
        return {"step_size": self.step_size,
                "final_step_size": self.final_step_size,
                "plateau_count": self.plateau_count,
                "num_cuts": self.num_cuts,
                "converged": self.converged,
                "last": self.last}

    def load_state_dict(self, state):
        """
        Restore the stage (step size, cuts, plateau count) from a checkpoint.

        The tolerances and floor stay as configured for the resumed run.
        """
        self.step_size = float(state["step_size"])
        self.plateau_count = int(state.get("plateau_count", 0))
        self.num_cuts = int(state.get("num_cuts", 0))
        self.converged = bool(state.get("converged", False))
        self.last = state.get("last")
