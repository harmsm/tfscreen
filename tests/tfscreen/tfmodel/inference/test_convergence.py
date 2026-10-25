"""
Tests for inference/convergence.py: the loss and parameter tests, and the
window-by-window decisions, on synthetic and recorded loss traces.
"""

import os

import numpy as np
import pytest
import jax.numpy as jnp

from tfscreen.tfmodel.inference import convergence as conv
from tfscreen.tfmodel.inference.convergence import (
    ConvergenceMonitor,
    loss_skew,
    loss_trend,
    param_drift,
    summarize_excess,
)

TRACES = os.path.join(os.path.dirname(__file__), "loss_traces")
WINDOW = 2000


def _windows(trace, window=WINDOW):
    """Consecutive full windows of a per-step loss trace."""
    for start in range(0, len(trace) - window + 1, window):
        yield start + window, trace[start:start + window]


def _feed(monitor, trace, window=WINDOW, param_excess=None):
    """Run a monitor over a trace (loss test only unless param_excess given)."""
    decisions = []
    for step, losses in _windows(trace, window):
        decisions.append(monitor.end_window(step, loss_trend(losses),
                                            param_excess))
        if decisions[-1] in (conv.CONVERGED, conv.DIVERGED):
            break
    return decisions


# -----------------------------------------------------------------------------
# loss_trend
# -----------------------------------------------------------------------------

def test_loss_trend_exact_line():
    losses = 1000.0 - 0.5 * np.arange(WINDOW)
    stats = loss_trend(losses)
    assert stats["drop"] == pytest.approx(0.5 * WINDOW, rel=1e-6)
    assert stats["t"] > 1e6 or np.isinf(stats["t"])
    # level: median of the last of 20 blocks
    assert stats["level"] == pytest.approx(np.median(losses[-100:]))


def test_loss_trend_rising_loss_has_negative_drop():
    stats = loss_trend(np.linspace(0, 100, WINDOW))
    assert stats["drop"] < 0


def test_loss_trend_constant_is_zero():
    stats = loss_trend(np.full(WINDOW, 5.0))
    assert stats["drop"] == 0
    assert stats["t"] == 0


def test_loss_trend_needs_three_losses():
    with pytest.raises(ValueError):
        loss_trend([1.0, 2.0])


def test_loss_trend_ignores_leading_partial_block():
    losses = np.concatenate([[1e9], np.full(2000, 3.0)])
    assert loss_trend(losses)["drop"] == 0


@pytest.mark.parametrize("noise", ["normal", "lognormal", "student_t"])
def test_loss_trend_flat_noise_rarely_looks_like_improvement(noise):
    """A flat loss with white noise -- symmetric, right-skewed or heavy
    tailed -- is called improving (t > 3) in only a few percent of windows."""
    rng = np.random.default_rng(0)
    false_calls = 0
    for _ in range(300):
        if noise == "normal":
            eps = rng.normal(size=WINDOW)
        elif noise == "lognormal":
            eps = rng.lognormal(sigma=1.0, size=WINDOW)
        else:
            eps = rng.standard_t(df=2, size=WINDOW)
        stats = loss_trend(1e5 + 1e4 * eps)
        false_calls += stats["t"] > 3
    assert false_calls / 300 < 0.03


def test_loss_trend_detects_decrease_well_below_step_noise():
    """A drop of half the per-step noise SD per window is still found (the
    expected t is ~5): the test is against the noise of the trend, not of
    a step."""
    rng = np.random.default_rng(1)
    losses = (1e5 - 5000 * np.arange(WINDOW) / WINDOW
              + 1e4 * rng.normal(size=WINDOW))
    assert loss_trend(losses)["t"] > 3


# -----------------------------------------------------------------------------
# loss_skew
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("noise", ["normal", "exponential", "lognormal",
                                   "student_t3"])
def test_loss_skew_benign_noise_is_small(noise):
    """Monte Carlo ELBO noise, right-skewed or heavy-tailed but without
    separate penalties, stays well under MAX_LOSS_SKEW."""
    rng = np.random.default_rng(0)
    draw = {"normal": lambda n: rng.normal(size=n),
            "exponential": lambda n: rng.exponential(size=n),
            "lognormal": lambda n: rng.lognormal(sigma=1.0, size=n),
            "student_t3": lambda n: rng.standard_t(3, size=n)}[noise]
    for _ in range(20):
        stats = loss_trend(1e5 + 1e3 * draw(WINDOW))
        assert stats["skew"] < 1.5 < conv.MAX_LOSS_SKEW


def test_loss_skew_rare_penalties_are_large():
    """Rare draws with a huge penalty: the median ignores them, the mean does
    not.  The skew is the mean shift (rate x penalty) in per-step noise SDs:
    here 5-30% of steps carrying 240 noise SDs."""
    rng = np.random.default_rng(1)
    for rate in (0.05, 0.1, 0.3):
        losses = (4e4 + 1e3 * rng.normal(size=WINDOW)
                  + 2.4e5 * (rng.random(WINDOW) < rate))
        stats = loss_trend(losses)
        assert stats["skew"] > 3 * conv.MAX_LOSS_SKEW
        assert stats["mean"] > stats["level"] + 3e3


def test_loss_skew_ignores_trend():
    """A falling loss is not read as skew (residuals are about the line)."""
    rng = np.random.default_rng(2)
    losses = 1e5 - 20.0 * np.arange(WINDOW) + 10.0 * rng.normal(size=WINDOW)
    assert abs(loss_trend(losses)["skew"]) < 0.5


def test_loss_skew_constant_and_empty():
    assert loss_trend(np.full(WINDOW, 5.0))["skew"] == 0.0
    assert loss_skew(np.zeros(10)) == 0.0
    assert loss_skew([]) == 0.0


def test_loss_skew_deterministic_repeats_are_not_skew():
    """A deterministic loss whose steps mostly repeat one value: the spread
    is floored at rtol * |level|, so float-level differences are not skew."""
    losses = np.full(WINDOW, 1e5)
    losses[::7] += 1e-3
    assert loss_skew(losses - 1e5, level=1e5) < conv.MAX_LOSS_SKEW


# -----------------------------------------------------------------------------
# param_drift / summarize_excess
# -----------------------------------------------------------------------------

def _block_centers(n=8, window=WINDOW):
    width = window / n
    return (np.arange(n) + 0.5) * width


def test_param_drift_linear_movement_in_sd_units():
    centers = _block_centers()
    sd = np.array([0.5, 2.0])
    # moves by exactly 1 SD per window, no noise
    means = (centers[:, None] / WINDOW) * sd[None, :]
    drift, excess = param_drift(means, centers, WINDOW, sd, z=3)
    np.testing.assert_allclose(drift, [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(excess, [1.0, 1.0], rtol=1e-6)


def test_param_drift_stationary_noise_has_little_excess():
    rng = np.random.default_rng(2)
    centers = _block_centers()
    means = rng.normal(scale=0.3, size=(8, 5000))
    _, excess = param_drift(means, centers, WINDOW, 1.0, z=3)
    assert summarize_excess(excess) < 0.3
    assert np.mean(np.asarray(excess) > 0) < 0.05


def test_param_drift_floor_absorbs_optimizer_jitter():
    """Movement of a few step sizes (Adam jitter) is under the floor even
    when the normalizer (a collapsed guide scale) is much smaller."""
    rng = np.random.default_rng(3)
    step_size = 1e-3
    centers = _block_centers()
    means = np.cumsum(rng.normal(scale=step_size, size=(8, 100)), axis=0)
    floor = conv.MIN_STEP_COHERENCE * step_size * WINDOW
    _, no_floor = param_drift(means, centers, WINDOW, 1e-4, z=3)
    _, with_floor = param_drift(means, centers, WINDOW, 1e-4, z=3,
                                floor=floor)
    assert summarize_excess(no_floor) > 1
    assert summarize_excess(with_floor) == 0


def test_param_drift_coherent_movement_beats_floor():
    step_size = 1e-3
    centers = _block_centers()
    means = (centers / WINDOW * 0.5 * step_size * WINDOW)[:, None]  # 50% speed
    floor = conv.MIN_STEP_COHERENCE * step_size * WINDOW
    _, excess = param_drift(means, centers, WINDOW, 0.01, z=3, floor=floor)
    assert summarize_excess(excess) > 1


def test_param_drift_accepts_jax_arrays():
    centers = _block_centers()
    means = jnp.asarray(np.outer(centers / WINDOW, [1.0, 2.0]))
    drift, excess = param_drift(means, centers, WINDOW, jnp.ones(2), z=3)
    np.testing.assert_allclose(np.asarray(drift), [1.0, 2.0], rtol=1e-5)


def test_summarize_excess_small_array_uses_max():
    assert summarize_excess(np.array([0.0, 0.0, 0.7])) == pytest.approx(0.7)


def test_summarize_excess_large_array_uses_quantile():
    excess = np.zeros(10000)
    excess[:5] = 10.0  # 0.05% of elements in the tail
    assert summarize_excess(excess) == 0.0
    excess[:500] = 10.0  # 5%
    assert summarize_excess(excess) == pytest.approx(10.0)


def test_summarize_excess_nonfinite_is_infinite():
    assert summarize_excess(np.array([0.0, np.nan])) == np.inf
    assert summarize_excess(np.array([])) == 0.0


# -----------------------------------------------------------------------------
# ConvergenceMonitor decisions
# -----------------------------------------------------------------------------

_FLAT = {"level": 100.0, "drop": 0.0, "drop_se": 1.0, "t": 0.0}
_IMPROVING = {"level": 100.0, "drop": 10.0, "drop_se": 1.0, "t": 10.0}


@pytest.mark.parametrize("kwargs", [{"step_size_cut": 1.0},
                                    {"step_size_cut": 0.0},
                                    {"patience": 0}])
def test_monitor_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ConvergenceMonitor(1e-3, 1e-6, **kwargs)


def test_monitor_cuts_after_patience_stalls():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    assert [m.end_window(i, _FLAT) for i in range(3)] == \
        [conv.CONTINUE, conv.CONTINUE, conv.CUT]
    assert m.step_size == pytest.approx(1e-4)
    assert m.plateau_count == 0
    assert m.num_cuts == 1


def test_monitor_cut_ignores_moving_parameters():
    """Above the floor the loss alone decides a cut."""
    m = ConvergenceMonitor(1e-3, 1e-6, patience=2)
    moving = {"p": 5.0}
    m.end_window(0, _FLAT, moving)
    assert m.end_window(1, _FLAT, moving) == conv.CUT


def test_monitor_improvement_resets_patience():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    m.end_window(0, _FLAT)
    m.end_window(1, _FLAT)
    m.end_window(2, _IMPROVING)
    assert m.plateau_count == 0
    assert m.end_window(3, _FLAT) == conv.CONTINUE


def test_monitor_rising_loss_counts_as_stall():
    rising = {"level": 100.0, "drop": -50.0, "drop_se": 1.0, "t": -50.0}
    m = ConvergenceMonitor(1e-3, 1e-6, patience=1)
    assert m.end_window(0, rising) == conv.CUT


def test_monitor_loss_rtol_floor():
    """A significant but negligible improvement (deterministic loss) is a
    stall."""
    tiny = {"level": 1e6, "drop": 0.1, "drop_se": 0.0, "t": np.inf}
    m = ConvergenceMonitor(1e-3, 1e-3, patience=1, loss_rtol=1e-6)
    assert m.end_window(0, tiny) == conv.CONVERGED
    m = ConvergenceMonitor(1e-3, 1e-3, patience=1, loss_rtol=1e-9)
    assert m.end_window(0, tiny) == conv.CONTINUE


def test_monitor_stop_needs_parameters_still_at_floor():
    m = ConvergenceMonitor(1e-6, 1e-6, patience=2, param_tolerance=0.05)
    moving = {"a": 0.0, "b": 0.2}
    for i in range(10):
        assert m.end_window(i, _FLAT, moving) == conv.CONTINUE
    assert m.last["worst_param"] == "b"
    assert m.last["params_moving"]
    still = {"a": 0.0, "b": 0.01}
    m.end_window(10, _FLAT, still)
    assert m.end_window(11, _FLAT, still) == conv.CONVERGED
    assert m.converged


def test_monitor_stop_needs_unskewed_loss_at_floor():
    """At the floor a skewed window is not a plateau: the run continues (and
    reports it) rather than converging on a median that hides the mean."""
    skewed = dict(_FLAT, skew=40.0, mean=500.0)
    m = ConvergenceMonitor(1e-6, 1e-6, patience=2)
    assert [m.end_window(i, skewed) for i in range(6)] == [conv.CONTINUE] * 6
    assert m.last["loss_skewed"] and m.last["loss_skew"] == 40.0
    assert m.last["loss_mean"] == 500.0
    assert "loss skewed" in m.describe()
    assert m.end_window(6, dict(_FLAT, skew=0.3)) == conv.CONTINUE
    assert m.end_window(7, dict(_FLAT, skew=0.3)) == conv.CONVERGED


def test_monitor_skew_never_blocks_a_cut():
    """A smaller step size does not remove rare penalties, so skew only gates
    the stop."""
    m = ConvergenceMonitor(1e-3, 1e-6, patience=2)
    skewed = dict(_FLAT, skew=40.0)
    m.end_window(0, skewed)
    assert m.end_window(1, skewed) == conv.CUT


def test_monitor_max_loss_skew_setting():
    m = ConvergenceMonitor(1e-6, 1e-6, patience=1, max_loss_skew=50.0)
    assert m.end_window(0, dict(_FLAT, skew=40.0)) == conv.CONVERGED
    m = ConvergenceMonitor(1e-6, 1e-6, patience=1)
    assert m.end_window(0, dict(_FLAT, skew=np.nan)) == conv.CONTINUE


def test_monitor_goes_through_all_stages():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=1)
    decisions = [m.end_window(i, _FLAT, {"p": 0.0}) for i in range(4)]
    assert decisions == [conv.CUT, conv.CUT, conv.CUT, conv.CONVERGED]
    assert m.step_size == pytest.approx(1e-6)
    assert m.num_cuts == 3


def test_monitor_cut_clamps_at_floor():
    m = ConvergenceMonitor(1e-3, 5e-4, patience=1)
    m.end_window(0, _FLAT)
    assert m.step_size == pytest.approx(5e-4)
    assert m.at_floor


def test_monitor_no_floor_means_no_cuts():
    m = ConvergenceMonitor(1e-3, None, patience=1)
    assert m.at_floor
    assert m.end_window(0, _FLAT) == conv.CONVERGED


def test_monitor_drift_floor():
    m = ConvergenceMonitor(1e-3, 1e-6)
    assert m.drift_floor(2000) == pytest.approx(
        conv.MIN_STEP_COHERENCE * 1e-3 * 2000)
    assert ConvergenceMonitor(np.nan).drift_floor(2000) == 0.0


def test_monitor_state_round_trip():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    m.end_window(0, _FLAT)
    m.end_window(1, _FLAT)
    m.end_window(2, _FLAT)  # cut
    m.end_window(3, _FLAT)
    state = m.state_dict()

    m2 = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    m2.load_state_dict(state)
    assert m2.step_size == pytest.approx(1e-4)
    assert m2.plateau_count == 1
    assert m2.num_cuts == 1
    assert m2.last == m.last


def test_monitor_describe():
    m = ConvergenceMonitor(1e-3, 1e-6)
    assert "no complete" in m.describe()
    m.end_window(2000, _FLAT, {"theta_loc": 0.3}, {"theta_loc": 0.4})
    text = m.describe()
    assert "theta_loc" in text and "step 2000" in text


# -----------------------------------------------------------------------------
# Synthetic loss traces
# -----------------------------------------------------------------------------

def _noisy(clean, rel_sd=0.05, abs_sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    sd = rel_sd * np.abs(clean) + abs_sd
    return clean + sd * rng.normal(size=clean.size)


def test_trace_monotone_decay_never_stalls_while_falling():
    """Exponential decay with noise proportional to the loss (the SVI
    pattern): improving in every window while it is still falling."""
    steps = np.arange(20 * WINDOW)
    trace = _noisy(1e3 + 1e6 * np.exp(-steps / 8000), rel_sd=0.25)
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace[:10 * WINDOW])
    assert conv.CUT not in decisions
    assert m.last["loss_improving"]


def test_trace_noisy_floor_cuts_after_patience():
    steps = np.arange(30 * WINDOW)
    trace = _noisy(1e5 + 1e6 * np.exp(-steps / 2000), rel_sd=0.0,
                   abs_sd=1e4)
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace)
    first_cut = decisions.index(conv.CUT)
    # decay is gone after ~6 windows; the cut follows within a few windows
    assert 5 <= first_cut <= 12


def test_trace_two_phase_plateau_shorter_than_patience():
    """drop, plateau of two windows, larger drop: no cut on the plateau."""
    seg = WINDOW
    drop1 = np.linspace(1e6, 5e5, 2 * seg)
    plateau = np.full(2 * seg, 5e5)
    drop2 = np.linspace(5e5, 1e4, 4 * seg)
    trace = _noisy(np.concatenate([drop1, plateau, drop2]), rel_sd=0.0,
                   abs_sd=2e3)
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace)
    assert conv.CUT not in decisions


def test_trace_two_phase_plateau_longer_than_patience_cuts():
    """A plateau as long as the patience span does trigger a cut -- the
    parameter test, not the loss, is what keeps a slow phase from being
    called converged."""
    seg = WINDOW
    trace = _noisy(np.concatenate([np.linspace(1e6, 5e5, 2 * seg),
                                   np.full(4 * seg, 5e5)]),
                   rel_sd=0.0, abs_sd=2e3)
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    assert conv.CUT in _feed(m, trace)


def test_trace_deterministic_slow_tail():
    """A noiseless tail: improving until the per-window drop falls under
    loss_rtol of the level."""
    steps = np.arange(40 * WINDOW)
    trace = 1e5 + 1e3 * np.exp(-steps / 5000)
    m = ConvergenceMonitor(1e-3, 1e-3, patience=3, loss_rtol=1e-6)
    decisions = _feed(m, trace)
    assert decisions[-1] == conv.CONVERGED
    # drop per window at the stop is below 1e-6 * 1e5 = 0.1
    assert m.last["loss_drop"] < 0.1


def test_trace_resume_matches_uninterrupted():
    rng = np.random.default_rng(4)
    trace = 1e4 + 1e5 * np.exp(-np.arange(24 * WINDOW) / 6000) \
        + 300 * rng.normal(size=24 * WINDOW)

    whole = ConvergenceMonitor(1e-3, 1e-6, patience=2)
    expected = _feed(whole, trace)

    first = ConvergenceMonitor(1e-3, 1e-6, patience=2)
    half = 10 * WINDOW
    got = _feed(first, trace[:half])
    resumed = ConvergenceMonitor(1e-3, 1e-6, patience=2)
    resumed.load_state_dict(first.state_dict())
    got += _feed(resumed, trace[half:])
    assert got == expected
    assert resumed.step_size == whole.step_size


# -----------------------------------------------------------------------------
# Recorded traces
# -----------------------------------------------------------------------------

def test_recorded_real_data_trace_is_still_improving():
    """Real data, ~200k genotypes, batch 65,536 (4 steps per epoch).  The run
    was stopped by the old rule; every window, including the last, is still
    a significant improvement."""
    trace = np.load(os.path.join(TRACES, "real_svi_200k_genotypes.npy"))
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace.astype(float))
    assert decisions and set(decisions) == {conv.CONTINUE}
    assert m.last["loss_t"] > 10


def test_recorded_simulation_trace_is_still_improving():
    """Step-3.5 anchored run 0002: the old rule stopped it at step ~8,250
    ('converged') while the loss was falling ~20% per 1000 steps."""
    trace = np.load(os.path.join(TRACES, "sim_svi_anchored_run0002.npy"))
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace.astype(float))
    assert set(decisions) == {conv.CONTINUE}
    assert m.last["loss_t"] > 10


def test_recorded_traces_are_not_skewed():
    """Neither recorded clean trace has a window near MAX_LOSS_SKEW."""
    for name in ("real_svi_200k_genotypes", "sim_svi_anchored_run0002",
                 "sim_svi_unclipped_converged_run0005"):
        trace = np.load(os.path.join(TRACES, name + ".npy")).astype(float)
        for _, losses in _windows(trace):
            assert loss_trend(losses)["skew"] < 1.0, name


def test_recorded_clipped_trace_never_converges():
    """Congression-calibration baseline run 0003 at step size 1e-6 under
    elementwise gradient clipping: ~15% of steps carry a ~2.4e5 binding
    penalty (a steep binding curve measured with SD 1e-3).  The block medians
    are flat, so the loss test alone called every window a plateau; the skew
    keeps the monitor from calling it converged."""
    trace = np.load(os.path.join(
        TRACES, "sim_svi_clipped_events_run0003.npy")).astype(float)
    skews = [loss_trend(losses)["skew"] for _, losses in _windows(trace)]
    assert min(skews) > 5 * conv.MAX_LOSS_SKEW

    m = ConvergenceMonitor(1e-6, 1e-6, patience=2)
    decisions = _feed(m, trace)
    assert set(decisions) == {conv.CONTINUE}
    assert m.last["loss_skewed"] and not m.last["loss_improving"]

    # the loss test alone would have stopped it
    blind = ConvergenceMonitor(1e-6, 1e-6, patience=2, max_loss_skew=np.inf)
    assert _feed(blind, trace)[-1] == conv.CONVERGED


def test_recorded_unclipped_trace_converges_at_floor():
    """The same simulation (run 0005) refit without clipping, around its
    convergence: flat and unskewed, so the loss test stops it."""
    trace = np.load(os.path.join(
        TRACES, "sim_svi_unclipped_converged_run0005.npy")).astype(float)
    m = ConvergenceMonitor(1e-6, 1e-6, patience=3)
    assert _feed(m, trace)[-1] == conv.CONVERGED


# -----------------------------------------------------------------------------
# Pooled check of a run of stalls
# -----------------------------------------------------------------------------

def test_pooled_loss_trend_matches_single_window_line():
    """Pooling windows of one exact line gives that line's drop per window."""
    trace = 1e6 - 50.0 * np.arange(3 * WINDOW)
    stats = [loss_trend(trace[i * WINDOW:(i + 1) * WINDOW]) for i in range(3)]
    pooled = conv.pooled_loss_trend(stats)
    assert pooled["drop"] == pytest.approx(50.0 * WINDOW, rel=1e-6)


def test_pooled_check_keeps_a_slow_noisy_descent():
    """Each window alone is not significant, but three pooled are."""
    rng = np.random.default_rng(1)
    trace = 1e6 - 20.0 * np.arange(12 * WINDOW) + 1.5e5 * rng.normal(size=12 * WINDOW)
    single = [loss_trend(trace[i:i + WINDOW])["t"]
              for i in range(0, 12 * WINDOW, WINDOW)]
    assert np.median(single) < conv.ConvergenceMonitor(1e-3).z
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    assert conv.CUT not in _feed(m, trace)


def test_pooled_check_still_cuts_a_real_plateau():
    rng = np.random.default_rng(2)
    trace = 1e6 + 1.5e5 * rng.normal(size=6 * WINDOW)
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace)
    assert decisions[2] == conv.CUT
    assert abs(m.last["pooled_loss_t"]) < 3


def test_pooled_check_blocks_a_stop_on_a_descent_at_the_floor():
    rng = np.random.default_rng(3)
    trace = 1e6 - 20.0 * np.arange(9 * WINDOW) + 1.5e5 * rng.normal(size=9 * WINDOW)
    m = ConvergenceMonitor(1e-6, 1e-6, patience=3)
    assert conv.CONVERGED not in _feed(m, trace)


def test_pooled_stalls_survive_a_resume():
    rng = np.random.default_rng(5)
    trace = 1e6 - 20.0 * np.arange(6 * WINDOW) + 1.5e5 * rng.normal(size=6 * WINDOW)
    whole = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    expected = _feed(whole, trace)
    first = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    got = _feed(first, trace[:2 * WINDOW])
    resumed = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    resumed.load_state_dict(first.state_dict())
    got += _feed(resumed, trace[2 * WINDOW:])
    assert got == expected


def test_recorded_count_likelihood_descent_is_not_cut():
    """Count-likelihood grid v2, run 0012 (poisson, seed 3, mixture), steps
    1-22,000 at step size 1e-3. The per-window test called windows
    16k/18k/20k stalls (t = 1.9, 3.0, 1.0) while the window medians kept
    falling 1.5-1.9e5 per window; the cut at 20k left the fit at twice the
    ELBO of its seed-2 twin, in a wrong optimum. Pooled, it is a descent."""
    trace = np.load(os.path.join(
        TRACES, "sim_svi_counts_noisy_descent_run0012.npy")).astype(float)
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _feed(m, trace)
    assert conv.CUT not in decisions


# -----------------------------------------------------------------------------
# runaway loss
# -----------------------------------------------------------------------------

def test_runaway_loss_is_diverged_not_converged():
    """
    A loss that runs off toward -inf (an unbounded posterior density) stops
    the run as diverged, where it used to plateau at ~-8e23 and converge.
    """
    rng = np.random.default_rng(0)
    start = 4e4 + rng.normal(0, 100, WINDOW)
    runaway = -8e23 * (1 + 0.1 * rng.normal(size=6 * WINDOW))
    m = ConvergenceMonitor(1e-3, None, patience=3)
    decisions = _feed(m, np.concatenate([start, runaway]))
    assert decisions[-1] == conv.DIVERGED
    assert m.diverged and not m.converged
    assert m.reference_loss == pytest.approx(4e4, rel=0.05)


def test_negative_but_bounded_loss_is_not_runaway():
    """A loss that settles below zero, far above the runaway line, is fine."""
    rng = np.random.default_rng(1)
    trace = np.concatenate([1e4 - np.linspace(0, 1.2e4, 4 * WINDOW),
                            -2e3 + rng.normal(0, 1, 8 * WINDOW)])
    m = ConvergenceMonitor(1e-3, None, patience=3)
    decisions = _feed(m, trace)
    assert conv.DIVERGED not in decisions
    assert decisions[-1] == conv.CONVERGED


def test_reference_loss_survives_checkpoint():
    m = ConvergenceMonitor(1e-3, 1e-6)
    m.end_window(WINDOW, loss_trend(np.full(WINDOW, 5e3)
                                    + np.random.default_rng(2).normal(0, 1, WINDOW)))
    resumed = ConvergenceMonitor(1e-3, 1e-6)
    resumed.load_state_dict(m.state_dict())
    assert resumed.reference_loss == pytest.approx(m.reference_loss)


# -----------------------------------------------------------------------------
# exact (full-batch) loss
# -----------------------------------------------------------------------------

# Mini-batch statistics that, alone, would read as a stall (noise hides the
# descent) or as heavily skewed.
_NOISY = {"level": 6e7, "mean": 6e7, "drop": 1e5, "drop_se": 5e4, "t": 2.0,
          "skew": 0.0}


def _exact_feed(monitor, exact_losses, stats=_NOISY):
    return [monitor.end_window(i, dict(stats), exact_loss=e)
            for i, e in enumerate(exact_losses)]


def test_exact_descent_hidden_by_minibatch_noise_is_not_a_stall():
    """A 1e5-per-window descent at mini-batch SE 5e4 stalls on the noisy
    test (t = 2 < 3) but is a descent on the exact loss."""
    noisy = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    assert [noisy.end_window(i, dict(_NOISY)) for i in range(3)][-1] == conv.CUT

    exact = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    decisions = _exact_feed(exact, 6e7 - 1e5 * np.arange(10))
    assert all(d == conv.CONTINUE for d in decisions)
    assert exact.step_size == 1e-3
    assert exact.last["loss_exact"] == pytest.approx(6e7 - 9e5)
    assert exact.last["loss_drop"] == pytest.approx(1e5)
    assert exact.last["loss_drop_se"] == pytest.approx(0.0, abs=1e-6)


def test_exact_first_window_counts_as_descent():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=1)
    assert m.end_window(0, dict(_NOISY), exact_loss=6e7) == conv.CONTINUE


def test_exact_flat_loss_cuts_then_converges():
    m = ConvergenceMonitor(1e-3, 1e-5, patience=2, loss_rtol=1e-6)
    # rises and falls by less than loss_rtol * |loss| (60 nats): a plateau
    decisions = _exact_feed(m, [6e7, 6e7 + 10, 6e7 - 5, 6e7 + 3, 6e7, 6e7 - 1,
                                6e7 + 2, 6e7])
    assert conv.CUT in decisions
    assert decisions[-1] == conv.CONVERGED
    assert m.step_size == pytest.approx(1e-5)


def test_exact_jittery_descent_is_improving_and_jitter_alone_is_not():
    """Single windows rise under the optimizer's jitter, but the line through
    recent windows descends; jitter about a level has no significant slope."""
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3, loss_rtol=1e-8, z=3.0)
    series = 1000.0 - 10.0 * np.arange(12) + np.tile([0.0, 3.0, -2.0, 1.0], 3)
    assert conv.CUT not in _exact_feed(m, series)

    flat = ConvergenceMonitor(1e-3, 1e-6, patience=3, loss_rtol=1e-8, z=3.0)
    decisions = _exact_feed(flat, [1000.0, 1001.0, 999.0, 1000.5, 999.5,
                                   1000.0, 1001.0, 999.0])
    assert conv.CUT in decisions


def test_exact_small_steady_drops_below_noise_floor_count():
    """A deterministic, steady descent of 2 nats per window is improving
    once the floor is below it, however small."""
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3, loss_rtol=1e-6)
    assert conv.CUT not in _exact_feed(m, 1000.0 - 2.0 * np.arange(10))


def test_exact_ignores_minibatch_skew():
    skewed = dict(_NOISY, skew=50.0)
    m = ConvergenceMonitor(1e-6, 1e-6, patience=2)
    decisions = _exact_feed(m, [100.0, 100.0, 100.0], stats=skewed)
    assert decisions[-1] == conv.CONVERGED
    assert not m.last["loss_skewed"]


def test_exact_runaway_uses_exact_loss():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    m.end_window(0, dict(_NOISY), exact_loss=100.0)
    assert m.end_window(1, dict(_NOISY), exact_loss=-1e7) == conv.DIVERGED


def test_exact_state_round_trip():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    _exact_feed(m, [100.0, 90.0])
    m2 = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    m2.load_state_dict(m.state_dict())
    # the restored exact losses join the line: 100, 90, 85 fall 7.5 per window
    m2.end_window(2, dict(_NOISY), exact_loss=85.0)
    assert m2.last["loss_drop"] == pytest.approx(7.5)


def test_exact_describe():
    m = ConvergenceMonitor(1e-3, 1e-6, patience=3)
    _exact_feed(m, [100.0, 90.0])
    assert "exact loss 90" in m.describe()
