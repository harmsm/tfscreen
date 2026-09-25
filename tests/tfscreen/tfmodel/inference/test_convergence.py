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
        if decisions[-1] == conv.CONVERGED:
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
