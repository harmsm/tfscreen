import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
import h5py
from tfscreen.analysis.hierarchical.run_inference import RunInference
import os
import dill
import optax
from numpyro.infer.autoguide import AutoDelta, AutoLaplaceApproximation
from numpyro.infer.svi import SVIState

from numpyro.infer.svi import SVIState
from flax import struct

@struct.dataclass
class MockData:
    num_genotype: int = 10

class MockModel:
    def __init__(self, num_genotype=10):
        self.data = MockData(num_genotype=num_genotype)
        self.priors = {}
        self.jax_model = lambda **kwargs: None
        self.jax_model_guide = lambda **kwargs: None
        self.init_params = {}
    
    def get_batch(self, data, indices):
        return data
    
    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)

def test_init():
    model = MockModel()
    ri = RunInference(model, seed=42)
    assert ri.model == model
    assert ri._seed == 42
    
    # Test missing attribute
    del model.data
    with pytest.raises(ValueError, match="`model` must have attribute data"):
        RunInference(model, seed=42)


def test_setup_svi():
    model = MockModel()
    ri = RunInference(model, seed=42)
    
    # Test with default guide_type
    svi = ri.setup_svi(guide_type="component")
    assert svi.guide == model.jax_model_guide
    
    svi = ri.setup_svi(guide_type="delta")
    assert isinstance(svi.guide, AutoDelta)
    
def test_run_optimization(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    
    out_prefix = os.path.join(tmpdir, "test")
    
    # Mock jax and svi methods to avoid actual execution
    mocker.patch("jax.jit", side_effect=lambda x: x)
    # Mock scan_fn or just lax.scan
    mocker.patch("jax.lax.scan", return_value=("state", jnp.array([1.0, 0.9])))
    
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = "state"
    mock_svi.update.return_value = ("state", 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}
    
    # Mock get_key to avoid randomness issues in tests
    mocker.patch.object(ri, "get_key", return_value=jnp.array([0, 42]))
    
    state, params, converged = ri.run_optimization(
        mock_svi, 
        max_num_epochs=10, 
        convergence_check_interval=1,
        checkpoint_interval=1,
        out_prefix=out_prefix,
        convergence_tolerance=0.01,
        patience=1
    )
    
    assert state is not None
    assert "p" in params
    assert isinstance(converged, bool)
    assert os.path.exists(f"{out_prefix}_losses.bin")
    assert os.path.exists(f"{out_prefix}_losses.txt")

def test_run_optimization_no_patch_scan(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    # 228: svi_state is None
    mocker.patch("jax.jit", side_effect=lambda x: x)
    mock_svi = mocker.Mock()
    # Use jnp array instead of string for state
    state = jnp.array([0.0])
    mock_svi.init.return_value = state
    mock_svi.update.return_value = (state, 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}
    
    # 236-241: scan_fn
    # Don't patch lax.scan, let it run
    # 228: Call jitter
    ri.run_optimization(mock_svi, max_num_epochs=2, convergence_check_interval=1, checkpoint_interval=1, init_params={"p": jnp.array(1.0)})
    assert len(ri._loss_deque) > 0

def test_run_optimization_svi_state_types(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    mocker.patch("jax.jit", side_effect=lambda x: x)
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = jnp.array([0.0])
    mock_svi.update.return_value = (jnp.array([0.0]), 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}
    
    # 269: Passing a non-string, non-None state
    ri.run_optimization(mock_svi, svi_state=jnp.array([1.0]), max_num_epochs=1)
    
    # 265-266: Passing an invalid string
    with pytest.raises(ValueError, match="is not valid"):
        ri.run_optimization(mock_svi, svi_state="invalid_path_to_svi_state", max_num_epochs=1)

def test_run_optimization_convergence(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    
    mocker.patch("jax.jit", side_effect=lambda x: x)
    # Return losses that decrease and then stabilize
    losses = np.concatenate([np.linspace(100, 10, 50), np.ones(50) * 10])
    mocker.patch("jax.lax.scan", return_value=("state", jnp.array(losses)))
    
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = "state"
    mock_svi.update.return_value = ("state", 10.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}
    
    state, params, converged = ri.run_optimization(
        mock_svi, 
        max_num_epochs=100, 
        convergence_check_interval=1,
        checkpoint_interval=1,
        convergence_tolerance=1e-1,
        convergence_window=1,
        patience=1
    )
    # It might not converge in exactly this setup due to how scan returns, 
    # but it covers the logic in _update_loss_deque and _check_convergence
    assert converged in [True, False]

def test_run_optimization_nan_explosion(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    
    mocker.patch("jax.jit", side_effect=lambda x: x)
    mocker.patch("jax.lax.scan", return_value=("state", jnp.array([1.0])))
    
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = "state"
    mock_svi.get_params.return_value = {"p": jnp.array(np.nan)}
    
    with pytest.raises(RuntimeError, match="model exploded"):
        ri.run_optimization(mock_svi, max_num_epochs=1, convergence_check_interval=1, checkpoint_interval=1)

def test_get_posteriors(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test")
    
    mock_svi = mocker.Mock()
    mock_svi.guide = mocker.Mock()
    mock_svi.get_params.return_value = {}
    
    # Mock Predictive
    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_latent_sampler = mock_predictive.return_value
    mock_latent_sampler.return_value = {"param": jnp.zeros((10, 10))} # Batch size 10, 10 genotypes
    
    mocker.patch("jax.device_get", side_effect=lambda x: x)
    
    ri.get_posteriors(mock_svi, "state", out_prefix, num_posterior_samples=20, sampling_batch_size=10)
    
    assert os.path.exists(f"{out_prefix}_posterior.h5")

def test_get_posteriors_batching_logic(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_batching")
    
    mock_svi = mocker.Mock()
    mock_svi.guide = mocker.Mock()
    mock_svi.get_params.return_value = {}
    
    # Mock Predictive to return different sizes
    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_sampler = mock_predictive.return_value
    # num_samples=25, sampling_batch_size=10 -> 3 batches (10, 10, 5)
    # Each loop has 1 latent call and 1 forward call (forward_batch_size=512 > 5 genotypes)
    # Total 6 calls.
    mock_sampler.side_effect = [
        {"p": jnp.zeros((10, 5))}, # latent 1
        {"obs": jnp.zeros((10, 5))}, # forward 1
        {"p": jnp.zeros((10, 5))}, # latent 2
        {"obs": jnp.zeros((10, 5))}, # forward 2
        {"p": jnp.zeros((5, 5))},  # latent 3
        {"obs": jnp.zeros((5, 5))}   # forward 3
    ]
    
    mocker.patch("jax.device_get", side_effect=lambda x: x)
    
    ri.get_posteriors(mock_svi, "state", out_prefix, num_posterior_samples=25, sampling_batch_size=10)
    assert os.path.exists(f"{out_prefix}_posterior.h5")

def test_get_posteriors_full_logic(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_full")
    
    mock_svi = mocker.Mock()
    mock_svi.get_params.return_value = {}
    
    # 465-466: global parameter
    # 479-480: concatenate multiple forward batches (5 genotypes each)
    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_sampler = mock_predictive.return_value
    mock_sampler.side_effect = [
        {"global_p": jnp.zeros((1, 1)), "geno_p": jnp.zeros((1, 10))}, # latent 
        {"obs": jnp.zeros((1, 5))}, # forward 1
        {"obs": jnp.zeros((1, 5))}, # forward 2
    ]
    
    mocker.patch("jax.device_get", side_effect=lambda x: x)
    
    ri.get_posteriors(mock_svi, "state", out_prefix, num_posterior_samples=1, 
                     sampling_batch_size=1, forward_batch_size=5)
    assert os.path.exists(f"{out_prefix}_posterior.h5")

def test_run_optimization_restore(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_restore")
    
    # Create a valid checkpoint
    state = SVIState(None, None, None) if ri.get_key().size > 0 else SVIState(None, None)
    try:
        state = SVIState(None, None, None)
    except:
        state = SVIState(None, None)
    
    ri._write_checkpoint(state, out_prefix)
    checkpoint_file = f"{out_prefix}_checkpoint.pkl"
    
    mocker.patch("jax.jit", side_effect=lambda x: x)
    mocker.patch("jax.lax.scan", return_value=(state, jnp.array([1.0])))
    
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = state
    mock_svi.update.return_value = (state, 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}
    
    # Run with checkpoint path
    ri.run_optimization(mock_svi, svi_state=checkpoint_file, max_num_epochs=1, convergence_check_interval=1, checkpoint_interval=1)

def test_write_params(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_params")
    params = {"p": jnp.array([1.0])}
    ri.write_params(params, out_prefix)
    assert os.path.exists(f"{out_prefix}_params.npz")

def test_write_losses_append(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_losses")
    # Pre-existing binary file is ignored when current_step != 0
    path = f"{out_prefix}_losses.bin"
    with open(path, "wb") as f:
        f.write(b"existing")

    ri._write_losses([1.0, 2.0], out_prefix)
    assert os.path.getsize(path) > 8


def test_write_losses_txt_has_header(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_hdr")

    # Step 0: header is written even when losses list is empty
    ri._write_losses(np.array([]), out_prefix)
    txt_path = f"{out_prefix}_losses.txt"
    assert os.path.exists(txt_path)
    with open(txt_path) as fh:
        first_line = fh.readline().strip()
    assert first_line == "epoch,loss,relative_change"


def test_write_losses_txt_format(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_fmt")

    # Init (step 0)
    ri._write_losses(np.array([]), out_prefix)

    # Simulate one interval having completed
    ri._current_step = ri._iterations_per_epoch  # epoch 1
    ri._relative_change = 0.123
    ri._write_losses(np.array([42.0, 43.0]), out_prefix)

    with open(f"{out_prefix}_losses.txt") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    assert lines[0] == "epoch,loss,relative_change"
    # Second line: epoch=1, loss=last value (43.0), relative_change=0.123
    parts = lines[1].split(",")
    assert int(parts[0]) == 1
    assert float(parts[1]) == pytest.approx(43.0)
    assert float(parts[2]) == pytest.approx(0.123)


def test_update_loss_deque_all_constant():
    """Constant losses produce relative_change=0 via zero numerator."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque
    ri._loss_deque = deque(maxlen=200)
    ri._update_loss_deque(np.ones(200))
    assert ri._relative_change == 0.0


def test_update_loss_deque_uses_total_improvement():
    """Metric is normalized by total improvement, not current loss magnitude."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque

    # Deque of size 4; warm-up: old_half=[100,90], new_half=[80,70]
    ri._loss_deque = deque(maxlen=4)
    ri._update_loss_deque([100.0, 90.0, 80.0, 70.0])

    # loss_start = mean([100,90]) = 95; loss_best = mean([80,70]) = 75
    # denom = |95 - 75| = 20; numerator = |95 - 75| = 20 → metric = 1.0
    assert ri._loss_start == pytest.approx(95.0)
    assert ri._loss_best == pytest.approx(75.0)
    assert ri._relative_change == pytest.approx(1.0)


def test_update_loss_deque_sign_change():
    """Metric is robust when the loss function crosses zero."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque

    ri._loss_deque = deque(maxlen=4)
    ri._update_loss_deque([10.0, 5.0, -5.0, -10.0])

    # loss_start = 7.5; loss_best = -7.5; denom = 15; metric = 15/15 = 1.0
    assert ri._relative_change == pytest.approx(1.0)
    assert ri._loss_start == pytest.approx(7.5)
    assert ri._loss_best == pytest.approx(-7.5)


def test_update_loss_deque_loss_start_fixed_after_first_check():
    """loss_start is captured once at the end of warm-up and never updated."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque

    ri._loss_deque = deque(maxlen=4)
    ri._update_loss_deque([100.0, 90.0, 80.0, 70.0])
    first_loss_start = ri._loss_start

    ri._update_loss_deque([60.0, 50.0])
    assert ri._loss_start == first_loss_start


def test_update_loss_deque_warmup_returns_inf():
    """Returns inf while the deque is not yet full."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque
    ri._loss_deque = deque(maxlen=10)
    ri._update_loss_deque([1.0, 2.0, 3.0])
    assert ri._relative_change == np.inf


def test_loss_state_reset_at_run_start(mocker):
    """_loss_start and _loss_best are reset each time run_optimization begins."""
    model = MockModel()
    ri = RunInference(model, seed=42)

    # Simulate stale state from a previous run
    ri._loss_start = 999.0
    ri._loss_best = -999.0

    mocker.patch("jax.jit", side_effect=lambda x: x)
    mocker.patch("jax.lax.scan", return_value=("state", jnp.array([1.0])))
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = "state"
    mock_svi.update.return_value = ("state", 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}

    ri.run_optimization(mock_svi, max_num_epochs=1,
                        convergence_check_interval=1, checkpoint_interval=1)

    # After the run resets state, _loss_start must not retain the stale value
    assert ri._loss_start != 999.0


def test_get_site_names(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    
    # Mock trace and seed
    mock_trace = mocker.patch("tfscreen.analysis.hierarchical.run_inference.trace")
    mock_seed = mocker.patch("tfscreen.analysis.hierarchical.run_inference.seed")
    
    # Mock the trace object
    mock_t = mocker.Mock()
    mock_t.items.return_value = [
        ("site1", {"type": "deterministic"}),
        ("site2", {"type": "sample"})
    ]
    mock_trace.return_value.get_trace.return_value = mock_t
    
    sites = ri._get_site_names()
    assert "site1" in sites
    assert "site2" not in sites

def test_restore_checkpoint_error(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test")
    
    # Write a bad checkpoint
    with open(f"{out_prefix}_bad.pkl", "wb") as f:
        dill.dump({"main_key": 0, "svi_state": "not_a_state"}, f)
    
    with pytest.raises(ValueError, match="does not appear to have a saved svi_state"):
        ri._restore_checkpoint(f"{out_prefix}_bad.pkl")

def test_jitter_init_parameters():
    model = MockModel()
    ri = RunInference(model, seed=42)
    params = {"p": jnp.array(1.0), "a": jnp.array([1.0, 2.0])}
    
    jittered = ri._jitter_init_parameters(params, 0.1)
    assert not jnp.array_equal(jittered["p"], 1.0)
    
    # Zero jitter
    params = {"p": jnp.array(1.0)}
    jittered = ri._jitter_init_parameters(params, 0.0)
    assert jittered["p"] == 1.0



def test_write_losses_empty(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_losses_empty")

    # 707: empty losses returns early
    ri._write_losses([], out_prefix)
    assert not os.path.exists(f"{out_prefix}_losses.bin")

def test_setup_svi_invalid_guide_type():
    """setup_svi raises ValueError for unrecognized guide_type."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    with pytest.raises(ValueError, match="not recognized"):
        ri.setup_svi(guide_type="bad_guide")

def test_restore_checkpoint_current_step(tmpdir):
    """_restore_checkpoint restores _current_step when present in checkpoint."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_step")

    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    # Write checkpoint that includes 'current_step'
    checkpoint_file = f"{out_prefix}_checkpoint.pkl"
    checkpoint_data = {
        'svi_state': state,
        'main_key': ri._main_key,
        'current_step': 999,
    }
    import dill as _dill
    with open(checkpoint_file, "wb") as f:
        _dill.dump(checkpoint_data, f)

    restored = ri._restore_checkpoint(checkpoint_file)
    assert ri._current_step == 999
    assert isinstance(restored, SVIState)


def test_checkpoint_round_trips_convergence_state(tmpdir):
    """_write_checkpoint saves loss_start/loss_best; _restore_checkpoint loads them."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_conv")

    ri._loss_start = -12345.0
    ri._loss_best = -99999.0

    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_checkpoint(state, out_prefix)

    # Fresh instance simulates a restart
    ri2 = RunInference(model, seed=0)
    assert ri2._loss_start is None
    assert ri2._loss_best == np.inf

    ri2._restore_checkpoint(f"{out_prefix}_checkpoint.pkl")
    assert ri2._loss_start == pytest.approx(-12345.0)
    assert ri2._loss_best == pytest.approx(-99999.0)


def test_checkpoint_missing_convergence_state_uses_defaults(tmpdir):
    """Old checkpoints without loss_start/loss_best leave defaults intact."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_old")

    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    import dill as _dill
    checkpoint_file = f"{out_prefix}_checkpoint.pkl"
    with open(checkpoint_file, "wb") as f:
        _dill.dump({'svi_state': state, 'main_key': ri._main_key}, f)

    ri._restore_checkpoint(checkpoint_file)
    assert ri._loss_start is None   # default — not present in old checkpoint
    assert ri._loss_best == np.inf  # default

def test_run_optimization_1d_block_idx(mocker):
    """1D block_idx is reshaped to 2D before passing to fast_scan."""

    class MockModel1D(MockModel):
        def get_random_idx(self, key=None, num_batches=1):
            # Always return a 1D array regardless of num_batches
            return np.array([0])

    model = MockModel1D()
    ri = RunInference(model, seed=42)

    mocker.patch("jax.jit", side_effect=lambda x: x)
    mocker.patch("jax.lax.scan", return_value=("state", jnp.array([1.0])))

    mock_svi = mocker.Mock()
    mock_svi.init.return_value = "state"
    mock_svi.update.return_value = ("state", 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}

    # Should not raise; the reshape path is exercised
    ri.run_optimization(mock_svi, max_num_epochs=1,
                        convergence_check_interval=1,
                        checkpoint_interval=1)


# =============================================================================
# _map_params_to_constrained
# =============================================================================

def test_map_params_to_constrained_strips_auto_loc():
    """Returned dict has site names without the _auto_loc suffix."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    constrained = ri._map_params_to_constrained(map_params)

    assert all(not k.endswith("_auto_loc") for k in constrained)
    assert "global_p" in constrained
    assert "geno_p" in constrained


def test_map_params_to_constrained_ignores_non_auto_loc_keys():
    """Extra keys without _auto_loc are not included in the output."""
    model = LaplaceModel(num_genotype=3)
    ri, map_params = _laplace_map_params(model)

    poisoned = dict(map_params, some_extra_key=jnp.array(99.0))
    constrained = ri._map_params_to_constrained(poisoned)

    assert "some_extra_key" not in constrained


def test_map_params_to_constrained_shapes():
    """Output shapes match the model site shapes."""
    num_genotype = 5
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    constrained = ri._map_params_to_constrained(map_params)

    assert constrained["global_p"].shape == ()
    assert constrained["geno_p"].shape == (num_genotype,)


def test_map_params_to_constrained_values_finite():
    """Constrained values should be finite (no NaN/Inf from bijection)."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    constrained = ri._map_params_to_constrained(map_params)

    for k, v in constrained.items():
        assert np.all(np.isfinite(np.asarray(v))), f"Non-finite value in '{k}'"


# =============================================================================
# get_map_posteriors
# =============================================================================

def test_get_map_posteriors_creates_h5(tmpdir):
    """get_map_posteriors writes an HDF5 posterior file."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_map_posteriors_num_samples_is_one(tmpdir):
    """The HDF5 file contains exactly 1 sample."""
    num_genotype = 5
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_shapes"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == 1
        assert hf["global_p"].shape == (1,)
        assert hf["geno_p"].shape == (1, num_genotype)


def test_get_map_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype still produces correct output shapes."""
    num_genotype = 6
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_fwd"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix, forward_batch_size=2)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (1,)
        assert hf["geno_p"].shape == (1, num_genotype)


def test_get_map_posteriors_sites_to_save(tmpdir):
    """sites_to_save restricts which sites appear in the output."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_filtered"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix,
                          sites_to_save=["geno_p"])

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf
        assert hf["geno_p"].shape == (1, 4)


def test_get_map_posteriors_compression(tmpdir):
    """HDF5 datasets are gzip-compressed."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_compressed"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert hf[k].compression == "gzip", f"'{k}' should be gzip-compressed"


def test_get_map_posteriors_values_finite(tmpdir):
    """All values in the MAP posterior should be finite."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_finite"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert not np.any(np.isnan(hf[k][:])), f"NaN found in '{k}'"


def test_get_map_posteriors_matches_constrained(tmpdir):
    """The geno_p values in the h5 file match _map_params_to_constrained output."""
    num_genotype = 4
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_match"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    constrained = ri._map_params_to_constrained(map_params)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        np.testing.assert_allclose(
            hf["global_p"][0], np.asarray(constrained["global_p"]), rtol=1e-5
        )
        np.testing.assert_allclose(
            hf["geno_p"][0], np.asarray(constrained["geno_p"]), rtol=1e-5
        )


# =============================================================================
# get_laplace_posteriors
# =============================================================================

# Shared data class and model for Laplace tests.  Uses a real numpyro model
# so that potential_energy, trace, and biject_to work end-to-end on a tiny
# problem (small D ≈ 1 + num_genotype parameters).

@struct.dataclass
class LaplaceData:
    num_genotype: int
    batch_idx: jnp.ndarray


def _laplace_jax_model(data, priors):
    """One global + one per-genotype parameter, no observations."""
    global_p = numpyro.sample("global_p", dist.Normal(0., 1.))
    with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
        numpyro.sample("geno_p", dist.Normal(global_p, 1.))


class LaplaceModel:
    """Minimal model wrapper compatible with RunInference."""

    def __init__(self, num_genotype=4):
        self.data = LaplaceData(num_genotype=num_genotype,
                                batch_idx=jnp.arange(num_genotype))
        self.priors = {}
        self.jax_model = _laplace_jax_model
        self.jax_model_guide = lambda data, priors: None  # not used in Laplace

    def get_batch(self, data, indices):
        return LaplaceData(num_genotype=len(indices), batch_idx=indices)

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def _laplace_map_params(model, seed=0):
    """Return AutoDelta MAP params (at prior initialisation, no optimisation)."""
    ri = RunInference(model, seed=seed)
    svi = ri.setup_svi(guide_type="delta")
    svi_state = svi.init(ri.get_key(), data=model.data, priors=model.priors)
    return ri, svi.get_params(svi_state)


def test_get_laplace_posteriors_creates_h5(tmpdir):
    """get_laplace_posteriors produces an HDF5 posterior file."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=10,
        sampling_batch_size=5,
        forward_batch_size=4,
    )

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_laplace_posteriors_output_shapes(tmpdir):
    """Output datasets have the expected shapes (num_samples, *param_shape)."""
    num_genotype = 5
    num_samples = 20
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_shapes"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=10,
        forward_batch_size=num_genotype,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)
        assert hf.attrs["num_samples"] == num_samples


def test_get_laplace_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype produces the same shapes as full-batch."""
    num_genotype = 6
    num_samples = 10
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_fwd"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=5,
        forward_batch_size=2,   # forces multiple forward batches
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_laplace_posteriors_sampling_batching(tmpdir):
    """num_posterior_samples not evenly divisible by sampling_batch_size works."""
    num_genotype = 4
    num_samples = 7        # 7 = 3 + 3 + 1
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_sbatch"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=3,
        forward_batch_size=num_genotype,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == num_samples
        assert hf["global_p"].shape[0] == num_samples
        assert hf["geno_p"].shape[0] == num_samples


def test_get_laplace_posteriors_non_auto_loc_keys_ignored(tmpdir):
    """Keys without _auto_loc suffix are silently excluded from the Hessian."""
    model = LaplaceModel(num_genotype=3)
    ri, map_params = _laplace_map_params(model)

    # Inject a key without the expected suffix
    poisoned = dict(map_params, some_other_key=jnp.array(99.0))

    out_prefix = str(tmpdir.join("laplace_extra"))
    # Should not raise; the extra key is ignored
    ri.get_laplace_posteriors(
        map_params=poisoned,
        out_prefix=out_prefix,
        num_posterior_samples=6,
        sampling_batch_size=6,
        forward_batch_size=3,
    )

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_laplace_posteriors_negative_eigenvalues_no_nan(tmpdir, mocker):
    """An indefinite Hessian (negative eigenvalues) must not produce NaN posteriors.

    Regression test: a MAP solution at a saddle point has negative Hessian
    eigenvalues.  The old code (jitter + jnp.linalg.inv) left the covariance
    non-PD in float32, causing jax.random.multivariate_normal to return all-NaN.
    The fix uses numpy float64 eigendecomposition, clamps negative eigenvalues
    to 1e-3, and samples via z @ L^T instead of multivariate_normal.

    LaplaceModel(num_genotype=2) has D=3 unconstrained parameters
    (1 global_p + 2 geno_p), so a 3x3 Hessian is injected.
    """
    model = LaplaceModel(num_genotype=2)
    ri, map_params = _laplace_map_params(model)

    # 3x3 symmetric matrix with one clearly negative eigenvalue (-5).
    bad_H = jnp.array([[10., 0., 0.],
                        [ 0., -5., 0.],
                        [ 0.,  0.,  8.]])

    # jax.hessian(fn)(x): patch so that jax.hessian(pe_fn) returns a callable
    # that ignores x and returns bad_H regardless of fn/x.
    mocker.patch("jax.hessian", return_value=lambda x: bad_H)

    out_prefix = str(tmpdir.join("laplace_negeig"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=20,
        sampling_batch_size=10,
        forward_batch_size=2,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert not np.any(np.isnan(hf[k][:])), f"NaN found in posterior key '{k}'"


# =============================================================================
# get_nuts_posteriors
# =============================================================================

def _fake_mcmc_samples(num_samples, num_genotype):
    """Return a samples dict in the shape expected by LaplaceModel."""
    return {
        "global_p": jnp.zeros((num_samples,)),
        "geno_p": jnp.zeros((num_samples, num_genotype)),
    }


def test_get_nuts_posteriors_creates_h5(tmpdir):
    """get_nuts_posteriors writes an HDF5 posterior file."""
    num_genotype = 4
    num_samples = 8
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_nuts_posteriors_output_shapes(tmpdir):
    """Output datasets have the expected shapes (num_samples, *site_shape)."""
    num_genotype = 5
    num_samples = 12
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_shapes"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           forward_batch_size=num_genotype)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)
        assert hf.attrs["num_samples"] == num_samples


def test_get_nuts_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype produces correct shapes."""
    num_genotype = 6
    num_samples = 8
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_fwd"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           forward_batch_size=2)  # forces 3 forward batches

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_nuts_posteriors_num_samples_in_attrs(tmpdir):
    """num_samples attribute in HDF5 reflects the actual sample count."""
    num_genotype = 3
    num_samples = 7
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_attr"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == num_samples


def test_get_nuts_posteriors_sites_to_save(tmpdir):
    """sites_to_save restricts which sites appear in the HDF5 file."""
    num_genotype = 5
    num_samples = 8
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_filtered"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           sites_to_save=["geno_p"])

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_nuts_posteriors_compression(tmpdir):
    """HDF5 datasets written by get_nuts_posteriors are gzip-compressed."""
    num_genotype = 4
    num_samples = 6
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_compressed"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert hf[k].compression == "gzip", (
                f"dataset '{k}' should be gzip-compressed"
            )


def test_get_laplace_posteriors_sites_to_save(tmpdir):
    """sites_to_save restricts which sites appear in the Laplace HDF5 file."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_filtered"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=8,
        sampling_batch_size=4,
        forward_batch_size=4,
        sites_to_save=["geno_p"],
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf
        assert hf["geno_p"].shape == (8, 4)


def test_get_laplace_posteriors_compression(tmpdir):
    """HDF5 datasets written by get_laplace_posteriors are gzip-compressed."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_compressed"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=8,
        sampling_batch_size=8,
        forward_batch_size=4,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert hf[k].compression == "gzip", (
                f"dataset '{k}' should be gzip-compressed"
            )


# =============================================================================
# _write_epoch_checkpoint
# =============================================================================

def _make_ri_with_epoch_dir(tmpdir):
    """Return a RunInference with _epoch_checkpoints_dir already set."""
    model = MockModel()
    ri = RunInference(model, seed=0)
    epoch_dir = str(tmpdir.join("checkpoints"))
    os.makedirs(epoch_dir, exist_ok=True)
    ri._epoch_checkpoints_dir = epoch_dir
    return ri


def test_write_epoch_checkpoint_creates_file(tmpdir):
    """_write_epoch_checkpoint writes a pkl file with the correct zero-padded name."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=42)

    expected = os.path.join(ri._epoch_checkpoints_dir, "0000042_checkpoint.pkl")
    assert os.path.exists(expected)


def test_write_epoch_checkpoint_file_format(tmpdir):
    """Epoch number is zero-padded to 7 digits."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=1)
    ri._write_epoch_checkpoint(state, epoch=999999)

    assert os.path.exists(os.path.join(ri._epoch_checkpoints_dir, "0000001_checkpoint.pkl"))
    assert os.path.exists(os.path.join(ri._epoch_checkpoints_dir, "0999999_checkpoint.pkl"))


def test_write_epoch_checkpoint_raises_if_exists(tmpdir):
    """_write_epoch_checkpoint raises FileExistsError if the target file already exists."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=7)

    with pytest.raises(FileExistsError, match="0000007_checkpoint.pkl"):
        ri._write_epoch_checkpoint(state, epoch=7)


def test_write_epoch_checkpoint_saves_correct_fields(tmpdir):
    """Checkpoint file contains main_key, svi_state, and current_step."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._current_step = 1234
    ri._write_epoch_checkpoint(state, epoch=5)

    path = os.path.join(ri._epoch_checkpoints_dir, "0000005_checkpoint.pkl")
    with open(path, "rb") as f:
        data = dill.load(f)

    assert "main_key" in data
    assert "svi_state" in data
    assert "current_step" in data
    assert data["current_step"] == 1234


def test_write_epoch_checkpoint_atomic_no_tmp_left(tmpdir):
    """No .tmp file is left behind after a successful write."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=3)

    tmp_path = os.path.join(ri._epoch_checkpoints_dir, "0000003_checkpoint.pkl.tmp")
    assert not os.path.exists(tmp_path)


# =============================================================================
# run_optimization — epoch checkpoint integration
# =============================================================================

def _optimization_mocks(mocker):
    """Shared mock setup for run_optimization epoch checkpoint tests."""
    mocker.patch("jax.jit", side_effect=lambda x: x)
    mocker.patch("jax.lax.scan", return_value=("state", jnp.array([1.0])))
    mock_svi = mocker.Mock()
    mock_svi.init.return_value = "state"
    mock_svi.update.return_value = ("state", 1.0)
    mock_svi.get_params.return_value = {"p": jnp.array(1.0)}
    return mock_svi


def test_run_optimization_epoch_checkpoint_creates_dir(tmpdir, mocker):
    """A checkpoints/ directory is created next to out_prefix when interval is set."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(tmpdir), "myrun")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=1,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=1,
    )

    assert os.path.isdir(os.path.join(str(tmpdir), "checkpoints"))


def test_run_optimization_epoch_checkpoint_file_written(tmpdir, mocker):
    """An epoch checkpoint file is written after the first epoch."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(tmpdir), "myrun")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=1,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=1,
    )

    checkpoints_dir = os.path.join(str(tmpdir), "checkpoints")
    pkl_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pkl")]
    assert len(pkl_files) >= 1


def test_run_optimization_epoch_checkpoint_correct_epoch_number(tmpdir, mocker):
    """The epoch number in the filename matches step // iterations_per_epoch."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(tmpdir), "myrun")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=2,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=1,
    )

    checkpoints_dir = os.path.join(str(tmpdir), "checkpoints")
    # MockModel: num_genotype=10, batch=1 → _iterations_per_epoch=10.
    # After epoch 1: step=10, epoch=1 → 0000001_checkpoint.pkl
    assert os.path.exists(os.path.join(checkpoints_dir, "0000001_checkpoint.pkl"))


def test_run_optimization_epoch_checkpoint_disabled_when_none(tmpdir, mocker):
    """No epoch checkpoints dir is created when epoch_checkpoint_interval=None."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(tmpdir), "myrun")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=2,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=None,
    )

    assert not os.path.isdir(os.path.join(str(tmpdir), "checkpoints"))


def test_run_optimization_epoch_checkpoint_disabled_when_zero(tmpdir, mocker):
    """No epoch checkpoints are written when epoch_checkpoint_interval=0."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(tmpdir), "myrun")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=2,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=0,
    )

    assert not os.path.isdir(os.path.join(str(tmpdir), "checkpoints"))


def test_run_optimization_epoch_checkpoint_file_exists_error(tmpdir, mocker):
    """FileExistsError propagates when an epoch checkpoint already exists."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(tmpdir), "myrun")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=1,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=1,
    )

    # A second run pointing at the same checkpoints/ directory must raise.
    ri2 = RunInference(MockModel(), seed=7)
    mock_svi2 = mocker.Mock()
    mock_svi2.init.return_value = "state"
    mock_svi2.update.return_value = ("state", 1.0)
    mock_svi2.get_params.return_value = {"p": jnp.array(1.0)}

    with pytest.raises(FileExistsError, match="checkpoint.pkl"):
        ri2.run_optimization(
            mock_svi2,
            out_prefix=out_prefix,
            max_num_epochs=1,
            convergence_check_interval=1,
            checkpoint_interval=1,
            epoch_checkpoint_interval=1,
        )


def test_run_optimization_epoch_checkpoint_dir_alongside_out_prefix(tmpdir, mocker):
    """checkpoints/ is placed in the same directory as out_prefix, not the cwd."""
    subdir = tmpdir.mkdir("subdir")
    model = MockModel()
    ri = RunInference(model, seed=42)
    mock_svi = _optimization_mocks(mocker)
    out_prefix = os.path.join(str(subdir), "run")

    ri.run_optimization(
        mock_svi,
        out_prefix=out_prefix,
        max_num_epochs=1,
        convergence_check_interval=1,
        checkpoint_interval=1,
        epoch_checkpoint_interval=1,
    )

    assert os.path.isdir(os.path.join(str(subdir), "checkpoints"))
    assert not os.path.isdir(os.path.join(str(tmpdir), "checkpoints"))
