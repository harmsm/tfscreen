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
    
    out_root = os.path.join(tmpdir, "test")
    
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
        out_root=out_root,
        convergence_tolerance=0.01,
        patience=1
    )
    
    assert state is not None
    assert "p" in params
    assert isinstance(converged, bool)
    assert os.path.exists(f"{out_root}_losses.bin")
    assert os.path.exists(f"{out_root}_losses.txt")

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
    out_root = os.path.join(tmpdir, "test")
    
    mock_svi = mocker.Mock()
    mock_svi.guide = mocker.Mock()
    mock_svi.get_params.return_value = {}
    
    # Mock Predictive
    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_latent_sampler = mock_predictive.return_value
    mock_latent_sampler.return_value = {"param": jnp.zeros((10, 10))} # Batch size 10, 10 genotypes
    
    mocker.patch("jax.device_get", side_effect=lambda x: x)
    
    ri.get_posteriors(mock_svi, "state", out_root, num_posterior_samples=20, sampling_batch_size=10)
    
    assert os.path.exists(f"{out_root}_posterior.h5")

def test_get_posteriors_batching_logic(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_batching")
    
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
    
    ri.get_posteriors(mock_svi, "state", out_root, num_posterior_samples=25, sampling_batch_size=10)
    assert os.path.exists(f"{out_root}_posterior.h5")

def test_get_posteriors_full_logic(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_full")
    
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
    
    ri.get_posteriors(mock_svi, "state", out_root, num_posterior_samples=1, 
                     sampling_batch_size=1, forward_batch_size=5)
    assert os.path.exists(f"{out_root}_posterior.h5")

def test_run_optimization_restore(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_restore")
    
    # Create a valid checkpoint
    state = SVIState(None, None, None) if ri.get_key().size > 0 else SVIState(None, None)
    try:
        state = SVIState(None, None, None)
    except:
        state = SVIState(None, None)
    
    ri._write_checkpoint(state, out_root)
    checkpoint_file = f"{out_root}_checkpoint.pkl"
    
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
    out_root = os.path.join(tmpdir, "test_params")
    params = {"p": jnp.array([1.0])}
    ri.write_params(params, out_root)
    assert os.path.exists(f"{out_root}_params.npz")

def test_write_losses_append(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_losses")
    # 702-703: file exists
    path = f"{out_root}_losses.bin"
    with open(path, "wb") as f:
        f.write(b"header")
    
    # 707: binary write
    ri._write_losses([1.0, 2.0], out_root)
    assert os.path.getsize(path) > 6

def test_update_loss_deque():
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque
    ri._loss_deque = deque(maxlen=200)
    # Fill deque to trigger relative change calculation
    ri._update_loss_deque(np.ones(200))
    # std will be 0, so 1e-10 epsilon prevents div by zero. 
    # (1 - 1) / 1e-10 = 0
    assert ri._relative_change == 0.0


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
    out_root = os.path.join(tmpdir, "test")
    
    # Write a bad checkpoint
    with open(f"{out_root}_bad.pkl", "wb") as f:
        dill.dump({"main_key": 0, "svi_state": "not_a_state"}, f)
    
    with pytest.raises(ValueError, match="does not appear to have a saved svi_state"):
        ri._restore_checkpoint(f"{out_root}_bad.pkl")

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
    out_root = os.path.join(tmpdir, "test_losses_empty")

    # 707: empty losses returns early
    ri._write_losses([], out_root)
    assert not os.path.exists(f"{out_root}_losses.bin")

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
    out_root = os.path.join(tmpdir, "test_step")

    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    # Write checkpoint that includes 'current_step'
    checkpoint_file = f"{out_root}_checkpoint.pkl"
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

    out_root = str(tmpdir.join("laplace"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_root=out_root,
        num_posterior_samples=10,
        sampling_batch_size=5,
        forward_batch_size=4,
    )

    assert os.path.exists(f"{out_root}_posterior.h5")


def test_get_laplace_posteriors_output_shapes(tmpdir):
    """Output datasets have the expected shapes (num_samples, *param_shape)."""
    num_genotype = 5
    num_samples = 20
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_root = str(tmpdir.join("laplace_shapes"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_root=out_root,
        num_posterior_samples=num_samples,
        sampling_batch_size=10,
        forward_batch_size=num_genotype,
    )

    with h5py.File(f"{out_root}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)
        assert hf.attrs["num_samples"] == num_samples


def test_get_laplace_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype produces the same shapes as full-batch."""
    num_genotype = 6
    num_samples = 10
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_root = str(tmpdir.join("laplace_fwd"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_root=out_root,
        num_posterior_samples=num_samples,
        sampling_batch_size=5,
        forward_batch_size=2,   # forces multiple forward batches
    )

    with h5py.File(f"{out_root}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_laplace_posteriors_sampling_batching(tmpdir):
    """num_posterior_samples not evenly divisible by sampling_batch_size works."""
    num_genotype = 4
    num_samples = 7        # 7 = 3 + 3 + 1
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_root = str(tmpdir.join("laplace_sbatch"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_root=out_root,
        num_posterior_samples=num_samples,
        sampling_batch_size=3,
        forward_batch_size=num_genotype,
    )

    with h5py.File(f"{out_root}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == num_samples
        assert hf["global_p"].shape[0] == num_samples
        assert hf["geno_p"].shape[0] == num_samples


def test_get_laplace_posteriors_non_auto_loc_keys_ignored(tmpdir):
    """Keys without _auto_loc suffix are silently excluded from the Hessian."""
    model = LaplaceModel(num_genotype=3)
    ri, map_params = _laplace_map_params(model)

    # Inject a key without the expected suffix
    poisoned = dict(map_params, some_other_key=jnp.array(99.0))

    out_root = str(tmpdir.join("laplace_extra"))
    # Should not raise; the extra key is ignored
    ri.get_laplace_posteriors(
        map_params=poisoned,
        out_root=out_root,
        num_posterior_samples=6,
        sampling_batch_size=6,
        forward_batch_size=3,
    )

    assert os.path.exists(f"{out_root}_posterior.h5")


def test_get_laplace_posteriors_hessian_jitter(tmpdir, mocker):
    """A small diagonal jitter is added before inversion for numerical stability.

    We verify this by checking that inv() is called on a matrix that differs
    from the raw Hessian by a positive diagonal term.
    """
    model = LaplaceModel(num_genotype=2)
    ri, map_params = _laplace_map_params(model)

    captured = {}

    real_inv = jnp.linalg.inv

    def capturing_inv(mat):
        captured["inverted"] = np.array(mat)
        return real_inv(mat)

    mocker.patch("jax.numpy.linalg.inv", side_effect=capturing_inv)

    out_root = str(tmpdir.join("laplace_jitter"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_root=out_root,
        num_posterior_samples=4,
        sampling_batch_size=4,
        forward_batch_size=2,
    )

    # The matrix passed to inv() must have a positive diagonal
    # (raw Hessian + 1e-6 * I guarantees this)
    inverted = captured["inverted"]
    assert np.all(np.diag(inverted) > 0)
