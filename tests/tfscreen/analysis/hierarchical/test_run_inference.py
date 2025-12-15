import pytest
from unittest.mock import MagicMock, patch, mock_open, call
import jax.numpy as jnp
import numpy as np
import os
from collections import deque

# Import the class to be tested
from tfscreen.analysis.hierarchical.run_inference import RunInference

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_data():
    """Mock data object."""
    data = MagicMock()
    data.num_genotype = 100
    return data

@pytest.fixture
def mock_model(mock_data):
    """
    Mock model object satisfying RunInference requirements.
    """
    model = MagicMock()
    model.data = mock_data
    model.priors = "priors"
    model.jax_model = MagicMock(__name__="jax_model")
    model.jax_model_guide = MagicMock(__name__="jax_model_guide")
    model.init_params = {"alpha": jnp.array(1.0)}
    
    # Mock batch functions
    model.get_random_idx.return_value = jnp.array([0, 1])
    model.get_batch.return_value = "dummy_batch_data"
    
    return model

@pytest.fixture
def run_inference(mock_model):
    """Standard RunInference instance seeded with 42."""
    return RunInference(model=mock_model, seed=42)

@pytest.fixture
def mock_svi_class(mocker):
    return mocker.patch("tfscreen.analysis.hierarchical.run_inference.SVI")

# =============================================================================
# Tests
# =============================================================================

# ----------------------------------------------------------------------------
# Initialization & Setup
# ----------------------------------------------------------------------------

def test_init_validates_model_attributes():
    """Test that __init__ raises ValueError if model attributes are missing."""
    bad_model = MagicMock()
    del bad_model.data # Missing 'data' attribute
    
    with pytest.raises(ValueError, match="must have attribute data"):
        RunInference(bad_model, seed=0)

def test_init_sets_seed_and_key(mock_model):
    """Test that PRNGKey is initialized correctly."""
    with patch("jax.random.PRNGKey") as mock_prng:
        ri = RunInference(mock_model, seed=123)
        mock_prng.assert_called_once_with(123)
        assert ri._current_step == 0

def test_setup_map(run_inference, mock_svi_class):
    """Test setup_map configuration."""
    svi = run_inference.setup_map(adam_step_size=0.01, elbo_num_particles=5)
    
    mock_svi_class.assert_called_once()
    args, kwargs = mock_svi_class.call_args
    assert args[0] == run_inference.model.jax_model
    assert args[1] == run_inference.model.jax_model_guide
    assert kwargs['loss'].num_particles == 5

def test_setup_svi(run_inference, mock_svi_class):
    """Test setup_svi configuration."""
    svi = run_inference.setup_svi(init_params={"x": 1.0}, elbo_num_particles=10)
    
    mock_svi_class.assert_called_once()
    args, kwargs = mock_svi_class.call_args
    assert args[0] == run_inference.model.jax_model
    assert args[1] == run_inference.model.jax_model_guide
    assert kwargs['loss'].num_particles == 10

# ----------------------------------------------------------------------------
# Run Optimization
# ----------------------------------------------------------------------------

@patch("jax.jit")
@patch("jax.device_put")
def test_run_optimization_standard_flow(mock_device_put, mock_jit, run_inference, mock_model):
    """Test the main optimization loop."""
    svi = MagicMock()
    mock_update_fn = MagicMock(return_value=("new_state", 1.5))
    mock_init_fn = MagicMock(return_value="initial_state")
    
    def jit_side_effect(fun):
        if fun == svi.update: return mock_update_fn
        elif fun == svi.init: return mock_init_fn
        return fun 
    
    mock_jit.side_effect = jit_side_effect
    mock_device_put.return_value = "gpu_data"

    run_inference._write_checkpoint = MagicMock()
    run_inference._write_losses = MagicMock()
    svi.get_params.return_value = {"x": np.array([1.0])}
    run_inference._relative_change = 0.0

    state, params, converged = run_inference.run_optimization(
        svi, 
        num_steps=5, 
        checkpoint_interval=2, 
        convergence_window=10
    )

    mock_init_fn.assert_called_once()
    assert mock_update_fn.call_count == 5
    assert run_inference._write_checkpoint.call_count == 4
    assert run_inference._write_losses.call_count == 4
    assert mock_model.get_random_idx.call_count == 6

@patch("jax.jit")
@patch("jax.device_put")
def test_run_optimization_resume_checkpoint(mock_device_put, mock_jit, run_inference):
    """Test resuming from a file path."""
    svi = MagicMock()
    mock_update_fn = MagicMock(return_value=("state", 1.0))
    mock_jit.side_effect = lambda f: mock_update_fn # return callable for all jit calls
    
    run_inference._write_checkpoint = MagicMock()
    run_inference._write_losses = MagicMock()
    run_inference._relative_change = 0.0
    
    with patch.object(run_inference, "_restore_checkpoint", return_value="restored_state") as mock_restore:
        with patch("os.path.isfile", return_value=True):
            run_inference.run_optimization(svi, svi_state="ckpt.pkl", num_steps=1)
            
            mock_restore.assert_called_once_with("ckpt.pkl")
            args, _ = mock_update_fn.call_args
            assert args[0] == "restored_state"

@patch("jax.jit")
@patch("jax.device_put")
def test_run_optimization_explosion(mock_device_put, mock_jit, run_inference):
    """Test RuntimeError is raised if params become NaN."""
    svi = MagicMock()
    mock_update_fn = MagicMock(return_value=("state", 1.0))
    mock_jit.side_effect = lambda f: mock_update_fn

    # Return NaN params to trigger explosion check
    svi.get_params.return_value = {"x": np.array([np.nan])}
    
    run_inference._update_loss_deque = MagicMock()
    run_inference._relative_change = 0.0 
    
    # Use a dummy object for svi_state, NOT a string, to bypass file check
    dummy_state = MagicMock()

    with pytest.raises(RuntimeError, match="model exploded"):
        run_inference.run_optimization(
            svi, 
            svi_state=dummy_state,
            num_steps=10, 
            checkpoint_interval=1
        )

@patch("jax.jit")
@patch("jax.device_put")
def test_run_optimization_convergence(mock_device_put, mock_jit, run_inference):
    """Test that optimization stops early if convergence tolerance is met."""
    svi = MagicMock()
    mock_update_fn = MagicMock(return_value=("state", 0.0))
    mock_jit.side_effect = lambda f: mock_update_fn

    svi.get_params.return_value = {"x": np.array([1.0])}
    
    run_inference._write_checkpoint = MagicMock()
    run_inference._write_losses = MagicMock()
    
    # Manually set relative change to be very small (converged)
    run_inference._relative_change = 1e-9 
    run_inference._update_loss_deque = MagicMock()

    # Use a dummy object for svi_state, NOT a string
    dummy_state = MagicMock()

    state, params, converged = run_inference.run_optimization(
        svi, 
        svi_state=dummy_state,
        num_steps=100, 
        checkpoint_interval=1,
        convergence_tolerance=1e-5
    )
    
    assert converged is True
    assert run_inference._write_checkpoint.call_count >= 1

# ----------------------------------------------------------------------------
# Posteriors & Predictions
# ----------------------------------------------------------------------------

def test_get_posteriors_batching_logic(run_inference, mock_model):
    """Verify the batching logic for large datasets in get_posteriors."""
    mock_model.data.num_genotype = 10
    svi = MagicMock()
    svi.get_params.return_value = {}
    
    with patch("tfscreen.analysis.hierarchical.run_inference.Predictive") as MockPredictive:
        mock_latent_pred = MagicMock()
        mock_latent_pred.return_value = {"alpha": np.zeros((1, 10)), "beta": np.zeros((1,))}
        
        mock_forward_pred = MagicMock()
        mock_forward_pred.side_effect = [
            {"obs": np.zeros((1, 4))}, # Batch 1
            {"obs": np.zeros((1, 4))}, # Batch 2
            {"obs": np.zeros((1, 2))}  # Batch 3
        ]
        
        def predictive_side_effect(*args, **kwargs):
            if 'posterior_samples' in kwargs: return mock_forward_pred
            return mock_latent_pred
            
        MockPredictive.side_effect = predictive_side_effect
        run_inference._write_posteriors = MagicMock()
        
        with patch("jax.device_get", side_effect=lambda x: x):
            run_inference.get_posteriors(
                svi, 
                svi_state=None, 
                out_root="test",
                num_posterior_samples=1,
                sampling_batch_size=1,
                forward_batch_size=4
            )
            
            assert mock_model.get_batch.call_count == 3
            run_inference._write_posteriors.assert_called_once()
            results = run_inference._write_posteriors.call_args[0][0]
            assert results["obs"].shape == (1, 10)

def test_predict(run_inference):
    """Test predict orchestration."""
    posterior_samples = {"a": 1}
    with patch("tfscreen.analysis.hierarchical.run_inference.Predictive") as MockPredictive:
        mock_instance = MockPredictive.return_value
        mock_instance.return_value = {"out": 1}
        
        res = run_inference.predict(posterior_samples, predict_sites=["out"])
        
        MockPredictive.assert_called_once()
        assert MockPredictive.call_args[1]['return_sites'] == ["out"]
        mock_instance.assert_called_once()

def test_predict_loads_file(run_inference):
    """Test predict loads from file if string passed."""
    with patch("jax.numpy.load", return_value={"a": 1}) as mock_load:
        with patch("tfscreen.analysis.hierarchical.run_inference.Predictive"):
            run_inference.predict("posteriors.npz")
            mock_load.assert_called_once_with("posteriors.npz")

# ----------------------------------------------------------------------------
# Helper Methods & IO
# ----------------------------------------------------------------------------

def test_jitter_init_parameters(run_inference):
    """Test jitter logic."""
    params = {"a": jnp.array([1.0, 1.0])}
    
    # Zero jitter
    res = run_inference._jitter_init_parameters(params.copy(), 0)
    assert jnp.array_equal(res["a"], params["a"])
    
    # Positive jitter
    with patch("jax.random.normal", return_value=jnp.array([0.1, -0.1])):
        res = run_inference._jitter_init_parameters(params.copy(), 0.1)
        assert res["a"][0] > 1.0
        assert res["a"][1] < 1.0

def test_update_loss_deque(run_inference):
    """Test deque smoothing and relative change calculation."""
    run_inference._loss_deque = deque(maxlen=4)
    run_inference._update_loss_deque([1.0, 1.0], convergence_window=2)
    assert run_inference._relative_change == np.inf
    run_inference._update_loss_deque([0.5, 0.5], convergence_window=2)
    assert np.isclose(run_inference._relative_change, 0.5)

def test_get_site_names(run_inference):
    """Test extracting site names via trace."""
    with patch("tfscreen.analysis.hierarchical.run_inference.seed") as mock_seed:
        with patch("tfscreen.analysis.hierarchical.run_inference.trace") as mock_trace:
            mock_traced_model = MagicMock()
            mock_trace.return_value = mock_traced_model
            mock_traced_model.get_trace.return_value = {
                "site_a": {"type": "deterministic"},
                "site_b": {"type": "sample"}
            }
            names = run_inference._get_site_names(target_sites="deterministic")
            assert "site_a" in names
            assert "site_b" not in names

def test_write_checkpoint(run_inference):
    """Test atomic checkpoint writing."""
    svi_state = "dummy_state"
    with patch("jax.device_get", return_value=svi_state): 
        with patch("dill.dump") as mock_dill:
            with patch("os.replace") as mock_replace:
                with patch("builtins.open", mock_open()):
                    run_inference._write_checkpoint(svi_state, "root")
                    mock_dill.assert_called_once()
                    mock_replace.assert_called_once_with("root_checkpoint.tmp.pkl", "root_checkpoint.pkl")

def test_restore_checkpoint(run_inference):
    """Test checkpoint restoration."""
    class DummySVIState: pass
    with patch("numpyro.infer.svi.SVIState", DummySVIState):
        fake_state = DummySVIState()
        ckpt_data = {'svi_state': fake_state, 'main_key': "key", 'current_step': 100}
        with patch("dill.load", return_value=ckpt_data):
            with patch("builtins.open", mock_open()):
                res = run_inference._restore_checkpoint("file.pkl")
                assert res == fake_state
                assert run_inference._current_step == 100

def test_write_losses(run_inference):
    """Test binary loss writing."""
    run_inference._current_step = 0
    with patch("os.path.exists", return_value=True):
        with patch("os.remove") as mock_rm:
            with patch("builtins.open", mock_open()) as mock_f:
                mock_f.return_value.fileno.return_value = 1
                run_inference._write_losses([1.0], "root")
                mock_rm.assert_called_once_with("root_losses.bin")