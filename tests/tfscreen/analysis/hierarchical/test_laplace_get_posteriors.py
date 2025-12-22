import pytest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoDelta
from numpyro.infer import Predictive

import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
sys.path.insert(0, src_path)

from tfscreen.analysis.hierarchical.run_inference import RunInference

def test_get_posteriors_laplace_handling():
    """
    Test that get_posteriors uses guide.sample_posterior for Laplace guides
    and Predictive for other guides.
    """
    
    # Mock Model and its data
    mock_model = MagicMock()
    mock_model.data.num_genotype = 10
    mock_model.priors = {"dummy": 1.0}
    mock_model.jax_model = MagicMock()
    mock_model.jax_model_guide = MagicMock()
    mock_model.init_params = {}
    mock_model.get_batch.return_value = {"data": jnp.zeros(10)}
    mock_model.extract_parameters.return_value = {} # Return empty dict for summaries
    
    # Initialize RunInference with mock model and seed
    ri = RunInference(mock_model, seed=42)
    ri.get_key = MagicMock(return_value=MagicMock()) # Return a dummy key
    
    # 1. Test with AutoLaplaceApproximation
    mock_guide_laplace = MagicMock()
    mock_guide_laplace.__class__ = AutoLaplaceApproximation
    mock_guide_laplace._unpack_latent.side_effect = lambda x: x
    
    # mock_posterior should be returned by get_posterior
    mock_posterior = MagicMock()
    mock_posterior.sample.return_value = {"x": jnp.zeros((5, 10))}
    mock_guide_laplace.get_posterior.return_value = mock_posterior
    
    mock_svi_laplace = MagicMock()
    mock_svi_laplace.guide = mock_guide_laplace
    mock_svi_laplace.get_params.return_value = {"auto_loc": jnp.zeros(5)}
    
    # mock_forward_sampler should be used for the ONE Predictive call (forward pass)
    mock_forward_sampler = MagicMock()
    mock_forward_sampler.return_value = {"obs": jnp.zeros((5, 10))}

    with patch("tfscreen.analysis.hierarchical.run_inference.Predictive", return_value=mock_forward_sampler) as mock_predictive:
        with patch("tfscreen.analysis.hierarchical.run_inference.tqdm", lambda x, **kwargs: x):
            with patch.object(ri, "_write_posteriors"):
                ri.get_posteriors(
                    svi=mock_svi_laplace,
                    svi_state="state",
                    out_root="test_root",
                    num_posterior_samples=5,
                    sampling_batch_size=5,
                    forward_batch_size=10,
                    write_csv=False
                )
            
                # Verify get_posterior was called once
                mock_guide_laplace.get_posterior.assert_called_once()
                # Verify sample was called on the posterior
                mock_posterior.sample.assert_called_once()
            # Verify Predictive was called ONCE (for the forward pass)
            assert mock_predictive.call_count == 1
            # Check the call was for the forward pass (has posterior_samples)
            args, kwargs = mock_predictive.call_args_list[0]
            assert "posterior_samples" in kwargs

    # 2. Test with AutoDelta (or anything else)
    mock_guide_delta = MagicMock()
    mock_guide_delta.__class__ = AutoDelta
    
    mock_svi_delta = MagicMock()
    mock_svi_delta.guide = mock_guide_delta
    mock_svi_delta.get_params.return_value = {"params": jnp.zeros(5)}
    
    # Side effect for Predictive mock to return different samplers
    # 1st call: latent, 2nd call: forward
    mock_latent_sampler = MagicMock()
    mock_latent_sampler.return_value = {"x": jnp.zeros((5, 10))}
    
    mock_forward_sampler = MagicMock()
    mock_forward_sampler.return_value = {"obs": jnp.zeros((5, 10))}
    
    def predictive_side_effect(*args, **kwargs):
        if "posterior_samples" in kwargs:
            return mock_forward_sampler
        return mock_latent_sampler

    with patch("tfscreen.analysis.hierarchical.run_inference.Predictive", side_effect=predictive_side_effect) as mock_predictive:
        with patch("tfscreen.analysis.hierarchical.run_inference.tqdm", lambda x, **kwargs: x):
            with patch.object(ri, "_write_posteriors"):
                ri.get_posteriors(
                    svi=mock_svi_delta,
                    svi_state="state",
                    out_root="test_root",
                    num_posterior_samples=5,
                    sampling_batch_size=5,
                    forward_batch_size=10,
                    write_csv=False
                )
            
            # Verify Predictive was called TWICE (one latent, one forward)
            assert mock_predictive.call_count == 2
            # Verify sample_posterior was NOT called on the guide (unless it's Laplace)
            if hasattr(mock_guide_delta, "sample_posterior"):
                mock_guide_delta.sample_posterior.assert_not_called()

def test_get_posteriors_diagonal_laplace_handling():
    """
    Test that get_posteriors uses the CPU-offload/pre-calc path for 
    AutoDiagonalLaplace.
    """
    from tfscreen.analysis.hierarchical.run_inference import AutoDiagonalLaplace
    
    # Mock Model and its data
    mock_model = MagicMock()
    mock_model.data.num_genotype = 10
    mock_model.priors = {"dummy": jnp.array(1.0)}
    mock_model.jax_model = MagicMock()
    mock_model.jax_model_guide = MagicMock()
    mock_model.init_params = {}
    mock_model.get_batch.return_value = {"data": jnp.zeros(10)}
    mock_model.extract_parameters.return_value = {}
    
    # Initialize RunInference with mock model and seed
    ri = RunInference(mock_model, seed=42)
    ri.get_key = MagicMock(return_value=MagicMock())
    
    # Test with AutoDiagonalLaplace
    mock_guide = MagicMock()
    mock_guide.__class__ = AutoDiagonalLaplace
    mock_guide._unpack_latent.side_effect = lambda x: x
    
    mock_posterior = MagicMock()
    mock_posterior.sample.return_value = {"x": jnp.zeros((5, 10))}
    mock_guide.get_posterior.return_value = mock_posterior
    
    mock_svi = MagicMock()
    mock_svi.guide = mock_guide
    mock_svi.get_params.return_value = {"auto_loc": jnp.zeros(5)}
    
    mock_forward_sampler = MagicMock()
    mock_forward_sampler.return_value = {"obs": jnp.zeros((5, 10))}

    with patch("tfscreen.analysis.hierarchical.run_inference.Predictive", return_value=mock_forward_sampler) as mock_predictive:
        with patch("tfscreen.analysis.hierarchical.run_inference.tqdm", lambda x, **kwargs: x):
            with patch.object(ri, "_write_posteriors"):
                ri.get_posteriors(
                    svi=mock_svi,
                    svi_state="state",
                    out_root="test_root",
                    num_posterior_samples=5,
                    sampling_batch_size=5,
                    forward_batch_size=10,
                    write_csv=False
                )
            
            # Verify get_posterior was called (indicates CPU offload path)
            mock_guide.get_posterior.assert_called_once()
            # Verify sample was called on the posterior
            mock_posterior.sample.assert_called_once()
            # Verify Predictive was called ONCE (for the forward pass)
            assert mock_predictive.call_count == 1
