import pytest
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta
from tfscreen.analysis.hierarchical.run_inference import RunInference
from flax import struct
import jax
import h5py

@struct.dataclass
class MockData:
    num_genotype: int
    batch_size: int
    batch_idx: jnp.ndarray

class MockModel:
    def __init__(self, num_genotype=10, batch_size=None):
        if batch_size is None:
            batch_size = num_genotype
        self.data = MockData(num_genotype=num_genotype, batch_size=batch_size, batch_idx=jnp.arange(batch_size))
        self.priors = {}
        
    def jax_model(self, data, priors):
        # Global parameter
        numpyro.sample("global_p", dist.Normal(0, 1))
        
        # Local parameter (genotype specific)
        with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
            p = numpyro.sample("geno_p", dist.Normal(0, 1))
            
        # Deterministic outside plate but matching genotype size
        numpyro.deterministic("geno_p_det", p * 2.0)
            
        # Matrix parameter (e.g. titrant x genotype)
        num_titrants = 3
        with numpyro.plate("titrant_plate", num_titrants, dim=-2):
            with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
                numpyro.sample("matrix_p", dist.Normal(0, 1))
                # Deterministic inside plate
                numpyro.deterministic("det_p_in", jnp.ones((num_titrants, data.num_genotype)))
        
        # Deterministic outside both plates but matching genotype size
        numpyro.deterministic("det_p_out", jnp.ones((num_titrants, data.num_genotype)))

    def jax_model_guide(self, data, priors):
        return AutoDelta(self.jax_model)(data, priors)

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)

    def get_batch(self, data, indices):
        # Simple slicing for Mock
        return MockData(num_genotype=len(indices), batch_size=len(indices), batch_idx=indices)

def test_get_genotype_dim_map():
    model = MockModel(num_genotype=5)
    ri = RunInference(model, seed=42)
    dim_map = ri._get_genotype_dim_map()
    
    # NumPyro normally uses negative indices for dims in plates
    assert dim_map["geno_p"] == -1
    assert dim_map["geno_p_det"] == -1
    assert dim_map["matrix_p"] == -1
    assert dim_map["det_p_in"] == -1
    assert dim_map["det_p_out"] == -1
    assert "global_p" not in dim_map

def test_get_posteriors_batching_mapping(tmpdir):
    num_genotypes = 10
    model = MockModel(num_genotype=num_genotypes)
    ri = RunInference(model, seed=42)
    
    svi = ri.setup_svi(guide_type="delta")
    svi_state = svi.init(ri.get_key(), data=model.data, priors=model.priors)
    
    out_root = str(tmpdir.join("test"))
    
    # Test 1: batch_size == num_genotypes
    ri.get_posteriors(svi, svi_state, out_root, 
                       num_posterior_samples=10, 
                       sampling_batch_size=5, 
                       forward_batch_size=10)
    
    with h5py.File(f"{out_root}_posterior.h5", 'r') as post:
        assert post["global_p"].shape == (10,)
        assert post["geno_p"].shape == (10, 10) # (samples, genotypes)
        assert post["geno_p_det"].shape == (10, 10)
        assert post["matrix_p"].shape == (10, 3, 10) # (samples, titrants, genotypes)
        assert post["det_p_in"].shape == (10, 3, 10)
        assert post["det_p_out"].shape == (10, 3, 10)

        # Cache for comparison
        geno_p = post["geno_p"][:]
        geno_p_det = post["geno_p_det"][:]
        matrix_p = post["matrix_p"][:]
        det_p_in = post["det_p_in"][:]
        det_p_out = post["det_p_out"][:]

    # Test 2: batch_size < num_genotypes (e.g. forward_batch_size=3)
    ri.get_posteriors(svi, svi_state, out_root + "_batched", 
                       num_posterior_samples=10, 
                       sampling_batch_size=5, 
                       forward_batch_size=3)
    
    with h5py.File(f"{out_root}_batched_posterior.h5", 'r') as post_batched:
        assert post_batched["global_p"].shape == (10,)
        assert post_batched["geno_p"].shape == (10, 10)
        assert post_batched["geno_p_det"].shape == (10, 10)
        assert post_batched["matrix_p"].shape == (10, 3, 10)
        assert post_batched["det_p_in"].shape == (10, 3, 10)
        assert post_batched["det_p_out"].shape == (10, 3, 10)
        
        # Verify values match (they should be identical since AutoDelta is deterministic given params)
        np.testing.assert_allclose(geno_p, post_batched["geno_p"][:])
        np.testing.assert_allclose(geno_p_det, post_batched["geno_p_det"][:])
        np.testing.assert_allclose(matrix_p, post_batched["matrix_p"][:])
        np.testing.assert_allclose(det_p_in, post_batched["det_p_in"][:])
        np.testing.assert_allclose(det_p_out, post_batched["det_p_out"][:])

def test_get_posteriors_shape_ambiguity(tmpdir):
    # Test with num_genotypes == num_titrants to ensure it doesn't get "mixed up"
    num_genotypes = 3
    num_titrants = 3
    model = MockModel(num_genotype=num_genotypes)
    # Overwrite model to have ambiguity
    def jax_model_ambiguous(data, priors):
        with numpyro.plate("titrant_plate", 3, dim=-2):
            with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
                # Shape is (3, 3)
                numpyro.sample("ambiguous_p", dist.Normal(0, 1))
    
    model.jax_model = jax_model_ambiguous
    ri = RunInference(model, seed=42)
    
    svi = ri.setup_svi(guide_type="delta")
    svi_state = svi.init(ri.get_key(), data=model.data, priors=model.priors)
    
    out_root = str(tmpdir.join("test_ambiguous"))
    
    ri.get_posteriors(svi, svi_state, out_root, 
                       num_posterior_samples=4, 
                       sampling_batch_size=2, 
                       forward_batch_size=1)
    
    with h5py.File(f"{out_root}_posterior.h5", 'r') as post:
        assert post["ambiguous_p"].shape == (4, 3, 3) # (samples, titrants, genotypes)

def test_get_posteriors_manual_guide_indexing(tmpdir):
    # This test mimics the failure in growth_hierarchical.py where a guide
    # does manual indexing on a parameter using a plate index.
    model = MockModel(num_genotype=10)
    
    # Custom guide with manual indexing (TracerArrayConversionError trigger)
    def manual_guide(data, priors):
        local_p_locs = numpyro.param("local_p_locs", jnp.zeros(10))
        local_p_scales = numpyro.param("local_p_scales", jnp.ones(10))
        
        with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1) as idx:
            # Indexing local_p_locs (which might be NumPy) with idx (which is a JAX tracer in Predictive)
            numpyro.sample("geno_p", dist.Normal(local_p_locs[idx], local_p_scales[idx]))

    model.jax_model_guide = manual_guide
    ri = RunInference(model, seed=42)
    
    # Mock parameters as NumPy arrays (simulating restored checkpoint)
    params = {
        "local_p_locs": np.zeros(10),
        "local_p_scales": np.ones(10)
    }
    
    # We need to wrap the guide to use these params
    svi = ri.setup_svi(guide_type="component")
    
    # Mock svi_state as a simple object that svi.get_params can handle if we mock it
    from unittest.mock import MagicMock
    svi_state = MagicMock()
    
    # Mock get_params to return our NumPy params
    with MagicMock() as mock_svi:
        mock_svi.guide = manual_guide
        mock_svi.get_params.return_value = params
        
        out_root = str(tmpdir.join("test_manual"))
        
        # This should NOT fail now because we device_put the params and data
        ri.get_posteriors(mock_svi, svi_state, out_root, 
                           num_posterior_samples=4, 
                           sampling_batch_size=2, 
                           forward_batch_size=5)
        
        with h5py.File(f"{out_root}_posterior.h5", 'r') as post:
            assert post["geno_p"].shape == (4, 10)

if __name__ == "__main__":
    pytest.main([__file__])
