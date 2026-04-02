import pytest
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from dataclasses import dataclass
from tfscreen.analysis.hierarchical.run_inference import RunInference
import h5py


@dataclass
class MockData:
    num_genotype: int
    batch_size: int
    batch_idx: torch.Tensor


class MockModel:
    def __init__(self, num_genotype=10, batch_size=None):
        if batch_size is None:
            batch_size = num_genotype
        self.data = MockData(
            num_genotype=num_genotype,
            batch_size=batch_size,
            batch_idx=torch.arange(batch_size)
        )
        self.priors = {}
        self.init_params = {}

    def pyro_model(self, data, priors):
        # Global parameter (no plate)
        pyro.sample("global_p", dist.Normal(0, 1))

        # Local parameter (genotype specific)
        with pyro.plate("model_genotype_plate", data.num_genotype, dim=-1):
            p = pyro.sample("geno_p", dist.Normal(0, 1))

        # Deterministic outside plate but matching genotype size
        pyro.deterministic("geno_p_det", torch.as_tensor(p * 2.0))

        # Matrix parameter (titrant x genotype)
        num_titrants = 3
        with pyro.plate("titrant_plate", num_titrants, dim=-2):
            with pyro.plate("matrix_genotype_plate", data.num_genotype, dim=-1):
                mp = pyro.sample("matrix_p", dist.Normal(0, 1))
                pyro.deterministic("det_p_in", torch.ones((num_titrants, data.num_genotype)))

        # Deterministic outside both plates but matching genotype size
        pyro.deterministic("det_p_out", torch.ones((num_titrants, data.num_genotype)))

    def pyro_model_guide(self, data, priors):
        pass  # AutoDelta guide used via setup_svi

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)

    def get_batch(self, data, indices):
        return MockData(
            num_genotype=len(indices),
            batch_size=len(indices),
            batch_idx=torch.as_tensor(indices)
        )


def test_get_genotype_dim_map():
    model = MockModel(num_genotype=5)
    ri = RunInference(model, seed=42)
    dim_map = ri._get_genotype_dim_map()

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
    pyro.clear_param_store()
    # One step to initialize param store
    svi.step(data=model.data, priors=model.priors)
    svi_state = pyro.get_param_store().get_state()

    out_root = str(tmpdir.join("test"))

    # Test 1: batch_size == num_genotypes
    ri.get_posteriors(svi, svi_state, out_root,
                      num_posterior_samples=10,
                      sampling_batch_size=5,
                      forward_batch_size=10)

    with h5py.File(f"{out_root}_posterior.h5", 'r') as post:
        assert post["global_p"].shape == (10,)
        assert post["geno_p"].shape == (10, 10)   # (samples, genotypes)
        assert post["geno_p_det"].shape == (10, 10)
        assert post["matrix_p"].shape == (10, 3, 10)  # (samples, titrants, genotypes)

        geno_p = post["geno_p"][:]
        matrix_p = post["matrix_p"][:]

    # Test 2: forward batching (forward_batch_size=3)
    ri.get_posteriors(svi, svi_state, out_root + "_batched",
                      num_posterior_samples=10,
                      sampling_batch_size=5,
                      forward_batch_size=3)

    with h5py.File(f"{out_root}_batched_posterior.h5", 'r') as post_batched:
        assert post_batched["global_p"].shape == (10,)
        assert post_batched["geno_p"].shape == (10, 10)
        assert post_batched["matrix_p"].shape == (10, 3, 10)

        # AutoDelta is deterministic: values should match across forward batch sizes
        np.testing.assert_allclose(geno_p, post_batched["geno_p"][:], rtol=1e-5)
        np.testing.assert_allclose(matrix_p, post_batched["matrix_p"][:], rtol=1e-5)


def test_get_posteriors_shape_ambiguity(tmpdir):
    """num_genotypes == num_titrants — genotype dim must still be identified correctly."""
    num_genotypes = 3

    class AmbiguousModel(MockModel):
        def pyro_model(self, data, priors):
            with pyro.plate("titrant_plate", 3, dim=-2):
                with pyro.plate("ambiguous_genotype_plate", data.num_genotype, dim=-1):
                    pyro.sample("ambiguous_p", dist.Normal(0, 1))

    model = AmbiguousModel(num_genotype=num_genotypes)
    ri = RunInference(model, seed=42)

    svi = ri.setup_svi(guide_type="delta")
    pyro.clear_param_store()
    svi.step(data=model.data, priors=model.priors)
    svi_state = pyro.get_param_store().get_state()

    out_root = str(tmpdir.join("test_ambiguous"))

    ri.get_posteriors(svi, svi_state, out_root,
                      num_posterior_samples=4,
                      sampling_batch_size=2,
                      forward_batch_size=1)

    with h5py.File(f"{out_root}_posterior.h5", 'r') as post:
        assert post["ambiguous_p"].shape == (4, 3, 3)  # (samples, titrants, genotypes)


def test_get_posteriors_manual_guide_indexing(tmpdir):
    """Guide with manually-indexed params — get_posteriors should not crash."""
    num_genotypes = 10

    class ManualGuideModel(MockModel):
        def pyro_model(self, data, priors):
            with pyro.plate("mg_genotype_plate", data.num_genotype, dim=-1):
                pyro.sample("geno_p", dist.Normal(0, 1))

        def pyro_model_guide(self, data, priors):
            locs = pyro.param("local_p_locs", torch.zeros(data.num_genotype))
            scales = pyro.param("local_p_scales", torch.ones(data.num_genotype),
                                constraint=torch.distributions.constraints.positive)
            with pyro.plate("mg_genotype_plate", data.num_genotype, dim=-1):
                pyro.sample("geno_p", dist.Normal(locs, scales))

    model = ManualGuideModel(num_genotype=num_genotypes)
    ri = RunInference(model, seed=42)

    svi = ri.setup_svi(guide_type="component")
    pyro.clear_param_store()
    svi.step(data=model.data, priors=model.priors)
    svi_state = pyro.get_param_store().get_state()

    out_root = str(tmpdir.join("test_manual"))

    ri.get_posteriors(svi, svi_state, out_root,
                      num_posterior_samples=4,
                      sampling_batch_size=2,
                      forward_batch_size=5)

    with h5py.File(f"{out_root}_posterior.h5", 'r') as post:
        assert post["geno_p"].shape == (4, 10)
