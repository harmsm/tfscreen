"""RunInference.full_batch_loss on a real (small) model."""
import os

import numpy as np
import pytest

from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

_SMOKE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                      "smoke-tests", "test_data")


@pytest.fixture(scope="module")
def ri_and_point():
    orch = ModelOrchestrator(
        growth_df=os.path.join(_SMOKE, "growth-smoke.csv"),
        binding_df=os.path.join(_SMOKE, "binding-smoke.csv"),
        theta="hill_geno", growth_likelihood="counts", sample_offset="level",
        batch_size=7)
    ri = RunInference(orch, 0)
    svi = ri.setup_svi(guide_type="delta")
    state = svi.init(ri.get_key(), priors=orch.priors,
                     data=orch.get_batch(orch.data, orch.get_random_idx()))
    values = {k[:-len("_auto_loc")]: v
              for k, v in svi.get_params(state).items()}
    return ri, values


def test_full_batch_loss_does_not_depend_on_chunking(ri_and_point):
    """Binding data (not sliced by genotype) and the library-sized latents
    repeat in every chunk; they must be counted once."""
    ri, values = ri_and_point
    G = ri.model.data.num_genotype
    one = ri.full_batch_loss(values, chunk_size=G)
    for chunk in (7, 13, G - 1):
        assert ri.full_batch_loss(values, chunk_size=chunk) == \
            pytest.approx(one, rel=1e-6)
    assert np.isfinite(one)


def test_full_batch_loss_moves_with_the_point(ri_and_point):
    ri, values = ri_and_point
    moved = dict(values)
    key = next(k for k in values if np.size(values[k]) > 1
               and "offset" in k)
    moved[key] = values[key] + 0.5
    assert ri.full_batch_loss(moved) != pytest.approx(ri.full_batch_loss(values))
