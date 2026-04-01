import pytest
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from collections import namedtuple

from tfscreen.analysis.hierarchical.growth_model.components.ln_cfu0.hierarchical import (
    ModelPriors,
    define_model,
    guide,
    get_hyperparameters,
    get_guesses,
    get_priors
)

MockGrowthData = namedtuple("MockGrowthData", [
    "num_replicate",
    "num_condition_pre",
    "num_genotype",
    "batch_size",
    "batch_idx",
    "scale_vector",
    "map_ln_cfu0"
])

@pytest.fixture
def mock_data():
    """
    Provides a mock data object for testing.
    - 2 replicates
    - 2 pre-selection conditions
    - 4 genotypes
    - Batch size 4 (1-to-1 mapping to genotypes for simplicity in this test)
    """
    num_replicate = 2
    num_condition_pre = 2
    num_genotype = 4
    batch_size = 4

    batch_idx = torch.arange(batch_size, dtype=torch.int32)
    scale_vector = torch.ones(batch_size)
    map_ln_cfu0 = torch.arange(batch_size, dtype=torch.int32)

    return MockGrowthData(
        num_replicate=num_replicate,
        num_condition_pre=num_condition_pre,
        num_genotype=num_genotype,
        batch_size=batch_size,
        batch_idx=batch_idx,
        scale_vector=scale_vector,
        map_ln_cfu0=map_ln_cfu0
    )


def test_get_hyperparameters():
    """Tests that get_hyperparameters returns the correct structure."""
    params = get_hyperparameters()
    assert isinstance(params, dict)
    assert "ln_cfu0_hyper_loc_loc" in params
    assert params["ln_cfu0_hyper_loc_loc"] == -2.5

def test_get_priors():
    """Tests that get_priors returns a correctly populated ModelPriors object."""
    priors = get_priors()
    assert isinstance(priors, ModelPriors)
    assert priors.ln_cfu0_hyper_loc_loc == -2.5

def test_get_guesses(mock_data):
    """Tests that get_guesses returns correctly named and shaped guesses."""
    name = "test_ln_cfu0"
    guesses = get_guesses(name, mock_data)

    assert isinstance(guesses, dict)

    assert f"{name}_hyper_loc" in guesses
    assert guesses[f"{name}_hyper_loc"] == -2.5

    assert f"{name}_offset" in guesses
    expected_shape = (mock_data.num_replicate,
                      mock_data.num_condition_pre,
                      mock_data.num_genotype)
    assert guesses[f"{name}_offset"].shape == expected_shape
    assert torch.all(guesses[f"{name}_offset"] == 0.0)

def test_define_model_logic_and_shapes(mock_data):
    """
    Tests the core logic of define_model for the hierarchical ln_cfu0.
    """
    name = "test_ln_cfu0"
    priors = get_priors()

    # Get base guesses (full genotype shape)
    base_guesses = get_guesses(name, mock_data)

    # Slice offsets to batch size for substitution
    batch_guesses = base_guesses.copy()
    full_offsets = base_guesses[f"{name}_offset"]
    batch_offsets = full_offsets[..., mock_data.batch_idx]
    batch_guesses[f"{name}_offset"] = batch_offsets

    substituted_model = poutine.condition(define_model, data=batch_guesses)

    final_ln_cfu0 = substituted_model(name=name, data=mock_data, priors=priors)

    model_trace = poutine.trace(substituted_model).get_trace(
        name=name,
        data=mock_data,
        priors=priors
    )

    # Check Deterministic Site
    assert name in model_trace.nodes
    ln_cfu0_site = model_trace.nodes[name]["value"]

    expected_site_shape = (mock_data.num_replicate,
                           mock_data.num_condition_pre,
                           mock_data.batch_size)
    assert ln_cfu0_site.shape == expected_site_shape

    # With 0 offsets, value should equal hyper_loc
    hyper_loc = base_guesses[f"{name}_hyper_loc"]
    assert torch.allclose(ln_cfu0_site, torch.tensor(float(hyper_loc)))

    # Check Final Expanded Shape: (R, 1, C, 1, 1, 1, Batch)
    expected_expanded_shape = (mock_data.num_replicate,
                               1,
                               mock_data.num_condition_pre,
                               1,
                               1,
                               1,
                               mock_data.batch_size)
    assert final_ln_cfu0.shape == expected_expanded_shape
    assert torch.allclose(final_ln_cfu0, torch.tensor(float(hyper_loc)))

def test_guide_logic_and_shapes(mock_data):
    """
    Tests the guide function shapes, parameter creation, and execution.
    """
    pyro.clear_param_store()

    name = "test_ln_cfu0_guide"
    priors = get_priors()

    torch.manual_seed(0)
    guide_trace = poutine.trace(guide).get_trace(
        name=name,
        data=mock_data,
        priors=priors
    )

    torch.manual_seed(0)
    final_ln_cfu0 = guide(name=name, data=mock_data, priors=priors)

    # Offset locs/scales should cover ALL genotypes: (R, C, Genotype)
    assert f"{name}_offset_locs" in guide_trace.nodes
    offset_locs = guide_trace.nodes[f"{name}_offset_locs"]["value"]

    expected_param_shape = (mock_data.num_replicate,
                            mock_data.num_condition_pre,
                            mock_data.num_genotype)
    assert offset_locs.shape == expected_param_shape

    # Sampled offsets should match the BATCH size: (R, C, Batch)
    assert f"{name}_offset" in guide_trace.nodes
    sampled_offsets = guide_trace.nodes[f"{name}_offset"]["value"]

    expected_sample_shape = (mock_data.num_replicate,
                             mock_data.num_condition_pre,
                             mock_data.batch_size)
    assert sampled_offsets.shape == expected_sample_shape

    # Check Return Shape: (R, 1, C, 1, 1, 1, Batch)
    expected_expanded_shape = (mock_data.num_replicate,
                               1,
                               mock_data.num_condition_pre,
                               1,
                               1,
                               1,
                               mock_data.batch_size)
    assert final_ln_cfu0.shape == expected_expanded_shape
