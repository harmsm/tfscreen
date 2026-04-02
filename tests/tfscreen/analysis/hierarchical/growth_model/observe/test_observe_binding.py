import pytest
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.observe.binding import observe, guide

# --- Mock Data Fixture ---

MockBindingData = namedtuple("MockBindingData", [
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "good_mask",
    "theta_std",
    "theta_obs",
    "batch_size",
    "scale_vector"
])

@pytest.fixture
def mock_data():
    """
    Provides mock data for the binding observation.
    - Shape: (2 titrant_names, 3 titrant_concs, 4 genotypes)
    - A mask is applied to half the data.
    """
    shape = (2, 3, 4)  # (titrant_name, titrant_conc, genotype)

    # Create a mask where half the data is good (True)
    mask_array = torch.ones(shape, dtype=torch.bool)
    mask_array[0, :, :] = False  # Mask first titrant_name

    batch_size = shape[2]  # Full batch for basic test
    scale_vector = torch.ones(batch_size)

    return MockBindingData(
        num_titrant_name=shape[0],
        num_titrant_conc=shape[1],
        num_genotype=shape[2],
        good_mask=mask_array,
        theta_std=torch.ones(shape) * 0.1,
        theta_obs=torch.ones(shape) * 0.5,
        batch_size=batch_size,
        scale_vector=scale_vector
    )

def test_observe_binding_site(mock_data):
    """
    Tests that the 'observe' function creates the correct observation site.
    - Checks the site name, type, and parameters (mean, std).
    - Checks that the observation mask is correctly applied.
    """
    name = "test"

    binding_pred = torch.ones_like(mock_data.theta_obs) * 0.45

    pyro.clear_param_store()
    model_trace = poutine.trace(observe).get_trace(
        name=name,
        data=mock_data,
        binding_pred=binding_pred
    )

    # --- 1. Check for the Observation Site ---
    obs_site_name = f"{name}_binding_obs"
    assert obs_site_name in model_trace.nodes

    site = model_trace.nodes[obs_site_name]

    # --- 2. Check Site Properties ---
    assert site["type"] == "sample"
    assert site["is_observed"]

    # --- 3. Check Distribution and Parameters ---
    # In Pyro, poutine.mask does not wrap the distribution; it's the raw Normal.
    assert isinstance(site["fn"], dist.Normal)

    # Check that the predicted mean was passed correctly
    assert torch.all(site["fn"].loc == binding_pred)
    # Check that the standard deviation was passed correctly
    assert torch.all(site["fn"].scale == mock_data.theta_std)

    # Check that the observations were passed correctly
    assert torch.all(site["value"] == mock_data.theta_obs)

    # --- 4. Check Masking ---
    # poutine.mask tracks the mask in site["mask"]; apply it to raw log_prob.
    actual_log_prob = (site["fn"].log_prob(site["value"]) * site["mask"]).sum()

    # Unmasked calculation on only good data
    unmasked_log_prob = dist.Normal(binding_pred, mock_data.theta_std).log_prob(mock_data.theta_obs)
    expected_log_prob = unmasked_log_prob[mock_data.good_mask].sum()

    assert torch.isclose(actual_log_prob, expected_log_prob)

def test_guide(mock_data):
    """
    Tests the guide function.
    The guide is currently empty/no-op, so we just check it runs without error.
    """
    name = "test_guide"
    binding_pred = torch.ones_like(mock_data.theta_obs) * 0.45

    # Should run without raising
    guide(name, mock_data, binding_pred)
