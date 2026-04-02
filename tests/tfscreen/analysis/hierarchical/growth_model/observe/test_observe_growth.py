import pytest
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.observe.growth import observe, guide

# --- Mock Data Fixture ---

MockGrowthData = namedtuple("MockGrowthData", [
    "ln_cfu",
    "ln_cfu_std",
    "num_replicate",
    "num_time",
    "num_condition_pre",
    "num_condition_sel",
    "num_titrant_name",
    "num_titrant_conc",
    "num_genotype",
    "batch_size",
    "scale_vector",
    "good_mask"
])

@pytest.fixture
def mock_data():
    """
    Provides mock data for the growth observation.

    Total Shape: (1, 1, 1, 1, 1, 1, 4)
    """
    num_replicate = 1
    num_time = 1
    num_condition_pre = 1
    num_condition_sel = 1
    num_titrant_name = 1
    num_titrant_conc = 1

    batch_size = 4
    total_genotypes = 100

    # Shape: (1, 1, 1, 1, 1, 1, 4)
    shape = (num_replicate, num_time, num_condition_pre, num_condition_sel,
             num_titrant_name, num_titrant_conc, batch_size)

    ln_cfu = torch.ones(shape) * 5.0
    ln_cfu_std = torch.ones(shape) * 0.2

    # Scale vector for subsampling
    scale_vector = torch.ones(batch_size) * (total_genotypes / batch_size)

    # Create mask: (0,0,0,0,0,0,0) is Bad
    good_mask = torch.ones(shape, dtype=torch.bool)
    good_mask[0, 0, 0, 0, 0, 0, 0] = False

    return MockGrowthData(
        ln_cfu=ln_cfu,
        ln_cfu_std=ln_cfu_std,
        num_replicate=num_replicate,
        num_time=num_time,
        num_condition_pre=num_condition_pre,
        num_condition_sel=num_condition_sel,
        num_titrant_name=num_titrant_name,
        num_titrant_conc=num_titrant_conc,
        num_genotype=total_genotypes,
        batch_size=batch_size,
        scale_vector=scale_vector,
        good_mask=good_mask
    )

def test_observe_structure_and_distribution(mock_data):
    """
    Verifies the site names, distribution types, and shapes.
    """
    name = "test"
    ln_cfu_pred = torch.ones_like(mock_data.ln_cfu) * 5.0

    pyro.clear_param_store()
    model_trace = poutine.trace(observe).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )

    # 1. Check 'nu' parameter (Pyro traces use .nodes)
    nu_name = f"{name}_nu"
    assert nu_name in model_trace.nodes
    assert isinstance(model_trace.nodes[nu_name]["fn"], dist.Gamma)

    # 2. Check Observation Site
    obs_name = f"{name}_growth_obs"
    assert obs_name in model_trace.nodes
    site = model_trace.nodes[obs_name]

    assert site["is_observed"]
    # In Pyro, poutine.mask does not wrap the distribution in MaskedDistribution;
    # the distribution is the raw StudentT and masking is tracked separately.
    assert isinstance(site["fn"], dist.StudentT)

    # 3. Check shapes match input
    assert site["value"].shape == mock_data.ln_cfu.shape

def test_observe_subsampling_scaling(mock_data):
    """
    CRITICAL: Verifies that the log_prob is correctly scaled.
    """
    name = "test"
    ln_cfu_pred = torch.ones_like(mock_data.ln_cfu) * 5.0

    fixed_nu = 10.0
    # Use poutine.do to substitute the latent nu without marking it observed
    conditioned_model = poutine.do(observe, data={f"{name}_nu": torch.tensor(fixed_nu)})

    pyro.clear_param_store()
    model_trace = poutine.trace(conditioned_model).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )

    site = model_trace.nodes[f"{name}_growth_obs"]

    # 1. Calculate Expected Masked Log Prob (manual)
    base_dist = dist.StudentT(df=fixed_nu, loc=ln_cfu_pred, scale=mock_data.ln_cfu_std)
    log_probs = base_dist.log_prob(mock_data.ln_cfu)
    masked_log_probs = torch.where(mock_data.good_mask, log_probs, torch.zeros_like(log_probs))
    sum_log_prob_batch = masked_log_probs.sum()

    # 2. Verify Scale Factor — poutine.scale stores it in site["scale"]
    scale_factor = 25.0
    assert torch.all(site["scale"] == scale_factor)

    # 3. Verify Masked Log Prob matches (apply site mask to raw distribution log_prob)
    trace_log_prob = site["fn"].log_prob(site["value"]) * site["mask"]
    assert torch.allclose(trace_log_prob.sum(), sum_log_prob_batch)

def test_observe_masking_logic(mock_data):
    """
    Verifies that masked data points do not contribute to the likelihood.
    """
    name = "test"

    # A prediction that is WAY OFF for the masked point
    ln_cfu_pred = torch.ones_like(mock_data.ln_cfu) * 5.0
    ln_cfu_pred = ln_cfu_pred.clone()
    ln_cfu_pred[0, 0, 0, 0, 0, 0, 0] = 1000.0

    fixed_nu = 30.0
    conditioned_model = poutine.do(observe, data={f"{name}_nu": torch.tensor(fixed_nu)})

    pyro.clear_param_store()
    model_trace = poutine.trace(conditioned_model).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )

    site = model_trace.nodes[f"{name}_growth_obs"]
    # Apply the mask tracked by poutine.mask to the raw log_prob
    log_probs = site["fn"].log_prob(site["value"]) * site["mask"]

    # Check the specific index (masked) -> 0.0
    assert log_probs[0, 0, 0, 0, 0, 0, 0] == 0.0

    # Check a valid index -> non-zero
    assert log_probs[0, 0, 0, 0, 0, 0, 1] != 0.0

def test_guide_structure(mock_data):
    """
    Tests that the guide creates the correct parameter site for 'nu'.
    """
    name = "test_guide"
    ln_cfu_pred = torch.ones_like(mock_data.ln_cfu) * 5.0

    pyro.clear_param_store()
    guide_trace = poutine.trace(guide).get_trace(
        name=name,
        data=mock_data,
        ln_cfu_pred=ln_cfu_pred
    )

    # In Pyro, pyro.param() creates "param" type sites in the trace
    assert f"{name}_nu_loc" in guide_trace.nodes
    assert f"{name}_nu_scale" in guide_trace.nodes
    assert f"{name}_nu" in guide_trace.nodes

    # Check distribution
    assert isinstance(guide_trace.nodes[f"{name}_nu"]["fn"], dist.LogNormal)
