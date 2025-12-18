import pytest
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, seed
from unittest.mock import MagicMock
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.model import jax_model
from tfscreen.analysis.hierarchical.growth_model.data_class import (
    DataClass, PriorsClass, GrowthData, BindingData, 
    GrowthPriors, BindingPriors
)

# --- Mocks ---

# Define minimal mocks for the nested data structures
MockGrowthData = namedtuple("MockGrowthData", ["t_pre", "t_sel", "congression_mask"])
MockBindingData = namedtuple("MockBindingData", [])
MockData = namedtuple("MockData", ["growth", "binding"])

MockGrowthPriors = namedtuple("MockGrowthPriors", [
    "theta_growth_noise", "condition_growth", "ln_cfu0", "dk_geno", "activity", "transformation"
])
MockBindingPriors = namedtuple("MockBindingPriors", ["theta_binding_noise"])
MockPriors = namedtuple("MockPriors", ["theta", "growth", "binding"])

@pytest.fixture
def mock_data():
    """Provides a mocked DataClass structure with shapes for calculations."""
    # Create simple scalar tensors for calculation verification
    # (1 + 2 * t_pre + 3 * t_sel) logic
    t_pre = jnp.array(2.0)
    t_sel = jnp.array(3.0)
    
    growth = MockGrowthData(t_pre=t_pre, t_sel=t_sel, congression_mask="mock_mask")
    binding = MockBindingData()
    return MockData(growth=growth, binding=binding)

@pytest.fixture
def mock_priors():
    """Provides a mocked PriorsClass structure."""
    growth = MockGrowthPriors(
        theta_growth_noise="prior_gn", 
        condition_growth="prior_cg",
        ln_cfu0="prior_cfu0", 
        dk_geno="prior_dk", 
        activity="prior_act",
        transformation="prior_trans"
    )
    binding = MockBindingPriors(theta_binding_noise="prior_bn")
    return MockPriors(theta="prior_theta", growth=growth, binding=binding)

@pytest.fixture
def mock_control():
    """
    Provides a dictionary of MagicMocks for all control functions.
    Configured to return specific values to test the calculation logic.
    """
    # Create mocks
    theta_model = MagicMock(return_value=10.0) # theta
    calc_theta = MagicMock(side_effect=lambda t, d: t * 2.0) # theta_growth/binding = 20.0
    
    # Growth Param Model returns (k_pre, m_pre, k_sel, m_sel)
    # k=1.0, m=1.0
    condition_growth_model = MagicMock(return_value=(1.0, 1.0, 1.0, 1.0))
    
    ln_cfu0_model = MagicMock(return_value=jnp.array([5.0])) # ln_cfu0 (must be array for softmax)
    activity_model = MagicMock(return_value=1.0) # activity
    dk_geno_model = MagicMock(return_value=0.0) # dk_geno
    
    transformation_model = MagicMock(return_value=(1.0, 1.0, 1.0)) # (lam, a, b)
    transformation_update = MagicMock(side_effect=lambda t, params, mask=None: t) # pass-through
    
    # Noise models just pass through or add noise. Let's pass through for simplicity.
    theta_binding_noise_model = MagicMock(side_effect=lambda n, x, p: x) 
    theta_growth_noise_model = MagicMock(side_effect=lambda n, x, p: x)
    
    binding_observer = MagicMock()
    growth_observer = MagicMock()

    return {
        "theta": (theta_model, calc_theta),
        "condition_growth": condition_growth_model,
        "ln_cfu0": ln_cfu0_model,
        "activity": activity_model,
        "dk_geno": dk_geno_model,
        "transformation": (transformation_model, transformation_update),
        "theta_binding_noise": theta_binding_noise_model,
        "theta_growth_noise": theta_growth_noise_model,
        "observe_binding": binding_observer,
        "observe_growth": growth_observer,
        "is_guide": False # Default to main model
    }

# --- Test Cases ---

def test_jax_model_execution_flow(mock_data, mock_priors, mock_control):
    """
    Tests the main execution path (is_guide=False).
    - Verifies all sub-models are called with correct args.
    - Verifies the final calculation logic.
    - Verifies deterministic sites are registered.
    """
    # Run the model
    # We trace it to check deterministic sites
    with numpyro.handlers.seed(rng_seed=0):
        model_trace = trace(jax_model).get_trace(mock_data, mock_priors, **mock_control)

    # --- 1. Verify Control Calls ---
    
    # Theta
    # theta_model called with (name, growth_data, theta_priors)
    mock_control["theta"][0].assert_called_once_with("theta", mock_data.growth, "prior_theta")
    
    # Calc Theta
    # Called twice: once for binding, once for growth
    assert mock_control["theta"][1].call_count == 2
    
    # Binding Noise
    mock_control["theta_binding_noise"].assert_called_once()
    
    # Growth Noise
    mock_control["theta_growth_noise"].assert_called_once()
    
    # Condition Growth
    mock_control["condition_growth"].assert_called_once_with(
        "condition_growth", mock_data.growth, "prior_cg"
    )
    
    # ln_cfu0
    mock_control["ln_cfu0"].assert_called_once_with(
        "ln_cfu0", mock_data.growth, "prior_cfu0"
    )
    
    # Transformation
    mock_control["transformation"][0].assert_called_once_with(
        "transformation", mock_data.growth, "prior_trans"
    )
    mock_control["transformation"][1].assert_called_once_with(
        jnp.array(20.0), # theta_growth
        params=(1.0, 1.0, 1.0),
        mask="mock_mask"
    )
    
    # Observers
    mock_control["observe_growth"].assert_called_once()
    mock_control["observe_binding"].assert_called_once()

    # --- 2. Verify Calculation Logic ---
    
    # Based on fixture return values:
    # theta = 10.0
    # theta_growth = 20.0 (calc_theta doubles it)
    # noisy_theta = 20.0 (noise passes through)
    # k_pre=1, m_pre=1, k_sel=1, m_sel=1
    # dk_geno=0, activity=1
    # ln_cfu0 = 5.0
    # t_pre = 2.0, t_sel = 3.0
    
    # g_pre = k_pre + dk + act*m*theta = 1 + 0 + 1*1*20 = 21.0
    # g_sel = k_sel + dk + act*m*theta = 1 + 0 + 1*1*20 = 21.0
    
    # ln_cfu_pred = ln_cfu0 + g_pre*t_pre + g_sel*t_sel
    #             = 5.0 + 21.0*2.0 + 21.0*3.0
    #             = 5.0 + 42.0 + 63.0 = 110.0
    
    expected_pred = 110.0
    
    # Check that the observer received this prediction
    args, _ = mock_control["observe_growth"].call_args
    assert args[0] == "final_binding_obs" # Name check (note: code uses confusing name here?)
    assert args[1] is mock_data.growth
    assert jnp.isclose(args[2], expected_pred)

    # --- 3. Verify Deterministic Sites ---
    assert "theta_binding_pred" in model_trace
    assert "theta_growth_pred" in model_trace
    assert "binding_pred" in model_trace
    assert "growth_pred" in model_trace
    
    assert jnp.isclose(model_trace["growth_pred"]["value"], expected_pred)

def test_jax_model_guide_flow(mock_data, mock_priors, mock_control):
    """
    Tests the guide execution path (is_guide=True).
    - Should call sub-models.
    - Should NOT perform the final complex calculation (g_pre, g_sel, etc).
    - Should pass None to observers as prediction.
    """
    # Set to guide mode
    mock_control["is_guide"] = True
    
    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data, mock_priors, **mock_control)
        
    # --- Verify Sub-models still called ---
    # The guide still needs to run the parameter guides
    mock_control["theta"][0].assert_called_once()
    mock_control["condition_growth"].assert_called_once()
    mock_control["transformation"][0].assert_called_once()
    
    # --- Verify Observers called with None ---
    # growth_observer("final_binding_obs", data.growth, None)
    args_growth, _ = mock_control["observe_growth"].call_args
    assert args_growth[2] is None
    
    args_binding, _ = mock_control["observe_binding"].call_args
    assert args_binding[2] is None