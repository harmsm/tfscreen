import pytest
import torch
import pyro
import pyro.poutine as poutine
from unittest.mock import MagicMock, ANY
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.model import pyro_model
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
    "theta_growth_noise", "condition_growth", "growth_transition", "ln_cfu0", "dk_geno", "activity", "transformation"
])
MockBindingPriors = namedtuple("MockBindingPriors", ["theta_binding_noise"])
MockPriors = namedtuple("MockPriors", ["theta", "growth", "binding"])

@pytest.fixture
def mock_data():
    """Provides a mocked DataClass structure with shapes for calculations."""
    t_pre = torch.tensor(2.0)
    t_sel = torch.tensor(3.0)

    growth = MockGrowthData(t_pre=t_pre, t_sel=t_sel, congression_mask="mock_mask")
    binding = MockBindingData()
    return MockData(growth=growth, binding=binding)

@pytest.fixture
def mock_priors():
    """Provides a mocked PriorsClass structure."""
    growth = MockGrowthPriors(
        theta_growth_noise="prior_gn",
        condition_growth="prior_cg",
        growth_transition="prior_gt",
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
    theta_model = MagicMock(return_value=10.0)  # theta
    calc_theta = MagicMock(side_effect=lambda t, d: t * 2.0)  # theta_growth/binding = 20.0
    get_moments = MagicMock(return_value=(0.0, 1.0))  # (mu, sigma) anchors

    # Growth Param Model returns params
    mock_params = MagicMock()
    condition_growth_model = MagicMock(return_value=mock_params)

    # calculate growth returns g_pre, g_sel
    calculate_growth = MagicMock(return_value=(21.0, 21.0))

    # Growth Transition returns total growth
    growth_transition_model = MagicMock(return_value=105.0)

    ln_cfu0_model = MagicMock(return_value=torch.tensor([5.0]))
    activity_model = MagicMock(return_value=1.0)
    dk_geno_model = MagicMock(return_value=0.0)

    transformation_model = MagicMock(return_value=(1.0, 1.0, 1.0))
    transformation_update = MagicMock(side_effect=lambda t, params, mask=None: t)

    theta_binding_noise_model = MagicMock(side_effect=lambda n, x, p: x)
    theta_growth_noise_model = MagicMock(side_effect=lambda n, x, p: x)

    binding_observer = MagicMock()
    growth_observer = MagicMock()

    return {
        "theta": (theta_model, calc_theta, get_moments),
        "condition_growth": condition_growth_model,
        "calculate_growth": calculate_growth,
        "growth_transition": growth_transition_model,
        "ln_cfu0": ln_cfu0_model,
        "activity": activity_model,
        "dk_geno": dk_geno_model,
        "transformation": (transformation_model, transformation_update),
        "theta_binding_noise": theta_binding_noise_model,
        "theta_growth_noise": theta_growth_noise_model,
        "observe_binding": binding_observer,
        "observe_growth": growth_observer,
        "is_guide": False  # Default to main model
    }

# --- Test Cases ---

def test_pyro_model_execution_flow(mock_data, mock_priors, mock_control):
    """
    Tests the main execution path (is_guide=False).
    - Verifies all sub-models are called with correct args.
    - Verifies the final calculation logic.
    - Verifies deterministic sites are registered.
    """
    pyro.clear_param_store()
    torch.manual_seed(0)
    model_trace = poutine.trace(pyro_model).get_trace(mock_data, mock_priors, **mock_control)

    # --- 1. Verify Control Calls ---

    # Theta
    mock_control["theta"][0].assert_called_once_with("theta", mock_data.growth, "prior_theta")

    # get_moments called with (theta, growth_data)
    mock_control["theta"][2].assert_called_once_with(10.0, mock_data.growth)

    # Calc Theta called twice: once for binding, once for growth
    assert mock_control["theta"][1].call_count == 2

    # Binding Noise
    mock_control["theta_binding_noise"].assert_called_once()

    # Growth Noise
    mock_control["theta_growth_noise"].assert_called_once()

    # Condition Growth
    mock_control["condition_growth"].assert_called_once_with(
        "condition_growth", mock_data.growth, "prior_cg"
    )

    # calculate_growth
    mock_control["calculate_growth"].assert_called_once()

    # Growth Transition
    mock_control["growth_transition"].assert_called_once_with(
        "growth_transition", mock_data.growth, ANY, g_pre=21.0, g_sel=21.0,
        t_pre=torch.tensor(2.0), t_sel=torch.tensor(3.0), theta=20.0
    )

    # ln_cfu0
    mock_control["ln_cfu0"].assert_called_once_with(
        "ln_cfu0", mock_data.growth, "prior_cfu0"
    )

    # Transformation
    mock_control["transformation"][0].assert_called_once_with(
        "transformation", mock_data.growth, "prior_trans", anchors=(0.0, 1.0)
    )
    mock_control["transformation"][1].assert_called_once_with(
        20.0,  # theta_growth (10.0 * 2.0)
        params=(1.0, 1.0, 1.0),
        mask="mock_mask"
    )

    # Observers
    mock_control["observe_growth"].assert_called_once()
    mock_control["observe_binding"].assert_called_once()

    # --- 2. Verify Calculation Logic ---
    # expected_pred = 5.0 + 105.0 = 110.0
    expected_pred = 110.0

    # Check that the observer received this prediction
    args, _ = mock_control["observe_growth"].call_args
    assert args[0] == "final_binding_obs"
    assert args[1] is mock_data.growth
    assert torch.isclose(args[2], torch.tensor(expected_pred)).all()

    # --- 3. Verify Deterministic Sites ---
    assert "theta_binding_pred" in model_trace.nodes
    assert "theta_growth_pred" in model_trace.nodes
    assert "binding_pred" in model_trace.nodes
    assert "growth_pred" in model_trace.nodes

    assert torch.isclose(model_trace.nodes["growth_pred"]["value"], torch.tensor(expected_pred)).all()


def test_pyro_model_guide_flow(mock_data, mock_priors, mock_control):
    """
    Tests the guide execution path (is_guide=True).
    - Should call sub-models.
    - Should NOT perform the final complex calculation.
    - Should pass None to observers as prediction.
    """
    mock_control["is_guide"] = True

    pyro.clear_param_store()
    torch.manual_seed(0)
    pyro_model(mock_data, mock_priors, **mock_control)

    # --- Verify Sub-models still called ---
    mock_control["theta"][0].assert_called_once()
    mock_control["condition_growth"].assert_called_once()
    mock_control["transformation"][0].assert_called_once()

    # --- Verify Observers called with None ---
    args_growth, _ = mock_control["observe_growth"].call_args
    assert args_growth[2] is None

    args_binding, _ = mock_control["observe_binding"].call_args
    assert args_binding[2] is None
