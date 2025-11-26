import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock
from collections import namedtuple

# --- Module Imports ---
from tfscreen.analysis.hierarchical.growth_model import model

# Import the functions we want to test
from tfscreen.analysis.hierarchical.growth_model.model import (
    _define_growth,
    _define_binding,
    jax_model,
    # Constants needed for control flags
    THETA_CATEGORICAL, THETA_HILL,
    CONDITION_GROWTH_INDEPENDENT, CONDITION_GROWTH_HIERARCHICAL, CONDITION_GROWTH_FIXED,
    LN_CFU0_HIERARCHICAL,
    DK_GENO_FIXED, DK_GENO_HIERARCHICAL,
    ACTIVITY_FIXED, ACTIVITY_HIERARCHICAL, ACTIVITY_HORSESHOE,
    THETA_GROWTH_NOISE_NONE, THETA_GROWTH_NOISE_BETA,
    THETA_BINDING_NOISE_NONE, THETA_BINDING_NOISE_BETA
)

# ControlClass mimicking the structure expected by the model
ControlClass = namedtuple(
    "ControlClass",
    [
        "condition_growth", 
        "ln_cfu0", 
        "dk_geno", 
        "activity", 
        "theta", 
        "theta_growth_noise", 
        "theta_binding_noise"
    ]
)

# --- Fixtures ---

@pytest.fixture
def mock_data():
    """Provides a mock DataClass object."""
    data = MagicMock(name="DataClass")
    data.growth = MagicMock(name="GrowthData")
    data.binding = MagicMock(name="BindingData")
    
    # Provide arrays for the time/math operations
    # Using 1.0 makes multiplication identity, easier to check math
    data.growth.t_pre = jnp.array(1.0)
    data.growth.t_sel = jnp.array(1.0)
    return data

@pytest.fixture
def mock_priors():
    """Provides a mock PriorsClass object."""
    priors = MagicMock(name="PriorsClass")
    priors.growth = MagicMock(name="GrowthPriors")
    priors.binding = MagicMock(name="BindingPriors")
    priors.theta = MagicMock(name="ThetaPriors")
    return priors

@pytest.fixture
def mock_theta_params():
    """Provides a mock theta parameter pytree."""
    return {"param1": 1.0}

@pytest.fixture
def mock_pyro(monkeypatch):
    """
    Mocks numpyro to intercept deterministic calls.
    Returns the mock object so we can assert calls on it.
    """
    pyro_mock = MagicMock(name="numpyro")
    # deterministic acts as identity function in mocks usually
    pyro_mock.deterministic.side_effect = lambda name, value: value
    monkeypatch.setattr("tfscreen.analysis.hierarchical.growth_model.model.pyro", pyro_mock)
    return pyro_mock

@pytest.fixture
def mock_growth_components(monkeypatch):
    """Mocks all dependencies for the _define_growth function."""
    
    mocks = {
        # Theta calculators
        "calc_theta_cat": MagicMock(return_value=jnp.array(0.5)),
        "calc_theta_hill": MagicMock(return_value=jnp.array(0.5)),
        
        # Noise models
        "define_no_noise": MagicMock(return_value=jnp.array(0.5)),
        "define_beta_noise": MagicMock(return_value=jnp.array(0.5)),
        
        # Growth Params (Return unpacked tuple)
        # k_pre, m_pre, k_sel, m_sel
        "define_growth_independent": MagicMock(return_value=(1.0, 1.0, 1.0, 1.0)),
        "define_growth_hierarchical": MagicMock(return_value=(2.0, 2.0, 2.0, 2.0)),
        "define_growth_fixed": MagicMock(return_value=(3.0, 3.0, 3.0, 3.0)),
        
        # Other params
        "define_ln_cfu0": MagicMock(return_value=10.0),
        "define_dk_geno_fixed": MagicMock(return_value=0.1),
        "define_dk_geno_hierarchical": MagicMock(return_value=0.2),
        "define_activity_fixed": MagicMock(return_value=1.0),
        "define_activity_hierarchical": MagicMock(return_value=1.1),
        "define_activity_horseshoe": MagicMock(return_value=1.2),
    }
    
    for name, mock_obj in mocks.items():
        monkeypatch.setattr(f"tfscreen.analysis.hierarchical.growth_model.model.{name}", mock_obj)
        
    return mocks

@pytest.fixture
def mock_binding_components(monkeypatch):
    """Mocks all dependencies for the _define_binding function."""
    mocks = {
        "calc_theta_cat": MagicMock(return_value=jnp.array(0.5)),
        "calc_theta_hill": MagicMock(return_value=jnp.array(0.5)),
        "define_no_noise": MagicMock(return_value=jnp.array(0.5)),
        "define_beta_noise": MagicMock(return_value=jnp.array(0.5)),
    }
    for name, mock_obj in mocks.items():
        monkeypatch.setattr(f"tfscreen.analysis.hierarchical.growth_model.model.{name}", mock_obj)
    return mocks

@pytest.fixture
def mock_main_model_components(monkeypatch):
    """Mocks all dependencies for the jax_model function."""
    mocks = {
        "define_theta_cat": MagicMock(return_value={"p": 1}),
        "define_theta_hill": MagicMock(return_value={"p": 2}),
        "_define_growth": MagicMock(return_value="growth_pred_tensor"),
        "_define_binding": MagicMock(return_value="binding_pred_tensor"),
        "observe_growth": MagicMock(),
        "observe_binding": MagicMock(),
    }
    for name, mock_obj in mocks.items():
        monkeypatch.setattr(f"tfscreen.analysis.hierarchical.growth_model.model.{name}", mock_obj)
    return mocks


# -------------------
# test _define_growth
# -------------------

class TestDefineGrowth:

    def test_path_independent_categorical_no_noise(self, mock_data, mock_priors, mock_theta_params, mock_growth_components, mock_pyro):
        """Tests a specific combination of control flags."""
        control = ControlClass(
            theta=THETA_CATEGORICAL, 
            theta_growth_noise=THETA_GROWTH_NOISE_NONE, 
            condition_growth=CONDITION_GROWTH_INDEPENDENT,
            ln_cfu0=LN_CFU0_HIERARCHICAL, 
            dk_geno=DK_GENO_FIXED, 
            activity=ACTIVITY_FIXED, 
            theta_binding_noise=0
        )
        
        # Run
        ln_cfu_pred = _define_growth(mock_data, mock_priors, control, mock_theta_params)
        
        # 1. Check Deterministic Registration
        mock_pyro.deterministic.assert_any_call("theta_growth_pred", jnp.array(0.5))

        # 2. Check Component Calls
        mock_growth_components["calc_theta_cat"].assert_called_once_with(mock_theta_params, mock_data.growth)
        mock_growth_components["define_no_noise"].assert_called_once()
        mock_growth_components["define_growth_independent"].assert_called_once()
        mock_growth_components["define_ln_cfu0"].assert_called_once()
        mock_growth_components["define_dk_geno_fixed"].assert_called_once()
        mock_growth_components["define_activity_fixed"].assert_called_once()
        
        # 3. Check Math Logic
        # Mocks return: theta=0.5, noise=0.5, k=1.0, m=1.0, ln_cfu0=10.0, dk=0.1, act=1.0
        # g = k + dk + act * m * noisy_theta
        # g = 1.0 + 0.1 + 1.0 * 1.0 * 0.5 = 1.6
        # pred = ln_cfu0 + g*t + g*t = 10.0 + 1.6*1.0 + 1.6*1.0 = 13.2
        
        # We assert approx equality for floating point
        assert jnp.isclose(ln_cfu_pred, 13.2)

    def test_path_hierarchical_hill_beta_noise(self, mock_data, mock_priors, mock_theta_params, mock_growth_components):
        """Tests the hierarchical / hill / beta noise path."""
        control = ControlClass(
            theta=THETA_HILL, 
            theta_growth_noise=THETA_GROWTH_NOISE_BETA, 
            condition_growth=CONDITION_GROWTH_HIERARCHICAL,
            ln_cfu0=LN_CFU0_HIERARCHICAL, 
            dk_geno=DK_GENO_HIERARCHICAL, 
            activity=ACTIVITY_HORSESHOE, 
            theta_binding_noise=0
        )
        
        _define_growth(mock_data, mock_priors, control, mock_theta_params)
        
        mock_growth_components["calc_theta_hill"].assert_called_once()
        mock_growth_components["define_beta_noise"].assert_called_once()
        mock_growth_components["define_growth_hierarchical"].assert_called_once()
        mock_growth_components["define_dk_geno_hierarchical"].assert_called_once()
        mock_growth_components["define_activity_horseshoe"].assert_called_once()

    def test_invalid_selection_raises_error(self, mock_data, mock_priors, mock_theta_params, mock_growth_components):
        """Tests that random invalid integers raise ValueErrors."""
        control = ControlClass(
            theta=99, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        with pytest.raises(ValueError, match="theta selection"):
             _define_growth(mock_data, mock_priors, control, mock_theta_params)

# -------------------
# test _define_binding
# -------------------

class TestDefineBinding:

    def test_path_hill_no_noise(self, mock_data, mock_priors, mock_theta_params, mock_binding_components, mock_pyro):
        control = ControlClass(
            theta=THETA_HILL, 
            theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, 
            theta_binding_noise=THETA_BINDING_NOISE_NONE
        )
        
        res = _define_binding(mock_data, mock_priors, control, mock_theta_params)
        
        mock_binding_components["calc_theta_hill"].assert_called_once_with(mock_theta_params, mock_data.binding)
        mock_pyro.deterministic.assert_called_with("theta_binding_pred", jnp.array(0.5))
        mock_binding_components["define_no_noise"].assert_called_once()
        
    def test_path_cat_beta_noise(self, mock_data, mock_priors, mock_theta_params, mock_binding_components):
        control = ControlClass(
            theta=THETA_CATEGORICAL, 
            theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, 
            theta_binding_noise=THETA_BINDING_NOISE_BETA
        )
        
        _define_binding(mock_data, mock_priors, control, mock_theta_params)
        
        mock_binding_components["calc_theta_cat"].assert_called_once()
        mock_binding_components["define_beta_noise"].assert_called_once()

# -------------------
# test jax_model
# -------------------

class TestJaxModel:

    def test_path_theta_cat(self, mock_data, mock_priors, mock_main_model_components, mock_pyro):
        """Tests the main model path with theta=0 (categorical)."""
        control = ControlClass(
            theta=THETA_CATEGORICAL, 
            theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        jax_model(mock_data, mock_priors, control)
        
        # 1. Check Data Flow for Theta Definition
        # CRITICAL UPDATE: The code passes mock_data.growth, not mock_data
        mock_main_model_components["define_theta_cat"].assert_called_once_with(
            "theta", mock_data.growth, mock_priors.theta
        )
        
        # 2. Check Helper Orchestration
        # define_theta_cat mock returns {"p": 1}, so helpers should receive that
        mock_main_model_components["_define_growth"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 1}
        )
        mock_main_model_components["_define_binding"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 1}
        )
        
        # 3. Check Deterministic Registration
        mock_pyro.deterministic.assert_any_call("growth_pred", "growth_pred_tensor")
        mock_pyro.deterministic.assert_any_call("binding_pred", "binding_pred_tensor")
        
        # 4. Check Final Observations
        mock_main_model_components["observe_growth"].assert_called_once_with(
            "final_obs", mock_data.growth, "growth_pred_tensor"
        )
        mock_main_model_components["observe_binding"].assert_called_once_with(
            "final_obs", mock_data.binding, "binding_pred_tensor"
        )
        
    def test_path_theta_hill(self, mock_data, mock_priors, mock_main_model_components):
        """Tests the main model path with theta=1 (hill)."""
        control = ControlClass(
            theta=THETA_HILL, 
            theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        jax_model(mock_data, mock_priors, control)
        
        # Check calling convention: passes data.growth
        mock_main_model_components["define_theta_hill"].assert_called_once_with(
            "theta", mock_data.growth, mock_priors.theta
        )
        
        # Check helpers receive result from define_theta_hill ({"p": 2})
        mock_main_model_components["_define_growth"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 2}
        )

    def test_value_error(self, mock_data, mock_priors):
        """Tests that an invalid theta flag in jax_model raises a ValueError."""
        control = ControlClass(
            theta=99, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        with pytest.raises(ValueError):
            jax_model(mock_data, mock_priors, control)