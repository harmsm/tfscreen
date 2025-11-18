import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock
from collections import namedtuple

# --- Module Imports ---
# Import the module that contains the functions we want to test
from tfscreen.analysis.hierarchical.growth_model import model

# Import the functions we want to test
from tfscreen.analysis.hierarchical.growth_model.model import (
    _define_growth,
    _define_binding,
    jax_model
)

# We use a namedtuple to create a simple, stand-in "ControlClass"
# This avoids needing to import the real dataclass
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
    
    # Provide real values for the math operations in _define_growth
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
    
    # Mock the nested priors
    priors.growth.theta_growth_noise = MagicMock(name="theta_growth_noise_prior")
    priors.growth.condition_growth = MagicMock(name="condition_growth_prior")
    priors.growth.ln_cfu0 = MagicMock(name="ln_cfu0_prior")
    priors.growth.dk_geno = MagicMock(name="dk_geno_prior")
    priors.growth.activity = MagicMock(name="activity_prior")
    priors.binding.theta_binding_noise = MagicMock(name="theta_binding_noise_prior")
    return priors

@pytest.fixture
def mock_theta_params():
    """Provides a mock theta parameter pytree."""
    return {"param1": 1.0, "param2": 2.0}

@pytest.fixture
def mock_growth_components(monkeypatch):
    """Mocks all dependencies for the _define_growth function."""
    
    # Create a dict of mocks
    mocks = {
        "calc_theta_cat": MagicMock(name="calc_theta_cat", return_value=jnp.array(0.5)),
        "calc_theta_hill": MagicMock(name="calc_theta_hill", return_value=jnp.array(0.5)),
        
        "define_no_noise": MagicMock(name="define_no_noise", return_value=jnp.array(0.5)),
        "define_beta_noise": MagicMock(name="define_beta_noise", return_value=jnp.array(0.5)),
        
        # Must return numerical values for the final math
        "define_growth_independent": MagicMock(name="define_growth_independent", return_value=(1.0, 1.0, 1.0, 1.0)),
        "define_growth_hierarchical": MagicMock(name="define_growth_hierarchical", return_value=(2.0, 2.0, 2.0, 2.0)),
        
        "define_ln_cfu0": MagicMock(name="define_ln_cfu0", return_value=10.0),
        
        "define_dk_geno_fixed": MagicMock(name="define_dk_geno_fixed", return_value=0.1),
        "define_dk_geno_hierarchical": MagicMock(name="define_dk_geno_hierarchical", return_value=0.2),
        
        "define_activity_fixed": MagicMock(name="define_activity_fixed", return_value=1.0),
        "define_activity_hierarchical": MagicMock(name="define_activity_hierarchical", return_value=1.1),
        "define_activity_horseshoe": MagicMock(name="define_activity_horseshoe", return_value=1.2),
    }
    
    # Patch all mocks into the 'model' module's namespace
    for name, mock_obj in mocks.items():
        monkeypatch.setattr(f"tfscreen.analysis.hierarchical.growth_model.model.{name}", mock_obj)
        
    return mocks


@pytest.fixture
def mock_binding_components(monkeypatch):
    """Mocks all dependencies for the _define_binding function."""
    mocks = {
        "calc_theta_cat": MagicMock(name="calc_theta_cat", return_value=jnp.array(0.5)),
        "calc_theta_hill": MagicMock(name="calc_theta_hill", return_value=jnp.array(0.5)),
        "define_no_noise": MagicMock(name="define_no_noise", return_value=jnp.array(0.5)),
        "define_beta_noise": MagicMock(name="define_beta_noise", return_value=jnp.array(0.5)),
    }
    for name, mock_obj in mocks.items():
        monkeypatch.setattr(f"tfscreen.analysis.hierarchical.growth_model.model.{name}", mock_obj)
    return mocks

@pytest.fixture
def mock_main_model_components(monkeypatch):
    """Mocks all dependencies for the jax_model function."""
    mocks = {
        "define_theta_cat": MagicMock(name="define_theta_cat", return_value={"p": 1}),
        "define_theta_hill": MagicMock(name="define_theta_hill", return_value={"p": 2}),
        "_define_growth": MagicMock(name="_define_growth", return_value="growth_pred"),
        "_define_binding": MagicMock(name="_define_binding", return_value="binding_pred"),
        "observe_growth": MagicMock(name="observe_growth"),
        "observe_binding": MagicMock(name="observe_binding"),
    }
    for name, mock_obj in mocks.items():
        monkeypatch.setattr(f"tfscreen.analysis.hierarchical.growth_model.model.{name}", mock_obj)
    return mocks


# -------------------
# test _define_growth
# -------------------

class TestDefineGrowth:

    def test_path_all_zero(self, mock_data, mock_priors, mock_theta_params, mock_growth_components):
        """Tests the all-default (0) control path."""
        control = ControlClass(
            theta=0, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        ln_cfu_pred = _define_growth(mock_data, mock_priors, control, mock_theta_params)
        
        # Check that the final math was sensible
        assert isinstance(ln_cfu_pred, jnp.ndarray)
        
        # Check theta calculation path
        mock_growth_components["calc_theta_cat"].assert_called_once_with(mock_theta_params, mock_data.growth)
        mock_growth_components["calc_theta_hill"].assert_not_called()
        
        # Check noise path
        mock_growth_components["define_no_noise"].assert_called_once_with(
            "theta_growth_noise", jnp.array(0.5), mock_priors.growth.theta_growth_noise
        )
        mock_growth_components["define_beta_noise"].assert_not_called()
        
        # Check condition_growth path
        mock_growth_components["define_growth_independent"].assert_called_once_with(
            "condition_growth", mock_data.growth, mock_priors.growth.condition_growth
        )
        mock_growth_components["define_growth_hierarchical"].assert_not_called()
        
        # Check ln_cfu0 path
        mock_growth_components["define_ln_cfu0"].assert_called_once_with(
            "ln_cfu0", mock_data.growth, mock_priors.growth.ln_cfu0
        )
        
        # Check dk_geno path
        mock_growth_components["define_dk_geno_fixed"].assert_called_once_with(
            "dk_geno", mock_data.growth, mock_priors.growth.dk_geno
        )
        mock_growth_components["define_dk_geno_hierarchical"].assert_not_called()

        # Check activity path
        mock_growth_components["define_activity_fixed"].assert_called_once_with(
            "activity", mock_data.growth, mock_priors.growth.activity
        )
        mock_growth_components["define_activity_hierarchical"].assert_not_called()
        mock_growth_components["define_activity_horseshoe"].assert_not_called()

    def test_path_all_one_and_two(self, mock_data, mock_priors, mock_theta_params, mock_growth_components):
        """Tests the non-default (1 or 2) control paths."""
        control = ControlClass(
            theta=1, theta_growth_noise=1, condition_growth=1,
            ln_cfu0=0, dk_geno=1, activity=2, theta_binding_noise=0
        )
        
        _define_growth(mock_data, mock_priors, control, mock_theta_params)
        
        # Check theta calculation path
        mock_growth_components["calc_theta_cat"].assert_not_called()
        mock_growth_components["calc_theta_hill"].assert_called_once_with(mock_theta_params, mock_data.growth)
        
        # Check noise path
        mock_growth_components["define_no_noise"].assert_not_called()
        mock_growth_components["define_beta_noise"].assert_called_once()
        
        # Check condition_growth path
        mock_growth_components["define_growth_independent"].assert_not_called()
        mock_growth_components["define_growth_hierarchical"].assert_called_once()
        
        # Check ln_cfu0 path (only has one path)
        mock_growth_components["define_ln_cfu0"].assert_called_once()
        
        # Check dk_geno path
        mock_growth_components["define_dk_geno_fixed"].assert_not_called()
        mock_growth_components["define_dk_geno_hierarchical"].assert_called_once()

        # Check activity path
        mock_growth_components["define_activity_fixed"].assert_not_called()
        mock_growth_components["define_activity_hierarchical"].assert_not_called()
        mock_growth_components["define_activity_horseshoe"].assert_called_once()

    @pytest.mark.parametrize("bad_control", [
        ControlClass(theta=2, theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0),
        ControlClass(theta=0, theta_growth_noise=2, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0),
        ControlClass(theta=0, theta_growth_noise=0, condition_growth=2, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0),
        ControlClass(theta=0, theta_growth_noise=0, condition_growth=0, ln_cfu0=1, dk_geno=0, activity=0, theta_binding_noise=0),
        ControlClass(theta=0, theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=2, activity=0, theta_binding_noise=0),
        ControlClass(theta=0, theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=3, theta_binding_noise=0),
    ])
    def test_value_errors(self, mock_data, mock_priors, mock_theta_params, mock_growth_components, bad_control):
        """Tests that invalid control flags raise ValueErrors."""
        with pytest.raises(ValueError):
            _define_growth(mock_data, mock_priors, bad_control, mock_theta_params)


# -------------------
# test _define_binding
# -------------------

class TestDefineBinding:

    def test_path_all_zero(self, mock_data, mock_priors, mock_theta_params, mock_binding_components):
        """Tests the all-default (0) control path."""
        control = ControlClass(
            theta=0, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        _define_binding(mock_data, mock_priors, control, mock_theta_params)
        
        # Check theta calculation path
        mock_binding_components["calc_theta_cat"].assert_called_once_with(mock_theta_params, mock_data.binding)
        mock_binding_components["calc_theta_hill"].assert_not_called()

        # Check noise path
        mock_binding_components["define_no_noise"].assert_called_once_with(
            "theta_binding_noise", jnp.array(0.5), mock_priors.binding.theta_binding_noise
        )
        mock_binding_components["define_beta_noise"].assert_not_called()
        
    def test_path_all_one(self, mock_data, mock_priors, mock_theta_params, mock_binding_components):
        """Tests the non-default (1) control path."""
        control = ControlClass(
            theta=1, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=1
        )
        
        _define_binding(mock_data, mock_priors, control, mock_theta_params)
        
        # Check theta calculation path
        mock_binding_components["calc_theta_cat"].assert_not_called()
        mock_binding_components["calc_theta_hill"].assert_called_once_with(mock_theta_params, mock_data.binding)

        # Check noise path
        mock_binding_components["define_no_noise"].assert_not_called()
        mock_binding_components["define_beta_noise"].assert_called_once_with(
            "theta_binding_noise", jnp.array(0.5), mock_priors.binding.theta_binding_noise
        )

    @pytest.mark.parametrize("bad_control", [
        ControlClass(theta=2, theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0),
        ControlClass(theta=0, theta_growth_noise=0, condition_growth=0, ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=2),
    ])
    def test_value_errors(self, mock_data, mock_priors, mock_theta_params, mock_binding_components, bad_control):
        """Tests that invalid control flags raise ValueErrors."""
        with pytest.raises(ValueError):
            _define_binding(mock_data, mock_priors, bad_control, mock_theta_params)


# -------------------
# test jax_model
# -------------------

class TestJaxModel:

    def test_path_theta_cat(self, mock_data, mock_priors, mock_main_model_components):
        """Tests the main model path with theta=0 (categorical)."""
        control = ControlClass(
            theta=0, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        jax_model(mock_data, mock_priors, control)
        
        # Check theta definition path
        mock_main_model_components["define_theta_cat"].assert_called_once_with(
            "theta", mock_data, mock_priors.theta
        )
        mock_main_model_components["define_theta_hill"].assert_not_called()
        
        # Check that helpers were called with the defined theta params
        mock_main_model_components["_define_growth"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 1}
        )
        mock_main_model_components["_define_binding"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 1}
        )
        
        # Check that observations were made
        mock_main_model_components["observe_growth"].assert_called_once_with(
            "final_obs", mock_data.growth, "growth_pred"
        )
        mock_main_model_components["observe_binding"].assert_called_once_with(
            "final_obs", mock_data.binding, "binding_pred"
        )
        
    def test_path_theta_hill(self, mock_data, mock_priors, mock_main_model_components):
        """Tests the main model path with theta=1 (hill)."""
        control = ControlClass(
            theta=1, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        jax_model(mock_data, mock_priors, control)
        
        # Check theta definition path
        mock_main_model_components["define_theta_cat"].assert_not_called()
        mock_main_model_components["define_theta_hill"].assert_called_once_with(
            "theta", mock_data.growth, mock_priors.theta
        )
        
        # Check that helpers were called with the defined theta params
        mock_main_model_components["_define_growth"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 2}
        )
        mock_main_model_components["_define_binding"].assert_called_once_with(
            mock_data, mock_priors, control, {"p": 2}
        )
        
        # Check that observations were made
        mock_main_model_components["observe_growth"].assert_called_once()
        mock_main_model_components["observe_binding"].assert_called_once()

    def test_value_error(self, mock_data, mock_priors, mock_main_model_components):
        """Tests that an invalid theta flag raises a ValueError."""
        control = ControlClass(
            theta=2, theta_growth_noise=0, condition_growth=0,
            ln_cfu0=0, dk_geno=0, activity=0, theta_binding_noise=0
        )
        
        with pytest.raises(ValueError):
            jax_model(mock_data, mock_priors, control)