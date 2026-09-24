import pytest
import jax.numpy as jnp
import numpyro
from numpyro.handlers import trace, seed
from unittest.mock import MagicMock, ANY
from collections import namedtuple

# --- Import Module Under Test (MUT) ---
from tfscreen.tfmodel.generative.model import jax_model
from tfscreen.tfmodel.data_class import (
    DataClass, PriorsClass, GrowthData, BindingData, 
    GrowthPriors, BindingPriors
)
from tfscreen.tfmodel.generative.components.transformation._classes import (
    single_class,
)

# --- Mocks ---

# Define minimal mocks for the nested data structures
class MockGrowthData(namedtuple("MockGrowthData", [
        "t_pre", "t_sel", "num_genotype",
        "external_theta_population", "external_dk_population",
        "external_activity_population", "batch_idx", "geno_theta_idx",
        "scatter_theta"])):
    """Namedtuple with a flax-struct-like ``.replace()`` for exercising the
    full-population congression branch in jax_model without a real GrowthData."""

    def replace(self, **kwargs):
        return self._replace(**kwargs)


MockBindingData = namedtuple("MockBindingData", [])
MockData = namedtuple("MockData", ["growth", "binding"])

MockGrowthPriors = namedtuple("MockGrowthPriors", [
    "theta_growth_noise", "condition_growth", "growth_transition", "ln_cfu0", "dk_geno", "activity", "transformation",
    "growth_noise", "sample_offset", "growth_obs"
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
    
    growth = MockGrowthData(
        t_pre=t_pre, t_sel=t_sel,
        num_genotype=1, external_theta_population=None,
        external_dk_population=None, external_activity_population=None,
        batch_idx=jnp.array([0]), geno_theta_idx=jnp.array([0]),
        # Deliberately non-default (1, not 0) so tests can confirm jax_model
        # preserves it rather than silently resetting it when building the
        # full-population data view (see
        # test_jax_model_population_needed_computes_locally).
        scatter_theta=1,
    )
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
        transformation="prior_trans",
        growth_noise="prior_grn",
        sample_offset="prior_so",
        growth_obs="prior_growth_obs",
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
    get_moments = MagicMock(return_value=(0.0, 1.0)) # (mu, sigma) anchors
    
    # Growth Param Model returns params
    # k=1.0, m=1.0
    mock_params = MagicMock()
    condition_growth_model = MagicMock(return_value=mock_params)
    
    # calculate growth returns g_pre, g_sel
    calculate_growth = MagicMock(return_value=(21.0, 21.0))
    
    # Growth Transition returns total growth
    growth_transition_model = MagicMock(return_value=105.0)
    
    ln_cfu0_model = MagicMock(return_value=jnp.array([5.0])) # ln_cfu0 (must be array for softmax)

    # activity / dk_geno: scalars; with return_population=True they also
    # return a library-ordered population.
    activity_model = MagicMock(
        side_effect=lambda *a, return_population=False, **k:
            (1.0, jnp.array([1.0])) if return_population else 1.0)
    dk_geno_model = MagicMock(
        side_effect=lambda *a, return_population=False, **k:
            (0.0, jnp.array([0.0])) if return_population else 0.0)

    transformation_model = MagicMock(return_value=1.0)  # lambda
    # One class holding every cell (the "single" behavior).
    cell_classes = MagicMock(
        side_effect=lambda focal, population, params, data:
            single_class(jnp.asarray(focal[0]), focal[1], focal[2]))
    
    # Noise models just pass through or add noise. Let's pass through for simplicity.
    theta_binding_noise_model = MagicMock(side_effect=lambda n, x, p, data=None: x)
    theta_growth_noise_model = MagicMock(side_effect=lambda n, x, p, data=None: x)
    growth_noise_model = MagicMock(return_value=0.0)  # sigma_k = 0 → no extra noise
    sample_offset_model = MagicMock(return_value=0.0)  # delta_sample = 0 → no offset

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
        "transformation": (transformation_model, cell_classes, False),
        "theta_binding_noise": theta_binding_noise_model,
        "theta_growth_noise": theta_growth_noise_model,
        "growth_noise": growth_noise_model,
        "sample_offset": sample_offset_model,
        "theta_rescale": lambda t: t,
        "observe_binding": binding_observer,
        "observe_growth": growth_observer,
        "is_guide": False  # Default to main model
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

    # get_moments is no longer used (its only consumer, logit_norm, is gone)
    mock_control["theta"][2].assert_not_called()
    
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
    
    # calculate_growth
    mock_control["calculate_growth"].assert_called_once()
    
    # Growth Transition
    mock_control["growth_transition"].assert_called_once_with(
        "growth_transition", mock_data.growth, ANY, g_pre=21.0, g_sel=21.0, 
        t_pre=jnp.array(2.0), t_sel=jnp.array(3.0), theta=jnp.array(20.0)
    )
    
    # ln_cfu0
    mock_control["ln_cfu0"].assert_called_once_with(
        "ln_cfu0", mock_data.growth, "prior_cfu0"
    )
    
    # Transformation: lambda, then the cell classes built from the
    # genotype's (noisy) theta, activity and dk_geno
    mock_control["transformation"][0].assert_called_once_with(
        "transformation", mock_data.growth, "prior_trans"
    )
    (focal, population, params, data), _ = \
        mock_control["transformation"][1].call_args
    assert jnp.isclose(focal[0], 20.0) and focal[1] == 1.0 and focal[2] == 0.0
    assert population is None
    assert params == 1.0
    assert data is mock_data.growth
    
    # Observers
    mock_control["observe_growth"].assert_called_once()
    mock_control["observe_binding"].assert_called_once()

    # --- 2. Verify Calculation Logic ---
    
    # expected_pred = 5.0 + 105.0 = 110.0
    
    expected_pred = 110.0
    
    # Check that the observer received this prediction
    args, kwargs = mock_control["observe_growth"].call_args
    assert args[0] == "growth" # Name check
    assert args[1] is mock_data.growth
    assert jnp.isclose(args[2], expected_pred)
    assert kwargs["priors"] == "prior_growth_obs"

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
    # growth_observer("growth", data.growth, None, priors=priors.growth.growth_obs)
    args_growth, kwargs_growth = mock_control["observe_growth"].call_args
    assert args_growth[2] is None
    assert kwargs_growth["priors"] == "prior_growth_obs"

    args_binding, _ = mock_control["observe_binding"].call_args
    assert args_binding[2] is None


# ---------------------------------------------------------------------------
# transformation_needs_population wiring (congression mixture)
# ---------------------------------------------------------------------------

def _needs_population(mock_control):
    transformation_model, cell_classes, _ = mock_control["transformation"]
    mock_control["transformation"] = (transformation_model, cell_classes, True)
    return cell_classes


def test_jax_model_population_not_needed_skips_extra_calc_theta(
        mock_data, mock_priors, mock_control):
    """
    Without a population need ("single"), calc_theta runs exactly twice
    (binding, growth), dk_geno/activity are called without
    return_population, and cell_classes gets no population.
    """
    assert mock_control["transformation"][2] is False

    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data, mock_priors, **mock_control)

    assert mock_control["theta"][1].call_count == 2
    assert "return_population" not in mock_control["dk_geno"].call_args.kwargs
    assert "return_population" not in mock_control["activity"].call_args.kwargs
    (_, population, _, _), _ = mock_control["transformation"][1].call_args
    assert population is None


def test_jax_model_population_needed_computes_locally(
        mock_data, mock_priors, mock_control):
    """
    With a population need and no external references (training), jax_model
    computes theta over a full-population data view (batch_idx /
    geno_theta_idx = arange(num_genotype), scatter_theta untouched) and takes
    dk_geno/activity populations from their components.
    """
    cell_classes = _needs_population(mock_control)

    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data, mock_priors, **mock_control)

    # binding, growth, and the extra population pass
    assert mock_control["theta"][1].call_count == 3
    third_call_data = mock_control["theta"][1].call_args_list[2].args[1]
    assert jnp.array_equal(third_call_data.batch_idx, jnp.arange(mock_data.growth.num_genotype))
    assert jnp.array_equal(third_call_data.geno_theta_idx, jnp.arange(mock_data.growth.num_genotype))
    assert third_call_data.scatter_theta == mock_data.growth.scatter_theta

    assert mock_control["dk_geno"].call_args.kwargs["return_population"] is True
    assert mock_control["activity"].call_args.kwargs["return_population"] is True

    (_, population, _, _), _ = cell_classes.call_args
    theta_pop, activity_pop, dk_pop = population
    assert jnp.isclose(theta_pop, 20.0)
    assert jnp.array_equal(activity_pop, jnp.array([1.0]))
    assert jnp.array_equal(dk_pop, jnp.array([0.0]))


def test_jax_model_population_needed_uses_external_override(
        mock_data, mock_priors, mock_control):
    """
    External references (prediction on a genotype subset) are used as-is: no
    local population theta pass, and they replace the components' values.
    """
    cell_classes = _needs_population(mock_control)

    growth = mock_data.growth.replace(
        external_theta_population=jnp.array(999.0),
        external_dk_population=jnp.array([7.0]),
        external_activity_population=jnp.array([3.0]),
    )
    with numpyro.handlers.seed(rng_seed=0):
        jax_model(mock_data._replace(growth=growth), mock_priors, **mock_control)

    assert mock_control["theta"][1].call_count == 2
    (_, population, _, _), _ = cell_classes.call_args
    assert population[0] is growth.external_theta_population
    assert population[1] is growth.external_activity_population
    assert population[2] is growth.external_dk_population


def test_jax_model_population_missing_raises(mock_data, mock_priors,
                                             mock_control):
    """
    If a component cannot supply its population (batch-sized latents) and no
    external reference is given, jax_model refuses rather than building
    congressed cells from nothing.
    """
    _needs_population(mock_control)
    mock_control["dk_geno"] = MagicMock(
        side_effect=lambda *a, return_population=False, **k:
            (0.0, None) if return_population else 0.0)

    with pytest.raises(ValueError, match="dk_geno"):
        with numpyro.handlers.seed(rng_seed=0):
            jax_model(mock_data, mock_priors, **mock_control)
