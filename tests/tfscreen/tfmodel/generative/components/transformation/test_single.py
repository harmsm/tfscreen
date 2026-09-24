import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock

from tfscreen.tfmodel.generative.components.transformation import single as transformation_single


def test_get_hyperparameters():
    """Verify get_hyperparameters returns empty dict."""
    params = transformation_single.get_hyperparameters()
    assert params == {}
    assert isinstance(params, dict)


def test_get_guesses():
    """Verify get_guesses returns empty dict."""
    guesses = transformation_single.get_guesses("test", MagicMock())
    assert guesses == {}
    assert isinstance(guesses, dict)


def test_get_priors():
    """Verify get_priors returns a ModelPriors instance."""
    priors = transformation_single.get_priors()
    assert isinstance(priors, transformation_single.ModelPriors)


def test_define_model_and_guide_have_no_parameters():
    assert transformation_single.define_model("test", MagicMock(), MagicMock()) is None
    assert transformation_single.guide("test", MagicMock(), MagicMock()) is None


def test_cell_classes_is_one_class_of_weight_one():
    """Every cell is clean: one class holding the genotype's own values."""
    theta = jnp.linspace(0.1, 0.9, 2 * 3 * 4).reshape(1, 1, 1, 1, 2, 3, 4)
    activity = jnp.array([1.0, 0.5, 2.0, 1.0]).reshape(1, 1, 1, 1, 1, 1, 4)
    dk_geno = jnp.array([0.0, -0.01, 0.02, 0.0]).reshape(1, 1, 1, 1, 1, 1, 4)

    classes = transformation_single.cell_classes((theta, activity, dk_geno),
                                                 None, None, MagicMock())

    np.testing.assert_array_equal(classes.theta, theta[None])
    np.testing.assert_array_equal(classes.activity, activity[None])
    np.testing.assert_array_equal(classes.dk_geno, dk_geno[None])
    assert classes.log_weight.shape[0] == 1
    np.testing.assert_array_equal(np.exp(classes.log_weight), 1.0)


def test_needs_population_flag():
    """No congression, so no library-wide population is needed."""
    assert transformation_single.NEEDS_POPULATION is False
