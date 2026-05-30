"""
Tests for simulate.sample_activity.sample_activity_prior.

Structure
---------
* Validation tests — unknown component name raises ValueError; error message
  lists available components.  These patch the registry to avoid JAX I/O.
* Output-shape / type tests — returns 1-D numpy array of length num_genotype.
  These use a mock activity module and patch handlers to skip real sampling.
* Argument-passing tests — priors_overrides merged correctly; define_model
  called with the right (name, sim_data, priors) triple.
* Integration tests — use real registered components with a minimal
  MagicMock SimData that provides exactly the fields each component reads.
  These touch JAX/NumPyro but do not run full inference.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
import jax
import jax.numpy as jnp

from tfscreen.simulate.sample_activity import sample_activity_prior


# ============================================================================
# Shared helpers
# ============================================================================

def _make_mock_module(G: int):
    """
    Return a mock activity module whose define_model returns a
    (1, 1, 1, 1, 1, 1, G) array of ones — the broadcast shape that real
    activity components produce.
    """
    mock_module = MagicMock()
    mock_module.get_hyperparameters.return_value = {"tau_scale": 0.1}
    mock_module.ModelPriors.return_value = MagicMock(name="mock_priors")
    mock_module.define_model.return_value = jnp.ones((1, 1, 1, 1, 1, 1, G))
    return mock_module


def _make_mock_sim_data(G: int = 5):
    """
    Minimal SimData-like mock supplying fields read by horseshoe / hierarchical.
    """
    sd = MagicMock(name="sim_data")
    sd.num_genotype = G
    sd.batch_size = G
    sd.batch_idx = jnp.arange(G)
    sd.wt_indexes = jnp.array([0], dtype=jnp.int32)
    sd.scale_vector = jnp.ones(G)
    return sd


# ============================================================================
# Validation tests  (registry is patched; no JAX executed)
# ============================================================================

class TestValidation:

    def test_raises_on_unknown_component(self):
        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"fixed": MagicMock()}}):
            with pytest.raises(ValueError, match="not found"):
                sample_activity_prior("does_not_exist", _make_mock_sim_data(),
                                      rng_key=0)

    def test_error_message_contains_component_name(self):
        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"fixed": MagicMock()}}):
            with pytest.raises(ValueError, match="does_not_exist"):
                sample_activity_prior("does_not_exist", _make_mock_sim_data(),
                                      rng_key=0)

    def test_error_message_lists_available_keys(self):
        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"fixed": MagicMock(), "horseshoe": MagicMock()}}):
            with pytest.raises(ValueError, match="fixed"):
                sample_activity_prior("bogus", _make_mock_sim_data(), rng_key=0)


# ============================================================================
# Output shape / type tests  (handlers patched; module mocked)
# ============================================================================

class TestOutputShapeAndType:

    def test_returns_1d_array(self):
        G = 5
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                result = sample_activity_prior("horseshoe", sim_data, rng_key=0)

        assert result.ndim == 1

    def test_returns_correct_length(self):
        G = 7
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                result = sample_activity_prior("horseshoe", sim_data, rng_key=0)

        assert result.shape == (G,)

    def test_returns_numpy_array(self):
        G = 4
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                result = sample_activity_prior("horseshoe", sim_data, rng_key=0)

        assert isinstance(result, np.ndarray)

    def test_strips_broadcast_dimensions(self):
        """The 7-D broadcast tensor must be collapsed to shape (G,)."""
        G = 6
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        # define_model returns (1,1,1,1,1,1,G); result must be (G,)
        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                result = sample_activity_prior("horseshoe", sim_data, rng_key=0)

        assert result.shape == (G,)
        np.testing.assert_allclose(result, 1.0)  # mock returns ones


# ============================================================================
# Argument-passing tests
# ============================================================================

class TestArgumentPassing:

    def test_priors_overrides_merged_into_hyperparameters(self):
        G = 4
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        overrides = {"tau_scale": 0.5}
        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                sample_activity_prior("horseshoe", sim_data, rng_key=0,
                                      priors_overrides=overrides)

        call_kwargs = mock_module.ModelPriors.call_args[1]
        assert call_kwargs["tau_scale"] == 0.5

    def test_priors_overrides_none_uses_defaults(self):
        G = 4
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                sample_activity_prior("horseshoe", sim_data, rng_key=0,
                                      priors_overrides=None)

        call_kwargs = mock_module.ModelPriors.call_args[1]
        assert call_kwargs["tau_scale"] == 0.1  # default from get_hyperparameters

    def test_define_model_called_with_activity_name(self):
        G = 3
        mock_module = _make_mock_module(G)
        mock_priors = mock_module.ModelPriors.return_value
        sim_data = _make_mock_sim_data(G)

        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                sample_activity_prior("horseshoe", sim_data, rng_key=0)

        mock_module.define_model.assert_called_once_with(
            "activity", sim_data, mock_priors
        )

    def test_define_model_receives_sim_data_unchanged(self):
        G = 3
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)

        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers"):
                sample_activity_prior("horseshoe", sim_data, rng_key=0)

        positional_args = mock_module.define_model.call_args[0]
        assert positional_args[1] is sim_data

    def test_rng_key_forwarded_to_handlers_seed(self):
        G = 3
        mock_module = _make_mock_module(G)
        sim_data = _make_mock_sim_data(G)
        mock_handlers = MagicMock()
        mock_handlers.seed.return_value.__enter__ = MagicMock(return_value=None)
        mock_handlers.seed.return_value.__exit__ = MagicMock(return_value=False)

        test_key = jax.random.PRNGKey(42)
        with patch("tfscreen.simulate.sample_activity.model_registry",
                   {"activity": {"horseshoe": mock_module}}):
            with patch("tfscreen.simulate.sample_activity.handlers", mock_handlers):
                sample_activity_prior("horseshoe", sim_data, rng_key=test_key)

        mock_handlers.seed.assert_called_once_with(rng_seed=test_key)


# ============================================================================
# Integration tests — real activity components, minimal mock SimData
# ============================================================================
# These tests require JAX and NumPyro but do not run full inference.
# They verify that the real registered components behave correctly when
# called through sample_activity_prior with the fields they actually read.

def _build_activity_sim_data(genotypes):
    """
    Minimal SimData-like mock with all fields that the genotype-level
    activity components (horseshoe, hierarchical) actually access.
    """
    G = len(genotypes)
    sd = MagicMock(name="activity_sim_data")
    sd.num_genotype = G
    sd.batch_size = G
    sd.batch_idx = jnp.arange(G, dtype=jnp.int32)
    sd.wt_indexes = jnp.array(
        [i for i, g in enumerate(genotypes) if g == "wt"], dtype=jnp.int32
    )
    sd.scale_vector = jnp.ones(G)
    return sd


class TestFixedComponent:

    def test_returns_all_ones(self):
        genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("fixed", sim_data,
                                       rng_key=jax.random.PRNGKey(0))
        np.testing.assert_allclose(result, 1.0)

    def test_correct_length(self):
        genotypes = ["wt"] + [f"M{i}" for i in range(10)]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("fixed", sim_data,
                                       rng_key=jax.random.PRNGKey(0))
        assert result.shape == (11,)

    def test_returns_numpy_array(self):
        genotypes = ["wt", "A1B"]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("fixed", sim_data,
                                       rng_key=jax.random.PRNGKey(0))
        assert isinstance(result, np.ndarray)


class TestHorseshoeComponent:

    def test_returns_positive_values(self):
        genotypes = ["wt"] + [f"M{i}" for i in range(10)]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("horseshoe", sim_data,
                                       rng_key=jax.random.PRNGKey(42))
        assert np.all(result > 0.0)

    def test_correct_shape(self):
        genotypes = ["wt", "A1B", "C2D"]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("horseshoe", sim_data,
                                       rng_key=jax.random.PRNGKey(0))
        assert result.shape == (3,)

    def test_wt_is_exactly_one(self):
        """Horseshoe component must pin wt genotype to activity = 1.0."""
        genotypes = ["wt", "A1B", "C2D", "A1B/C2D"]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("horseshoe", sim_data,
                                       rng_key=jax.random.PRNGKey(5))
        wt_idx = genotypes.index("wt")
        assert np.isclose(result[wt_idx], 1.0), (
            f"Expected wt activity = 1.0, got {result[wt_idx]}"
        )

    def test_reproducible_with_same_key(self):
        genotypes = ["wt"] + [f"M{i}" for i in range(6)]
        sim_data = _build_activity_sim_data(genotypes)
        key = jax.random.PRNGKey(99)
        r1 = sample_activity_prior("horseshoe", sim_data, rng_key=key)
        r2 = sample_activity_prior("horseshoe", sim_data, rng_key=key)
        np.testing.assert_array_equal(r1, r2)

    def test_different_keys_give_different_values(self):
        """Two distinct keys should (with overwhelming probability) differ."""
        genotypes = ["wt"] + [f"M{i}" for i in range(8)]
        sim_data = _build_activity_sim_data(genotypes)
        r1 = sample_activity_prior("horseshoe", sim_data,
                                    rng_key=jax.random.PRNGKey(1))
        r2 = sample_activity_prior("horseshoe", sim_data,
                                    rng_key=jax.random.PRNGKey(2))
        assert not np.allclose(r1, r2)

    def test_priors_override_tau_scale(self):
        """A very small tau_scale should pull activities very close to 1.0."""
        genotypes = ["wt"] + [f"M{i}" for i in range(15)]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior(
            "horseshoe", sim_data,
            rng_key=jax.random.PRNGKey(0),
            priors_overrides={"global_scale_tau_scale": 1e-6},
        )
        # With a vanishingly small tau, all activities should be ≈ 1.0.
        np.testing.assert_allclose(result, 1.0, atol=1e-3)


class TestHierarchicalComponent:

    def test_returns_positive_values(self):
        genotypes = ["wt"] + [f"M{i}" for i in range(8)]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("hierarchical", sim_data,
                                       rng_key=jax.random.PRNGKey(7))
        assert np.all(result > 0.0)

    def test_correct_shape(self):
        genotypes = ["wt", "X1Y", "X2Y"]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("hierarchical", sim_data,
                                       rng_key=jax.random.PRNGKey(3))
        assert result.shape == (3,)

    def test_wt_is_exactly_one(self):
        """Hierarchical component must pin wt genotype to activity = 1.0."""
        genotypes = ["wt", "A1B", "C2D"]
        sim_data = _build_activity_sim_data(genotypes)
        result = sample_activity_prior("hierarchical", sim_data,
                                       rng_key=jax.random.PRNGKey(1))
        wt_idx = genotypes.index("wt")
        assert np.isclose(result[wt_idx], 1.0)

    def test_reproducible_with_same_key(self):
        genotypes = ["wt", "A1B", "C2D", "D4E"]
        sim_data = _build_activity_sim_data(genotypes)
        key = jax.random.PRNGKey(55)
        r1 = sample_activity_prior("hierarchical", sim_data, rng_key=key)
        r2 = sample_activity_prior("hierarchical", sim_data, rng_key=key)
        np.testing.assert_array_equal(r1, r2)
