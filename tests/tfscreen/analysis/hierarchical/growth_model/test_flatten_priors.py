import pytest
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from flax.struct import dataclass
from typing import Any
from unittest.mock import MagicMock

from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass

@dataclass(frozen=True)
class MockComponentPriors:
    scale_param: float
    std_param: float
    beta_kappa_scale: float
    some_rate: float
    other_param: float

@dataclass(frozen=True)
class MockPriorsClass:
    theta: Any
    growth: Any
    binding: Any

def test_flatten_priors():
    """
    Test that flatten_priors recursively updates all fields:
    - Containing 'scale' or 'std' to 100.0 (unless they are rates).
    - Containing 'rate' to 1e-6.
    - 'beta_kappa_scale' to 1e-6.
    """
    
    # Create a mock model class
    model = MagicMock(spec=ModelClass)
    
    # Define a nested prior structure
    initial_priors = MockPriorsClass(
        theta=MockComponentPriors(scale_param=1.0, std_param=0.1, beta_kappa_scale=1.0, some_rate=1.0, other_param=5.0),
        growth=MockComponentPriors(scale_param=2.0, std_param=0.2, beta_kappa_scale=1.0, some_rate=1.0, other_param=10.0),
        binding=MockComponentPriors(scale_param=3.0, std_param=0.3, beta_kappa_scale=1.0, some_rate=1.0, other_param=15.0)
    )
    
    model._priors = initial_priors
    
    # Bind the method to the mock instance
    model.flatten_priors = ModelClass.flatten_priors.__get__(model, ModelClass)
    
    # Run flatten_priors
    model.flatten_priors()
    
    # Verify updates
    updated_priors = model._priors
    
    # Check theta
    assert updated_priors.theta.scale_param == 100.0
    assert updated_priors.theta.std_param == 100.0
    assert updated_priors.theta.beta_kappa_scale == 1e-6
    assert updated_priors.theta.some_rate == 1e-6
    assert updated_priors.theta.other_param == 5.0
    
    # Check growth
    assert updated_priors.growth.scale_param == 100.0
    assert updated_priors.growth.std_param == 100.0
    assert updated_priors.growth.beta_kappa_scale == 1e-6
    assert updated_priors.growth.some_rate == 1e-6
    assert updated_priors.growth.other_param == 10.0

def test_flatten_priors_nested():
    """
    Test that flatten_priors handles deeper nesting and specific rate names.
    """
    @dataclass(frozen=True)
    class DeepChild:
        deep_scale: float
        deep_rate: float
        val: float

    @dataclass(frozen=True)
    class MidChild:
        child: DeepChild
        mid_std: float

    @dataclass(frozen=True)
    class Root:
        mid: MidChild
        top_val: float

    model = MagicMock(spec=ModelClass)
    model._priors = Root(
        mid=MidChild(
            child=DeepChild(deep_scale=1.0, deep_rate=1.0, val=2.0),
            mid_std=3.0
        ),
        top_val=4.0
    )
    
    model.flatten_priors = ModelClass.flatten_priors.__get__(model, ModelClass)
    model.flatten_priors()
    
    updated = model._priors
    assert updated.mid.child.deep_scale == 100.0
    assert updated.mid.child.deep_rate == 1e-6
    assert updated.mid.child.val == 2.0
    assert updated.mid.mid_std == 100.0
    assert updated.top_val == 4.0
