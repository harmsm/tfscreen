
import numpy as np

class LinkageModel:
    """Base class for theta-to-growth linkage models."""
    name = "base"
    
    def predict(self, theta, params):
        raise NotImplementedError

    def get_param_defs(self):
        """Return list of (suffix, guess, transform, description)."""
        raise NotImplementedError

class LinearModel(LinkageModel):
    """
    Linear linkage: growth = intercept + slope * theta
    
    Params:
    - b: intercept (at theta=0)
    - m: slope (change in growth per unit theta)
    """
    name = "linear"

    def predict(self, theta, params):
        # params is formatted as [b, m] relative to the condition
        return params[0] + params[1] * theta

    def get_param_defs(self):
        return [
            ("b", 0.0, "none", "Intercept (theta=0)"),
            ("m", 0.0, "none", "Slope (d_growth/d_theta)")
        ]

class PowerLawModel(LinkageModel):
    """
    Power law linkage: growth = intercept + coeff * theta^power
    
    Params:
    - b: intercept (at theta=0)
    - a: coefficient/amplitude
    - n: power exponent
    """
    name = "power_law"

    def predict(self, theta, params):
        # params: [b, a, n]
        b, a, n = params
        return b + a * (theta ** n)

    def get_param_defs(self):
        return [
            ("b", 0.0, "none", "Intercept (theta=0)"),
            ("a", 0.0, "none", "Coefficient"),
            ("n", 1.0, "none", "Power exponent")
        ]

class SaturationModel(LinkageModel):
    """
    Saturation linkage: growth = min + (max - min) * (theta / (1 + theta))
    This is effectively a Hill function with n=1 and K=1, scaled.
    Alternatively: growth = b + range * (theta / (K + theta)) ?
    User asked for: min + (max - min)*(theta/(1 + theta))
    
    Params:
    - min: growth at theta=0
    - max: asymptotic maximum growth (at theta -> infinity)
    """
    name = "saturation"

    def predict(self, theta, params):
        # params: [min, max]
        min_val, max_val = params
        return min_val + (max_val - min_val) * (theta / (1.0 + theta))

    def get_param_defs(self):
        return [
            ("min", 0.0, "none", "Minimum growth (theta=0)"),
            ("max", 0.0, "none", "Maximum growth (theta -> inf)")
        ]

MODEL_REGISTRY = {
    "linear": LinearModel,
    "power_law": PowerLawModel,
    "saturation": SaturationModel
}

def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
