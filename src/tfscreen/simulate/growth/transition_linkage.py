
import numpy as np

class TransitionModel:
    """Base class for theta-to-transition (tau, k_sharp) linkage models."""
    name = "base"
    
    def predict_tau(self, theta, params):
        """
        Predict shift time (tau) based on theta and parameters.
        """
        raise NotImplementedError

    def predict_k_sharp(self, theta, params):
        """
        Predict transition sharpness (k_sharp) based on theta and parameters.
        """
        raise NotImplementedError

    def get_param_defs(self):
        """Return list of (suffix, guess, transform, description)."""
        raise NotImplementedError

class ConstantTransition(TransitionModel):
    """
    Constant transition parameters: tau and k_sharp are independent of theta.
    
    Params:
    - tau: shift time
    - k_sharp: transition sharpness
    """
    name = "constant"

    def predict_tau(self, theta, params):
        # params: [tau, k_sharp]
        return params[0]

    def predict_k_sharp(self, theta, params):
        # params: [tau, k_sharp]
        return params[1]

    def get_param_defs(self):
        return [
            ("tau", 0.0, "none", "Shift time (lag)"),
            ("k_sharp", 1.0, "none", "Transition sharpness")
        ]

class ProteinDilutionTransition(TransitionModel):
    """
    Protein-dilution inspired transition: tau depends on theta.
    
    tau = tau0 + tau_scale / (tau_offset + theta)
    k_sharp = 100.0 (fixed, sharp transition)
    
    Params:
    - tau0: baseline shift
    - tau_scale: scale factor for theta dependence
    - tau_offset: offset for theta (prevents division by zero)
    """
    name = "protein_dilution"

    def predict_tau(self, theta, params):
        # params: [tau0, tau_scale, tau_offset]
        tau0, tau_scale, tau_offset = params
        return tau0 + tau_scale / (tau_offset + theta)

    def predict_k_sharp(self, theta, params):
        # Fixed sharp transition
        return 100.0

    def get_param_defs(self):
        return [
            ("tau0", 0.0, "none", "Baseline shift"),
            ("tau_scale", 1.0, "none", "Shift scale factor"),
            ("tau_offset", 0.1, "none", "Shift offset")
        ]

TRANSITION_REGISTRY = {
    "constant": ConstantTransition,
    "protein_dilution": ProteinDilutionTransition
}

def get_transition_model(name):
    if name not in TRANSITION_REGISTRY:
        raise ValueError(f"Unknown transition model: {name}. Available: {list(TRANSITION_REGISTRY.keys())}")
    return TRANSITION_REGISTRY[name]()
