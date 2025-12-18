
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from tfscreen.models.lac_model.lac_model import LacModel

# Dummy model class to simulate behavior of MicroscopicDimerModel etc
class DummyModel:
    def __init__(self, e_total, o_total, r_total):
        self.equilibrium_constants = ["K1", "K2"]
        self.species_names = ["E", "O", "R", "RO"]
        self.repressor_reactant = ["R", "R"]
        self.repressor_product = ["R", "RO"] # Dummy mapping
        
    def get_fx_operator(self, K_array):
        # K_array corresponds to equilibrium_constants
        return np.sum(K_array)

@pytest.fixture
def dummy_models_dict():
    return {"dummy": DummyModel}

def test_lac_model_init(dummy_models_dict):
    with patch("tfscreen.models.lac_model.lac_model.AVAILABLE_MODELS", dummy_models_dict):
        # Missing constant
        with pytest.raises(ValueError, match="Missing constants"):
            LacModel("dummy", {"K1": 1.0}, 1, 1, 1, 1, 1)

        # Extra constant
        with pytest.raises(ValueError, match="Extra constants"):
            LacModel("dummy", {"K1": 1.0, "K2": 1.0, "K3": 1.0}, 1, 1, 1, 1, 1)
            
        # Success
        model = LacModel("dummy", {"K1": 1.0, "K2": 2.0}, 1, 1, 1.0, 1.0, 298.0)
        
        # Check species: excludes E, O.
        # Dummy internal species: E, O, R, RO.
        # Should contain R, RO.
        assert "R" in model.species
        assert "RO" in model.species
        assert "E" not in model.species
        assert "O" not in model.species

def test_lac_model_get_obs(dummy_models_dict):
    with patch("tfscreen.models.lac_model.lac_model.AVAILABLE_MODELS", dummy_models_dict):
        R = 0.001987
        T = 298.0
        
        # wt_K: K1=1, K2=1. wt_dG = 0.
        model = LacModel("dummy", {"K1": 1.0, "K2": 1.0}, 1, 1, 1.0, R, T)
        
        # Reactant: R, R. Product: R, RO.
        # Indices in species list [R, RO]: R=0, RO=1.
        # dG array size 2 (for K1, K2).
        
        # genotype_ddG: array of length len(species). [ddG_R, ddG_RO].
        # ddG = [0, 0]. No change.
        obs = model.get_obs(np.array([0.0, 0.0]))
        # mut_dG = 0 + (0 - 0) = 0.
        # mut_K = exp(0) = 1.
        # get_fx_operator returns sum(K) = 1+1 = 2.
        assert np.allclose(obs, 2.0)
        
        # Change ddG_RO by -RTln(10). (Stabilize RO by factor of 10).
        # ddG_RO = -RT * ln(10).
        # mut_dG = 0 + (ddG_RO - ddG_R). K2 uses RO as product.
        # K1 also uses RO? No, my dummy:
        # self.repressor_reactant = ["R", "R"]
        # self.repressor_product = ["R", "RO"]
        # So K1 reaction: R -> R. (Silly but dummy). ddG = ddG_R - ddG_R = 0.
        # K2 reaction: R -> RO. ddG = ddG_RO - ddG_R (if ddG_R=0).
        
        val = -R * T * np.log(10.0)
        obs_mut = model.get_obs(np.array([0.0, val]))
        
        # K1: dG = 0 -> K=1.
        # K2: dG = val -> K = exp(- val / RT) = exp(ln(10)) = 10.
        # Sum = 11.
        assert np.allclose(obs_mut, 11.0)

def test_lac_model_unknown_model():
    with pytest.raises(ValueError, match="model 'bad' not recognized"):
        LacModel("bad", {}, 1, 1, 1, 1, 1)
