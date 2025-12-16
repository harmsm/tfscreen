
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# We need to mock 'eee' module before importing EEEModel if we want to avoid import error
# But EEEModel imports it inside __init__. 
# So we can import EEEModel, but instantiating it requires mocking.

from tfscreen.models.eee_model import EEEModel

def test_eee_model_init(tmp_path):
    # Test initialization with mocked eee
    with patch.dict(sys.modules, {"eee": MagicMock(), "eee.io": MagicMock()}):
        import eee
        eee.io.read_ensemble = MagicMock()
        mock_ens = MagicMock()
        eee.io.read_ensemble.return_value = mock_ens
        
        # Test file input (doesn't need to exist if mocked)
        # But read_ensemble is called with it.
        
        model = EEEModel(
            ensemble_spreadsheet="dummy.xlsx",
            scale_obs_by=10.0,
            e_total=1e-3,
            R=0.001987,
            T=298.0
        )
        
        eee.io.read_ensemble.assert_called_with("dummy.xlsx", gas_constant=0.001987)
        mock_ens.read_ligand_dict.assert_called()
        
        # Check properties
        mock_ens.species = ["s1", "s2"]
        assert model.species == ["s1", "s2"]

def test_eee_model_get_obs():
    with patch.dict(sys.modules, {"eee": MagicMock(), "eee.io": MagicMock()}):
        import eee
        mock_ens = MagicMock()
        eee.io.read_ensemble = MagicMock(return_value=mock_ens)
        
        # Setup mock behavior for get_fx_obs_fast
        # It takes mut_energy_array and temperature
        # Returns fx_occupied, fx_folded
        
        model = EEEModel("dummy", scale_obs_by=2.0, e_total=1.0, R=1.0, T=1.0)
        
        mock_ens.get_fx_obs_fast.return_value = (np.array([0.5]), np.array([0.8]))
        
        ddG = np.array([0.1, 0.2])
        obs = model.get_obs(ddG)
        
        # obs = 0.5 * 0.8 * 2.0 = 0.4 * 2.0 = 0.8
        assert np.allclose(obs, 0.8)
        
        mock_ens.get_fx_obs_fast.assert_called()

def test_eee_model_import_error():
    # Simulate eee missing
    with patch.dict(sys.modules):
        if "eee" in sys.modules:
            del sys.modules["eee"]
        # We also need to ensure it can't be imported.
        # One way is to set sys.modules['eee'] = None? No that raises ModuleNotFoundError or ImportError directly.
        # Or mock __import__?
        # Or SideEffect on import?
        
        # Simpler: just ensure init raises if Import fails.
        # But if 'eee' is installed in environment, we need to hide it.
        # patch.dict(sys.modules) restores after, but deleting from it inside might trigger reload if code tries 'import eee'.
        # We need check if `import eee` inside `__init__` fails.
        
        # Let's try mocking sys.modules so 'eee' is not there, and providing a loader that fails?
        # Easier: patch builtins.__import__?
        pass

    # Actually simpler:
    with patch.dict(sys.modules, {"eee": None}):
         with pytest.raises(ImportError, match="eee library not found"):
             EEEModel("dummy", 1, 1, 1, 1)

