
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scipy.optimize import OptimizeResult

from tfscreen.models.lac_model.linkage_dimer import LinkageDimerModel
from tfscreen.models.lac_model.mwc_dimer import MWCDimerModel
from tfscreen.models.lac_model.linkage_dimer_tetramer import LinkageDimerTetramerModel
from tfscreen.models.lac_model.microscopic_dimer import MicroscopicDimerModel

# ----------------------------------------------------------------------------
# Parameter sets for each model to be used in parametrization
# ----------------------------------------------------------------------------

MODELS_TO_TEST = [
    (LinkageDimerModel, 5, {'R', 'O', 'E'}),
    (MWCDimerModel, 5, {'H', 'L', 'O', 'E'}),
    (LinkageDimerTetramerModel, 12, {'R2', 'R4', 'O', 'E'}),
    (MicroscopicDimerModel, 12, {'L', 'H', 'U', 'O', 'E'}),
]

@pytest.mark.parametrize("ModelClass, num_K, expected_species_subset", MODELS_TO_TEST)
class TestCommonModelBehavior:
    """
    Tests that are common to all LacModel implementations.
    """

    def test_initialization_scalars(self, ModelClass, num_K, expected_species_subset):
        """Test initialization with scalar inputs."""
        r_tot, o_tot, e_tot = 1e-9, 1e-9, 1e-9
        model = ModelClass(r_tot, o_tot, e_tot)
        
        assert model.num_conditions == 1
        assert np.allclose(model.r_total_dimer if hasattr(model, 'r_total_dimer') else model.r_total, 
                           r_tot/2 if hasattr(model, 'r_total_dimer') else r_tot)

    def test_initialization_broadcasting(self, ModelClass, num_K, expected_species_subset):
        """Test proper broadcasting of inputs."""
        r_tot = np.array([1e-9, 2e-9])
        o_tot = 1e-9
        e_tot = np.array([1e-6, 1e-5]) # Shape (2,) matches r_tot
        
        model = ModelClass(r_tot, o_tot, e_tot)
        assert model.num_conditions == 2

        # Test incompatible shapes
        with pytest.raises(ValueError, match="broadcast"):
            ModelClass(np.zeros(3), 1.0, np.zeros(2))

    def test_get_concs_bad_K_array(self, ModelClass, num_K, expected_species_subset):
        """Test validation of K_array input."""
        model = ModelClass(1e-9, 1e-9, 1e-9)
        
        # Wrong type
        with pytest.raises(ValueError, match="1D NumPy array"):
            model.get_concs([1]*num_K)
            
        # Wrong shape
        with pytest.raises(ValueError, match="1D NumPy array"):
            model.get_concs(np.ones(num_K + 1))

    def test_mass_conservation_zero_interaction(self, ModelClass, num_K, expected_species_subset):
        """
        Test that mass is conserved. 
        We use a dummy K_array. Exact solution depends on values, but mass conservation must hold.
        To avoid solver issues, we pick 'weak' binding constants (large Kd or small Ka).
        
        Since definitions vary (Ka vs Kd), we try to pick values that allow convergence.
        """
        r_tot, o_tot, e_tot = 1e-7, 1e-8, 1e-5
        model = ModelClass(r_tot, o_tot, e_tot)
        
        # Construct K_array
        # LinkageDimer (Kd): Large values -> weak binding
        # Others (Ka): Small values -> weak binding
        is_dissociation = (ModelClass == LinkageDimerModel)
        
        if is_dissociation:
            K_val = 1.0 # High Kd = weak binding
        else:
            # Special case for MicroscopicDimer: K_l_2u (dissociation), K_l_h (equilibrium)
            # We treat generic filler.
            K_val = 1e-6 # Low Ka = weak binding
            
        K_array = np.full(num_K, K_val)
        
        # Specifically tweak constants for convergence if needed per model logic
        if ModelClass == MicroscopicDimerModel:
            # K_l_2u (dissoc), K_l_h (eq), others (assoc)
            K_array[0] = 1e-9 # K_l_2u
            K_array[1] = 1.0  # K_l_h
        elif ModelClass == LinkageDimerTetramerModel:
             # All Association
             K_array[:] = 1e3 # Moderate association
        elif ModelClass == MWCDimerModel:
             # K_h_l unitless, others assoc
             K_array[0] = 1.0
             K_array[1:] = 1e6
        
        concs = model.get_concs(K_array)
        
        # Check for NaNs (solver failure) - finding a stable set for all is tricky generally,
        # but with these simple inputs it should work.
        assert not np.isnan(concs).any(), "Solver returned NaNs"
        
        # Check Mass Conservation
        # E total
        e_calc = np.sum(concs * model.e_stoich, axis=1)
        assert np.allclose(e_calc, model.e_total, rtol=1e-3), f"Effector mass not conserved for {ModelClass.__name__}"
        
        # O total
        o_calc = np.sum(concs * model.o_stoich, axis=1)
        assert np.allclose(o_calc, model.o_total, rtol=1e-3), f"Operator mass not conserved for {ModelClass.__name__}"
        
        # R total
        # Model stores r_total differently.
        # r_stoich is in terms of Monomers usually?
        # LinkageDimer: Species R, RE, etc are dimers. r_stoich=[2, 2...].
        # r_total passed to init is MONOMER units.
        r_calc = np.sum(concs * model.r_stoich, axis=1)
        
        # Check against r_total property
        target_r = model.r_total if hasattr(model, 'r_total') else model.r_total_dimer * 2
        
        assert np.allclose(r_calc, target_r, rtol=1e-3), f"Repressor mass not conserved for {ModelClass.__name__}"


    def test_get_fx_operator_returns_valid_range(self, ModelClass, num_K, expected_species_subset):
        """Test that get_fx_operator returns values between 0 and 1."""
        model = ModelClass(1e-7, 1e-8, 1e-5)
        # Use ones, should converge to something
        if ModelClass == LinkageDimerModel:
            K_array = np.full(num_K, 1e-6) # Tight binding Kd
        else:
            K_array = np.full(num_K, 1e6) # Tight binding Ka
            if ModelClass == MicroscopicDimerModel:
                K_array[0] = 1e-9; K_array[1] = 1.0
            if ModelClass == MWCDimerModel:
                 K_array[0] = 1.0

        fx = model.get_fx_operator(K_array)
        assert np.all(fx >= 0.0)
        assert np.all(fx <= 1.0 + 1e-6) # Allow slight numerical tolerance

    def test_repressor_reactant_product_consistency(self, ModelClass, num_K, expected_species_subset):
        """Check that reactant/product definitions match species."""
        assert len(ModelClass.repressor_reactant) == num_K
        assert len(ModelClass.repressor_product) == num_K
        
        for s in ModelClass.repressor_reactant:
            assert s in ModelClass.species_names
        for s in ModelClass.repressor_product:
            assert s in ModelClass.species_names


    def test_solver_failure_handling(self, ModelClass, num_K, expected_species_subset):
        """Test that solver failure emits warning and returns NaNs."""
        model = ModelClass(1e-9, 1e-9, 1e-9)
        K_array = np.ones(num_K)
        
        # Path to root varies slightly or we mock it globally in the module
        # But 'root' is imported into the module namespace.
        module_path = ModelClass.__module__
        
        with patch(f"{module_path}.root") as mock_root:
            mock_res = MagicMock(spec=OptimizeResult)
            mock_res.success = False
            mock_res.message = "Mock failure"
            mock_root.return_value = mock_res
            
            with pytest.warns(UserWarning, match="Solver failed"):
                concs = model.get_concs(K_array)
                
            assert np.all(np.isnan(concs))
