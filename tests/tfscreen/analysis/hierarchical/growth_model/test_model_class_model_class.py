# ----------------------------------------------------------------------------
# test ModelClass
# ----------------------------------------------------------------------------

import pytest
import pandas as pd
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock, call, patch


# Import the class and its dependencies
from tfscreen.analysis.hierarchical.growth_model.model_class import (
    ModelClass,
    jax_model,
    sample_batch,
    deterministic_batch
)

@pytest.fixture
def mock_modelclass_deps(mocker):
    """
    Mocks all dependencies for ModelClass __init__.
    This is a large fixture to set up the entire environment.
    """
    
    # 1. Mock all helper functions
    mock_growth_df = pd.DataFrame({"map_theta_group": [0, 1, 1, 0]})
    mock_binding_df = pd.DataFrame({"map_theta_group": [1]})
    
    mock_read_growth = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class._read_growth_df",
        return_value=mock_growth_df
    )
    mock_read_binding = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class._read_binding_df",
        return_value=mock_binding_df
    )
    
    # Mock TensorManagers
    mock_growth_tm = MagicMock()
    mock_growth_tm.df = mock_growth_df
    mock_growth_tm.tensors = {
        "ln_cfu": "ln_cfu_tensor", "ln_cfu_std": "std_tensor", "t_pre": "t_pre_tensor",
        "t_sel": "t_sel_tensor", "map_ln_cfu0": "map_ln_cfu0_tensor",
        "map_condition_pre": "map_cond_pre_tensor", 
        "map_condition_sel": "map_cond_sel_tensor",
        "map_genotype": "map_geno_tensor", "map_theta": "map_theta_tensor",
        "good_mask": "growth_good_mask"
    }
    mock_growth_tm.map_sizes = {
        "ln_cfu0": 10, "genotype": 5, "activity": 5, "dk_geno": 5,
        "theta": 15, "condition": 8, "titrant": 6, "replicate": 4
    }
    mock_growth_tm.tensor_shape = (4, 3, 8, 5) # (rep, time, treat, geno)

    mock_growth_theta_tm = MagicMock()
    mock_growth_theta_tm.tensors = {
        "titrant_conc": "growth_titrant_conc_tensor",
        "map_theta_group": "growth_map_theta_group_tensor"
    }
    mock_growth_theta_tm.tensor_shape = (2, 3, 5) # (name, conc, geno)

    mock_binding_tm = MagicMock()
    mock_binding_tm.tensors = {
        "theta_obs": "theta_obs_tensor", "theta_std": "theta_std_tensor",
        "map_theta_group": "binding_map_theta_group_tensor",
        "titrant_conc": "binding_titrant_conc_tensor",
        "good_mask": "binding_good_mask"
    }
    mock_binding_tm.tensor_shape = (2, 3, 5) # (name, conc, geno)

    mock_build_growth_tm = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_tm",
        return_value=mock_growth_tm
    )
    mock_build_growth_theta_tm = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class._build_growth_theta_tm",
        return_value=mock_growth_theta_tm
    )
    mock_build_binding_tm = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class._build_binding_tm",
        return_value=mock_binding_tm
    )
    
    mock_get_wt_info = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class._get_wt_info",
        return_value={"wt_index": 0, "not_wt_mask": jnp.array([1]), "num_not_wt": 1}
    )
    
    # 2. Mock dataclass constructors and populate_dataclass
    mock_growth_data = MagicMock(name="GrowthDataInstance")
    mock_binding_data = MagicMock(name="BindingDataInstance")
    mock_data_class = MagicMock(name="DataClassInstance")
    mock_data_class.growth = mock_growth_data
    mock_data_class.binding = mock_binding_data

    mock_populate = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.populate_dataclass",
        side_effect=[
            mock_growth_data,    # First call returns GrowthData
            mock_binding_data,   # Second call returns BindingData
            mock_data_class      # Third call returns DataClass
        ]
    )
    
    mock_control_class = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.ControlClass"
    )
    mock_priors_class = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.PriorsClass"
    )
    mock_growth_priors = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.GrowthPriors"
    )
    mock_binding_priors = mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.BindingPriors"
    )
    
    # --- FIX: Replace MockComponent class with a MagicMock ---
    # 3. Mock the MODEL_COMPONENT_NAMES
    mock_comp = MagicMock(name="MockComponent")
    mock_comp.get_priors.return_value = "mock_prior"
    # Use side_effect to dynamically create the guess dict
    mock_comp.get_guesses.side_effect = lambda name, data: {f"guess_{name}": 1.0}
    
    mock_comp_dict = {
        "hierarchical": (0, mock_comp),
        "fixed": (1, mock_comp),
        "hill": (2, mock_comp),
        "beta": (3, mock_comp)
    }
    # --- END FIX ---
    
    mock_model_components = {
        "condition_growth": mock_comp_dict,
        "ln_cfu0": mock_comp_dict,
        "dk_geno": mock_comp_dict,
        "activity": mock_comp_dict,
        "theta": mock_comp_dict,
        "theta_growth_noise": mock_comp_dict,
        "theta_binding_noise": mock_comp_dict
    }
    mocker.patch(
        "tfscreen.analysis.hierarchical.growth_model.model_class.MODEL_COMPONENT_NAMES",
        mock_model_components
    )
    
    # Return all mocks for assertions
    return {
        "read_growth": mock_read_growth,
        "read_binding": mock_read_binding,
        "build_growth_tm": mock_build_growth_tm,
        "build_growth_theta_tm": mock_build_growth_theta_tm,
        "build_binding_tm": mock_build_binding_tm,
        "get_wt_info": mock_get_wt_info,
        "populate_dataclass": mock_populate,
        "ControlClass": mock_control_class,
        "PriorsClass": mock_priors_class,
        "GrowthPriors": mock_growth_priors,
        "BindingPriors": mock_binding_priors,
        "mock_component": mock_comp, # This is now a MagicMock
        "mock_growth_df": mock_growth_df,
        "mock_growth_tm": mock_growth_tm,
        "mock_data_class": mock_data_class
    }

class TestModelClass:

    def test_modelclass_init_success(self, mock_modelclass_deps):
        """
        Tests the entire successful initialization sequence of ModelClass.
        """
        # Call the constructor
        model = ModelClass(
            ln_cfu_df="growth.csv",
            theta_obs_df="binding.csv",
            condition_growth="hierarchical",
            ln_cfu0="hierarchical",
            dk_geno="hierarchical",
            activity="fixed",
            theta="hill",
            theta_growth_noise="beta",
            theta_binding_noise="beta"
        )

        # --- 1. Assert _initialize_data() calls ---
        
        # Check helpers
        mock_modelclass_deps["read_growth"].assert_called_once_with("growth.csv")
        mock_modelclass_deps["build_growth_tm"].assert_called_once_with(
            mock_modelclass_deps["mock_growth_df"]
        )
        mock_modelclass_deps["build_growth_theta_tm"].assert_called_once_with(
            mock_modelclass_deps["mock_growth_df"]
        )
        mock_modelclass_deps["get_wt_info"].assert_called_once_with(
            mock_modelclass_deps["mock_growth_tm"]
        )
        mock_modelclass_deps["read_binding"].assert_called_once_with(
            "binding.csv",
            existing_df=mock_modelclass_deps["mock_growth_tm"].df
        )
        
        # Check populate_dataclass calls (the most complex part)
        populate_calls = mock_modelclass_deps["populate_dataclass"].call_args_list
        assert len(populate_calls) == 3
        
        # Call 1: GrowthData
        growth_sources = populate_calls[0].kwargs['sources'] 
        assert growth_sources[0]["ln_cfu"] == "ln_cfu_tensor" # from growth_tm
        assert growth_sources[0]["titrant_conc"] == "growth_titrant_conc_tensor" # from growth_theta_tm
        assert growth_sources[1]["num_ln_cfu0"] == 10 # from sizes
        assert growth_sources[1]["num_titrant_conc"] == 3 # from growth_theta_tm shape
        assert growth_sources[2]["wt_index"] == 0 # from wt_info
        assert growth_sources[3]["scatter_theta"] == 1 # from other_data
        
        # Call 2: BindingData
        binding_sources = populate_calls[1].kwargs['sources']
        assert binding_sources[0]["theta_obs"] == "theta_obs_tensor"
        assert binding_sources[1]["num_genotype"] == 5
        assert binding_sources[2]["scatter_theta"] == 0
        
        # Call 3: DataClass
        data_sources = populate_calls[2].kwargs['sources']
        
        assert data_sources["num_genotype"] == 5
        # Check the growth_to_binding_idx calculation
        # max_idx = 1, so N = 2. binding_values = [1].
        # expected = [-1, 0]
        expected_idx = jnp.array([-1, 0])
        assert jnp.array_equal(data_sources["growth_to_binding_idx"], expected_idx)
        
        # Assert final data object is set
        assert model._data is mock_modelclass_deps["mock_data_class"]

        # --- 2. Assert _initialize_classes() calls ---
        
        # --- FIX: This will now pass ---
        # Check that get_guesses was called
        mock_comp = mock_modelclass_deps["mock_component"]
        assert mock_comp.get_guesses.call_count == 7
        assert mock_comp.get_priors.call_count == 7
        # --- END FIX ---
        
        # Check ControlClass
        expected_control_kwargs = {
            "condition_growth": 0, "ln_cfu0": 0, "dk_geno": 0,
            "activity": 1, "theta": 2, "theta_growth_noise": 3,
            "theta_binding_noise": 3
        }
        mock_modelclass_deps["ControlClass"].assert_called_once_with(
            **expected_control_kwargs
        )
        
        # Check Priors
        mock_modelclass_deps["GrowthPriors"].assert_called_once()
        mock_modelclass_deps["BindingPriors"].assert_called_once()
        mock_modelclass_deps["PriorsClass"].assert_called_once()

        # Check final attributes
        assert model._control is mock_modelclass_deps["ControlClass"].return_value
        assert model._priors is mock_modelclass_deps["PriorsClass"].return_value
        assert model._init_params["guess_theta"] == 1.0
        
    def test_modelclass_init_bad_model_name(self, mock_modelclass_deps):
        """
        Tests that a ValueError is raised for an unrecognized model name.
        """
        with pytest.raises(ValueError, match="theta 'bad_name' not recognized"):
            ModelClass(
                ln_cfu_df="g.csv",
                theta_obs_df="b.csv",
                theta="bad_name" # This is the invalid name
            )
            
        # Check that we failed *during* _initialize_classes
        # _initialize_data should have been called
        mock_modelclass_deps["read_growth"].assert_called_once()
        # ControlClass should *not* have been called
        mock_modelclass_deps["ControlClass"].assert_not_called()

    def test_modelclass_properties(self, mocker):
        """
        Tests that all properties return the correct private attributes.
        """
        # Disable __init__ logic
        mocker.patch.object(ModelClass, "_initialize_data")
        mocker.patch.object(ModelClass, "_initialize_classes")
        
        # Create a "blank" instance
        model = ModelClass("g.csv", "b.csv")
        
        # Manually set private attributes
        model._data = "data_obj"
        model._priors = "priors_obj"
        model._control = "control_obj"
        model._init_params = "init_params_dict"
        
        # Test properties
        assert model.data == "data_obj"
        assert model.priors == "priors_obj"
        assert model.control == "control_obj"
        assert model.init_params == "init_params_dict"
        
        # Test properties that return imported functions/vars
        assert model.jax_model is jax_model
        assert model.sample_batch is sample_batch
        assert model.deterministic_batch is deterministic_batch
        
        # Test settings property
        assert model.settings["theta"] == "hill" # The default
        model._theta = "custom_theta"
        assert model.settings["theta"] == "custom_theta"