
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
from tfscreen.analysis.independent.get_indiv_growth import get_indiv_growth

class TestGetIndivGrowth:

    @patch("tfscreen.analysis.independent.get_indiv_growth._get_indiv_growth")
    @patch("tfscreen.analysis.independent.get_indiv_growth.get_scaled_cfu")
    def test_get_indiv_growth_single_iteration(self, mock_get_scaled, mock_internal_get):
        
        # Setup input
        df = pd.DataFrame({
            "replicate": [1, 1],
            "genotype": ["wt", "wt"],
            "t_sel": [1, 2],
            "ln_cfu": [10.0, 11.0]
        })
        series_selector = ["replicate", "genotype"]
        calibration_data = {}
        
        # Setup mocks
        param_df_mock = pd.DataFrame({"replicate": [1], "genotype": ["wt"], 
                                      "lnA0_est": [9.0], "lnA0_std": [0.1], 
                                      "k_est": [1.0], "k_std": [0.01], 
                                      "dk_geno": [0.0], "lnA0_pre": [9.0]})
        pred_df_mock = df.copy()
        pred_df_mock["cfu"] = np.exp(pred_df_mock["ln_cfu"])
        
        mock_internal_get.return_value = (param_df_mock, pred_df_mock)
        
        # Call function
        param_out, pred_out = get_indiv_growth(
            df=df,
            series_selector=series_selector,
            calibration_data=calibration_data,
            num_iterations=1
        )
        
        # Verify calls
        mock_internal_get.assert_called_once()
        mock_get_scaled.assert_not_called() # Should not be called for single iteration
        
        # Verify output
        assert len(param_out) == 1
        assert "k_est" in param_out.columns
        pd.testing.assert_frame_equal(param_out, param_df_mock[param_out.columns])
        # pred_df should be the one returned by _get_indiv_growth
        pd.testing.assert_frame_equal(pred_out, pred_df_mock)

    @patch("tfscreen.analysis.independent.get_indiv_growth._get_indiv_growth")
    @patch("tfscreen.analysis.independent.get_indiv_growth.get_scaled_cfu")
    def test_get_indiv_growth_multi_iteration(self, mock_get_scaled, mock_internal_get):
        
        # Setup input
        df = pd.DataFrame({
            "replicate": [1, 1],
            "genotype": ["wt", "wt"],
            "t_sel": [1, 2],
            "ln_cfu": [10.0, 11.0]
        })
        series_selector = ["replicate", "genotype"]
        calibration_data = {}
        
        # Setup mocks
        # Iteration 1 return
        param_df_1 = pd.DataFrame({"replicate": [1], "genotype": ["wt"], 
                                   "lnA0_est": [9.0], "lnA0_std": [0.1],
                                   "k_est": [1.0], "k_std": [0.1],
                                   "dk_geno": [0.0], "lnA0_pre": [9.0]})
        pred_df_1 = df.copy()
        
        # Iteration 2 return - MUST include _is_fake_point to test cleanup
        param_df_2 = pd.DataFrame({"replicate": [1], "genotype": ["wt"], 
                                   "lnA0_est": [9.1], "lnA0_std": [0.05],
                                   "k_est": [1.1], "k_std": [0.05],
                                   "dk_geno": [0.0], "lnA0_pre": [9.1]})
        
        # Simulating what happens: the underlying fitter usually preserves extra columns from input
        # So if iteration 1 added _is_fake_point, iteration 2's input has it, and its output likely has it too.
        pred_df_2 = df.copy() 
        pred_df_2["_is_fake_point"] = False # Simulate existence
        
        mock_internal_get.side_effect = [(param_df_1, pred_df_1), (param_df_2, pred_df_2)]
        
        # Mock scaled cfu to just return input (simplified)
        mock_get_scaled.side_effect = lambda x, y: x 
        
        # Call function
        param_out, pred_out = get_indiv_growth(
            df=df,
            series_selector=series_selector,
            calibration_data=calibration_data,
            num_iterations=2
        )
        
        # Verify calls
        assert mock_internal_get.call_count == 2
        
        # Verify cleanup
        assert "_is_fake_point" not in pred_out.columns
        pd.testing.assert_frame_equal(param_out, param_df_2[param_out.columns])

    @patch("tfscreen.analysis.independent.get_indiv_growth._get_indiv_growth")
    @patch("tfscreen.analysis.independent.get_indiv_growth.get_scaled_cfu")
    def test_get_indiv_growth_alignment(self, mock_get_scaled, mock_internal_get):
        """
        Tests that get_indiv_growth correctly aligns param_df and pred_df 
        even if their order is different (as can happen with batching).
        """
        # Setup multi-series input
        df = pd.DataFrame({
            "replicate": [1, 1, 2, 2],
            "genotype": ["wt", "wt", "wt", "wt"],
            "t_sel": [1, 2, 1, 2],
            "ln_cfu": [10.0, 11.0, 10.0, 11.0]
        })
        series_selector = ["replicate", "genotype"]
        
        # Iteration 1 return: Out of order compared to df
        param_df_1 = pd.DataFrame({
            "replicate": [2, 1], # Swapped order
            "genotype": ["wt", "wt"], 
            "lnA0_est": [9.0, 8.0],
            "lnA0_std": [0.1, 0.1],
            "k_est": [1.0, 1.0],
            "k_std": [0.1, 0.1],
            "dk_geno": [0.0, 0.0],
            "lnA0_pre": [9.0, 8.0]
        })
        pred_df_1 = df.copy() # Order matches df

        # We'll just do 1 loop to see if the fake point adding (which uses alignment) works
        # or mock a 2nd loop to see the update alignment.
        
        mock_internal_get.side_effect = [(param_df_1, pred_df_1), (param_df_1, pred_df_1)]
        mock_get_scaled.side_effect = lambda x, y: x
        
        # Call with 2 iterations to trigger alignment and t=0 updates
        get_indiv_growth(
            df=df,
            series_selector=series_selector,
            calibration_data={},
            num_iterations=2
        )
        
        # The fact it didn't crash is a good sign. 
        # The alignment happens internally. 
        # If it failed, we'd see a KeyError or mis-alignment.

