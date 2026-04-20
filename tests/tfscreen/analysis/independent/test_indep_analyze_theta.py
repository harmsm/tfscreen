
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from tfscreen.analysis.independent.analyze_theta import analyze_theta, main

class TestAnalyzeTheta:

    @patch("tfscreen.analysis.independent.analyze_theta.cfu_to_theta")
    @patch("tfscreen.analysis.independent.analyze_theta.process_theta_fit")
    def test_analyze_theta_basic(self, mock_process, mock_cfu_to_theta):
        
        # Setup mocks
        df = MagicMock(spec=pd.DataFrame)
        calibration_data = {"some": "data"}
        
        param_df_mock = MagicMock(spec=pd.DataFrame)
        pred_df_mock = MagicMock(spec=pd.DataFrame)
        mock_cfu_to_theta.return_value = (param_df_mock, pred_df_mock)
        
        # Create mocks for the output components of process_theta_fit
        out_mocks = {
            "theta": MagicMock(spec=pd.DataFrame),
            "dk_geno": MagicMock(spec=pd.DataFrame),
            "ln_cfu0": MagicMock(spec=pd.DataFrame),
            "pred": MagicMock(spec=pd.DataFrame)
        }
        mock_process.return_value = out_mocks
        
        # Call function
        analyze_theta(
            df=df,
            calibration_data=calibration_data,
            non_sel_conditions="test_cond",
            out_root="test_root",
            max_batch_size=100
        )
        
        # Verify calls
        mock_cfu_to_theta.assert_called_once_with(
            df=df,
            non_sel_conditions="test_cond",
            calibration_data=calibration_data,
            max_batch_size=100,
            logistic_theta=False,
            model_name=None,
            transition_model_name=None
        )
        
        mock_process.assert_called_once_with(param_df_mock, pred_df_mock)
        
        # Verify file writes
        param_df_mock.to_csv.assert_called_once_with("test_root_param.csv", index=False)
        pred_df_mock.to_csv.assert_called_once_with("test_root_pred.csv", index=False)
        
        out_mocks["theta"].to_csv.assert_called_once_with("test_root_theta.csv", index=False)
        out_mocks["dk_geno"].to_csv.assert_called_once_with("test_root_dk_geno.csv", index=False)
        out_mocks["ln_cfu0"].to_csv.assert_called_once_with("test_root_ln_cfu0.csv", index=False)
        out_mocks["pred"].to_csv.assert_called_once_with("test_root_pred_proc.csv", index=False)

    @patch("tfscreen.analysis.independent.analyze_theta.cfu_to_theta")
    @patch("tfscreen.analysis.independent.analyze_theta.process_theta_fit")
    def test_analyze_theta_logistic(self, mock_process, mock_cfu_to_theta):
        
        # Setup mocks
        df = MagicMock(spec=pd.DataFrame)
        calibration_data = {"some": "data"}
        
        param_df_mock = MagicMock(spec=pd.DataFrame)
        pred_df_mock = MagicMock(spec=pd.DataFrame)
        mock_cfu_to_theta.return_value = (param_df_mock, pred_df_mock)
        
        # Create mocks for the output components of process_theta_fit
        out_mocks = {
            "theta": MagicMock(spec=pd.DataFrame),
            "dk_geno": MagicMock(spec=pd.DataFrame),
            "ln_cfu0": MagicMock(spec=pd.DataFrame),
            "pred": MagicMock(spec=pd.DataFrame)
        }
        mock_process.return_value = out_mocks
        
        # Call function with logistic_theta=True
        analyze_theta(
            df=df,
            calibration_data=calibration_data,
            logistic_theta=True
        )
        
        # Verify call to cfu_to_theta has logistic_theta=True
        mock_cfu_to_theta.assert_called_once_with(
            df=df,
            non_sel_conditions=None,
            calibration_data=calibration_data,
            max_batch_size=250,
            logistic_theta=True,
            model_name=None,
            transition_model_name=None
        )

    @patch("tfscreen.analysis.independent.analyze_theta.generalized_main")
    def test_main(self, mock_generalized_main):
        main()
        mock_generalized_main.assert_called_once_with(
            analyze_theta,
            manual_arg_nargs={"non_sel_conditions": "+"},
            manual_arg_types={"non_sel_conditions": str}
        )
