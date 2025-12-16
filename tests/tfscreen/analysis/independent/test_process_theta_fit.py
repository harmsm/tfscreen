
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from tfscreen.analysis.independent.process_theta_fit import (
    process_theta_fit, 
    _parse_parameter, 
    _parse_theta,
    _parse_ln_cfu0,
    _parse_dk_geno,
    _build_ground_truth,
    _clean_pred_df
)

class TestProcessThetaFitHelpers:
    
    def test_parse_parameter_basic(self):
        # Setup basic params
        param_df = pd.DataFrame({
            "name": ["theta_wt_IPTG_0.1", "theta_mut_IPTG_0.1"],
            "class": ["theta", "theta"],
            "est": [10.0, 5.0],
            "std": [0.1, 0.2]
        })
        
        # Test basic parsing
        res = _parse_parameter(
            param_df=param_df,
            ground_truth_df=None,
            param_class="theta",
            split_cols_to_drop=[0],
            rename_map={1: "genotype", 2: "titrant_name", 3: "titrant_conc"},
            output_prefix="theta",
            sort_keys=["genotype", "titrant_name", "titrant_conc"],
            ground_truth_keys=["genotype", "titrant_name", "titrant_conc"],
            ground_truth_col_name="theta",
            type_conversions={"titrant_conc": float}
        )
        
        assert len(res) == 2
        assert "theta_est" in res.columns
        assert "theta_std" in res.columns
        
        # set_categorical_genotype likely puts wt first. 
        # let's just assert we have both rows we expect
        wt_row = res[res["genotype"] == "wt"].iloc[0]
        mut_row = res[res["genotype"] == "mut"].iloc[0]
        
        assert wt_row["theta_est"] == 10.0
        assert mut_row["theta_est"] == 5.0
        assert mut_row["titrant_conc"] == 0.1

    def test_parse_parameter_empty(self):
        param_df = pd.DataFrame({"class": ["other"], "name": ["foo"], "est": [1], "std": [1]})
        res = _parse_parameter(param_df, None, "theta", [], {}, "t", [], [], "", {})
        assert res.empty

    def test_parse_parameter_ground_truth(self):
        param_df = pd.DataFrame({
            "name": ["theta_wt_IPTG_0.1"],
            "class": ["theta"],
            "est": [10.0],
            "std": [0.1]
        })
        
        gt_df = pd.DataFrame({
            "genotype": ["wt"],
            "titrant_name": ["IPTG"],
            "titrant_conc": [0.1],
            "theta": [10.1]
        })
        
        res = _parse_parameter(
            param_df=param_df,
            ground_truth_df=gt_df,
            param_class="theta",
            split_cols_to_drop=[0],
            rename_map={1: "genotype", 2: "titrant_name", 3: "titrant_conc"},
            output_prefix="theta",
            sort_keys=["genotype", "titrant_name", "titrant_conc"],
            ground_truth_keys=["genotype", "titrant_name", "titrant_conc"],
            ground_truth_col_name="theta",
            type_conversions={"titrant_conc": float}
        )
        
        assert "theta_real" in res.columns
        assert res.iloc[0]["theta_real"] == 10.1
        
    def test_parse_wrappers(self):
         # Just quick smoke tests that wrappers call logic correctly
         param_df = pd.DataFrame({
             "name": ["theta_wt_IPTG_1.0", "lnA0_wt_lib1_rep1", "dk_geno_wt"],
             "class": ["theta", "lnA0", "dk_geno"],
             "est": [1.0, 2.0, 3.0],
             "std": [0.1, 0.2, 0.3]
         })
         
         # Theta
         res = _parse_theta(param_df)
         assert len(res) == 1
         assert res.iloc[0]["genotype"] == "wt"
         
         # LnA0
         res = _parse_ln_cfu0(param_df)
         assert len(res) == 1
         assert res.iloc[0]["replicate"] == "rep1"
         
         # Dk_geno
         res = _parse_dk_geno(param_df)
         assert len(res) == 1
         assert res.iloc[0]["genotype"] == "wt"

    def test_build_ground_truth(self):
        counts = pd.DataFrame({"sample": [1], "ln_cfu_0": [10.0]})
        sample = pd.DataFrame({
            "sample": [1],
            "genotype": ["wt"],
            "replicate": ["1"],
            "titrant_conc": [0.0],
            "cfu": [100] # Should be dropped
        })
        
        res = _build_ground_truth(counts, sample)
        assert len(res) == 1
        assert "ln_cfu_0" in res.columns
        assert "genotype" in res.columns
        assert "cfu" not in res.columns
        assert res["replicate"].dtype == object # str
        assert res["titrant_conc"].dtype == float

    def test_clean_pred_df(self):
        df = pd.DataFrame({
            "genotype": ["wt"],
            "y_obs": [1.0],
            "extra": [2.0]
        })
        res = _clean_pred_df(df)
        assert "genotype" in res.columns
        assert "y_obs" in res.columns
        assert "extra" not in res.columns

class TestProcessThetaFit:
    
    @patch("tfscreen.analysis.independent.process_theta_fit._build_ground_truth")
    @patch("tfscreen.analysis.independent.process_theta_fit._parse_theta")
    @patch("tfscreen.analysis.independent.process_theta_fit._parse_dk_geno")
    @patch("tfscreen.analysis.independent.process_theta_fit._parse_ln_cfu0")
    @patch("tfscreen.analysis.independent.process_theta_fit._clean_pred_df")
    def test_process_theta_fit(self, mock_clean, mock_ln, mock_dk, mock_theta, mock_build):
        
        param_df = MagicMock()
        pred_df = MagicMock()
        counts_df = pd.DataFrame({"col": [1]})
        sample_df = pd.DataFrame({"col": [1]})
        
        # Setup ground truth return
        gt_df = pd.DataFrame({
            "theta": [1],
            "dk_geno": [1],
            "ln_cfu_0": [1] # Note mapping expects specific names
        })
        mock_build.return_value = gt_df
        
        # Wrappers return dummy dfs
        mock_theta.return_value = pd.DataFrame({"t": [1]})
        mock_dk.return_value = pd.DataFrame({"d": [1]})
        mock_ln.return_value = pd.DataFrame({"l": [1]})
        mock_clean.return_value = pd.DataFrame({"p": [1]})
        
        res = process_theta_fit(param_df, pred_df, counts_df, sample_df)
        
        # Verify
        mock_build.assert_called_once_with(counts_df, sample_df)
        mock_theta.assert_called_once()
        # Verify passed GT is correct (pandas filtering makes direct arg comparison slightly hard, but we can check calls)
        assert res["theta"].iloc[0]["t"] == 1
        
        # Check that we passed the ground truth correctly
        _, kwargs = mock_theta.call_args
        # Positional args: param_df, ground_truth_df
        args = mock_theta.call_args[0]
        pd.testing.assert_frame_equal(args[1], gt_df) # Passed the whole GT df? wrapper filters it? Ah wrapper takes specific GT df if split?
        
        # looking at code:
        # ground_truth_dict["theta"] = ground_truth_df (if "theta" in columns)
        # _parse_theta(param_df, ground_truth_dict["theta"])
        
        # So yes, it passes the full dataframe if the column exists.

    def test_process_theta_fit_no_gt(self):
        param_df = MagicMock()
        pred_df = MagicMock()
        
        res = process_theta_fit(param_df, pred_df)
        
        assert res["theta"] is not None 
        # calls with None for gt
        
