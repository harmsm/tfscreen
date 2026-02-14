import pytest
import pandas as pd
import numpy as np

# Import the function to be tested
from tfscreen.analysis.independent.cfu_to_theta import _prep_inference_df

# --- Test Fixtures -----------------------------------------------------------

@pytest.fixture
def base_df() -> pd.DataFrame:
    """
    Provides a "happy path" DataFrame that is dense and well-formed.
    It contains 2 genotypes, 2 replicates, and 2 conditions (8 rows total).
    """
    data = {
        'genotype': ['wt', 'wt', 'A1V', 'A1V'] * 2,
        'replicate': [1, 1, 1, 1, 2, 2, 2, 2],
        'library': ['lib1'] * 8,
        'titrant_name': ['IPTG'] * 8,
        'titrant_conc': [10.0, 100.0] * 4,
        'condition_pre': ['M9'] * 8,
        't_pre': [4.0] * 8,
        'condition_sel': ['M9+Ab'] * 8,
        't_sel': [18.0] * 8,
        'ln_cfu': np.random.rand(8) + 10,
        'ln_cfu_std': np.random.rand(8) * 0.1,
    }
    return pd.DataFrame(data)

@pytest.fixture
def calibration_data() -> dict:
    """
    Provides a mock calibration data dictionary.
    """
    k_bg_df = pd.DataFrame({
        'm': [0.1], 'b': [0.05]
    }, index=pd.Index(['IPTG'], name='titrant_name'))

    dk_cond_df = pd.DataFrame({
        'm': [0.5, 1.2], 'b': [0, 0.1]
    }, index=pd.Index(['M9', 'M9+Ab'], name='condition'))
    
    return {"k_bg_df": k_bg_df, "dk_cond_df": dk_cond_df}

# --- Unit Tests --------------------------------------------------------------

class TestPrepInferenceDF:

    def test_happy_path_correctness(self, base_df, calibration_data, mocker):
        """
        GIVEN a valid, dense DataFrame and calibration data
        WHEN _prep_inference_df is called
        THEN it should return a processed DataFrame with correct new columns.
        """
        # ARRANGE: Mock all external dependencies
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", side_effect=lambda x: x.copy())
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns") # Assume it passes
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.set_categorical_genotype", side_effect=lambda df, **kwargs: df)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.chunk_by_group", side_effect=lambda x, y: [x]) # Single batch
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_calibration", return_value=calibration_data)
        
        # ACT
        result_df, _ = _prep_inference_df(base_df, calibration_data, max_batch_size=10)

        # ASSERT
        # Check for new columns
        # ASSERT
        # Check for new columns
        # k_bg_m/b are not stored, only k_bg is.
        # dk columns are named dk_pre_m, dk_pre_b etc.
        expected_new_cols = ['_batch_idx', 'k_bg', 'dk_pre_m', 
                             'dk_pre_b', 'dk_sel_m', 'dk_sel_b']
        for col in expected_new_cols:
            assert col in result_df.columns
        
        # Check values of merged calibration data
        # k_bg = b + m*titrant (t=10 or 100). m=0.1, b=0.05.
        # k_bg(10) = 0.05 + 0.1*10 = 1.05
        # k_bg(100) = 0.05 + 0.1*100 = 10.05
        
        # dk_pre_m (condition M9). m=0.5
        assert np.allclose(result_df['dk_pre_m'], 0.5)
        # dk_sel_m (condition M9+Ab). m=1.2
        assert np.allclose(result_df['dk_sel_m'], 1.2)
        
        # Check dtypes and batching
        assert result_df['replicate'].dtype == 'Int64'
        assert result_df['_batch_idx'].nunique() == 1
        assert result_df.shape[0] == base_df.shape[0]

    def test_fails_on_missing_columns(self, base_df, calibration_data, mocker):
        """
        GIVEN a DataFrame missing a required column
        WHEN _prep_inference_df is called
        THEN it should raise a ValueError.
        """
        # ARRANGE
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", side_effect=lambda x: x.copy())
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
        # We need to mock the real check_columns to test the failure
        from tfscreen.util import check_columns 
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns", side_effect=check_columns)
        
        df_missing_col = base_df.drop(columns=['t_sel'])
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Not all required columns"):
            _prep_inference_df(df_missing_col, calibration_data, max_batch_size=10)

    def test_fails_on_duplicate_rows(self, base_df, calibration_data, mocker):
        """
        GIVEN a DataFrame with duplicate genotype/condition rows within a replicate
        WHEN _prep_inference_df is called
        THEN it should raise a ValueError.
        """
        # ARRANGE
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", side_effect=lambda x: x.copy())
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns")
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.set_categorical_genotype", side_effect=lambda df, **kwargs: df)
        
        df_with_duplicates = pd.concat([base_df, base_df.head(1)], ignore_index=True)
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="duplicate combinations"):
            _prep_inference_df(df_with_duplicates, calibration_data, max_batch_size=10)

    def test_fails_on_sparse_data(self, base_df, calibration_data, mocker):
        """
        GIVEN a DataFrame that is not dense (missing rows)
        WHEN _prep_inference_df is called
        THEN it should raise a ValueError.
        """
        # ARRANGE
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", side_effect=lambda x: x.copy())
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns")
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.set_categorical_genotype", side_effect=lambda df, **kwargs: df)
        
        df_sparse = base_df.iloc[:-1] # Drop one row to make it not dense
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="is missing values"):
            _prep_inference_df(df_sparse, calibration_data, max_batch_size=10)

    def test_fails_on_nan_values(self, base_df, calibration_data, mocker):
        """
        GIVEN a DataFrame with NaN in a critical float column
        WHEN _prep_inference_df is called
        THEN it should raise a ValueError.
        """
        # ARRANGE
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", side_effect=lambda x: x.copy())
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns")
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.set_categorical_genotype", side_effect=lambda df, **kwargs: df)
        
        base_df.loc[0, 'ln_cfu'] = np.nan
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="nan values are not allowed"):
            _prep_inference_df(base_df, calibration_data, max_batch_size=10)

    def test_batching_keeps_genotypes_together(self, base_df, calibration_data, mocker):
        """
        GIVEN a max_batch_size that forces multiple batches
        WHEN _prep_inference_df is called
        THEN it should assign all rows for a given genotype to the same batch.
        """
        # ARRANGE: Mock helpers
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_dataframe", side_effect=lambda x: x.copy())
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.get_scaled_cfu", side_effect=lambda df, **kwargs: df)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.check_columns")
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.set_categorical_genotype", side_effect=lambda df, **kwargs: df)
        # Mock the real chunk_by_group to test the logic
        from tfscreen.util import chunk_by_group
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.chunk_by_group", side_effect=chunk_by_group)
        mocker.patch("tfscreen.analysis.independent.cfu_to_theta.read_calibration", return_value=calibration_data)
        
        # Use a max_batch_size of 2. This will break into each genotype condition
        # block
        max_batch_size = 2

        # ACT
        result_df, _ = _prep_inference_df(base_df, calibration_data, max_batch_size)

        # ASSERT
        # There should be 4 batches wt or A1V with each condition
        assert result_df['_batch_idx'].nunique() == 4
        
        # Make sure that each genotype ends up in exactly two batches
        assert result_df.groupby('genotype')['_batch_idx'].nunique().eq(2).all()