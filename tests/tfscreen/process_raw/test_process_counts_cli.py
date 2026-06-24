import pytest
import pandas as pd
from unittest.mock import patch

from tfscreen.process_raw.scripts.process_counts_cli import process_counts


# ------------------------
# Fixture
# ------------------------

@pytest.fixture
def fx_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sample": ["s1", "s2"],
        "sample_cfu": [1e8, 1.6e8],
        "sample_cfu_std": [1e7, 1.6e7],
    })


# ------------------------
# Test process_counts
# ------------------------

@patch('pandas.DataFrame.to_csv')
@patch('tfscreen.process_raw.scripts.process_counts_cli.counts_to_lncfu')
@patch('tfscreen.process_raw.scripts.process_counts_cli._aggregate_counts')
@patch('tfscreen.process_raw.scripts.process_counts_cli._prep_sample_df')
def test_process_counts_orchestration(
    mock_prep, mock_agg, mock_lncfu, mock_to_csv, fx_sample_df
):
    """Verify the overall workflow and data handoffs."""
    df_prepped = fx_sample_df.set_index("sample")
    df_aggregated = pd.DataFrame({"s": ["s1"], "aggregated": [True]})
    df_final = pd.DataFrame({"s": ["s1"], "final_lncfu": [True]})

    mock_prep.return_value = df_prepped
    mock_agg.return_value = df_aggregated
    mock_lncfu.return_value = df_final

    process_counts(
        sample_df=fx_sample_df,
        counts_csv_path="/counts/path",
        output_file="/out/final.csv",
        counts_glob_prefix="test_prefix",
        min_genotype_obs=20,
        pseudocount=2,
        verbose=False
    )

    mock_prep.assert_called_once_with(fx_sample_df, "/counts/path", "test_prefix", False)
    mock_agg.assert_called_once_with(df_prepped)
    mock_lncfu.assert_called_once_with(
        df_prepped, df_aggregated, min_genotype_obs=20, pseudocount=2
    )
    mock_to_csv.assert_called_once_with("/out/final.csv", index=False)


def test_process_counts_missing_cfu_columns(tmp_path):
    """ValueError raised when sample_cfu_std is absent."""
    bad_df = pd.DataFrame({
        "sample": ["s1", "s2"],
        "sample_cfu": [1e8, 1.6e8],
        # sample_cfu_std intentionally omitted
    })
    prepped = bad_df.set_index("sample")
    prepped["obs_file"] = "dummy.csv"

    with patch('tfscreen.process_raw.scripts.process_counts_cli._prep_sample_df', return_value=prepped):
        with pytest.raises(ValueError, match="Not all required columns seen"):
            process_counts(
                sample_df=bad_df,
                counts_csv_path=str(tmp_path),
                output_file="/out/final.csv",
            )
