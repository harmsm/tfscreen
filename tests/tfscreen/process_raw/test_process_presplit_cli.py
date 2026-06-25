import pytest
import numpy as np
import pandas as pd
import pandas.testing as pd_testing
from unittest.mock import patch, MagicMock, ANY

from tfscreen.process_raw.scripts.process_presplit_cli import process_presplit


# ------------------------
# Fixtures
# ------------------------

@pytest.fixture
def fx_sample_df() -> pd.DataFrame:
    """Minimal presplit sample metadata."""
    return pd.DataFrame({
        "sample": ["ps_rep1_kanR", "ps_rep2_kanR"],
        "replicate": [1, 2],
        "condition_pre": ["kanR-cond", "kanR-cond"],
        "sample_cfu": [1e8, 1.2e8],
        "sample_cfu_std": [1e7, 1.2e7],
    })


@pytest.fixture
def fx_lncfu_df() -> pd.DataFrame:
    """Simulated output from counts_to_lncfu, including ln_cfu_var."""
    return pd.DataFrame({
        "sample": ["ps_rep1_kanR", "ps_rep1_kanR", "ps_rep2_kanR", "ps_rep2_kanR"],
        "replicate": [1, 1, 2, 2],
        "condition_pre": ["kanR-cond", "kanR-cond", "kanR-cond", "kanR-cond"],
        "library": ["default"] * 4,
        "genotype": ["wt", "A1V", "wt", "A1V"],
        "ln_cfu": [10.0, 9.5, 10.1, 9.4],
        "ln_cfu_var": [0.01, 0.02, 0.01, 0.02],
    })


# ------------------------
# Tests for process_presplit
# ------------------------

@patch('pandas.DataFrame.to_csv')
@patch('tfscreen.process_raw.scripts.process_presplit_cli.get_scaled_cfu')
@patch('tfscreen.process_raw.scripts.process_presplit_cli.counts_to_lncfu')
@patch('tfscreen.process_raw.scripts.process_presplit_cli._aggregate_counts')
@patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df')
def test_process_presplit_orchestration(
    mock_prep, mock_agg, mock_lncfu, mock_scaled, mock_to_csv,
    fx_sample_df, fx_lncfu_df
):
    """Verify the overall workflow, data handoffs, and column selection."""
    df_prepped = fx_sample_df.set_index("sample")
    df_counts = pd.DataFrame({"sample": ["ps_rep1_kanR"], "genotype": ["wt"], "counts": [100]})
    # get_scaled_cfu adds ln_cfu_std to the df
    df_scaled = fx_lncfu_df.copy()
    df_scaled["ln_cfu_std"] = np.sqrt(df_scaled["ln_cfu_var"])

    mock_prep.return_value = df_prepped
    mock_agg.return_value = df_counts
    mock_lncfu.return_value = fx_lncfu_df
    mock_scaled.return_value = df_scaled

    process_presplit(
        sample_df=fx_sample_df,
        counts_csv_path="/counts/path",
        output_file="/out/presplit.csv",
        counts_glob_prefix="counts",
        min_genotype_obs=5,
        pseudocount=1,
        verbose=False
    )

    mock_prep.assert_called_once_with(fx_sample_df, "/counts/path", "counts", False)
    # _aggregate_counts receives the sample_df with 'library' injected (ANY because
    # the exact df object varies when library is added)
    mock_agg.assert_called_once_with(ANY)
    mock_lncfu.assert_called_once_with(ANY, df_counts,
                                        min_genotype_obs=5, pseudocount=1)
    mock_scaled.assert_called_once()

    # Verify to_csv is called once on the result
    mock_to_csv.assert_called_once_with("/out/presplit.csv", index=False)



def test_process_presplit_output_columns_integration(fx_sample_df, fx_lncfu_df, tmp_path):
    """Integration-style: verify the CSV is written with the right columns."""
    df_prepped = fx_sample_df.set_index("sample")
    df_scaled = fx_lncfu_df.copy()
    df_scaled["ln_cfu_std"] = np.sqrt(df_scaled["ln_cfu_var"])
    output_file = str(tmp_path / "presplit.csv")

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=df_prepped), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli._aggregate_counts',
               return_value=pd.DataFrame()), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.counts_to_lncfu',
               return_value=fx_lncfu_df), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.get_scaled_cfu',
               return_value=df_scaled):

        process_presplit(
            sample_df=fx_sample_df,
            counts_csv_path="/counts/path",
            output_file=output_file,
            verbose=False
        )

    result = pd.read_csv(output_file)
    assert set(result.columns) == {"replicate", "condition_pre", "genotype",
                                    "ln_cfu", "ln_cfu_std"}
    # Extra columns from counts_to_lncfu (library, sample, ln_cfu_var) must be absent
    assert "library" not in result.columns
    assert "ln_cfu_var" not in result.columns
    assert "sample" not in result.columns


def test_process_presplit_output_sorted(fx_sample_df, fx_lncfu_df, tmp_path):
    """Output rows are sorted by (replicate, condition_pre, genotype)."""
    df_prepped = fx_sample_df.set_index("sample")
    # Deliberately shuffle the order in the lncfu df
    df_shuffled = fx_lncfu_df.iloc[::-1].reset_index(drop=True)
    df_scaled = df_shuffled.copy()
    df_scaled["ln_cfu_std"] = np.sqrt(df_scaled["ln_cfu_var"])
    output_file = str(tmp_path / "presplit_sorted.csv")

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=df_prepped), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli._aggregate_counts',
               return_value=pd.DataFrame()), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.counts_to_lncfu',
               return_value=df_shuffled), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.get_scaled_cfu',
               return_value=df_scaled):

        process_presplit(
            sample_df=fx_sample_df,
            counts_csv_path="/counts/path",
            output_file=output_file,
            verbose=False
        )

    result = pd.read_csv(output_file)
    sorted_result = result.sort_values(
        by=["replicate", "condition_pre", "genotype"]
    ).reset_index(drop=True)
    pd_testing.assert_frame_equal(result, sorted_result)


def test_process_presplit_missing_replicate_column(tmp_path, fx_sample_df):
    """ValueError raised when 'replicate' is absent from sample_df."""
    bad_df = fx_sample_df.drop(columns=["replicate"])
    prepped = bad_df.set_index("sample")
    prepped["obs_file"] = "dummy.csv"

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=prepped):
        with pytest.raises(ValueError, match="Not all required columns seen"):
            process_presplit(
                sample_df=bad_df,
                counts_csv_path=str(tmp_path),
                output_file="/out/presplit.csv",
            )


def test_process_presplit_missing_condition_pre_column(tmp_path, fx_sample_df):
    """ValueError raised when 'condition_pre' is absent from sample_df."""
    bad_df = fx_sample_df.drop(columns=["condition_pre"])
    prepped = bad_df.set_index("sample")
    prepped["obs_file"] = "dummy.csv"

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=prepped):
        with pytest.raises(ValueError, match="Not all required columns seen"):
            process_presplit(
                sample_df=bad_df,
                counts_csv_path=str(tmp_path),
                output_file="/out/presplit.csv",
            )


def test_process_presplit_missing_cfu_column(tmp_path, fx_sample_df):
    """ValueError raised when 'sample_cfu_std' is absent from sample_df."""
    bad_df = fx_sample_df.drop(columns=["sample_cfu_std"])
    prepped = bad_df.set_index("sample")
    prepped["obs_file"] = "dummy.csv"

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=prepped):
        with pytest.raises(ValueError, match="Not all required columns seen"):
            process_presplit(
                sample_df=bad_df,
                counts_csv_path=str(tmp_path),
                output_file="/out/presplit.csv",
            )


def test_process_presplit_default_library_added(fx_sample_df, fx_lncfu_df, tmp_path):
    """When sample_df has no 'library' column, 'default' is injected before counts_to_lncfu."""
    df_prepped = fx_sample_df.set_index("sample")
    assert "library" not in df_prepped.columns

    df_scaled = fx_lncfu_df.copy()
    df_scaled["ln_cfu_std"] = np.sqrt(df_scaled["ln_cfu_var"])

    lncfu_calls = []

    def capture_lncfu(sdf, cdf, **kwargs):
        lncfu_calls.append(sdf.copy())
        return fx_lncfu_df

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=df_prepped), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli._aggregate_counts',
               return_value=pd.DataFrame()), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.counts_to_lncfu',
               side_effect=capture_lncfu), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.get_scaled_cfu',
               return_value=df_scaled), \
         patch('pandas.DataFrame.to_csv'):

        process_presplit(
            sample_df=fx_sample_df,
            counts_csv_path="/counts/path",
            output_file=str(tmp_path / "out.csv"),
            verbose=False
        )

    assert len(lncfu_calls) == 1
    assert "library" in lncfu_calls[0].columns
    assert (lncfu_calls[0]["library"] == "default").all()


def test_process_presplit_existing_library_preserved(fx_sample_df, fx_lncfu_df, tmp_path):
    """When sample_df already has a 'library' column, it is not overwritten."""
    df_with_lib = fx_sample_df.copy()
    df_with_lib["library"] = ["lib_A", "lib_B"]
    df_prepped = df_with_lib.set_index("sample")

    df_scaled = fx_lncfu_df.copy()
    df_scaled["ln_cfu_std"] = np.sqrt(df_scaled["ln_cfu_var"])

    lncfu_calls = []

    def capture_lncfu(sdf, cdf, **kwargs):
        lncfu_calls.append(sdf.copy())
        return fx_lncfu_df

    with patch('tfscreen.process_raw.scripts.process_presplit_cli._prep_sample_df',
               return_value=df_prepped), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli._aggregate_counts',
               return_value=pd.DataFrame()), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.counts_to_lncfu',
               side_effect=capture_lncfu), \
         patch('tfscreen.process_raw.scripts.process_presplit_cli.get_scaled_cfu',
               return_value=df_scaled), \
         patch('pandas.DataFrame.to_csv'):

        process_presplit(
            sample_df=df_with_lib,
            counts_csv_path="/counts/path",
            output_file=str(tmp_path / "out.csv"),
            verbose=False
        )

    assert list(lncfu_calls[0]["library"]) == ["lib_A", "lib_B"]
