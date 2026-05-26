import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
from tfscreen.analysis.hierarchical.growth_model.extraction import (
    extract_theta_curves,
    extract_theta_unmeasured,
)

@pytest.fixture
def mock_model():
    """Create a ModelClass instance with minimal mocked internals."""
    model = MagicMock(spec=ModelClass)
    model._theta = "hill"
    
    # Mock TensorManager and its DataFrame
    mock_tm = MagicMock()
    mock_tm.df = pd.DataFrame({
        "genotype": ["wt", "wt", "mut", "mut"],
        "titrant_name": ["iptg", "iptg", "iptg", "iptg"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
        "map_theta_group": [0, 0, 1, 1]
    })
    model.growth_tm = mock_tm
    model.training_tm = mock_tm

    return model

@pytest.fixture
def mock_posteriors():
    """Create a dictionary of mock posterior samples."""
    # 10 samples, 2 theta groups
    num_samples = 10
    num_groups = 2
    return {
        "theta_hill_n": np.ones((num_samples, num_groups)) * 2,
        "theta_log_hill_K": np.ones((num_samples, num_groups)) * -1.0, # log(0.36) approx
        "theta_theta_high": np.ones((num_samples, num_groups)) * 0.9,
        "theta_theta_low": np.ones((num_samples, num_groups)) * 0.1
    }

def test_extract_theta_curves_basic(mock_model, mock_posteriors):
    """Test default behavior using data from growth_tm.df."""
    results = extract_theta_curves(mock_model, mock_posteriors)
    
    # Check output structure
    assert isinstance(results, pd.DataFrame)
    assert "genotype" in results.columns
    assert "titrant_name" in results.columns
    assert "titrant_conc" in results.columns
    assert "median" in results.columns
    
    # Should have unique (genotype, titrant_name, titrant_conc)
    # wt: 0.0, 1.0; mut: 0.0, 1.0 -> 4 rows
    assert len(results) == 4
    
    # Verify values for wt at conc=1.0
    # hill_n=2, log_K=-1.0 (K=0.367), high=0.9, low=0.1
    # occupancy = 1 / (1 + exp(-2 * (log(1.0) - (-1.0)))) = 1 / (1 + exp(-2)) = 1 / (1 + 0.135) = 0.88
    # theta = 0.1 + (0.9 - 0.1) * 0.88 = 0.1 + 0.8 * 0.88 = 0.804
    wt_1 = results[(results["genotype"] == "wt") & (results["titrant_conc"] == 1.0)]
    assert np.allclose(wt_1["median"], 0.804, atol=1e-3)

def test_extract_theta_curves_manual_df(mock_model, mock_posteriors):
    """Test providing a manual titrant DataFrame."""
    manual_df = pd.DataFrame({
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.5, 2.0]
    })
    
    results = extract_theta_curves(mock_model, mock_posteriors, manual_titrant_df=manual_df)
    
    # Should broadcast across 'wt' and 'mut' -> 4 rows
    assert len(results) == 4
    assert set(results["genotype"]) == {"wt", "mut"}
    assert set(results["titrant_conc"]) == {0.5, 2.0}

def test_extract_theta_curves_manual_df_with_genotypes(mock_model, mock_posteriors):
    """Test providing a manual titrant DataFrame with explicit genotypes."""
    manual_df = pd.DataFrame({
        "genotype": ["wt", "mut"],
        "titrant_name": ["iptg", "iptg"],
        "titrant_conc": [0.5, 2.0]
    })
    
    results = extract_theta_curves(mock_model, mock_posteriors, manual_titrant_df=manual_df)
    
    assert len(results) == 2
    assert results.iloc[0]["genotype"] == "wt"
    assert results.iloc[1]["genotype"] == "mut"

def test_extract_theta_curves_wrong_theta_model(mock_model, mock_posteriors):
    """Test that it raises ValueError for a theta model without the extraction interface."""
    mock_model._theta = "simple"
    with pytest.raises(ValueError, match="does not support this interface"):
        extract_theta_curves(mock_model, mock_posteriors)

def test_extract_theta_curves_missing_columns(mock_model, mock_posteriors):
    """Test error handling for missing columns in manual_df."""
    manual_df = pd.DataFrame({"titrant_name": ["iptg"]}) # Missing conc
    with pytest.raises(Exception): # check_columns raises an error
         extract_theta_curves(mock_model, mock_posteriors, manual_titrant_df=manual_df)

def test_extract_theta_curves_invalid_genotype(mock_model, mock_posteriors):
    """Test error handling for genotype not in model."""
    manual_df = pd.DataFrame({
        "genotype": ["non_existent"],
        "titrant_name": ["iptg"],
        "titrant_conc": [1.0]
    })
    with pytest.raises(ValueError, match="were not found in the model data"):
        extract_theta_curves(mock_model, mock_posteriors, manual_titrant_df=manual_df)

def test_extract_theta_curves_file_loading(mock_model):
    """Test loading posteriors from file."""
    mock_model._theta = "hill"
    mock_npz = {
        "theta_hill_n": np.ones((5, 2)),
        "theta_log_hill_K": np.ones((5, 2)),
        "theta_theta_high": np.ones((5, 2)),
        "theta_theta_low": np.ones((5, 2))
    }
    
    with patch("numpy.load", return_value=mock_npz) as mock_load:
        extract_theta_curves(mock_model, "mock.npz")
        mock_load.assert_called_once_with("mock.npz")


# ---------------------------------------------------------------------------
# extract_theta_unmeasured — genotype batching
# ---------------------------------------------------------------------------

def _make_unmeasured_model(theta_name="hill_mut"):
    """Minimal model mock for extract_theta_unmeasured."""
    model = MagicMock(spec=ModelClass)
    model._theta = theta_name
    mock_tm = MagicMock()
    mock_tm.tensor_dim_names = ["titrant_name", "genotype"]
    mock_tm.tensor_dim_labels = [["IPTG"], ["wt", "A1B"]]
    mock_tm.df = pd.DataFrame({
        "genotype": ["wt", "wt", "A1B", "A1B"],
        "titrant_name": ["IPTG", "IPTG", "IPTG", "IPTG"],
        "titrant_conc": [0.0, 1.0, 0.0, 1.0],
    })
    model.training_tm = mock_tm
    model.mut_labels = ["M1A", "K2L"]
    model.pair_labels = []
    mock_priors = MagicMock()
    mock_priors.theta = MagicMock(spec=[])  # no theta_tf_total_M etc.
    model.priors = mock_priors
    return model


def _fake_predict_unmeasured(target_genotypes, titrant_names,
                             manual_titrant_df, mut_labels,
                             pair_labels, param_posteriors, q_to_get):
    """Returns one row per (genotype, titrant_conc) with median=0.5."""
    rows = []
    for g in target_genotypes:
        for _, row in manual_titrant_df.iterrows():
            rows.append({"genotype": g,
                         "titrant_name": row["titrant_name"],
                         "titrant_conc": row["titrant_conc"],
                         "median": 0.5})
    return pd.DataFrame(rows)


@pytest.fixture
def patched_unmeasured_module():
    """Patch the registry so hill_mut has a predict_unmeasured we can spy on."""
    fake_module = MagicMock()
    fake_module.predict_unmeasured = MagicMock(side_effect=_fake_predict_unmeasured)
    registry_patch = {"hill_mut": fake_module}
    with patch(
        "tfscreen.analysis.hierarchical.growth_model.extraction.model_registry",
        {"theta": registry_patch},
    ), patch(
        "tfscreen.analysis.hierarchical.growth_model.extraction.load_posteriors",
        return_value=({"median": 0.5}, {}),
    ):
        yield fake_module


class TestExtractThetaUnmeasuredBatching:

    def _manual_df(self):
        return pd.DataFrame({
            "titrant_name": ["IPTG", "IPTG"],
            "titrant_conc": [0.0, 1.0],
        })

    def test_small_list_single_call(self, patched_unmeasured_module):
        """When N <= batch_size, predict_unmeasured is called exactly once."""
        model = _make_unmeasured_model()
        genotypes = ["wt", "A1B", "C3D"]
        extract_theta_unmeasured(
            model, {}, genotypes, self._manual_df(),
            genotype_batch_size=10,
        )
        assert patched_unmeasured_module.predict_unmeasured.call_count == 1
        called_genos = patched_unmeasured_module.predict_unmeasured.call_args.kwargs[
            "target_genotypes"
        ]
        assert called_genos == genotypes

    def test_large_list_splits_into_batches(self, patched_unmeasured_module):
        """When N > batch_size, predict_unmeasured is called ceil(N/batch) times."""
        model = _make_unmeasured_model()
        genotypes = [f"G{i}" for i in range(7)]
        extract_theta_unmeasured(
            model, {}, genotypes, self._manual_df(),
            genotype_batch_size=3,
        )
        # ceil(7/3) = 3 calls
        assert patched_unmeasured_module.predict_unmeasured.call_count == 3

    def test_batched_result_identical_to_single_call(self, patched_unmeasured_module):
        """Batched output has the same rows as a single-call output, in the same order."""
        model = _make_unmeasured_model()
        genotypes = [f"G{i}" for i in range(5)]
        manual_df = self._manual_df()

        # Single call (batch_size > N)
        result_single = extract_theta_unmeasured(
            model, {}, genotypes, manual_df, genotype_batch_size=100,
        )
        patched_unmeasured_module.predict_unmeasured.reset_mock()

        # Batched call (batch_size < N)
        result_batched = extract_theta_unmeasured(
            model, {}, genotypes, manual_df, genotype_batch_size=2,
        )

        # Same shape and same genotype ordering
        assert len(result_single) == len(result_batched)
        assert list(result_single["genotype"]) == list(result_batched["genotype"])
        assert list(result_single["titrant_conc"]) == list(result_batched["titrant_conc"])

    def test_batch_boundaries_cover_all_genotypes(self, patched_unmeasured_module):
        """Every genotype appears in exactly one batch call."""
        model = _make_unmeasured_model()
        genotypes = [f"G{i}" for i in range(10)]
        extract_theta_unmeasured(
            model, {}, genotypes, self._manual_df(), genotype_batch_size=4,
        )
        seen = []
        for c in patched_unmeasured_module.predict_unmeasured.call_args_list:
            seen.extend(c.kwargs["target_genotypes"])
        assert sorted(seen) == sorted(genotypes)

    def test_missing_predict_unmeasured_raises(self):
        """ValueError when the theta component has no predict_unmeasured."""
        model = _make_unmeasured_model(theta_name="no_such_component")
        with patch(
            "tfscreen.analysis.hierarchical.growth_model.extraction.model_registry",
            {"theta": {}},
        ), patch(
            "tfscreen.analysis.hierarchical.growth_model.extraction.load_posteriors",
            return_value=({"median": 0.5}, {}),
        ):
            with pytest.raises(ValueError, match="predict_unmeasured"):
                extract_theta_unmeasured(
                    model, {}, ["wt"], self._manual_df()
                )
