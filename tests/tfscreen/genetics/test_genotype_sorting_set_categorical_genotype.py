import pytest
import pandas as pd
import numpy as np

# Import the function to be tested
from tfscreen.genetics.genotype_sorting import set_categorical_genotype

@pytest.fixture
def mock_standardize_genotypes(mocker):
    """Mocks the standardize_genotypes function to return a fixed list."""
    return mocker.patch("tfscreen.genetics.genotype_sorting.standardize_genotypes")

@pytest.fixture
def mock_argsort_genotypes(mocker):
    """Mocks the argsort_genotypes function to return a fixed array of indices."""
    return mocker.patch("tfscreen.genetics.genotype_sorting.argsort_genotypes")

# ----------------------------------------------------------------------------------
# Scenario 1: standardize=False (Default Behavior)
# ----------------------------------------------------------------------------------

def test_no_standardize_no_sort(mock_argsort_genotypes, mock_standardize_genotypes):
    """
    Test case: `standardize=False`, `sort=False`.
    Should create categories from original names and preserve row order.
    """
    # Arrange: Create test data and configure mocks
    data = {'genotype': ['B10A/A5G', 'A5G', 'wt'], 'value': [3, 2, 1]}
    df = pd.DataFrame(data)
    
    # The unique genotypes are ['B10A/A5G', 'A5G', 'wt'].
    # We will mock argsort to return the indices that would sort them to
    # ['wt', 'A5G', 'B10A/A5G'].
    mock_argsort_genotypes.return_value = np.array([2, 1, 0])
    
    # Act: Run the function
    result_df = set_categorical_genotype(df, standardize=False, sort=False)

    # Assert: Verify the results
    mock_standardize_genotypes.assert_not_called()
    mock_argsort_genotypes.assert_called_once()
    
    # Check that category order is correct
    expected_order = ['wt', 'A5G', 'B10A/A5G']
    assert list(result_df['genotype'].cat.categories) == expected_order

    # Check that original row order and values are preserved
    assert np.array_equal(df["value"],result_df["value"])




def test_no_standardize_with_sort(mock_argsort_genotypes, mock_standardize_genotypes):
    """
    Test case: `standardize=False`, `sort=True`.
    Should create categories from original names and then sort the DataFrame.
    """
    # Arrange
    data = {'genotype': ['B10A/A5G', 'A5G', 'wt'], 'value': [3, 2, 1]}
    df = pd.DataFrame(data)
    mock_argsort_genotypes.return_value = np.array([2, 1, 0])
    
    # Act
    result_df = set_categorical_genotype(df, standardize=False, sort=True)

    # Assert
    mock_standardize_genotypes.assert_not_called()
    
    # Check that the rows are now sorted by the new categorical order
    expected_genotypes = ['wt', 'A5G', 'B10A/A5G']
    assert result_df['genotype'].tolist() == expected_genotypes
    assert result_df['value'].tolist() == [1, 2, 3]

# ----------------------------------------------------------------------------------
# Scenario 2: standardize=True
# ----------------------------------------------------------------------------------

def test_standardize_no_sort(mock_argsort_genotypes, mock_standardize_genotypes):
    """
    Test case: `standardize=True`, `sort=False`.
    Should replace genotypes with standardized names but preserve row order.
    """
    # Arrange
    data = {'genotype': ['B10A/A5G', 'C30C', 'A5G'], 'value': [3, 1, 2]}
    df = pd.DataFrame(data)
    
    # Mock `standardize_genotypes` to return a new list of names
    standardized_names = ["A5G/B10A", "wt", "A5G"]
    mock_standardize_genotypes.return_value = standardized_names
    
    # After standardization, unique genotypes will be ['A5G/B10A', 'wt', 'A5G'].
    # We mock argsort to sort them to ['wt', 'A5G', 'A5G/B10A'].
    mock_argsort_genotypes.return_value = np.array([1, 2, 0])
    
    # Act
    result_df = set_categorical_genotype(df, standardize=True, sort=False)
    
    # Assert
    mock_standardize_genotypes.assert_called_once()
    
    # Check that genotype values were replaced, but row order was preserved
    assert result_df['genotype'].tolist() == standardized_names
    assert result_df['value'].tolist() == [3, 1, 2]
    
    # Check that the categories are based on the *standardized* names
    expected_order = ['wt', 'A5G', 'A5G/B10A']
    assert list(result_df['genotype'].cat.categories) == expected_order


def test_standardize_with_sort(mock_argsort_genotypes, mock_standardize_genotypes):
    """
    Test case: `standardize=True`, `sort=True`.
    Should replace genotypes with standardized names and then sort the DataFrame.
    """
    # Arrange
    data = {'genotype': ['B10A/A5G', 'C30C', 'A5G'], 'value': [3, 1, 2]}
    df = pd.DataFrame(data)
    mock_standardize_genotypes.return_value = ["A5G/B10A", "wt", "A5G"]
    mock_argsort_genotypes.return_value = np.array([1, 2, 0])
    
    # Act
    result_df = set_categorical_genotype(df, standardize=True, sort=True)

    # Assert
    # Check that the DataFrame is now sorted by the standardized names
    assert result_df['genotype'].tolist() == ['wt', 'A5G', 'A5G/B10A']
    assert result_df['value'].tolist() == [1, 2, 3]

# ----------------------------------------------------------------------------------
# Edge Cases
# ----------------------------------------------------------------------------------

def test_no_genotype_column():
    """Tests that the function works correctly if 'genotype' column is missing."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    result_df = set_categorical_genotype(df)
    pd.testing.assert_frame_equal(df, result_df)
    assert id(df) != id(result_df)


def test_empty_dataframe():
    """Tests that the function handles an empty DataFrame."""
    df = pd.DataFrame({'genotype': [], 'value': []})
    result_df = set_categorical_genotype(df, sort=True)
    assert result_df.empty