import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.tensor_manager import TensorManager

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "genotype": ["A", "A", "B", "B"],
        "condition": ["C1", "C2", "C1", "C2"],
        "replicate": [1, 1, 1, 1],
        "value": [1.0, 2.0, 3.0, 4.0],
        "other_val": [10.0, 20.0, 30.0, 40.0]
    })

def test_init(sample_df):
    tm = TensorManager(sample_df)
    assert tm.df is sample_df
    assert tm.map_sizes == {}
    assert tm.tensors is None

def test_add_map_tensor_simple(sample_df):
    tm = TensorManager(sample_df)
    tm.add_map_tensor("genotype", name="geno")
    assert "map_geno" in tm.df.columns
    assert tm.map_sizes["geno"] == 2
    assert "map_geno" in tm._to_tensor_columns

    # Test default name
    tm = TensorManager(sample_df)
    tm.add_map_tensor("genotype")
    assert "map_genotype" in tm.df.columns
    assert tm.map_sizes["genotype"] == 2

    # Test error already added
    with pytest.raises(ValueError, match="map_geno already added"):
        tm = TensorManager(sample_df)
        tm.add_map_tensor("genotype", name="geno")
        tm.add_map_tensor("genotype", name="geno")

def test_add_map_tensor_multi(sample_df):
    tm = TensorManager(sample_df)
    tm.add_map_tensor(select_cols=["genotype"], select_pool_cols=["condition"], name="pool")
    assert "map_condition" in tm.df.columns
    assert tm.map_sizes["pool"] == 4 # A-C1, A-C2, B-C1, B-C2
    assert "map_condition" in tm._to_tensor_columns

    # The current implementation of _add_multi_map_tensor has a bug where it 
    # doesn't add `map_name` to `_to_tensor_columns`, but checks for it.
    # To trigger the error, we need to manually add it or call a simple map first.
    tm.add_map_tensor("genotype", name="pool")
    with pytest.raises(ValueError, match="map_pool already added"):
        tm.add_map_tensor(select_cols=["genotype"], select_pool_cols=["condition"], name="pool")

def test_add_data_tensor(sample_df):
    tm = TensorManager(sample_df)
    tm.add_data_tensor("value")
    assert "value" in tm._to_tensor_columns
    assert tm._tensor_column_dtypes["value"] == jnp.float32

    with pytest.raises(ValueError, match="column 'nonexistent' not found"):
        tm.add_data_tensor("nonexistent")

def test_add_pivot_index(sample_df):
    tm = TensorManager(sample_df)
    tm.add_pivot_index("geno", "genotype")
    assert "geno_idx" in tm.df.columns
    assert "geno_idx" in tm._pivot_index
    assert tm.tensor_dim_names == ["geno"]
    assert list(tm.tensor_dim_labels[0]) == ["A", "B"]

    with pytest.raises(ValueError, match="cat_column 'nonexistent' is invalid"):
        tm.add_pivot_index("bad", "nonexistent")

def test_add_ranked_pivot_index(sample_df):
    tm = TensorManager(sample_df)
    tm.add_ranked_pivot_index("rank", "value", "genotype")
    assert "rank_idx" in tm.df.columns
    # A has values 1.0, 2.0 -> ranks 0, 1
    # B has values 3.0, 4.0 -> ranks 0, 1
    assert list(tm.df["rank_idx"]) == [0, 1, 0, 1]

def test_create_tensors(sample_df):
    tm = TensorManager(sample_df)
    tm.add_pivot_index("geno", "genotype")
    tm.add_pivot_index("cond", "condition")
    tm.add_data_tensor("value")
    tm.add_map_tensor("genotype", name="gm")
    
    tm.create_tensors()
    assert tm.tensor_shape == (2, 2)
    assert "value" in tm.tensors
    assert "map_gm" in tm.tensors
    assert "good_mask" in tm.tensors
    assert tm.tensors["value"].shape == (2, 2)
    assert jnp.issubdtype(tm.tensors["map_gm"].dtype, jnp.integer)

def test_create_tensors_errors(sample_df):
    tm = TensorManager(sample_df)
    with pytest.raises(ValueError, match="pivot indexes must be added"):
        tm.create_tensors()
    
    tm.add_pivot_index("geno", "genotype")
    with pytest.raises(ValueError, match="tensor columns must be specified"):
        tm.create_tensors()

def test_properties(sample_df):
    tm = TensorManager(sample_df)
    tm.add_map_tensor("genotype", name="g")
    assert tm.map_groups["g"] is not None
    assert tm.tensor_dim_names == []
    assert tm.tensor_dim_labels == []
    assert tm.tensor_shape is None

def test_add_multi_map_tensor_name(sample_df):
    # Cover line 150 (name is None in _add_multi_map_tensor)
    tm = TensorManager(sample_df)
    # Use multiple select_cols so name becomes "genotype-replicate", 
    # which doesn't conflict with existing columns.
    tm.add_map_tensor(select_cols=["genotype", "replicate"], select_pool_cols=["condition"])
    assert "map_condition" in tm.df.columns
    assert "genotype-replicate" in tm.map_sizes

def test_add_map_tensor_select_cols_str(sample_df):
    # Cover line 238-240 (select_cols as str)
    tm = TensorManager(sample_df)
    tm.add_map_tensor("genotype")
    assert "map_genotype" in tm.df.columns

def test_add_map_tensor_pool_cols_str(sample_df):
    # Cover line 244-245 (select_pool_cols as str)
    tm = TensorManager(sample_df)
    tm.add_map_tensor(select_cols=["genotype"], select_pool_cols="condition", name="p2")
    assert "map_condition" in tm.df.columns

def test_create_tensors_dtype_default(sample_df):
    # Cover line 483 (default dtype jnp.float32)
    tm = TensorManager(sample_df)
    tm.add_pivot_index("geno", "genotype")
    tm.add_data_tensor("value", dtype=None) # Forces line 483
    tm.create_tensors()
    assert tm.tensors["value"].dtype == jnp.float32
