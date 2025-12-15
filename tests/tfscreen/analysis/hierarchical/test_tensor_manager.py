import pytest
import pandas as pd
import numpy as np
import jax.numpy as jnp
from pandas.api.types import is_categorical_dtype, is_integer_dtype

# Import the class as requested by the user
from tfscreen.analysis.hierarchical import TensorManager

# This fixture provides a consistent, clean DataFrame for each test
@pytest.fixture
def sample_df():
    """
    A full-factorial DataFrame fixture for testing.
    Pivots: replicate (2), treatment (2)
    Ranked Pivot: time_value (2)
    Map: genotype (2)
    Data: ln_cfu, cell_count
    Pooled: g_control, t_control
    """
    data = {
        # Pivot Dims
        'replicate':  ['R1', 'R1', 'R1', 'R1', 'R1', 'R1', 'R1', 'R1',
                       'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2'],
        'treatment':  ['T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2',
                       'T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2'],
        # Map/Data Dims
        'genotype':   ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B',
                       'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        # Rank Dim
        'time_value': [0, 10, 0, 10, 0, 10, 0, 10,
                       0, 10, 0, 10, 0, 10, 0, 10],
        # Pooled Map Cols
        'g_control':  ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B',
                       'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        't_control':  ['T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2',
                       'T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2'],
        # Data Cols
        'ln_cfu':     [1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2,
                       5.1, 5.2, 6.1, 6.2, 7.1, 7.2, 8.1, 8.2],
        'cell_count': [10, 20, 30, 40, 50, 60, 70, 80,
                       90, 100, 110, 120, 130, 140, 150, np.nan], # One NaN
    }
    return pd.DataFrame(data)

@pytest.fixture
def tensor_manager(sample_df):
    """Returns a TensorManager instance initialized with sample_df."""
    return TensorManager(sample_df)


class TestTensorManager:

    def test_init(self, sample_df):
        tm = TensorManager(sample_df)
        assert tm.df is sample_df
        assert tm.map_sizes == {}
        assert tm.tensors is None
        assert tm.tensor_shape is None
        assert tm.tensor_dim_names == []
        assert tm.tensor_dim_labels == []

    # --- Test add_map_tensor ---

    def test_add_simple_map_single_col_str(self, tensor_manager):
        tensor_manager.add_map_tensor(select_cols='genotype')
        
        assert 'map_genotype' in tensor_manager.df.columns
        assert 'map_genotype' in tensor_manager._to_tensor_columns
        assert tensor_manager.map_sizes['genotype'] == 2 # A, B
        assert is_integer_dtype(tensor_manager.df['map_genotype'])
        # Check C-ordering
        assert tensor_manager.df[tensor_manager.df['genotype'] == 'A']['map_genotype'].iloc[0] == 0
        assert tensor_manager.df[tensor_manager.df['genotype'] == 'B']['map_genotype'].iloc[0] == 1

    def test_add_simple_map_multi_col_list(self, tensor_manager):
        name = 'gt_map'
        tensor_manager.add_map_tensor(select_cols=['genotype', 'treatment'], name=name)
        
        map_name = f'map_{name}'
        assert map_name in tensor_manager.df.columns
        assert map_name in tensor_manager._to_tensor_columns
        assert tensor_manager.map_sizes[name] == 4 # (A,T1), (A,T2), (B,T1), (B,T2)
        # Check C-ordering
        df = tensor_manager.df
        assert df[(df['genotype'] == 'A') & (df['treatment'] == 'T1')][map_name].iloc[0] == 0
        assert df[(df['genotype'] == 'A') & (df['treatment'] == 'T2')][map_name].iloc[0] == 1
        assert df[(df['genotype'] == 'B') & (df['treatment'] == 'T1')][map_name].iloc[0] == 2
        assert df[(df['genotype'] == 'B') & (df['treatment'] == 'T2')][map_name].iloc[0] == 3

    def test_add_simple_map_raises_duplicate(self, tensor_manager):
        tensor_manager.add_map_tensor(select_cols='genotype')
        with pytest.raises(ValueError, match="map_genotype already added"):
            tensor_manager.add_map_tensor(select_cols='genotype')

    def test_add_multi_map(self, tensor_manager):
        tensor_manager.add_map_tensor(name="test",
                                      select_cols=['replicate'], 
                                      select_pool_cols=['g_control', 't_control'])
        
        assert 'map_g_control' in tensor_manager.df.columns
        assert 'map_t_control' in tensor_manager.df.columns
        assert 'map_g_control' in tensor_manager._to_tensor_columns
        assert 'map_t_control' in tensor_manager._to_tensor_columns
        
        # R1-A, R1-B, R1-T1, R1-T2, R2-A, R2-B, R2-T1, R2-T2
        # But g_control and t_control are correlated
        # Unique pooled values are A, B, T1, T2. Size = 4
        assert tensor_manager.map_sizes['test'] == 8
        #assert tensor_manager.map_sizes['map_t_control'] == 4

    def test_add_multi_map_with_name(self, tensor_manager):
        tensor_manager.add_map_tensor(select_cols=['replicate'], 
                                      select_pool_cols='g_control',
                                      name='pooled_map')
        # Name should be recorded for overall size
        assert tensor_manager.map_sizes['pooled_map'] == 4 # R1-A, R1-B, R2-A, R2-B

    # --- Test add_data_tensor ---

    def test_add_data_tensor_success(self, tensor_manager):
        tensor_manager.add_data_tensor('ln_cfu')
        assert 'ln_cfu' in tensor_manager._to_tensor_columns
        assert tensor_manager._tensor_column_dtypes['ln_cfu'] == jnp.float32

    def test_add_data_tensor_with_dtype(self, tensor_manager):
        tensor_manager.add_data_tensor('cell_count', dtype=jnp.int32)
        assert 'cell_count' in tensor_manager._to_tensor_columns
        assert tensor_manager._tensor_column_dtypes['cell_count'] == jnp.int32

    def test_add_data_tensor_raises_not_found(self, tensor_manager):
        with pytest.raises(ValueError, match="column 'nonexistent' not found"):
            tensor_manager.add_data_tensor('nonexistent')

    # --- Test add_pivot_index ---

    def test_add_pivot_index_not_categorical(self, tensor_manager):
        tensor_manager.add_pivot_index('rep_dim', 'replicate')
        
        assert 'rep_dim_idx' in tensor_manager.df.columns
        assert isinstance(tensor_manager.df['replicate'].dtype, pd.CategoricalDtype)
        assert is_integer_dtype(tensor_manager.df['rep_dim_idx'])
        assert 'rep_dim_idx' in tensor_manager._pivot_index
        assert 'rep_dim' in tensor_manager._tensor_dim_names
        assert (tensor_manager._tensor_dim_labels[0] == ['R1', 'R2']).all()
        assert (tensor_manager._tensor_dim_codes[0] == [0, 1]).all()

    def test_add_pivot_index_is_categorical_preserves_order(self, tensor_manager):
        # Force a non-alphabetical categorical order
        custom_cats = ['R2', 'R1']
        tensor_manager.df['replicate'] = pd.Categorical(
            tensor_manager.df['replicate'], 
            categories=custom_cats
        )
        tensor_manager.add_pivot_index('rep_dim', 'replicate')

        # Order should be ['R2', 'R1']
        assert (tensor_manager._tensor_dim_labels[0] == custom_cats).all()
        assert (tensor_manager._tensor_dim_codes[0] == [0, 1]).all()
        # R2 should be code 0
        assert tensor_manager.df[tensor_manager.df['replicate'] == 'R2']['rep_dim_idx'].iloc[0] == 0
        # R1 should be code 1
        assert tensor_manager.df[tensor_manager.df['replicate'] == 'R1']['rep_dim_idx'].iloc[0] == 1

    def test_add_pivot_index_raises_invalid_col(self, tensor_manager):
        with pytest.raises(ValueError, match="cat_column .* should be a single column"):
            tensor_manager.add_pivot_index('rep_dim', ['replicate', 'treatment'])
            
        with pytest.raises(ValueError, match="cat_column .* should be a single column"):
            tensor_manager.add_pivot_index('rep_dim', 'nonexistent')

    # --- Test add_ranked_pivot_index ---

    def test_add_ranked_pivot_index(self, tensor_manager):
        select_cols = ['replicate', 'treatment', 'genotype']
        tensor_manager.add_ranked_pivot_index('time_dim', 'time_value', select_cols=select_cols)
        
        assert 'time_dim_idx' in tensor_manager.df.columns
        assert '_cat_time_dim_idx' in tensor_manager.df.columns
        assert 'time_dim_idx' in tensor_manager._pivot_index
        assert 'time_dim' in tensor_manager._tensor_dim_names
        assert (tensor_manager._tensor_dim_labels[0] == [0, 1]).all() # Ranks 0 and 1
        
        # Check ranks. time_value 0 should have rank 0, time_value 10 should have rank 1
        df = tensor_manager.df
        assert df[df['time_value'] == 0]['time_dim_idx'].iloc[0] == 0
        assert df[df['time_value'] == 10]['time_dim_idx'].iloc[0] == 1

    def test_add_ranked_pivot_index_select_str(self, tensor_manager):
        # Just test that str input for select_cols works
        tensor_manager.add_ranked_pivot_index('time_dim', 'time_value', select_cols='replicate')
        assert 'time_dim_idx' in tensor_manager.df.columns

    def test_add_ranked_pivot_index_raises_missing_col(self, tensor_manager):
        # This relies on the imported `check_columns` raising an error.
        # We test both rank_column and select_cols
        with pytest.raises(Exception): # Catching base Exception as we don't know what check_columns raises
            tensor_manager.add_ranked_pivot_index(
                'time_dim',
                'nonexistent_rank', # This column doesn't exist
                select_cols='replicate'
            )
        
        with pytest.raises(Exception):
            tensor_manager.add_ranked_pivot_index(
                'time_dim',
                'time_value',
                select_cols='nonexistent_select' # This column doesn't exist
            )

    # --- Test create_tensors ---

    def test_create_tensors_raises_no_pivot(self, tensor_manager):
        tensor_manager.add_data_tensor('ln_cfu')
        with pytest.raises(ValueError, match="pivot indexes must be added"):
            tensor_manager.create_tensors()

    def test_create_tensors_raises_no_data(self, tensor_manager):
        tensor_manager.add_pivot_index('rep_dim', 'replicate')
        with pytest.raises(ValueError, match="tensor columns must be specified"):
            tensor_manager.create_tensors()

    def test_create_tensors_end_to_end(self, tensor_manager):
        # 1. Setup
        # Pivots: (rep, treat, time) -> (2, 2, 2)
        tensor_manager.add_pivot_index('rep_dim', 'replicate') # R1, R2
        tensor_manager.add_pivot_index('treat_dim', 'treatment') # T1, T2
        tensor_manager.add_ranked_pivot_index(
            'time_dim', 
            'time_value', 
            select_cols=['replicate', 'treatment', 'genotype']
        ) # 0, 10 -> 0, 1
        
        # Maps
        tensor_manager.add_map_tensor('genotype') # A, B -> 0, 1
        
        # Data
        tensor_manager.add_data_tensor('ln_cfu') # default float32
        tensor_manager.add_data_tensor('cell_count', dtype=jnp.int16) # custom int16
        
        # 2. Run
        tensor_manager.create_tensors()
        
        # 3. Check
        expected_shape = (2, 2, 2)
        assert tensor_manager.tensor_shape == expected_shape
        assert tensor_manager.tensors is not None
        
        # Check all tensors exist and have correct shape
        expected_tensors = ['good_mask', 'map_genotype', 'ln_cfu', 'cell_count']
        for name in expected_tensors:
            assert name in tensor_manager.tensors
            assert tensor_manager.tensors[name].shape == expected_shape
            
        # Check dtypes
        assert jnp.issubdtype(tensor_manager.tensors['map_genotype'].dtype, jnp.integer)
        assert tensor_manager.tensors['ln_cfu'].dtype == jnp.float32
        assert tensor_manager.tensors['cell_count'].dtype == jnp.int16
        assert tensor_manager.tensors['good_mask'].dtype == bool

        # Check NaN handling (and aggregation)
        # The NaN was at (R2, T2, B, time=10), but it gets aggregated
        # with (R2, T2, A, time=10).
        # The index is (rep=1, treat=1, time=1)
        idx = (1, 1, 1)
        
        # The pivot aggregation (mean) of [150, np.nan] is 150.0, which is NOT NaN.
        # Therefore, the good_mask should be True.
        assert tensor_manager.tensors['good_mask'][idx] == True
        
        # Check data tensors at the aggregated index
        # cell_count: mean(150, nan) = 150. Cast to int16 is 150.
        assert tensor_manager.tensors['cell_count'][idx] == 140
        
        # ln_cfu: mean(7.2, 8.2) = 7.7
        assert jnp.isclose(tensor_manager.tensors['ln_cfu'][idx], 7.7)
        
        # map_genotype: mean(0, 1) = 0.5. Cast to int is 0.
        assert tensor_manager.tensors['map_genotype'][idx] == 0
        # -----------
        
        # Check a non-NaN, non-aggregated index
        # (rep=0, treat=0, time=0) -> R1, T1, A, time=0
        # Note: This is also aggregated (R1, T1, A, 0) and (R1, T1, B, 0)
        idx_good = (0, 0, 0)
        assert tensor_manager.tensors['good_mask'][idx_good] == True
        
        # ln_cfu: mean(1.1, 2.1) = 1.6
        assert jnp.isclose(tensor_manager.tensors['ln_cfu'][idx_good], 1.6)
        
        # cell_count: mean(10, 30) = 20
        assert tensor_manager.tensors['cell_count'][idx_good] == 20
        
        # map_genotype: mean(0, 1) = 0.5. Cast to int is 0.
        assert tensor_manager.tensors['map_genotype'][idx_good] == 0
        
    # --- Test Properties ---

    def test_properties(self, tensor_manager, sample_df):
        assert tensor_manager.df is sample_df
        
        tensor_manager._map_sizes = {'test': 1}
        assert tensor_manager.map_sizes == {'test': 1}
        
        tensor_manager._tensors = {'test': jnp.array([1])}
        assert tensor_manager.tensors == {'test': jnp.array([1])}
        
        tensor_manager._tensor_shape = (1,)
        assert tensor_manager.tensor_shape == (1,)
        
        tensor_manager._tensor_dim_names = ['test_dim']
        assert tensor_manager.tensor_dim_names == ['test_dim']
        
        tensor_manager._tensor_dim_labels = [['A']]
        assert tensor_manager.tensor_dim_labels == [['A']]

    def test_create_tensors_sparse_data_filling(self, tensor_manager):
        """
        Tests that missing combinations in the input DF result in:
        1. good_mask = False
        2. data tensors filled with 1.0 (to avoid NaNs in JAX)
        """
        # Create a sparse dataframe: R1 exists, R2 is MISSING entirely
        sparse_df = pd.DataFrame({
            'replicate': ['R1'], 
            'treatment': ['T1'],
            'genotype': ['A'],
            'ln_cfu': [5.0]
        })
        
        tm = TensorManager(sparse_df)
        
        # We define pivot dimensions that imply R2 should exist (by adding it as a category)
        tm.df['replicate'] = pd.Categorical(tm.df['replicate'], categories=['R1', 'R2'])
        tm.add_pivot_index('rep_dim', 'replicate') # Codes: 0 (R1), 1 (R2)
        
        # Other pivots
        tm.add_pivot_index('treat_dim', 'treatment')
        tm.add_map_tensor('genotype')
        tm.add_data_tensor('ln_cfu')
        
        tm.create_tensors()
        
        # Shape should be (2, 1) because R1 and R2 are categories
        assert tm.tensor_shape == (2, 1)
        
        # Index 0 (R1) should be good
        assert tm.tensors['good_mask'][0, 0] == True
        assert tm.tensors['ln_cfu'][0, 0] == 5.0
        
        # Index 1 (R2) was missing from data -> Generated by reindex
        # Should be masked False
        assert tm.tensors['good_mask'][1, 0] == False
        # Should be filled with 1.0 (NOT NaN)
        assert tm.tensors['ln_cfu'][1, 0] == 1.0