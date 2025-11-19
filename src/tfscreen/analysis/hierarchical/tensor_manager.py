
from tfscreen.util import (
    check_columns,
)

import jax.numpy as jnp

import numpy as np
import pandas as pd

import copy

class TensorManager:
    """
    Manages the wrangling of experimental data from a DataFrame into
    JAX-compatible tensors.

    This class handles data preprocessing and pivoting to create
    dense, multi-dimensional tensors required for hierarchical models in
    JAX/Numpyro. It creates integer-based parameter maps based on column
    values and builds a final dictionary of tensors.

    Attributes
    ----------
    df : pd.DataFrame
        The fully preprocessed pandas DataFrame.
    map_sizes : dict
        A dictionary holding the sizes (max index + 1) of the generated
        parameter maps.
    tensors : dict
        A dictionary of the final, dense JAX tensors (e.g., "ln_cfu",
        "good_mask"). This is populated after create_tensors() is called.
    tensor_shape : tuple
        The shape of the generated tensors. Populated after
        create_tensors() is called.
    tensor_dim_names : list
        A list of names for each tensor dimension.
    tensor_dim_labels : list
        A list of pandas.CategoricalIndex objects, where each provides
        the labels for a tensor dimension.

    Notes
    -----
    The output tensor dimensions are determined by each call to
    `add_pivot_index` or `add_ranked_pivot_index`. Each call will result in a
    new pivot index, and thus a new tensor dimension.

    The `tensors` dict will always include the "good_mask" tensor, which is a
    bool mask that is only True if an element is non-NaN across all tensors.
    """

    def __init__(self,df): 
        """
        Initializes the TensorManager.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be wrangled. This
            class assumes the DataFrame is already loaded.
        """

        self._df = df
        self._map_sizes = {}
        self._map_groups = {}

        self._to_tensor_columns = []
        self._tensor_column_dtypes = {}
        
        # Empty tensor data for now
        self._tensors = None
        self._tensor_shape = None
        self._pivot_index = []

        self._tensor_dim_names = []
        self._tensor_dim_labels = []
        self._tensor_dim_codes = []


    def _add_simple_map_tensor(self,
                               select_cols,
                               name=None):
        """
        Add a new parameter map to the tensor. Create c-order indexes for 
        unique combinations of select_cols.
        """

        if name is None:
            name = "-".join(select_cols)

        # Group on selected columns
        map_name = f"map_{name}"

        if map_name in self._to_tensor_columns:
            raise ValueError(
                f"{map_name} already added"
            )

        # Get all unique combinations of the group columns, then sort and 
        # extract indexes. This creates a map with C-order/row-major sorting.
        unique_groups = self._df[select_cols].drop_duplicates().copy()
        sorted_groups = unique_groups.sort_values(by=select_cols).reset_index(drop=True)
        sorted_groups[map_name] = sorted_groups.index    

        # Record the parameter mapping 
        self._map_groups[name] = sorted_groups
        self._map_sizes[name] = len(sorted_groups)

        # Merge the map back onto the target dataframe.
        self._df = self._df.merge(sorted_groups, 
                                  on=select_cols, 
                                  how="left",
                                  sort=False)
        
        # Record that this should be a new tensor
        self._to_tensor_columns.append(map_name)

    def _add_multi_map_tensor(self,
                              select_cols,
                              select_pool_cols,
                              name=None):
        """
        Add a new parameter map to the tensor. Harder case that requires pooling
        across select_pool_cols. Create c-order indexes for unique combinations
        of select_cols + select_pool_cols[0] OR select_cols + select_pool_cols[1]
        ... 
        """
        
        if name is None:
            name = "-".join(select_cols)

        # Group on selected columns
        map_name = f"map_{name}"

        if map_name in self._to_tensor_columns:
            raise ValueError(
                f"{map_name} already added"
            )

        # Build a pooled dataframe that has only columns in select_cols plus a
        # new column `pooled_col_name`. The pooled column has the values taken 
        # from each select_pool_col stacked into rows.
        to_concat = []
        for c in select_pool_cols:
            sel_on = copy.copy(select_cols)
            sel_on.append(c)
            new_df = self._df[sel_on].rename(columns={c: name})
            to_concat.append(new_df)

        # Build list of columns for sorting
        sort_by = copy.copy(select_cols)
        sort_by.append(name)

        # Get unique groups, sort, and record index
        unique_groups = pd.concat(to_concat).drop_duplicates().copy()
        sorted_groups = unique_groups.sort_values(by=sort_by).reset_index(drop=True)
        sorted_groups[map_name] = sorted_groups.index

        # Record the parameter mapping 
        self._map_groups[name] = sorted_groups
        self._map_sizes[name] = len(sorted_groups)

        # Now go back through the select_pool_cols and merge the newly 
        # constructed parameter map back to each of those columns. 
        for c in select_pool_cols:
            
            left_on = copy.copy(select_cols)
            left_on.append(c)

            right_on = copy.copy(select_cols)
            right_on.append(name)

            self._df = self._df.merge(
                sorted_groups,
                left_on=left_on,
                right_on=right_on,
                how="left"
            ).rename(columns={map_name: f"map_{c}"}).drop(columns=name)

            # Build a new tensor from this column. 
            self._to_tensor_columns.append(f"map_{c}")


    def add_map_tensor(self,
                       select_cols,
                       select_pool_cols=None,
                       name=None):
        """
        Creates an integer map for unique values in DataFrame column(s).

        Groups the DataFrame by `select_cols` and creates a new column
        `map_{name}` containing a unique, 0-indexed integer for
        each group. The map is created by finding all unique combinations of
        `select_cols`, sorting them, and then assigning a C-order index.
        It stores the total count in `self.map_sizes`. The
        resulting `map_{name}` column is added to `self._to_tensor_columns`.

        Parameters
        ----------
        select_cols : list of str or str
            The DataFrame columns to group by to create the map.
        select_pool_cols : list of str or str, optional
            These DataFrame columns are "pooled" with `select_cols`. The
            function finds unique rows in (select_cols + select_pool_cols[0])
            OR (select_cols + select_pool_cols[1]) OR ...
            (select_cols + select_pool_cols[n]). This will create a new map for
            *each* column in `select_pool_cols` (e.g., `map_col1`, `map_col2`).
        name : str, optional
            The base name for the map (e.g. `map_{name}`). If not specified, the
            name will be "-".join(select_cols). If used in conjunction with
            `select_pool_cols`, this `name` is also used as a key in
            `self.map_sizes` to store the total count of all unique
            pooled entries.
        """

        # Deal with select_cols. Ensure it is a list of columns rather than a 
        # single value
        if isinstance(select_cols,str):
            select_cols = [select_cols]
        select_cols = list(select_cols)

        # Deal with pool columns
        if select_pool_cols is not None:
            if isinstance(select_pool_cols,str):
                select_pool_cols = [select_pool_cols]

        if select_pool_cols is None:
            self._add_simple_map_tensor(name=name,
                                        select_cols=select_cols)

        else:
            self._add_multi_map_tensor(name=name,
                                       select_cols=select_cols,
                                       select_pool_cols=select_pool_cols)

    def add_data_tensor(self,name,dtype=jnp.float32):
        """
        Registers a DataFrame column to be included in the final tensors.

        The values in the column should be coercible to floats. The column
        name will be used as the key in the `self.tensors` dictionary.

        Parameters
        ----------
        name : str
            The name of the column in `self._df` to be converted into a tensor.
        dtype : type, optional
            Cast to this type. Default is jnp.float32
        """

        if name not in self._df.columns:
            raise ValueError(
                f"column '{name}' not found in dataframe"
            )

        self._to_tensor_columns.append(name)

        if dtype is not None:
            self._tensor_column_dtypes[name] = dtype


    def _register_pivot_dimension(self, tensor_dim_name, new_index_column, cat_column):
        """
        Private helper to register a new tensor dimension.
        """
        
        # Record the new index column
        self._df[new_index_column] = self._df[cat_column].cat.codes

        # Record that we are going to pivot on this to make the tensor
        self._pivot_index.append(new_index_column)

        # Record the identities of the name of each dimension (dim_names), 
        # labels for each element along the dimension (dim_labels), and their
        # integer codes (dim_codes).
        cats = self._df[cat_column].cat.categories
        self._tensor_dim_names.append(tensor_dim_name)
        self._tensor_dim_labels.append(cats)
        self._tensor_dim_codes.append(np.arange(cats.size,dtype=int))

    def add_pivot_index(self,
                        tensor_dim_name,
                        cat_column):
        """
        Create a new tensor column/pivot index that is based on a categorical 
        variable in `cat_column`. If that column is not already categorical, 
        it will be coerced into a categorical value.

        Populates self._df with a new column (name_idx) containing the category
        codes for the row. It also appends to self._pivot_index,
        self._tensor_dim_names self._tensor_dim_labels, and self._tensor_dim_codes, 
        to set up the final tensor creation operation. 
        
        Parameters
        ----------
        tensor_dim_name : str
            name of the new tensor dimension
        cat_column : str
            column to use for categorization
        """
        
        # Will create this column 
        new_index_column = f"{tensor_dim_name}_idx"

        if not isinstance(cat_column,str) or cat_column not in self._df.columns:
            raise ValueError(
                f"cat_column '{cat_column}' is invalid. It should be a single column in the dataframe."
            )

        # Do a categorical cast *unless* already done. This lets the user 
        # pre-specify some categories with complexity (e.g. genotype) but leaves
        # the rest simple.

        if not isinstance(self._df[cat_column].dtype, pd.CategoricalDtype):
            self._df[cat_column] = pd.Categorical(self._df[cat_column])

        # Call helper to register the dimension
        self._register_pivot_dimension(tensor_dim_name,
                                       new_index_column,
                                       cat_column)

    def add_ranked_pivot_index(self,
                               tensor_dim_name,
                               rank_column,
                               select_cols):
        """
        Create a new tensor column/pivot index that is based on the rank-order
        of the values in rank_column. This is helpful for things like 
        time-series. We want to build a tensor with a dimension that is
        num_time_points long, in order of the time points. Our categories need
        to be defined by the time point number (0,1,2,...), not the time point
        value (20 min, 27.3 s, etc.) that may differ between samples.  

        Populates self._df with a new column (name_idx) containing the category
        codes for the row. It also appends to self._pivot_index,
        self._tensor_dim_names self._tensor_dim_labels, and self._tensor_dim_codes, 
        to set up the final tensor creation operation. 
        
        Parameters
        ----------
        tensor_dim_name : str
            name of the new tensor dimension
        rank_column : str
            column to use for ranking (e.g. time)
        select_cols : str or list of str
            columns to use to select unique values of rank column.
        """
        
        # Make sure select_cols is a list
        if isinstance(select_cols,str):
            select_cols = [select_cols]

        # Will make this column
        new_index_column = f"{tensor_dim_name}_idx"

        # Make sure the all requested columns exist
        all_columns = []
        all_columns.append(rank_column)
        all_columns.extend(select_cols)
        check_columns(self._df, required_columns=all_columns)

        # Create a new, 'hidden/private' category column for this aggregated
        # value.
        cat_column = f"_cat_{new_index_column}"

        # The categorical column is a the rank of each `rank_column` value 
        # within each condition defined by select_cols. This is in sorted order
        self._df[cat_column] = (self._df
                                .groupby(select_cols,observed=False)[rank_column]
                                .rank(method='first')
                                .astype(int) - 1)
        self._df[cat_column] = pd.Categorical(self._df[cat_column])

        # Call helper to register the dimension
        self._register_pivot_dimension(tensor_dim_name, new_index_column, cat_column)
        
    def create_tensors(self):
        """
        Converts the DataFrame into a dictionary of JAX tensors.
        
        This pivots the tidy DataFrame into a dense, multi-dimensional structure, 
        keeping track of columns the user asked to grab. Each of those columns
        becomes its own named tensor. It uses `pandas.pivot_table` to reshape
        the data from a long format to a wide, dense format matching the tensor
        dimensions. It builds an exhaustive `MultiIndex` to ensure all tensor
        elements are present, filling missing data with NaN. It pivots on the
        pre-computed integer codes for each dimension (e.g., `_tensor_dim_codes`)
        to ensure the final tensor axes are correctly aligned with the parameter
        maps.

        This method populates `self._tensors` and `self._tensor_shape`. It also
        generates the tensor `good_mask` which records whether each element is
        non-NaN across all tensors.   
        """

        if len(self._tensor_dim_codes) == 0:
            raise ValueError(
                "pivot indexes must be added using add_pivot_index before generating tensors."
            )
        
        if len(self._to_tensor_columns) == 0:
            raise ValueError(
                "tensor columns must be specified prior to generating tensors."
            )

        exhaustive_index = pd.MultiIndex.from_product(self._tensor_dim_codes, 
                                                      names=self._pivot_index)

        # Use pivot_table to reshape data.
        pivoted_df = pd.pivot_table(
            self._df,
            values=self._to_tensor_columns,
            index=self._pivot_index,
            observed=False
        )

        # Reindex, forcing the complete set of indices
        pivoted_df = pivoted_df.reindex(exhaustive_index)

        # Get the final tensor shape
        self._tensor_shape = tuple(len(lab) for lab in self._tensor_dim_labels)
        
        # We're going to put tensors here. 
        self._tensors = {}

        # Build mask indicating good values. This will have the shape of the
        # rest of the tensors with False for nan values created during the
        # pivot. Look for a row with **any** nan. This corresponds to a tensor
        # coordinate that is a nan in at least one of the tensor. 
        pivot_non_nan = ~np.any(pivoted_df[self._to_tensor_columns]
                                .isna().to_numpy(),axis=1)
        self._tensors["good_mask"] = jnp.asarray(
            pivot_non_nan.reshape(self._tensor_shape)
        )

        # df cleanup -- remove nan prior to building tensors. We can set them 
        # to whatever we want as long as we apply good_mask when doing the 
        # analysis. Has to be above zero because jax/numpyro demands non-zero
        # values when it checks std on final obs/Normal samples -- even with 
        # the mask
        pivoted_df = pivoted_df.fillna(1)

        # Convert to tensors, reshaping and casting as needed
        for c in self._to_tensor_columns:

            # Figure out the datatype we should cast to. Unless manually 
            # defined, set `map_` to int and everything else to float32. 
            if c in self._tensor_column_dtypes:
                dtype = self._tensor_column_dtypes[c]
            elif c.startswith("map_"):
                dtype = int
            else:
                dtype = jnp.float32

            # Resize, cast, and convert to jnumpy array
            values = pivoted_df[c].to_numpy().reshape(self._tensor_shape)
            tensor = jnp.asarray(values,dtype=dtype)

            # Record the tensor
            self._tensors[c] = tensor


    @property
    def df(self):
        """The final, preprocessed pandas DataFrame."""
        return self._df

    @property
    def map_sizes(self):
        """A dictionary holding the sizes of the maps."""
        return self._map_sizes
    
    @property
    def map_groups(self):
        """A dictionary holding the parameters the maps point to."""
        return self._map_groups

    @property
    def tensors(self):
        """A dictionary of the final, dense JAX tensors."""
        return self._tensors
    
    @property
    def tensor_shape(self):
        """The shape of the generated tensors."""
        return self._tensor_shape
    
    @property
    def tensor_dim_names(self):
        """A list of names for each tensor dimension."""
        return self._tensor_dim_names

    @property
    def tensor_dim_labels(self):
        """A list of labels for each tensor dimension."""
        return self._tensor_dim_labels


