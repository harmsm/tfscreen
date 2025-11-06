from tfscreen.genetics import (
    set_categorical_genotype
)

from tfscreen.util import (
    read_dataframe,
    get_scaled_cfu,
    check_columns,
)

import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd

import inspect
import collections

class TensorManager:
    """
    Manages the wrangling of experimental data from a DataFrame into
    JAX-compatible tensors.

    This class handles data loading, preprocessing, and pivoting to create
    dense, multi-dimensional tensors required for hierarchical models in
    JAX/Numpyro. It creates parameter maps, handles NaN values, and builds a
    final data dictionary.
    
    Attributes
    ----------
    df : pd.DataFrame
        The fully preprocessed pandas DataFrame.
    parameter_indexers : dict
        A dictionary mapping parameter names (e.g., "genotype") to their
        own mapping dictionaries.
    data_dict : dict
        A dictionary of scalar values and small arrays needed by the model
        (e.g., "wt_index", "num_genotype").
    tensors : dict
        A dictionary of the final, dense JAX tensors (e.g., "ln_cfu",
        "good_mask").
    tensor_shape : tuple
        The shape of the generated tensors (replicates, timepoints,
        treatments, genotypes).
    tensor_dim_labels : list
        A list of pandas.CategoricalIndex objects, where each provides
        the labels for a tensor dimension.

    Notes
    -----
    The output tensor dimensions are always:
        (num_replicates, num_timepoints, num_treatments, num_genotypes)

    The `tensors` dict will always include:
    + ln_cfu: observed ln_cfu
    + ln_cfu_std: uncertainty in observed ln_cfu (standard error)
    + t_pre: amount of time the sample grew in pre-selection conditions
    + t_sel: amount of time the sample grew in selection conditions
    + map_genotype: map indicating which genotype corresponds to what cell
      in the tensor.
    + good_mask: bool mask that indicates which values of ln_cfu are non-nan

    The `data_dict` will always include:
    + wt_index: position of the wildtype genotype on the tensor genotype axis
    + not_wt_mask: integer index mask selecting non-wildtype genotypes
    + num_not_wt: number of non-wildtype genotypes
    """

    def __init__(self,df,treatment_columns=None):
        """
        Initializes the TensorManager.

        Parameters
        ----------
        df : pd.DataFrame or str
            The input DataFrame or a path to a file (e.g., CSV, Excel)
            that can be read by `read_dataframe`.
        treatment_columns : list of str, optional
            A list of column names that, together, uniquely define an
            experimental treatment. If None (default), uses a standard set:
            ['condition_pre', 'condition_sel', 'titrant_name', 'titrant_conc'].

        """

        self._prep_dataframe(df)

        # Columns that select a unique treatment experienced by a genotype, then
        # followed over time. 
        if treatment_columns is None:
            treatment_columns = ['condition_pre',
                                 'condition_sel',
                                 'titrant_name',
                                 'titrant_conc']
        self._treatment_columns = treatment_columns
        check_columns(self._df, required_columns=self._treatment_columns)

        self._parameter_indexers = {}
        self._data_dict = {}
    
        # These tensors will always be generated
        self._to_tensor_columns = ['ln_cfu','ln_cfu_std','t_pre','t_sel']
        
        # Empty tensor data for now
        self._tensors = None
        self._tensor_shape = None
        self._tensor_dim_labels = None

        # add genotype parameter map, then wildtype information. 
        self.add_parameter_map("genotype",["genotype"])
        self._load_wt_info()

    def _prep_dataframe(self,df):
        """
        Prepares the input DataFrame for tensor creation.

        This method reads the DataFrame if it's a path, calculates `ln_cfu`
        and `ln_cfu_std` if missing, ensures a 'replicate' column exists,
        checks for all required columns, and standardizes genotype/replicate
        columns as categorical.

        Parameters
        ----------
        df : pd.DataFrame or str
            Input DataFrame or file path.

        Raises
        ------
        ValueError
            If required columns are missing and cannot be calculated.
        
        """

        # read from file or work on a copy
        self._df = read_dataframe(df)    
    
        # Make sure we have required ln_cfu/std columns, calculating if
        # needed/possible
        self._df = get_scaled_cfu(self._df,need_columns=["ln_cfu","ln_cfu_std"])

        # make a replicate column if not defined
        if "replicate" not in self._df.columns:
            self._df["replicate"] = 1

        # check for all required columns
        required = ["ln_cfu","ln_cfu_std","replicate"]
        required.extend(["condition_pre","condition_sel"])
        required.extend(["t_pre","t_sel"])
        required.extend(["titrant_name","titrant_conc"])
        check_columns(self._df,required_columns=required)
                                                    
        # Set genotype and replicate to categorical.
        self._df = set_categorical_genotype(self._df,standardize=True)
        self._df['replicate'] = pd.Categorical(self._df['replicate'])

    def _load_wt_info(self):
        """
        Extracts and stores wildtype genotype information.

        Finds the integer index for the 'wt' genotype from the 'genotype'
        parameter map. It then creates and stores the 'wt_index', a mask for
        non-wildtype genotypes ('not_wt_mask'), and the count of non-wildtype
        genotypes ('num_not_wt') in `self._data_dict`.

        Raises
        ------
        ValueError
            If 'wt' is not found in the genotype parameter map.
        """

        if "wt" not in self._parameter_indexers["genotype"]:
            raise ValueError("df['genotype'] must have a wt entry")
        
        self._data_dict["wt_index"] = self._parameter_indexers["genotype"]["wt"]
        
        # Create an array of all genotype indices [0, 1, ..., num_genotype-1]
        all_geno_indices = jnp.arange(self._data_dict["num_genotype"],dtype=int)
        
        # Build the mask on this 1D array
        not_wt_mask_1d = all_geno_indices[all_geno_indices != self._data_dict["wt_index"]]
        
        self._data_dict["not_wt_mask"] = not_wt_mask_1d
        self._data_dict["num_not_wt"] = len(not_wt_mask_1d)

    def add_parameter_map(self,name,select_cols):
        """
        Creates an integer map for a new parameter based on DataFrame columns.

        Groups the DataFrame by `select_cols` and creates a new column
        `map_{name}` containing a unique integer index (via `ngroup`) for
        each group. It stores the mapping dictionary in
        `self._parameter_indexers` and the total count in `self._data_dict`.

        The resulting `map_{name}` column is added to `self._to_tensor_columns`.

        Parameters
        ----------
        name : str
            The base name for the parameter (e.g., "genotype").
        select_cols : list of str
            The DataFrame columns to group by to create the map.
        """

        map_name = f"map_{name}"
        self._df[map_name] = self._df.groupby(select_cols,
                                              sort=False,
                                              observed=True).ngroup()
        to_idx = (self._df[[map_name] + select_cols]
                  .drop_duplicates()
                  .set_index(select_cols))[map_name].to_dict()
        
        num_entries = self._df[map_name].drop_duplicates().size

        self._parameter_indexers[name] = to_idx
        self._data_dict[f"num_{name}"] = num_entries
        self._to_tensor_columns.append(map_name)

    def add_condition_map(self):
        """
        Creates unified integer maps for pre-selection and selection conditions.

        This method is specialized for experimental conditions. It finds all
        unique (replicate, condition) pairs across both `condition_pre` and
        `condition_sel` columns, creates a single unified integer map
        (`map_cond`), and then merges this map back onto the DataFrame as
        `map_cond_pre` and `map_cond_sel`.

        This ensures identical conditions (e.g., "LB") have the same index
        regardless of whether they appeared in the 'pre' or 'sel' column.
        The resulting maps are added to `self._to_tensor_columns`.
        """

        # Build array mapping unique replicate/combination combos to indexes
        pre = self._df[["replicate", "condition_pre"]].rename(columns={"condition_pre": "condition"})
        sel = self._df[["replicate", "condition_sel"]].rename(columns={"condition_sel": "condition"})
        cond_df = pd.concat([pre, sel]).drop_duplicates().reset_index(drop=True)
        cond_df["map_cond"] = cond_df.groupby(["condition","replicate"], sort=False, observed=True).ngroup()

        # Merge back onto df for condition_pre, creating 'map_cond_pre'
        self._df = self._df.merge(
            cond_df,
            left_on=["replicate", "condition_pre"],
            right_on=["replicate", "condition"],
            how="left"
        ).rename(columns={"map_cond": "map_cond_pre"}).drop(columns="condition")

        # Merge back onto df for condition_sel, creating 'map_cond_sel'
        self._df = self._df.merge(
            cond_df,
            left_on=["replicate", "condition_sel"],
            right_on=["replicate", "condition"],
            how="left"
        ).rename(columns={"map_cond": "map_cond_sel"}).drop(columns="condition")

        # dictionary will map (replicate,condition) to index
        condition_to_idx = dict(zip(zip(cond_df["condition"],cond_df["replicate"]),
                                    cond_df["map_cond"]))

        self._to_tensor_columns.extend(["map_cond_pre","map_cond_sel"])
        self._parameter_indexers["k"] = condition_to_idx
        self._parameter_indexers["m"] = condition_to_idx
        self._data_dict["num_condition"] = cond_df["condition"].drop_duplicates().size

    def add_data_tensor(self,name):
        """
        Registers a DataFrame column to be included in the final tensors.

        The values in the column should be coercible to floats. The column
        name will be used as the key in the `self.tensors` dictionary.

        Parameters
        ----------
        name : str
            The name of the column in `self._df` to be converted into a tensor.
        """

        self._to_tensor_columns.append(name)

    def _pivot_df(self):
        """
        Pivots the tidy DataFrame into a dense, multi-dimensional structure.

        This method is the core of the tensor creation. It uses
        `pandas.pivot_table` to reshape the data from a long format to a wide,
        dense format matching the tensor dimensions. It builds an exhaustive
        `MultiIndex` to ensure all tensor cells are present, filling missing
        data with NaN.

        Crucially, it pivots on the pre-computed integer codes for each
        dimension (e.g., `replicate.cat.codes`, `map_genotype`) to ensure
        the final tensor axes are correctly aligned with the parameter maps.

        Returns
        -------
        pd.DataFrame
            A pivoted DataFrame, reindexed to be exhaustive.
        """

        pivot_index = ['rep_idx', 'time_idx', 'treat_idx', 'geno_idx']
        
        # Build aggregate treatment column
        self._df['treatment'] = pd.Categorical(
            self._df[self._treatment_columns].apply(tuple, axis=1)
        )

        # These columns are going to be our tensor dimension (used for a pivot)
        self._df['rep_idx'] = self._df['replicate'].cat.codes
        self._df['time_idx'] = (self._df
                                .groupby(['replicate','genotype','treatment'],observed=False)['t_sel']
                                .rank(method='first')
                                .astype(int) - 1)
        self._df['treat_idx'] = self._df['treatment'].cat.codes
        self._df['geno_idx'] = self._df['genotype'].cat.codes

        # Build an exhaustive multi-index across our target tensor dimensions
        dim_sizes = [
            len(self._df['replicate'].cat.categories),
            self._df['time_idx'].max() + 1,
            len(self._df['treatment'].cat.categories),
            self._data_dict['num_genotype'] 
        ]
        dim_codes = [np.arange(size) for size in dim_sizes]
        exhaustive_index = pd.MultiIndex.from_product(dim_codes, 
                                                      names=pivot_index)

        # Use pivot_table to reshape the data.
        pivoted_df = pd.pivot_table(
            self._df,
            values=self._to_tensor_columns,
            index=pivot_index,
            observed=False
        )

        # Reindex, forcing the complete set of indices
        return pivoted_df.reindex(exhaustive_index)


    def create_tensors(self):
        """
        Converts the pivoted DataFrame into a dictionary of JAX tensors.

        This method first calls `_pivot_df`. It then iterates over the
        pivoted data, converts each column to a `jnp.ndarray`, reshapes it
        to the final tensor shape, and stores it in `self._tensors`.

        It also creates the `good_mask` tensor to identify non-NaN values.
        All NaN values are filled with `1.0` to avoid JAX/Numpyro errors
        during sampling, but the mask ensures these values are ignored.

        This method populates `self._tensors`, `self._tensor_shape`, and
        `self._tensor_dim_labels`.    
        """

        # Pivot the dataframe
        pivoted_df = self._pivot_df()

        # The 'time_idx' is the only one not from a categorical, so we get its
        # unique sorted values.
        time_labels = np.sort(self._df['time_idx'].unique())

        # Get dimension labels
        self._tensor_dim_labels = [
            self._df['replicate'].cat.categories,  # Labels for dim 0
            time_labels,                           # Labels for dim 1
            self._df['treatment'].cat.categories,  # Labels for dim 2
            self._df['genotype'].cat.categories    # Labels for dim 3
        ]
    
        # Get the labels along each tensor dimension, as well as the tensor 
        # shape. 
        self._tensor_shape = tuple(len(lab) for lab in self._tensor_dim_labels)
        
        # Build mask indicating good values. This will have the shape of the rest of
        # the tensors with False for any nan values created during the pivot.
        self._tensors = {}
        self._tensors["good_mask"] = jnp.asarray(
            pivoted_df["ln_cfu"].notna().to_numpy().reshape(self._tensor_shape)
        )

        # df cleanup -- remove nan prior to building tensors. We can set them 
        # to whatever we want as long as we apply good_mask when doing the 
        # analysis. Has to be above zero because jax/numpyro demands non-zero
        # values when it checks std on final obs/Normal samples -- even with 
        # the mask
        pivoted_df.fillna(1,inplace=True)

        # Convert to tensors, reshaping and casting as needed
        for c in self._to_tensor_columns:

            # Cast to float and reshape
            tensor = jnp.asarray(pivoted_df[c].to_numpy(dtype=float).reshape(self._tensor_shape))

            # Final cast to int (for maps) and 32 bit (everything else)
            if c.startswith("map_"):
                tensor = jnp.asarray(tensor,dtype=int)
            else:
                tensor = jnp.asarray(tensor,dtype=jnp.float32)
            
            # Record the tensor
            self._tensors[c] = tensor

        self._data_dict["num_replicate"] = self._tensor_shape[0]
        self._data_dict["num_time"] = self._tensor_shape[1]
        self._data_dict["num_treatment"] = self._tensor_shape[2]
        # This is set above -- here for parallelism
        #self._data_dict["num_genotype"] = self._tensor_shape[3]

    def build_dataclass(self,target_dataclass):
        """
        Populates a target dataclass with data from the manager.

        This method inspects the `__init__` signature of `target_dataclass`
        and populates its arguments using keys from `self.tensors` and
        `self.data_dict`.

        It includes validation to ensure no Python lists or tuples are
        passed, as these are not valid JAX Pytree nodes.

        Parameters
        ----------
        target_dataclass : type
            The (e.g., flax) dataclass to instantiate.

        Returns
        -------
        object
            An instance of `target_dataclass` populated with tensor data.

        Raises
        ------
        ValueError
            If a required dataclass parameter is not found in
            `self.tensors` or `self.data_dict`.
        ValueError
            If a value is a Python `list` or `tuple`, which are
            not valid JAX Pytree nodes.
        """

        # Get required parameters
        required_keys = inspect.signature(target_dataclass).parameters

        # Construct dataclass kwargs from data in dataclass. 
        dataclass_kwargs = {}
        for k in required_keys:
        
            if k in self.tensors:
                value = self.tensors[k]
            elif k in self.data_dict:
                value = self.data_dict[k]
            else:
                raise ValueError(
                    f"could not find required parameter '{k}' in {self.__class__.__name__}"
                )

            # Check if the value is an iterable (like list or tuple) but NOT a
            # string/bytes and NOT a JAX array.
            is_iterable = isinstance(value, collections.abc.Iterable)
            is_string = isinstance(value, (str, bytes))
            is_jax_array = isinstance(value, jax.Array) 

            if is_iterable and not is_string and not is_jax_array:
                raise ValueError(
                    f"Parameter '{k}' is a '{type(value)}', but must be a "
                    f"jnp.ndarray, scalar, or bool. Python lists/tuples are not "
                    f"valid JAX Pytree nodes."
                )        
            
            dataclass_kwargs[k] = value
         
        return target_dataclass(**dataclass_kwargs) 

    @property
    def df(self):
        """The final, preprocessed pandas DataFrame."""
        return self._df
    
    @property
    def parameter_indexers(self):
        """A dictionary of parameter-to-integer mapping dictionaries."""
        return self._parameter_indexers
    
    @property
    def data_dict(self):
        """A dictionary of scalar values and small arrays for the model."""
        return self._data_dict

    @property
    def tensors(self):
        """A dictionary of the final, dense JAX tensors."""
        return self._tensors
    
    @property
    def tensor_shape(self):
        """The shape of the generated tensors (replicates, timepoints, treatments, genotypes)."""
        return self._tensor_shape
    
    @property
    def tensor_dim_labels(self):
        """A list of labels for each tensor dimension."""
        return self._tensor_dim_labels


