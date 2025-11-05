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
    Tensor dimensions will be:
        (num_replicates,num_timepoints,num_treatments,num_genotypes)

    The tensors will always include the following:
    + ln_cfu: observed ln_cfu 
    + ln_cfu_std: uncertainty in observed ln_cfu (standard error)
    + t_pre: amount of time the sample grew in pre-selection conditions
    + t_sel: amount of time the sample grew in selection conditions
    + map_genotype: map indicating which genotype corresponds to what cell in 
      the tensor. 
    + good_mask: bool mask that indicates which values of ln_cfu are non-nan

    The data_dict will always include:
    + wt_index: position of the wildtype genotype on the tensor genotype axis
    + not_wt_mask: bool mask selecting non-wildtype genotypes along the tensor
      genotype axis. 
    + num_not_wt: number of non-wildtype genotypes
    
    """

    def __init__(self,df,treatment_columns=None):

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
        Will create a tensor with the name `map_{name}` """

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
        Map things like replicate + pheS-4CP and kanR+kan to integer indexes. 
        This is its own method so we can merge identical conditions in 
        condition_pre and condition_sel into a single set of maps. map_cond_pre
        and map_cond_sel map specific rows to the integer index for the
        (replicate,condition_pre) and (replicate,condition_sel) combos. 
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
        Add a specific data column to the tensor list. This will appear in 
        self.tensors[name] and will be returned as float32. 
        """

        self._to_tensor_columns.append(name)

    def _pivot_df(self):
        """
        Pivot the dataframe on columns in `pivot_index`, keeping the values in 
        to_tensor_columns.
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
        dim_labels = [np.sort(self._df[idx].unique()) for idx in pivot_index]
        exhaustive_index = pd.MultiIndex.from_product(dim_labels,names=pivot_index)

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
            
        # Pivot the dataframe
        pivoted_df = self._pivot_df()

        # Hold what each entry in each tensor dimension holds on to
        self._tensor_dim_labels = [level.to_numpy() for level in pivoted_df.index.levels]
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

        self._data_dict["num_time"] = self._tensor_shape[1]
        self._data_dict["tensor_shape_i"] = self._tensor_shape[0]
        self._data_dict["tensor_shape_j"] = self._tensor_shape[1]
        self._data_dict["tensor_shape_k"] = self._tensor_shape[2]
        self._data_dict["tensor_shape_l"] = self._tensor_shape[3]

    def build_dataclass(self,target_dataclass):
        """
        Construct a target flax dataclass from the loaded tensors. 
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
        return self._df
    
    @property
    def parameter_indexers(self):
        return self._parameter_indexers
    
    @property
    def data_dict(self):
        return self._data_dict

    @property
    def tensors(self):
        return self._tensors
    
    @property
    def tensor_shape(self):
        return self._tensor_shape
    
    @property
    def tensor_dim_labels(self):
        return self._tensor_dim_labels


