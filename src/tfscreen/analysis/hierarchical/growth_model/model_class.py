import tfscreen
from tfscreen.__version__ import __version__

import yaml

from tfscreen.util.dataframe import add_group_columns
from tfscreen.analysis.hierarchical import (
    TensorManager,
    populate_dataclass
)

from tfscreen.analysis.hierarchical.growth_model.model import jax_model
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry
from tfscreen.analysis.hierarchical.growth_model.batch import get_batch

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    DataClass,
    BindingData,
    GrowthData,
    PriorsClass,
    GrowthPriors,
    BindingPriors
)

import jax
from jax import numpy as jnp
import pandas as pd
import numpy as np

from functools import partial
import os
import warnings

# Declare float datatype
FLOAT_DTYPE = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

# Set zero conc to this when taking log
ZERO_CONC_VALUE = 1e-20


def _read_growth_df(growth_df,
                    theta_group_cols=None,
                    treatment_cols=None):
    """
    Reads and preprocesses a DataFrame containing growth curve data.

    This function performs several key operations:
    1.  Loads the DataFrame (if a path is given).
    2.  Sets the 'genotype' column to a standardized categorical.
    3.  Validates or calculates 'ln_cfu' and 'ln_cfu_std'.
    4.  Ensures a 'replicate' column exists.
    5.  Checks for all required data columns.
    6.  Adds group columns for 'treatment' and 'map_theta_group' using
        `add_group_columns`.

    Parameters
    ----------
    growth_df : pd.DataFrame or str
        DataFrame or path to a file holding growth data. Must have columns
        ln_cfu, ln_cfu_std, t_pre, and t_sel. It also must have all columns
        in theta_group_cols and treatment_cols.
    theta_group_cols : list, optional
        Column names used to define unique theta parameter groups.
        If not specified, defaults to ["genotype", "titrant_name"].
    treatment_cols : list, optional
        Column names used to define unique growth treatment conditions.
        If not specified, defaults to ["condition_pre", "condition_sel",
        "titrant_name", "titrant_conc"].

    Returns
    -------
    pd.DataFrame
        A copy of the processed growth DataFrame with new group columns.
    """
    
    # Get default theta_group columns
    if theta_group_cols is None:
        theta_group_cols = ["genotype","titrant_name"]

    # Get default treatment columns
    if treatment_cols is None:
        treatment_cols =  ["condition_pre","condition_sel",
                           "titrant_name","titrant_conc"]

    # Read dataframe, make sure genotypes are categorical, and validate or
    # calculate ln_cfu and ln_cfu_std
    growth_df = tfscreen.util.io.read_dataframe(growth_df)
    growth_df = tfscreen.genetics.set_categorical_genotype(growth_df,standardize=True)
    growth_df = tfscreen.util.dataframe.get_scaled_cfu(growth_df,need_columns=["ln_cfu","ln_cfu_std"])

    # make a replicate column if not defined
    if "replicate" not in growth_df.columns:
        growth_df["replicate"] = 1
    
    # check for all required columns
    required = theta_group_cols[:]
    required.extend(treatment_cols)
    required.extend(["ln_cfu","ln_cfu_std","replicate","t_pre","t_sel"])
    tfscreen.util.dataframe.check_columns(growth_df,required_columns=required)

    # These two maps are used to look up parameters after sampling posteriors
    growth_df = add_group_columns(target_df=growth_df,
                                  group_cols=treatment_cols,
                                  group_name="treatment")

    growth_df = add_group_columns(target_df=growth_df,
                                  group_cols=theta_group_cols,
                                  group_name="map_theta_group")
    
    mapper = {}
    for _, sub_df in growth_df.groupby(["condition_pre"]):
        cond_sel = list(pd.unique(sub_df["condition_sel"]))
        mapper.update({c:i for i, c in enumerate(cond_sel)})

    growth_df["condition_sel_reduced"] = growth_df["condition_sel"].map(mapper)

        
    return growth_df

def _build_growth_tm(growth_df):
    """
    Builds a TensorManager for the main growth data.

    This function configures a TensorManager to pivot the growth data into
    a dense 4D tensor with the shape:
    (replicate, time, treatment, genotype)

    It also registers all necessary data columns (e.g., 'ln_cfu') and
    parameter mapping columns (e.g., 'map_ln_cfu0') to be included in
    the final tensor dictionary.

    Parameters
    ----------
    growth_df : pd.DataFrame
        The processed growth DataFrame, typically from `_read_growth_df`.
        Must contain all columns required for pivots and maps.

    Returns
    -------
    tfscreen.analysis.hierarchical.TensorManager
        A fully-processed TensorManager instance after `create_tensors()`
        has been called.

    """

    # Add pivot column so we can both use this as a pivot and value
    growth_df["pivot_titrant_conc"] = growth_df["titrant_conc"]

    # Create tensor manager for construction of growth experiment tensors
    growth_tm = TensorManager(growth_df)

    # Define that we want a 4D tensor (replicate,time,treatment,genotype)
    growth_tm.add_pivot_index(tensor_dim_name="replicate",cat_column="replicate")
    growth_tm.add_ranked_pivot_index(tensor_dim_name="time",
                                     rank_column="t_sel",
                                     select_cols=['replicate','genotype','treatment'])
    growth_tm.add_pivot_index(tensor_dim_name="condition_pre",cat_column="condition_pre")
    growth_tm.add_pivot_index(tensor_dim_name="condition_sel",cat_column="condition_sel_reduced")
    growth_tm.add_pivot_index(tensor_dim_name="titrant_name",cat_column="titrant_name")
    growth_tm.add_pivot_index(tensor_dim_name="titrant_conc",cat_column="pivot_titrant_conc")
    growth_tm.add_pivot_index(tensor_dim_name="genotype",cat_column="genotype")
    
    # Register that we want tensors from these data columns
    growth_tm.add_data_tensor("ln_cfu",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("ln_cfu_std",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("t_pre",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("t_sel",dtype=FLOAT_DTYPE)

    # This creates a full-dimension map tensor that lets us look up the growth
    # conditions (pre and sel) for each element in the tensor. 
    # By calling with `select_pool_cols`, we pool names of conditions on 
    # condition_pre and condition_sel. Basically this does the operation 
    # unique(replicate + condition_pre OR replicate + condition_sel). This
    # will create map_condition_pre and map_condition_sel. 
    growth_tm.add_map_tensor(select_cols=["replicate"],
                             select_pool_cols=["condition_pre","condition_sel"],
                             name="condition")

    # These maps will allow us to extract parameter values from posterior 
    # samples. 
    growth_tm.add_map_tensor(["replicate","condition_pre","genotype"],
                            name="ln_cfu0")
    growth_tm.add_map_tensor("genotype",name="genotype")
    growth_tm.add_map_tensor(["titrant_name","titrant_conc","genotype"],name="theta")

    # Do final tensor creation
    growth_tm.create_tensors()

    return growth_tm


def _read_binding_df(binding_df,
                     growth_df,
                     theta_group_cols=None):
    """
    Reads and processes a DataFrame containing direct binding data.

    This function:
    1.  Loads the DataFrame.
    2.  Standardizes the 'genotype' column.
    3.  Checks for required columns ('theta_obs', 'theta_std', etc.).
    4.  Calls `add_group_columns` using `existing_df` (from the growth data)
        to ensure that the 'map_theta_group' column is *identical* to the
        one used for the growth data, linking the two datasets.

    Parameters
    ----------
    binding_df : pd.DataFrame or str
        DataFrame or path to file holding binding data.
    existing_df : pd.DataFrame, optional
        The processed growth DataFrame (`growth_tm.df`). This is used to
        enforce consistent 'map_theta_group' indexing.
    theta_group_cols : list, optional
        Column names used to define unique theta parameter groups.
        If not specified, defaults to ["genotype", "titrant_name"].

    Returns
    -------
    pd.DataFrame
        A copy of the processed binding DataFrame.
    """

    # Get default theta_group columns
    if theta_group_cols is None:
        theta_group_cols = ["genotype","titrant_name"]

    # Load dataframe (either loads from file or works on copy)
    binding_df = tfscreen.util.io.read_dataframe(binding_df)
    binding_df = tfscreen.genetics.set_categorical_genotype(binding_df,
                                                            standardize=True)

    # check for all required columns
    required = theta_group_cols[:]
    required.extend(["theta_obs","theta_std","titrant_conc"])
    tfscreen.util.dataframe.check_columns(binding_df,required_columns=required)


    cols = ["genotype","titrant_name"]
    binding_seen = binding_df[cols].drop_duplicates().set_index(cols)
    growth_seen = growth_df[cols].drop_duplicates().set_index(cols)
    is_subset = binding_seen.index.isin(growth_seen.index).all()
    if not is_subset:
        raise ValueError(
            "binding_df contains genotype/titrant_name pairs that were not seen "
            "in the growth_df."
        )

    return binding_df

def _build_binding_tm(binding_df):
    """
    Builds a TensorManager for the direct binding data.

    This function pivots the binding data into a dense 3D tensor with the shape:
    (titrant_name, titrant_conc, genotype)

    It registers the 'theta_obs', 'theta_std', 'titrant_conc', and
    'map_theta_group' columns to be included in the final tensors.

    Parameters
    ----------
    binding_df : pd.DataFrame
        The processed binding DataFrame, typically from `_read_binding_df`.

    Returns
    -------
    tfscreen.analysis.hierarchical.TensorManager
        A processed TensorManager instance after `create_tensors()`
        has been called.
    """

    # Create a titrant_conc column to use as a pivot for the tensor. The pivot
    # consumes the column. This move allows us to preserve titrant_conc and use
    # its values to populate a tensor.
    binding_df["pivot_titrant_conc"] = binding_df["titrant_conc"]

    # Create tensor manager
    binding_tm = TensorManager(binding_df)

    # Add pivots on titrant_conc and theta_group to create [num_titrant,num_theta_group]
    # output tensors
    binding_tm.add_pivot_index(tensor_dim_name="titrant_name",cat_column="titrant_name")
    binding_tm.add_pivot_index(tensor_dim_name="titrant_conc",cat_column="pivot_titrant_conc")
    binding_tm.add_pivot_index(tensor_dim_name="genotype",cat_column="genotype")

    # add data columns 
    binding_tm.add_data_tensor("theta_obs",dtype=FLOAT_DTYPE)
    binding_tm.add_data_tensor("theta_std",dtype=FLOAT_DTYPE)

    # Build tensors
    binding_tm.create_tensors()
    
    return binding_tm


def _setup_batching(growth_genotypes,
                    binding_genotypes,
                    batch_size):
    """
    Calculates batching indices and scale factors.

    This function determines how to batch genotypes, handling the fact that
    binding data is only available for a subset of genotypes. It ensures
    that binding genotypes are always prioritized and handled correctly
    relative to the growth-only genotypes.

    Parameters
    ----------
    growth_genotypes : np.ndarray
        Array of genotype names in the growth dataset.
    binding_genotypes : np.ndarray
        Array of genotype names in the binding dataset.
    batch_size : int, optional
        The desired size of the batch.

    Returns
    -------
    dict
        A dictionary containing:
        - `batch_idx`: Indices for the current batch.
        - `batch_size`: The actual size of the batch.
        - `scale_vector`: Scaling factors for subsampling correction.
        - `num_binding`: Number of overlapping binding genotypes.
        - `not_binding_idx`: Indices of genotypes without binding data.
        - `not_binding_batch_size`: Number of non-binding genotypes in batch.
    """
    
    if batch_size is None:
        batch_size = len(growth_genotypes)
    
    if batch_size > len(growth_genotypes):
        batch_size = len(growth_genotypes)

    # Use growth_genotype order as the source-of-truth for indexing
    binding_idx = np.where(np.isin(growth_genotypes,binding_genotypes))[0]
    not_binding_idx = np.where(~np.isin(growth_genotypes,binding_genotypes))[0]
    num_binding = len(binding_idx)
    num_not_binding = len(not_binding_idx)
    not_binding_batch_size = batch_size - num_binding

    # Build idx array. The first entries correspond to the binding data and are 
    # the same for all rounds
    idx = np.zeros(batch_size,dtype=int)
    idx[:num_binding] = binding_idx

    # Calculate scale vector, which will be the same for all rounds
    scale_vector = np.ones(batch_size,dtype=float)
    if not_binding_batch_size > 0:
        scale_vector[num_binding:] = num_not_binding/not_binding_batch_size

    # Return output as a dictionary so this can be loaded into the dataclass
    out = {}
    out["batch_idx"] = idx
    out["batch_size"] = batch_size
    out["scale_vector"] = scale_vector
    out["num_binding"] = num_binding
    out["not_binding_idx"] = not_binding_idx
    out["not_binding_batch_size"] = not_binding_batch_size

    return out


def _extract_param_est(input_df,
                       params_to_get,
                       map_column,
                       get_columns,
                       in_run_prefix,
                       param_posteriors,
                       q_to_get):
    """
    Extract parameter estimates and quantiles from posterior samples.

    This function creates a DataFrame for each parameter in `params_to_get`, mapping
    parameter indices to metadata columns, and fills in quantile columns for each
    requested quantile in `q_to_get` using the posterior samples in `param_posteriors`.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing metadata and mapping columns for the parameters.
    params_to_get : list of str
        List of parameter names to extract from the posterior samples.
    map_column : str
        Name of the column in `input_df` that maps rows to parameter indices.
    get_columns : list of str
        List of metadata columns to include in the output DataFrame.
    in_run_prefix : str
        Prefix to prepend to parameter names when looking them up in `param_posteriors`.
    param_posteriors : dict
        Dictionary mapping parameter names (with prefix) to posterior samples,
        where each value is a NumPy array of shape (num_samples, num_params).
    q_to_get : dict
        Dictionary mapping output column names to quantile values (between 0 and 1)
        to extract from the posterior samples.

    Returns
    -------
    out_dfs : dict
        Dictionary mapping each parameter name to a DataFrame containing the
        requested quantiles for each parameter, with metadata columns.

    Raises
    ------
    KeyError
        If a requested parameter or quantile is not found in `param_posteriors`.
    """
    # Create dataframe with unique rows for map_column that has columns 
    # get_columns + map_column. Rows will be sorted by map_column. 
    get_columns.append(map_column)
    df = (input_df
          .drop_duplicates(map_column)[get_columns]
          .sort_values(map_column)
          .reset_index(drop=True)
          .copy())

    # Go through all parameters requested
    out_dfs = {}
    for param in params_to_get:

        # Grab the posterior distribution of this parameters and flatten. 
        model_param = f"{in_run_prefix}{param}"
        flat_param = (param_posteriors[model_param].reshape(param_posteriors[model_param].shape[0],-1))

        # Create dataframe for loading the data
        to_write = df.copy()

        # Go through quantiles
        for q_name in q_to_get:

            # Calculate quantile and load into the output dataframe
            q = np.quantile(flat_param,q_to_get[q_name],axis=0)
            to_write[q_name] = q[to_write[map_column].values]

        # Record the final dataframe
        out_dfs[param] = to_write.drop(columns=[map_column])
    
    return out_dfs
    
class ModelClass:
    """
    Manages the data wrangling and configuration for the JAX growth model.

    This class serves as the main entry point for fitting the hierarchical
    growth and binding model. It takes raw DataFrames for ln(CFU) and
    theta observations, processes them into JAX-compatible Pytrees, and
    assembles the final `data`, `priors`, and `control` objects required
    by the `jax_model`.

    Parameters
    ----------
    growth_df : pd.DataFrame or str
        DataFrame or path to file with growth data.
    binding_df : pd.DataFrame or str
        DataFrame or path to file with binding data.
    condition_growth : str, optional
        Model name for condition-specific growth.
    ln_cfu0 : str, optional
        Model name for initial cell counts.
    dk_geno : str, optional
        Model name for genotype-specific death rate.
    activity : str, optional
        Model name for genotype activity.
    theta : str, optional
        Model name for theta calculation (e.g., "hill").
    transformation : str, optional
        Model name for transformation correction (e.g., "congression" or "single").
    theta_growth_noise : str, optional
        Model name for noise on theta in the growth model.
    theta_binding_noise : str, optional
        Model name for noise on theta in the binding model.

    Attributes
    ----------
    data : DataClass
        A JAX Pytree (flax dataclass) holding all data tensors.
    priors : PriorsClass
        A JAX Pytree holding all prior definitions.
    control : ControlClass
        A JAX Pytree holding integer codes that control model behavior.
    init_params : dict
        A dictionary of initial parameter guesses for optimization.
    jax_model : function
        The top-level JAX model function (from `model.py`).
    settings : dict
        A dictionary of the model choice strings used.
    """

    def __init__(self,
                 growth_df,
                 binding_df,
                 batch_size=None,
                 condition_growth="hierarchical",
                 ln_cfu0="hierarchical",
                 dk_geno="hierarchical",
                 activity="horseshoe",
                 theta="hill",
                 transformation="congression",
                 theta_growth_noise="none",
                 theta_binding_noise="none",
                 spiked_genotypes=None):

        self._ln_cfu_df = growth_df
        self._binding_df = binding_df

        self._batch_size = batch_size

        self._condition_growth = condition_growth
        self._ln_cfu0 = ln_cfu0
        self._dk_geno = dk_geno
        self._activity = activity
        self._theta = theta
        self._transformation = transformation
        self._theta_growth_noise = theta_growth_noise
        self._theta_binding_noise = theta_binding_noise
        self._spiked_genotypes = spiked_genotypes

        self._initialize_data()
        self._initialize_classes()


    def _initialize_data(self):
        """
        Loads, processes, and wrangles all data into JAX Pytrees.

        This method orchestrates the entire data pipeline:
        1.  Calls `_read_growth_df` and `_build_growth_tm`.
        2.  Calls `_read_binding_df` and `_build_binding_tm`, passing the
            growth data to ensure map consistency.
        3.  Extracts tensors and metadata from the TensorManagers.
        4.  Uses `populate_dataclass` to build the `GrowthData`, `BindingData`,
            and final `DataClass` Pytrees.
        5.  Sets the final `self._data` attribute.
        """
        # ---------------------------------------------------------------------
        # growth dataclass

        # Load in growth data, creating two blocks of tensors. One holds the
        # growth (replicate,time,condition,genotype) data. The other holds 
        # the theta (titrant_conc,theta_group) tensor. 
        self.growth_df = _read_growth_df(self._ln_cfu_df)
        self.growth_tm = _build_growth_tm(self.growth_df)
                   
        # Assemble tensors. 
        tensors = {}
        
        from_growth_tm = ["ln_cfu","ln_cfu_std","t_pre","t_sel",
                          "map_condition_pre","map_condition_sel","good_mask"]
                          
        for k in from_growth_tm:
            tensors[k] = self.growth_tm.tensors[k]

        # build dictionaries of sizes. 
        sizes = dict([(f"num_{s}",self.growth_tm.map_sizes[s]) for s in self.growth_tm.map_sizes])
        sizes["num_replicate"] = self.growth_tm.tensor_shape[0]
        sizes["num_time"] = self.growth_tm.tensor_shape[1]
        sizes["num_condition_pre"] = self.growth_tm.tensor_shape[2]
        sizes["num_condition_sel"] = self.growth_tm.tensor_shape[3]
        sizes["num_titrant_name"] = self.growth_tm.tensor_shape[4]        
        sizes["num_titrant_conc"] = self.growth_tm.tensor_shape[5]
        sizes["num_genotype"] = self.growth_tm.tensor_shape[6]

        # Get wildtype information
        genotype_idx = self.growth_tm.tensor_dim_names.index("genotype")
        wt_loc = np.where(self.growth_tm.tensor_dim_labels[genotype_idx] == "wt")
        wt_info = {"wt_indexes":jnp.array(wt_loc[0], dtype=jnp.int32)}

         # scatter_theta tells the theta model caller to return a full-sized
        # growth_tm tensor instead of the smaller growth_theta_tm-sized tensor
        # congression_mask is a boolean array of shape (num_genotype,) that 
        # tells the model which genotypes should be corrected for congression.
        # Initialize to all True (no masking).
        mask = np.ones(sizes["num_genotype"],dtype=bool)
        if self._spiked_genotypes is not None:
            
            # Make sure spiked_genotypes is a list or array
            if isinstance(self._spiked_genotypes,str):
                self._spiked_genotypes = [self._spiked_genotypes]

            # Get names of genotypes in the growth dataset
            genotype_idx = self.growth_tm.tensor_dim_names.index("genotype")
            genotype_names = self.growth_tm.tensor_dim_labels[genotype_idx]

            # Check for genotypes not in the dataset
            missing = []
            for g in self._spiked_genotypes:
                if g not in genotype_names:
                    missing.append(g)
            
            if len(missing) > 0:
                raise ValueError(
                    f"The following spiked_genotypes were not found in the growth "
                    f"dataset: {missing}"
                )

            # Update mask
            spiked_idx = np.where(np.isin(genotype_names,self._spiked_genotypes))[0]
            mask[spiked_idx] = False

        other_data = {"scatter_theta":1,
                      "congression_mask":jnp.array(mask,dtype=bool)}

        # Grab the titrant concentration and log_titrant_conc (1D array from 
        # the tensor labels along dimension 6)
        idx = np.where(np.array(self.growth_tm.tensor_dim_names) == "titrant_conc")[0][0]
        titrant_conc = np.array(self.growth_tm.tensor_dim_labels[idx])
        log_titrant_conc = titrant_conc.copy()
        log_titrant_conc[log_titrant_conc == 0] = ZERO_CONC_VALUE
        log_titrant_conc = np.log(log_titrant_conc)
        
        other_data["titrant_conc"] = titrant_conc
        other_data["log_titrant_conc"] = log_titrant_conc

        growth_data_sources = [tensors,sizes,wt_info,other_data]
        
        # ---------------------------------------------------------------------
        # binding dataclass
  
        # Load in the binding data, creating one block of tensors. This is
        # only (titrant_conc,theta_group). Use the growth tensor manager to
        # make sure that the mapping to parameters matches between the growth 
        # and the binding data. 
        self.binding_df = _read_binding_df(self._binding_df,self.growth_tm.df)
        self.binding_tm = _build_binding_tm(self.binding_df)

        # Grab the sizes
        sizes = {"num_titrant_name":self.binding_tm.tensor_shape[0],
                 "num_titrant_conc":self.binding_tm.tensor_shape[1],
                 "num_genotype":self.binding_tm.tensor_shape[2]}
        other_data = {"scatter_theta":0}

        # Grab the titrant concentration and log_titrant_conc (1D array from 
        # the tensor labels along dimension 6)
        idx = np.where(np.array(self.binding_tm.tensor_dim_names) == "titrant_conc")[0][0]
        titrant_conc = np.array(self.binding_tm.tensor_dim_labels[idx])
        log_titrant_conc = titrant_conc.copy()
        log_titrant_conc[log_titrant_conc == 0] = ZERO_CONC_VALUE
        log_titrant_conc = np.log(log_titrant_conc)

        other_data["titrant_conc"] = titrant_conc
        other_data["log_titrant_conc"] = log_titrant_conc

        binding_data_sources = [self.binding_tm.tensors,sizes,other_data]

        # ---------------------------------------------------------------------
        # Create batching information 
       
        batch_data = _setup_batching(self.growth_tm.tensor_dim_labels[-1],
                                     self.binding_tm.tensor_dim_labels[-1],
                                     self._batch_size)
        
        # Record relevant batch data for the growth dataset
        growth_batch_data = {}
        growth_batch_data["batch_idx"] = batch_data["batch_idx"]
        growth_batch_data["batch_size"] = batch_data["batch_size"]
        growth_batch_data["scale_vector"] = batch_data["scale_vector"]
        growth_batch_data["geno_theta_idx"] = np.arange(batch_data["batch_size"],dtype=int)
        growth_data_sources.append(growth_batch_data)
        
        # Record relevant batch data for the binding dataset
        binding_batch_data = {}
        binding_batch_data["batch_idx"] = batch_data["batch_idx"][:batch_data["num_binding"]]
        binding_batch_data["batch_size"] = batch_data["num_binding"]
        binding_batch_data["scale_vector"] = batch_data["scale_vector"][:batch_data["num_binding"]]
        binding_batch_data["geno_theta_idx"] = np.arange(batch_data["num_binding"],dtype=int)
        binding_data_sources.append(binding_batch_data)

        # ---------------------------------------------------------------------
        # Populate dataclasses

        # Populate a GrowthData flax dataclass with all keys in `sources`. 
        growth_dataclass = populate_dataclass(GrowthData,
                                              sources=growth_data_sources)

        # Populate a BindingData flax dataclass with all keys in `sources`
        binding_dataclass = populate_dataclass(BindingData,
                                               sources=binding_data_sources)

        source_data = [{"growth":growth_dataclass,
                        "binding":binding_dataclass,
                        "num_genotype":self.growth_tm.tensor_shape[-1]}]
        source_data.append(batch_data)

        # Build the aggregated `DataClass` flax dataclass with the growth
        # and binding dataclasses. Expose as a model attribute. 
        self._data = populate_dataclass(DataClass,
                                        sources=source_data)
        
    
    def _initialize_classes(self):
        """
        Initializes the control, priors, and init_params objects.

        This method:
        1.  Maps the model name strings (e.g., "hierarchical") to their
            corresponding integer codes and component objects from
            `model_registry`.
        2.  Populates and creates the `ControlClass` instance.
        3.  Populates and creates the `PriorsClass` instance by calling
            `get_priors()` on each component.
        4.  Generates the `init_params` dictionary by calling `get_guesses()`
            on each component.
        5.  Sets `self._control`, `self._priors`, and `self._init_params`.

        Raises
        ------
        ValueError
            If an unrecognized model name string is provided.
        """
        
        # The first value is the prefix of the parameter passed into the `name`
        # arguments of all model components. It is also the key to 
        # model_registry. The second value is the model name passed by 
        # the user/caller. It should match one of the possible model types for
        # the given model component in model_registry. For example, 
        # for "dk_geno", `self._dk_geno` could be 'fixed' or 'hierarchical'. 
        # The last value determines whether this is used to initialize the 
        # data.growth and data.binding. 
        load_map = [("condition_growth",self._condition_growth,"growth"),
                    ("ln_cfu0",self._ln_cfu0,"growth"),
                    ("dk_geno",self._dk_geno,"growth"),
                    ("activity",self._activity,"growth"),
                    ("theta",self._theta,"theta"),
                    ("transformation",self._transformation,"growth"),
                    ("theta_growth_noise",self._theta_growth_noise,"growth"),
                    ("theta_binding_noise",self._theta_binding_noise,"binding")]
        
        main_control_kwargs = {"is_guide":False}
        guide_control_kwargs = {"is_guide":True}
        priors_class_kwargs = {"growth":{},"binding":{},"theta":{}}
        init_params = {}
        for to_load in load_map:
    
            key, value, prior_group = to_load
    
            if key not in model_registry:
                raise ValueError(
                    f"{key} is not in model_registry. "
                    f"It should be one of: {list(model_registry.keys())}"
                )

            if value not in model_registry[key]:
                raise ValueError(
                    f"{key} '{value}' not recognized. "
                    f"It should be one of: {list(model_registry[key].keys())}"
                )
    
            # get the component module 
            component_module = model_registry[key][value]
    
            # Record priors
            priors_class_kwargs[prior_group][key] = component_module.get_priors()
    
            # Record guesses
            if prior_group == "binding":
                guesses = component_module.get_guesses(name=key,
                                                       data=self._data.binding)
            else:
                guesses = component_module.get_guesses(name=key,
                                                       data=self._data.growth)
            init_params.update(guesses)

            # Record control parameters for the main and guide functions
            if key == "theta":
                main_control_kwargs[key] = (component_module.define_model, 
                                            component_module.run_model,
                                            component_module.get_population_moments)
                guide_control_kwargs[key] = (component_module.guide, 
                                             component_module.run_model,
                                             component_module.get_population_moments)
            elif key == "transformation":
                main_control_kwargs[key] = (component_module.define_model, 
                                            component_module.update_thetas)
                guide_control_kwargs[key] = (component_module.guide, 
                                             component_module.update_thetas)
            else:
                main_control_kwargs[key] = component_module.define_model
                guide_control_kwargs[key] = component_module.guide
    

        main_control_kwargs["batch_size"] = self._batch_size
        guide_control_kwargs["batch_size"] = self._batch_size

        # Set the observables
        main_control_kwargs["observe_binding"] = model_registry["observe_binding"].observe
        main_control_kwargs["observe_growth"] = model_registry["observe_growth"].observe
        
        guide_control_kwargs["observe_binding"] = model_registry["observe_binding"].guide
        guide_control_kwargs["observe_growth"] = model_registry["observe_growth"].guide

        # Store main control kwargs for later
        self.main_control_kwargs = main_control_kwargs

        # bake the control arguments into the main and guide models
        self._jax_model = partial(jax_model, **main_control_kwargs)
        self._jax_model_guide = partial(jax_model, **guide_control_kwargs)
                
        # Build priors class
        growth_priors = populate_dataclass(GrowthPriors,
                                           sources=priors_class_kwargs["growth"])
        binding_priors = populate_dataclass(BindingPriors,
                                            sources=priors_class_kwargs["binding"])
        priors = populate_dataclass(PriorsClass,
                                    sources=dict(theta=priors_class_kwargs["theta"]["theta"],
                                                 growth=growth_priors,
                                                 binding=binding_priors))
    
        self._priors = priors
        self._init_params = init_params

    def extract_parameters(self,
                           posteriors,
                           q_to_get=None):
        """
        Extract parameter quantiles from posterior samples.

        This method extracts specified quantiles for each model parameter of
        interest, returning a dictionary of DataFrames with parameter estimates
        and associated metadata.

        Parameters
        ----------
        posteriors : dict or str
            Assumes this is a dictionary of posteriors keying parameters to 
            numpy arrays or a path to a .npz file containing posterior samples
            for model parameters.
        q_to_get : dict, optional
            Dictionary mapping output column names to quantile values (between 0 and 1)
            to extract from the posterior samples. If None, a default set of quantiles
            is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
            upper_quartile, upper_95, max).

        Returns
        -------
        params : dict
            Dictionary mapping parameter names to DataFrames containing the requested
            quantiles and metadata columns for each parameter.

        Raises
        ------
        ValueError
            If `q_to_get` is not a dictionary.
        """

        # Load the posterior file
        if isinstance(posteriors,(dict,np.lib.npyio.NpzFile)):
            param_posteriors = posteriors
        else:
            param_posteriors = np.load(posteriors)
        

        # Named quantiles to pull from the posterior distribution
        if q_to_get is None:
            q_to_get = {"min":0.0,
                        "lower_95":0.025,
                        "lower_std":0.159,
                        "lower_quartile":0.25,
                        "median":0.5,
                        "upper_quartile":0.75,
                        "upper_std":0.841,
                        "upper_95":0.975,
                        "max":1.0}
            
        # make sure q_to_get is a dictionary
        if not isinstance(q_to_get,dict):
            raise ValueError(
                "q_to_get should be a dictionary keying column names to quantiles"
            )

        # Define how to go about constructing dataframes to store the parameter
        # estimates. 
        extract = []

        # theta
        if self._theta == "hill":
            extract.append(
                dict(
                    input_df = self.growth_tm.df,
                    params_to_get = ["hill_n","log_hill_K","theta_high","theta_low"],
                    map_column = "map_theta_group",
                    get_columns = ["genotype","titrant_name"],
                    in_run_prefix = "theta_"
                )
            )
        elif self._theta == "categorical":
            extract.append(
                dict(
                    input_df = self.growth_tm.df,
                    params_to_get = ["theta"],
                    map_column = "map_theta",
                    get_columns = ["genotype","titrant_name","titrant_conc"],
                    in_run_prefix = "theta_"
                )
            )
        
        # condition
        if self._condition_growth in ["independent","hierarchical"]:
            extract.append(
                dict(
                    input_df = self.growth_tm.map_groups['condition'],
                    params_to_get = ["growth_m","growth_k"],
                    map_column = "map_condition",
                    get_columns = ["replicate","condition"],
                    in_run_prefix = "condition_"
                )
            )

        # ln_cfu0
        if self._dk_geno == "hierarchical":
            extract.append(
                dict(
                    input_df = self.growth_tm.df,
                    params_to_get = ["ln_cfu0"],
                    map_column = "map_ln_cfu0",
                    get_columns = ["replicate","condition_pre","genotype"],
                    in_run_prefix = ""
                )
            )

        # dk_geno
        if self._dk_geno == "none":
            pass
        elif self._dk_geno == "hierarchical":
            extract.append(
                dict(
                    input_df = self.growth_tm.df,
                    params_to_get = ["dk_geno"],
                    map_column = "map_genotype",
                    get_columns = ["genotype"],
                    in_run_prefix = ""
                )
            )

        # activity
        if self._activity == "fixed":
            pass
        elif self._activity in ["hierarchical","horseshoe"]:
            extract.append(
                dict(
                    input_df = self.growth_tm.df,
                    params_to_get = ["activity"],
                    map_column = "map_genotype",
                    get_columns = ["genotype"],
                    in_run_prefix = ""
                )
            )

        # transformation
        if self._transformation == "congression":
            # lam is global
            lam_df = pd.DataFrame({"parameter":["lam"], "map_all":[0]})
            extract.append(
                dict(
                    input_df = lam_df,
                    params_to_get = ["lam"],
                    map_column = "map_all",
                    get_columns = ["parameter"],
                    in_run_prefix = "transformation_"
                )
            )

            # mu and sigma are (titrant_name, titrant_conc)
            # We can use the growth_tm.df to find the unique (titrant_name, titrant_conc) pairs 
            # and their indices.
            trans_df = (self.growth_tm.df[["titrant_name", "titrant_conc", 
                                          "titrant_name_idx", "titrant_conc_idx"]]
                        .drop_duplicates()
                        .copy())
            
            # num_titrant_conc
            idx = np.where(np.array(self.growth_tm.tensor_dim_names) == "titrant_conc")[0][0]
            num_titrant_conc = len(self.growth_tm.tensor_dim_labels[idx])
            
            # Map column
            trans_df["map_trans"] = (trans_df["titrant_name_idx"] * num_titrant_conc + 
                                     trans_df["titrant_conc_idx"])
            
            extract.append(
                dict(
                    input_df = trans_df,
                    params_to_get = ["mu","sigma"],
                    map_column = "map_trans",
                    get_columns = ["titrant_name","titrant_conc"],
                    in_run_prefix = "transformation_"
                )
            )

        params = {}
        for kwargs in extract:
            params.update(_extract_param_est(param_posteriors=param_posteriors,
                                             q_to_get=q_to_get,
                                             **kwargs))

        return params

    def extract_theta_curves(self,
                             posteriors,
                             q_to_get=None,
                             manual_titrant_df=None):
        """
        Extract theta curves by sampling from the joint posterior distribution.

        This method calculates fractional occupancy (theta) across a range of
        titrant concentrations by sampling from the joint posterior of Hill
        parameters (hill_n, log_hill_K, theta_high, theta_low).

        Parameters
        ----------
        posteriors : dict or str
            Assumes this is a dictionary of posteriors keying parameters to
            numpy arrays or a path to a .npz file containing posterior samples
            for model parameters.
        q_to_get : dict, optional
            Dictionary mapping output column names to quantile values (between 0 and 1)
            to extract from the posterior samples. If None, a default set of quantiles
            is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
            upper_quartile, upper_95, max).
        manual_titrant_df : pd.DataFrame, optional
            A DataFrame specifying 'titrant_name' and 'titrant_conc' values
            at which to calculate theta. If provided, it overrides the default
            calculation at the concentrations present in the input data.
            If 'genotype' is present, it will be used; otherwise, the method
            will calculate theta for all genotypes in the model.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: 'genotype', 'titrant_name', 'titrant_conc',
            and requested quantiles of theta.

        Raises
        ------
        ValueError
            If the model was not initialized with theta='hill'.
            If `q_to_get` is not a dictionary.
            If `manual_titrant_df` is missing required columns.
        """

        if self._theta != "hill":
            raise ValueError(
                "extract_theta_curves is only available for models where "
                "theta='hill'."
            )

        # Load the posterior file
        if isinstance(posteriors,(dict,np.lib.npyio.NpzFile)):
            param_posteriors = posteriors
        else:
            param_posteriors = np.load(posteriors)

        # Named quantiles to pull from the posterior distribution
        if q_to_get is None:
            q_to_get = {"min":0.0,
                        "lower_95":0.025,
                        "lower_std":0.159,
                        "lower_quartile":0.25,
                        "median":0.5,
                        "upper_quartile":0.75,
                        "upper_std":0.841,
                        "upper_95":0.975,
                        "max":1.0}

        # make sure q_to_get is a dictionary
        if not isinstance(q_to_get,dict):
            raise ValueError(
                "q_to_get should be a dictionary keying column names to quantiles"
            )

        # Construct calculation DataFrame
        if manual_titrant_df is None:
            # Use unique (genotype, titrant_name, titrant_conc) from input data
            calc_df = (self.growth_tm.df[["genotype", "titrant_name", "titrant_conc", "map_theta_group"]]
                       .drop_duplicates()
                       .reset_index(drop=True))
        else:
            tfscreen.util.dataframe.check_columns(manual_titrant_df,
                                                  required_columns=["titrant_name", "titrant_conc"])
            
            # If genotype is not provided, broadcast across all genotypes
            if "genotype" not in manual_titrant_df.columns:
                genotypes = self.growth_tm.df["genotype"].unique()
                dfs = []
                for g in genotypes:
                    tmp = manual_titrant_df.copy()
                    tmp["genotype"] = g
                    dfs.append(tmp)
                calc_df = pd.concat(dfs).reset_index(drop=True)
            else:
                calc_df = manual_titrant_df.copy()

            # Map to theta groups
            # We need to reach into the GROWTH_TM to find the mapping
            # This is a bit tricky because the manual_titrant_df might have new concentrations.
            # BUT the parameters (hill_n, etc) are mapped to (genotype, titrant_name).
            # The mapper used in _extract_param_est for Hill model is "map_theta_group"
            # which maps (genotype, titrant_name) to an index.
            
            # Find the (genotype, titrant_name) -> map_theta_group mapping
            mapping = (self.growth_tm.df[["genotype", "titrant_name", "map_theta_group"]]
                       .drop_duplicates()
                       .set_index(["genotype", "titrant_name"])["map_theta_group"]
                       .to_dict())
            
            # Apply mapping. If a (genotype, titrant_name) pair is not in the model, it's an error.
            try:
                calc_df["map_theta_group"] = calc_df.set_index(["genotype", "titrant_name"]).index.map(mapping)
            except Exception as e:
                raise ValueError(
                    "Some (genotype, titrant_name) pairs in manual_titrant_df "
                    "were not found in the model data."
                ) from e
            
            if calc_df["map_theta_group"].isna().any():
                missing = calc_df[calc_df["map_theta_group"].isna()]
                raise ValueError(
                    f"The following (genotype, titrant_name) pairs were not found in the model data: "
                    f"{missing[['genotype', 'titrant_name']].drop_duplicates().values}"
                )

        # Extract posterior parameters and flatten (num_samples, num_groups)
        # Note: their common prefix is "theta_" (see extract_parameters)
        hill_n = param_posteriors["theta_hill_n"].reshape(param_posteriors["theta_hill_n"].shape[0], -1)
        log_hill_K = param_posteriors["theta_log_hill_K"].reshape(param_posteriors["theta_log_hill_K"].shape[0], -1)
        theta_high = param_posteriors["theta_theta_high"].reshape(param_posteriors["theta_theta_high"].shape[0], -1)
        theta_low = param_posteriors["theta_theta_low"].reshape(param_posteriors["theta_theta_low"].shape[0], -1)

        # Vectorized calculation of theta for all samples
        # titrant_conc shape: (N_points,)
        # mapping shape: (N_points,)
        # params shape: (N_samples, N_groups)
        
        # log_titrant shape: (1, N_points)
        log_titrant = calc_df["titrant_conc"].values.copy()
        log_titrant[log_titrant == 0] = ZERO_CONC_VALUE
        log_titrant = np.log(log_titrant)[None, :]
        
        # indices shape: (N_points,)
        indices = calc_df["map_theta_group"].values.astype(int)
        
        # Indexed params shape: (N_samples, N_points)
        h_n = hill_n[:, indices]
        l_K = log_hill_K[:, indices]
        t_h = theta_high[:, indices]
        t_l = theta_low[:, indices]
        
        # Calculate theta using Hill equation: (N_samples, N_points)
        # occupancy = 1 / (1 + exp(-hill_n * (log(conc) - log_K)))
        occupancy = 1.0 / (1.0 + np.exp(-h_n * (log_titrant - l_K)))
        theta_samples = t_l + (t_h - t_l) * occupancy
        
        # Calculate quantiles across samples (axis 0)
        for q_name, q_val in q_to_get.items():
            calc_df[q_name] = np.quantile(theta_samples, q_val, axis=0)

        return calc_df.drop(columns=["map_theta_group"])

    def extract_growth_predictions(self,
                                  posteriors,
                                  q_to_get=None):
        """
        Extract predicted ln_cfu values matching the input growth data.

        This method pulls the 'growth_pred' values from the posterior samples
        and maps them back to the original rows in `self.growth_df`.

        Parameters
        ----------
        posteriors : dict or str
            Assumes this is a dictionary of posteriors keying parameters to
            numpy arrays or a path to a .npz file containing posterior samples
            for model parameters.
        q_to_get : dict, optional
            Dictionary mapping output column names to quantile values (between 0 and 1)
            to extract from the posterior samples. If None, a default set of quantiles
            is used (min, lower_95, lower_std, lower_quartile, median, upper_std,
            upper_quartile, upper_95, max).

        Returns
        -------
        pd.DataFrame
            A copy of `self.growth_df` with new columns for the requested
            quantiles of 'ln_cfu_pred'.

        Raises
        ------
        ValueError
            If 'growth_pred' is not in the posterior samples.
            If `q_to_get` is not a dictionary.
        """

        # Load the posterior file
        if isinstance(posteriors,(dict,np.lib.npyio.NpzFile)):
            param_posteriors = posteriors
        else:
            param_posteriors = np.load(posteriors)

        if "growth_pred" not in param_posteriors:
            raise ValueError(
                "'growth_pred' not found in posterior samples. Make sure the "
                "model was run in a way that generates growth predictions."
            )

        # Named quantiles to pull from the posterior distribution
        if q_to_get is None:
            q_to_get = {"min":0.0,
                        "lower_95":0.025,
                        "lower_std":0.159,
                        "lower_quartile":0.25,
                        "median":0.5,
                        "upper_quartile":0.75,
                        "upper_std":0.841,
                        "upper_95":0.975,
                        "max":1.0}

        # make sure q_to_get is a dictionary
        if not isinstance(q_to_get,dict):
            raise ValueError(
                "q_to_get should be a dictionary keying column names to quantiles"
            )

        # Grab the growth_pred tensor
        growth_pred = param_posteriors["growth_pred"]

        # The tensor shape is (num_samples, replicate, time, condition_pre, 
        # condition_sel, titrant_name, titrant_conc, genotype)
        
        # Get the index columns from the growth_df
        rep_idx = self.growth_df["replicate_idx"].values
        time_idx = self.growth_df["time_idx"].values
        pre_idx = self.growth_df["condition_pre_idx"].values
        sel_idx = self.growth_df["condition_sel_idx"].values
        name_idx = self.growth_df["titrant_name_idx"].values
        conc_idx = self.growth_df["titrant_conc_idx"].values
        geno_idx = self.growth_df["genotype_idx"].values

        # Pull predictions for each row across all samples
        # shape: (num_samples, num_rows)
        preds = growth_pred[:, rep_idx, time_idx, pre_idx, sel_idx, name_idx, conc_idx, geno_idx]

        # Calculate quantiles and load into output dataframe
        out_df = self.growth_df.copy()
        for q_name, q_val in q_to_get.items():
            out_df[q_name] = np.quantile(preds, q_val, axis=0)

        return out_df


    def flatten_priors(self):
        """
        Flatten all priors by setting scale and std hyperparameters to large values.
        This effectively turns MAP into MLE.
        """
        
        def _flatten_recursive(obj):
            # If not a flax/standard dataclass, return as is
            if not hasattr(obj, "__dataclass_fields__"):
                return obj
            
            updates = {}
            for field_name in obj.__dataclass_fields__:
                val = getattr(obj, field_name)
                
                # Heuristic for scale/std parameters (Normal, HalfNormal, LogNormal)
                if "scale" in field_name or "std" in field_name:
                    # Special case: Growth model rate parameters are sometimes named 'scale'
                    # but behave as rates in Gamma distributions so they should be small
                    # to imply high uncertainty.
                    if "beta_kappa_scale" in field_name or "rate" in field_name:
                        updates[field_name] = 1e-6
                    else:
                        updates[field_name] = 100.0
                
                # Heuristic for rate parameters (Gamma, Exponential)
                elif "rate" in field_name:
                    updates[field_name] = 1e-6
                
                # Recurse if the value is itself a dataclass
                elif hasattr(val, "__dataclass_fields__"):
                    updates[field_name] = _flatten_recursive(val)
            
            # Apply all updates to this dataclass
            if updates:
                return obj.replace(**updates)
            
            return obj

        self._priors = _flatten_recursive(self._priors)


    @property
    def init_params(self):
        """A dictionary of initial parameter guesses for optimization."""
        return self._init_params

    @property
    def jax_model(self):
        """The top-level JAX model function."""
        return self._jax_model
    
    @property
    def jax_model_guide(self):
        """Guide for the jax function."""
        return self._jax_model_guide

    @property
    def get_batch(self):
        """Get a deterministic batch of data."""
        return get_batch
    
    def get_random_idx(self, batch_key=None, num_batches=1):
        """
        Get a random set of integers corresponding to genotypes for mini-batching.
        The first `num_binding` entries are always the fixed binding genotypes.
        The remaining entries are sampled from the non-binding genotypes.

        Parameters
        ----------
        batch_key : int, optional
            If provided, re-initializes the NumPy random number generator. 
            Must be called once to initialize.
        num_batches : int, optional
            Number of batches of indices to generate. Default is 1.

        Returns
        -------
        jnp.ndarray
            If `num_batches == 1`, returns an array of shape `(batch_size,)`.
            If `num_batches > 1`, returns an array of shape `(num_batches, batch_size)`.
        """

        if batch_key is not None:
            self._batch_rng = np.random.default_rng(batch_key)
            self._batch_idx = np.array(self.data.growth.batch_idx, dtype=int)
            self._batch_choose_from = np.array(self.data.not_binding_idx)
            self._batch_choose_size = self.data.not_binding_batch_size

        if not hasattr(self, "_batch_rng"):
            raise ValueError(
                "get_random_idx must be called with an integer batch key the "
                "first time it is called."
            )

        # Generate a single batch
        if num_batches == 1:
            self._batch_idx[-self._batch_choose_size:] = self._batch_rng.choice(
                self._batch_choose_from,
                self._batch_choose_size,
                replace=False
            )
            return jnp.array(self._batch_idx, dtype=jnp.int32)
        
        # Generate a block of batches
        else:
            block_idx = np.zeros((num_batches, len(self._batch_idx)), dtype=int)
            for i in range(num_batches):
                self._batch_idx[-self._batch_choose_size:] = self._batch_rng.choice(
                    self._batch_choose_from,
                    self._batch_choose_size,
                    replace=False
                )
                block_idx[i, :] = self._batch_idx
                
            return jnp.array(block_idx, dtype=jnp.int32)

    @property
    def data(self):
        """The DataClass Pytree holding all input data."""
        return self._data

    @property
    def priors(self):
        """The PriorsClass Pytree holding all model priors."""
        return self._priors
            
    @property
    def settings(self):
        """A dictionary of the model component names used."""

        return {
            "batch_size":self._batch_size,
            "condition_growth":self._condition_growth,
            "ln_cfu0":self._ln_cfu0,
            "dk_geno":self._dk_geno,
            "activity":self._activity,
            "theta":self._theta,
            "transformation":self._transformation,
            "theta_growth_noise":self._theta_growth_noise,
            "theta_binding_noise":self._theta_binding_noise,
            "spiked_genotypes":self._spiked_genotypes,
        }

    @staticmethod
    def load_config(config_file):
        """
        Load model configuration from a YAML file.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.

        Returns
        -------
        growth_df : str
            Path to the growth data CSV file.
        binding_df : str
            Path to the binding data CSV file.
        settings : dict
            Dictionary of model settings.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Check for required fields
        required_fields = ["growth_df", "binding_df", "settings","tfscreen_version"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if config["tfscreen_version"] != __version__:
            warnings.warn(f"Configuration file version {config['tfscreen_version']} does not match current tfscreen version {__version__}")

        return config["growth_df"], config["binding_df"], config["settings"]

    def write_config(self, 
                     growth_df_path, 
                     binding_df_path, 
                     out_root):
        """
        Write model configuration to a YAML file.

        Parameters
        ----------
        growth_df_path : str
            Path to the growth data CSV file.
        binding_df_path : str
            Path to the binding data CSV file.
        out_root : str
            Root filename for the configuration file ({out_root}_config.yaml).
        """
        config = {
            "tfscreen_version": __version__,
            "growth_df": growth_df_path,
            "binding_df": binding_df_path,
            "settings": self.settings
        }

        with open(f"{out_root}_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

