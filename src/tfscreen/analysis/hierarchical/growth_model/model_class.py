import tfscreen

from tfscreen.util import add_group_columns
from tfscreen.analysis.hierarchical import (
    TensorManager,
    populate_dataclass
)

from tfscreen.analysis.hierarchical.growth_model.model import jax_model
from tfscreen.analysis.hierarchical.growth_model.registry import model_registry

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
    growth_df = tfscreen.util.read_dataframe(growth_df)
    growth_df = tfscreen.genetics.set_categorical_genotype(growth_df,standardize=True)
    growth_df = tfscreen.util.get_scaled_cfu(growth_df,need_columns=["ln_cfu","ln_cfu_std"])

    # make a replicate column if not defined
    if "replicate" not in growth_df.columns:
        growth_df["replicate"] = 1
    
    # check for all required columns
    required = theta_group_cols[:]
    required.extend(treatment_cols)
    required.extend(["ln_cfu","ln_cfu_std","replicate","t_pre","t_sel"])
    tfscreen.util.check_columns(growth_df,required_columns=required)

    # These two maps are used to look up parameters after sampling posteriors
    growth_df = add_group_columns(target_df=growth_df,
                                  group_cols=treatment_cols,
                                  group_name="treatment")

    growth_df = add_group_columns(target_df=growth_df,
                                  group_cols=theta_group_cols,
                                  group_name="map_theta_group")
        
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
    growth_tm.add_pivot_index(tensor_dim_name="condition_sel",cat_column="condition_sel")
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
    binding_df = tfscreen.util.read_dataframe(binding_df)
    binding_df = tfscreen.genetics.set_categorical_genotype(binding_df,
                                                            standardize=True)

    # check for all required columns
    required = theta_group_cols[:]
    required.extend(["theta_obs","theta_std","titrant_conc"])
    tfscreen.util.check_columns(binding_df,required_columns=required)


    cols = ["genotype","titrant_name"]
    binding_seen = binding_df[cols].drop_duplicates().set_index(cols)
    growth_seen = growth_df[cols].drop_duplicates().set_index(cols)
    is_subset = binding_seen.index.isin(growth_seen.index).all()
    if not is_subset:
        raise ValueError(
            "binding_df contains genotype/titrant_name pairs that were not seen "
            "in the growth_df."
        )

    growth_ref = growth_df[cols].drop_duplicates()

    # Merge the binding data into the growth data. This makes sure the binding
    # data has a row for every genotype/titrant_name combination seen in the 
    # growth data. Pairs not seen in the binding data will have `nan` entries.
    binding_df = growth_ref.merge(binding_df,
                                  on=["genotype","titrant_name"],
                                  how="left",
                                  sort=False)
    
    # Reassert that genotype must be categorical after the merge
    binding_df = tfscreen.genetics.set_categorical_genotype(binding_df)

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
                 theta_growth_noise="none",
                 theta_binding_noise="none"):

        self._ln_cfu_df = growth_df
        self._binding_df = binding_df

        self._batch_size = batch_size

        self._condition_growth = condition_growth
        self._ln_cfu0 = ln_cfu0
        self._dk_geno = dk_geno
        self._activity = activity
        self._theta = theta
        self._theta_growth_noise = theta_growth_noise
        self._theta_binding_noise = theta_binding_noise

        self._initialize_data()
        self._initialize_classes()


    def _initialize_data(self):
        """
        Loads, processes, and wrangles all data into JAX Pytrees.

        This method orchestrates the entire data pipeline:
        1.  Calls `_read_growth_df` and `_build_growth_tm` / `_build_growth_theta_tm`.
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
        
        # Get wildtype information
        genotype_idx = self.growth_tm.tensor_dim_names.index("genotype")
        wt_loc = np.where(self.growth_tm.tensor_dim_labels[genotype_idx] == "wt")
    
        wt_info = {"wt_indexes":jnp.array(wt_loc[0], dtype=jnp.int32),
                   "batch_idx":np.arange(self.growth_tm.tensor_shape[-1],dtype=int),
                   "batch_size":self.growth_tm.tensor_shape[-1]}
        
        # Assemble tensors. We draw some from growth_tm and some from 
        # growth_theta_tm    
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

        # scatter_theta tells the theta model caller to return a full-sized
        # growth_tm tensor instead of the smaller growth_theta_tm-sized tensor
        other_data = {"scatter_theta":1}

        # Grab the titrant concentration and log_titrant_conc (1D array from 
        # the tensor labels along dimension 6)
        idx = np.where(np.array(self.growth_tm.tensor_dim_names) == "titrant_conc")[0][0]
        titrant_conc = np.array(self.growth_tm.tensor_dim_labels[idx])
        log_titrant_conc = titrant_conc.copy()
        log_titrant_conc[log_titrant_conc == 0] = ZERO_CONC_VALUE
        log_titrant_conc = np.log(log_titrant_conc)
        
        other_data["titrant_conc"] = titrant_conc
        other_data["log_titrant_conc"] = log_titrant_conc

        # Populate a GrowthData flax dataclass with all keys in `sources`. 
        growth_dataclass = populate_dataclass(GrowthData,
                                              sources=[tensors,
                                                       sizes,
                                                       wt_info,
                                                       other_data])
        
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
        other_data = {"scatter_theta":0,
                      "batch_idx":np.arange(self.growth_tm.tensor_shape[-1],dtype=int),
                      "batch_size":self.binding_tm.tensor_shape[2]}

        # Grab the titrant concentration and log_titrant_conc (1D array from 
        # the tensor labels along dimension 6)
        idx = np.where(np.array(self.binding_tm.tensor_dim_names) == "titrant_conc")[0][0]
        titrant_conc = np.array(self.binding_tm.tensor_dim_labels[idx])
        log_titrant_conc = titrant_conc.copy()
        log_titrant_conc[log_titrant_conc == 0] = ZERO_CONC_VALUE
        log_titrant_conc = np.log(log_titrant_conc)

        other_data["titrant_conc"] = titrant_conc
        other_data["log_titrant_conc"] = log_titrant_conc


        # Populate a BindingData flax dataclass with all keys in `sources`
        binding_dataclass = populate_dataclass(BindingData,
                                               sources=[self.binding_tm.tensors,
                                                        sizes,
                                                        other_data])
        
        # ---------------------------------------------------------------------
        # combined dataclass
        
        source_data = {"growth":growth_dataclass,
                       "binding":binding_dataclass,
                       "num_genotype":self.growth_tm.tensor_shape[-1]}

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
            `MODEL_COMPONENT_NAMES`.
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
                                            component_module.run_model)
                guide_control_kwargs[key] = (component_module.guide, 
                                             component_module.run_model)
            else:
                main_control_kwargs[key] = component_module.define_model
                guide_control_kwargs[key] = component_module.guide
    
        # Record the batch size
        if self._batch_size is None:
            self._batch_size = self.growth_tm.tensor_shape[-1]
        if self._batch_size > self.growth_tm.tensor_shape[-1]:
            self._batch_size = self.growth_tm.tensor_shape[-1]
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
        growth_priors = GrowthPriors(**(priors_class_kwargs["growth"]))
        binding_priors = BindingPriors(**(priors_class_kwargs["binding"]))
        priors = PriorsClass(theta=priors_class_kwargs["theta"]["theta"],
                             growth=growth_priors,
                             binding=binding_priors)
    
        self._priors = priors
        self._init_params = init_params

    def define_deterministic_model(self,batch_idx):
        """
        Create a jax model that uses specific genotype indices.
        """

        batch_idx = jnp.array(batch_idx,dtype=jnp.int32)

        # Temporarily store batch_idx the control kwargs
        self._main_control_kwargs["batch_idx"] = batch_idx

        # Bake in the new deterministic model
        self._jax_model_deterministic = partial(jax_model, **self._main_control_kwargs)

        # Pop the batch_idx back out of control kwargs (we use this to avoid 
        # copying the control kwargs while also not contaminating it with our 
        # our batch_idx)
        self._main_control_kwargs.pop("batch_idx")

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
        if isinstance(posteriors,dict):
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
                        "upper_std":0.659,
                        "upper_quartile":0.75,
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

        params = {}
        for kwargs in extract:
            params.update(_extract_param_est(param_posteriors=param_posteriors,
                                             q_to_get=q_to_get,
                                             **kwargs))

        return params


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
    def jax_model_deterministic(self):
        """jax model that does calculation for a specific set of genotype indexes"""
        if not hasattr(self,"_jax_model_deterministic"):
            raise ValueError(
                "jax_model_deterministic is only defined after define_deterministic_model "
                "is called."
            )
        
        return self._jax_model_deterministic


    @property
    def data(self):
        """The DataClass Pytree holding all input data."""
        return self._data

    @property
    def priors(self):
        """The PriorsClass Pytree holding all model priors."""
        return self._priors

    @property
    def control(self):
        """The ControlClass Pytree holding model-switching integers."""
        return self._control
            
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
            "theta_growth_noise":self._theta_growth_noise,
            "theta_binding_noise":self._theta_binding_noise,
        }

