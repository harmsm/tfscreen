import tfscreen

from tfscreen.util import add_group_columns
from tfscreen.analysis.hierarchical import (
    TensorManager,
    populate_dataclass
)

from tfscreen.analysis.hierarchical.growth_model.model import (
    jax_model,
    MODEL_COMPONENT_NAMES
)

from tfscreen.analysis.hierarchical.growth_model.batch import (
    sample_batch,
    deterministic_batch
)

from tfscreen.analysis.hierarchical.growth_model.data_class import (
    DataClass,
    BindingData,
    GrowthData,
    
    ControlClass,

    PriorsClass,
    GrowthPriors,
    BindingPriors
)

import jax
from jax import numpy as jnp
import pandas as pd
import numpy as np

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

    growth_df = add_group_columns(target_df=growth_df,
                                  group_cols=treatment_cols,
                                  group_name="treatment")

    growth_df = add_group_columns(target_df=growth_df,
                                  group_cols=theta_group_cols,
                                  group_name="map_theta_group")
    
    # Create a log_titrant_conc column, replacing 0 with 1000x less than lowest
    # non-zero value. 
    titrant_conc = np.array(growth_df["titrant_conc"],dtype=float)
    titrant_conc[titrant_conc == 0] = ZERO_CONC_VALUE
    growth_df["log_titrant_conc"] = np.log(titrant_conc)
    
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

    # Create tensor manager for construction of growth experiment tensors
    growth_tm = TensorManager(growth_df)
    
    # Define that we want a 4D tensor (replicate,time,treatment,genotype)
    growth_tm.add_pivot_index(tensor_dim_name="replicate",cat_column="replicate")
    growth_tm.add_ranked_pivot_index(tensor_dim_name="time",
                                     rank_column="t_sel",
                                     select_cols=['replicate','genotype','treatment'])
    growth_tm.add_pivot_index(tensor_dim_name="treatment",cat_column="treatment_tuple")
    growth_tm.add_pivot_index(tensor_dim_name="genotype",cat_column="genotype")
    
    # Register that we want tensors from these data columns
    growth_tm.add_data_tensor("ln_cfu",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("ln_cfu_std",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("t_pre",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("t_sel",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("titrant_conc",dtype=FLOAT_DTYPE)
    growth_tm.add_data_tensor("log_titrant_conc",dtype=FLOAT_DTYPE)
    
    # These calls will create full-sized integer tensors for parameter 
    # mapping. For example, ln_cfu0 should have a unique parameter for each 
    # replicate, condition_pre, and genotype. Each unique combination of 
    # those columns will be assigned an integer index, then a tensor 
    # created where the element declares which ln_cfu0 parameter should be 
    # applied at that tensor coordinate. 
    growth_tm.add_map_tensor(["replicate","condition_pre","genotype"],
                             name="ln_cfu0")
    growth_tm.add_map_tensor("genotype",name="genotype")
    growth_tm.add_map_tensor("genotype",name="activity")
    growth_tm.add_map_tensor("genotype",name="dk_geno")
    growth_tm.add_map_tensor(["titrant_name","titrant_conc","genotype"],name="theta")

    # We already created this map above to integrate with the binding data. It's
    # living in a data column, so grab it to create this map. 
    growth_tm.add_data_tensor("map_theta_group",dtype=int) 
    
    # By calling with `select_pool_cols`, we pool names of conditions on 
    # condition_pre and condition_sel. Basically this does the operation 
    # unique(replicate + condition_pre OR replicate + condition_sel). This
    # will create map_condition_pre and map_condition_sel. 
    growth_tm.add_map_tensor(select_cols=["replicate"],
                             select_pool_cols=["condition_pre","condition_sel"],
                             name="condition")

    growth_tm.add_map_tensor(["titrant_name","titrant_conc"],name="titrant")
    growth_tm.add_map_tensor("replicate",name="replicate")

    # Do final tensor creation
    growth_tm.create_tensors()

    return growth_tm

def _build_growth_theta_tm(growth_df):
    """
    Builds a TensorManager for the growth-derived theta data.

    This function creates a smaller, dense 3D tensor containing the unique
    values needed to calculate theta from the growth data. The final tensor
    shape is:
    (titrant_name, titrant_conc, genotype)

    The resulting tensors (e.g., 'titrant_conc', 'map_theta_group') are
    used to populate the `GrowthData` dataclass, ensuring a compact
    representation for theta calculations.

    Parameters
    ----------
    growth_df : pd.DataFrame
        The processed growth DataFrame.

    Returns
    -------
    tfscreen.analysis.hierarchical.TensorManager
        A processed TensorManager instance after `create_tensors()`
        has been called.
    """

    # growth theta df has all unique theta_group and titrant_conc combos
    growth_theta_df = (growth_df
                       .drop_duplicates(["titrant_name","titrant_conc","genotype"])
                       .copy())
    
    # Create a titrant_conc column to use as a pivot for the tensor. The pivot
    # consumes the column. This move allows us to preserve titrant_conc and use
    # its values to populate a tensor.
    growth_theta_df["pivot_titrant_conc"] = growth_theta_df["titrant_conc"]

    # Create tensor manager
    tm = TensorManager(growth_theta_df)

    tm.add_pivot_index(tensor_dim_name="titrant_name",cat_column="titrant_name")
    tm.add_pivot_index(tensor_dim_name="titrant_conc",cat_column="pivot_titrant_conc")
    tm.add_pivot_index(tensor_dim_name="genotype",cat_column="genotype")

    # map_theta_group will allow us to map back from these local values to the 
    # parameter values
    tm.add_data_tensor("map_theta_group",dtype=int)

    # This will give us the array of titrant concs
    tm.add_data_tensor("titrant_conc",dtype=FLOAT_DTYPE)
    tm.add_data_tensor("log_titrant_conc",dtype=FLOAT_DTYPE)

    # Create tensors
    tm.create_tensors()

    return tm

def _read_binding_df(binding_df,
                     existing_df=None,
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
                                                            standardize=True,
                                                            sort=True)

    # check for all required columns
    required = theta_group_cols[:]
    required.extend(["theta_obs","theta_std","titrant_conc"])
    tfscreen.util.check_columns(binding_df,required_columns=required)

        
    # This map will map back to the theta groups in existing_df, letting us
    # grab the same theta_groups from this binding_df and whatever was in 
    # existing_df. 
    binding_df = add_group_columns(binding_df,
                                   group_cols=theta_group_cols,
                                   group_name="map_theta_group",
                                   existing_df=existing_df)
    
    # Create a log_titrant_conc column, replacing 0 ZERO_CONC_VALUE
    titrant_conc = np.array(binding_df["titrant_conc"],dtype=float)
    titrant_conc[titrant_conc == 0] = ZERO_CONC_VALUE
    binding_df["log_titrant_conc"] = np.log(titrant_conc)

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
    binding_tm = tfscreen.analysis.hierarchical.TensorManager(binding_df)

    # Add pivots on titrant_conc and theta_group to create [num_titrant,num_theta_group]
    # output tensors
    binding_tm.add_pivot_index(tensor_dim_name="titrant_name",cat_column="titrant_name")
    binding_tm.add_pivot_index(tensor_dim_name="titrant_conc",cat_column="pivot_titrant_conc")
    binding_tm.add_pivot_index(tensor_dim_name="genotype",cat_column="genotype")

    # add data columns 
    binding_tm.add_data_tensor("theta_obs",dtype=FLOAT_DTYPE)
    binding_tm.add_data_tensor("theta_std",dtype=FLOAT_DTYPE)
    binding_tm.add_data_tensor("map_theta_group",dtype=int)
    binding_tm.add_data_tensor("titrant_conc",dtype=FLOAT_DTYPE)
    binding_tm.add_data_tensor("log_titrant_conc",dtype=FLOAT_DTYPE)

    # Build tenors
    binding_tm.create_tensors()
    
    return binding_tm

def _get_wt_info(tm):
    """
    Extracts wild-type index and non-wild-type mask from a TensorManager.

    This function inspects the 'genotype' dimension of the TensorManager
    to find the integer index corresponding to the 'wt' genotype and
    a mask for all other genotypes.

    Parameters
    ----------
    tm : tfscreen.analysis.hierarchical.TensorManager
        A TensorManager that has a 'genotype' dimension.

    Returns
    -------
    dict
        A dictionary with keys "wt_index", "not_wt_mask", and "num_not_wt".

    Raises
    ------
    ValueError
        If the 'genotype' dimension is not found, or if exactly one
        'wt' entry is not present.
    """

    try:
        genotype_dim_idx = tm.tensor_dim_names.index("genotype")
    except ValueError:
        raise ValueError(
            "Could not find 'genotype' in tensor dimension names."
        )

    mask = tm.tensor_dim_labels[genotype_dim_idx] != "wt"
    not_wt_mask = jnp.arange(len(mask),dtype=int)[mask]
    
    wt_index = jnp.arange(len(mask),dtype=int)[~mask]
    if len(wt_index) != 1:
        raise ValueError(
            f"Exactly one 'wt' entry is allowed in tensor dimension "
            f"{tm.tensor_dim_names[genotype_dim_idx]}, but we found {len(wt_index)}."
        )
    wt_index = wt_index[0]

    num_not_wt = len(not_wt_mask)
    
    out = {"wt_index":wt_index,
           "not_wt_mask":not_wt_mask,
           "num_not_wt":num_not_wt}

    return out

    
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
    ln_cfu_df : pd.DataFrame or str
        DataFrame or path to file with growth data.
    theta_obs_df : pd.DataFrame or str
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
                 ln_cfu_df,
                 theta_obs_df,
                 condition_growth="hierarchical",
                 ln_cfu0="hierarchical",
                 dk_geno="hierarchical",
                 activity="fixed",
                 theta="hill",
                 theta_growth_noise="beta",
                 theta_binding_noise="beta"):

        self._ln_cfu_df = ln_cfu_df
        self._theta_obs_df = theta_obs_df

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
        4.  Generates the `growth_to_binding_idx` map.
        5.  Uses `populate_dataclass` to build the `GrowthData`, `BindingData`,
            and final `DataClass` Pytrees.
        6.  Sets the final `self._data` attribute.
        """
        # ---------------------------------------------------------------------
        # growth dataclass

        # Load in growth data, creating two blocks of tensors. One holds the
        # growth (replicate,time,condition,genotype) data. The other holds 
        # the theta (titrant_conc,theta_group) tensor. 
        self.growth_df = _read_growth_df(self._ln_cfu_df)
        self.growth_tm = _build_growth_tm(self.growth_df)
        self.growth_theta_tm = _build_growth_theta_tm(self.growth_df)
        
        # Get wildtype information
        wt_info = _get_wt_info(self.growth_tm)
        
        # Assemble tensors. We draw some from growth_tm and some from 
        # growth_theta_tm        
        tensors = {}
        
        from_growth_tm = ["ln_cfu","ln_cfu_std","t_pre","t_sel",
                          "map_ln_cfu0","map_condition_pre","map_condition_sel",
                          "map_genotype","map_theta","good_mask"]
        for k in from_growth_tm:
            tensors[k] = self.growth_tm.tensors[k]

        from_growth_theta_tm = ["titrant_conc","log_titrant_conc","map_theta_group"]
        for k in from_growth_theta_tm:
            tensors[k] = self.growth_theta_tm.tensors[k]

        # build dictionaries of sizes. 
        sizes = dict([(f"num_{s}",self.growth_tm.map_sizes[s]) for s in self.growth_tm.map_sizes])
        sizes["num_time"] = self.growth_tm.tensor_shape[1]
        sizes["num_treatment"] = self.growth_tm.tensor_shape[2]
        sizes["num_titrant_name"] = self.growth_theta_tm.tensor_shape[0]
        sizes["num_titrant_conc"] = self.growth_theta_tm.tensor_shape[1]
        
        # Other data for control. scatter_theta tells the theta model caller 
        # to return a full-sized growth_tm tensor instead of the smaller 
        # growth_theta_tm-sized tensor
        other_data = {"scatter_theta":1}

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
        self.binding_df = _read_binding_df(self._theta_obs_df,
                                           existing_df=self.growth_tm.df)
        self.binding_tm = _build_binding_tm(self.binding_df)

        # Grab the sizes
        sizes = {"num_titrant_name":self.binding_tm.tensor_shape[0],
                 "num_titrant_conc":self.binding_tm.tensor_shape[1],
                 "num_genotype":self.binding_tm.tensor_shape[2]}
        other_data = {"scatter_theta":0}

        # Populate a BindingData flax dataclass with all keys in `sources`
        binding_dataclass = populate_dataclass(BindingData,
                                               sources=[self.binding_tm.tensors,
                                                        sizes,
                                                        other_data])
        
        # ---------------------------------------------------------------------
        # combined dataclass

        # Build a map for batch sampling on binding data. 
        N = np.max(self.growth_df["map_theta_group"]) + 1
        binding_values = pd.unique(self.binding_df["map_theta_group"]).astype(int)
        growth_to_binding_idx = np.full(N,-1,dtype=int)
        growth_to_binding_idx[binding_values] = np.arange(len(binding_values),dtype=int)
        growth_to_binding_idx = jnp.array(growth_to_binding_idx,dtype=int)
        
        source_data = {"growth":growth_dataclass,
                       "binding":binding_dataclass,
                       "num_genotype":self.growth_tm.tensor_shape[-1],
                       "growth_to_binding_idx":growth_to_binding_idx}

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
        # MODEL_COMPONENT_NAMES. The second value is the model name passed by 
        # the user/caller. It should match one of the possible model types for
        # the given model component in MODEL_COMPONENT_NAMES. For example, 
        # for "dk_geno", `dk_geno_model` could be 'fixed' or 'hierarchical'. 
        # The last value determines whether this is used to initialize the 
        # data.growth and data.binding data. 
        load_map = [("condition_growth",self._condition_growth,"growth"),
                    ("ln_cfu0",self._ln_cfu0,"growth"),
                    ("dk_geno",self._dk_geno,"growth"),
                    ("activity",self._activity,"growth"),
                    ("theta",self._theta,"theta"),
                    ("theta_growth_noise",self._theta_growth_noise,"growth"),
                    ("theta_binding_noise",self._theta_binding_noise,"binding")]
        
        control_class_kwargs = {}
        priors_class_kwargs = {"growth":{},"binding":{},"theta":{}}
        init_params = {}
        for to_load in load_map:
    
            key, value, prior_group = to_load
    
            if value not in MODEL_COMPONENT_NAMES[key]:
                raise ValueError(
                    f"{key} '{value}' not recognized. "
                    f"It should be one of: {list(MODEL_COMPONENT_NAMES[key].keys())}"
                )
    
            component_int, component = MODEL_COMPONENT_NAMES[key][value]
    
            # Integer model selector
            control_class_kwargs[key] = component_int
    
            # Record priors
            priors_class_kwargs[prior_group][key] = component.get_priors()
    
            # Record guesses
            if prior_group == "binding":
                guesses = component.get_guesses(name=key,
                                                data=self._data.binding)
            else:
                guesses = component.get_guesses(name=key,
                                                data=self._data.growth)

            init_params.update(guesses)
    
        # Build control class
        control = ControlClass(**control_class_kwargs)

        # Build priors class
        growth_priors = GrowthPriors(**(priors_class_kwargs["growth"]))
        binding_priors = BindingPriors(**(priors_class_kwargs["binding"]))
        priors = PriorsClass(theta=priors_class_kwargs["theta"]["theta"],
                             growth=growth_priors,
                             binding=binding_priors)
    
        self._control = control
        self._priors = priors
        self._init_params = init_params
    
    @property
    def init_params(self):
        """A dictionary of initial parameter guesses for optimization."""
        return self._init_params

    @property
    def jax_model(self):
        """The top-level JAX model function."""
        return jax_model

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
    def sample_batch(self):
        """The batch sampling function (from `batch.py`)."""
        return sample_batch
    
    @property
    def deterministic_batch(self):
        """The deterministic batch-creation function (from `batch.py`)."""
        return deterministic_batch

    @property
    def settings(self):
        """A dictionary of the model component names used."""

        return {
            "condition_growth":self._condition_growth,
            "ln_cfu0":self._ln_cfu0,
            "dk_geno":self._dk_geno,
            "activity":self._activity,
            "theta":self._theta,
            "theta_growth_noise":self._theta_growth_noise,
            "theta_binding_noise":self._theta_binding_noise,
        }

