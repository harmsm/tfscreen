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


from jax import numpy as jnp
import pandas as pd
import numpy as np


def _read_growth_df(growth_df,
                    theta_group_cols=None,
                    treatment_cols=None):
    """
    Read a dataframe holding growth data and process for the model. 

    Parameters
    ----------
    growth_df : pd.DataFrame or path
        dataframe or spreadsheet holding growth data. must have columns 
        ln_cfu,ln_cfu_std,t_pre, and t_sel. it also must have all columns
        in theta_group_cols and treatment cols. 
    theta_group_cols : list, optional
        make groups of rows that should use the same theta_model parameters.
        If not specified, uses genotype,titrant_name.
    treatment_cols: list, optional
        make groups of rows that have the same treatment and should thus 
        grow the same way. If not specified, uses condition_pre,condition_sel,
        titrant_name,titrant_conc. 

    Returns
    -------
    growth_df : pd.DataFrame
        processed growth dataframe
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
    
    return growth_df

def _build_growth_tm(growth_df):

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
    growth_tm.add_data_tensor("ln_cfu")
    growth_tm.add_data_tensor("ln_cfu_std")
    growth_tm.add_data_tensor("t_pre")
    growth_tm.add_data_tensor("t_sel")
    growth_tm.add_data_tensor("titrant_conc")
    
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

def _build_growth_theta_tm(growth_df,
                           theta_group_cols=None):
    """
    Creates tensors that have dimensions [titrant_name,titrant_conc,genotype]. 
    These will be smaller than full growth tensors because they only have 
    unique combinations of these values. The tensors this creates are:
    + 'map_theta_group': which maps back to the unique titrant_name/genotype
       combos that use unique parameters
    + 'titrant_conc': titrant concentration
    """

    # Get default theta_group columns
    if theta_group_cols is None:
        theta_group_cols = ["genotype","titrant_name"]

    # growth theta df has all unique theta_group and titrant_conc combos
    growth_theta_df = (growth_df
                       .drop_duplicates(["titrant_name","titrant_conc","genotype"])
                       .copy())
    
    # Create a titrant_conc column to use as a pivot for the tensor. The pivot
    # consumes the column. This move allows us to preserve titrant_conc and use
    # its values to populate a tensor.
    growth_theta_df["pivot_titrant_conc"] = growth_df["titrant_conc"]

    # Create tensor manager
    tm = TensorManager(growth_theta_df)

    tm.add_pivot_index(tensor_dim_name="titrant_name",cat_column="titrant_name")
    tm.add_pivot_index(tensor_dim_name="titrant_conc",cat_column="pivot_titrant_conc")
    tm.add_pivot_index(tensor_dim_name="genotype",cat_column="genotype")

    # map_theta_group will allow us to map back from these local values to the 
    # parameter values
    tm.add_data_tensor("map_theta_group",dtype=int)

    # This will give us the array of titrant concs
    tm.add_data_tensor("titrant_conc")

    # Create tensors
    tm.create_tensors()

    return tm

def _read_binding_df(binding_df,
                     existing_df=None,
                     theta_group_cols=None):

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
    
    # theta_titrant_conc is the name of the titrant_conc accessed in 
    # theta.run_model
    binding_df["theta_titrant_conc"] = binding_df["titrant_conc"]
        
    return binding_df

def _build_binding_tm(binding_df):
    """
    Creates tensors that have dimensions [titrant_name,titrant_conc,genotype].
    The tensors this creates are:
    + 'map_theta_group': which maps back to the unique titrant_name/genotype
       combos that use unique parameters
    + 'titrant_conc': titrant concentration
    + 'theta_obs': observed theta
    + 'theta_std': standard error on the observed theta
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
    binding_tm.add_data_tensor("theta_obs")
    binding_tm.add_data_tensor("theta_std")
    binding_tm.add_data_tensor("map_theta_group",dtype=int)
    binding_tm.add_data_tensor("titrant_conc")

    # Build tenors
    binding_tm.create_tensors()
    
    return binding_tm

def _get_wt_info(tm):

    mask = tm.tensor_dim_labels[-1] != "wt"
    not_wt_mask = jnp.arange(len(mask),dtype=int)[mask]
    
    wt_index = jnp.arange(len(mask),dtype=int)[~mask]
    if len(wt_index) != 1:
        raise ValueError(
            f"Exactly one 'wt' entry is allowed in tensor dimension "
            f"{tm.tensor_dim_names[-1]}, but we found {len(wt_index)}."
        )
    wt_index = wt_index[0]

    num_not_wt = len(not_wt_mask)
    
    out = {"wt_index":wt_index,
           "not_wt_mask":not_wt_mask,
           "num_not_wt":num_not_wt}

    return out

    
class ModelClass:

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

        from_growth_theta_tm = ["titrant_conc","map_theta_group"]
        for k in from_growth_theta_tm:
            tensors[k] = self.growth_theta_tm.tensors[k]

        # build dictionaries of sizes. Most are from self.growth_tm (except
        # num_theta_group)
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
                 "num_titrant_conc":self.binding_tm.tensor_shape[1]}
        other_data = {"scatter_theta":0,
                      "obs_mask":jnp.ones(self.growth_tm.tensor_shape[-1],dtype=bool)}

        # Populate a BindingData flax dataclass with all keys in `sources`
        binding_dataclass = populate_dataclass(BindingData,
                                               sources=[self.binding_tm.tensors,
                                                        sizes,
                                                        other_data])
        

        # ---------------------------------------------------------------------
        # combined dataclass

        # Build a map for batch sampling on binding data. 
        N = np.max(self.growth_df["map_theta_group"])
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
                init_params.update(component.get_guesses(name=key,
                                                         data=self._data.binding))
            else:
                init_params.update(component.get_guesses(name=key,
                                                         data=self._data.growth))
    
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
        return self._init_params

    @property
    def jax_model(self):
        return jax_model

    @property
    def data(self):
        return self._data

    @property
    def priors(self):
        return self._priors

    @property
    def control(self):
        return self._control
        
    @property
    def sample_batch(self):
        return sample_batch
    
    @property
    def deterministic_batch(self):
        return deterministic_batch

    @property
    def settings(self):

        return {
            "condition_growth":self._condition_growth,
            "ln_cfu0":self._ln_cfu0,
            "dk_geno":self._dk_geno,
            "activity":self._activity,
            "theta":self._theta,
            "theta_growth_noise":self._theta_growth_noise,
            "theta_binding_noise":self._theta_binding_noise,
        }

