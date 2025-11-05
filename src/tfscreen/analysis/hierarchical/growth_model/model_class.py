import tfscreen

from .model import (
    jax_model
)

from . import components

from .factory import (
    sample_batch,
    deterministic_batch
)

from .data_class import DataClass
from tfscreen.analysis.hierarchical import TensorManager

from jax import numpy as jnp

class ModelClass:

    # These must match the static arguments in the model
    static_arg_names = ("use_growth_indep",
                        "use_fixed_dk_geno",
                        "use_fixed_activity",
                        "use_horseshoe_activity",
                        "use_categorical_theta")

    def __init__(self,
                 df,
                 measured_hill,
                 use_indep_growth=False,
                 use_fixed_dk_geno=False,
                 use_fixed_activity=False,
                 use_categorical_theta=False,
                 use_horseshoe_activity=True):
        
        # Read dataframe
        df = tfscreen.util.read_dataframe(df)

        # read measured hill
        self.measured_hill = self._read_measured_hill(measured_hill)
        self.num_measured_hill_theta = len(self.measured_hill)

        # Control parameters
        self.use_indep_growth = bool(use_indep_growth)
        self.use_fixed_dk_geno = bool(use_fixed_dk_geno)
        self.use_fixed_activity = bool(use_fixed_activity)
        self.use_categorical_theta = bool(use_categorical_theta)
        self.use_horseshoe_activity = bool(use_horseshoe_activity)

        # Creates .tm and .df attributes.
        self._build_tensors(df)

        # flax dataclass with all required fit data
        self.data = self.tm.build_dataclass(DataClass)

        self._get_components()
        self._get_guesses()
        self._get_hyperparameters()
        self._get_priors()

        self._jax_model_kwargs = {}
        self._jax_model_kwargs.update(**self._priors)
        self._init_params = self._guesses

    def _build_tensors(self,df):

        self.tm = TensorManager(df)
        self.df = self.tm.df

        # These maps should create all tensors we need for all possible growth
        # model combos
        self.tm.add_parameter_map("ln_cfu0",
                                  ["replicate","condition_pre","genotype"])
        self.tm.add_parameter_map("activity",
                                  ["genotype"])
        self.tm.add_parameter_map("dk_geno",
                                  ["genotype"])
        self.tm.add_parameter_map("theta",
                                  ["genotype","titrant_name","titrant_conc"])
        self.tm.add_parameter_map("titrant",
                                  ["titrant_name","titrant_conc"])
        self.tm.add_parameter_map("theta_group",
                                  ["genotype","titrant_name"])
        self.tm.add_parameter_map("replicate",
                                  ["replicate"])

        self.tm.add_condition_map()
        self.tm.add_data_tensor("titrant_conc")

        self.tm.create_tensors()

    def _get_components(self):

        # Growth
        if self.use_indep_growth:
            self._growth_compontent = components.growth_indep
        else:
            self._growth_compontent = components.growth_hyper

        # ln_cfu0
        self._ln_cfu0_component = components.ln_cfu0

        # dk_geno
        if self.use_fixed_dk_geno:
            self._dk_geno_component = components.dk_geno_fixed
        else:
            self._dk_geno_component = components.dk_geno

        # activity
        if self.use_fixed_activity:
            self._activity_component = components.activity_fixed
        else:
            if self.use_horseshoe_activity:
                self._activity_component = components.activity_horseshoe
            else:
                self._activity_component = components.activity

        # theta
        if self.use_categorical_theta:
            self._theta_component = components.theta_cat
        else:
            self._theta_component = components.theta_hill


    def _get_guesses(self):

        guesses = {}
        guesses.update(self._growth_compontent.get_guesses("growth",self.data))
        guesses.update(self._ln_cfu0_component.get_guesses("ln_cfu0",self.data))
        guesses.update(self._dk_geno_component.get_guesses("dk_geno",self.data))
        guesses.update(self._activity_component.get_guesses("activity",self.data))
        guesses.update(self._theta_component.get_guesses("theta",
                                                         self.data,
                                                         self.num_measured_hill_theta))
        self._guesses = guesses

    def _get_hyperparameters(self):

        hyperparameters = {}
        hyperparameters["growth"] = self._growth_compontent.get_hyperparameters()
        hyperparameters["ln_cfu0"] = self._ln_cfu0_component.get_hyperparameters()
        hyperparameters["dk_geno"] = self._dk_geno_component.get_hyperparameters()
        hyperparameters["activity"] = self._activity_component.get_hyperparameters()
        hyperparameters["theta"] = self._theta_component.get_hyperparameters(
            data=self.data,
            num_measured_theta_hill=self.num_measured_hill_theta
        )

        self._hyperparameters = hyperparameters


    def _get_priors(self):

        # -------------------------------------------------------------------------
        # Load in measured values.

        measured = []
        unmeasured = list(self.tm.parameter_indexers["theta_group"].values())
        logit_theta_min = []
        logit_theta_max = []
        log_hill_K = []
        log_hill_n = []

        for i, key in enumerate(self.measured_hill):

            # Set indices for measured and unmeasured values
            idx = self.tm.parameter_indexers["theta_group"][key]
            measured.append(idx)
            unmeasured.remove(idx)

            # Grab measured values
            logit_theta_min.append(self.measured_hill[key]["logit_theta_min"])
            logit_theta_max.append(self.measured_hill[key]["logit_theta_max"])
            log_hill_K.append(self.measured_hill[key]["log_hill_K"])
            log_hill_n.append(self.measured_hill[key]["log_hill_n"])

        self._hyperparameters["theta"]["measured_theta_hill_indices"] = jnp.asarray(measured,dtype=int)
        self._hyperparameters["theta"]["unmeasured_theta_hill_indices"] = jnp.asarray(unmeasured,dtype=int)
        self._hyperparameters["theta"]["measured_logit_theta_min_loc"] = jnp.asarray(logit_theta_min,dtype=float)
        self._hyperparameters["theta"]["measured_logit_theta_max_loc"] = jnp.asarray(logit_theta_max,dtype=float)
        self._hyperparameters["theta"]["measured_log_theta_hill_K_loc"] = jnp.asarray(log_hill_K,dtype=float)
        self._hyperparameters["theta"]["measured_log_theta_hill_n_loc"] = jnp.asarray(log_hill_n,dtype=float)

        self._guesses["theta_measured_logit_theta_min"] = jnp.asarray(logit_theta_min,dtype=float)
        self._guesses["theta_measured_logit_theta_max"] = jnp.asarray(logit_theta_max,dtype=float)
        self._guesses["theta_measured_log_hill_K"] = jnp.asarray(log_hill_K,dtype=float)
        self._guesses["theta_measured_log_hill_n"] = jnp.asarray(log_hill_n,dtype=float)

        # Build final priors
        growth_priors = self._growth_compontent.ModelPriors(**self._hyperparameters["growth"])
        ln_cfu0_priors = self._ln_cfu0_component.ModelPriors(**self._hyperparameters["ln_cfu0"])
        dk_geno_priors = self._dk_geno_component.ModelPriors(**self._hyperparameters["dk_geno"])
        activity_priors = self._activity_component.ModelPriors(**self._hyperparameters["activity"])
        theta_priors = self._theta_component.ModelPriors(**self._hyperparameters["theta"])

        # priors
        priors = dict(growth_priors=growth_priors,
                      ln_cfu0_priors=ln_cfu0_priors,
                      dk_geno_priors=dk_geno_priors,
                      activity_priors=activity_priors,
                      theta_priors=theta_priors)

        self._priors = priors

    def _read_measured_hill(self,measured_hill):
        """
        Read measured hill coefficients from a dataframe.
        """

        # already a dict
        if isinstance(measured_hill,dict):
            return measured_hill
        
        # Read measured_hill dataframe
        measured_df = tfscreen.util.read_dataframe(measured_hill)
        
        # Make sure all required columns are present
        required_columns = ["genotype","titrant_name",
                            "log_hill_K","log_hill_n",
                            "logit_theta_min","logit_theta_max"]
        tfscreen.util.check_columns(measured_df,required_columns)

        # Build a dictionary like {(genotype,titrant_name):{param1:value,param2:value}}
        keys = list(zip(measured_df["genotype"],measured_df["titrant_name"]))
        columns_to_get = ["log_hill_K","log_hill_n","logit_theta_min","logit_theta_max"]
        list_of_dicts = measured_df[columns_to_get].to_dict(orient='records')

        measured_hill_dict = dict(zip(keys,list_of_dicts))
        
        return measured_hill_dict

    @property
    def init_params(self):
        return self._init_params

    @property
    def jax_model_kwargs(self):
        return self._jax_model_kwargs

    @property
    def jax_model(self):
        return jax_model
        
    @property
    def sample_batch(self):
        return sample_batch
    
    @property
    def deterministic_batch(self):
        return deterministic_batch