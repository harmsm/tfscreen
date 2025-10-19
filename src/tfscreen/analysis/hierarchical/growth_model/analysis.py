from . import model, guide, preprocess, initialize, dataclasses

import jax.numpy as jnp

class GrowthAnalysis:
    def __init__(self, df, **kwargs):
        
        self.df, self.model_data, self.param_map = preprocess.df_to_tensors(df, **kwargs)
        self.model_priors = self.get_default_priors()

    @property
    def model(self):
        return model.growth_model

    @property
    def guide(self):
        return guide.AutoNormal(model.growth_model) 

    def get_default_priors(self):

        wt_theta_est = jnp.asarray([1,0.000251286,0.999960216,0.996037239,0.96543106,0.715382967,0.218308477,0.024518657])
        wt_theta_std = jnp.asarray([0.02,5.02572E-06,0.019999204,0.019920745,0.019308621,0.014307659,0.00436617,0.000490373])

        return dataclasses.GrowthModelPriors(
            ln_cfu0_hyper_loc_loc=0,
            ln_cfu0_hyper_loc_scale=1,
            ln_cfu0_hyper_scale_loc=1,
            A_hyper_loc_loc=1,
            A_hyper_loc_scale=0.1,
            A_hyper_scale_loc=0.1,
            dk_geno_hyper_shift_loc=0,
            dk_geno_hyper_shift_scale=0.1,
            dk_geno_hyper_loc_loc=0,
            dk_geno_hyper_loc_scale=0.1,
            dk_geno_hyper_scale_loc=0.1,
            wt_theta_loc=wt_theta_est,
            wt_theta_scale=wt_theta_std,
            log_alpha_hyper_loc_loc=jnp.log(0.5),
            log_alpha_hyper_loc_scale=0.1,
            log_beta_hyper_loc_loc=jnp.log(0.5),
            log_beta_hyper_loc_scale=1,
            growth_k_loc=0.2,
            growth_k_scale=0.01,
            growth_m_loc=0.1,
            growth_m_scale=0.1
        )

    def get_initial_params(self):
        
        # Get initial values for SVI
        return initialize.get_init_values(self.df,
                                          self.model,
                                          self.model_data,
                                          self.model_priors)
                                          
    def run_svi(self, num_steps=5000, learning_rate=1e-3):
        # 3. Set up and run the SVI inference
        # ... SVI logic using self.model, self.guide, and self.model_data
        # ... Returns the SVI result
        pass

    def get_posterior_summary(self, svi_result):
        # 4. Post-process the results into a user-friendly format (e.g., a DataFrame)
        pass