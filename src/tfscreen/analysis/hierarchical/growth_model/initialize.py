import jax
import jax.numpy as jnp
import pandas as pd
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

def get_init_values(df, model, data, priors):
    """
    Generates a function that provides sensible initial values for the SVI guide.

    This function creates an initialization dictionary with data-driven guesses
    for key parameters and sets all other location parameters to zero. All scale
    parameters are initialized to a small positive value.

    Args:
        df (pd.DataFrame): The original long-form dataframe.
        model (callable): The NumPyro model function (e.g., growth_model).
        data (GrowthModelData): The dataclass holding JAX arrays for the model.
        priors (GrowthModelPriors): The dataclass holding priors.

    Returns:
        A callable that returns a dictionary of initial parameter values,
        suitable for the `init_loc_fn` argument of an AutoGuide.
    """
    
    # Create a dummy SVI object to discover parameter shapes. We need a dummy
    # optimizer and loss, which are not used beyond this step.
    optimizer = Adam(step_size=0.01)
    guide = AutoNormal(model)
    svi_for_init = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Run svi.init() to get the initial, randomly-initialized parameters.
    # This requires a PRNG key and the model's arguments.
    # The returned object contains the names and correctly-shaped JAX arrays.
    initial_svi_state = svi_for_init.init(
        jax.random.PRNGKey(0),  # Use a fixed key; it's just for shape discovery
        data=data, 
        priors=priors, 
        fix_A=False, 
        fix_dk_geno=False
    )
    proto_params = svi_for_init.get_params(initial_svi_state)
    
    init_values = {}

    # Initializations

    # Guess ln_cfu0 from the mean of the earliest time points for each
    # idx = df.groupby(['map_ln_cfu0'],observed=False)['t_sel'].idxmin()
    # init_df = df.loc[idx]
    # init_ln_cfu0_locs = jnp.zeros(data.num_ln_cfu0)
    # block_map = {name: i for i, name in enumerate(df['map_ln_cfu0'].cat.categories)}
    
    # for _, row in init_df.iterrows():
    #     block_idx = block_map[row['map_ln_cfu0']]
    #     init_ln_cfu0_locs = init_ln_cfu0_locs.at[block_idx].set(row['ln_cfu'])
        
    # super hack
    init_values["ln_cfu0_hyper_locs_auto_loc"] = jnp.zeros(data.num_ln_cfu0) #init_ln_cfu0_locs
    
    # Use pre-estimated values for wild-type theta
    init_values["theta_wt_auto_loc"] = priors.wt_theta_loc

    # This should initialize to 1.
    init_values["A_offset_auto_loc"] = jnp.ones_like(proto_params["A_offset_auto_loc"])
    
    # Set Beta distribution to be around 0.5, 0.5
    target_alpha_beta = 0.5
    init_values["log_alpha_hyper_loc_auto_loc"] = jnp.log(target_alpha_beta)
    init_values["log_beta_hyper_loc_auto_loc"] = jnp.log(target_alpha_beta)

    # Set positive growth as our guess
    init_values["growth_k_loc_auto"] = 0.025

    # Default Initializations (for all other parameters) ---
    for name, proto_value in proto_params.items():

        # Skip parameters we've already set
        if name in init_values:
            continue

        # For scale parameters, start with a small positive value (e.g., 0.1)
        if "auto_scale" in name:
            init_values[name] = jnp.full_like(proto_value, 0.1)
        
        # For all other location parameters, start at zero
        elif "auto_loc" in name:
            init_values[name] = jnp.zeros_like(proto_value)


    # Return a function that SVI can call to get these values
    return init_to_value(values=init_values)