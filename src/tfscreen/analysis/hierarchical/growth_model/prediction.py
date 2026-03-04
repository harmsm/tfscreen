import pandas as pd
import numpy as np
import itertools
from .model_class import ModelClass
from tfscreen.analysis.hierarchical.posteriors import load_posteriors, get_posterior_samples
import jax
from jax import numpy as jnp
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

def copy_model_class(model_class,
                     t_pre=None,
                     t_sel=None,
                     titrant_conc=None,
                     genotypes=None,
                     condition_pre=None,
                     condition_sel=None,
                     titrant_name=None,
                     replicate=None):
    """
    Generate a fresh ModelClass instance with model components from an old
    ModelClass and new data passed in by the user.

    The default behavior is to use values from `model_class.growth_df`.
    Quantitative inputs (t_pre, t_sel, titrant_conc) can be expanded beyond 
    those in the original dataframe. Categorical inputs must be a subset 
    of those present in the original dataframe.

    Parameters
    ----------
    model_class : ModelClass
        The original ModelClass instance to copy.
    t_pre : float, optional
        A single timepoint for pre-growth. Must be >= 0. If None, uses the 
        value(s) from `model_class.growth_df`. If multiple exist in the 
        original dataframe and none is specified, an error is raised.
    t_sel : list, optional
        List of timepoints for selection. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    titrant_conc : list, optional
        List of titrant concentrations. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    genotypes : list, optional
        List of genotypes to include. Must be a subset of those in 
        `model_class.growth_df`. If None, uses all genotypes.
    condition_pre : list, optional
        List of pre-growth conditions to include. Must be a subset 
        of those in `model_class.growth_df`. If None, uses all.
    condition_sel : list, optional
        List of selection conditions to include. Must be a subset 
        of those in `model_class.growth_df`. If None, uses all.
    titrant_name : list, optional
        List of titrant names to include. Must be a subset of 
        those in `model_class.growth_df`. If None, uses all.
    replicate : list, optional
        List of replicates to include. Must be a subset of 
        those in `model_class.growth_df`. If None, uses all.

    Returns
    -------
    ModelClass
        A new ModelClass instance initialized with the exhaustive 
        combinations of all inputs.
    """

    df = model_class.growth_df

    def _get_input(value, col_name, is_quantitative=False, single_value=False):
        """
        Validate and process input values.
        """
        if value is None:
            vals = pd.unique(df[col_name]).tolist()
        else:
            if not isinstance(value, (list, np.ndarray, tuple)):
                vals = [value]
            else:
                vals = list(value)

        # Quantitative checks
        if is_quantitative:
            for v in vals:
                # Handle potential numpy types
                if float(v) < 0:
                    raise ValueError(f"{col_name} must be >= 0, got {v}")
            
            if single_value and len(vals) != 1:
                raise ValueError(
                    f"{col_name} must have exactly one value. "
                    f"Got {len(vals)}: {vals}"
                )
        
        # Categorical checks
        else:
            seen = pd.unique(df[col_name])
            missing = [v for v in vals if v not in seen]
            if missing:
                raise ValueError(
                    f"The following {col_name} values were not found in "
                    f"model_class.growth_df: {missing}"
                )

        return vals

    # Process all inputs
    t_pre_list = _get_input(t_pre, "t_pre", is_quantitative=True, single_value=True)
    t_sel_list = _get_input(t_sel, "t_sel", is_quantitative=True)
    titrant_conc_list = _get_input(titrant_conc, "titrant_conc", is_quantitative=True)

    genotypes_list = _get_input(genotypes, "genotype")
    condition_pre_list = _get_input(condition_pre, "condition_pre")
    condition_sel_list = _get_input(condition_sel, "condition_sel")
    titrant_name_list = _get_input(titrant_name, "titrant_name")
    replicate_list = _get_input(replicate, "replicate")

    # Build exhaustive combinations
    combos = list(itertools.product(
        t_pre_list,
        t_sel_list,
        titrant_conc_list,
        genotypes_list,
        condition_pre_list,
        condition_sel_list,
        titrant_name_list,
        replicate_list
    ))

    # Construct new growth_df
    new_growth_df = pd.DataFrame(combos, columns=[
        "t_pre", "t_sel", "titrant_conc", "genotype",
        "condition_pre", "condition_sel", "titrant_name", "replicate"
    ])

    # Add required data columns with dummy values
    new_growth_df["ln_cfu"] = 0.0
    new_growth_df["ln_cfu_std"] = 1.0

    # Subset binding_df to ensure that every (genotype, titrant_name) in the 
    # binding data is also present in the new growth dataframe. This is 
    # required for ModelClass validation. 
    new_binding_df = model_class.binding_df.copy()
    mask = (new_binding_df["genotype"].isin(genotypes_list)) & \
           (new_binding_df["titrant_name"].isin(titrant_name_list))
    new_binding_df = new_binding_df[mask].copy()

    # Create new ModelClass using settings from the old one
    settings = model_class.settings.copy()
    
    return ModelClass(
        growth_df=new_growth_df,
        binding_df=new_binding_df,
        **settings
    )

def predict(model_class,
            param_posteriors,
            predict_sites=None,
            q_to_get=None,
            num_samples=100,
            t_pre=None,
            t_sel=None,
            titrant_conc=None,
            genotypes=None,
            condition_pre=None,
            condition_sel=None,
            titrant_name=None,
            replicate=None):
    """
    Predict values for specified sites in the model using posterior samples,
    handling subsetting and expansion of inputs.

    Parameters
    ----------
    model_class : ModelClass
        The original ModelClass used for training. 
    param_posteriors : dict or str
        Posterior samples. Can be a dictionary, .npz file, or .h5 file.
    predict_sites : list of str, optional
        List of model sites to predict. If None, defaults to ["growth_pred"].
    q_to_get : dict, optional
        Quantiles to calculate. If None, uses default.
    num_samples : int, optional
        Number of posterior samples to use for prediction. 
    t_pre : float, optional
        A single timepoint for pre-growth. Must be >= 0. If None, uses the 
        value(s) from `model_class.growth_df`. If multiple exist in the 
        original dataframe and none is specified, an error is raised.
    t_sel : list, optional
        List of timepoints for selection. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    titrant_conc : list, optional
        List of titrant concentrations. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    genotypes : list, optional
        List of genotypes to include. Must be a subset of those in 
        `model_class.growth_df`. If None, uses all genotypes.
    condition_pre : list, optional
        List of pre-growth conditions to include. Must be a subset 
        of those in `model_class.growth_df`. If None, uses all.
    condition_sel : list, optional
        List of selection conditions to include. Must be a subset 
        of those in `model_class.growth_df`. If None, uses all.
    titrant_name : list, optional
        List of titrant names to include. Must be a subset of 
        those in `model_class.growth_df`. If None, uses all.
    replicate : list, optional
        List of replicates to include. Must be a subset of 
        those in `model_class.growth_df`. If None, uses all.

    Returns
    -------
    pd.DataFrame or dict
        If a single site is requested, returns a pd.DataFrame with quantile 
        columns. If multiple sites are requested, returns a dictionary mapping 
        site names to pd.DataFrames.
    """

    # Load and validate quantiles
    q_to_get, param_posteriors = load_posteriors(param_posteriors, q_to_get)

    if predict_sites is None:
        predict_sites = ["growth_pred"]
    
    if isinstance(predict_sites, str):
        predict_sites = [predict_sites]

    # Create the prediction model
    new_mc = copy_model_class(model_class,
                              t_pre=t_pre,
                              t_sel=t_sel,
                              titrant_conc=titrant_conc,
                              genotypes=genotypes,
                              condition_pre=condition_pre,
                              condition_sel=condition_sel,
                              titrant_name=titrant_name,
                              replicate=replicate)

    # -------------------------------------------------------------------------
    # Sample selection from posterior
    
    # Identify how many samples we have
    first_key = next(iter(param_posteriors.keys()))
    total_available = param_posteriors[first_key].shape[0]
    
    # Sample indices
    num_samples = min(num_samples, total_available)
    rng = np.random.default_rng()
    sample_indices = rng.choice(total_available, size=num_samples, replace=False)

    # -------------------------------------------------------------------------
    # Categorical re-mapping logic
    
    # Run a trace of the original model to identify plate structure
    seeded_model = seed(model_class.jax_model, rng_seed=0)
    traced_model = trace(seeded_model)
    model_trace = traced_model.get_trace(data=model_class.data,
                                         priors=model_class.priors)

    # Identify which dimensions are associated with which plates
    # We only slice categorical dimensions (subsetting). Quantitative dimensions
    # like titrant_conc and time can be expanded and do not have parameters
    # associated with specific concentrations/timepoints in the same way.
    categorical_dims = ["replicate", "condition_pre", "condition_sel", 
                        "titrant_name", "genotype"]
    
    plate_to_dim = {}
    for i, name in enumerate(model_class.growth_tm.tensor_dim_names):
        if name.lower() in categorical_dims:
            plate_to_dim[name.lower()] = i

    # Sliced samples for Predictive
    sliced_samples = {}
    
    for site_name, site in model_trace.items():
        if site["type"] not in ["sample", "deterministic"]:
            continue
        
        # Get the actual parameter from posteriors (handling suffixes)
        try:
            val = get_posterior_samples(param_posteriors, site_name)
        except KeyError:
            continue
            
        # Extract only the requested samples
        val = val[sample_indices]

        # Handle plate slicing
        for frame in site.get("cond_indep_stack", []):
            plate_name = frame.name.lower()
            dim_idx = None
            for p_name, d_idx in plate_to_dim.items():
                if p_name in plate_name:
                    dim_idx = d_idx
                    break
            
            if dim_idx is not None:
                # Find labels in old and new models
                old_labels = model_class.growth_tm.tensor_dim_labels[dim_idx]
                new_labels = new_mc.growth_tm.tensor_dim_labels[dim_idx]
                
                # Find indices of new labels in old labels
                indices = np.array([np.where(old_labels == lab)[0][0] for lab in new_labels])
                
                # Slice the array along frame.dim
                # frame.dim is relative to the right-most dimension of the trace value.
                val = jnp.take(val, indices, axis=frame.dim)

        sliced_samples[site_name] = val

    # -------------------------------------------------------------------------
    # Run Prediction
    
    predictive = Predictive(new_mc.jax_model, 
                            posterior_samples=sliced_samples, 
                            return_sites=predict_sites)
    
    # We need a key, even if not used for randomness in deterministic sites
    predict_key = jax.random.PRNGKey(0) 
    predictions = predictive(predict_key, 
                             data=new_mc.data, 
                             priors=new_mc.priors)
    
    # -------------------------------------------------------------------------
    # Calculate Quantiles and Join
    
    # Get the flat index for each row in new_mc.growth_df
    tm = new_mc.growth_tm
    base_df = new_mc.growth_df.copy()
    
    # tm._pivot_index columns in df contain the integer codes for each dimension
    # (replicate_idx, time_idx, etc.)
    indices = [base_df[f"{dim}_idx"].values for dim in tm.tensor_dim_names]
    
    all_dfs = {}
    for site in predict_sites:
        site_samples = predictions[site] # shape: (num_samples, ...)
        df = base_df.copy()

        for q_name, q_val in q_to_get.items():
            # Calculate quantiles along the sample dimension (axis 0)
            q_arr = np.quantile(site_samples, q_val, axis=0)
            
            # Pull values for each row. 
            # If q_arr matches the tensor shape, we use the indices.
            if q_arr.shape == tm.tensor_shape:
                df[q_name] = q_arr[tuple(indices)]
            else:
                # Try to broadcast to tensor shape (handles scalars and partial plates)
                try:
                    broadcast_q = np.broadcast_to(q_arr, tm.tensor_shape)
                    df[q_name] = broadcast_q[tuple(indices)]
                except ValueError:
                    # If we can't broadcast, it means there's a shape mismatch we 
                    # can't easily resolve here without more complex metadata.
                    # We'll just assign a dummy value or raise a helpful error.
                    raise ValueError(
                        f"Site '{site}' with shape {q_arr.shape} cannot be "
                        f"mapped to the growth dataframe (tensor shape {tm.tensor_shape})."
                    )
        
        all_dfs[site] = df

    if len(predict_sites) == 1:
        return all_dfs[predict_sites[0]]

    return all_dfs
