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
                     titrant_conc=None):
    """
    Generate a fresh ModelClass instance with model components from an old
    ModelClass and new quantitative data passed in by the user.

    The default behavior is to use values from `model_class.growth_df`.
    Quantitative inputs (t_pre, t_sel, titrant_conc) can be expanded beyond 
    those in the original dataframe. 

    Parameters
    ----------
    model_class : ModelClass
        The original ModelClass instance to copy.
    t_pre : list, optional
        List of timepoints for pre-growth. Must be >= 0. If None, uses the 
        value(s) from `model_class.growth_df`. 
    t_sel : list, optional
        List of timepoints for selection. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    titrant_conc : list, optional
        List of titrant concentrations. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.

    Returns
    -------
    ModelClass
        A new ModelClass instance initialized with the exhaustive 
        combinations of all inputs.
    """

    df = model_class.growth_df

    def _get_input(value, col_name):
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

        # coerce to float
        try:
            vals = [float(v) for v in vals]
        except ValueError:
            raise ValueError(f"{col_name} must be numeric, got {vals}")

        # Quantitative checks
        for v in vals:
            if v < 0:
                raise ValueError(f"{col_name} must be >= 0, got {v}")

        return vals

    # Process quantitative inputs
    t_pre_list = _get_input(t_pre, "t_pre")
    t_sel_list = _get_input(t_sel, "t_sel")
    titrant_conc_list = _get_input(titrant_conc, "titrant_conc")

    # Get all unique categorical combinations
    categorical_cols = ["replicate", "condition_pre", "condition_sel", 
                        "titrant_name", "genotype"]
    unique_cats = df[categorical_cols].drop_duplicates()

    # Build exhaustive combinations of quantitative values
    quant_combos = list(itertools.product(t_pre_list, t_sel_list, titrant_conc_list))
    quant_df = pd.DataFrame(quant_combos, columns=["t_pre", "t_sel", "titrant_conc"])

    # Cross-join categorical and quantitative data
    new_growth_df = unique_cats.merge(quant_df, how="cross")

    # Add required data columns with dummy values
    new_growth_df["ln_cfu"] = 0.0
    new_growth_df["ln_cfu_std"] = 1.0

    # We keep the binding_df as is, as it's keyed by genotype/titrant_name
    # and we aren't subsetting those in this step.
    new_binding_df = model_class.binding_df.copy()

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
            num_samples=None,
            t_pre=None,
            t_sel=None,
            titrant_conc=None,
            genotypes=None):
    """
    Predict values for specified sites in the model using posterior samples,
    handling subsetting of genotypes and expansion of quantitative inputs.

    Parameters
    ----------
    model_class : ModelClass
        The original ModelClass used for training. 
    param_posteriors : dict or str
        Posterior samples. Can be a dictionary, or path to .h5 file.
    predict_sites : list of str, optional
        List of model sites to predict. If None, defaults to ["growth_pred"].
    q_to_get : dict, optional
        Quantiles to calculate. If None, uses default.
    num_samples : int, optional
        Number of posterior samples to use for prediction. 
    t_pre : list, optional
        List of timepoints for pre-growth. Must be >= 0. If None, uses the 
        value(s) from `model_class.growth_df`. 
    t_sel : list, optional
        List of timepoints for selection. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    titrant_conc : list, optional
        List of titrant concentrations. Must be >= 0. If None, uses 
        values from `model_class.growth_df`.
    genotypes : list, optional
        List of genotypes to include in the prediction. Must be a subset 
        of those in `model_class.growth_df`. If None, uses all genotypes.

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

    # Create the expanded prediction model
    new_mc = copy_model_class(model_class,
                              t_pre=t_pre,
                              t_sel=t_sel,
                              titrant_conc=titrant_conc)


    # -------------------------------------------------------------------------
    # Genotype subsetting
    
    all_genotypes = new_mc.growth_tm.tensor_dim_labels[-1]
    if genotypes is None:
        genotypes = all_genotypes.tolist()
        genotype_indices = np.arange(len(all_genotypes))
    else:
        # Validate and find indices
        genotype_indices = []
        for g in genotypes:
            matches = np.where(all_genotypes == g)[0]
            if len(matches) == 0:
                # Try string fallback
                matches = np.where(all_genotypes.astype(str) == str(g))[0]
            
            if len(matches) == 0:
                 raise ValueError(f"Genotype {g} not found in model.")
            genotype_indices.append(matches[0])
        genotype_indices = np.array(genotype_indices)

    # Use existing get_batch to subset the data tensors
    subset_data = new_mc.get_batch(new_mc.data, jnp.array(genotype_indices))

    # -------------------------------------------------------------------------
    # Sample selection from posterior
    
    # Identify how many samples we have
    first_key = next(iter(param_posteriors.keys()))
    total_available = param_posteriors[first_key].shape[0]
    
    # Sample indices
    if num_samples is None:
        num_samples = total_available
    num_samples = min(num_samples, total_available)
    rng = np.random.default_rng()
    sample_indices = rng.choice(total_available, size=num_samples, replace=False)
    sample_indices = np.sort(sample_indices)

    # -------------------------------------------------------------------------
    # Parameter slicing
    
    # Run a trace of the original model to identify plate structure
    # We use the original model class because posteriors match its structure.
    seeded_model = seed(model_class.jax_model, rng_seed=0)
    traced_model = trace(seeded_model)
    model_trace = traced_model.get_trace(data=model_class.data,
                                         priors=model_class.priors)

    sliced_samples = {}
    for site_name, site in model_trace.items():
        if site["type"] != "sample":
            continue
        
        try:
            val = get_posterior_samples(param_posteriors, site_name)
        except KeyError:
            continue
            
        val = val[sample_indices]

        # Slice any plated dimension to match the new data labels. 
        # This handles genotype subsetting and any other model plates (like 
        # titrant_conc in congression/categorical models).
        for frame in site.get("cond_indep_stack", []):
            plate_name = frame.name.lower()
            
            # Find the corresponding dimension in the TensorManager
            dim_idx = None
            for i, name in enumerate(model_class.growth_tm.tensor_dim_names):
                if name.lower() in plate_name:
                    dim_idx = i
                    break
            
            if dim_idx is not None:
                old_labels = model_class.growth_tm.tensor_dim_labels[dim_idx]
                new_labels = new_mc.growth_tm.tensor_dim_labels[dim_idx]

                if not np.array_equal(old_labels, new_labels):
                    try:
                        # Map new labels to old indices via list lookup
                        old_list = old_labels.tolist()
                        indices = [old_list.index(l) for l in new_labels.tolist()]
                        val = jnp.take(val, jnp.array(indices), axis=frame.dim)
                    except ValueError:
                         raise ValueError(
                            f"Site '{site_name}' is plated on '{plate_name}' "
                            f"and cannot be expanded to new values."
                        )

        sliced_samples[site_name] = val

    # -------------------------------------------------------------------------
    # Run Prediction
    
    predictive = Predictive(new_mc.jax_model, 
                            posterior_samples=sliced_samples, 
                            return_sites=predict_sites)
    
    # We need a key, even if not used for randomness in deterministic sites
    predict_key = jax.random.PRNGKey(0) 
    predictions = predictive(predict_key, 
                             data=subset_data, 
                             priors=new_mc.priors)
    
    # -------------------------------------------------------------------------
    # Calculate Quantiles and Join
    
    # tm._pivot_index columns in df contain the integer codes for each dimension
    # (replicate_idx, time_idx, etc.)
    # We want a dataframe that only has the subsetted genotypes.
    base_df = new_mc.growth_df.copy()
    base_df = base_df[base_df["genotype"].isin(genotypes)].copy()
    
    # Re-calculate indices for the subsetted dataframe relative to the 
    # using the new_mc TM but subset the df.
    tm = new_mc.growth_tm
    indices = [base_df[f"{dim}_idx"].values for dim in tm.tensor_dim_names]
    
    # Since we used get_batch, the predictions only have the subsetted genotypes.
    # The last dimension of predictions[site] will match the length of genotype_indices.
    # To map back to the tensor indices, we need to map genotype_idx in base_df
    # to its relative position in the genotype_indices array.
    
    # Mapping from original genotype index to new relative index
    geno_map = {orig_idx: i for i, orig_idx in enumerate(genotype_indices)}
    relative_geno_indices = base_df["genotype_idx"].map(geno_map).values
    
    # Update indices for genotype to be relative for the prediction tensor
    indices[-1] = relative_geno_indices

    all_dfs = {}
    for site in predict_sites:
        site_samples = predictions[site] # shape: (num_samples, ...)
        df = base_df.copy()

        for q_name, q_val in q_to_get.items():
            # Calculate quantiles along the sample dimension (axis 0)
            q_arr = np.quantile(site_samples, q_val, axis=0)
            
            # Pull values for each row using the relative indices.
            # q_arr shape will reflect the subsetted genotypes (and potentially expanded others).
            # We use the full tensor indexing but with the subsetted genotype indices.
            df[q_name] = q_arr[tuple(indices)]
        
        all_dfs[site] = df

    if len(predict_sites) == 1:
        return all_dfs[predict_sites[0]]

    return all_dfs
