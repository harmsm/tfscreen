import pandas as pd
import numpy as np
import itertools
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel.inference.posteriors import load_posteriors, get_posterior_samples
from tfscreen.tfmodel.tensors.batch import get_batch as _get_batch
import jax
from jax import numpy as jnp
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

def _align_site_to_tm_dims(spatial_shape, tm_sizes):
    """
    Map each axis of a predicted site's spatial_shape to the correct
    TensorManager dimension index.

    Right-to-left alignment is tried first.  It is correct when the
    non-broadcast site dimensions are tail-aligned with the TM ordering —
    e.g. growth_pred (7-D, full TM rank) or theta_growth_pred (7-D with
    broadcast size-1 dims on the left).

    For sites whose non-broadcast dims are NOT tail-aligned — notably ln_cfu0
    with shape (n_rep, n_cp, n_geno) where n_rep and n_cp sit at TM positions
    0 and 2 rather than at the tail — we fall back to a greedy left-to-right
    subsequence search through tm_sizes.

    Parameters
    ----------
    spatial_shape : tuple of int
        Shape of the prediction site with the sample axis removed.
    tm_sizes : tuple of int
        Sizes of each TensorManager dimension in order.

    Returns
    -------
    list of int
        tm_dims[i] is the TM dimension index for spatial_shape[i].
    """
    n_tm = len(tm_sizes)
    n_sp = len(spatial_shape)

    rt_dims = [n_tm - n_sp + i for i in range(n_sp)]

    # Check whether non-broadcast axes agree with their right-to-left positions.
    rt_ok = all(
        spatial_shape[i] == tm_sizes[rt_dims[i]]
        for i in range(n_sp)
        if spatial_shape[i] != 1
    )
    if rt_ok:
        return rt_dims

    # Right-to-left is wrong: greedy left-to-right subsequence match on
    # non-broadcast dims.  Size-1 (broadcast) dims keep their rt_dims entry
    # because modulo makes the exact TM position irrelevant.
    result = list(rt_dims)
    j = 0
    for i in range(n_sp):
        if spatial_shape[i] == 1:
            continue
        while j < n_tm and tm_sizes[j] != spatial_shape[i]:
            j += 1
        if j >= n_tm:
            return rt_dims  # No match — fall back to right-to-left
        result[i] = j
        j += 1

    return result


def copy_orchestrator(orchestrator,
                      t_pre=None,
                      t_sel=None,
                      titrant_conc=None,
                      genotypes=None):
    """
    Generate a fresh ModelOrchestrator instance with model components from an old
    ModelOrchestrator and new quantitative data passed in by the user.

    The default behavior is to use values from `orchestrator.growth_df`.
    Quantitative inputs (t_pre, t_sel, titrant_conc) can be expanded beyond
    those in the original dataframe.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        The original ModelOrchestrator instance to copy.
    t_pre : list, optional
        List of timepoints for pre-growth. Must be >= 0. If None, uses the
        value(s) from `orchestrator.growth_df`.
    t_sel : list, optional
        List of timepoints for selection. Must be >= 0. If None, uses
        values from `orchestrator.growth_df`.
    titrant_conc : list, optional
        List of titrant concentrations. Must be >= 0. If None, uses
        values from `orchestrator.growth_df`.
    genotypes : list, optional
        Subset of genotypes to include. Must be a subset of those in
        ``orchestrator.growth_df``. If None, uses all genotypes.

    Returns
    -------
    ModelOrchestrator
        A new ModelOrchestrator instance initialized with the exhaustive
        combinations of all inputs.
    """

    df = orchestrator.growth_df

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

    # Subset genotypes when requested.  Doing this here ensures the new
    # orchestrator's TM labels differ from the original's, which causes the
    # parameter-slicing loop in predict() to fire correctly.
    if genotypes is not None:
        all_genos = set(str(g) for g in unique_cats["genotype"].tolist())
        geno_strs = [str(g) for g in genotypes]
        missing = [g for g in geno_strs if g not in all_genos]
        if missing:
            raise ValueError(f"Genotype(s) not found in orchestrator: {missing}")
        unique_cats = unique_cats[
            unique_cats["genotype"].astype(str).isin(set(geno_strs))
        ]

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
    new_binding_df = orchestrator.binding_df.copy()

    # Create new ModelOrchestrator using settings from the old one
    settings = orchestrator.settings.copy()

    # When subsetting genotypes, restrict spiked_genotypes in settings to
    # only those present in the subset.  Spiked genotypes affect per-genotype
    # masks (spiked_mask) which control per_geno_loc/scale at prediction time
    # via the non-centred parameterisation — so any spiked genotype that IS in
    # the subset must remain marked as spiked; those outside the subset are
    # simply omitted.  Dropping all spiked_genotypes would silently mis-classify
    # in-subset spiked genotypes as library genotypes and corrupt predictions.
    if genotypes is not None:
        geno_set = set(geno_strs)
        orig_spiked = settings.get("spiked_genotypes") or []
        settings["spiked_genotypes"] = [
            g for g in orig_spiked if str(g) in geno_set
        ]

    return ModelOrchestrator(
        growth_df=new_growth_df,
        binding_df=new_binding_df,
        **settings
    )

def _convert_map_params(map_params, model_trace):
    """
    Convert a raw MAP parameter dict to a posterior dict suitable for Predictive.

    ``AutoDelta`` guide parameters are stored in *constrained* space with
    ``{site}_auto_loc`` keys because ``RunInference.write_params`` calls
    ``svi.get_params(svi_state)``, which applies the SVI ``constrain_fn``
    before saving.  ``Predictive(posterior_samples=...)`` expects values keyed
    by bare site name with a leading sample dimension.  This function strips
    the ``_auto_loc`` suffix and adds that dimension so the rest of
    :func:`predict` can treat the MAP point as a 1-sample posterior.

    .. note::
        Do **not** apply ``biject_to`` or any other transform here.  The values
        are already constrained; a second transform would corrupt positive-
        constrained sites (e.g. ``ln_cfu0_hyper_scale``) by applying ``exp``
        to a value that is already in positive space.

    Parameters
    ----------
    map_params : dict-like
        Raw MAP parameter dict (e.g., from ``np.load("_params.npz")``).
        Keys end with ``_auto_loc``; values are already in constrained space.
    model_trace : dict
        NumPyro model trace (unused; retained for interface compatibility).

    Returns
    -------
    dict
        Constrained values keyed by bare site name, each with shape
        ``(1, *site_shape)``.
    """
    constrained = {}
    for k, v in map_params.items():
        k = str(k)
        if not k.endswith("_auto_loc"):
            continue
        site_name = k[: -len("_auto_loc")]
        val = jnp.array(np.asarray(v))
        # Values are already constrained (svi.get_params() applied constrain_fn).
        # Add the leading sample dimension expected by Predictive.
        constrained[site_name] = jnp.expand_dims(val, 0)

    return constrained


def predict(orchestrator,
            param_posteriors,
            predict_sites=None,
            q_to_get=None,
            num_samples=100,
            num_marginal_samples=None,
            t_pre=None,
            t_sel=None,
            titrant_conc=None,
            genotypes=None):
    """
    Predict values for specified sites in the model using posterior samples,
    handling subsetting of genotypes and expansion of quantitative inputs.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        The original ModelOrchestrator used for training.
    param_posteriors : dict or str
        Posterior samples. Can be a dictionary, or path to .h5 file.
    predict_sites : list of str, optional
        List of model sites to predict. If None, defaults to ["growth_pred"].
    q_to_get : dict, optional
        Quantiles to calculate. If None, uses default.
    num_samples : int or None, optional
        Randomly select this many joint draws from the predicted samples and
        return them as ``sample_0`` … ``sample_{N-1}`` columns alongside the
        quantile columns, preserving joint parameter uncertainty across rows.
        Sampling is with replacement when ``num_samples`` exceeds the number
        of draws available. Set to ``None`` to suppress sample columns and
        return only quantiles. Default 100.
    num_marginal_samples : int or None, optional
        Number of posterior samples to run through the model for computing
        quantile predictions. If None, uses all available samples.
    t_pre : list, optional
        List of timepoints for pre-growth. Must be >= 0. If None, uses the
        value(s) from `orchestrator.growth_df`.
    t_sel : list, optional
        List of timepoints for selection. Must be >= 0. If None, uses
        values from `orchestrator.growth_df`.
    titrant_conc : list, optional
        List of titrant concentrations. Must be >= 0. If None, uses
        values from `orchestrator.growth_df`.
    genotypes : list, optional
        List of genotypes to include in the prediction. Must be a subset
        of those in `orchestrator.growth_df`. If None, uses all genotypes.

    Returns
    -------
    pd.DataFrame or dict
        If a single site is requested, returns a pd.DataFrame with quantile
        columns and, unless ``num_samples`` is ``None``, ``sample_0`` …
        ``sample_{N-1}`` columns. If multiple sites are requested, returns a
        dictionary mapping site names to such DataFrames.
    """

    # Load and validate quantiles
    q_to_get, param_posteriors = load_posteriors(param_posteriors, q_to_get)

    if predict_sites is None:
        predict_sites = ["growth_pred"]
    
    if isinstance(predict_sites, str):
        predict_sites = [predict_sites]

    # Create the expanded prediction model, subsetting genotypes if requested.
    # Passing genotypes to copy_orchestrator ensures the new orchestrator's TM
    # labels are a strict subset of the original's, which triggers parameter
    # slicing in the loop below.  Validation of unknown genotype names is also
    # handled inside copy_orchestrator.
    new_orchestrator = copy_orchestrator(orchestrator,
                                         t_pre=t_pre,
                                         t_sel=t_sel,
                                         titrant_conc=titrant_conc,
                                         genotypes=genotypes)

    # After copy_orchestrator the new TM contains exactly the requested
    # genotypes (or all genotypes when genotypes=None).
    genotypes = new_orchestrator.growth_tm.tensor_dim_labels[-1].tolist()

    # -------------------------------------------------------------------------
    # Model trace — run first so bijections are available for MAP detection.
    #
    # We use the original orchestrator because posteriors match its structure.

    seeded_model = seed(orchestrator.jax_model, rng_seed=0)
    traced_model = trace(seeded_model)
    model_trace = traced_model.get_trace(data=orchestrator.data,
                                         priors=orchestrator.priors)

    # -------------------------------------------------------------------------
    # MAP param conversion (if needed)
    #
    # Raw MAP params from RunInference.write_params() / np.load("_params.npz")
    # have keys like ``{site}_auto_loc`` in *constrained* space (because
    # write_params receives the output of svi.get_params(), which applies
    # constrain_fn) and no leading sample dimension.  Rename the keys and add
    # a size-1 leading dim so the rest of this function treats them as a
    # 1-sample posterior.
    #
    # Genuine posterior files produced by get_posteriors() / get_map_posteriors()
    # already store constrained values keyed by bare site name and are left
    # unchanged.

    if any(str(k).endswith("_auto_loc") for k in param_posteriors.keys()):
        param_posteriors = _convert_map_params(param_posteriors, model_trace)

    # -------------------------------------------------------------------------
    # Sample selection from posterior

    # Identify how many samples we have
    first_key = next(iter(param_posteriors.keys()))
    total_available = param_posteriors[first_key].shape[0]

    # Sample indices for running through the model (quantile computation)
    n_for_quantiles = total_available if num_marginal_samples is None else min(num_marginal_samples, total_available)
    rng = np.random.default_rng()
    sample_indices = rng.choice(total_available, size=n_for_quantiles, replace=False)
    sample_indices = np.sort(sample_indices)

    # -------------------------------------------------------------------------
    # Parameter slicing

    sliced_samples = {}
    for site_name, site in model_trace.items():
        if site["type"] != "sample":
            continue
            
        if site.get("is_observed", False):
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
            for i, name in enumerate(orchestrator.growth_tm.tensor_dim_names):
                if name.lower() in plate_name:
                    dim_idx = i
                    break
            
            if dim_idx is not None:
                old_labels = orchestrator.growth_tm.tensor_dim_labels[dim_idx]
                new_labels = new_orchestrator.growth_tm.tensor_dim_labels[dim_idx]

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
    
    predictive = Predictive(new_orchestrator.jax_model,
                            posterior_samples=sliced_samples,
                            return_sites=predict_sites)

    # We need a key, even if not used for randomness in deterministic sites
    predict_key = jax.random.PRNGKey(0)

    # Build a canonical-indexed data view for prediction.  The static
    # new_orchestrator.data.growth.batch_idx uses a binding-first ordering
    # (binding genotypes at positions 0..num_binding-1, then library genotypes
    # at positions num_binding..N-1).  Posterior samples, however, are always
    # stored in canonical genotype order (index g = genotype g).  Passing the
    # static data to Predictive would misalign posterior params vs. mask
    # lookups for all non-canonical positions.
    #
    # get_batch with arange(num_genotype) sets batch_idx = arange(num_genotype)
    # (canonical), which is exactly what get_posteriors uses when saving
    # samples.  This makes per-genotype lookups (masks, offsets) consistent
    # with the stored posterior values.
    num_geno = new_orchestrator.data.growth.num_genotype
    all_indices = jnp.arange(num_geno, dtype=jnp.int32)
    pred_data = _get_batch(new_orchestrator.data, all_indices)

    # Use the ORIGINAL orchestrator's priors so that any pinned hyperpriors
    # (e.g. activity_hyper_loc/activity_hyper_scale in the calibration model)
    # remain pinned at their calibrated values rather than being sampled from
    # the default prior.  new_orchestrator is rebuilt from settings-only
    # (component names, no priors) so its priors are all-default.
    predictions = predictive(predict_key,
                             data=pred_data,
                             priors=orchestrator.priors)

    # -------------------------------------------------------------------------
    # Calculate Quantiles and Join

    # new_orchestrator already contains exactly the requested genotypes, so
    # growth_df needs no further filtering.
    base_df = new_orchestrator.growth_df.copy()

    # Replace the dummy ln_cfu/ln_cfu_std zeros with observed values from the
    # original orchestrator.growth_df (NaN where there is no matching observation,
    # e.g. for expanded prediction grids).
    merge_keys = ["replicate", "condition_pre", "condition_sel",
                  "titrant_name", "genotype", "t_pre", "t_sel", "titrant_conc"]
    obs_cols = merge_keys + ["ln_cfu", "ln_cfu_std"]
    orig_obs = orchestrator.growth_df[obs_cols].drop_duplicates(subset=merge_keys)
    base_df = base_df.drop(columns=["ln_cfu", "ln_cfu_std"]).merge(
        orig_obs, on=merge_keys, how="left"
    )

    tm = new_orchestrator.growth_tm
    indices = [base_df[f"{dim}_idx"].values for dim in tm.tensor_dim_names]

    # TM dimension sizes matching the current (possibly genotype-subsetted) indices.
    tm_sizes = tuple(int(idx.max()) + 1 if len(idx) > 0 else 1 for idx in indices)

    all_dfs = {}
    for site in predict_sites:
        site_samples = predictions[site]  # shape: (num_samples, ...)

        # Map each axis of the site's spatial shape to the correct TM dimension.
        # Right-to-left alignment works for full-rank and tail-aligned sites;
        # _align_site_to_tm_dims falls back to subsequence matching for sites
        # like ln_cfu0 whose batch dims are not at the tail of the TM ordering.
        spatial_shape = site_samples.shape[1:]
        tm_dims = _align_site_to_tm_dims(spatial_shape, tm_sizes)
        aligned_indices = tuple(
            indices[tm_dims[i]] % spatial_shape[i]
            for i in range(len(spatial_shape))
        )

        # Sites whose spatial tensor has size-1 (broadcast) dimensions — e.g.
        # theta_growth_pred with shape (1,1,1,1,1,n_conc,n_geno) — would require
        # indexing into a len(base_df)-sized result even though only the
        # non-trivial dimensions carry unique information.  Deduplicate base_df to
        # the unique combinations of those non-trivial TM dimensions so that
        # downstream allocations scale with the compact tensor, not the full
        # cross-joined dataframe.
        has_broadcast = any(sz == 1 for sz in spatial_shape)
        if has_broadcast:
            nontrivial_tm_idx = [
                tm_dims[i]
                for i, sz in enumerate(spatial_shape) if sz > 1
            ]
            if nontrivial_tm_idx:
                dedup_cols = [f"{tm.tensor_dim_names[j % len(tm.tensor_dim_names)]}_idx"
                              for j in nontrivial_tm_idx]
                unique_mask = ~base_df.duplicated(subset=dedup_cols)
            else:
                # All dims are size-1; the whole tensor is a scalar per sample.
                unique_mask = pd.Series(False, index=base_df.index)
                unique_mask.iloc[0] = True
            unique_mask_values = unique_mask.values
            df = base_df[unique_mask].copy()
            eff_aligned = tuple(ai[unique_mask_values] for ai in aligned_indices)
        else:
            df = base_df.copy()
            eff_aligned = aligned_indices

        for q_name, q_val in q_to_get.items():
            q_arr = np.quantile(site_samples, q_val, axis=0)
            df[q_name] = q_arr[eff_aligned]

        if num_samples is not None and num_samples > 0:
            site_samples_np = np.array(site_samples)  # (S, ...)
            S = site_samples_np.shape[0]
            chosen = np.random.choice(S, size=num_samples,
                                      replace=num_samples > S)
            sample_arr = np.stack(
                [site_samples_np[s][eff_aligned] for s in chosen],
                axis=1,
            )  # (n_rows, num_samples)
            samples_df = pd.DataFrame(
                sample_arr,
                columns=[f"sample_{i}" for i in range(num_samples)],
                index=df.index,
            )
            df = pd.concat([df, samples_df], axis=1)

        all_dfs[site] = df

    if len(predict_sites) == 1:
        return all_dfs[predict_sites[0]]

    return all_dfs
