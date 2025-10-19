from tfscreen.genetics import (
    set_categorical_genotype
)

from tfscreen.util import (
    read_dataframe,
    get_scaled_cfu,
    check_columns,
)

from .dataclasses import (
    GrowthModelData,
    GrowthModelParameters
)

import jax.numpy as jnp
import pandas as pd

def _get_condition_maps(condition_to_idx):
    """
    Build arrays mapping experimental conditions to parameter locations
    along appropriate tensor dimensions.
    """

    cond_seen = list(set([c[0] for c in condition_to_idx] +
                         [c[1] for c in condition_to_idx]))
    cond_seen.sort()
    cond_seen = {c:i for i, c in enumerate(cond_seen)}
    
    titr_conc_seen = list(set([c[3] for c in condition_to_idx]))
    titr_conc_seen.sort()
    titr_conc_seen = {t:i for i, t in enumerate(titr_conc_seen)}
    
    cond_pre_map = []
    cond_sel_map = []
    theta_map = []
    for c in condition_to_idx:
        
        cond_pre = c[0]
        cond_pre_map.append(cond_seen[cond_pre])
        
        cond_sel = c[1]
        cond_sel_map.append(cond_seen[cond_sel])
        
        titr_conc = c[3]
        theta_map.append(titr_conc_seen[titr_conc])

    cond_pre_map = jnp.asarray(cond_pre_map,dtype=int)
    cond_sel_map = jnp.asarray(cond_sel_map,dtype=int)
    theta_map = jnp.asarray(theta_map,dtype=int)

    return cond_pre_map, cond_sel_map, theta_map, cond_seen, titr_conc_seen


def df_to_tensors(df,condition_cols=None,ln_cfu0_block_cols=None):
    """
    Take an input dataframe and generate 4D tensors for model regression.
    """

    # -------------------------------------------------------------------------
    # Prep dataframe for processing 
    # -------------------------------------------------------------------------
    
    # Columns defining a unique condition (within a replicate) that should have
    # its own growth model. Generally pre/sel/titrant. 
    if condition_cols is None:
        condition_cols = ['condition_pre','condition_sel','titrant_name','titrant_conc']

    # Columns defining unique ln_cfu0 start blocks (e.g., a genotype should 
    # have the same ln_cfu0 across all conditions that share replicate &
    # condition_pre). 
    if ln_cfu0_block_cols is None:
        ln_cfu0_block_cols = ["replicate","condition_pre"]

    # read from file or work on a copy
    df = read_dataframe(df)
    

    # Make sure we have required ln_cfu/std columns, calculating if
    # needed/possible
    df = get_scaled_cfu(df,need_columns=["ln_cfu","ln_cfu_std"])

    # make a replicate column if not defined
    if "replicate" not in df.columns:
        df["replicate"] = 1

    # check for all required columns
    required = ["ln_cfu","ln_cfu_std","replicate"]
    required.extend(condition_cols)
    required.extend(ln_cfu0_block_cols)
    required.extend(["t_pre","t_sel"])
    check_columns(df,required_columns=required)
                                                  
    # Set genotype and replicate to categorical.
    df = set_categorical_genotype(df,standardize=True)
    df['replicate'] = pd.Categorical(df['replicate'])

    # Build aggregate condition and ln_cfu0_block columns
    df['condition'] = pd.Categorical(df[condition_cols].apply(tuple, axis=1))
    df['ln_cfu0_block'] = pd.Categorical(df[ln_cfu0_block_cols].apply(tuple, axis=1))

    # -------------------------------------------------------------------------
    # Set up idx columns for the pivot. 
    # -------------------------------------------------------------------------
    
    # Indices provide a powerful way to pivot later
    df['geno_idx'] = df['genotype'].cat.codes
    df['rep_idx'] = df['replicate'].cat.codes
    df['cond_idx'] = df['condition'].cat.codes
    df['time_idx'] = (df
                      .groupby(['replicate','genotype', 'condition'],observed=False)['t_sel']
                      .rank(method='first')
                      .astype(int) - 1)
    df['ln_cfu0_block_idx'] = df['ln_cfu0_block'].cat.codes

    # Make mapping to  genotype, replicate, and condition to the
    # appropriate index. 
    geno_to_idx = {name: i for i, name in enumerate(df['genotype'].cat.categories)}
    replicate_to_idx = {name: i for i, name in enumerate(df['replicate'].cat.categories)}
    condition_to_idx = {name: i for i, name in enumerate(df['condition'].cat.categories)}
    ln_cfu0_block_to_idx = {name: i for i, name in enumerate(df['ln_cfu0_block'].cat.categories)}

    # Get final array dimensions
    num_geno = len(geno_to_idx)
    num_rep = len(replicate_to_idx)
    num_cond = len(condition_to_idx)
    num_time = df['time_idx'].max() + 1
    shape = (num_rep,num_cond,num_geno,num_time)

    # -------------------------------------------------------------------------
    # Pivot dataframe
    # -------------------------------------------------------------------------
    
    # Build an exhaustive multiindex
    all_reps = df['rep_idx'].unique()
    all_conds = df['cond_idx'].unique()
    all_genos = df['geno_idx'].unique()
    all_times = df['time_idx'].unique()

    exhaustive_index = pd.MultiIndex.from_product(
        [all_reps, all_conds, all_genos, all_times],
        names=['rep_idx', 'cond_idx', 'geno_idx', 'time_idx']
    )

    # Use pivot_table to reshape the data directly.
    pivoted_df = pd.pivot_table(df,
        values=['ln_cfu','ln_cfu_std','t_sel','t_pre','ln_cfu0_block_idx'],
        index=['rep_idx', 'cond_idx', 'geno_idx', 'time_idx'],
        observed=False
    )

    # Reindex, forcing the complete set of indices
    pivoted_df = pivoted_df.reindex(exhaustive_index)

    # -------------------------------------------------------------------------
    # Build output tensors
    # -------------------------------------------------------------------------
    
    # Reshape output arrays
    ln_cfu_obs = jnp.asarray(pivoted_df["ln_cfu"].to_numpy(dtype=float).reshape(shape))
    ln_cfu_std = jnp.asarray(pivoted_df["ln_cfu_std"].to_numpy(dtype=float).reshape(shape))
    t_sel = jnp.asarray(pivoted_df["t_sel"].to_numpy(dtype=float).reshape(shape))
    t_pre = jnp.asarray(pivoted_df["t_pre"].to_numpy(dtype=float).reshape(shape))
    ln_cfu0_block_idx = jnp.asarray(pivoted_df["ln_cfu0_block_idx"].to_numpy(dtype=float).reshape(shape))
    
    # Get mask for values that are seen (pivot may have introduced nan for
    # missing obs)
    good_mask = ~jnp.isnan(ln_cfu_obs)
    
    # Replace NaNs in data tensors with 0. good_mask will handle them in the
    # model
    ln_cfu_obs = jnp.nan_to_num(ln_cfu_obs)
    ln_cfu_std = jnp.nan_to_num(ln_cfu_std)
    t_sel = jnp.nan_to_num(t_sel)
    t_pre = jnp.nan_to_num(t_pre)
    ln_cfu0_block_idx = jnp.nan_to_num(ln_cfu0_block_idx)

    # -------------------------------------------------------------------------
    # Build various mapping arrays and masks
    # -------------------------------------------------------------------------
    
    # Get wildtype -- will apply on the the genotype dimension
    wt_index = int(geno_to_idx["wt"])
    not_wt_mask = df['genotype'].cat.categories != "wt"
    
    # Map arrays expanding condition_to_idx to map condition_pre, condition_sel
    # and titr_conc
    cond_pre_map, cond_sel_map, theta_map, raw_cond_to_idx, titr_conc_to_idx = _get_condition_maps(condition_to_idx)
    num_raw_cond = len(raw_cond_to_idx)
    num_theta = len(titr_conc_to_idx)

    # map in replicate, genotype, condition that gives ln_cfu0
    ln_cfu0_block_map = jnp.asarray(ln_cfu0_block_idx[:,:,:,0],dtype=int)
    num_ln_cfu0_block = len(ln_cfu0_block_to_idx)

    # replicate map
    replicate_map = jnp.asarray(df['rep_idx'],dtype=int)

    # -------------------------------------------------------------------------
    # Construct output classes
    # -------------------------------------------------------------------------
    
    # Build model data class
    gmd = GrowthModelData(ln_cfu_obs=ln_cfu_obs,
                          ln_cfu_std=ln_cfu_std,
                          t_sel=t_sel,
                          t_pre=t_pre,
                          wt_index=wt_index,
                          cond_pre_map=cond_pre_map,
                          cond_sel_map=cond_sel_map,
                          theta_map=theta_map,
                          ln_cfu0_block_map=ln_cfu0_block_map,
                          good_mask=good_mask,
                          not_wt_mask=not_wt_mask,
                          num_geno=num_geno,
                          num_rep=num_rep,
                          num_cond=num_cond,
                          num_time=num_time,
                          num_raw_cond=num_raw_cond,
                          num_theta=num_theta,
                          num_ln_cfu0_block=num_ln_cfu0_block)

    gmp = GrowthModelParameters(geno_to_idx=geno_to_idx,
                                replicate_to_idx=replicate_to_idx,
                                condition_to_idx=condition_to_idx,
                                raw_cond_to_idx=raw_cond_to_idx,
                                titr_conc_to_idx=titr_conc_to_idx)
                      
    return df, gmd, gmp
    