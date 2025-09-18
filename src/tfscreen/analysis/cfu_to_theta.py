
from tfscreen.calibration import (
    read_calibration,
    get_background,
    get_wt_theta,
    get_k_vs_theta
)

from tfscreen.util import (
    chunk_by_group,
    argsort_genotypes
)

from tfscreen.fitting import (
    run_least_squares,
    predict_with_error,
    scale,
    unscale,
    get_scaling    
)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy

def _get_theta(theta_unconstrained):
    """
    Logistic transform of unconstrained theta parameters to bound
    between 0 and 1.
    """
    
    return 1/(1 + np.exp(-theta_unconstrained))   

def _get_k(params,
           k_wt0,
           m_cond,
           theta_indexer,
           kshift_indexer,
           k_shift_scaling):
    """
    Get predicted bacterial growth rate under a set of conditions given
    the global effect of the genotype and the fractional saturation of the 
    operator.
    
    Parameters
    ----------
    params : np.ndarray
        1D numpy array of parameters. These are accessed using theta_indexer,
        kshift_indexer, and A0_indexer. 
    k_wt0 : np.ndarray
        1D array of wildtype growth rates under the conditions of interest. 
        There should be one entry per genotype/condition pair. 
    m_cond : np.ndarray
        1D array of slopes relating theta to growth rate under the conditions 
        of interest. There should be one entry per genotype/condition pair. 
    theta_indexer : np.ndarray
        1D array that indexes params, pulling out the appropriate theta for 
        each genotype/condition. There should be one entry per
        genotype/condition pair. 
    kshift_indexer : np.ndarray
        1D array that indexes params, pulling out the appropriate global effect
        of the genotype. There should be one entry per genotype/condition pair.

    Returns
    -------
    np.ndarray
        1D array of growth rates. 
    """

    # Get theta from unconstrained parameters
    theta = _get_theta(params[theta_indexer])
    
    # scale k_shift
    k_shift = unscale(params[kshift_indexer],k_shift_scaling)

    return k_wt0 + k_shift + theta*m_cond 
          

def _theta_to_lncfu(params,
                    theta_indexer,
                    kshift_indexer,
                    A0_indexer,
                    t_pre,
                    k_wt_pre,
                    m_pre,
                    t_sel,
                    k_wt_sel,
                    m_sel,
                    k_shift_scaling,
                    A0_scaling):
    """
    Calculate ln_cfu over time given the global effect of the genotype on
    growth, the fractional saturation of the operator, and the conditions. 

    Parameters
    ----------
    params : np.ndarray
        1D numpy array of parameters. These are accessed using theta_indexer,
        kshift_indexer, and A0_indexer. 
    t : np.ndarray
        1D or 2D array of times at which to do the calculation. The first 
        dimension should have one entry per genotype/condition pair. The second
        dimension can encode multiple time points. 
    k_wt0 : np.ndarray
        1D array of wildtype growth rates at theta = 0 under the conditions of
        interest. There should be one entry per genotype/condition pair. 
    m_cond : np.ndarray
        1D array of slopes relating theta to growth rate under the conditions 
        of interest. There should be one entry per genotype/condition pair. 
    theta_indexer : np.ndarray
        1D array that indexes params, pulling out the appropriate theta for 
        each genotype/condition. There should be one entry per
        genotype/condition pair. 
    kshift_indexer : np.ndarray
        1D array that indexes params, pulling out the appropriate global effect
        of the genotype. There should be one entry per genotype/condition pair.
    A0_indexer : np.ndarray
        1D array that indexes params, pulling out the appropriate initial
        ln_cfu for the sample. There should be one entry per genotype/condition
        pair.

    Returns
    -------
    np.ndarray
        1D array of ln_cfu. The array will be len(k_wt0)*num_times long. 
    """

    # Get initial populations
    A0 = params[A0_indexer]

    # Get growth rate during pre-selection interval
    k_pre = _get_k(params,
                   k_wt_pre,
                   m_pre,
                   theta_indexer,
                   kshift_indexer,
                   k_shift_scaling)
    
    # Get growth rate during selection interval
    k_sel = _get_k(params,
                   k_wt_sel,
                   m_sel,
                   theta_indexer,
                   kshift_indexer,
                   k_shift_scaling)

    # Scale A0 
    A0 = unscale(params[A0_indexer],A0_scaling)

    # Expand to match t if necessary
    if len(t_sel.shape) > 1:
        A0 = A0[:,np.newaxis]
        k_pre = k_pre[:,np.newaxis]
        k_sel = k_sel[:,np.newaxis]

    # Calculate population
    A = A0 + k_pre*t_pre + k_sel*t_sel

    return A.flatten()
    

def _run_regression(df,
                    calibration_data):

    # Work on a copy of the dataframe
    df = df.copy()
    
    # Extract 1D arrays from the data frame
    t_sel = df["time"].to_numpy()
    t_pre = df["pre_time"].to_numpy()
    ln_cfu = df["ln_cfu"].to_numpy()
    ln_cfu_std = np.sqrt(df["ln_cfu_var"].to_numpy())

    # Load calibration dictionary
    calibration_dict = read_calibration(calibration_data)
    
    # --------------------------------------------------------------------------
    # Get relationship between theta and growth for these conditions

    # Get background growth at this titrant concentration
    k_bg = get_background(df["titrant_name"],
                          df["titrant_conc"],
                          calibration_dict)

    # Get wildtype growth for theta = 0 in pre condition
    m_pre, b_pre = get_k_vs_theta(df["pre_condition"],
                                  df["titrant_name"],
                                  calibration_dict)    
    k_wt_pre = k_bg + b_pre

    # Get wildtype growth for theta = 0 in selection condition
    m_sel, b_sel = get_k_vs_theta(df["condition"],
                                  df["titrant_name"],
                                  calibration_dict)    
    k_wt_sel = k_bg + b_sel

     
    # --------------------------------------------------------------------------
    # Create indexer and rev_indexer arrays. The "indexer" arrays map from 
    # dataframe rows to parameters. The rev_indexer arrays map from parameters
    # to rows. (rev_indexer[indexer] would give the rows in the dataframe). 

    # theta_indexer maps each row to its genotype/titrant_name/titrant_conc
    # theta parameter in the parameter array
    theta_slicer = ['genotype', 'titrant_name','titrant_conc']
    _theta_tuples = df[theta_slicer].itertuples(index=False, name=None)
    theta_indexer, theta_rev_indexer = pd.factorize(pd.Series(_theta_tuples))
    
    # kshift_indexer maps each row to its k_shift parameter in the parameter
    # array
    kshift_start = np.max(theta_indexer) + 1
    _idx, _rev = pd.factorize(df["genotype"])
    kshift_indexer = _idx + kshift_start
    kshift_rev_indexer = np.full(np.max(kshift_indexer)+1,object)
    kshift_rev_indexer[kshift_start:] = _rev

    # A0_indexer maps each row to its A0 parameter in the parameter array
    A0_start = np.max(kshift_indexer) + 1
    A0_slicer = ['genotype','replicate','library']    
    _rep_geno_tuples = df[A0_slicer].itertuples(index=False, name=None)
    _idx, _rev = pd.factorize(pd.Series(_rep_geno_tuples))
    A0_indexer = _idx + A0_start
    A0_rev_indexer = np.full(np.max(A0_indexer)+1,object)
    A0_rev_indexer[A0_start:] = _rev
    
    # --------------------------------------------------------------------------
    # Build guesses

    # empty guesses
    guesses = np.zeros(np.max(A0_indexer)+1,dtype=float)
    
    # Initialize theta parameters with wildtype theta
    guesses[theta_indexer] = get_wt_theta(df["titrant_name"],
                                          df["titrant_conc"],
                                          calibration_dict)

    # Get pre-loaded lnA0 and k shift guesses
    guesses[A0_indexer] = df["lnA_pre0_guess"].to_numpy()
    guesses[kshift_indexer] = df["k_shift_guess"].to_numpy()

    # Get scale of k_shift and transform
    k_shift_scaling = get_scaling(guesses[kshift_indexer])
    guesses[kshift_indexer] = scale(guesses[kshift_indexer], k_shift_scaling)
    
    # Get scale of A0 and transform
    A0_scaling = get_scaling(guesses[A0_indexer])  
    guesses[A0_indexer] = scale(guesses[A0_indexer], A0_scaling)

    # --------------------------------------------------------------------------
    # Construct a dataframe holding information about the fit parameters (for
    # the fit_out object with detailed internal information about the fit)

    param_class = []
    param_class.extend(["theta" for _ in theta_rev_indexer[0:]])
    param_class.extend(["k_shift" for _ in kshift_rev_indexer[kshift_start:]])
    param_class.extend(["lnA0" for _ in A0_rev_indexer[A0_start:]])
    
    param_genotype = []
    param_genotype.extend([v[0] for v in theta_rev_indexer[0:]])
    param_genotype.extend([v for v in kshift_rev_indexer[kshift_start:]])
    param_genotype.extend([v[0] for v in A0_rev_indexer[A0_start:]])
    
    param_replicate = []
    param_replicate.extend([None for _ in theta_rev_indexer[0:]])
    param_replicate.extend([None for _ in kshift_rev_indexer[kshift_start:]])
    param_replicate.extend([(v[1],v[2]) for v in A0_rev_indexer[A0_start:]])

    param_titrant_name = []
    param_titrant_name.extend([v[1] for v in theta_rev_indexer[0:]])
    param_titrant_name.extend([None for _ in kshift_rev_indexer[kshift_start:]])
    param_titrant_name.extend([None for _ in A0_rev_indexer[A0_start:]])

    param_titrant_conc = []
    param_titrant_conc.extend([v[2] for v in theta_rev_indexer[0:]])
    param_titrant_conc.extend([np.nan for _ in kshift_rev_indexer[kshift_start:]])
    param_titrant_conc.extend([np.nan for _ in A0_rev_indexer[A0_start:]])

    fit_df = pd.DataFrame({"class":param_class,
                           "genotype":param_genotype,
                           "replicate":param_replicate,
                           "titrant_name":param_titrant_name,
                           "titrant_conc":param_titrant_conc,
                           "guess":guesses})
     
    # --------------------------------------------------------------------------
    # Non-params args for the _theta_to_lncfu model
    args = (theta_indexer,
            kshift_indexer,
            A0_indexer,
            t_pre,
            k_wt_pre,
            m_pre,
            t_sel,
            k_wt_sel,
            m_pre,
            k_shift_scaling,
            A0_scaling)

    # --------------------------------------------------------------------------
    # Do fit,
    
    # Run least squares, optimizing the parameters against ln_cfu with the
    # _theta_to_lncfu model. 
    lower_bounds = np.full(len(guesses),-np.inf)
    upper_bounds = np.full(len(guesses), np.inf)
    lower_bounds[theta_indexer] = -15
    upper_bounds[theta_indexer] = 15
    
    params, std_errors, cov_matrix, _ = run_least_squares(
        _theta_to_lncfu,
        ln_cfu,
        ln_cfu_std,
        guesses,
        lower_bounds,
        upper_bounds,
        args
    )


    # --------------------------------------------------------------------------
    # Build fit_out object with detailed internal information about the fit
    fit_df["est"] = params
    fit_df["std"] = std_errors

    # Prep output dictionary holding fit outputs
    fit_out = {"cov_matrix":cov_matrix,
               "fit_df":fit_df}
        
    
    # -------------------------------------------------------------------------
    # Build dataframes holding parameter outputs on real scale

    # Dataframe holding fractional saturation
    theta_est = _get_theta(params[theta_indexer])
    theta_std = theta_est*(1 - theta_est)*std_errors[theta_indexer]

    theta_df = pd.DataFrame({"genotype":[v[0] for v in theta_rev_indexer[theta_indexer]],
                             "titrant_name":[v[1] for v in theta_rev_indexer[theta_indexer]],
                             "titrant_conc":[float(v[2]) for v in theta_rev_indexer[theta_indexer]],
                             "theta_est":theta_est,
                             "theta_std":theta_std})

    # Dataframe holding global growth rate effects of genotypes
    k_shift_est = unscale(params[kshift_indexer],k_shift_scaling)
    k_shift_std = std_errors[kshift_indexer]*k_shift_scaling[1]
    
    growth_df = pd.DataFrame({"genotype":kshift_rev_indexer[kshift_indexer],
                              "k_shift_est":k_shift_est,
                              "k_shift_std":k_shift_std})

    # Dataframe holding observed and predicted ln(cfu/mL) using model. 
    pred_est, pred_std = predict_with_error(
        _theta_to_lncfu,
        params,
        cov_matrix,
        args
    )
  
    pred_df = pd.DataFrame({"time":t_sel,
                            "obs_est":ln_cfu,
                            "obs_std":ln_cfu_std,
                            "pred_est":pred_est,
                            "pred_std":pred_std})
    
    # Build a dataframe holding growth rates and A0 for each sample

    # Predict the growth rates under all conditions using our estimated 
    # parameters. 
    k_est, k_std = predict_with_error(
        _get_k,
        params,
        cov_matrix,
        args=[k_wt_sel,
              m_sel,
              theta_indexer,
              kshift_indexer,
              k_shift_scaling]
    )

    lnA0_est = unscale(params[A0_indexer], A0_scaling)
    lnA0_std = std_errors[A0_indexer]*A0_scaling[1]

    k_df = df.copy()[["genotype","replicate","library","condition","titrant_name","titrant_conc"]]
    k_df["k_est"] = k_est
    k_df["k_std"] = k_std
    k_df["lnA0_est"] = lnA0_est
    k_df["lnA0_std"] = lnA0_std

    return theta_df, growth_df, pred_df, k_df, fit_out

def cfu_to_theta(df,
                 calibration_data,
                 max_block_size=250):
    """
    Take read counts under different conditions and use them to estimate the 
    fractional saturation of the operator (theta) for each genotype as a
    function of inducer concentration. This estimate also yields the intrinsic
    effect of each genotype on growth rate and the initial population of each 
    genotype in each replicate. 
    
    Parameters
    ----------
    df : 
    calibration_data : str or dict
        Path to the calibration file or loaded calibration dictionary.
    fit_method : str, optional
        which fitting method to use to estimate parameters. should be 'mle' 
        (maximum likelihood, default) or 'map' (maximum a posteriori). 
    max_block_size : int, default=250
        when doing regression, grab no more than max_block_size rows from the
        combined_df when doing regression. It pools rows until it reaches
        max_block_size rows. The function keeps all rows corresponding to a
        specific genotype together. If adding a genotype goes exceeds this, 
        it is placed in the next block. If a genotype has more rows than 
        max_block_size by itself, this limit is ignored and the genotype is
        analyzed on its own.

    Returns
    -------
    length 4 tuple
        + theta_df: pandas.DataFrame holding theta vs. titrant for each genotype
        + growth_df: pandas.DataFrame holding the intrinsic effect of each 
          genotype on growth rate
        + pred_df: pandas.DataFrame holding the observed and predicted
          ln(cfu/mL) under all conditions and times
        + k_df: pandas.DataFrame holding the A0 and k values for each genotype
    """

    # Genotypes come out of replicate_aware_load sorted
    all_genotypes = pd.unique(df["genotype"])
    idx = argsort_genotypes(all_genotypes)
    genotype_order = all_genotypes[idx]

    # Lists to store block-wise results
    theta_dfs = []
    growth_dfs = []
    pred_dfs = []
    k_dfs = []
    fit_outs = []

    # For genotype blocks...
    block_counter = 0
    genotype_blocks = chunk_by_group(df["genotype"],max_block_size)
    for block in tqdm(genotype_blocks):

        this_df = df.loc[block,:]
        theta_df, growth_df, pred_df, k_df, fit_out = _run_regression(
            this_df,
            calibration_data,
        )
    
        theta_dfs.append(theta_df)
        growth_dfs.append(growth_df)
        pred_dfs.append(pred_df)
        k_dfs.append(k_df)

        fit_out["fit_df"]["block"] = block_counter
        fit_outs.append(fit_out)
        
        block_counter += 1
    
    
    # -------------------------------------------------------------------------
    # Finalize theta_df

    theta_df = pd.concat(theta_dfs,ignore_index=True)
    
    # Collapse theta_df to single genotype/titrant name/titrant conc values 
    theta_df["genotype"] = pd.Categorical(theta_df["genotype"],
                                          categories=genotype_order,
                                          ordered=True)    
    theta_df = theta_df.groupby(['genotype', 'titrant_name', 'titrant_conc'],
                                 as_index=False,
                                 observed=False).first()
    
    # -------------------------------------------------------------------------
    # Finalize growth_df

    growth_df = pd.concat(growth_dfs,ignore_index=True)
    
    # Collapse growth_df to single genotype values
    growth_df["genotype"] = pd.Categorical(growth_df["genotype"],
                                           categories=genotype_order,
                                           ordered=True)    
    growth_df = growth_df.groupby(['genotype'],
                                 as_index=False,
                                 observed=False).first()

    # -------------------------------------------------------------------------
    # Finalize pred_df

    pred_df = pd.concat(pred_dfs,ignore_index=True)
    
    # Build a dataframe holding observed and predicted ln(cfu/mL)
    pred_df = pd.concat([df,pred_df],axis=1)
    
    # Sort on genotype, replicate, library, condition, titrant_name, titrant_conc
    pred_df["genotype"] = pd.Categorical(pred_df["genotype"],
                                         categories=genotype_order,
                                         ordered=True)
    sort_on = ["genotype","replicate","library",
               "condition","titrant_name","titrant_conc"]
    pred_df = pred_df.sort_values(sort_on)


    # -------------------------------------------------------------------------
    # Finalize k_df

    k_df = pd.concat(k_dfs,ignore_index=True)
    
    # -------------------------------------------------------------------------
    # Finalize fit_out

    fit_out = {}
    fit_out["fit_df"] = pd.concat([f["fit_df"] for f in fit_outs],
                                  ignore_index=True)
    cov_matrices = [f["cov_matrix"] for f in fit_outs]
    fit_out["cov_matrix"] = scipy.linalg.block_diag(*cov_matrices)

    
    return theta_df, growth_df, pred_df, k_df, fit_out
