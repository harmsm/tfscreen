
from tfscreen.calibration import (
    read_calibration,
    get_wt_k,
    get_wt_theta,
    get_slopes
)

from tfscreen.util import (
    chunk_by_group,
)

from tfscreen.fitting import (
    run_least_squares,
    run_map,
    predict_with_error,
    scale,
    unscale,
    get_scaling    
)

from tfscreen.analysis import (
    counts_to_lncfu,
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
                    t_noselect,
                    k_wt0_noselect,
                    m_noselect,
                    t_select,
                    k_wt0_select,
                    m_select,
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
    k_noselect = _get_k(params,
                        k_wt0_noselect,
                        m_noselect,
                        theta_indexer,
                        kshift_indexer,
                        k_shift_scaling)
    
    # Get growth rate during selection interval
    k_select = _get_k(params,
                      k_wt0_select,
                      m_select,
                      theta_indexer,
                      kshift_indexer,
                      k_shift_scaling)

    # Scale A0 
    A0 = unscale(params[A0_indexer],A0_scaling)

    # Flatten t if necessary
    if len(t_select.shape) > 1:
        A0 = A0[:,np.newaxis]
        k_noselect = k_noselect[:,np.newaxis]
        k_select = k_select[:,np.newaxis]

    # Calculate population
    A = A0 + k_noselect*t_noselect + k_select*t_select

    return A.flatten()
    

def _run_regression(df,
                    times,
                    ln_cfu,
                    ln_cfu_var,
                    calibration_data,
                    fit_method="mle",
                    unique_A0=False,
                    pre_select_t=30):

    # Work on a copy of the dataframe
    df = df.copy()
    num_rows = len(df["genotype"])
    
    # Load calibration dictionary
    calibration_dict = read_calibration(calibration_data)
    
    # --------------------------------------------------------------------------
    # Get growth rates of wildtype in all conditions with theta = 0. Then get
    # slope of growth vs. theta in these conditions. DO this for noselect (out
    # growth in iptg without selection) and select (full selection conditions)

    k_wt0_noselect, _ = get_wt_k(marker=df["marker"],
                                 select=np.zeros(len(df["iptg"]),dtype=int),
                                 iptg=df["iptg"],
                                 calibration_data=calibration_dict,
                                 theta=np.zeros(len(df["iptg"]),dtype=float),
                                 calc_err=False)

    m_noselect = get_slopes(marker=df["marker"],
                            select=np.zeros(len(df["iptg"]),dtype=int),
                            calibration_data=calibration_dict)

    # pre-calculate k_wt0 for all rows. This is the growth rate of 
    # wildtype for theta = 0 under these conditions. 
    k_wt0_select, _ = get_wt_k(marker=df["marker"],
                               select=df["select"],
                               iptg=df["iptg"],
                               calibration_data=calibration_dict,
                               theta=np.zeros(len(df["iptg"]),dtype=float),
                               calc_err=False)
    
    # pre-calculate m_cond for all rows. This is the slope relating theta to
    # growth rate perturbation. 
    m_select = get_slopes(marker=df["marker"],
                          select=df["select"],
                          calibration_data=calibration_dict)
     
    # --------------------------------------------------------------------------
    # Create indexer and rev_indexer arrays. The "indexer" arrays map from 
    # dataframe rows to parameters. The rev_indexer arrays map from parameters
    # to rows. (rev_indexer[indexer] would give the rows in the dataframe). 

    # params are organized like this:
    # 
    #  [genotype_0_iptg_0_theta,
    #   genotype_0_iptg_0.1_theta,
    #   genotype_0_iptg_0.5_theta,
    #   genotype_0_iptg_1.0_theta,
    #   genotype_1_iptg_0_theta,
    #   genotype_1_iptg_0.1_theta,
    #     ...
    #   genotype_n_iptg_1.0_theta,
    #   genotype_0_k_shift,
    #   genotype_1_k_shift,
    #     ...
    #   genotype_n_k_shift,
    #   sample_0_A0,
    #   sample_1_A0,
    #   sample_2_A0, 
    #     ...
    #   sample_m_A0]

    # theta_indexer maps unique each row to its genotype/iptg theta
    # parameter in the parameter array
    _geno_iptg_tuples = df[['genotype', 'iptg']].itertuples(index=False, name=None)
    theta_indexer, theta_rev_indexer = pd.factorize(pd.Series(_geno_iptg_tuples))
    
    # kshift_indexer maps each row in the dataframe to its k_shift
    # parameter in the parameter array
    kshift_start = np.max(theta_indexer) + 1
    _idx, _rev = pd.factorize(df["genotype"])
    kshift_indexer = _idx + kshift_start
    kshift_rev_indexer = np.full(np.max(kshift_indexer)+1,object)
    kshift_rev_indexer[kshift_start:] = _rev

    # A0_indexer maps each row to its A0 parameter in the parameter array
    A0_start = np.max(kshift_indexer) + 1
    if unique_A0:
        A0_indexer = A0_start + np.arange(num_rows,dtype=int)
        A0_rev_indexer = np.full(np.max(A0_indexer)+1,0,dtype=int)
        A0_rev_indexer[A0_start:] = np.arange(0,len(A0_indexer),dtype=int)
    else:
        _rep_geno_tuples = df[['genotype','replicate']].itertuples(index=False, name=None)
        _idx, _rev = pd.factorize(pd.Series(_rep_geno_tuples))
        A0_indexer = _idx + A0_start
        A0_rev_indexer = np.full(np.max(A0_indexer)+1,object)
        A0_rev_indexer[A0_start:] = _rev
    
    # --------------------------------------------------------------------------
    # Build guesses

    # empty guesses
    guesses = np.zeros(np.max(A0_indexer)+1,dtype=float)
    
    # Initialize theta parameters with wildtype theta
    guesses[theta_indexer] = get_wt_theta(iptg=df["iptg"],
                                          calibration_data=calibration_dict)

    # Get pre-loaded lnA0 and k shift guesses
    guesses[A0_indexer] = df["lnA_pre0_guess"]
    guesses[kshift_indexer] = df["k_shift_guess"]

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
    
    param_replicate = []
    param_replicate.extend([None for _ in theta_rev_indexer[0:]])
    param_replicate.extend([None for _ in kshift_rev_indexer[kshift_start:]])
    
    if unique_A0:
        param_genotype.extend(list(df["genotype"]))
        param_replicate.extend(list(df["replicate"]))
    else:
        param_genotype.extend([v[0] for v in A0_rev_indexer[A0_start:]])
        param_replicate.extend([v[1] for v in A0_rev_indexer[A0_start:]])
    
    param_iptg = []
    param_iptg.extend([v[1] for v in theta_rev_indexer[0:]])
    param_iptg.extend([np.nan for _ in kshift_rev_indexer[kshift_start:]])
    param_iptg.extend([np.nan for _ in A0_rev_indexer[A0_start:]])

    fit_df = pd.DataFrame({"class":param_class,
                           "genotype":param_genotype,
                           "replicate":param_replicate,
                           "iptg":param_iptg,
                           "guess":guesses})
     
    # --------------------------------------------------------------------------
    # Non-params args for the _theta_to_lncfu model
    args = (theta_indexer,
            kshift_indexer,
            A0_indexer,
            pre_select_t,
            k_wt0_noselect,
            m_noselect,
            times,
            k_wt0_select,
            m_select,
            k_shift_scaling,
            A0_scaling)

    # --------------------------------------------------------------------------
    # Do fit, either mle or map. 

    if fit_method == "mle":
    
        # Run least squares, optimizing the parameters against ln_cfu with the
        # _theta_to_lncfu model. 
        lower_bounds = np.full(len(guesses),-np.inf)
        upper_bounds = np.full(len(guesses), np.inf)
        lower_bounds[theta_indexer] = -15
        upper_bounds[theta_indexer] = 15
        
        params, std_errors, cov_matrix = run_least_squares(
            _theta_to_lncfu,
            ln_cfu.flatten(),
            np.sqrt(ln_cfu_var.flatten()),
            guesses,
            lower_bounds,
            upper_bounds,
            args
        )

    elif fit_method == "map":
        
        prior_types = np.full(len(guesses), object)
        prior_types[theta_indexer] = 'normal'
        prior_types[kshift_indexer] = 'normal'
        prior_types[A0_indexer] = 'normal'

        prior_0_params = np.zeros(len(guesses),dtype=float)
        prior_0_params[theta_indexer] = 0
        prior_0_params[kshift_indexer] = guesses[kshift_indexer]
        prior_0_params[A0_indexer] = guesses[A0_indexer]

        prior_1_params = np.zeros(len(guesses),dtype=float)
        prior_1_params[theta_indexer] = 15
        prior_1_params[kshift_indexer] = 15
        prior_1_params[A0_indexer] = 15

        prior_params = list(zip(prior_0_params,prior_1_params))

        params, cov_matrix = run_map(
            _theta_to_lncfu,
            ln_cfu.flatten(),
            np.sqrt(ln_cfu_var.flatten()),
            guesses,
            prior_types,
            prior_params,
            args
        )

        std_errors = np.sqrt(np.diag(cov_matrix))

    else: 
        err = f"fit_method '{fit_method}' not recognized. Should be 'mle' or 'map'.\n"
        raise ValueError(err)
    
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
                             "iptg":[float(v[1]) for v in theta_rev_indexer[theta_indexer]],
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
  
    pred_df = pd.DataFrame({"time":times.flatten(),
                            "obs_est":ln_cfu.flatten(),
                            "obs_std":np.sqrt(ln_cfu_var.flatten()),
                            "pred_est":pred_est,
                            "pred_std":pred_std})
    
    # Build a dataframe holding growth rates and A0 for each sample

    # Predict the growth rates under all conditions using our estimated 
    # parameters. 
    k_est, k_std = predict_with_error(
        _get_k,
        params,
        cov_matrix,
        args=[k_wt0_select,
              m_select,
              theta_indexer,
              kshift_indexer,
              k_shift_scaling]
    )

    lnA0_est = unscale(params[A0_indexer], A0_scaling)
    lnA0_std = std_errors[A0_indexer]*A0_scaling[1]

    k_df = df.copy()[["genotype","replicate","marker","select","iptg"]]
    k_df["k_est"] = k_est
    k_df["k_std"] = k_std
    k_df["lnA0_est"] = lnA0_est
    k_df["lnA0_std"] = lnA0_std

    return theta_df, growth_df, pred_df, k_df, fit_out

def counts_to_theta(combined_df,
                    sample_df,
                    calibration_data,
                    fit_method="mle",
                    max_block_size=250,
                    pseudocount=1,
                    fit_unique_A0=False,
                    pre_select_time=30):
    """
    Take read counts under different conditions and use them to estimate the 
    fractional saturation of the operator (theta) for each genotype as a
    function of inducer concentration. This estimate also yields the intrinsic
    effect of each genotype on growth rate and the initial population of each 
    genotype in each replicate. 
    
    Parameters
    ----------
    combined_df : pandas.DataFrame
        A dataframe that minimally has columns "genotype", "sample",
        "time", "counts", "total_counts_at_time", and "total_cfu_at_time".  The
        values in the "sample" column should be indexes in the sample_df 
        dataframe. The dataframe must be sorted by genotype, then sample. 
        The combined_df should be exhaustive, having all genotypes in all 
        samples. Genotypes not seen in a particular sample should still be 
        present, just given counts of zero. 
    sample_df : pandas.DataFrame
        Dataframe containing information about the samples. This function assumes
        it is indexed by the values seen in the "sample" column of the 
        combined_df, and that it minimally has columns "replicate", "marker", 
        "select", and "iptg".
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
    pseudocount : int, optional
        Pseudocount to add to CFU values before taking the logarithm. 
        Default is 1.
    fit_unique_A0 : bool, optional
        If True, fit a unique initial population (A0) for each
        genotype in each sample. If False, fit an initial population for each 
        genotype/replicate that is shared across all samples from that
        replicate. Default is False.

    Returns
    -------
    length 4 tuple
        + theta_df: pandas.DataFrame holding theta vs. iptg for each genotype
        + growth_df: pandas.DataFrame holding the intrinsic effect of each 
          genotype on growth rate
        + pred_df: pandas.DataFrame holding the observed and predicted
          ln(cfu/mL) under all conditions and times
        + k_df: pandas.DataFrame holding the A0 and k values for each genotype
    """

    to_regress_df, times, ln_cfu, ln_cfu_var = counts_to_lncfu(
        combined_df,
        sample_df,
        calibration_data,
        pseudocount=pseudocount,
        pre_select_time=pre_select_time
    )

    # Genotypes come out of replicate_aware_load sorted
    genotype_order = pd.unique(to_regress_df["genotype"])

    # This records whether we actually had data against which to do a 
    # regression. 
    enough_obs = to_regress_df.groupby("genotype",
                                       observed=False).first()["enough_obs"]
    
    # Lists to store block-wise results
    theta_dfs = []
    growth_dfs = []
    pred_dfs = []
    k_dfs = []
    fit_outs = []

    # For genotype blocks...
    block_counter = 0
    genotype_blocks = chunk_by_group(to_regress_df["genotype"],max_block_size)
    for block in tqdm(genotype_blocks):

        this_df = to_regress_df.loc[block,:]
        theta_df, growth_df, pred_df, k_df, fit_out = _run_regression(
            this_df,
            times[block,:],
            ln_cfu[block,:],
            ln_cfu_var[block,:],
            calibration_data,
            fit_method,
            fit_unique_A0,
            pre_select_time
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
    
    # Collapse theta_df to single genotype/iptg values 
    theta_df["genotype"] = pd.Categorical(theta_df["genotype"],
                                          categories=genotype_order,
                                          ordered=True)    
    theta_df = theta_df.groupby(['genotype', 'iptg'],
                                 as_index=False,
                                 observed=False).first()
    theta_df["enough_obs"] = enough_obs[theta_df["genotype"]].values

    
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
    growth_df["enough_obs"] = enough_obs[growth_df["genotype"]].values

    # -------------------------------------------------------------------------
    # Finalize pred_df

    pred_df = pd.concat(pred_dfs,ignore_index=True)
    
    # Build a dataframe holding observed and predicted ln(cfu/mL)
    pred_df["genotype"] = combined_df["genotype"].values
    pred_df["sample"] = combined_df["sample"].values
    col_to_grab = ["replicate","marker","select","iptg"]
    for col in col_to_grab:
        pred_df[col] = sample_df.loc[combined_df["sample"],col].values
    
    # Sort on genotype, replicate, marker, select, iptg
    pred_df["genotype"] = pd.Categorical(pred_df["genotype"],
                                         categories=genotype_order,
                                         ordered=True)
    sort_on = ["genotype","replicate","marker",
               "select","iptg","time","sample"]
    pred_df = pred_df.sort_values(sort_on)
    pred_df["enough_obs"] = enough_obs[pred_df["genotype"]].values

    column_order = ["genotype","replicate","marker","select","iptg","sample",
                    "time","obs_est","obs_std","pred_est","pred_std","enough_obs"]
    pred_df = pred_df.loc[:,column_order]

    # -------------------------------------------------------------------------
    # Finalize k_df

    k_df = pd.concat(k_dfs,ignore_index=True)
    k_df["enough_obs"] = enough_obs[k_df["genotype"]].values
    
    
    # -------------------------------------------------------------------------
    # Finalize fit_out

    fit_out = {}
    fit_out["fit_df"] = pd.concat([f["fit_df"] for f in fit_outs],
                                  ignore_index=True)
    cov_matrices = [f["cov_matrix"] for f in fit_outs]
    fit_out["cov_matrix"] = scipy.linalg.block_diag(*cov_matrices)

    
    return theta_df, growth_df, pred_df, k_df, fit_out
