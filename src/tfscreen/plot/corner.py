import corner
import numpy as np

def corner_plot(fit_df,
                cov_matrix,
                plot_mask=None,
                num_samples=10000,
                max_allowed_param=10):
    """
    Generate correlation plot illustrating covariance in the estimates between
    fit parameters. WARNING: going above about ~10 parameters gets slow and 
    hard to visualize. 

    Parameters
    ----------
    fit_df : pd.DataFrame
        dataframe with fit parameter information. expects columns 
        "est', "genotype", "class","iptg", and "block". indexes are expected to 
        be integers that match dimensions of the covariance matrix. 
    cov_matrix : np.ndarray (or scipy.sparse)
        covariance matrix for the fit parameters.
    plot_mask : np.ndarray, optional
        boolean array that pulls parameter information from rows in df and
        indexes out of cov_matrix. if None, take all rows.
    num_samples : int, default=100000
        generate this number of samples to build covariance plot. the greater the
        number of samples, the smoother the distribution
    max_allowed_param : int, default = 10
        only plot correlations for this many parameters

    Returns
    -------
    fig : matplotlib.Figure
        figure holding the corner plot
    """


    if plot_mask is not None:
        fit_df = fit_df.loc[plot_mask,:]

    if len(fit_df) > max_allowed_param:
        err = "too many rows to make a corner plot. stopping to avoid a memory leak\n"
        raise RuntimeError(err)
    
    slicer = fit_df.index.values
    
    cov_matrix = cov_matrix[np.ix_(slicer,slicer)]

    est_values = fit_df["est"].values

    param_df = fit_df.loc[:,["genotype","class","iptg","block"]] 
    param_df["iptg"] = param_df["iptg"].astype(str)
    param_df["block"] = param_df["block"].astype(str)
    param_names = list(param_df.agg("_".join,axis=1))
    
    samples = np.random.multivariate_normal(mean=est_values,
                                            cov=cov_matrix,
                                            size=num_samples)
    
    fig = corner.corner(samples,
                        truths=est_values,
                        labels=param_names)

    return fig