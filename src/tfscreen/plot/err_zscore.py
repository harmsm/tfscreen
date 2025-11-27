from tfscreen.plot.helper import (
    clean_arrays,
)

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def err_zscore(
        param_est,
        param_std,
        param_real,
        z_min=-8,
        z_max=8,
        step_size=0.1,
        ax=None
    ):
    """
    Plot the distribution of Z-scores for the error between estimated and real
    values. If the model is well calibrated, the solid red line will exactly 
    track with the histogram values. 

    Parameters
    ----------
    real_values : array_like
        The true values.
    est_values : array_like
        The estimated values.
    est_std : array_like
        The standard error on the estimated parameters.
    z_min : float, optional
        The minimum Z-score to plot.
    z_max : float, optional
        The maximum Z-score to plot.
    step_size : float, optional
        The step size for the histogram bins.
    ax : matplotlib.axes._axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The axes object with the plot.
    """

    param_est, param_std, param_real = clean_arrays(param_est,
                                                    param_std,
                                                    param_real)

    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))
    
    input_bins = np.arange(z_min,z_max + step_size,step_size)

    obs_z_score = (param_est - param_real)/(param_std)
    counts, bins = np.histogram(obs_z_score,bins=input_bins)
        
    centers = (bins[1:] - bins[:-1])/2 + bins[:-1]
    freq = counts/np.sum(counts)

    # Normalized gaussian PDF
    pdf = stats.norm.pdf(centers)
    pdf = pdf/np.sum(pdf)

    # Draw observed histogram
    for i in range(len(freq)):
        if i == 0:
            label = "observed"
        else:
            label = None
        
        ax.fill([bins[i],bins[i],bins[i+1],bins[i+1]],
                [0,freq[i],freq[i],0],
                facecolor='lightgray',
                edgecolor='gray',
                label=label)
    
    ax.plot(centers,pdf,lw=3,color='red',label="perfect calibration")
    ax.set_xlabel('z-score')
    ax.set_ylabel("probability density function")
    ax.legend()

    return ax
