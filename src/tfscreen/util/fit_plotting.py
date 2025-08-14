import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

import copy

DEFAULT_SCATTER_KWARGS = {"s":10,
                          "alpha":0.1,
                          "edgecolor":"royalblue",
                          "facecolor":"none"}

def plot_corr(x_values,
              y_values,
              ax=None,
              scatter_kwargs=None,
              scale_by=0.01):

    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))

    final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
    if scatter_kwargs is not None:
        for k in scatter_kwargs:
            final_scatter_kwargs[k] = scatter_kwargs[k]
    
    ax_min = np.min([np.min(x_values),np.min(y_values)])
    ax_max = np.max([np.max(x_values),np.max(y_values)])
    span = ax_max - ax_min
    ax_min = ax_min - scale_by*span
    ax_max = ax_max + scale_by*span
    
    ax.scatter(x_values,
               y_values,
               **final_scatter_kwargs)
    ax.plot([ax_min,ax_max],[ax_min,ax_max],'--',color='gray',lw=2,zorder=5)
    
    ax.set_xlim(ax_min,ax_max)
    ax.set_ylim(ax_min,ax_max)
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_err(real_values,
               est_values,
               est_err,
               plot_as_real_err=False,
               range_mask=None,
               ax=None,
               scatter_kwargs=None,
               pct_cut=0.01):

    if range_mask is None:
        range_mask = np.ones(len(real_values),dtype=bool)

    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))

    final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
    if scatter_kwargs is not None:
        for k in scatter_kwargs:
            final_scatter_kwargs[k] = scatter_kwargs[k]

    diff = (est_values - real_values)[range_mask] 
    err = est_err[range_mask]

    if not plot_as_real_err:
        normalize = np.abs(real_values[range_mask])
        diff = diff/normalize
        err = err/normalize

    abs_diffs = np.abs(diff)
    abs_diffs.sort()
    pct_cut_index = int(np.round(len(abs_diffs)*(1-pct_cut),0))
    ax_max = abs_diffs[pct_cut_index]

    ax.scatter(diff,err,**final_scatter_kwargs)
    ax.set_xlabel("est_value - real_value")
    ax.set_ylabel("est_err")
    ax.plot([0,0],[0,2*ax_max],'--',color='gray',zorder=-20)

    for i in range(1,4):
        ax.plot([0,ax_max],[0,ax_max/i],'--',color='gray',zorder=-20)
        ax.plot([-ax_max,0],[ax_max/i,0],'--',color='gray',zorder=-20)
    

    ax.set_xlim(-ax_max,ax_max)
    ax.set_ylim(0,ax_max*2)
    ax.set_aspect('equal', adjustable='box')

    return ax

def plot_err_zscore(real_values,
                    est_values,
                    est_std,
                    z_min=-8,
                    z_max=8,
                    step_size=0.1,
                    ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))
    
    input_bins = np.arange(z_min,z_max + step_size,step_size)
    counts, bins = np.histogram((est_values - real_values)/(est_std),bins=input_bins)
        
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
    ax.set_ylabel("PDF")
    ax.legend()

    return ax


def plot_summary(k_est,
                 k_std,
                 k_real,
                 suptitle=None,
                 subsample=10000):

    good_mask = np.logical_not(np.isnan(k_est))

    k_est = k_est[good_mask]
    k_std = k_std[good_mask]
    k_real = k_real[good_mask]
    
    if subsample is not None:
        index = np.arange(len(k_est),dtype=int)
        index = np.random.choice(index,size=subsample,replace=False)
    
    fig, ax = plt.subplots(1,3,figsize=(14,6))
    
    plot_corr(k_real[index],
              k_est[index],
              ax=ax[0])

    ax[0].set_xlabel("k_real")
    ax[0].set_ylabel("k_est")
    
    plot_err(k_real[index],
             k_est[index],
             k_std[index],
             ax=ax[1],plot_as_real_err=True)

    ax[1].set_xlabel("k_est - k_real")
    ax[1].set_ylabel("k_std")

    plot_err_zscore(k_real,
                    k_est,
                    k_std,
                    ax=ax[2])

    ax[2].set_xlabel("Z-score (k_est - k_real)")
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    fig.tight_layout()
                     
    return fig, ax