
from tfscreen.plot.default_styles import (
        DEFAULT_SCATTER_KWARGS,
        DEFAULT_FIT_LINE_KWARGS
)

from matplotlib import pyplot as plt

import copy

def hill_fit(obs_df,
			 pred_df,
			 genotype,
			 zero_iptg_value=1e-6,
			 scatter_kwargs=None,
			 fit_line_kwargs=None,
			 ax=None):
  
	this_obs_df = obs_df[obs_df["genotype"] == genotype]
	this_obs_df.loc[this_obs_df["iptg"] == 0,"iptg"] = zero_iptg_value
	this_pred_df = pred_df[pred_df["genotype"] == genotype]

	if ax is None:
		_, ax = plt.subplots(1,figsize=(6,6))

	final_scatter_kwargs = copy.deepcopy(DEFAULT_SCATTER_KWARGS)
	final_scatter_kwargs["alpha"] = 1 # override default alpha = 0.1
	if scatter_kwargs is not None:
		for k in scatter_kwargs:
			final_scatter_kwargs[k] = scatter_kwargs[k]

	final_fit_line_kwargs = copy.deepcopy(DEFAULT_FIT_LINE_KWARGS)
	if fit_line_kwargs is not None:
		for k in fit_line_kwargs:
			final_fit_line_kwargs[k] = fit_line_kwargs[k]


	# Plot error bars
	ax.errorbar(this_obs_df["iptg"],
				this_obs_df["theta_est"],
				this_obs_df["theta_std"],
				lw=0,
				elinewidth=1,
				capsize=5,
				zorder=10)
	ax.scatter(this_obs_df["iptg"],
			   this_obs_df["theta_est"],
			   **final_scatter_kwargs)

	ax.plot(this_pred_df["iptg"],
			this_pred_df["hill_est"],
			"-",
			zorder=30,
			**final_fit_line_kwargs)

	ax.fill_between(this_pred_df["iptg"],
					this_pred_df["hill_est"] - this_pred_df["hill_std"],
					this_pred_df["hill_est"] + this_pred_df["hill_std"],
					color="lightgray",
					zorder=0)

	ax.set_xscale('log')
	ax.set_xlabel("[iptg] (mM)")
	ax.set_ylabel("fractional occupancy")
	ax.set_ylim(0,1)

	return ax