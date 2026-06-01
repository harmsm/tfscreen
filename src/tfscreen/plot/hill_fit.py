
from tfscreen.plot.default_styles import (
        DEFAULT_SCATTER_KWARGS,
        DEFAULT_FIT_LINE_KWARGS
)

from matplotlib import pyplot as plt

def hill_fit(obs_df,
             pred_df,
             genotype,
             zero_titrant_value=1e-6,
             scatter_kwargs=None,
             fit_line_kwargs=None,
             ax=None):
    """
    Plot observed theta values and a Hill-function fit for one genotype.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Observed data.  Required columns: ``genotype``, ``titrant_conc``,
        ``theta_est``, ``theta_std``.
    pred_df : pandas.DataFrame
        Model predictions.  Required columns: ``genotype``, ``titrant_conc``,
        ``hill_est``, ``hill_std``.
    genotype : str
        Genotype label used to filter both ``obs_df`` and ``pred_df``.
    zero_titrant_value : float, optional
        Replacement value for zero concentrations so the log-scaled x-axis
        does not break.  Defaults to ``1e-6``.
    scatter_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.scatter``.  Override the defaults
        in ``default_styles.DEFAULT_SCATTER_KWARGS``.
    fit_line_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.plot`` for the Hill fit line.
        Override the defaults in ``default_styles.DEFAULT_FIT_LINE_KWARGS``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new 6×6 figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the finished plot.
    """

    this_obs_df = obs_df[obs_df["genotype"] == genotype].copy()
    this_obs_df["titrant_conc"] = this_obs_df["titrant_conc"].astype(float)
    this_obs_df.loc[this_obs_df["titrant_conc"] == 0, "titrant_conc"] = zero_titrant_value
    this_pred_df = pred_df[pred_df["genotype"] == genotype]

    if ax is None:
        _, ax = plt.subplots(1, figsize=(6, 6))

    final_scatter_kwargs = DEFAULT_SCATTER_KWARGS | {"alpha": 1} | (scatter_kwargs or {})
    final_fit_line_kwargs = DEFAULT_FIT_LINE_KWARGS | (fit_line_kwargs or {})

    # Plot error bars
    ax.errorbar(this_obs_df["titrant_conc"],
                this_obs_df["theta_est"],
                this_obs_df["theta_std"],
                lw=0,
                elinewidth=1,
                capsize=5,
                zorder=10)
    ax.scatter(this_obs_df["titrant_conc"],
               this_obs_df["theta_est"],
               **final_scatter_kwargs)

    ax.plot(this_pred_df["titrant_conc"],
            this_pred_df["hill_est"],
            "-",
            zorder=30,
            **final_fit_line_kwargs)

    ax.fill_between(this_pred_df["titrant_conc"],
                    this_pred_df["hill_est"] - this_pred_df["hill_std"],
                    this_pred_df["hill_est"] + this_pred_df["hill_std"],
                    color="lightgray",
                    zorder=0)

    ax.set_xscale('log')
    ax.set_xlabel("[titrant] (mM)")
    ax.set_ylabel("fractional occupancy")
    ax.set_ylim(0, 1)

    return ax
