import numpy as np
from matplotlib import pyplot as plt
import tfscreen
from tfscreen.plot import default_styles

def plot_theta_fits(df,
                    title=None,
                    ax=None,
                    scatter_kwargs=None,
                    colors=None,
                    markers=None):
    """
    Plot observed theta values with model median and optional uncertainty bands.

    Each unique (genotype, titrant_name) pair is drawn as a separate series
    with its own color and marker, cycling through the ``colors`` and
    ``markers`` lists.  The x-axis is always log-scaled; zero-concentration
    points are replaced with ``min_positive / 100`` so they appear on the
    axis without error.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot.  Required columns:

        * ``genotype`` — genotype label (string).
        * ``titrant_name`` — titrant identifier (string).
        * ``titrant_conc`` — titrant concentration (float, ≥ 0).
        * ``theta_obs`` — observed operator occupancy.
        * ``theta_std`` — standard deviation of ``theta_obs``; used for
          error bars.
        * ``median`` — posterior/model median of theta; drawn as a line.

        Optional columns (each pair must be present together):

        * ``lower_std``, ``upper_std`` — ±1 SD credible band (filled).
        * ``lower_95``, ``upper_95`` — 95 % credible band (filled).
    title : str, optional
        Axes title.  If ``None``, no title is set.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new 6×6 figure is created when ``None``.
    scatter_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.scatter`` for the observed
        data points.  These override the defaults in
        ``default_styles.DEFAULT_EXPT_SCATTER_KWARGS``; ``edgecolor`` is
        always overridden by the per-series color.
    colors : list of str, optional
        Cycle of hex/named colors for each series.  Defaults to
        ``default_styles.DEFAULT_COLORS``.
    markers : list of str, optional
        Cycle of marker codes for each series.  Defaults to
        ``default_styles.DEFAULT_MARKERS``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the finished plot.

    Notes
    -----
    Zero-concentration values are replaced in a *copy* of ``df`` and do not
    modify the caller's DataFrame.
    """

    # Deal with user scatter kwargs
    if scatter_kwargs is None:
        scatter_kwargs = {}
    # defaults first so that user-supplied values take precedence
    scatter_kwargs = default_styles.DEFAULT_EXPT_SCATTER_KWARGS | scatter_kwargs
    # edgecolor is controlled per-series via colors; drop any stale value
    if "edgecolor" in scatter_kwargs:
        scatter_kwargs.pop("edgecolor")

    # Deal with user colors and markers
    if colors is None:
        colors = default_styles.DEFAULT_COLORS[:]
    if markers is None:
        markers = default_styles.DEFAULT_MARKERS[:]
    
    # make sure genotypes are categorical and sort
    df = (tfscreen.genetics.set_categorical_genotype(df)
          .sort_values(["genotype","titrant_name","titrant_conc"])
          .reset_index(drop=True))
    
    # Set 0 values to lowest / 100
    titrant_conc_floor = np.nanmin(df["titrant_conc"][df["titrant_conc"]>0])/100
    df.loc[df["titrant_conc"] == 0,"titrant_conc"] = titrant_conc_floor

    # expand color and markers lists
    num_series = len(df.drop_duplicates(["genotype","titrant_name"]))
    colors = num_series*list(colors)
    markers = num_series*list(markers)
    
    # Create ax if needed
    if ax is None:
        _, ax = plt.subplots(1,figsize=(6,6))
    
    # Go through groups and plot
    counter = 0
    for g, g_df in df.groupby(["genotype","titrant_name"],observed=True):
        
        series_name = f"{g[0]} ({g[1]})"
        ax.scatter(g_df["titrant_conc"],g_df["theta_obs"],
                   marker=markers[counter],
                   edgecolor=colors[counter],
                   **scatter_kwargs,zorder=10,label=series_name)
        ax.errorbar(x=g_df["titrant_conc"],
                    y=g_df["theta_obs"],
                    yerr=g_df["theta_std"],
                    lw=0,
                    elinewidth=1,
                    capsize=5,
                    color=colors[counter],zorder=0)

        # Median
        ax.plot(g_df["titrant_conc"],
                g_df["median"],
                lw=2,color=colors[counter])

        # Standard error
        if set(["lower_std","upper_std"]) <= set(g_df.columns):
            
            ax.fill_between(x=g_df["titrant_conc"],
                            y1=g_df["lower_std"],
                            y2=g_df["upper_std"],
                            color=colors[counter],
                            alpha=0.7,zorder=-10)

        # 95% CI
        if set(["lower_95","upper_95"]) <= set(g_df.columns):
        
            ax.fill_between(x=g_df["titrant_conc"],
                            y1=g_df["lower_95"],
                            y2=g_df["upper_95"],
                            color=colors[counter],
                            alpha=0.4,zorder=-20)
        counter += 1
    
    ax.set_xscale("log")
    ax.set_xlabel("titrant conc (mM)")
    ax.set_ylabel("$\\theta$")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title is not None:
        ax.set_title(title)

    return ax

