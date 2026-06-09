"""
Functions for plotting per-genotype growth trajectory predictions.

plot_geno_trajectory
    Pure plotting function.  Accepts a prediction DataFrame and returns a
    ``matplotlib.figure.Figure``.

predict_and_plot_geno_trajectory
    Composite entry point.  Calls ``predict()`` with a fine t_sel grid, merges
    in observed data and a pre-selection anchor from the ln_cfu0 site, then
    delegates to ``plot_geno_trajectory``.
"""

import numpy as np
import pandas as pd

_T_FINE = 50

_CONDITION_COLS = [
    "condition_pre", "condition_sel", "titrant_name", "titrant_conc"
]
_KEEP_COLS = [
    "replicate", "condition_pre", "condition_sel",
    "titrant_name", "titrant_conc", "genotype",
    "t_sel", "ln_cfu", "ln_cfu_std", "q05", "median", "q95",
]


def plot_geno_trajectory(
    pred_df,
    figsize=None,
    colors=None,
    markers=None,
):
    """
    Plot per-genotype growth trajectories from a prediction DataFrame.

    One subplot is generated per unique
    ``(condition_pre, condition_sel, titrant_name, titrant_conc)`` combination.
    Every ``(genotype, replicate)`` pair present in ``pred_df`` is drawn as a
    separate series on each subplot.  Genotypes are distinguished by color;
    replicates are distinguished by marker and linestyle.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame.  Required columns:

        * ``replicate``, ``condition_pre``, ``condition_sel``,
          ``titrant_name``, ``titrant_conc``, ``genotype`` — grouping keys.
        * ``t_sel`` — time coordinate (may include negative values for the
          pre-selection phase).
        * ``ln_cfu`` — observed ln(CFU); ``NaN`` where no observation exists.
        * ``ln_cfu_std`` — observed std; ``NaN`` where no observation exists.
        * ``median`` — predicted median trajectory; ``NaN`` where unavailable.

        Optional columns (both must be present together to draw a credible
        interval):

        * ``q05``, ``q95`` — 5th/95th percentile of the posterior predictive.

    figsize : tuple of (float, float), optional
        Figure ``(width, height)`` in inches.  Defaults to
        ``(5 * n_cols, 4 * n_rows)``.
    colors : list of str, optional
        Color cycle for genotypes.  Defaults to ``DEFAULT_COLORS``.
    markers : list of str, optional
        Marker cycle for replicates.  Defaults to ``DEFAULT_MARKERS``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    from tfscreen.plot.default_styles import DEFAULT_COLORS, DEFAULT_MARKERS

    if colors is None:
        colors = DEFAULT_COLORS[:]
    if markers is None:
        markers = DEFAULT_MARKERS[:]

    conditions = (
        pred_df[_CONDITION_COLS]
        .drop_duplicates()
        .sort_values(_CONDITION_COLS)
        .reset_index(drop=True)
    )
    genotypes = sorted(pred_df["genotype"].unique().tolist(), key=str)
    replicates = sorted(pred_df["replicate"].unique().tolist(), key=str)

    n_combos = len(conditions)
    n_cols = min(3, n_combos)
    n_rows = (n_combos + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharey=True
    )

    # One color per (genotype, replicate) pair
    series_pairs = [
        (g, r) for g in genotypes for r in replicates
    ]
    pair_color = {
        pair: colors[i % len(colors)] for i, pair in enumerate(series_pairs)
    }
    rep_marker = {r: markers[i % len(markers)] for i, r in enumerate(replicates)}

    has_ci = {"q05", "q95"} <= set(pred_df.columns)

    for combo_i, cond_row in conditions.iterrows():
        ax = axes[combo_i // n_cols][combo_i % n_cols]

        cp = cond_row["condition_pre"]
        cs = cond_row["condition_sel"]
        tn = cond_row["titrant_name"]
        tc = float(cond_row["titrant_conc"])

        cond_df = pred_df[
            (pred_df["condition_pre"] == cp)
            & (pred_df["condition_sel"] == cs)
            & (pred_df["titrant_name"] == tn)
            & (pred_df["titrant_conc"] == tc)
        ].copy()

        ax.set_title(f"{cp} → {cs}\n{tn} = {tc:.3g}", fontsize=9)
        ax.set_xlabel("Time")
        ax.set_ylabel("ln(CFU)")
        ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for geno in genotypes:
            geno_df = cond_df[cond_df["genotype"] == geno]
            if geno_df.empty:
                continue

            for rep in replicates:
                rep_df = (
                    geno_df[geno_df["replicate"] == rep]
                    .sort_values("t_sel")
                )
                if rep_df.empty:
                    continue

                color = pair_color[(geno, rep)]
                label = f"{geno} ({rep})"
                mk = rep_marker[rep]
                ls = "-"

                # Observed data points
                obs = rep_df.dropna(subset=["ln_cfu"])
                if not obs.empty:
                    ax.errorbar(
                        obs["t_sel"],
                        obs["ln_cfu"],
                        yerr=obs["ln_cfu_std"],
                        fmt=mk,
                        color=color,
                        ms=5,
                        lw=1,
                        capsize=3,
                        zorder=3,
                    )

                # Predicted median line
                line_df = rep_df.dropna(subset=["median"])
                if not line_df.empty:
                    ax.plot(
                        line_df["t_sel"],
                        line_df["median"],
                        ls,
                        color=color,
                        lw=1.8,
                        label=label,
                        zorder=4,
                    )

                # 90 % credible interval band
                if has_ci:
                    ci_df = rep_df.dropna(subset=["q05", "q95"])
                    if not ci_df.empty:
                        ax.fill_between(
                            ci_df["t_sel"],
                            ci_df["q05"],
                            ci_df["q95"],
                            color=color,
                            alpha=0.2,
                            zorder=2,
                        )

        ax.legend(fontsize=8, loc="best")

    for extra_i in range(n_combos, n_rows * n_cols):
        axes[extra_i // n_cols][extra_i % n_cols].set_visible(False)

    fig.tight_layout()
    return fig


def predict_geno_trajectory_df(
    orchestrator,
    param_posteriors,
    genotypes=None,
    titrant_names=None,
    t_fine=_T_FINE,
    num_marginal_samples=200,
):
    """
    Run forward predictions and return the merged trajectory DataFrame.

    Calls :func:`~tfscreen.tfmodel.analysis.prediction.predict` with a fine
    ``t_sel`` grid, merges in observed data and a pre-selection anchor from
    the ``ln_cfu0`` site, and returns a single DataFrame ready for
    :func:`plot_geno_trajectory`.  Only ``(condition_pre, condition_sel,
    titrant_name, titrant_conc)`` combinations present in
    ``orchestrator.growth_df`` are included.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Fitted model orchestrator.
    param_posteriors : dict or str
        MAP parameter dict (keys ending in ``_auto_loc``), path to a
        ``_params.npz`` file, or path to a ``_posterior.h5`` file.
    genotypes : list of str, optional
        Subset of genotypes to include.  If ``None``, uses all genotypes in
        the orchestrator.
    titrant_names : list of str, optional
        Subset of titrant names to include.  If ``None``, uses all.
    t_fine : int, optional
        Number of equally-spaced selection-phase time points for the fine grid.
        Defaults to ``_T_FINE`` (50).
    num_marginal_samples : int, optional
        Number of posterior samples to run through the model when computing
        quantiles.  Passed directly to :func:`~.prediction.predict`.
        Defaults to 200.

    Returns
    -------
    pd.DataFrame
        Merged prediction DataFrame with columns matching ``_KEEP_COLS``.
    """
    from tfscreen.tfmodel.analysis.prediction import predict

    if isinstance(param_posteriors, str) and param_posteriors.endswith(".npz"):
        param_posteriors = dict(np.load(param_posteriors))

    # Fine selection-phase time grid: 0 … max observed t_sel
    gd = orchestrator.data.growth
    good_mask = np.asarray(gd.good_mask)
    t_sel_tensor = np.asarray(gd.t_sel)
    max_t_sel = float(np.nanmax(t_sel_tensor[good_mask]))
    t_sel_grid = np.linspace(0.0, max_t_sel, t_fine).tolist()

    q_spec = {"q05": 0.05, "median": 0.5, "q95": 0.95}

    all_dfs = predict(
        orchestrator,
        param_posteriors,
        predict_sites=["growth_pred", "ln_cfu0"],
        q_to_get=q_spec,
        num_samples=None,
        num_marginal_samples=num_marginal_samples,
        t_sel=t_sel_grid,
        genotypes=genotypes,
    )
    fine_df = all_dfs["growth_pred"]
    ln_cfu0_raw = all_dfs["ln_cfu0"]

    # Build anchor rows at t_sel = -t_pre using ln_cfu0 quantiles.
    # ln_cfu0 is indexed by (replicate, condition_pre, genotype); expand across
    # all valid (condition_sel, titrant_name, titrant_conc) for each condition_pre.
    t_pre_df = (
        orchestrator.growth_df[["replicate", "condition_pre", "t_pre"]]
        .drop_duplicates(subset=["replicate", "condition_pre"])
    )
    ln_cfu0_vals = (
        ln_cfu0_raw[["replicate", "condition_pre", "genotype", "q05", "median", "q95"]]
        .drop_duplicates(subset=["replicate", "condition_pre", "genotype"])
    )
    valid_combos = orchestrator.growth_df[_CONDITION_COLS].drop_duplicates()
    anchor_df = (
        ln_cfu0_vals
        .merge(valid_combos, on="condition_pre", how="inner")
        .merge(t_pre_df, on=["replicate", "condition_pre"], how="left")
    )
    anchor_df["t_sel"] = -anchor_df["t_pre"]
    anchor_df["ln_cfu"] = np.nan
    anchor_df["ln_cfu_std"] = np.nan

    # Overlay observed ln_cfu0 measurements when presplit_df is available.
    presplit_df = getattr(orchestrator, "presplit_df", None)
    if presplit_df is not None:
        ps = presplit_df[
            ["replicate", "condition_pre", "genotype", "ln_cfu", "ln_cfu_std"]
        ].rename(columns={"ln_cfu": "_ps_ln", "ln_cfu_std": "_ps_ln_std"})
        anchor_df = anchor_df.merge(
            ps, on=["replicate", "condition_pre", "genotype"], how="left"
        )
        anchor_df["ln_cfu"] = anchor_df["_ps_ln"]
        anchor_df["ln_cfu_std"] = anchor_df["_ps_ln_std"]
        anchor_df = anchor_df.drop(columns=["_ps_ln", "_ps_ln_std"])

    # Observed data from the orchestrator (no model predictions)
    obs_cols = [
        "replicate", "condition_pre", "condition_sel",
        "titrant_name", "titrant_conc", "genotype",
        "t_sel", "ln_cfu", "ln_cfu_std",
    ]
    obs_df = orchestrator.growth_df[obs_cols].copy()
    if genotypes is not None:
        obs_df = obs_df[obs_df["genotype"].isin(list(genotypes))]
    for col in ("q05", "median", "q95"):
        obs_df[col] = np.nan

    # Titrant-name filter (predict() has no titrant_name argument)
    if titrant_names is not None:
        tn_set = {str(t) for t in titrant_names}
        fine_df = fine_df[fine_df["titrant_name"].isin(tn_set)]
        obs_df = obs_df[obs_df["titrant_name"].isin(tn_set)]
        anchor_df = anchor_df[anchor_df["titrant_name"].isin(tn_set)]

    pred_df = pd.concat(
        [fine_df[_KEEP_COLS], obs_df[_KEEP_COLS], anchor_df[_KEEP_COLS]],
        ignore_index=True,
    )

    # Restrict to condition combos that actually exist in the training data.
    # copy_orchestrator() produces a Cartesian product of categorical columns
    # that can include invalid (condition_pre, condition_sel) pairings; this
    # filter removes them.
    valid_combos = (
        orchestrator.growth_df[_CONDITION_COLS]
        .drop_duplicates()
    )
    pred_df = pred_df.merge(valid_combos, on=_CONDITION_COLS, how="inner")
    return pred_df


def predict_and_plot_geno_trajectory(
    orchestrator,
    param_posteriors,
    genotypes=None,
    titrant_names=None,
    t_fine=_T_FINE,
    num_marginal_samples=200,
    figsize=None,
    colors=None,
    markers=None,
):
    """
    Run forward predictions and plot per-genotype growth trajectories.

    Calls :func:`predict_geno_trajectory_df` to build the merged prediction
    DataFrame, then delegates to :func:`plot_geno_trajectory`.  Only
    ``(condition_pre, condition_sel, titrant_name, titrant_conc)`` combinations
    present in ``orchestrator.growth_df`` are included in the output.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Fitted model orchestrator.
    param_posteriors : dict or str
        MAP parameter dict (keys ending in ``_auto_loc``), path to a
        ``_params.npz`` file, or path to a ``_posterior.h5`` file.
    genotypes : list of str, optional
        Subset of genotypes to include.  If ``None``, uses all genotypes in
        the orchestrator.
    titrant_names : list of str, optional
        Subset of titrant names to include.  If ``None``, uses all.
    t_fine : int, optional
        Number of equally-spaced selection-phase time points for the fine grid.
        Defaults to ``_T_FINE`` (50).
    num_marginal_samples : int, optional
        Number of posterior samples to run through the model when computing
        quantiles.  Passed directly to :func:`~.prediction.predict`.
        Defaults to 200.
    figsize : tuple of (float, float), optional
        Passed to :func:`plot_geno_trajectory`.
    colors : list of str, optional
        Passed to :func:`plot_geno_trajectory`.
    markers : list of str, optional
        Passed to :func:`plot_geno_trajectory`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pred_df = predict_geno_trajectory_df(
        orchestrator,
        param_posteriors,
        genotypes=genotypes,
        titrant_names=titrant_names,
        t_fine=t_fine,
        num_marginal_samples=num_marginal_samples,
    )
    return plot_geno_trajectory(
        pred_df, figsize=figsize, colors=colors, markers=markers
    )
