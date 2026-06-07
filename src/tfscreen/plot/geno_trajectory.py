"""
Public function for plotting per-genotype growth trajectory fits overlaid
on experimental data.

The core value of :func:`plot_geno_trajectory` is the interpolative step:
it runs the forward model on a fine time grid to produce smooth trajectory
curves that can be compared visually against the raw ln_cfu observations.
Simpler downstream diagnostics (correlation plots, goodness-of-fit statistics)
are intentionally left to the caller so they can be combined or skipped as
needed.
"""

import sys

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# Number of equally-spaced time points used for the fine-grid selection-phase
# trajectory in the MAP prediction path.
_T_FINE = 50


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _compute_map_predictions(gm, params):
    """
    Run the forward model at the MAP point and return prediction tensors.

    Two forward passes are performed: one at the observed timepoints and one
    on a :data:`_T_FINE`-point fine grid spanning ``[0, global_max_t_sel]``.
    The fine-grid pass is used to draw smooth selection-phase trajectory curves.

    Parameters
    ----------
    gm : ModelOrchestrator
        Fitted model orchestrator.
    params : dict
        MAP parameter dict with ``{site}_auto_loc`` keys (unconstrained space).

    Returns
    -------
    growth_pred : np.ndarray
        Shape ``(R, T, CP, CS, TN, TC, G)`` — predictions at observed timepoints.
    ln_cfu0_map : np.ndarray or None
        Shape ``(R, CP, G)`` — MAP ``ln_cfu0`` deterministic site, or ``None``
        if absent from the trace.
    growth_pred_fine : np.ndarray
        Shape ``(R, T_FINE, CP, CS, TN, TC, G)`` — predictions on the fine grid.
    t_fine_1d : np.ndarray
        Shape ``(T_FINE,)`` — time values for ``growth_pred_fine``.

    Raises
    ------
    ValueError
        If ``growth_pred`` is not present in the model trace.
    """
    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoDelta

    guide = AutoDelta(gm.jax_model)
    all_indices = jnp.arange(gm.data.num_genotype)
    full_data = gm.get_batch(gm.data, all_indices)
    pred_fn = Predictive(gm.jax_model, guide=guide, params=params, num_samples=1)
    map_samples = pred_fn(
        jax.random.PRNGKey(0), data=full_data, priors=gm.priors
    )

    if "growth_pred" not in map_samples:
        raise ValueError(
            "growth_pred not found in the model trace; "
            "cannot generate trajectory plots."
        )

    # Remove leading num_samples dimension (num_samples=1).
    growth_pred = np.asarray(map_samples["growth_pred"][0])  # (R,T,CP,CS,TN,TC,G)
    ln_cfu0_map = (
        np.asarray(map_samples["ln_cfu0"][0])
        if "ln_cfu0" in map_samples
        else None
    )

    # --- Fine-grid forward pass ---
    gd = gm.data.growth
    good_mask    = np.asarray(gd.good_mask)        # (R, T, CP, CS, TN, TC, G)
    t_sel_tensor = np.asarray(gd.t_sel)
    n_rep, n_t, n_cp, n_cs, n_tn, n_tc, n_geno = good_mask.shape

    global_max_t_sel = float(np.nanmax(t_sel_tensor[good_mask]))
    t_fine_1d  = np.linspace(0.0, global_max_t_sel, _T_FINE)
    fine_shape = (n_rep, _T_FINE, n_cp, n_cs, n_tn, n_tc, n_geno)

    # Broadcast a 7-D array from T=1 to T_FINE; all per-condition tensors are
    # constant along the T axis so slicing any timepoint is exact.
    def _bc_t(arr, *, dtype=None):
        a = np.asarray(arr)
        r = np.broadcast_to(a[:, 0:1, ...], fine_shape).copy()
        return jnp.array(r if dtype is None else r.astype(dtype))

    t_sel_fine = np.broadcast_to(
        t_fine_1d[None, :, None, None, None, None, None], fine_shape
    ).copy()
    # Active wherever any observed timepoint was valid for that cell.
    has_data_bc    = good_mask.any(axis=1, keepdims=True)
    good_mask_fine = np.broadcast_to(has_data_bc, fine_shape).copy()

    gd_full = full_data.growth
    fine_gd = gd_full.replace(
        num_time=_T_FINE,
        t_sel=jnp.array(t_sel_fine),
        t_pre=_bc_t(gd_full.t_pre),
        good_mask=jnp.array(good_mask_fine),
        map_condition_pre=_bc_t(gd_full.map_condition_pre, dtype=int),
        map_condition_sel=_bc_t(gd_full.map_condition_sel, dtype=int),
        ln_cfu=jnp.zeros(fine_shape),
        ln_cfu_std=jnp.ones(fine_shape),
    )
    fine_data = full_data.replace(growth=fine_gd)
    fine_samples = pred_fn(jax.random.PRNGKey(1), data=fine_data, priors=gm.priors)
    growth_pred_fine = np.asarray(fine_samples["growth_pred"][0])

    return growth_pred, ln_cfu0_map, growth_pred_fine, t_fine_1d


def _compute_posterior_predictions(gm, posterior_file):
    """
    Load ``growth_pred`` from a posterior HDF5 file and compute summary statistics.

    The posterior file must have been produced by
    ``RunInference.get_posteriors()``, ``get_laplace_posteriors()``, or
    ``get_map_posteriors()``.

    Parameters
    ----------
    gm : ModelOrchestrator
        Fitted model orchestrator (used only for shape validation; not called).
    posterior_file : str or h5py.File
        Path to an HDF5 posterior file or an already-open ``h5py.File`` handle.

    Returns
    -------
    growth_pred_median : np.ndarray
        Shape ``(R, T, CP, CS, TN, TC, G)`` — median across posterior samples.
    growth_pred_lo : np.ndarray
        Shape ``(R, T, CP, CS, TN, TC, G)`` — 5th-percentile (lower bound of
        90 % credible interval).
    growth_pred_hi : np.ndarray
        Shape ``(R, T, CP, CS, TN, TC, G)`` — 95th-percentile (upper bound).
    ln_cfu0_median : np.ndarray or None
        Shape ``(R, CP, G)`` — median ``ln_cfu0`` if present in the file,
        otherwise ``None``.

    Raises
    ------
    ValueError
        If ``growth_pred`` is not a dataset in the HDF5 file.
    """
    import h5py

    def _read(fh):
        if "growth_pred" not in fh:
            raise ValueError(
                "growth_pred not found in posterior file; "
                "cannot generate trajectory plots."
            )
        # Shape: (S, R, T, CP, CS, TN, TC, G)
        samples = fh["growth_pred"][:]
        median = np.median(samples, axis=0)
        lo     = np.percentile(samples, 5,  axis=0)
        hi     = np.percentile(samples, 95, axis=0)
        ln_cfu0 = (
            np.median(fh["ln_cfu0"][:], axis=0)
            if "ln_cfu0" in fh
            else None
        )
        return median, lo, hi, ln_cfu0

    if isinstance(posterior_file, str):
        with h5py.File(posterior_file, "r") as fh:
            return _read(fh)
    else:
        return _read(posterior_file)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_geno_trajectory(
    gm,
    out_prefix,
    params=None,
    posterior_file=None,
    genotypes=None,
    titrant_names=None,
    write_csv=True,
    growth_pred_std=None,
):
    """
    Generate per-genotype growth trajectory plots from a fitted model.

    For each genotype a PDF is written containing one subplot per
    ``(condition_pre, condition_sel, titrant_name, titrant_conc)``
    combination.  Each subplot shows:

    * **Observed data** — ``ln_cfu ± ln_cfu_std`` error bars at their
      ``t_sel`` x-coordinates (circles, one colour per replicate).
    * **Pre-split observations** *(optional)* — ``ln_cfu_t0 ± ln_cfu_t0_std``
      as square markers at ``x = −t_pre`` when ``gm.data.presplit`` is not
      ``None``.  The same pooled aliquot value appears in every subplot that
      shares the same ``(replicate, condition_pre, genotype)`` triple, which
      is correct because the measurement predates the condition split.
    * **Model prediction**:

      - *MAP path* — a smooth 50-point trajectory from ``−t_pre`` through
        ``max(t_sel)``, computed via a fine-grid forward pass.
      - *Posterior path* — a median line from ``−t_pre`` through the
        observed timepoints, with a 90 % credible-interval shaded band
        covering the selection phase (``t ≥ 0``).

    Parameters
    ----------
    gm : ModelOrchestrator
        Fitted model orchestrator.
    out_prefix : str
        Prefix for output files.  Each genotype's PDF is written to
        ``{out_prefix}_calib_{genotype}.pdf``.
    params : dict or str, optional
        MAP parameter dict with ``{site}_auto_loc`` keys, or a path to a
        ``_params.npz`` file produced by ``RunInference.write_params()``.
        Exactly one of *params* or *posterior_file* must be supplied.
    posterior_file : str or h5py.File, optional
        Path to a ``_posterior.h5`` file (or an open ``h5py.File`` handle)
        produced by ``RunInference.get_posteriors()``.  The plot shows the
        posterior median with a 90 % credible-interval shaded band.
    genotypes : list of str, optional
        Subset of genotype names to plot.  If ``None``, all genotypes in
        ``gm`` are plotted.  Names not found in the model are warned about
        and silently skipped.
    titrant_names : list of str, optional
        Subset of titrant names to include.  If ``None``, all titrant names
        are included.
    write_csv : bool, optional
        If ``True`` (default), write
        ``{out_prefix}_calib_growth_df.csv`` containing the observed and
        predicted ``ln_cfu`` values.
    growth_pred_std : np.ndarray, optional
        Laplace-based per-cell prediction standard deviations, same shape
        as the ``growth_pred`` tensor ``(R, T, CP, CS, TN, TC, G)``.
        Only used with the MAP path (ignored for posteriors).  When
        *write_csv* is ``True`` this is written as the ``ln_cfu_pred_std``
        column.

    Returns
    -------
    growth_df_out : pd.DataFrame or None
        Merged DataFrame of observed and predicted ``ln_cfu`` values (one
        row per valid tensor cell), or ``None`` if matplotlib is not
        available.  Columns include all columns from the growth
        ``TensorManager`` DataFrame plus ``ln_cfu_pred`` and, depending on
        the prediction path:

        - MAP + *growth_pred_std*: ``ln_cfu_pred_std``
        - Posterior: ``ln_cfu_pred_lo``, ``ln_cfu_pred_hi``

    Raises
    ------
    ValueError
        If neither or both of *params* / *posterior_file* are supplied, or
        if required data are missing from the posterior file.
    """
    if (params is None) == (posterior_file is None):
        raise ValueError(
            "Exactly one of params or posterior_file must be provided."
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  warning: matplotlib not available; skipping calibration plots.",
            file=sys.stderr,
        )
        return None

    # -----------------------------------------------------------------------
    # Compute predictions
    # -----------------------------------------------------------------------
    is_map = params is not None
    growth_pred_lo = growth_pred_hi = None
    growth_pred_fine = t_fine_1d = None

    if is_map:
        if isinstance(params, str):
            params = dict(np.load(params))
        growth_pred, ln_cfu0_map, growth_pred_fine, t_fine_1d = (
            _compute_map_predictions(gm, params)
        )
    else:
        growth_pred, growth_pred_lo, growth_pred_hi, ln_cfu0_map = (
            _compute_posterior_predictions(gm, posterior_file)
        )

    # -----------------------------------------------------------------------
    # Data tensors
    # -----------------------------------------------------------------------
    gd = gm.data.growth
    good_mask    = np.asarray(gd.good_mask)        # (R, T, CP, CS, TN, TC, G)
    t_pre_tensor = np.asarray(gd.t_pre)
    t_sel_tensor = np.asarray(gd.t_sel)
    ln_cfu_obs   = np.asarray(gd.ln_cfu)
    ln_cfu_std_t = np.asarray(gd.ln_cfu_std)

    n_rep, n_t, n_cp, n_cs, n_tn, n_tc, n_geno = good_mask.shape

    # -----------------------------------------------------------------------
    # Dimension labels and helper maps
    # -----------------------------------------------------------------------
    tm = gm.growth_tm
    dn = tm.tensor_dim_names

    geno_labels = list(tm.tensor_dim_labels[dn.index("genotype")])
    rep_labels  = list(tm.tensor_dim_labels[dn.index("replicate")])
    cp_labels   = list(tm.tensor_dim_labels[dn.index("condition_pre")])
    tn_labels   = list(tm.tensor_dim_labels[dn.index("titrant_name")])
    tc_labels   = list(tm.tensor_dim_labels[dn.index("titrant_conc")])

    df = tm.df
    # (cp_idx, cs_idx) → condition_sel label string
    cs_name_map = {}
    for _, row in df.drop_duplicates(
        ["condition_pre_idx", "condition_sel_idx"]
    ).iterrows():
        cs_name_map[
            (int(row["condition_pre_idx"]), int(row["condition_sel_idx"]))
        ] = str(row["condition_sel"])

    # (genotype, titrant_name, titrant_conc) → mean observed theta
    theta_map: dict = {}
    for _, row in gm.binding_df.iterrows():
        key = (
            str(row["genotype"]),
            str(row["titrant_name"]),
            float(row["titrant_conc"]),
        )
        theta_map.setdefault(key, []).append(float(row["theta_obs"]))
    theta_map = {k: float(np.nanmean(v)) for k, v in theta_map.items()}

    # -----------------------------------------------------------------------
    # Pre-split data (optional)
    # -----------------------------------------------------------------------
    presplit = gm.data.presplit
    if presplit is not None:
        ps_ln_cfu     = np.asarray(presplit.ln_cfu_t0)      # (R, CP, G)
        ps_ln_cfu_std = np.asarray(presplit.ln_cfu_t0_std)  # (R, CP, G)
        ps_good_mask  = np.asarray(presplit.good_mask)       # (R, CP, G)
    else:
        ps_ln_cfu = ps_ln_cfu_std = ps_good_mask = None

    # -----------------------------------------------------------------------
    # Genotype filter
    # -----------------------------------------------------------------------
    avail_geno = {str(g): i for i, g in enumerate(geno_labels)}
    if genotypes is not None:
        requested = [str(g) for g in genotypes]
        missing = sorted(set(requested) - set(avail_geno))
        if missing:
            print(
                f"  warning: requested genotype(s) not found in model: {missing}",
                file=sys.stderr,
            )
        geno_iter = [
            (avail_geno[g], g)
            for g in requested
            if g in avail_geno
        ]
    else:
        geno_iter = [(i, str(g)) for i, g in enumerate(geno_labels)]

    # -----------------------------------------------------------------------
    # Titrant_name filter
    # -----------------------------------------------------------------------
    if titrant_names is not None:
        requested_tn = set(str(t) for t in titrant_names)
        tn_indices = frozenset(
            i for i, tn in enumerate(tn_labels) if str(tn) in requested_tn
        )
    else:
        tn_indices = frozenset(range(n_tn))

    # -----------------------------------------------------------------------
    # Build predictions DataFrame (used for CSV and returned to caller)
    # -----------------------------------------------------------------------
    r_idx, t_idx, cp_idx, cs_idx, tn_idx, tc_idx, g_idx = np.where(good_mask)
    flat_idx = (r_idx, t_idx, cp_idx, cs_idx, tn_idx, tc_idx, g_idx)

    pred_dict: dict = {
        "replicate_idx":     r_idx.astype(int),
        "condition_pre_idx": cp_idx.astype(int),
        "condition_sel_idx": cs_idx.astype(int),
        "titrant_name_idx":  tn_idx.astype(int),
        "titrant_conc_idx":  tc_idx.astype(int),
        "genotype_idx":      g_idx.astype(int),
        "t_sel":             t_sel_tensor[flat_idx],
        "ln_cfu_pred":       growth_pred[flat_idx],
    }
    if is_map and growth_pred_std is not None:
        pred_dict["ln_cfu_pred_std"] = growth_pred_std[flat_idx]
    elif not is_map:
        pred_dict["ln_cfu_pred_lo"] = growth_pred_lo[flat_idx]
        pred_dict["ln_cfu_pred_hi"] = growth_pred_hi[flat_idx]

    merge_cols = [
        "replicate_idx", "condition_pre_idx", "condition_sel_idx",
        "titrant_name_idx", "titrant_conc_idx", "genotype_idx", "t_sel",
    ]
    growth_df_out = df.merge(pd.DataFrame(pred_dict), on=merge_cols, how="left")

    if write_csv:
        csv_path = f"{out_prefix}_calib_growth_df.csv"
        growth_df_out.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}", flush=True)

    # -----------------------------------------------------------------------
    # Per-genotype PDF plots
    # -----------------------------------------------------------------------
    print("Generating calibration quality plots ...", flush=True)

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for g_i, geno_name in geno_iter:

        # Collect valid (cp, cs, tn, tc) combos → list of replicates.
        # Apply titrant_name filter here.
        condition_combos: dict = {}
        for r_i in range(n_rep):
            for cp_i in range(n_cp):
                for cs_i in range(n_cs):
                    for tn_i in range(n_tn):
                        if tn_i not in tn_indices:
                            continue
                        for tc_i in range(n_tc):
                            if good_mask[r_i, :, cp_i, cs_i, tn_i, tc_i, g_i].any():
                                condition_combos.setdefault(
                                    (cp_i, cs_i, tn_i, tc_i), []
                                ).append(r_i)

        if not condition_combos:
            continue

        n_combos = len(condition_combos)
        n_cols   = min(3, n_combos)
        n_rows   = (n_combos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
            sharey=True,
        )
        fig.suptitle(
            f"Calibration fit — genotype: {geno_name}",
            fontsize=13,
            fontweight="bold",
        )

        for combo_i, ((cp_i, cs_i, tn_i, tc_i), rep_list) in enumerate(
            condition_combos.items()
        ):
            ax = axes[combo_i // n_cols][combo_i % n_cols]

            cp_name = str(cp_labels[cp_i])
            cs_name = cs_name_map.get((cp_i, cs_i), f"sel_{cs_i}")
            tn_name = str(tn_labels[tn_i])
            tc_val  = float(tc_labels[tc_i])

            theta_obs = theta_map.get((geno_name, tn_name, tc_val))
            theta_str = f", θ = {theta_obs:.3f}" if theta_obs is not None else ""
            ax.set_title(
                f"{cp_name} → {cs_name}\n{tn_name} = {tc_val:.3g}{theta_str}",
                fontsize=9,
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("ln(CFU)")
            ax.axvline(0.0, color="0.6", lw=0.8, ls="--")

            # Build per-replicate observation records.
            rep_data = []
            for r_i in rep_list:
                mask    = good_mask[r_i, :, cp_i, cs_i, tn_i, tc_i, g_i]
                valid_t = np.where(mask)[0]
                if not len(valid_t):
                    continue
                rep_data.append({
                    "r_i":     r_i,
                    "valid_t": valid_t,
                    "t_sel":   t_sel_tensor[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i],
                    "obs":     ln_cfu_obs[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i],
                    "std":     ln_cfu_std_t[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i],
                    "t_pre":   float(np.nanmedian(
                        t_pre_tensor[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i]
                    )),
                })

            if not rep_data:
                ax.set_visible(False)
                continue

            max_t_sel = float(max(np.nanmax(rd["t_sel"]) for rd in rep_data))
            max_t_pre = float(max(rd["t_pre"] for rd in rep_data))

            for rd in rep_data:
                r_i       = rd["r_i"]
                rep_color = prop_colors[r_i % len(prop_colors)]
                rep_label = str(rep_labels[r_i])

                # --- Pre-split observed point (square marker) ---
                # Shown at x = −t_pre; same pooled aliquot for all conditions
                # sharing this (replicate, condition_pre, genotype) triple.
                if ps_good_mask is not None and ps_good_mask[r_i, cp_i, g_i]:
                    ax.errorbar(
                        -rd["t_pre"],
                        float(ps_ln_cfu[r_i, cp_i, g_i]),
                        yerr=float(ps_ln_cfu_std[r_i, cp_i, g_i]),
                        fmt="s",
                        color=rep_color,
                        ms=5,
                        lw=1,
                        capsize=3,
                        zorder=3,
                    )

                # --- Selection-phase observed data (circle marker) ---
                ax.errorbar(
                    rd["t_sel"],
                    rd["obs"],
                    yerr=rd["std"],
                    fmt="o",
                    color=rep_color,
                    ms=5,
                    lw=1,
                    capsize=3,
                    label=rep_label,
                    zorder=3,
                )

                # --- Model prediction ---
                if is_map:
                    # Smooth fine-grid line (pre-selection + selection phases).
                    y_fine    = growth_pred_fine[r_i, :, cp_i, cs_i, tn_i, tc_i, g_i]
                    calc_at_0 = float(y_fine[0])
                    ln_cfu0_r = (
                        float(ln_cfu0_map[r_i, cp_i, g_i])
                        if (ln_cfu0_map is not None and ln_cfu0_map.ndim == 3)
                        else calc_at_0
                    )
                    t_smooth = np.concatenate([[-rd["t_pre"], 0.0], t_fine_1d])
                    y_smooth = np.concatenate([[ln_cfu0_r, calc_at_0], y_fine])
                    ax.plot(t_smooth, y_smooth, "-", color=rep_color, lw=1.8, zorder=4)

                else:
                    # Posterior: median line (both phases) + 90 % credible
                    # interval shaded band (selection phase only).
                    valid_t = rd["valid_t"]
                    t_obs   = rd["t_sel"]
                    y_med   = growth_pred[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i]
                    y_lo    = growth_pred_lo[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i]
                    y_hi    = growth_pred_hi[r_i, valid_t, cp_i, cs_i, tn_i, tc_i, g_i]

                    ln_cfu0_r = (
                        float(ln_cfu0_map[r_i, cp_i, g_i])
                        if (ln_cfu0_map is not None and ln_cfu0_map.ndim == 3)
                        else float(y_med[0])
                    )

                    # Sort by time for clean connected lines.
                    sort_idx  = np.argsort(t_obs)
                    t_sorted  = t_obs[sort_idx]
                    y_med_s   = y_med[sort_idx]
                    y_lo_s    = y_lo[sort_idx]
                    y_hi_s    = y_hi[sort_idx]

                    # Full line: pre-selection anchor → observed medians.
                    t_line = np.concatenate([[-rd["t_pre"]], t_sorted])
                    y_line = np.concatenate([[ln_cfu0_r], y_med_s])
                    ax.plot(t_line, y_line, "-", color=rep_color, lw=1.8, zorder=4)

                    # Shaded credible interval — selection phase only.
                    ax.fill_between(
                        t_sorted,
                        y_lo_s,
                        y_hi_s,
                        color=rep_color,
                        alpha=0.2,
                        zorder=2,
                    )

            ax.legend(fontsize=8, loc="best")
            x_pad = (max_t_sel + max_t_pre) * 0.03
            ax.set_xlim(-max_t_pre - x_pad, max_t_sel + x_pad)

        # Hide unused subplot cells.
        for extra_i in range(n_combos, n_rows * n_cols):
            axes[extra_i // n_cols][extra_i % n_cols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe_name = geno_name.replace("/", "-")
        pdf_path  = f"{out_prefix}_calib_{safe_name}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {pdf_path}", flush=True)

    print("Calibration plots complete.", flush=True)
    return growth_df_out
