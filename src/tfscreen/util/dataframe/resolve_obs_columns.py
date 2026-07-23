def _quantile_col(q):
    """Column name storing quantile ``q`` (e.g. 0.159 -> 'q0.159')."""
    return f"q{q}"


def resolve_obs_columns(df,
                        y_obs=None,
                        y_std=None,
                        point_quantile=0.5,
                        sigma_quantiles=(0.159, 0.841),
                        sigma_col="_sigma"):
    """
    Resolve default observable / uncertainty columns for quantile tables.

    Several ``tfs-*`` tools consume a long-form CSV and need a single point
    estimate (``y_obs``) and, optionally, a per-row standard deviation
    (``y_std``). When the table stores a posterior as quantile columns (e.g.
    ``q0.5``, ``q0.159``, ``q0.841`` as written by ``tfs-predict-theta``), the
    caller can omit both and let this helper fill in sane defaults:

    - ``y_obs`` defaults to the ``point_quantile`` column (``q0.5``).
    - ``y_std`` defaults to the symmetric one-sigma half-width from
      ``sigma_quantiles``, ``(q0.841 - q0.159) / 2``, added as a new column.

    An explicitly supplied name always takes precedence and is returned
    unchanged (no quantile inspection is done for that axis).

    Parameters
    ----------
    df : pandas.DataFrame
        The input table. Not modified in place; a copy is returned only when a
        sigma column has to be added.
    y_obs : str or None, optional
        Name of the observable column. If None, the ``point_quantile`` column is
        used when present.
    y_std : str or None, optional
        Name of the standard-deviation column. If None, the symmetric quantile
        half-width is computed from ``sigma_quantiles`` when both bounding
        columns are present; otherwise it stays None (unweighted).
    point_quantile : float, optional
        Quantile used as the ``y_obs`` fallback. Default 0.5 (median).
    sigma_quantiles : tuple of float, optional
        ``(lo, hi)`` quantiles bracketing one sigma. Default (0.159, 0.841).
    sigma_col : str, optional
        Name of the column created to hold the derived sigma. Default
        ``"_sigma"``.

    Returns
    -------
    df : pandas.DataFrame
        Either the original DataFrame (unchanged) or a copy with the derived
        ``sigma_col`` added.
    y_obs : str
        The resolved observable column name.
    y_std : str or None
        The resolved standard-deviation column name, or None if no explicit
        column was given and the quantile columns were unavailable.

    Raises
    ------
    ValueError
        If ``y_obs`` is None and the ``point_quantile`` column is not present --
        an observable cannot be guessed.
    """
    # Resolve the observable. Fall back to the point-estimate quantile.
    if y_obs is None:
        point_col = _quantile_col(point_quantile)
        if point_col not in df.columns:
            raise ValueError(
                f"No 'y_obs' column specified and the default '{point_col}' "
                f"column is not present. Specify an observable column "
                f"explicitly. Available columns: {list(df.columns)}"
            )
        y_obs = point_col

    # Resolve the standard deviation. Fall back to the symmetric quantile
    # half-width, mirroring the convention used across the codebase.
    if y_std is None:
        lo, hi = sigma_quantiles
        lo_col = _quantile_col(lo)
        hi_col = _quantile_col(hi)
        if lo_col in df.columns and hi_col in df.columns:
            df = df.copy()
            df[sigma_col] = (df[hi_col] - df[lo_col]) / 2
            y_std = sigma_col

    return df, y_obs, y_std
