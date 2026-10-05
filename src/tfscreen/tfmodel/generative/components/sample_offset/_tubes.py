"""
Tube labels for the per-tube sample offsets.

Both ``level`` and ``normal`` sample one value per position of the growth
tensor's tube axes (replicate, time, condition_pre, condition_sel,
titrant_name, titrant_conc), flattened in C order. ``tube_extract_spec``
labels each flat position with the tube design read off the growth rows
that occupy it, so ``tfs-extract-params`` writes one row per measured tube.
"""

import numpy as np
import pandas as pd

TUBE_DIMS = ("replicate", "time", "condition_pre", "condition_sel",
             "titrant_name", "titrant_conc")

TUBE_COLUMNS = ["replicate", "library", "condition_pre", "condition_sel",
                "titrant_name", "titrant_conc", "t_pre", "t_sel"]


def tube_index(tm):
    """
    Flat tube position of every growth row, from the tensor's ``{dim}_idx``
    columns, in the order the offset components reshape it.
    """
    df = tm.df
    names = list(tm.tensor_dim_names)
    sizes = {d: len(lab) for d, lab in zip(names, tm.tensor_dim_labels)}
    missing = [d for d in TUBE_DIMS if f"{d}_idx" not in df.columns]
    if missing:
        raise ValueError(f"growth rows lack the tube index columns "
                         f"{[f'{d}_idx' for d in missing]}")
    idx = tuple(df[f"{d}_idx"].to_numpy(dtype=int) for d in TUBE_DIMS)
    shape = tuple(sizes[d] for d in TUBE_DIMS)
    return np.ravel_multi_index(idx, shape)


def tube_extract_spec(ctx, name, offset_site, sigma_site):
    """
    Extraction specs for a per-tube offset site and its population SD.

    Parameters
    ----------
    ctx : ExtractionContext
    name : str
        The component's registry name (``sample_offset``).
    offset_site : str
        Site suffix of the per-tube values (``offset``, ``delta_k``).
    sigma_site : str
        Site suffix of their SD (``sigma``, ``sigma_env``).

    Returns
    -------
    list of dict
        One spec for the per-tube values (``{name}_{offset_site}``, one row per
        measured tube labeled by ``TUBE_COLUMNS``) and one for the SD. A held
        SD is a deterministic site, absent from a MAP checkpoint;
        ``extract_parameters`` then skips it with a warning.
    """
    tm = ctx.growth_tm
    df = tm.df
    flat = tube_index(tm)
    columns = [c for c in TUBE_COLUMNS if c in df.columns]
    # every genotype in a tube shares its labels; keep one row per tube
    tubes = (df[columns].assign(_tube_flat=flat)
             .drop_duplicates("_tube_flat"))
    sigma = pd.DataFrame({"parameter": [f"{name}_{sigma_site}"],
                          "_map": [0]})
    return [
        dict(input_df=tubes,
             params_to_get=[f"{name}_{offset_site}"],
             map_column="_tube_flat",
             get_columns=columns,
             in_run_prefix=""),
        dict(input_df=sigma,
             params_to_get=[f"{name}_{sigma_site}"],
             map_column="_map",
             get_columns=["parameter"],
             in_run_prefix=""),
    ]
