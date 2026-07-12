"""
Generate simulated ground-truth growth-model parameters (condition_growth).

Mirrors ``tfs_sim_parameters.csv``'s genotype-level ground truth, but at the
per-condition level: the ``growth`` block of the simulate config directly
specifies the true condition_growth parameters (linear/power/saturation),
keyed by condition string. Those same condition strings are used, unchanged,
as ``condition_rep`` on the tfmodel inference side (see
``model_orchestrator._build_growth_tm``), so no further mapping is needed to
join this file against a fit's ``*_params_growth_{name}.csv`` outputs.

Each growth model uses different parameter names in the YAML (``b``/``m``
for linear, ``b``/``a``/``n`` for power, ``kmin``/``kmax`` for saturation --
see ``simulate.growth.growth_linkage``) which must be renamed to match the
inference-side extract names (``growth_k``, ``growth_m``, ``growth_n``,
``growth_min``, ``growth_max`` respectively -- see
``generative/components/growth/*.py``'s ``get_extract_specs``) before this
ground truth can be joined against a fit's extracted parameters. The two
sides' formulas must stay in sync by hand (compare
``simulate/growth/growth_linkage.py`` against
``generative/components/growth/{linear,power,saturation}.py``); this module
is the single place that mapping is recorded.
"""

import pandas as pd


_MODEL_KEY_MAP = {
    "linear":     {"b": "growth_k", "m": "growth_m"},
    "power":      {"b": "growth_k", "a": "growth_m", "n": "growth_n"},
    "saturation": {"kmin": "growth_min", "kmax": "growth_max"},
}


def generate_growth_parameters_df(growth_cfg):
    """
    Generate the ground-truth condition_growth parameters DataFrame.

    Parameters
    ----------
    growth_cfg : dict
        The 'growth' top-level config block: a mapping from condition
        string to a parameter dict. Each parameter dict may contain a
        'model' key ('linear', 'power', or 'saturation'; defaults to
        'linear' to match simulate.thermo_to_growth._apply_growth_params)
        plus the model's own parameter keys.

    Returns
    -------
    pandas.DataFrame
        One row per condition. Columns: 'condition_rep' plus the union of
        growth_k/growth_m/growth_n/growth_min/growth_max used by any
        condition present (NaN where a given condition's model doesn't use
        that parameter).

    Raises
    ------
    ValueError
        If a condition specifies an unrecognized 'model', or is missing one
        of that model's required parameter keys.
    """
    rows = []
    for condition, params in growth_cfg.items():
        params = dict(params)
        model_name = params.pop("model", "linear")
        if model_name not in _MODEL_KEY_MAP:
            raise ValueError(
                f"Unknown growth model '{model_name}' for condition "
                f"'{condition}'. Available models: "
                f"{sorted(_MODEL_KEY_MAP)}"
            )

        key_map = _MODEL_KEY_MAP[model_name]
        row = {"condition_rep": condition}
        for src_key, dst_key in key_map.items():
            if src_key not in params:
                raise ValueError(
                    f"Growth condition '{condition}' (model '{model_name}') "
                    f"is missing required parameter '{src_key}'."
                )
            row[dst_key] = float(params[src_key])
        rows.append(row)

    return pd.DataFrame(rows)
