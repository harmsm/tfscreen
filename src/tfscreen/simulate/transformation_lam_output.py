"""
Generate the ground-truth congression Poisson-rate (lambda) echo.

``transformation_poisson_lambda`` is a top-level simulate-config scalar
(number of plasmids/cell during transformation, drawn from a zero-truncated
Poisson; see ``simulate.selection_experiment``). It has a fit-side analog:
the ``transformation`` component category's ``empirical``/``logit_norm``
variants both sample a global congression rate (see
``generative/components/transformation/_congression.py``) and extract it as
``lam`` (see those modules' ``get_extract_specs``, which writes
``*_params_lam.csv``). This mirrors ``base_growth_data.generate_k_ref_df``:
a single global scalar, not genotype- or condition-indexed, echoed to its
own single-row ground-truth file.
"""

import pandas as pd


def generate_transformation_lam_df(cf):
    """
    Generate a one-row ground-truth DataFrame for the global lambda scalar.

    Parameters
    ----------
    cf : dict
        The full simulate config. May contain 'transformation_poisson_lambda'
        (a top-level key; see selection_experiment.py's SIMULATE_KNOWN_KEYS /
        _check_dict_number(..., allow_none=True)). Unlike
        generate_k_ref_df's required base_growth_data.k_ref, this key is
        read with .get() since it is always semantically present (defaulting
        to None) rather than required.

    Returns
    -------
    pandas.DataFrame
        Single row with columns 'parameter' ("lam") and 'ref' (the
        configured value). None or an absent key means "exactly one plasmid
        per cell" (see selection_experiment.py's
        ``transformation_poisson_lambda is None or == 0`` check), echoed
        here as ``ref=0.0``.
    """
    lam = cf.get("transformation_poisson_lambda")
    ref = float(lam) if lam else 0.0
    return pd.DataFrame({"parameter": ["lam"], "ref": [ref]})
