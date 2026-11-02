"""
Operator and TF-level contexts of the θ conditions.

A tube table may name each tube's operator and TF level, in the optional
text columns ``operator`` and ``tf_level`` (``tfs-process-counts`` carries
them into the growth table). Binding data may name each measurement's
operator and its protein concentration, ``operator`` and ``tf_total_uM``
(µM protomer; pipetted, so known). Without these columns there is one
operator and one TF level, as before contexts existed.

Inside the fit, the titrant (the effector), the operator and the TF level
fold into one θ-condition index on the existing ``titrant_name`` tensor
dimension. Every growth component already indexes tubes by that dimension,
so tube identity stays unique and nothing outside the θ layer changes
shape. :class:`ThetaConditions` records what each index is, as static
tuples the θ components read (``effector_idx``, ``operator_idx``,
``tf_level_idx`` and, for binding, the known ``tf_total_M``).

With one operator, one TF level and no binding ``tf_total_uM`` column the
θ-condition labels are the titrant names themselves, so data without
contexts build exactly the tensors they did before. Otherwise a label is
``titrant|operator|tf_level`` (growth) or ``titrant|operator|<conc>uM``
(binding), and only a θ component that declares ``SUPPORTS_CONTEXTS``
reads the context arrays. Curve models (``hill_geno`` and the like) fit
each growth θ condition as its own curve, which is what they mean, but
cannot link a binding measurement to one; the thermodynamic models without
context support would read each context as a different effector.

The contexts file (:func:`read_contexts`) gives the operators' offsets and
the TF levels' concentrations as priors:

.. code-block:: yaml

    units: kcal/mol            # operator shifts; kT, kcal/mol or kJ/mol
    temperature_K: 310.15
    operators:
      O1: {reference: true}
      O2: {shift: 1.5, shift_sd: 0.3}
    tf_levels:                 # total TF per cell, µM protomer
      low:  {conc_uM: 3,  ln_sd: 0.5}
      high: {conc_uM: 30, ln_sd: 0.5}
"""

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from tfscreen.util.io.read_yaml import read_yaml


OPERATOR_COLUMN = "operator"
TF_LEVEL_COLUMN = "tf_level"
BINDING_TF_COLUMN = "tf_total_uM"
CONTEXT_COLUMNS = (OPERATOR_COLUMN, TF_LEVEL_COLUMN)

THETA_CONDITION_COLUMN = "theta_condition"
DEFAULT_LABEL = "default"
SEPARATOR = "|"

UM_TO_M = 1e-6

_TOP_KEYS = {"units", "temperature_K", "operators", "tf_levels"}
_OPERATOR_KEYS = {"reference", "shift", "shift_sd"}
_TF_LEVEL_KEYS = {"conc_uM", "ln_sd"}


# ---------------------------------------------------------------------------
# Context columns

def _labels(df, column, where):
    """Validated text labels of ``column`` (None when the column is absent)."""
    if column not in df.columns:
        return None
    values = df[column]
    if values.isna().any():
        raise ValueError(f"{where} column '{column}' has missing values; "
                         f"give every row a label.")
    labels = values.astype(str).str.strip()
    if (labels == "").any():
        raise ValueError(f"{where} column '{column}' has empty labels.")
    if labels.str.contains(SEPARATOR, regex=False).any():
        raise ValueError(f"{where} column '{column}' labels may not contain "
                         f"'{SEPARATOR}'.")
    return labels


def _binding_tf(df):
    """Validated binding protein concentrations in µM (None if absent)."""
    if BINDING_TF_COLUMN not in df.columns:
        return None
    conc = pd.to_numeric(df[BINDING_TF_COLUMN], errors="coerce")
    bad = ~np.isfinite(conc.to_numpy(dtype=float)) | (conc <= 0)
    if bad.any():
        raise ValueError(f"binding_df column '{BINDING_TF_COLUMN}' must be a "
                         f"finite, positive concentration (µM) on every row.")
    return conc.astype(float)


def named_labels(names):
    """An axis's labels, or () when it has only the default one."""
    names = tuple(names) if names is not None else ()
    return () if names in ((), (DEFAULT_LABEL,)) else names


def _conc_label(conc_uM):
    return f"{float(conc_uM):.6g}uM"


@dataclass(frozen=True)
class ThetaConditions:
    """
    What each θ-condition index is, for the growth and the binding data.

    Attributes
    ----------
    composite : bool
        True when the labels carry an operator and TF level (more than one
        operator or TF level, or binding protein concentrations given).
    effector_names, operator_names, tf_level_names : tuple of str
        Model-wide names. Effectors are the titrant names of the growth
        data (of the binding data in a binding-only model), sorted; an
        operator or TF level absent from the data is ``"default"``.
    growth_labels, binding_labels : tuple of str or None
        The θ-condition labels in tensor order.
    growth_*_idx, binding_*_idx : tuple of int
        For each θ condition, its index into the names. A binding
        measurement has no TF level label (``-1``); its known total TF is
        ``binding_tf_total_M`` (NaN where no concentration was given).
    """

    composite: bool
    effector_names: tuple
    operator_names: tuple
    tf_level_names: tuple
    growth_labels: tuple = None
    growth_effector_idx: tuple = None
    growth_operator_idx: tuple = None
    growth_tf_level_idx: tuple = None
    binding_labels: tuple = None
    binding_effector_idx: tuple = None
    binding_operator_idx: tuple = None
    binding_tf_total_M: tuple = None
    growth_table: pd.DataFrame = field(default=None, compare=False,
                                       repr=False)
    binding_table: pd.DataFrame = field(default=None, compare=False,
                                        repr=False)

    def data_fields(self, which):
        """The static fields a GrowthData/BindingData carries."""
        names = {"theta_effector_names": self.effector_names,
                 "theta_operator_names": self.operator_names,
                 "theta_tf_level_names": self.tf_level_names,
                 "theta_contexts": self.composite}
        if which == "growth":
            n = len(self.growth_labels)
            names.update(
                theta_effector_idx=self.growth_effector_idx,
                theta_operator_idx=self.growth_operator_idx,
                theta_tf_level_idx=self.growth_tf_level_idx,
                theta_tf_total_M=(math.nan,) * n)
        elif which == "binding":
            n = len(self.binding_labels)
            names.update(
                theta_effector_idx=self.binding_effector_idx,
                theta_operator_idx=self.binding_operator_idx,
                theta_tf_level_idx=(-1,) * n,
                theta_tf_total_M=self.binding_tf_total_M)
        else:
            raise ValueError(f"which must be 'growth' or 'binding', "
                             f"got {which!r}")
        return names


def _growth_context(df):
    """(titrant, operator, tf_level) Series of a growth frame."""
    if "titrant_name" not in df.columns:
        raise ValueError("θ conditions need a 'titrant_name' column.")
    titrant = df["titrant_name"].astype(str)
    op = _labels(df, OPERATOR_COLUMN, "growth_df")
    tf = _labels(df, TF_LEVEL_COLUMN, "growth_df")
    if op is None:
        op = pd.Series(DEFAULT_LABEL, index=df.index)
    if tf is None:
        tf = pd.Series(DEFAULT_LABEL, index=df.index)
    return titrant, op, tf


def _binding_context(df, default_operator):
    """(titrant, operator, tf_total_uM) Series of a binding frame."""
    if "titrant_name" not in df.columns:
        raise ValueError("θ conditions need a 'titrant_name' column.")
    if TF_LEVEL_COLUMN in df.columns:
        raise ValueError(
            f"binding_df has a '{TF_LEVEL_COLUMN}' column. Binding "
            f"measurements give their protein concentration instead, in "
            f"'{BINDING_TF_COLUMN}' (µM protomer).")
    titrant = df["titrant_name"].astype(str)
    op = _labels(df, OPERATOR_COLUMN, "binding_df")
    if op is None:
        op = pd.Series(default_operator, index=df.index)
    conc = _binding_tf(df)
    if conc is None:
        conc = pd.Series(math.nan, index=df.index)
    return titrant, op, conc


def theta_conditions(growth_df=None, binding_df=None):
    """
    The θ conditions of an experiment.

    Parameters
    ----------
    growth_df, binding_df : pandas.DataFrame or None
        Need ``titrant_name``; may have the context columns. At least one
        must be given. Pass the binding data as read, before it is synced
        to the growth data: the conditions depend only on which operators
        and concentrations it names.

    Returns
    -------
    ThetaConditions

    Notes
    -----
    Without contexts the binding θ conditions are the growth ones (the
    growth titrant names, in order), so a binding measurement has the index
    of the growth condition with its titrant even when it measures only
    some of the titrants.
    """
    if growth_df is None and binding_df is None:
        raise ValueError("theta_conditions needs growth_df or binding_df.")

    g = _growth_context(growth_df) if growth_df is not None else None
    growth_ops = sorted(g[1].unique()) if g is not None else []
    if binding_df is not None:
        if OPERATOR_COLUMN not in binding_df.columns and len(growth_ops) > 1:
            raise ValueError(
                f"binding_df needs an '{OPERATOR_COLUMN}' column: the growth "
                f"data have operators {growth_ops}.")
        default_op = growth_ops[0] if growth_ops else DEFAULT_LABEL
        b = _binding_context(binding_df, default_op)
    else:
        b = None

    operator_names = tuple(sorted(set(growth_ops) |
                                  (set(b[1].unique()) if b is not None
                                   else set())))
    tf_level_names = (tuple(sorted(g[2].unique())) if g is not None
                      else (DEFAULT_LABEL,))
    has_binding_tf = (b is not None and
                      BINDING_TF_COLUMN in binding_df.columns)
    composite = (len(operator_names) > 1 or len(tf_level_names) > 1
                 or has_binding_tf)

    source = g if g is not None else b
    effector_names = tuple(sorted(source[0].unique()))
    out = dict(composite=composite, effector_names=effector_names,
               operator_names=operator_names, tf_level_names=tf_level_names)

    if g is not None:
        label = _join(g, composite, binding=False)
        table = (pd.DataFrame({"theta_condition": label,
                               "titrant_name": g[0], "operator": g[1],
                               "tf_level": g[2]})
                 .drop_duplicates().sort_values("theta_condition")
                 .reset_index(drop=True))
        if table["theta_condition"].duplicated().any():
            raise ValueError("Growth θ-condition labels are not unique.")
        out.update(
            growth_labels=tuple(table["theta_condition"]),
            growth_effector_idx=tuple(effector_names.index(t)
                                      for t in table["titrant_name"]),
            growth_operator_idx=tuple(operator_names.index(o)
                                      for o in table["operator"]),
            growth_tf_level_idx=tuple(tf_level_names.index(t)
                                      for t in table["tf_level"]),
            growth_table=table)

    if b is not None:
        if not composite and g is not None:
            # Aligned with the growth conditions (index = growth index).
            table = out["growth_table"][["theta_condition", "titrant_name",
                                         "operator"]].copy()
            table[BINDING_TF_COLUMN] = math.nan
        else:
            label = _join(b, composite, binding=True)
            table = (pd.DataFrame({"theta_condition": label,
                                   "titrant_name": b[0], "operator": b[1],
                                   BINDING_TF_COLUMN: b[2]})
                     .drop_duplicates(subset=["theta_condition"])
                     .sort_values("theta_condition").reset_index(drop=True))
        out.update(
            binding_labels=tuple(table["theta_condition"]),
            binding_effector_idx=tuple(effector_names.index(t)
                                       if t in effector_names else -1
                                       for t in table["titrant_name"]),
            binding_operator_idx=tuple(operator_names.index(o)
                                       for o in table["operator"]),
            binding_tf_total_M=tuple(float(c) * UM_TO_M
                                     for c in table[BINDING_TF_COLUMN]),
            binding_table=table)

    return ThetaConditions(**out)


def _join(context, composite, binding):
    titrant, op, last = context
    if not composite:
        return titrant
    if binding:
        last = last.map(lambda c: "" if math.isnan(c) else _conc_label(c))
    return titrant + SEPARATOR + op + SEPARATOR + last


def assign_theta_condition(df, conditions, which):
    """
    Add the categorical ``theta_condition`` column to a growth or binding
    frame, with the categories (so the tensor order) of ``conditions``.

    Raises
    ------
    ValueError
        If a row's condition is not one of ``conditions`` (a frame other
        than the one the conditions were built from).
    """
    df = df.copy()
    if which == "growth":
        label = _join(_growth_context(df), conditions.composite,
                      binding=False)
        labels = conditions.growth_labels
    elif which == "binding":
        default_op = conditions.operator_names[0]
        label = _join(_binding_context(df, default_op),
                      conditions.composite, binding=True)
        labels = conditions.binding_labels
    else:
        raise ValueError(f"which must be 'growth' or 'binding', "
                         f"got {which!r}")
    names = (("operator", conditions.operator_names, df, OPERATOR_COLUMN),
             ("TF level", conditions.tf_level_names, df, TF_LEVEL_COLUMN))
    for kind, known, frame, column in names:
        if which == "binding" and column == TF_LEVEL_COLUMN:
            continue
        if column in frame.columns:
            extra = sorted(set(frame[column].astype(str)) - set(known))
            if extra:
                raise ValueError(f"{which} rows name {kind} {extra}, not "
                                 f"among {list(known)}.")
    unknown = sorted(set(label) - set(labels))
    if unknown:
        raise ValueError(f"{which} rows have θ conditions {unknown} that "
                         f"are not among {list(labels)}.")
    df[THETA_CONDITION_COLUMN] = pd.Categorical(label, categories=labels)
    return df


def split_theta_condition(labels, conditions):
    """
    The context columns for a sequence of growth θ-condition labels.

    Returns a DataFrame with ``titrant_name``, ``operator`` and
    ``tf_level`` (one row per label), for writing outputs keyed by the
    titrant the user named.
    """
    table = conditions.growth_table.set_index("theta_condition")
    return table.loc[list(labels)].reset_index()


# ---------------------------------------------------------------------------
# Contexts file

def _number(value, where, positive=False, nonnegative=False):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a number, got {value!r}.")
    if not math.isfinite(x):
        raise ValueError(f"{where} must be finite, got {value!r}.")
    if positive and x <= 0:
        raise ValueError(f"{where} must be > 0, got {value!r}.")
    if nonnegative and x < 0:
        raise ValueError(f"{where} must be >= 0, got {value!r}.")
    return x


def read_contexts(contexts):
    """
    Read and normalize a contexts file.

    Parameters
    ----------
    contexts : str or dict
        Path to the YAML file, or its parsed contents. A dict this function
        returned is passed through.

    Returns
    -------
    dict
        ``{"operators": {label: {"reference": bool, "shift_kT": float,
        "shift_sd_kT": float}}, "tf_levels": {label: {"conc_M": float,
        "ln_sd": float}}, "units": str, "temperature_K": float,
        "normalized": True}``. Operator
        shifts are converted to kT once, here; the reference operator's
        shift is 0 with SD 0.

    Raises
    ------
    ValueError
        On an unknown key, a missing value, more or less than one
        reference operator, or a non-positive concentration.
    """
    from tfscreen.tfmodel.generative.components.theta.thermo.io import (
        ddG_to_kT_factor,
    )

    if isinstance(contexts, dict) and contexts.get("normalized"):
        return contexts
    cf = read_yaml(contexts) if not isinstance(contexts, dict) else contexts
    if cf is None:
        cf = {}
    if not isinstance(cf, dict):
        raise ValueError("A contexts file must be a mapping.")
    extra = sorted(set(cf) - _TOP_KEYS)
    if extra:
        raise ValueError(f"Unknown contexts keys {extra}; expected "
                         f"{sorted(_TOP_KEYS)}.")

    units = str(cf.get("units", "kT"))
    temperature_K = _number(cf.get("temperature_K", 298.15),
                            "temperature_K", positive=True)
    factor = ddG_to_kT_factor(units, temperature_K)

    operators = {}
    raw_ops = cf.get("operators") or {}
    if not isinstance(raw_ops, dict):
        raise ValueError("'operators' must map each operator label to its "
                         "settings.")
    for label, spec in raw_ops.items():
        label = str(label)
        where = f"operator '{label}'"
        spec = {} if spec is None else spec
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: settings must be a mapping.")
        extra = sorted(set(spec) - _OPERATOR_KEYS)
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}; expected "
                             f"{sorted(_OPERATOR_KEYS)}.")
        reference = bool(spec.get("reference", False))
        if reference:
            if "shift" in spec or "shift_sd" in spec:
                raise ValueError(f"{where} is the reference; it has no shift.")
            operators[label] = {"reference": True, "shift_kT": 0.0,
                                "shift_sd_kT": 0.0}
            continue
        for key in ("shift", "shift_sd"):
            if key not in spec:
                raise ValueError(f"{where} needs '{key}' (or reference: "
                                 f"true).")
        operators[label] = {
            "reference": False,
            "shift_kT": _number(spec["shift"], f"{where} shift") * factor,
            "shift_sd_kT": _number(spec["shift_sd"], f"{where} shift_sd",
                                   nonnegative=True) * factor}
    if operators:
        refs = [k for k, v in operators.items() if v["reference"]]
        if len(refs) != 1:
            raise ValueError(f"Exactly one operator must be the reference "
                             f"(reference: true); found {refs or 'none'}.")

    tf_levels = {}
    raw_tf = cf.get("tf_levels") or {}
    if not isinstance(raw_tf, dict):
        raise ValueError("'tf_levels' must map each TF level label to its "
                         "settings.")
    for label, spec in raw_tf.items():
        label = str(label)
        where = f"tf_level '{label}'"
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: settings must be a mapping with "
                             f"conc_uM and ln_sd.")
        extra = sorted(set(spec) - _TF_LEVEL_KEYS)
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}; expected "
                             f"{sorted(_TF_LEVEL_KEYS)}.")
        for key in ("conc_uM", "ln_sd"):
            if key not in spec:
                raise ValueError(f"{where} needs '{key}'.")
        tf_levels[label] = {
            "conc_M": _number(spec["conc_uM"], f"{where} conc_uM",
                              positive=True) * UM_TO_M,
            "ln_sd": _number(spec["ln_sd"], f"{where} ln_sd",
                             nonnegative=True)}

    return {"units": units, "temperature_K": temperature_K,
            "operators": operators, "tf_levels": tf_levels,
            "normalized": True}


def check_contexts(contexts, conditions):
    """
    Check a contexts file against the labels in the data.

    Every operator and TF level the data name must be in the file. A file
    is required once the data have more than one operator or TF level;
    with one, the file is optional and its entries for that label are
    checked when given.

    Parameters
    ----------
    contexts : dict or None
        From :func:`read_contexts`.
    conditions : ThetaConditions

    Raises
    ------
    ValueError
    """
    ops = [o for o in conditions.operator_names if o != DEFAULT_LABEL]
    tfs = [t for t in conditions.tf_level_names if t != DEFAULT_LABEL]
    many = len(conditions.operator_names) > 1 or \
        len(conditions.tf_level_names) > 1
    if contexts is None:
        if many:
            raise ValueError(
                f"The data name operators {list(conditions.operator_names)} "
                f"and TF levels {list(conditions.tf_level_names)}; give "
                f"their priors in a contexts file (--contexts).")
        return
    if len(conditions.operator_names) > 1 and DEFAULT_LABEL in \
            conditions.operator_names:
        raise ValueError(
            f"Some rows have no '{OPERATOR_COLUMN}' and others do; label "
            f"every row once there is more than one operator.")
    if len(conditions.tf_level_names) > 1 and DEFAULT_LABEL in \
            conditions.tf_level_names:
        raise ValueError(
            f"Some rows have no '{TF_LEVEL_COLUMN}' and others do; label "
            f"every row once there is more than one TF level.")
    for kind, names, given in (("operator", ops, contexts["operators"]),
                               ("tf_level", tfs, contexts["tf_levels"])):
        missing = sorted(set(names) - set(given))
        if missing and (given or len(names) > 1):
            raise ValueError(f"The contexts file has no entry for "
                             f"{kind} {missing} (given: {sorted(given)}).")
        unused = sorted(set(given) - set(names))
        if unused:
            raise ValueError(f"The contexts file names {kind} {unused}, "
                             f"which no data row has (data: {names}).")
