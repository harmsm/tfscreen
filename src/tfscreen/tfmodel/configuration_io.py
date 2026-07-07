import os
import yaml
import pandas as pd
import numpy as np
import jax.numpy as jnp
import warnings
from tfscreen.__version__ import __version__
from tfscreen.util.validation import check_unknown_keys

# All recognized top-level keys for a tfmodel config file.
TFMODEL_KNOWN_KEYS = frozenset({
    "tfscreen_version",
    "data",
    "components",
    "priors_file",
    "guesses_file",
})

def _extract_scalars(obj, prefix=""):
    """Recursively extract scalar values from PriorsClass."""
    out = {}
    for k in dir(obj):
        if k.startswith("_") or k in ['replace', 'asdict', '__class__', 'tree_flatten', 'tree_unflatten']:
            continue
        v = getattr(obj, k)
        if hasattr(v, '__dataclass_fields__'):
            out.update(_extract_scalars(v, prefix + k + "."))
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if not hasattr(sub_v, 'shape') or len(sub_v.shape) == 0:
                    out[f"{prefix}{k}.{sub_k}"] = float(sub_v) if isinstance(sub_v, (int, float, np.number)) else str(sub_v)
        else:
            if not hasattr(v, 'shape') or len(np.array(v).shape) == 0:
                out[f"{prefix}{k}"] = float(v) if isinstance(v, (int, float, np.number)) else str(v)
    return out

def _gather_dict_field(full_key, flat_dict):
    """
    Collect a one-level-deep sub-dictionary for a dict-typed dataclass field.

    Looks for keys of the form ``"{full_key}.{sub_key}"`` in ``flat_dict``
    (with no further dots) and returns ``{sub_key: value}``.  Values are
    coerced to ``float`` when possible, falling back to the raw value.

    Returns
    -------
    dict
        The collected sub-dictionary; empty if no matching keys were found.
    """
    sub_dict = {}
    sub_prefix = f"{full_key}."
    for k, v in flat_dict.items():
        if not k.startswith(sub_prefix):
            continue
        suffix = k[len(sub_prefix):]
        # Only pick up keys that name a single suffix; deeper nesting
        # (suffix containing another '.') indicates a nested dataclass and
        # is handled by the recursive walk in _update_dataclass.
        if "." in suffix:
            continue
        try:
            sub_dict[suffix] = float(v)
        except (ValueError, TypeError):
            sub_dict[suffix] = v
    return sub_dict


def _update_dataclass(dc, prefix, flat_dict):
    """
    Recursively update a dataclass from a flat dictionary of values.

    Supports three field types:
    - Nested dataclasses (recurse with extended prefix).
    - Scalar leaves (consume ``flat_dict[full_key]``).
    - Dict-typed fields (consume any ``flat_dict[full_key.<sub>]`` keys
      via :func:`_gather_dict_field`).  Used by the ``pinned`` field on
      component ``ModelPriors`` to inject empirical-Bayes hyperprior
      values from the calibration pre-fit.
    """
    import dataclasses
    updates = {}

    # Check if this is a standard dataclass or a flax struct dataclass
    if hasattr(dc, '__dataclass_fields__'):
        fields = dataclasses.fields(dc) if dataclasses.is_dataclass(dc) else dc.__dataclass_fields__.values()

        for field in fields:
            field_name = field.name
            full_key = f"{prefix}{field_name}" if prefix else field_name
            attr_val = getattr(dc, field_name)

            if hasattr(attr_val, '__dataclass_fields__'):
                updates[field_name] = _update_dataclass(attr_val, full_key + ".", flat_dict)
            elif isinstance(attr_val, dict):
                # Dict-typed field — gather any "{full_key}.<suffix>" rows.
                # Merge with the existing dict so previously-set entries
                # are preserved when the CSV doesn't override them.
                sub = _gather_dict_field(full_key, flat_dict)
                if sub:
                    merged = dict(attr_val)
                    merged.update(sub)
                    updates[field_name] = merged
            elif isinstance(attr_val, tuple):
                # Static structural tuples (e.g. condition_growth.m_is_selection)
                # are re-derived at construction and serialised by
                # _extract_scalars as a *string* (e.g. "(True, False)").  Never
                # overwrite the correctly-typed tuple with that string — keep
                # the freshly-built value.
                continue
            elif full_key in flat_dict:
                new_val = flat_dict[full_key]
                if isinstance(attr_val, int) and not isinstance(attr_val, bool):
                    new_val = int(new_val)
                updates[field_name] = new_val

        if len(updates) > 0:
            if hasattr(dc, "replace"):
                return dc.replace(**updates)
            else:
                return dataclasses.replace(dc, **updates)
    return dc

def _extract_prior_arrays(obj, prefix=""):
    """
    Recursively collect 1-D *floating-point* array leaves from a PriorsClass,
    keyed by dotted path.

    Companion to :func:`_extract_scalars`, which emits only scalar leaves and
    silently skips arrays.  Per-condition priors (e.g. ``condition_growth``'s
    ``k_loc`` after the pre-fit calibration writes per-condition values) live in
    array leaves, so they need their own extraction to reach the priors CSV.

    Only floating arrays are returned: static config tuples such as
    ``m_is_selection`` (a tuple of bools) must never be mistaken for a
    per-condition prior.
    """
    out = {}
    for k in dir(obj):
        if k.startswith("_") or k in ['replace', 'asdict', '__class__',
                                      'tree_flatten', 'tree_unflatten']:
            continue
        v = getattr(obj, k)
        if hasattr(v, '__dataclass_fields__'):
            out.update(_extract_prior_arrays(v, prefix + k + "."))
        elif isinstance(v, dict) or not hasattr(v, 'shape'):
            # dicts (e.g. `pinned`) and non-array leaves (scalars, static
            # tuples like m_is_selection) are not per-condition arrays.
            continue
        else:
            try:
                arr = np.asarray(v)
            except (ValueError, TypeError):
                continue
            if arr.ndim >= 1 and arr.size > 0 and arr.dtype.kind == "f":
                out[f"{prefix}{k}"] = arr
    return out


def _assemble_condition_array(param, grp, cond_rep_map):
    """
    Assemble one per-condition prior array from its indexed CSV rows.

    The rows are joined to the model's condition order by ``condition_rep``
    *name* (and ``replicate`` when present), not by raw integer index, so a
    priors CSV stays correct even if conditions are reordered.  Fails fast if
    the CSV references an unknown condition or is missing any condition — a
    stale priors file should error, not silently mis-map (see the fail-fast
    config-validation convention).

    Falls back to ``flat_index`` ordering when no ``condition_rep`` map or
    label columns are available (e.g. non-growth arrays).
    """
    if cond_rep_map is None or getattr(cond_rep_map, "empty", True):
        ordered = grp.sort_values("flat_index") if "flat_index" in grp else grp
        return jnp.asarray(ordered["value"].to_numpy(dtype=float))

    sorted_map = cond_rep_map.sort_values("map_condition_rep").reset_index(drop=True)
    label_cols = [c for c in ("replicate", "condition_rep")
                  if c in sorted_map.columns and c in grp.columns]

    if not label_cols:
        ordered = grp.sort_values("flat_index") if "flat_index" in grp else grp
        arr = ordered["value"].to_numpy(dtype=float)
        if len(arr) != len(sorted_map):
            raise ValueError(
                f"Prior '{param}' has {len(arr)} indexed row(s) in the priors "
                f"CSV but the model has {len(sorted_map)} condition(s), and no "
                f"condition_rep labels are present to join on. Regenerate the "
                f"priors file with tfs-configure-model."
            )
        return jnp.asarray(arr)

    key_to_pos = {
        tuple(str(sorted_map.loc[i, c]) for c in label_cols): i
        for i in range(len(sorted_map))
    }

    arr = np.full(len(sorted_map), np.nan, dtype=float)
    for _, r in grp.iterrows():
        key = tuple(str(r[c]) for c in label_cols)
        pos = key_to_pos.get(key)
        if pos is None:
            raise ValueError(
                f"Prior '{param}' references condition {key} (columns "
                f"{label_cols}) in the priors CSV, which is not one of the "
                f"model's conditions {sorted(key_to_pos)}. The priors file is "
                f"likely stale — regenerate it with tfs-configure-model."
            )
        arr[pos] = float(r["value"])

    missing = [k for k, i in key_to_pos.items() if np.isnan(arr[i])]
    if missing:
        raise ValueError(
            f"Prior '{param}' is missing per-condition row(s) for {missing} in "
            f"the priors CSV; every condition_rep must have a row. Regenerate "
            f"with tfs-configure-model / tfs-prefit-calibration."
        )
    return jnp.asarray(arr)


def _read_priors_flat(priors_df, cond_rep_map):
    """
    Build the flat prior dict from a priors CSV.

    Scalar rows (no ``flat_index``, or ``flat_index`` NaN) are read as scalars
    exactly as before.  Rows carrying a ``flat_index`` are grouped by parameter
    and assembled into per-condition arrays via
    :func:`_assemble_condition_array`.  A legacy 2-column CSV (no ``flat_index``)
    takes the pure-scalar path untouched.
    """
    def _coerce(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v

    if "flat_index" not in priors_df.columns:
        return {k: _coerce(v)
                for k, v in zip(priors_df["parameter"], priors_df["value"])}

    scalar_mask = priors_df["flat_index"].isna()
    flat = {row["parameter"]: _coerce(row["value"])
            for _, row in priors_df[scalar_mask].iterrows()}

    indexed = priors_df[~scalar_mask]
    for param, grp in indexed.groupby("parameter", observed=True):
        flat[param] = _assemble_condition_array(param, grp, cond_rep_map)

    return flat


def write_configuration(orchestrator,
                        out_prefix,
                        growth_df_path=None,
                        binding_df_path=None,
                        presplit_df_path=None,
                        base_growth_df_path=None):
    """
    Write model configuration and extracted priors/guesses to files.

    Parameters
    ----------
    orchestrator : ModelOrchestrator
        Initialized ModelOrchestrator object.
    out_prefix : str
        Root filename for output files.
    growth_df_path : str
        Path to growth data CSV.
    binding_df_path : str
        Path to binding data CSV.
    presplit_df_path : str, optional
        Path to the pre-split observation CSV.
    base_growth_df_path : str, optional
        Path to the base_growth_df CSV (direct growth-rate measurements
        anchoring the k_ref latent; see model_orchestrator._read_base_growth_df).
    """
    # Construct priors and guesses dataframes
    priors_list = []
    guesses_list = []

    # Extract priors
    priors_raw = _extract_scalars(orchestrator.priors)
    for k, v in priors_raw.items():
        priors_list.append(pd.DataFrame({"parameter": [k], "value": [v]}))

    # Extract guesses
    for k, v in orchestrator.init_params.items():
        if not hasattr(v, 'shape') or len(v.shape) == 0:
            guesses_list.append(pd.DataFrame({"parameter": [k], "value": [v]}))

    yaml_path = f"{out_prefix}_config.yaml"
    priors_path = f"{out_prefix}_priors.csv"
    guesses_path = f"{out_prefix}_guesses.csv"

    data_paths = {}
    if growth_df_path is not None:
        data_paths["growth"] = growth_df_path
    if binding_df_path is not None:
        data_paths["binding"] = binding_df_path
    if presplit_df_path is not None:
        data_paths["presplit"] = presplit_df_path
    if base_growth_df_path is not None:
        data_paths["base_growth"] = base_growth_df_path

    config = {
        "tfscreen_version": __version__,
        "data": data_paths,
        "components": orchestrator.settings,
        "priors_file": os.path.basename(priors_path),
        "guesses_file": os.path.basename(guesses_path)
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote configuration to {yaml_path}")

    # Process array guesses and any others
    tm = orchestrator.growth_tm
    if tm is not None:
        cond_rep_map = tm.map_groups.get("condition_rep", pd.DataFrame())
        geno_map = tm.map_groups.get("genotype", pd.DataFrame())
        theta_map = tm.map_groups.get("theta", pd.DataFrame())
        ln_cfu0_map = tm.map_groups.get("ln_cfu0", pd.DataFrame())
    else:
        cond_rep_map = pd.DataFrame()
        geno_map = pd.DataFrame()
        theta_map = pd.DataFrame()
        ln_cfu0_map = pd.DataFrame()

    for k, v in orchestrator.init_params.items():
        if not hasattr(v, 'shape') or len(v.shape) == 0:
            continue
            
        arr = np.array(v)
        flat_val = arr.flatten()
        
        df = pd.DataFrame({"parameter": k, "value": flat_val, "flat_index": range(len(flat_val))})
        
        if "condition_growth" in k and not "hyper" in k:
            if not cond_rep_map.empty and len(flat_val) == len(cond_rep_map):
                sorted_map = cond_rep_map.sort_values("map_condition_rep").reset_index(drop=True)
                for col in ["replicate", "condition_rep"]:
                    if col in sorted_map.columns:
                        df[col] = sorted_map[col].values
        
        elif "theta" in k and not "hyper" in k:
            if not theta_map.empty and len(flat_val) == len(theta_map):
                sorted_map = theta_map.sort_values("map_theta").reset_index(drop=True)
                for col in ["titrant_name", "titrant_conc", "genotype"]:
                    if col in sorted_map.columns:
                        df[col] = sorted_map[col].values
            else: 
                df["flat_index"] = range(len(flat_val))

        elif k.startswith("dk_geno") or k.startswith("activity"):
            if not geno_map.empty and len(flat_val) == len(geno_map):
                sorted_map = geno_map.sort_values("map_genotype").reset_index(drop=True)
                if "genotype" in sorted_map.columns:
                    df["genotype"] = sorted_map["genotype"].values

        elif k.startswith("ln_cfu0"):
            if not ln_cfu0_map.empty and len(flat_val) == len(ln_cfu0_map):
                sorted_map = ln_cfu0_map.sort_values("map_ln_cfu0").reset_index(drop=True)
                for col in ["replicate", "condition_pre", "genotype"]:
                    if col in sorted_map.columns:
                        df[col] = sorted_map[col].values

        if len(df.columns) == 3: # parameter, value, flat_index
            if len(arr.shape) == 1:
                df["dim_0"] = range(arr.shape[0])
            elif len(arr.shape) == 2:
                idx = np.indices(arr.shape)
                df["dim_0"] = idx[0].flatten()
                df["dim_1"] = idx[1].flatten()
            elif len(arr.shape) == 3:
                idx = np.indices(arr.shape)
                df["dim_0"] = idx[0].flatten()
                df["dim_1"] = idx[1].flatten()
                df["dim_2"] = idx[2].flatten()
                
        guesses_list.append(df)

    # Per-condition array priors (e.g. condition_growth k_loc/k_scale once the
    # pre-fit calibration has written per-condition values).  _extract_scalars
    # skips array leaves, so emit them here as indexed rows tagged with
    # condition_rep labels, mirroring the per-condition guesses rows.  Restricted
    # to the growth linking-function components so other components' internal
    # arrays are untouched.  On a fresh configure these fields are still scalar,
    # so nothing is emitted here and the priors CSV stays 2-column (legacy).
    for dotted_key, arr in _extract_prior_arrays(orchestrator.priors).items():
        if ("condition_growth" not in dotted_key
                and "growth_transition" not in dotted_key):
            continue
        flat_val = np.asarray(arr).flatten()
        df = pd.DataFrame({"parameter": dotted_key,
                           "value": flat_val,
                           "flat_index": range(len(flat_val))})
        if (not cond_rep_map.empty) and len(flat_val) == len(cond_rep_map):
            sorted_map = cond_rep_map.sort_values("map_condition_rep").reset_index(drop=True)
            for col in ("replicate", "condition_rep"):
                if col in sorted_map.columns:
                    df[col] = sorted_map[col].values
        priors_list.append(df)

    if len(priors_list) > 0:
        final_priors = pd.concat(priors_list, ignore_index=True)
        final_priors.to_csv(priors_path, index=False)
        print(f"Wrote priors to {priors_path}")

    if len(guesses_list) > 0:
        final_guesses = pd.concat(guesses_list, ignore_index=True)
        cols = ["parameter", "value"] + [c for c in final_guesses.columns if c not in ["parameter", "value"]]
        final_guesses[cols].to_csv(guesses_path, index=False)
        print(f"Wrote guesses to {guesses_path}")

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

def read_configuration(config_file):
    """
    Read the configuration file and initialize the ModelOrchestrator and init_params.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    orchestrator : ModelOrchestrator
        Initialized ModelOrchestrator object.
    init_params : dict
        Dictionary of initial parameters for the model.
    """

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    check_unknown_keys(config, TFMODEL_KNOWN_KEYS, label="tfmodel config")

    # Check sanity of format and read in data paths and components
    if "data" in config and "components" in config:
        growth_df_path = config["data"].get("growth")
        binding_df_path = config["data"].get("binding")
        presplit_df_path = config["data"].get("presplit")
        base_growth_df_path = config["data"].get("base_growth")
        settings = config["components"]
    else:
        raise ValueError(f"Configuration file '{config_file}' has an unrecognized format.")

    if "tfscreen_version" in config:
        if config["tfscreen_version"] != __version__:
            warnings.warn(f"Configuration file version {config['tfscreen_version']} does not match current tfscreen version {__version__}")

    batch_size = settings.pop("batch_size", None)
    # presplit_df / base_growth_df are data paths, not component settings;
    # pop them since they're already passed explicitly below (they end up
    # in settings because ModelOrchestrator.settings includes them for
    # round-tripping the raw path the user originally supplied).
    settings.pop("presplit_df", None)
    settings.pop("base_growth_df", None)

    orchestrator = ModelOrchestrator(growth_df_path,
                     binding_df_path,
                     batch_size=batch_size,
                     presplit_df=presplit_df_path,
                     base_growth_df=base_growth_df_path,
                     **settings)

    # Update Priors from CSV
    priors_file = config.get("priors_file")
    if priors_file is None:
        raise ValueError(f"priors_file not specified in {config_file}")
    
    priors_path = os.path.join(os.path.dirname(config_file), priors_file)
    if not os.path.exists(priors_path):
        raise FileNotFoundError(f"Priors file not found: {priors_path}")

    priors_df = pd.read_csv(priors_path)
    cond_rep_map = None
    if orchestrator.growth_tm is not None:
        cond_rep_map = orchestrator.growth_tm.map_groups.get("condition_rep")
    flat_priors = _read_priors_flat(priors_df, cond_rep_map)

    new_priors = _update_dataclass(orchestrator.priors, "", flat_priors)
    orchestrator._priors = new_priors

    # Construct init_params from CSV
    guesses_file = config.get("guesses_file")
    if guesses_file is None:
        raise ValueError(f"guesses_file not specified in {config_file}")

    guesses_path = os.path.join(os.path.dirname(config_file), guesses_file)
    if not os.path.exists(guesses_path):
        raise FileNotFoundError(f"Guesses file not found: {guesses_path}")

    guesses_df = pd.read_csv(guesses_path)
    init_params = {}
    
    for param_name, df_group in guesses_df.groupby("parameter", observed=True):
        if "flat_index" in df_group or any(c.startswith("dim_") for c in df_group.columns):
            sorted_group = df_group.sort_values("flat_index") if "flat_index" in df_group else df_group
            val_array = sorted_group["value"].values
            
            if param_name in orchestrator.init_params:
                orig_val = orchestrator.init_params[param_name]
                if hasattr(orig_val, 'shape') and orig_val.shape != ():
                    orig_shape = orig_val.shape
                    if val_array.size != np.prod(orig_shape):
                        raise ValueError(
                            f"Parameter '{param_name}' has {val_array.size} "
                            f"{'value' if val_array.size == 1 else 'values'} in "
                            f"'{guesses_file}' but the current model expects shape "
                            f"{orig_shape} ({int(np.prod(orig_shape))} values).  "
                            f"The guesses file is likely stale — regenerate it with "
                            f"tfs-configure-model and re-run tfs-prefit-calibration."
                        )
                    init_params[param_name] = jnp.array(val_array.reshape(orig_shape))
                else:
                    init_params[param_name] = float(val_array[0])
        else:
            init_params[param_name] = float(df_group["value"].iloc[0])

    missing_params = []
    for k in orchestrator.init_params.keys():
        if k not in init_params:
            missing_params.append(k)

    if len(missing_params) > 0:
        raise ValueError(f"Missing initial guesses for parameters: {missing_params}")

    return orchestrator, init_params
