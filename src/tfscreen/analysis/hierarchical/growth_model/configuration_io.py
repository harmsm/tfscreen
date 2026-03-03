import os
import yaml
import pandas as pd
import numpy as np
import jax.numpy as jnp
import dataclasses
import warnings
import tfscreen
from tfscreen.__version__ import __version__

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

def _update_dataclass(dc, prefix, flat_dict):
    """Recursively update a dataclass from a flat dictionary of values."""
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
            elif full_key in flat_dict:
                updates[field_name] = flat_dict[full_key]
                
        if len(updates) > 0:
            if hasattr(dc, "replace"):
                return dc.replace(**updates)
            else:
                return dataclasses.replace(dc, **updates)
    return dc

def write_configuration(gm, 
                        out_root, 
                        growth_df_path, 
                        binding_df_path):
    """
    Write model configuration and extracted priors/guesses to files.

    Parameters
    ----------
    gm : GrowthModel
        Initialized GrowthModel object.
    out_root : str
        Root filename for output files.
    growth_df_path : str
        Path to growth data CSV.
    binding_df_path : str
        Path to binding data CSV.
    """
    # Construct priors and guesses dataframes
    priors_list = []
    guesses_list = []

    # Extract priors
    priors_raw = _extract_scalars(gm.priors)
    for k, v in priors_raw.items():
        priors_list.append(pd.DataFrame({"parameter": [k], "value": [v]}))

    # Extract guesses
    for k, v in gm.init_params.items():
        if not hasattr(v, 'shape') or len(v.shape) == 0:
            guesses_list.append(pd.DataFrame({"parameter": [k], "value": [v]}))

    yaml_path = f"{out_root}_config.yaml"
    priors_path = f"{out_root}_priors.csv"
    guesses_path = f"{out_root}_guesses.csv"

    config = {
        "tfscreen_version": __version__,
        "data": {
            "growth": growth_df_path,
            "binding": binding_df_path
        },
        "components": gm.settings,
        "priors_file": os.path.basename(priors_path),
        "guesses_file": os.path.basename(guesses_path)
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote configuration to {yaml_path}")

    # Process array guesses and any others
    tm = gm.growth_tm
    cond_map = tm.map_groups.get("condition", pd.DataFrame())
    geno_map = tm.map_groups.get("genotype", pd.DataFrame())
    theta_map = tm.map_groups.get("theta", pd.DataFrame())
    ln_cfu0_map = tm.map_groups.get("ln_cfu0", pd.DataFrame())

    for k, v in gm.init_params.items():
        if not hasattr(v, 'shape') or len(v.shape) == 0:
            continue
            
        arr = np.array(v)
        flat_val = arr.flatten()
        
        df = pd.DataFrame({"parameter": k, "value": flat_val, "flat_index": range(len(flat_val))})
        
        if "condition_growth" in k and not "hyper" in k:
            if not cond_map.empty and len(flat_val) == len(cond_map):
                sorted_map = cond_map.sort_values("map_condition").reset_index(drop=True)
                for col in ["replicate", "condition"]:
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

    if len(priors_list) > 0:
        final_priors = pd.concat(priors_list, ignore_index=True)
        final_priors.to_csv(priors_path, index=False)
        print(f"Wrote priors to {priors_path}")

    if len(guesses_list) > 0:
        final_guesses = pd.concat(guesses_list, ignore_index=True)
        cols = ["parameter", "value"] + [c for c in final_guesses.columns if c not in ["parameter", "value"]]
        final_guesses[cols].to_csv(guesses_path, index=False)
        print(f"Wrote guesses to {guesses_path}")

from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass as GrowthModel

def read_configuration(config_file):
    """
    Read the configuration file and initialize the GrowthModel and init_params.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    gm : GrowthModel
        Initialized GrowthModel object.
    init_params : dict
        Dictionary of initial parameters for the model.
    """

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Handle both old and new formats
    if "data" in config and "components" in config:
        growth_df_path = config["data"]["growth"]
        binding_df_path = config["data"]["binding"]
        settings = config["components"]
    elif "growth_df" in config and "binding_df" in config and "settings" in config:
        growth_df_path = config["growth_df"]
        binding_df_path = config["binding_df"]
        settings = config["settings"]
    else:
        raise ValueError(f"Configuration file '{config_file}' has an unrecognized format.")

    if "tfscreen_version" in config:
        if config["tfscreen_version"] != __version__:
            warnings.warn(f"Configuration file version {config['tfscreen_version']} does not match current tfscreen version {__version__}")

    batch_size = settings.pop("batch_size", None)

    gm = GrowthModel(growth_df_path,
                     binding_df_path,
                     batch_size=batch_size,
                     **settings)

    # Update Priors from CSV
    priors_file = config.get("priors_file")
    if priors_file is None:
        raise ValueError(f"priors_file not specified in {config_file}")
    
    priors_path = os.path.join(os.path.dirname(config_file), priors_file)
    if not os.path.exists(priors_path):
        raise FileNotFoundError(f"Priors file not found: {priors_path}")

    priors_df = pd.read_csv(priors_path)
    flat_priors = {}
    for k, v in zip(priors_df["parameter"], priors_df["value"]):
        try:
            flat_priors[k] = float(v)
        except (ValueError, TypeError):
            flat_priors[k] = v
    
    new_priors = _update_dataclass(gm.priors, "", flat_priors)
    gm._priors = new_priors

    # Construct init_params from CSV
    guesses_file = config.get("guesses_file")
    if guesses_file is None:
        raise ValueError(f"guesses_file not specified in {config_file}")

    guesses_path = os.path.join(os.path.dirname(config_file), guesses_file)
    if not os.path.exists(guesses_path):
        raise FileNotFoundError(f"Guesses file not found: {guesses_path}")

    guesses_df = pd.read_csv(guesses_path)
    init_params = {}
    
    for param_name, df_group in guesses_df.groupby("parameter"):
        if "flat_index" in df_group or any(c.startswith("dim_") for c in df_group.columns):
            sorted_group = df_group.sort_values("flat_index") if "flat_index" in df_group else df_group
            val_array = sorted_group["value"].values
            
            if param_name in gm.init_params:
                orig_val = gm.init_params[param_name]
                if hasattr(orig_val, 'shape') and orig_val.shape != ():
                    orig_shape = orig_val.shape
                    init_params[param_name] = jnp.array(val_array.reshape(orig_shape))
                else:
                    init_params[param_name] = float(val_array[0])
        else:
            init_params[param_name] = float(df_group["value"].iloc[0])

    missing_params = []
    for k in gm.init_params.keys():
        if k not in init_params:
            missing_params.append(k)
    
    if len(missing_params) > 0:
        raise ValueError(f"Missing initial guesses for parameters: {missing_params}")

    return gm, init_params
