import os
import yaml
import pandas as pd
import numpy as np
import tfscreen
from tfscreen.analysis.hierarchical.growth_model import GrowthModel
from tfscreen.util.cli.generalized_main import generalized_main

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

def configure_growth_analysis(growth_df=None,
                              binding_df=None,
                              out_root="tfs",
                              condition_growth_model="linear",
                              growth_transition_model="instant",
                              ln_cfu0_model="hierarchical",
                              dk_geno_model="hierarchical",
                              activity_model="horseshoe",
                              theta_model="hill",
                              transformation_model="empirical",
                              theta_growth_noise_model="zero",
                              theta_binding_noise_model="zero",
                              spiked=None):
    """
    Construct the analysis configuration step. This creates a tfs_config.yaml file
    along with tfs_priors.csv and tfs_guesses.csv (if any array parameters exist).

    Parameters
    ----------
    growth_df : pandas.DataFrame or str, optional
        Input data for the growth model (e.g., genotype/cfu measurements).
    binding_df : pandas.DataFrame or str, optional
        Input data for the binding model (e.g., theta vs. titrant measurements).
    out_root : str, optional
        Output file root for the generated configuration files (default 'tfs').
    condition_growth_model: str, optional
        Model to use to describe growth under different conditions (e.g., 
        pheS+4CP). Allowed values are 'linear' (default), 'linear_independent',
        'linear_fixed', 'power', or 'saturation'.
    growth_transition_model : str, optional
        Model to use to describe the transition between the pre-selection
        and selection phases. Allowed values are 'instant' (default), 'memory',
        or 'baranyi'.
    ln_cfu0_model : str, optional
        Model to use to describe ln_cfu0, the initial populations of genotypes
        in each replicate. Only 'hierarchical' is allowed at this point. 
    dk_geno_model : str, optional
        Model to use to describe dk_geno, the pleiotropic effect of a genotype
        on growth, independent of occupancy. Allowed values are 'hierarchical' 
        (default) or 'fixed'.
    activity_model : str, optional
        Model to use to describe activity, a scalar multiplied against 
        occupancy that defines how strongly a genotype alters transcription 
        given its occupancy. Allowed values are 'fixed' (default), 'hierarchical',
        and 'horseshoe'.
    theta_model : str, optional
        Model to use to describe theta, the fractional occupancy of a genotype
        on the transcription factor binding site. Allowed values are 'hill' 
        (default) or 'categorical'. 
    transformation_model : str, optional
        Model for transformation correction. Allowed values are 'single', 
        'empirical', 'congression', or 'logit_norm'. Default 'empirical'.
    theta_growth_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by 
        bacterial growth. Allowed values are 'beta' (default) or 'zero'.
    theta_binding_noise_model : str, optional
        Model to use for stochastic experimental noise in theta measured by 
        binding. Allowed values are 'beta' (default) or 'zero'.
    spiked : list or str, optional
        Names of genotypes that should be excluded from congression
        correction.

    Returns
    -------
    None
    """

    if growth_df is None or binding_df is None:
        raise ValueError("growth_df and binding_df must be provided.")

    # Initialize model to build mappings and get guesses
    gm = GrowthModel(growth_df,
                     binding_df,
                     condition_growth=condition_growth_model,
                     growth_transition=growth_transition_model,
                     ln_cfu0=ln_cfu0_model,
                     dk_geno=dk_geno_model,
                     activity=activity_model,
                     theta=theta_model,
                     transformation=transformation_model,
                     theta_growth_noise=theta_growth_noise_model,
                     theta_binding_noise=theta_binding_noise_model,
                     spiked_genotypes=spiked)

    config = {
        "tfscreen_version": str(tfscreen.__version__),
        "data": {
            "growth": growth_df if isinstance(growth_df, str) else "growth.csv",
            "binding": binding_df if isinstance(binding_df, str) else "binding.csv"
        },
        "components": gm.settings,
        "priors": {},
        "init_params": {}
    }

    # Extract scalar priors
    config["priors"] = _extract_scalars(gm.priors)

    # Separate scalar and array init params
    array_guesses = {}
    for k, v in gm.init_params.items():
        if hasattr(v, 'shape') and len(v.shape) > 0:
            array_guesses[k] = np.array(v)
        else:
            config["init_params"][k] = float(v)

    yaml_path = f"{out_root}_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote configuration to {yaml_path}")

    if len(array_guesses) > 0:
        # Build tfs_guesses.csv 
        dfs = []
        
        # Extract map dataframes to join against
        tm = gm.growth_tm
        
        # Create full dataframe of all unique identifiers we might need for melting
        # We can just iterate over array_guesses. Depending on their prefix, we know how to map them!
        cond_map = tm.map_groups.get("condition", pd.DataFrame())
        geno_map = tm.map_groups.get("genotype", pd.DataFrame())
        theta_map = tm.map_groups.get("theta", pd.DataFrame())
        ln_cfu0_map = tm.map_groups.get("ln_cfu0", pd.DataFrame())

        for k, arr in array_guesses.items():
            flat_val = arr.flatten()
            
            df = pd.DataFrame({"parameter": k, "value": flat_val, "flat_index": range(len(flat_val))})
            
            # Identify which mapping goes with this parameter based on known model components
            if "condition_growth" in k and not "hyper" in k:
                # corresponds to map_condition
                if not cond_map.empty and len(flat_val) == len(cond_map):
                    # cond_map is sorted by map_condition
                    sorted_map = cond_map.sort_values("map_condition").reset_index(drop=True)
                    for col in ["replicate", "condition"]:
                        if col in sorted_map.columns:
                            df[col] = sorted_map[col].values
            
            elif "theta" in k and not "hyper" in k:
                # corresponds to map_theta_group or map_theta
                # Note: some parameters are (num_titrant_name, num_genotype) - wait, map_theta handles this?
                # Actually, in model_class.py it's map_theta_group. Let's see if lengths match!
                if not theta_map.empty and len(flat_val) == len(theta_map):
                    sorted_map = theta_map.sort_values("map_theta").reset_index(drop=True)
                    for col in ["titrant_name", "titrant_conc", "genotype"]:
                        if col in sorted_map.columns:
                            df[col] = sorted_map[col].values
                else: 
                    # If it's a 2D array of (titrant_name, genotype), we might need to manually unroll it or rely on the dimensions it has.
                    # As a fallback:
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

            # Generic fallback if no columns were added except parameter and value
            if len(df.columns) == 2:
                # Give it generic indices based on shape
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
                    
            dfs.append(df)

        if len(dfs) > 0:
            final_guesses = pd.concat(dfs, ignore_index=True)
            guesses_path = f"{out_root}_guesses.csv"
            # Organize columns so parameter and value are first
            cols = ["parameter", "value"] + [c for c in final_guesses.columns if c not in ["parameter", "value"]]
            final_guesses[cols].to_csv(guesses_path, index=False)
            print(f"Wrote array guesses to {guesses_path}")

def main():
    return generalized_main(configure_growth_analysis,
                            manual_arg_types={"growth_df":str,
                                              "binding_df":str,
                                              "spiked":list},
                            manual_arg_nargs={"spiked":"+"})

if __name__ == "__main__":
    main()
