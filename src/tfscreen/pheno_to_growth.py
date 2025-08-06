import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def pheno_to_growth(lib_phenotype,
                    sel_name,
                    pheno_0_growth,
                    pheno_1_growth,
                    ln_base_growth_rate=-4,
                    ln_base_growth_rate_std=0.5,
                    pheno_name="obs",
                    return_df=False,
                    save_path=None,
                    scale_pheno_by=1):
    """
    Convert the phenotype into a growth rate given growth rates.
    """

    for lib in lib_phenotype:
        print(f"Calculating growth rates for library {lib} with {sel_name}", flush=True)

        for clone in tqdm(lib_phenotype[lib]):
            # Assign base growth rate if missing
            if "base_growth_rate" not in clone:
                if len(clone["clone"]) == 0:
                    ln_k = ln_base_growth_rate
                else:
                    if ln_base_growth_rate_std <= 0:
                        ln_k = ln_base_growth_rate
                    else:
                        ln_k = np.random.normal(ln_base_growth_rate, ln_base_growth_rate_std)
                clone["base_growth_rate"] = np.exp(ln_k)

            # Scale phenotype growth rates
            p0g = clone["base_growth_rate"] * pheno_0_growth
            p1g = clone["base_growth_rate"] * pheno_1_growth

            # Compute growth rate under selection
            clone[f"sel_{sel_name}"] = clone[pheno_name]*scale_pheno_by*(p1g - p0g) + p0g
            clone[f"{sel_name}_p0g"] = p0g
            clone[f"{sel_name}_p1g"] = p1g

    if return_df:
        all_clones = []

        params = {
            "sel_name": sel_name,
            "pheno_0_growth": pheno_0_growth,
            "pheno_1_growth": pheno_1_growth,
            "ln_base_growth_rate": ln_base_growth_rate,
            "ln_base_growth_rate_std": ln_base_growth_rate_std,
            "pheno_name": pheno_name,
        }

        for lib_name, clones in lib_phenotype.items():
            for clone in clones:
                clone_copy = clone.copy()
                clone_copy["library"] = lib_name
                clone_copy.update(params)
                all_clones.append(clone_copy)

        df_all = pd.DataFrame(all_clones)

        if save_path:
            df_all.to_csv(save_path, index=False)
            print(f"Saved combined DataFrame to: {save_path}")

        return lib_phenotype, df_all

    return lib_phenotype
