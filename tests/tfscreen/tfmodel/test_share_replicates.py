import pytest
import pandas as pd
import numpy as np

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

def get_mock_df():
    # 4 genotypes, 2 condition pre, 2 condition sel, 2 titrant concs, 3 reps
    data = []
    
    # Let's say we have genotypes A, B
    for rep in [1, 2]:
        for cond_pre in ['pre-1', 'pre-2']:
            for cond_sel in ['sel+1']:
                for tname in ['inducer']:
                    for tconc in [0.0, 1.0]:
                        for geno in ['wt', 'A1T']:
                            data.append({
                                "library": "lib",
                                "genotype": geno,
                                "condition_pre": cond_pre,
                                "condition_sel": cond_sel,
                                "titrant_name": tname,
                                "titrant_conc": tconc,
                                "replicate": rep,
                                "ln_cfu": 1.0,
                                "ln_cfu_std": 0.1,
                                "t_pre": 10.0,
                                "t_sel": 10.0,
                                "theta_obs": 0.5,
                                "theta_std": 0.1
                            })
                            
    df = pd.DataFrame(data)
    growth_df = df.copy()
    binding_df = df[df['cond_pre'] == 'pre-1'].copy() if False else df.copy() # Just pass identical df for mock
    return growth_df, binding_df


def test_growth_shares_replicates():
    
    growth_df, binding_df = get_mock_df()
    
    # 1. Test without sharing replicates (Default)
    model_default = ModelOrchestrator(growth_df=growth_df, binding_df=binding_df)
    
    # Map condition conceptually defines (rep, cond) combinations pooled across pre and sel
    # Reps = 2, cond_pre = 2, cond_sel = 1 -> Total 6 combinations
    assert model_default.data.growth.num_condition_rep == 6
    assert model_default.settings["growth_shares_replicates"] is False
    
    # 2. Test WITH sharing replicates
    model_shared = ModelOrchestrator(growth_df=growth_df, binding_df=binding_df, growth_shares_replicates=True)
    
    # Should only be 3 combinations because replicate is ignored (cond_pre=2 + cond_sel=1 = 3)
    assert model_shared.data.growth.num_condition_rep == 3
    assert model_shared.settings["growth_shares_replicates"] is True
    
    # Ensure that the mappings naturally map both reps to the same parameter index
    # We can inspect the map tensor created under the hood
    df_mapped = model_shared.growth_tm.df
    
    # Find rows for rep 1 and rep 2 for the same condition
    rep1_map = df_mapped[(df_mapped["replicate"] == 1) & (df_mapped["condition_pre"] == 'pre-1')]["map_condition_pre"].iloc[0]
    rep2_map = df_mapped[(df_mapped["replicate"] == 2) & (df_mapped["condition_pre"] == 'pre-1')]["map_condition_pre"].iloc[0]
    
    assert rep1_map == rep2_map, "Replicates did not receive the same parameter mapping index!"
