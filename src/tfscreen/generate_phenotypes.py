import eee

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def _read_ddG(ddG_spreadsheet,ens):
    
    ddG_table = pd.read_excel(ddG_spreadsheet)
    
    ddG_dict = {}
    for idx in ddG_table.index:
        mut = str(ddG_table.loc[idx,"mut"])
        ddG_values = np.array(ddG_table.loc[idx,ens.species],dtype=float)
        ddG_dict[mut] = ddG_values
    
    
    # HACK FOR NOW --> CALL C AS S
    for mut in list(ddG_dict.keys()):
        if mut[-1] == "S":
            ddG_dict[f"{mut[:-1]}C"] = ddG_dict[mut]

    return ddG_dict

def _calc_lib_phenos(library,
                    ensemble_spreadsheet,
                    ddG_spreadsheet,
                    concs_mM,
                    T=310,
                    R=0.001987):

    ens = eee.io.read_ensemble(ensemble_spreadsheet)

    ddG_dict = _read_ddG(ddG_spreadsheet,ens=ens)

    concs_for_eee = -R*T*np.log(concs_mM*1e-3)
    ens.read_ligand_dict(ligand_dict={"iptg":concs_for_eee})
    T_array = np.array([T])

    lib_details = []
    for clone in tqdm(library):
        
        clone_ddG = np.zeros(len(ens.species),dtype=float)
        for mut in clone:
            clone_ddG += ddG_dict[mut]
    
        fx_occupied, fx_folded = ens.get_fx_obs_fast(mut_energy_array=clone_ddG,
                                                     temperature=T_array)


        
    
        lib_details.append({})
        lib_details[-1]["clone"] = clone
        lib_details[-1]["ddG"] = clone_ddG
        lib_details[-1]["fx_occupied"] = fx_occupied
        lib_details[-1]["fx_folded"] = fx_folded
        lib_details[-1]["obs"] = fx_occupied*fx_folded

    return lib_details
    
def generate_phenotypes(libraries,
                        ensemble_spreadsheet,
                        ddG_spreadsheet,
                        concs_mM,
                        T=310,
                        R=0.001987):


    lib_details = {}
    for lib in libraries:
        print(f"calculating phenotypes for library {lib}",flush=True)
        lib_details[lib] = _calc_lib_phenos(libraries[lib],
                                            ensemble_spreadsheet=ensemble_spreadsheet,
                                            ddG_spreadsheet=ddG_spreadsheet,
                                            concs_mM=concs_mM,
                                            T=T,
                                            R=R)

    return lib_details