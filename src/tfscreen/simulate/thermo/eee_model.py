import numpy as np

class EEEModel:

    def __init__(self,
                 ensemble_spreadsheet,
                 scale_obs_by,
                 e_total,
                 R,
                 T):
            
        try:
            import eee
        except Exception as e:
            err = "eee library not found. eee phenotype generation not possible.\n"
            raise ImportError(err) from e

        # Create thermodynamic ensemble
        self._ens = eee.io.read_ensemble(ensemble_spreadsheet,
                                         gas_constant=R)
        iptg_chemical_potential = R*T*np.log(e_total)
        self._ens.read_ligand_dict(ligand_dict={"iptg":iptg_chemical_potential})
        
        self._T_array = np.array([T])
        self._scale_obs_by = scale_obs_by

    @property
    def species(self):
        return self._ens.species

    def get_obs(self,
                genotype_ddG):

        # Calculate ensemble features for the genotype
        fx_occupied, fx_folded = self._ens.get_fx_obs_fast(mut_energy_array=genotype_ddG,
                                                           temperature=self._T_array)
        obs = fx_occupied * fx_folded * self._scale_obs_by

        return obs

