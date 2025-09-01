import numpy as np

from .microscopic_dimer import MicroscopicDimerModel
from .linkage_dimer import LinkageDimerModel
from .linkage_dimer_tetramer import LinkageDimerTetramerModel
from .mwc_dimer import MWCDimerModel

AVAILABLE_MODELS = {"microscopic_dimer":MicroscopicDimerModel,
                    "mwc_dimer":MWCDimerModel,
                    "linkage_dimer":LinkageDimerModel,
                    "linkage_dimer_tetramer":LinkageDimerTetramerModel}

class LacModel:

    def __init__(self,
                 model_to_use,
                 wt_K_dict,
                 r_total,
                 o_total,
                 e_total,
                 R,
                 T):

        self._R = R
        self._T = T
        self._RT = self._R*self._T

        if model_to_use not in AVAILABLE_MODELS:
            err = f"model '{model_to_use}' not recognized. Should be one of:\n"
            raise ValueError(err)

        self._model = AVAILABLE_MODELS[model_to_use](e_total=e_total,
                                                     o_total=o_total,
                                                     r_total=r_total)
        
        expected_K = set(self._model.equilibrium_constants)
        input_K = set(wt_K_dict.keys())
        if input_K != expected_K:
            err = "wt_K_dict does not have the correct equilibrium constants.\n"
            
            missing = expected_K - input_K
            if len(missing) > 0:
                err += "Missing constants:\n"
                missing = list(missing)
                missing.sort()
                for m in missing:
                    err += f"    {m}\n"
            
            extra = input_K - expected_K
            if len(extra) > 0:
                err += "Extra constants:\n"
                extra = list(extra)
                extra.sort()
                for e in extra:
                    err += f"    {e}\n"

            err += "\n"
            raise ValueError(err)
            
        # store wildtype equilibrium constants as an array
        wt_K_array = []
        for K in self._model.equilibrium_constants:
            wt_K_array.append(wt_K_dict[K])
        self._wt_K_array = np.array(wt_K_array)

        # Store wildtype dG values
        self._wt_dG_array = -self._RT*np.log(wt_K_array)

        # Get the names of all lac species
        self._species = [s for s in self._model.species_names
                         if s not in ["E","O"]]

        # These let us quickly look up the reactants and products for each 
        # equilibrium. When we load in ddG, we can then add to the right 
        # reactions. 
        self._species_to_idx = dict([(s,i) for i, s in enumerate(self._species)])
        self._reactant_idx = np.array([self._species_to_idx[s] for s in self._model.repressor_reactant])
        self._product_idx = np.array([self._species_to_idx[s] for s in self._model.repressor_product])

    @property
    def species(self):
        return self._species

    def get_obs(self,genotype_ddG):

        mut_dG_array = self._wt_dG_array + (genotype_ddG[self._product_idx] - genotype_ddG[self._reactant_idx])
        mut_K_array = np.exp(-mut_dG_array/self._RT)

        return self._model.get_fx_operator(mut_K_array)


