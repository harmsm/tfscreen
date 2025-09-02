import numpy as np
from scipy.optimize import root
import warnings

class MWCDimerModel:
    """A Monod-Wyman-Changeux (MWC) style model of a lac repressor dimer.

    This model describes a dimeric repressor that exists in two conformational
    states, H (high-affinity for operator) and L (low-affinity for operator).

    Parameters
    ----------
    r_total : float or array-like
        Total concentration of repressor in **monomer** units.
    o_total : float or array-like
        Total concentration of operator (O).
    e_total : float or array-like
        Total concentration of effector (E).

    Notes
    -----
    This model is based on the formalism presented by Sochor et al. (2014),
    where the H and L states correspond to the R and R* states in the paper,
    respectively.

    **Species**
        - E, O: Free effector and operator
        - H, L: High and low DNA-affinity repressor dimer conformations
        - HE, HE2: H dimer bound to one or two effectors
        - LE, LE2: L dimer bound to one or two effectors
        - HO, LO: H or L dimer bound to operator
        - HOE, HOE2: HO complex bound to one or two effectors
        - LOE, LOE2: LO complex bound to one or two effectors

    **Equilibrium Constants**
    The `get_concs` method requires a `K_array` with 5 **association** constants:
        - K_h_l: H -> L (unitless)
        - K_h_o: H + O -> HO (M-1)
        - K_h_e: H + E -> HE (M-1)
        - K_l_o: L + O -> LO (M-1)
        - K_l_e: L + E -> LE (M-1)

    References
    ----------
    Sochor, M. A., et al. (2014). In vitro transcription accurately predicts
    lac repressor phenotype in vivo in Escherichia coli. PeerJ, 2, e498. 
    DOI: 10.7717/peerj.498
    """

    species_names = (
        'E', 'O', 'H', 'L', 'HE', 'HE2', 'LE', 'LE2',
        'HO', 'LO', 'HOE', 'HOE2', 'LOE', 'LOE2'
    )
    r_stoich = np.array([0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    e_stoich = np.array([1, 0, 0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2])
    o_stoich = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    repressor_reactant = np.array(['H', 'H', 'H', 'L', 'L'])
    repressor_product = np.array(['L', 'HO', 'HE', 'LO', 'LE'])

    equilibrium_constants = ["K_h_l","K_h_o","K_h_e","K_l_o","K_l_e"]

    def __init__(self, r_total, o_total, e_total):
        # Convert input r_total (monomer units) to the dimer units used
        # by this model's internal solver.
        r_total_as_dimer = np.asarray(r_total) / 2.0

        try:
            b_r, b_o, b_e = np.broadcast_arrays(r_total_as_dimer, o_total, e_total)
        except ValueError as e:
            raise ValueError("Input concentration shapes cannot be broadcast together.") from e

        self.r_total_dimer = b_r.ravel()
        self.o_total = b_o.ravel()
        self.e_total = b_e.ravel()
        self.num_conditions = self.e_total.size
        self._species_map = {name: i for i, name in enumerate(self.species_names)}

    def get_concs(self, K_array):
        """
        Calculates equilibrium concentrations for all species.

        Parameters
        ----------
        K_array : np.ndarray
            A 1D NumPy array of 5 **association** constants. See class
            docstring for the required order.

        Returns
        -------
        np.ndarray
            A (num_conditions x 14) array of species concentrations.
        """
        if not isinstance(K_array, np.ndarray) or K_array.shape != (5,):
            raise ValueError("K_array must be a 1D NumPy array of length 5.")
        results = np.zeros((self.num_conditions, len(self.species_names)))
        for i in range(self.num_conditions):
            solution = self._solve_single(self.e_total[i], self.o_total[i], self.r_total_dimer[i], K_array)
            results[i, :] = solution
        return results

    def get_fx_operator(self, K_array):
        """Calculates the fraction of operator bound by repressor."""
        all_concs = self.get_concs(K_array)
        o_free_conc = all_concs[:, self._species_map['O']]
        o_free_conc = np.minimum(o_free_conc, self.o_total)
        return (self.o_total - o_free_conc) / (self.o_total + 1e-30)

    def _solve_single(self, e_total, o_total, r_total_dimer, K_array):
        def _equations_log(log_free_concs, *args):

            # Ignore overflows within this calculation --> np.inf
            with np.errstate(over='ignore'):
                e_free, o_free, h_free = np.exp(log_free_concs)
                e_total, o_total, r_total_dimer, K_array = args
                K_h_l, K_h_o, K_h_e, K_l_o, K_l_e = K_array

                l_free = K_h_l * h_free
                he, he2 = K_h_e * h_free * e_free, K_h_e**2 * h_free * e_free**2
                le, le2 = K_l_e * l_free * e_free, K_l_e**2 * l_free * e_free**2
                ho, lo = K_h_o * h_free * o_free, K_l_o * l_free * o_free
                hoe, hoe2 = K_h_e * ho * e_free, K_h_e**2 * ho * e_free**2
                loe, loe2 = K_l_e * lo * e_free, K_l_e**2 * lo * e_free**2

                e_calc = e_free + he + 2*he2 + le + 2*le2 + hoe + 2*hoe2 + loe + 2*loe2
                o_calc = o_free + ho + lo + hoe + hoe2 + loe + loe2
                r_calc = h_free + l_free + he + he2 + le + le2 + ho + lo + hoe + hoe2 + loe + loe2

                return (e_total-e_calc, o_total-o_calc, r_total_dimer-r_calc)

        initial_guess = np.log(np.maximum(1e-20, [e_total, o_total, r_total_dimer]))
        solution = root(_equations_log, initial_guess, args=(e_total, o_total, r_total_dimer, K_array), method='lm')
        if not solution.success:
            warnings.warn(f"Solver failed for condition E={e_total}, O={o_total}, R={r_total_dimer*2}: {solution.message}")
            return np.full(len(self.species_names), np.nan)

        e_free, o_free, h_free = np.exp(solution.x)
        K_h_l, K_h_o, K_h_e, K_l_o, K_l_e = K_array
        concs = np.zeros(len(self.species_names))
        concs[self._species_map['E']], concs[self._species_map['O']], concs[self._species_map['H']] = e_free, o_free, h_free
        concs[self._species_map['L']] = K_h_l * h_free
        concs[self._species_map['HE']] = K_h_e * h_free * e_free
        concs[self._species_map['HE2']] = K_h_e * concs[self._species_map['HE']] * e_free
        concs[self._species_map['LE']] = K_l_e * concs[self._species_map['L']] * e_free
        concs[self._species_map['LE2']] = K_l_e * concs[self._species_map['LE']] * e_free
        concs[self._species_map['HO']] = K_h_o * h_free * o_free
        concs[self._species_map['LO']] = K_l_o * concs[self._species_map['L']] * o_free
        concs[self._species_map['HOE']] = K_h_e * concs[self._species_map['HO']] * e_free
        concs[self._species_map['HOE2']] = K_h_e * concs[self._species_map['HOE']] * e_free
        concs[self._species_map['LOE']] = K_l_e * concs[self._species_map['LO']] * e_free
        concs[self._species_map['LOE2']] = K_l_e * concs[self._species_map['LOE']] * e_free
        return concs