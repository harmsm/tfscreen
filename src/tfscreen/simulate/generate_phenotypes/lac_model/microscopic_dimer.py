import numpy as np
from scipy.optimize import root
import warnings

class MicroscopicDimerModel:
    """
    A microscopic linked equilibria model of a lac repressor dimer.

    This class is initialized with the total concentrations of the components.
    Its primary method, `get_concs`, takes a set of 12 equilibrium constants
    and calculates the concentrations of all 15 species for each condition.
    The model includes an unfolded monomer (U) and two dimer conformations (L
    and H).

    Parameters
    ----------
    r_total : float or array-like
        Total concentration of repressor in **monomer** units.
    o_total : float or array-like
        Total concentration of operator (O).
    e_total : float or array-like
        Total concentration of effector (E).

    Attributes
    ----------
    species_names : tuple
        Names of the 15 molecular species in the model.
    e_stoich, o_stoich, r_stoich : np.ndarray
        Stoichiometry of each component (effector, operator, repressor)
        in each of the 15 species.
    repressor_reactant, repressor_product : np.ndarray
        Names of repressor species on the reactant and product sides of the
        12 equilibria, respectively.

    Notes
    -----
    **Species**
        - O: free operator
        - E: free effector
        - U: unfolded repressor monomer
        - L: folded, low DNA-affinity repressor dimer
        - H: folded, high DNA-affinity repressor dimer
        - LO: L dimer bound to operator
        - LE: L dimer bound to one effector
        - LE2: L dimer bound to two effectors
        - LOE: L dimer bound to operator and one effector
        - LOE2: L dimer bound to operator and two effectors
        - HO: H dimer bound to operator
        - HE: H dimer bound to one effector
        - HE2: H dimer bound to two effectors
        - HOE: H dimer bound to operator and one effector
        - HOE2: H dimer bound to operator and two effectors

    **Equilibrium Constants**
    The `get_concs` method requires a `K_array` with 12 constants:
        - K_l_2u: L -> 2U (M, dissociation)
        - K_l_h: L -> H (unitless, isomerization)
        - K_l_o: L + O -> LO (M-1, association)
        - K_h_o: H + O -> HO (M-1, association)
        - K_l_e: L + E -> LE (M-1, association)
        - K_le_e: LE + E -> LE2 (M-1, association)
        - K_h_e: H + E -> HE (M-1, association)
        - K_he_e: HE + E -> HE2 (M-1, association)
        - K_lo_e: LO + E -> LOE (M-1, association)
        - K_loe_e: LOE + E -> LOE2 (M-1, association)
        - K_ho_e: HO + E -> HOE (M-1, association)
        - K_hoe_e: HOE + E -> HOE2 (M-1, association)
    """

    species_names = (
        'E', 'O', 'U', 'L', 'H', 'LO', 'LE', 'LE2', 'LOE', 'LOE2',
        'HO', 'HE', 'HE2', 'HOE', 'HOE2'
    )
    r_stoich = np.array([0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    e_stoich = np.array([1, 0, 0, 0, 0, 0, 1, 2, 1, 2, 0, 1, 2, 1, 2])
    o_stoich = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    r_mask = r_stoich > 0
    e_mask = e_stoich > 0
    o_mask = o_stoich > 0
    repressor_reactant = np.array([
        'L', 'L', 'L', 'H', 'L', 'LE', 'H', 'HE',
        'LO', 'LOE', 'HO', 'HOE'
    ])
    repressor_product = np.array([
        'U', 'H', 'LO', 'HO', 'LE', 'LE2', 'HE', 'HE2',
        'LOE', 'LOE2', 'HOE', 'HOE2'
    ])

    equilibrium_constants = ["K_l_2u", "K_l_h", 
                             "K_l_o", "K_h_o",
                             "K_l_e", "K_le_e",
                             "K_h_e", "K_he_e",
                             "K_lo_e", "K_loe_e",
                             "K_ho_e", "K_hoe_e"]

    def __init__(self, r_total, o_total, e_total):
        try:
            b_e, b_o, b_r = np.broadcast_arrays(e_total, o_total, r_total)
        except ValueError as e:
            raise ValueError("Input concentration shapes cannot be broadcast together.") from e
        self.e_total = b_e.ravel()
        self.o_total = b_o.ravel()
        self.r_total = b_r.ravel()
        self.num_conditions = self.e_total.size
        self._species_map = {name: i for i, name in enumerate(self.species_names)}

    def get_concs(self, K_array):
        if not isinstance(K_array, np.ndarray) or K_array.shape != (12,):
            raise ValueError("K_array must be a 1D NumPy array of length 12.")

        results = np.zeros((self.num_conditions, len(self.species_names)))
        for i in range(self.num_conditions):
            solution = self._solve_single_condition(self.e_total[i], self.o_total[i], self.r_total[i], K_array)
            results[i, :] = solution
        return results

    def _solve_single_condition(self, e_total, o_total, r_total, K_array):
        def _equations_log(log_free_concs, *args):
            e_free, o_free, u_free = np.exp(log_free_concs)
            e_total, o_total, r_total, K_array = args
            (K_l_2u, K_l_h, K_l_o, K_h_o, K_l_e, K_le_e,
             K_h_e, K_he_e, K_lo_e, K_loe_e, K_ho_e, K_hoe_e) = K_array

            l = u_free**2 / K_l_2u if K_l_2u > 0 else (0.0 if u_free > 1e-30 else np.inf)
            h = K_l_h * l
            lo = K_l_o * l * o_free
            le = K_l_e * l * e_free
            le2 = K_le_e * le * e_free
            loe = K_lo_e * lo * e_free
            loe2 = K_loe_e * loe * e_free
            ho = K_h_o * h * o_free
            he = K_h_e * h * e_free
            he2 = K_he_e * he * e_free
            hoe = K_ho_e * ho * e_free
            hoe2 = K_hoe_e * hoe * e_free

            e_calc = (e_free + le + 2*le2 + loe + 2*loe2 + he + 2*he2 + hoe + 2*hoe2)
            o_calc = o_free + lo + loe + loe2 + ho + hoe + hoe2
            r_calc = (u_free + 2*l + 2*h + 2*lo + 2*le + 2*le2 + 2*loe + 2*loe2 + 2*ho + 2*he + 2*he2 + 2*hoe + 2*hoe2)
            
            return (e_total - e_calc, o_total - o_calc, r_total - r_calc)

        initial_concs = np.array([e_total, o_total, r_total])
        initial_guess_log = np.log(np.maximum(initial_concs, 1e-20))
        solution = root(_equations_log, initial_guess_log, args=(e_total, o_total, r_total, K_array), method='lm')

        if not solution.success:
            warnings.warn(f"Solver failed for condition E={e_total}, O={o_total}, R={r_total}: {solution.message}")
            return np.full(len(self.species_names), np.nan)

        e_free, o_free, u_free = np.exp(solution.x)
        (K_l_2u, K_l_h, K_l_o, K_h_o, K_l_e, K_le_e,
         K_h_e, K_he_e, K_lo_e, K_loe_e, K_ho_e, K_hoe_e) = K_array

        concs = np.zeros(len(self.species_names))
        concs[self._species_map['E']] = e_free
        concs[self._species_map['O']] = o_free
        concs[self._species_map['U']] = u_free
        concs[self._species_map['L']] = concs[self._species_map['U']]**2 / K_l_2u if K_l_2u > 0 else 0.0
        concs[self._species_map['H']] = K_l_h * concs[self._species_map['L']]
        concs[self._species_map['LO']] = K_l_o * concs[self._species_map['L']] * concs[self._species_map['O']]
        concs[self._species_map['LE']] = K_l_e * concs[self._species_map['L']] * concs[self._species_map['E']]
        concs[self._species_map['LE2']] = K_le_e * concs[self._species_map['LE']] * concs[self._species_map['E']]
        concs[self._species_map['LOE']] = K_lo_e * concs[self._species_map['LO']] * concs[self._species_map['E']]
        concs[self._species_map['LOE2']] = K_loe_e * concs[self._species_map['LOE']] * concs[self._species_map['E']]
        concs[self._species_map['HO']] = K_h_o * concs[self._species_map['H']] * concs[self._species_map['O']]
        concs[self._species_map['HE']] = K_h_e * concs[self._species_map['H']] * concs[self._species_map['E']]
        concs[self._species_map['HE2']] = K_he_e * concs[self._species_map['HE']] * concs[self._species_map['E']]
        concs[self._species_map['HOE']] = K_ho_e * concs[self._species_map['HO']] * concs[self._species_map['E']]
        concs[self._species_map['HOE2']] = K_hoe_e * concs[self._species_map['HOE']] * concs[self._species_map['E']]
        return concs

    def get_fx_operator(self, K_array):
        """Calculates the fraction of operator bound by repressor."""
        all_concs = self.get_concs(K_array)
        o_free_conc = all_concs[:, self._species_map['O']]
        o_free_conc = np.minimum(o_free_conc, self.o_total)
        return (self.o_total - o_free_conc) / (self.o_total + 1e-30)