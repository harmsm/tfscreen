import numpy as np
from scipy.optimize import root
import warnings

class LinkageDimerModel:
    """
    A general linkage model for a dimeric repressor.

    This model describes a dimeric repressor (R) that can bind up to two
    effector molecules (E) and one operator (O). The equilibria are defined
    by **dissociation** constants.

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
    **Species**
        - E, O, R: Free effector, operator, and repressor dimer
        - RE, RE2: Repressor bound to one or two effectors
        - RO: Repressor bound to operator
        - ROE: Repressor bound to one effector and operator
        - ROE2: Repressor bound to two effectors and operator

    **Equilibrium Constants**
    The `get_concs` method requires a `K_array` with 5 **dissociation** constants:
        - Kd_e1: RE -> R + E (M)
        - Kd_e2: RE2 -> RE + E (M)
        - Kd_o: RO -> R + O (M)
        - Kd_oe1: ROE -> RE + O (M)
        - Kd_oe2: ROE2 -> RE2 + O (M)

    References
    ----------
    O'Gorman, R. B., et al. ((1980). Thermodynamics of the lac
    repressor-operator interaction. Journal of Molecular Biology, 144(2),
    179-192.
    """

    species_names = ('E', 'O', 'R', 'RE', 'RE2', 'RO', 'ROE', 'ROE2')
    
    r_stoich = np.array([0, 0, 2, 2, 2, 2, 2, 2])
    e_stoich = np.array([1, 0, 0, 1, 2, 0, 1, 2])
    o_stoich = np.array([0, 1, 0, 0, 0, 1, 1, 1])

    repressor_reactant = np.array(['RE', 'RE2', 'RO', 'ROE', 'ROE2'])
    repressor_product = np.array(['R', 'RE', 'R', 'RE', 'RE2'])

    equilbrium_constants = ["Kd_e1","Kd_e2","Kd_o","Kd_oe1","Kd_oe2"]

    def __init__(self, r_total, o_total, e_total):
        # Convert input r_total (monomer units) to the dimer units used
        # by this model's internal solver.
        r_total_as_dimer = np.asarray(r_total) / 2.0

        try:
            # Note the new broadcast order: r, o, e
            b_r, b_o, b_e = np.broadcast_arrays(r_total_as_dimer, o_total, e_total)
        except ValueError as e:
            raise ValueError("Input concentration shapes cannot be broadcast together.") from e

        self.r_total_dimer = b_r.ravel() # Stored internally as dimer concentration
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
            A 1D NumPy array of 5 **dissociation** constants. See class
            docstring for the required order.

        Returns
        -------
        np.ndarray
            A (num_conditions x 8) array of species concentrations.
        """
        if not isinstance(K_array, np.ndarray) or K_array.shape != (5,):
            raise ValueError("K_array must be a 1D NumPy array of length 5.")
        results = np.zeros((self.num_conditions, len(self.species_names)))
        for i in range(self.num_conditions):
            # Pass concentrations in the order the solver expects: e, o, r
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
            e_free, o_free, r_free = np.exp(log_free_concs)
            e_total, o_total, r_total_dimer, K_array = args
            Kd_e1, Kd_e2, Kd_o, Kd_oe1, Kd_oe2 = K_array

            # Add a small epsilon to Kd values to avoid division by zero if a K is 0
            re = r_free * e_free / (Kd_e1 + 1e-30)
            re2 = re * e_free / (Kd_e2 + 1e-30)
            ro = r_free * o_free / (Kd_o + 1e-30)
            roe = re * o_free / (Kd_oe1 + 1e-30)
            roe2 = re2 * o_free / (Kd_oe2 + 1e-30)

            e_calc = e_free + re + 2*re2 + roe + 2*roe2
            o_calc = o_free + ro + roe + roe2
            r_calc = r_free + re + re2 + ro + roe + roe2

            return (e_total-e_calc, o_total-o_calc, r_total_dimer-r_calc)

        initial_guess = np.log(np.maximum(1e-20, [e_total, o_total, r_total_dimer]))
        solution = root(_equations_log, initial_guess, args=(e_total, o_total, r_total_dimer, K_array), method='lm')
        if not solution.success:
            warnings.warn(f"Solver failed for condition E={e_total}, O={o_total}, R={r_total_dimer*2}: {solution.message}")
            return np.full(len(self.species_names), np.nan)

        e_free, o_free, r_free = np.exp(solution.x)
        Kd_e1, Kd_e2, Kd_o, Kd_oe1, Kd_oe2 = K_array
        concs = np.zeros(len(self.species_names))
        concs[self._species_map['E']], concs[self._species_map['O']], concs[self._species_map['R']] = e_free, o_free, r_free
        concs[self._species_map['RE']] = r_free * e_free / (Kd_e1 + 1e-30)
        concs[self._species_map['RE2']] = concs[self._species_map['RE']] * e_free / (Kd_e2 + 1e-30)
        concs[self._species_map['RO']] = r_free * o_free / (Kd_o + 1e-30)
        concs[self._species_map['ROE']] = concs[self._species_map['RE']] * o_free / (Kd_oe1 + 1e-30)
        concs[self._species_map['ROE2']] = concs[self._species_map['RE2']] * o_free / (Kd_oe2 + 1e-30)
        return concs