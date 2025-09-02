import numpy as np
from scipy.optimize import root
import warnings

class LinkageDimerTetramerModel:
    """
    A lac repressor model with explicit dimer-tetramer repressor equilibrium.

    This model includes repressor dimers (R2) which can associate into
    tetramers (R4). Both forms can bind effector and operator.

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
        - E, O: Free effector and operator
        - R2: Repressor dimer
        - R4: Repressor tetramer
        - R2E, R2E2: Dimer bound to one or two effectors
        - R4E, R4E2, R4E3, R4E4: Tetramer bound to 1-4 effectors
        - R2O, R4O: Operator bound to a dimer or tetramer
        - R2OE, R2OE2: R2O complex bound to one or two effectors
        - R4OE: R4O complex bound to one effector

    **Equilibrium Constants**
    The `get_concs` method requires a `K_array` with 12 **association** constants:
        - K_r2_e: R2 + E <=> R2E (M-1)
        - K_r2e_e: R2E + E <=> R2E2 (M-1)
        - K_2r2_r4: 2R2 <=> R4 (M-1)
        - K_r4_e: R4 + E <=> R4E (M-1)
        - K_r4e_e: R4E + E <=> R4E2 (M-1)
        - K_r4e2_e: R4E2 + E <=> R4E3 (M-1)
        - K_r4e3_e: R4E3 + E <=> R4E4 (M-1)
        - K_r2_o: R2 + O <=> R2O (M-1)
        - K_r4_o: R4 + O <=> R4O (M-1)
        - K_r2o_e: R2O + E <=> R2OE (M-1)
        - K_r2oe_e: R2OE + E <=> R2OE2 (M-1)
        - K_r4o_e: R4O + E <=> R4OE (M-1)
    """

    species_names = (
        'E', 'O', 'R2', 'R2E', 'R2E2', 'R2O', 'R2OE', 'R2OE2',
        'R4', 'R4E', 'R4E2', 'R4E3', 'R4E4', 'R4O', 'R4OE'
    )
    r_stoich = np.array([0, 0, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4])
    e_stoich = np.array([1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1])
    o_stoich = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])

    repressor_reactant = np.array([
        'R2', 'R2E', 'R2', 'R4', 'R4E', 'R4E2', 'R4E3',
        'R2', 'R4', 'R2O', 'R2OE', 'R4O'
    ])
    repressor_product = np.array([
        'R2E', 'R2E2', 'R4', 'R4E', 'R4E2', 'R4E3', 'R4E4',
        'R2O', 'R4O', 'R2OE', 'R2OE2', 'R4OE'
    ])
    
    equilibrium_constants = (
        'K_r2_e', 'K_r2e_e', 'K_2r2_r4', 'K_r4_e', 'K_r4e_e', 'K_r4e2_e',
        'K_r4e3_e', 'K_r2_o', 'K_r4_o', 'K_r2o_e', 'K_r2oe_e', 'K_r4o_e'
    )

    def __init__(self, r_total, o_total, e_total):
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
        Calculate equilibrium concentrations for all species.

        Parameters
        ----------
        K_array : np.ndarray
            A 1D NumPy array of 12 **association** constants. See class
            docstring for the required order.

        Returns
        -------
        np.ndarray
            A (num_conditions x 15) array of species concentrations.
        """
        if not isinstance(K_array, np.ndarray) or K_array.shape != (12,):
            raise ValueError("K_array must be a 1D NumPy array of length 12.")
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
                e_free, o_free, r2_free = np.exp(log_free_concs)
                e_total, o_total, r_total_dimer, K_array = args
                (K_r2_e, K_r2e_e, K_2r2_r4, K_r4_e, K_r4e_e, K_r4e2_e, K_r4e3_e,
                K_r2_o, K_r4_o, K_r2o_e, K_r2oe_e, K_r4o_e) = K_array

                r2e = K_r2_e * r2_free * e_free
                r2e2 = K_r2e_e * r2e * e_free
                r4 = K_2r2_r4 * r2_free**2
                r4e = K_r4_e * r4 * e_free
                r4e2 = K_r4e_e * r4e * e_free
                r4e3 = K_r4e2_e * r4e2 * e_free
                r4e4 = K_r4e3_e * r4e3 * e_free
                r2o = K_r2_o * o_free * r2_free
                r4o = K_r4_o * o_free * r4
                r2oe = K_r2o_e * r2o * e_free
                r2oe2 = K_r2oe_e * r2oe * e_free
                r4oe = K_r4o_e * r4o * e_free

                e_calc = e_free + r2e + 2*r2e2 + r4e + 2*r4e2 + 3*r4e3 + 4*r4e4 + r2oe + 2*r2oe2 + r4oe
                o_calc = o_free + r2o + r4o + r2oe + r2oe2 + r4oe
                r_calc = r2_free + r2e + r2e2 + 2*(r4 + r4e + r4e2 + r4e3 + r4e4) + r2o + r2oe + r2oe2 + 2*(r4o + r4oe)

                return (e_total-e_calc, o_total-o_calc, r_total_dimer-r_calc)

        initial_guess = np.log(np.maximum(1e-20, [e_total, o_total, r_total_dimer]))
        solution = root(_equations_log, initial_guess, args=(e_total, o_total, r_total_dimer, K_array), method='lm')
        if not solution.success:
            warnings.warn(f"Solver failed for condition E={e_total}, O={o_total}, R={r_total_dimer*2}: {solution.message}")
            return np.full(len(self.species_names), np.nan)

        e_free, o_free, r2_free = np.exp(solution.x)
        (K_r2_e, K_r2e_e, K_2r2_r4, K_r4_e, K_r4e_e, K_r4e2_e, K_r4e3_e,
         K_r2_o, K_r4_o, K_r2o_e, K_r2oe_e, K_r4o_e) = K_array
        
        concs = np.zeros(len(self.species_names))
        concs[self._species_map['E']], concs[self._species_map['O']], concs[self._species_map['R2']] = e_free, o_free, r2_free
        concs[self._species_map['R2E']] = K_r2_e * r2_free * e_free
        concs[self._species_map['R2E2']] = K_r2e_e * concs[self._species_map['R2E']] * e_free
        concs[self._species_map['R4']] = K_2r2_r4 * r2_free**2
        concs[self._species_map['R4E']] = K_r4_e * concs[self._species_map['R4']] * e_free
        concs[self._species_map['R4E2']] = K_r4e_e * concs[self._species_map['R4E']] * e_free
        concs[self._species_map['R4E3']] = K_r4e2_e * concs[self._species_map['R4E2']] * e_free
        concs[self._species_map['R4E4']] = K_r4e3_e * concs[self._species_map['R4E3']] * e_free
        concs[self._species_map['R2O']] = K_r2_o * o_free * r2_free
        concs[self._species_map['R4O']] = K_r4_o * o_free * concs[self._species_map['R4']]
        concs[self._species_map['R2OE']] = K_r2o_e * concs[self._species_map['R2O']] * e_free
        concs[self._species_map['R2OE2']] = K_r2oe_e * concs[self._species_map['R2OE']] * e_free
        concs[self._species_map['R4OE']] = K_r4o_e * concs[self._species_map['R4O']] * e_free
        return concs