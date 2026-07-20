"""
Mutation / genotype layer on top of the four-state equilibrium core.

Mutations perturb the free energy (stability) of the four protein states
HD, H, L, LE independently, in units of kT (positive = destabilizing). The
wild-type is pinned to the gauge g_L = 0, so its state energies follow from the
three wild-type association constants:

    g_L  = 0
    g_H  = -ln K_conf
    g_HD = -ln K_conf - ln K_dna
    g_LE = -ln K_eff

A genotype's energies are the wild-type energies plus every present mutation's
per-state effect plus every present pair's per-state epistasis. The perturbed
energies are converted back to K_conf/K_dna/K_eff and handed to the core.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .core import fraction_bound

STATES = ("HD", "H", "L", "LE")


def _energies_from_ks(ln_K_conf, ln_K_dna, ln_K_eff):
    """Wild-type state energies (g_L gauge = 0) from log association constants."""
    return {"L": 0.0,
            "H": -ln_K_conf,
            "HD": -ln_K_conf - ln_K_dna,
            "LE": -ln_K_eff}


def _ks_from_energies(g):
    """Association constants (Kc, Kd, Ke) from a state-energy dict."""
    ln_K_conf = -(g["H"] - g["L"])
    ln_K_dna = -(g["HD"] - g["H"])
    ln_K_eff = -(g["LE"] - g["L"])
    return np.exp(ln_K_conf), np.exp(ln_K_dna), np.exp(ln_K_eff)


def parse_genotype(genotype):
    """Split a genotype string into its mutation names ('wt' -> [])."""
    if genotype in (None, "wt", ""):
        return []
    return genotype.split("/")


class MutationEffects:
    """
    Catalog of per-mutation state effects and per-pair in-state epistasis.

    All effects are ddG values in kT keyed by state name (HD, H, L, LE);
    omitted states default to 0.
    """

    def __init__(self):
        self._singles = {}   # name -> {state: ddG}
        self._pairs = {}     # frozenset({a, b}) -> {state: ddG}

    @staticmethod
    def _as_state_dict(kwargs):
        bad = set(kwargs) - set(STATES)
        if bad:
            raise ValueError(f"unknown state(s) {sorted(bad)}; valid: {STATES}")
        return {s: float(kwargs.get(s, 0.0)) for s in STATES}

    def add_mutation(self, name, **ddg):
        """Register a mutation's main effect, e.g. add_mutation('A', HD=2.0)."""
        self._singles[name] = self._as_state_dict(ddg)
        return self

    def add_epistasis(self, m1, m2, **ddg):
        """Register in-state epistasis for a pair, e.g. add_epistasis('A','B', HD=1)."""
        if m1 == m2:
            raise ValueError("epistasis requires two distinct mutations")
        self._pairs[frozenset((m1, m2))] = self._as_state_dict(ddg)
        return self

    def ddg_for(self, genotype):
        """Summed per-state ddG for a genotype string ('wt', 'A', 'A/B', ...)."""
        muts = parse_genotype(genotype)
        total = {s: 0.0 for s in STATES}
        for m in muts:
            if m not in self._singles:
                raise KeyError(f"mutation {m!r} not in catalog")
            for s in STATES:
                total[s] += self._singles[m][s]
        for a, b in combinations(muts, 2):
            pair = self._pairs.get(frozenset((a, b)))
            if pair:
                for s in STATES:
                    total[s] += pair[s]
        return total

    @property
    def mutations(self):
        return list(self._singles)


class ThermoModel:
    """
    Wild-type four-state system plus a mutation layer.

    Parameters
    ----------
    ln_K_conf, ln_K_dna, ln_K_eff : float
        Natural-log wild-type association constants.
    protein_total, dna_total : float
        Total protein and DNA (same concentration unit).
    """

    def __init__(self, ln_K_conf, ln_K_dna, ln_K_eff,
                 protein_total, dna_total):
        self.wt_energies = _energies_from_ks(ln_K_conf, ln_K_dna, ln_K_eff)
        self.protein_total = float(protein_total)
        self.dna_total = float(dna_total)

    def genotype_ks(self, genotype="wt", effects=None, ddg=None):
        """
        Association constants (K_conf, K_dna, K_eff) for a genotype.

        Mutations are supplied the same way as in ``observable``: either a
        direct ``ddg`` dict or a ``MutationEffects`` catalog plus a ``genotype``
        string. Useful for feeding ``core.solve_species`` to inspect the full
        occupancy breakdown.
        """
        g = dict(self.wt_energies)
        if ddg is not None and effects is not None:
            raise ValueError("pass either ddg= or effects=, not both")
        if ddg is not None:
            delta = MutationEffects._as_state_dict(ddg)
        elif effects is not None:
            delta = effects.ddg_for(genotype)
        else:
            delta = {s: 0.0 for s in STATES}
        for s in STATES:
            g[s] += delta[s]
        return _ks_from_energies(g)

    def observable(self, effector_total, genotype="wt", effects=None, ddg=None):
        """
        Observable HD/(HD+D) over effector concentration(s).

        Provide mutations either as a direct ``ddg={'HD': ..., ...}`` dict (one
        genotype, manual) or as a ``MutationEffects`` catalog plus a
        ``genotype`` string.
        """
        Kc, Kd, Ke = self.genotype_ks(genotype, effects, ddg)
        return fraction_bound(Kc, Kd, Ke, self.protein_total,
                              self.dna_total, effector_total)


def enumerate_genotypes(mutations, order=2):
    """['wt', singles..., pairs...] up to the requested combination order."""
    genos = ["wt"]
    for k in range(1, order + 1):
        genos += ["/".join(c) for c in combinations(mutations, k)]
    return genos


def build_titration_df(model, effector_conc, genotypes=None, effects=None,
                       titrant_name="effector", observable_std=None,
                       noise_sd=None, rng=None):
    """
    Long-form titration table for tfs-cat-response / tfs-extract-epistasis.

    Columns: genotype, titrant_name, titrant_conc, observable
    (+ observable_std if requested). ``noise_sd`` adds Gaussian scatter to the
    observable (drawn from ``rng``); ``observable_std`` is the *reported* error
    written to every row (independent of the noise actually applied).

    Notes
    -----
    ``cat_response`` accepts any genotype labels, but ``extract_epistasis``
    parses them through ``standardize_genotypes`` and requires the ``XsiteY``
    mutation convention (e.g. ``A1V``, ``A2V`` -> double ``A1V/A2V``). Name
    mutations accordingly if you plan to run the epistasis extractor.
    """
    if genotypes is None:
        genotypes = (["wt"] if effects is None
                     else enumerate_genotypes(effects.mutations, order=2))
    conc = np.asarray(effector_conc, dtype=float)
    if noise_sd is not None and rng is None:
        rng = np.random.default_rng()

    rows = []
    for geno in genotypes:
        y = np.atleast_1d(model.observable(conc, genotype=geno, effects=effects))
        if noise_sd:
            y = np.clip(y + rng.normal(0.0, noise_sd, size=y.shape), 0.0, 1.0)
        block = pd.DataFrame({"genotype": geno, "titrant_name": titrant_name,
                              "titrant_conc": conc, "observable": y})
        if observable_std is not None:
            block["observable_std"] = observable_std
        rows.append(block)
    return pd.concat(rows, ignore_index=True)
