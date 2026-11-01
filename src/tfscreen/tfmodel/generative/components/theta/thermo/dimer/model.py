"""
Shared parts of the thermo.dimer theta components (PK, PddG, PnnC).

The physics is ``species.py``. This module turns a component's priors into
per-genotype terms and back: the variant flags, the priors class, the wt,
per-mutation and per-pair sites (one code path for model and guide), the
assembly, ``run_model`` and the extraction helpers.

Variant flags
-------------
Six static (``pytree_node=False``) bool fields of every ModelPriors pick
the species and terms (``species.Variant``): ``theta_monomers``,
``theta_unfolded_dimer`` (default on), ``theta_unfolded_monomer``,
``theta_second_engagement`` (default on), ``theta_effector_coupling`` and
``theta_operator_coupling``. Set them with ``tfs-configure-model
--set_priors theta_monomers=1``, in the priors CSV, or in a simulate
config's ``theta_priors``.

Sites (``name`` is the component's prefix, "theta")
-----------------------------------------------------
wt, one per active term, ``{name}_{term}_wt`` ~ Normal(loc, scale); per
titrant inside ``{name}_wt_plate`` for the effector terms. The operator
terms are put on the wt *effective* constant, ``{name}_ln_K_h_o_eff_wt``
= ln[K_o (1 + K_oii)], which is what the data see; the first engagement
``{name}_ln_K_h_o_wt`` (deterministic) is that minus softplus(ln K_oii).
So the prior on the wt operator does not depend on whether the second
engagement is in the model.

Per mutation, a shared shift on every active term. PK (and the
non-structural terms of PddG/PnnC): ``{name}_sigma_d`` (K,) ~
HalfNormal(``theta_sigma_d_{term}_scale``) and ``{name}_d_offset``
(K, M) ~ Normal(0, 1), shift = offset · σ. PddG/PnnC put the structural
prior (``thermo/prior.py``) on the six C12 constants (ln_K_h_l, ln_K_h_o,
ln_K_h_e, ln_K_l_o, ln_K_l_e, ln_K_u). With more than one titrant, each
mutation's effect on an effector term is its shared shift plus a
deviation per titrant, ``{name}_d_dev`` (K_t, T, M) ~
Normal(0, ``theta_d_dev_scale``): the study's per-effector remedy (step
5), which ties the effectors together instead of fitting each freely.

Per pair (with epistasis), one shift on every active term, shared across
titrants: a regularized horseshoe, ``{name}_epi_tau``, ``{name}_epi_c2``,
``{name}_epi_lambda`` and ``{name}_epi_offset`` (K, P), with the local
scale's prior HalfCauchy(exp(−d/d0)), d the pair's closest contact over
the structures when the data carry contact distances (PnnC), else 1.

Deterministic sites, the names extraction and the simulator's truth
tables use: ``{name}_d_{term}`` ((M,), or (T, M) for effector terms),
``{name}_epi_{term}`` (P,), ``{name}_{term}`` per genotype ((G,) or
(T, G)) and ``{name}_ln_K_h_o_eff``/``{name}_ln_K_l_o_eff`` per genotype.
"""

import math
from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from flax.struct import dataclass, field

from tfscreen.genetics.build_mut_geno_matrix import (
    apply_mut_matrix,
    apply_pair_matrix,
)
from tfscreen.tfmodel.generative.components._horseshoe import (
    HalfCauchy,
    regularized_scale,
)
from tfscreen.tfmodel.generative.components.theta.thermo import mwc
from tfscreen.tfmodel.generative.components.theta.thermo.dimer import species as sp

DEFAULT_TF_TOTAL_M = mwc.DEFAULT_TF_TOTAL_M
DEFAULT_CONC_UNIT_SCALE = mwc.DEFAULT_CONC_UNIT_SCALE
_ZERO_CONC_VALUE = 0.0

# Effective operator constants are what the wt prior sits on.
_OPERATOR_TERMS = {"ln_K_h_o": "ln_K_h_oii", "ln_K_l_o": "ln_K_l_oii"}

# wt prior (loc, scale) per wt site, natural-log units. With every
# optional term off these are the O2_C12 U1 PK defaults (thermo/mwc.py).
# The monomer and Oii values follow the study's lac-like truth
# (planning/studies/thermo-identifiability, step 3): a tight dimer
# (1e10 M⁻¹), monomer half-site binding 1e6 (H) and 1e3 (L) M⁻¹, ln K_oii 4.
WT_PRIORS = {
    "ln_K_h_l": (mwc.DEFAULT_LN_K_H_L, 2.0),
    "ln_K_h_o_eff": (mwc.DEFAULT_LN_K_H_O, 2.0),
    "ln_K_l_o_eff": (mwc.DEFAULT_LN_K_L_O, 3.0),
    "ln_K_u": (-12.0, 3.0),
    "ln_K_h_oii": (4.0, 3.0),
    "ln_K_l_oii": (4.0, 3.0),
    "ln_K_d_h": (math.log(1e10), 3.0),
    "ln_K_m_h_l": (0.5 * mwc.DEFAULT_LN_K_H_L, 2.0),
    "ln_K_m_h_o": (math.log(1e6), 3.0),
    "ln_K_m_l_o": (math.log(1e3), 3.0),
    "ln_K_m_u": (-12.0, 3.0),
    "ln_K_h_e": (mwc.DEFAULT_LN_K_H_E, 2.0),
    "ln_K_l_e": (mwc.DEFAULT_LN_K_L_E, 2.0),
    "ln_K_h_ee": (0.0, 1.0),
    "ln_K_l_ee": (0.0, 1.0),
    "ln_K_h_eo": (0.0, 1.0),
    "ln_K_l_eo": (0.0, 1.0),
}

FLAG_FIELDS = {
    "theta_monomers": "monomers",
    "theta_unfolded_dimer": "unfolded_dimer",
    "theta_unfolded_monomer": "unfolded_monomer",
    "theta_second_engagement": "second_engagement",
    "theta_effector_coupling": "effector_coupling",
    "theta_operator_coupling": "operator_coupling",
}

# The six constants of the O2_C12 structural prior, in the row order of
# prior.projection(mwc.PROJ_C12, mwc.PROJ_U_C12).
STRUCT_TERMS = ("ln_K_h_l", "ln_K_h_o", "ln_K_h_e", "ln_K_l_o", "ln_K_l_e",
                "ln_K_u")


def wt_site(term):
    """The wt sample-site suffix of a term (effective for the operator)."""
    return f"{term}_eff" if term in _OPERATOR_TERMS else term


def variant_from_priors(priors):
    """The :class:`species.Variant` a ModelPriors (or None) selects."""
    defaults = sp.Variant()
    kwargs = {}
    for f, attr in FLAG_FIELDS.items():
        val = getattr(priors, f, getattr(defaults, attr)) if priors is not None \
            else getattr(defaults, attr)
        kwargs[attr] = bool(val)
    return sp.Variant(**kwargs)


# ---------------------------------------------------------------------------
# Priors class
# ---------------------------------------------------------------------------

def common_hyperparameters():
    """Hyperparameters every thermo.dimer variant has."""
    p = {}
    for site, (loc, scale) in WT_PRIORS.items():
        p[f"theta_{site}_wt_loc"] = float(loc)
        p[f"theta_{site}_wt_scale"] = float(scale)
    p["theta_tf_total_M"] = DEFAULT_TF_TOTAL_M     # M, protomer units
    p["theta_conc_unit_scale"] = DEFAULT_CONC_UNIT_SCALE   # mM -> M
    p["theta_d_dev_scale"] = 0.5     # per-titrant deviation of effector effects
    p["theta_epi_tau_scale"] = 0.1
    p["theta_epi_slab_scale"] = 2.0
    p["theta_epi_slab_df"] = 4.0
    p["theta_epi_d0"] = 8.0          # Å
    return p


def sigma_hyperparameters(terms):
    """``theta_sigma_d_{term}_scale`` (HalfNormal scale of σ) per term."""
    return {f"theta_sigma_d_{t}_scale": 1.0 for t in terms}


def flag_defaults():
    """The variant flags and their defaults."""
    v = sp.Variant()
    return {f: bool(getattr(v, attr)) for f, attr in FLAG_FIELDS.items()}


def make_priors_class(module_name, float_fields, static_fields):
    """
    A flax ModelPriors dataclass with the given fields.

    ``float_fields`` are pytree leaves; ``static_fields`` maps name to a
    default and becomes ``pytree_node=False`` fields (hashable under jit).
    The class is named ``ModelPriors`` in ``module_name`` so it pickles.
    """
    ann = {}
    ns = {"__module__": module_name, "__qualname__": "ModelPriors",
          "__doc__": "Hyperparameters of a thermo.dimer theta component "
                     "(thermo/dimer/model.py)."}
    for f in float_fields:
        ann[f] = float
    for f, default in static_fields.items():
        ann[f] = type(default)
        ns[f] = field(pytree_node=False, default=default)
    ns["__annotations__"] = ann
    cls = type("ModelPriors", (), ns)
    return dataclass(frozen=True)(cls)


# ---------------------------------------------------------------------------
# ThetaParam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThetaParam:
    """
    Per-genotype terms of a thermo.dimer model.

    One field per term (None when the variant lacks it): (G,) for the
    genotype terms, (T, G) for the effector terms.
    """

    ln_K_h_l: jnp.ndarray = None
    ln_K_h_o: jnp.ndarray = None
    ln_K_l_o: jnp.ndarray = None
    ln_K_u: jnp.ndarray = None
    ln_K_h_oii: jnp.ndarray = None
    ln_K_l_oii: jnp.ndarray = None
    ln_K_d_h: jnp.ndarray = None
    ln_K_m_h_l: jnp.ndarray = None
    ln_K_m_h_o: jnp.ndarray = None
    ln_K_m_l_o: jnp.ndarray = None
    ln_K_m_u: jnp.ndarray = None
    ln_K_h_e: jnp.ndarray = None
    ln_K_l_e: jnp.ndarray = None
    ln_K_h_ee: jnp.ndarray = None
    ln_K_l_ee: jnp.ndarray = None
    ln_K_h_eo: jnp.ndarray = None
    ln_K_l_eo: jnp.ndarray = None
    # Effective operator constants K_o (1 + K_oii), (G,); for the truth
    # tables and parameters_df, not used by run_model.
    ln_K_h_o_eff: jnp.ndarray = None
    ln_K_l_o_eff: jnp.ndarray = None
    tf_total: float = DEFAULT_TF_TOTAL_M    # M, protomer units
    conc_unit_scale: float = field(pytree_node=False,
                                   default=DEFAULT_CONC_UNIT_SCALE)
    variant: sp.Variant = field(pytree_node=False, default=sp.Variant())

    def terms(self):
        """Term name → array for the variant's terms."""
        return {t: getattr(self, t) for t in sp.active_terms(self.variant)}


def _broadcast_terms(terms, variant, genotype_idx=None):
    """
    Terms shaped for a (T, C, G) evaluation: genotype terms (1, 1, G),
    effector terms (T, 1, G); optionally indexed to ``genotype_idx``.
    """
    out = {}
    tt = set(sp.titrant_terms(variant))
    for t, v in terms.items():
        if genotype_idx is not None:
            v = v[..., genotype_idx]
        out[t] = v[:, None, :] if t in tt else v[None, None, :]
    return out


def compute_theta(terms, titrant_conc, tf_total, variant, genotype_idx=None):
    """θ (T, C, G') from per-genotype terms; ``titrant_conc`` (C,) in M."""
    b = _broadcast_terms(terms, variant, genotype_idx)
    return sp.theta(b, jnp.asarray(titrant_conc)[None, :, None], tf_total,
                    variant)


def run_model(theta_param, data):
    """
    θ at the concentrations and genotypes of ``data``.

    Returns
    -------
    (T, C, num_obs), or (1, 1, 1, 1, T, C, num_obs) when
    ``data.scatter_theta`` is 1.
    """
    geno_idx = data.batch_idx[data.geno_theta_idx]
    theta_calc = compute_theta(theta_param.terms(),
                               data.titrant_conc * theta_param.conc_unit_scale,
                               theta_param.tf_total, theta_param.variant,
                               genotype_idx=geno_idx)
    if data.scatter_theta == 1:
        theta_calc = theta_calc[None, None, None, None, :, :, :]
    return theta_calc


def get_population_moments(theta_param, data):
    """
    Not computed: nothing consumes population moments of θ since the
    ``logit_norm`` transformation was removed, and computing them costs a
    full-library θ evaluation per step.
    """
    return None, None


# ---------------------------------------------------------------------------
# Sampling helpers (model and guide share these)
# ---------------------------------------------------------------------------

def _normal(site, loc, scale, guide, shape, plural, loc_init=None):
    """A Normal site, or in the guide its mean-field Normal."""
    if not guide:
        return pyro.sample(site, dist.Normal(loc, scale))
    s = "s" if plural else ""
    init = loc if loc_init is None else loc_init
    q_loc = pyro.param(f"{site}_loc{s}", jnp.broadcast_to(
        jnp.asarray(init, dtype=jnp.float32), shape))
    q_scale = pyro.param(f"{site}_scale{s}", jnp.ones(shape),
                         constraint=constraints.positive)
    return q_loc, q_scale


def _sample_wt(name, priors, variant, T, guide):
    """wt terms: dict term → () or (T,); operator terms first engagement."""
    wt = {}
    gterms = sp.genotype_terms(variant)
    tterms = sp.titrant_terms(variant)
    for t in gterms:
        site = f"{name}_{wt_site(t)}_wt"
        loc = getattr(priors, f"theta_{wt_site(t)}_wt_loc")
        scale = getattr(priors, f"theta_{wt_site(t)}_wt_scale")
        if guide:
            q_loc, q_scale = _normal(site, loc, scale, True, (), False)
            wt[t] = pyro.sample(site, dist.Normal(q_loc, q_scale))
        else:
            wt[t] = _normal(site, loc, scale, False, (), False)
    q = {}
    if guide:
        for t in tterms:
            site = f"{name}_{t}_wt"
            q[t] = _normal(site, getattr(priors, f"theta_{t}_wt_loc"), None,
                           True, (T,), True)
    with pyro.plate(f"{name}_wt_plate", T, dim=-1):
        for t in tterms:
            site = f"{name}_{t}_wt"
            if guide:
                wt[t] = pyro.sample(site, dist.Normal(*q[t]))
            else:
                wt[t] = pyro.sample(site, dist.Normal(
                    getattr(priors, f"theta_{t}_wt_loc"),
                    getattr(priors, f"theta_{t}_wt_scale")))
    # The wt sites of the operator terms are effective constants.
    for t, oii in _OPERATOR_TERMS.items():
        if variant.second_engagement:
            wt[t] = wt[t] - jnp.logaddexp(0.0, wt[oii])
        if not guide:
            pyro.deterministic(f"{name}_{t}_wt", wt[t])
    return wt


def _sample_free_shifts(name, priors, terms, num_mut, guide):
    """Hierarchical per-mutation shifts for ``terms``: dict term → (M,)."""
    K = len(terms)
    if K == 0:
        return {}
    if num_mut == 0:
        return {t: jnp.zeros((0,)) for t in terms}
    scales = jnp.array([getattr(priors, f"theta_sigma_d_{t}_scale")
                        for t in terms])
    if guide:
        s_loc = pyro.param(f"{name}_sigma_d_locs", jnp.full(K, -1.0))
        s_scale = pyro.param(f"{name}_sigma_d_scales", jnp.full(K, 0.1),
                             constraint=constraints.positive)
        o_loc = pyro.param(f"{name}_d_offset_locs", jnp.zeros((K, num_mut)))
        o_scale = pyro.param(f"{name}_d_offset_scales", jnp.ones((K, num_mut)),
                             constraint=constraints.positive)
    with pyro.plate(f"{name}_sigma_d_plate", K, dim=-1):
        sigma = pyro.sample(f"{name}_sigma_d",
                            dist.LogNormal(s_loc, s_scale) if guide
                            else dist.HalfNormal(scales))
    with pyro.plate(f"{name}_d_term_plate", K, dim=-2):
        with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
            offset = pyro.sample(f"{name}_d_offset",
                                 dist.Normal(o_loc, o_scale) if guide
                                 else dist.Normal(0.0, 1.0))
    d = offset * sigma[:, None]
    return {t: d[i] for i, t in enumerate(terms)}


def _sample_deviations(name, priors, terms, T, num_mut, guide):
    """Per-titrant deviations of effector effects: dict term → (T, M)."""
    K = len(terms)
    if K == 0 or T < 2 or num_mut == 0:
        return {t: jnp.zeros((T, num_mut)) for t in terms}
    dev_scale = priors.theta_d_dev_scale
    if guide:
        q_loc = pyro.param(f"{name}_d_dev_locs", jnp.zeros((K, T, num_mut)))
        q_scale = pyro.param(f"{name}_d_dev_scales",
                             jnp.full((K, T, num_mut), dev_scale),
                             constraint=constraints.positive)
    with pyro.plate(f"{name}_dev_term_plate", K, dim=-3):
        with pyro.plate(f"{name}_dev_titrant_plate", T, dim=-2):
            with pyro.plate(f"{name}_mutation_plate", num_mut, dim=-1):
                dev = pyro.sample(f"{name}_d_dev",
                                  dist.Normal(q_loc, q_scale) if guide
                                  else dist.Normal(0.0, dev_scale))
    return {t: dev[i] for i, t in enumerate(terms)}


def _sample_pair_shifts(name, priors, terms, num_pair, distances, guide):
    """Horseshoe pair shifts on every term: dict term → (P,)."""
    K = len(terms)
    if guide:
        tau_loc = pyro.param(f"{name}_epi_tau_loc", jnp.array(-2.0))
        tau_scale = pyro.param(f"{name}_epi_tau_scale", jnp.array(0.5),
                               constraint=constraints.positive)
        c2_loc = pyro.param(f"{name}_epi_c2_loc", jnp.array(1.4))
        c2_scale = pyro.param(f"{name}_epi_c2_scale", jnp.array(0.5),
                              constraint=constraints.positive)
        lam_loc = pyro.param(f"{name}_epi_lambda_locs",
                             jnp.zeros((K, num_pair)))
        lam_scale = pyro.param(f"{name}_epi_lambda_scales",
                               jnp.ones((K, num_pair)),
                               constraint=constraints.positive)
        off_loc = pyro.param(f"{name}_epi_offset_locs",
                             jnp.zeros((K, num_pair)))
        off_scale = pyro.param(f"{name}_epi_offset_scales",
                               jnp.ones((K, num_pair)),
                               constraint=constraints.positive)
        tau = pyro.sample(f"{name}_epi_tau", dist.LogNormal(tau_loc, tau_scale))
        c2 = pyro.sample(f"{name}_epi_c2", dist.LogNormal(c2_loc, c2_scale))
    else:
        tau = pyro.sample(f"{name}_epi_tau",
                          HalfCauchy(priors.theta_epi_tau_scale))
        df = priors.theta_epi_slab_df
        c2 = pyro.sample(f"{name}_epi_c2", dist.InverseGamma(
            df / 2.0, df * priors.theta_epi_slab_scale ** 2 / 2.0))
        if distances is None:
            prior_lam = jnp.ones((num_pair,))
        else:
            prior_lam = jnp.maximum(
                jnp.exp(-jnp.asarray(distances) / priors.theta_epi_d0), 1e-12)
    with pyro.plate(f"{name}_epi_term_plate", K, dim=-2):
        with pyro.plate(f"{name}_pair_plate", num_pair, dim=-1):
            lam = pyro.sample(f"{name}_epi_lambda",
                              dist.LogNormal(lam_loc, lam_scale) if guide
                              else HalfCauchy(prior_lam[None, :]))
            offset = pyro.sample(f"{name}_epi_offset",
                                 dist.Normal(off_loc, off_scale) if guide
                                 else dist.Normal(0.0, 1.0))
    epi = offset * tau * regularized_scale(lam, tau, c2)   # (K, P)
    return {t: epi[i] for i, t in enumerate(terms)}


def define(name, data, priors, guide=False, structural=None,
           pair_distances=None):
    """
    The model (``guide=False``) or mean-field guide of a thermo.dimer model.

    Parameters
    ----------
    name : str
        Site prefix.
    data : GrowthData or BindingData
        Needs the mutation (and, with pairs, pair) scatter indices.
    priors : ModelPriors
    guide : bool
    structural : callable or None
        ``structural(guide) -> dict`` of per-mutation shared shifts (M,) for
        some terms (PddG/PnnC's :data:`STRUCT_TERMS`); those terms get no
        hierarchical prior here. None (PK): every term is hierarchical.
    pair_distances : array (P,) or None
        Closest contact distance of each pair (Å), for the horseshoe scale.

    Returns
    -------
    ThetaParam
    """
    variant = variant_from_priors(priors)
    T = data.num_titrant_name
    M = data.num_mutation
    has_pairs = data.num_pair > 0
    gterms = sp.genotype_terms(variant)
    tterms = sp.titrant_terms(variant)
    terms = gterms + tterms

    mut_scatter = partial(apply_mut_matrix,
                          mut_nnz_mut_idx=jnp.array(data.mut_nnz_mut_idx),
                          mut_nnz_geno_idx=jnp.array(data.mut_nnz_geno_idx),
                          num_genotype=data.num_genotype)

    wt = _sample_wt(name, priors, variant, T, guide)

    shared = dict(structural(guide)) if structural is not None else {}
    free = [t for t in terms if t not in shared]
    shared.update(_sample_free_shifts(name, priors, free, M, guide))
    dev = _sample_deviations(name, priors, tterms, T, M, guide)

    d = {t: shared[t] for t in gterms}
    for t in tterms:
        d[t] = shared[t][None, :] + dev[t]                    # (T, M)

    epi = {}
    if has_pairs:
        epi = _sample_pair_shifts(name, priors, terms, data.num_pair,
                                  pair_distances, guide)
        pair_scatter = partial(
            apply_pair_matrix,
            pair_nnz_pair_idx=jnp.array(data.pair_nnz_pair_idx),
            pair_nnz_geno_idx=jnp.array(data.pair_nnz_geno_idx),
            num_genotype=data.num_genotype)

    values = {}
    for t in terms:
        if t in tterms:
            v = wt[t][:, None] + mut_scatter(d[t])            # (T, G)
            if has_pairs:
                v = v + pair_scatter(epi[t])[None, :]
        else:
            v = wt[t] + mut_scatter(d[t])                     # (G,)
            if has_pairs:
                v = v + pair_scatter(epi[t])
        values[t] = v

    eff = {}
    for t, oii in _OPERATOR_TERMS.items():
        eff[f"{t}_eff"] = values[t]
        if variant.second_engagement:
            eff[f"{t}_eff"] = values[t] + jnp.logaddexp(0.0, values[oii])

    if not guide:
        for t in terms:
            pyro.deterministic(f"{name}_d_{t}", d[t])
            if has_pairs:
                pyro.deterministic(f"{name}_epi_{t}", epi[t])
            pyro.deterministic(f"{name}_{t}", values[t])
        for k, v in eff.items():
            pyro.deterministic(f"{name}_{k}", v)

    return ThetaParam(**values, **eff, tf_total=priors.theta_tf_total_M,
                      conc_unit_scale=priors.theta_conc_unit_scale,
                      variant=variant)


def wt_guesses(name, data):
    """
    Starting values for the wt sites every variant has.

    Guesses are made before the priors CSV is read (so before the variant
    is known), and a guess for a site the model lacks would not be used;
    the variant-specific sites start at their prior medians instead.
    """
    T = data.num_titrant_name
    g = {}
    for t in ("ln_K_h_l", "ln_K_h_o", "ln_K_l_o"):
        g[f"{name}_{wt_site(t)}_wt"] = jnp.array(WT_PRIORS[wt_site(t)][0])
    for t in ("ln_K_h_e", "ln_K_l_e"):
        g[f"{name}_{t}_wt"] = jnp.full(T, WT_PRIORS[t][0])
    return g


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def get_extract_specs(ctx, extra=()):
    """``tfs-extract-params`` tables for a thermo.dimer model."""
    variant = variant_from_priors(getattr(ctx, "theta_priors", None))
    gterms = list(sp.genotype_terms(variant))
    tterms = list(sp.titrant_terms(variant))
    tm = ctx.growth_tm
    num_genotype = len(tm.tensor_dim_labels[tm.tensor_dim_names.index("genotype")])
    titrant_names = list(tm.tensor_dim_labels[tm.tensor_dim_names.index("titrant_name")])
    mut_labels = list(ctx.mut_labels)
    num_mut = len(mut_labels)

    geno_df = tm.df[["genotype", "genotype_idx"]].drop_duplicates().copy()
    geno_df["map_geno"] = geno_df["genotype_idx"]
    specs = [dict(input_df=geno_df,
                  params_to_get=gterms + ["ln_K_h_o_eff", "ln_K_l_o_eff"],
                  map_column="map_geno", get_columns=["genotype"],
                  in_run_prefix="theta_")]

    te_df = (tm.df[["genotype", "titrant_name", "genotype_idx",
                    "titrant_name_idx"]].drop_duplicates().copy())
    te_df["map_theta_KE"] = te_df["titrant_name_idx"] * num_genotype \
        + te_df["genotype_idx"]
    specs.append(dict(input_df=te_df, params_to_get=tterms,
                      map_column="map_theta_KE",
                      get_columns=["genotype", "titrant_name"],
                      in_run_prefix="theta_"))

    if num_mut > 0:
        mut_df = pd.DataFrame({"mutation": mut_labels, "map_mut": range(num_mut)})
        specs.append(dict(input_df=mut_df,
                          params_to_get=[f"d_{t}" for t in gterms],
                          map_column="map_mut", get_columns=["mutation"],
                          in_run_prefix="theta_"))
        rows = [{"titrant_name": tn, "mutation": m,
                 "map_theta_d_KE": ti * num_mut + mi}
                for ti, tn in enumerate(titrant_names)
                for mi, m in enumerate(mut_labels)]
        specs.append(dict(input_df=pd.DataFrame(rows),
                          params_to_get=[f"d_{t}" for t in tterms],
                          map_column="map_theta_d_KE",
                          get_columns=["titrant_name", "mutation"],
                          in_run_prefix="theta_"))

    if ctx.pair_labels:
        pair_df = pd.DataFrame({"pair": list(ctx.pair_labels),
                                "map_pair": range(len(ctx.pair_labels))})
        specs.append(dict(input_df=pair_df,
                          params_to_get=[f"epi_{t}" for t in gterms + tterms],
                          map_column="map_pair", get_columns=["pair"],
                          in_run_prefix="theta_"))
    specs.extend(extra)
    return specs


def build_calc_df(model, manual_titrant_df):
    """Concentration grid for θ extraction (as the O2 models)."""
    from tfscreen.tfmodel.generative.components.theta.thermo.O2_C12_K5_U1_a.thermo import (
        build_calc_df as _o2_build_calc_df,
    )
    calc_df, cols, extra = _o2_build_calc_df(model, manual_titrant_df)
    extra["variant"] = variant_from_priors(model.priors.theta)
    return calc_df, cols, extra


def _loader(param_posteriors):
    from tfscreen.tfmodel.inference.posteriors import get_posterior_samples

    def load(key):
        v = get_posterior_samples(param_posteriors, key)
        if hasattr(v, "shape") and not hasattr(v, "reshape"):
            v = v[:]
        return np.asarray(v)
    return load


def compute_theta_samples(calc_df, param_posteriors, *, tf_total,
                          conc_unit_scale=1.0, variant=None):
    """
    Posterior θ samples (S, N) at the rows of ``calc_df``.

    Reads the per-genotype term sites ``theta_{term}``.
    """
    variant = sp.Variant() if variant is None else variant
    load = _loader(param_posteriors)
    g = calc_df["genotype_idx"].values.astype(int)
    ti = calc_df["titrant_name_idx"].values.astype(int)
    tt = set(sp.titrant_terms(variant))
    terms = {}
    for t in sp.active_terms(variant):
        v = load(f"theta_{t}")
        terms[t] = v[:, ti, g] if t in tt else v[:, g]
    conc = calc_df["titrant_conc"].values.astype(float) * conc_unit_scale
    return sp.theta_np(terms, conc[None, :], tf_total, variant)


def predict_unmeasured(target_genotypes, titrant_names, manual_titrant_df,
                       mut_labels, pair_labels, param_posteriors, q_to_get, *,
                       tf_total, conc_unit_scale=1.0, theta_priors=None):
    """
    θ quantiles for genotypes assembled from wt, per-mutation and per-pair
    posteriors (genotypes need not be in the training data).
    """
    from tfscreen.tfmodel.analysis.predict_unmeasured import (
        _build_genotype_indicators,
        _build_prediction_grid,
    )

    variant = variant_from_priors(theta_priors)
    target_genotypes = list(target_genotypes)
    mut_mat, pair_mat, is_valid = _build_genotype_indicators(
        target_genotypes, mut_labels, pair_labels)
    calc_df, geno_idx, titrant_idx = _build_prediction_grid(
        target_genotypes, titrant_names, manual_titrant_df)
    load = _loader(param_posteriors)
    tt = set(sp.titrant_terms(variant))
    has_pairs = len(pair_labels) > 0

    rows = {}
    for t in sp.active_terms(variant):
        wt = load(f"theta_{t}_wt")
        d = load(f"theta_d_{t}")
        if t in tt:                                   # wt (S, T), d (S, T, M)
            v = wt[:, :, None] + np.einsum("stm,nm->stn", d, mut_mat)
            if has_pairs:
                v = v + np.einsum("sp,np->sn", load(f"theta_epi_{t}"),
                                  pair_mat)[:, None, :]
            rows[t] = v[:, titrant_idx, geno_idx]
        else:                                         # wt (S,), d (S, M)
            v = wt[:, None] + np.einsum("sm,nm->sn", d, mut_mat)
            if has_pairs:
                v = v + np.einsum("sp,np->sn", load(f"theta_epi_{t}"), pair_mat)
            rows[t] = v[:, geno_idx]

    conc = calc_df["titrant_conc"].values.astype(float) * conc_unit_scale
    theta_samples = sp.theta_np(rows, conc[None, :], tf_total, variant)
    theta_samples[:, ~is_valid[geno_idx]] = np.nan

    result_df = calc_df[["genotype", "titrant_name", "titrant_conc"]].copy()
    for q_name, q_val in q_to_get.items():
        result_df[q_name] = np.quantile(theta_samples, q_val, axis=0)
    return result_df
