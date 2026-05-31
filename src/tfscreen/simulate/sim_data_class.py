"""
Lightweight data container for simulation with tfmodel theta components.

SimData carries only the structural metadata fields that theta
define_model / run_model actually read.  Observation tensors
(ln_cfu, theta_obs, good_mask, …) are absent.

Field names and dtypes mirror GrowthData / BindingData so that any theta
component can accept a SimData without modification.
"""

import numpy as np
import jax.numpy as jnp
from flax.struct import dataclass, field
from typing import Any

from tfscreen.genetics.build_mut_geno_matrix import (
    build_mut_geno_matrix,
    build_mut_sparse_indices,
)

_ZERO_CONC_SENTINEL = 1e-20


@dataclass(frozen=True)
class SimData:
    """
    Minimal data container for prior-predictive theta simulation.

    Required scalar fields
    ----------------------
    num_titrant_name : int  — always 1 for simulation.
    num_titrant_conc : int  — C, number of unique effector concentrations.
    num_genotype     : int  — G.
    batch_size       : int  — G (all genotypes in one batch).
    scatter_theta    : int  — always 0 for simulation.

    Required JAX arrays
    -------------------
    titrant_conc     : (C,)  mM values (data units; theta models apply
                              conc_unit_scale internally to convert to M).
    log_titrant_conc : (C,)  log of above; zeros replaced by sentinel.
    batch_idx        : (G,)  identity arange(G).
    geno_theta_idx   : (G,)  identity arange(G).

    Optional — mutation-decomp models (lnK_mut, hill_mut, …)
    ---------------------------------------------------------
    num_mutation     : int
    num_pair         : int
    mut_nnz_mut_idx  : (nnz,) int32 COO row indices (mutation axis).
    mut_nnz_geno_idx : (nnz,) int32 COO col indices (genotype axis).
    pair_nnz_pair_idx: (nnz,) int32
    pair_nnz_geno_idx: (nnz,) int32

    Optional — struct NN / ddG-prior models (lnK_nn_prior, lnK_ddG_prior)
    -----------------------------------------------------------------------
    num_struct              : int
    struct_names            : tuple of str, length S
    struct_features         : (M, S, F) float32  where F=60 for NN features
                              or F=S for ddG-prior CSV values
    struct_n_chains         : (S,) int32 or None
    struct_contact_pair_idx : (P, 2) int32 or None
    struct_contact_distances: (P, S) float32 or None
    """

    # ── Required: shape scalars ──────────────────────────────────────────────
    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype:     int = field(pytree_node=False)
    batch_size:       int = field(pytree_node=False)
    scatter_theta:    int = field(pytree_node=False)

    # ── Required: JAX arrays ─────────────────────────────────────────────────
    titrant_conc:     jnp.ndarray
    log_titrant_conc: jnp.ndarray
    batch_idx:        jnp.ndarray
    geno_theta_idx:   jnp.ndarray

    # ── Optional: mutation-decomp ─────────────────────────────────────────────
    num_mutation:      int = field(pytree_node=False, default=0)
    num_pair:          int = field(pytree_node=False, default=0)
    mut_nnz_mut_idx:   Any = field(pytree_node=False, default=None)
    mut_nnz_geno_idx:  Any = field(pytree_node=False, default=None)
    pair_nnz_pair_idx: Any = field(pytree_node=False, default=None)
    pair_nnz_geno_idx: Any = field(pytree_node=False, default=None)

    # ── Optional: structural ensemble ─────────────────────────────────────────
    num_struct:               int = field(pytree_node=False, default=0)
    struct_names:             Any = field(pytree_node=False, default=None)
    struct_features:          Any = field(pytree_node=False, default=None)
    struct_n_chains:          Any = field(pytree_node=False, default=None)
    struct_contact_pair_idx:  Any = field(pytree_node=False, default=None)
    struct_contact_distances: Any = field(pytree_node=False, default=None)



def build_sim_data(library_df,
                   sample_df,
                   struct_ensemble_path=None,
                   skip_pairs=False):
    """
    Build a SimData from the simulation pipeline's core inputs.

    Parameters
    ----------
    library_df : pd.DataFrame
        Must contain a ``"genotype"`` column.  Genotype order defines the G
        dimension and must be consistent with the order used in thermo_to_growth.
    sample_df : pd.DataFrame
        Must contain a ``"titrant_conc"`` column (mM).  Only unique values
        are used; replicate / time structure is ignored.
    struct_ensemble_path : str or None
        Path to the structural data file required by lnK_nn_prior and
        lnK_ddG_prior theta components.
          - ``.h5`` / ``.hdf5``: HDF5 LigandMPNN ensemble (for lnK_nn_prior).
          - ``.csv``: per-mutation per-structure ddG prior means
            (for lnK_ddG_prior).
        Pass None for components that do not need structural data
        (e.g. hill, mwc_dimer_lnK_mut).
    skip_pairs : bool, default False
        Skip building pairwise epistasis indices.  Set True for theta
        components that do not model pairwise epistasis to avoid the
        quadratic enumeration over double mutants.

    Returns
    -------
    SimData
    """
    # ── Genotypes ─────────────────────────────────────────────────────────────
    genotypes = library_df["genotype"].tolist()
    G = len(genotypes)

    # ── Concentrations ────────────────────────────────────────────────────────
    concs = np.sort(sample_df["titrant_conc"].unique()).astype(float)
    safe_concs = np.where(concs == 0.0, _ZERO_CONC_SENTINEL, concs)
    log_concs = np.log(safe_concs)
    C = len(concs)

    # ── Mutation-genotype COO mapping ─────────────────────────────────────────
    mut_labels, pair_labels, mut_geno_matrix, pair_nnz_pair_idx, pair_nnz_geno_idx = \
        build_mut_geno_matrix(genotypes, skip_pairs=skip_pairs)
    mut_nnz_mut_idx, mut_nnz_geno_idx = build_mut_sparse_indices(mut_geno_matrix)
    M = len(mut_labels)
    P = len(pair_labels)

    # ── Structural ensemble ───────────────────────────────────────────────────
    struct_kw = dict(
        num_struct=0,
        struct_names=None,
        struct_features=None,
        struct_n_chains=None,
        struct_contact_pair_idx=None,
        struct_contact_distances=None,
    )
    if struct_ensemble_path is not None:
        path_lower = str(struct_ensemble_path).lower()
        if path_lower.endswith(".csv"):
            from tfscreen.tfmodel.generative.components.theta.struct.io import (
                load_ddG_prior_csv,
            )
            raw = load_ddG_prior_csv(struct_ensemble_path, mut_labels)
        else:
            from tfscreen.tfmodel.generative.components.theta.struct.io import (
                load_struct_ensemble,
            )
            raw = load_struct_ensemble(
                h5_path=struct_ensemble_path,
                mut_labels=mut_labels,
                pair_labels=pair_labels if not skip_pairs else None,
            )
        struct_kw = dict(
            num_struct=len(raw["struct_names"]),
            struct_names=raw["struct_names"],
            struct_features=raw["struct_features"],
            struct_n_chains=raw["struct_n_chains"],
            struct_contact_pair_idx=raw["struct_contact_pair_idx"],
            struct_contact_distances=raw["struct_contact_distances"],
        )

    return SimData(
        num_titrant_name=1,
        num_titrant_conc=C,
        num_genotype=G,
        batch_size=G,
        scatter_theta=0,
        titrant_conc=jnp.array(concs),
        log_titrant_conc=jnp.array(log_concs),
        batch_idx=jnp.arange(G),
        geno_theta_idx=jnp.arange(G),
        num_mutation=M,
        num_pair=P,
        mut_nnz_mut_idx=mut_nnz_mut_idx,
        mut_nnz_geno_idx=mut_nnz_geno_idx,
        pair_nnz_pair_idx=pair_nnz_pair_idx,
        pair_nnz_geno_idx=pair_nnz_geno_idx,
        **struct_kw,
    )
