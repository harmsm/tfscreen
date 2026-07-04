
import jax.numpy as jnp
from flax.struct import (
    dataclass,
    field
)
from typing import Any


@dataclass(frozen=True)
class GrowthData:
    
    # Batch information
    batch_size: int = field(pytree_node=False)
    batch_idx: jnp.ndarray
    scale_vector: jnp.ndarray
    geno_theta_idx: jnp.ndarray

    # Data tensors
    ln_cfu: jnp.ndarray
    ln_cfu_std: jnp.ndarray
    t_pre: jnp.ndarray
    t_sel: jnp.ndarray
    good_mask: jnp.ndarray
    congression_mask: jnp.ndarray

        
    # Tensor shape
    num_replicate: int = field(pytree_node=False)
    num_time: int = field(pytree_node=False)
    num_condition_pre: int = field(pytree_node=False)
    num_condition_sel: int = field(pytree_node=False)
    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False)

    # mappers
    num_condition_rep: int = field(pytree_node=False)
    map_condition_pre: jnp.ndarray
    map_condition_sel: jnp.ndarray

    # 1D arrays of titrant concentration (corresponds to the second-to-last 
    # tensor dimension)
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray

    # meta data
    wt_indexes: jnp.ndarray
    scatter_theta: int = field(pytree_node=False)

    # Boolean mask, shape (num_genotype,), True = spiked genotype.
    # Used by ln_cfu0 component to apply a separate prior location.
    ln_cfu0_spiked_mask: jnp.ndarray

    # Boolean mask, shape (num_genotype,), True = wildtype genotype.
    # Used by ln_cfu0 component to apply a separate prior location for WT.
    ln_cfu0_wt_mask: jnp.ndarray

    # Boolean masks, shape (num_classes, num_genotype), partitioning the
    # library genotypes (non-spiked, non-wt) into hierarchical subgroups.
    # Row i is True for genotypes belonging to library class i.  Genotypes
    # not covered by any class row default to class-0 hyper-parameters.
    # With num_classes == 1 the single row covers all library genotypes.
    # None means fall back to treating all library genotypes as one class.
    ln_cfu0_library_masks: Any = field(default=None)
    num_ln_cfu0_library_classes: int = field(pytree_node=False, default=1)

    growth_shares_replicates: bool = field(pytree_node=False, default=False)

    # Optional mutation-decomposition matrices (set when using *_mut_decomp components).
    # Stored as pytree_node=False so they are treated as static by JAX tracing.
    # Shape: mut_geno_matrix (num_mutation, num_genotype).
    # Both the mutation-genotype matrix and the pair-genotype indicator matrix
    # are stored in COO format as two int32 index arrays of shape (nnz,) rather
    # than dense matrices, which would be O(100 GiB+) for large libraries.
    # Use apply_mut_matrix / apply_pair_matrix for memory-efficient scatter.
    num_mutation: int = field(pytree_node=False, default=0)
    num_pair: int = field(pytree_node=False, default=0)
    mut_geno_matrix: Any = field(pytree_node=False, default=None)
    mut_nnz_mut_idx: Any = field(pytree_node=False, default=None)
    mut_nnz_geno_idx: Any = field(pytree_node=False, default=None)
    pair_nnz_pair_idx: Any = field(pytree_node=False, default=None)
    pair_nnz_geno_idx: Any = field(pytree_node=False, default=None)

    # Optional structural ensemble data (set when using struct theta components,
    # e.g. struct.lac_dimer.lnK_nn_prior).  All fields stored as
    # pytree_node=False (static); downstream components convert to jnp arrays.
    #
    #   struct_names           : tuple of str, length S — structure names in
    #                            column order matching struct_features axis 1
    #   struct_features        : (M, S, 60) float32 — per-mutation per-structure
    #                            feature vector [logP row | one_hot_wt | one_hot_mut]
    #   struct_n_chains        : (S,) int32 — chains bearing the mutation per struct
    #   struct_contact_pair_idx: (P, 2) int32 — mutation-index pairs (i, j)
    #   struct_contact_distances: (P, S) float32 — min Cα-Cα distance in Å;
    #                            999.0 where the pair is not a contact in that struct
    num_struct: int = field(pytree_node=False, default=0)
    struct_names: Any = field(pytree_node=False, default=None)
    struct_features: Any = field(pytree_node=False, default=None)
    struct_n_chains: Any = field(pytree_node=False, default=None)
    struct_contact_pair_idx: Any = field(pytree_node=False, default=None)
    struct_contact_distances: Any = field(pytree_node=False, default=None)

    # Precomputed theta at growth titrant concentrations for *every* genotype
    # in the library (shape (num_titrant_name, num_titrant_conc, true_num_genotype)),
    # used only by transformation components whose congression correction needs
    # a population-wide reference distribution (see transformation/_congression.py).
    # None (the default) tells `jax_model` to compute this locally from
    # `data.growth` itself, which is only correct when `data.growth` already
    # spans the full genotype population (true during SVI training, since
    # genotype minibatching never shrinks `num_genotype`).  Prediction code
    # paths that subset genotypes (e.g. `analysis.prediction.predict`) must
    # supply this explicitly, computed against the true, unsubsetted library.
    external_theta_population: Any = field(default=None)

@dataclass(frozen=True)
class BindingData:

    # Batch information
    batch_size: int = field(pytree_node=False)
    batch_idx: jnp.ndarray
    scale_vector: jnp.ndarray
    geno_theta_idx: jnp.ndarray

    # Main data tensors
    theta_obs: jnp.ndarray
    theta_std: jnp.ndarray
    good_mask: jnp.ndarray

    # Tensor dimensions
    num_titrant_name: int = field(pytree_node=False)
    num_titrant_conc: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False) 

    # 1D arrays of titrant concentration (corresponds to the second-to-last 
    # tensor dimension)
    titrant_conc: jnp.ndarray
    log_titrant_conc: jnp.ndarray

    scatter_theta: int = field(pytree_node=False)

    # Optional mutation-decomposition matrices.  Stored as pytree_node=False
    # so they are treated as static by JAX tracing.  Defaults match GrowthData.
    # Both the mutation-genotype and pair-genotype matrices use COO format
    # (see GrowthData for details).  Use apply_mut_matrix / apply_pair_matrix.
    num_mutation: int = field(pytree_node=False, default=0)
    num_pair: int = field(pytree_node=False, default=0)
    mut_geno_matrix: Any = field(pytree_node=False, default=None)
    mut_nnz_mut_idx: Any = field(pytree_node=False, default=None)
    mut_nnz_geno_idx: Any = field(pytree_node=False, default=None)
    pair_nnz_pair_idx: Any = field(pytree_node=False, default=None)
    pair_nnz_geno_idx: Any = field(pytree_node=False, default=None)

    # Optional structural ensemble data.  Matches GrowthData struct fields;
    # see GrowthData for full field descriptions.
    num_struct: int = field(pytree_node=False, default=0)
    struct_names: Any = field(pytree_node=False, default=None)
    struct_features: Any = field(pytree_node=False, default=None)
    struct_n_chains: Any = field(pytree_node=False, default=None)
    struct_contact_pair_idx: Any = field(pytree_node=False, default=None)
    struct_contact_distances: Any = field(pytree_node=False, default=None)


@dataclass(frozen=True)
class PreSplitData:
    """
    Holds the optional pre-split (t = -t_pre) sequencing observations.

    These are taken from a single pooled aliquot just before the culture is
    split into titrant-concentration conditions, so they carry no
    condition_sel or titrant_conc index.  The prediction for each observation
    is simply ``ln_cfu0[replicate, condition_pre, genotype]``, making this a
    direct side-channel constraint on the initial-population parameter.

    Tensors are shaped ``(num_replicate, num_condition_pre, num_genotype)``
    and use the same categorical orderings as the companion GrowthData so
    that slicing with ``data.growth.batch_idx`` aligns the genotype axis.
    """

    # Data tensors
    ln_cfu_t0: jnp.ndarray
    ln_cfu_t0_std: jnp.ndarray
    good_mask: jnp.ndarray

    # Tensor dimensions (aligned with GrowthData)
    num_replicate: int = field(pytree_node=False)
    num_condition_pre: int = field(pytree_node=False)
    num_genotype: int = field(pytree_node=False)


@dataclass(frozen=True)
class BaseGrowthData:
    """
    Holds the optional base (reference-condition) growth-rate observations
    used to directly constrain dk_geno for a subset of genotypes.

    Unlike growth/binding, this measurement is taken independent of the
    titrant/selection system entirely -- it is a direct read of "how fast
    does this genotype grow" in whatever reference condition/medium the
    measurement was made in, used to anchor a new global scalar (k_ref) and
    tie it, via the existing per-genotype dk_geno latent, to genotypes with
    directly-measured growth rates. See generative/model.py's
    base_growth_obs block.

    Tensors are shaped ``(num_genotype,)`` using the same genotype ordering
    as the companion GrowthData, so that slicing with
    ``data.growth.batch_idx`` aligns the genotype axis -- the same pattern
    PreSplitData uses. Genotypes with no measurement (or multiple raw
    measurements collapsed via inverse-variance weighting during
    construction -- see model_orchestrator._read_base_growth_df) have
    good_mask == False and are excluded from the likelihood.
    """

    # Data tensors
    rate_obs: jnp.ndarray
    rate_std: jnp.ndarray
    good_mask: jnp.ndarray

    # Tensor dimension (aligned with GrowthData)
    num_genotype: int = field(pytree_node=False)


@dataclass(frozen=True)
class DataClass:
    """
    A container holding data needed to specify tfmodel, treated as a JAX
    Pytree.
    """

    num_genotype: int = field(pytree_node=False)

    batch_idx: jnp.ndarray
    batch_size: int = field(pytree_node=False)

    not_binding_idx: jnp.ndarray
    not_binding_batch_size: int = field(pytree_node=False)
    num_binding: int = field(pytree_node=False)

    # GrowthData when running the joint growth+binding model; None in binding_only mode.
    growth: Any = field(default=None)
    binding: BindingData = field(default=None)
    # Optional pre-split observations (t = -t_pre aliquot before condition split).
    presplit: Any = field(default=None)
    # Optional direct growth-rate measurements used to constrain dk_geno.
    base_growth: Any = field(default=None)


@dataclass(frozen=True)
class GrowthPriors:
    condition_growth: Any
    growth_transition: Any
    ln_cfu0: Any
    dk_geno: Any
    activity: Any
    transformation: Any
    theta_growth_noise: Any
    growth_noise: Any
    sample_offset: Any
    # Prior (loc, scale) for the optional base_growth k_ref scalar; None when
    # base_growth_df was not supplied (see BaseGrowthPriors).
    base_growth: Any = field(default=None)


@dataclass(frozen=True)
class BaseGrowthPriors:
    """
    Prior for the optional k_ref scalar (see BaseGrowthData / model.py's
    base_growth_obs block). k_ref_loc is derived empirically from wt's row
    in base_growth_df (dk_geno_wt == 0 makes wt's own measurement a direct
    read of k_ref) -- see model_orchestrator._derive_k_ref_guess.
    """
    k_ref_loc: float
    k_ref_scale: float

@dataclass(frozen=True)
class BindingPriors:
    theta_binding_noise: Any


@dataclass(frozen=True)
class PriorsClass:
    
    ## GrowthPriors and BindingPriors
    theta: BindingPriors
    growth: GrowthPriors
    binding: BindingPriors
