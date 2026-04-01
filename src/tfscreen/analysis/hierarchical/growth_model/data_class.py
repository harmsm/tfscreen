
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GrowthData:

    # Batch information
    batch_size: int
    batch_idx: torch.Tensor
    scale_vector: torch.Tensor
    geno_theta_idx: torch.Tensor

    # Data tensors
    ln_cfu: torch.Tensor
    ln_cfu_std: torch.Tensor
    t_pre: torch.Tensor
    t_sel: torch.Tensor
    good_mask: torch.Tensor
    congression_mask: torch.Tensor


    # Tensor shape
    num_replicate: int
    num_time: int
    num_condition_pre: int
    num_condition_sel: int
    num_titrant_name: int
    num_titrant_conc: int
    num_genotype: int

    # mappers
    num_condition_rep: int
    map_condition_pre: torch.Tensor
    map_condition_sel: torch.Tensor

    # 1D arrays of titrant concentration (corresponds to the second-to-last
    # tensor dimension)
    titrant_conc: torch.Tensor
    log_titrant_conc: torch.Tensor

    # meta data
    wt_indexes: torch.Tensor
    scatter_theta: int
    growth_shares_replicates: bool = False

    # Optional mutation-decomposition matrices (set when using *_mut_decomp components).
    # Shape: mut_geno_matrix (num_mutation, num_genotype),
    #        pair_geno_matrix (num_pair, num_genotype).
    num_mutation: int = 0
    num_pair: int = 0
    mut_geno_matrix: Any = None
    pair_geno_matrix: Any = None

@dataclass(frozen=True)
class BindingData:

    # Batch information
    batch_size: int
    batch_idx: torch.Tensor
    scale_vector: torch.Tensor
    geno_theta_idx: torch.Tensor

    # Main data tensors
    theta_obs: torch.Tensor
    theta_std: torch.Tensor
    good_mask: torch.Tensor

    # Tensor dimensions
    num_titrant_name: int
    num_titrant_conc: int
    num_genotype: int

    # 1D arrays of titrant concentration (corresponds to the second-to-last
    # tensor dimension)
    titrant_conc: torch.Tensor
    log_titrant_conc: torch.Tensor

    scatter_theta: int


@dataclass(frozen=True)
class DataClass:
    """
    A container holding data needed to specify growth_model.
    """

    num_genotype: int

    batch_idx: torch.Tensor
    batch_size: int

    not_binding_idx: torch.Tensor
    not_binding_batch_size: int
    num_binding: int

    # This will be a GrowthData and BindingData
    growth: GrowthData
    binding: BindingData


@dataclass(frozen=True)
class GrowthPriors:
    condition_growth: Any
    growth_transition: Any
    ln_cfu0: Any
    dk_geno: Any
    activity: Any
    transformation: Any
    theta_growth_noise: Any

@dataclass(frozen=True)
class BindingPriors:
    theta_binding_noise: Any


@dataclass(frozen=True)
class PriorsClass:

    ## GrowthPriors and BindingPriors
    theta: BindingPriors
    growth: GrowthPriors
    binding: BindingPriors
