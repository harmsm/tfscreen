import pytest
import torch
from dataclasses import dataclass
from typing import Optional

from tfscreen.analysis.hierarchical.growth_model.batch import get_batch

# Must be proper stdlib dataclasses so dataclasses.replace() works inside get_batch.

@dataclass
class MockGrowthData:
    batch_size: int
    num_genotype: int
    batch_idx: torch.Tensor
    scale_vector: torch.Tensor
    geno_theta_idx: torch.Tensor
    ln_cfu: torch.Tensor
    ln_cfu_std: torch.Tensor
    t_pre: torch.Tensor
    t_sel: torch.Tensor
    map_condition_pre: torch.Tensor
    map_condition_sel: torch.Tensor
    good_mask: torch.Tensor
    congression_mask: torch.Tensor

@dataclass
class MockDataClass:
    growth: MockGrowthData


def test_get_batch_metadata_updates():
    """
    Test that get_batch correctly updates geno_theta_idx and batch_size.
    """
    total_size = 10
    batch_size = 2

    growth = MockGrowthData(
        batch_size=batch_size,
        num_genotype=batch_size,
        batch_idx=torch.arange(batch_size, dtype=torch.int64),
        scale_vector=torch.ones((1, total_size)),
        geno_theta_idx=torch.arange(batch_size, dtype=torch.int32),
        ln_cfu=torch.zeros((1, 1, 1, 1, 1, 1, total_size)),
        ln_cfu_std=torch.zeros((1, 1, 1, 1, 1, 1, total_size)),
        t_pre=torch.zeros((1, 1, 1, 1, 1, 1, total_size)),
        t_sel=torch.zeros((1, 1, 1, 1, 1, 1, total_size)),
        map_condition_pre=torch.zeros((1, total_size), dtype=torch.int64),
        map_condition_sel=torch.zeros((1, total_size), dtype=torch.int64),
        good_mask=torch.ones((1, total_size), dtype=torch.bool),
        congression_mask=torch.ones((total_size,), dtype=torch.bool),
    )

    full_data = MockDataClass(growth=growth)

    new_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    new_batch_size = len(new_indices)

    batch_data = get_batch(full_data, new_indices)

    assert batch_data.growth.batch_size == new_batch_size
    assert batch_data.growth.num_genotype == batch_size  # original value preserved
    assert torch.equal(batch_data.growth.batch_idx, new_indices)
    assert torch.equal(batch_data.growth.geno_theta_idx,
                       torch.arange(new_batch_size, dtype=torch.int32))

    assert batch_data.growth.ln_cfu.shape[-1] == new_batch_size
    assert batch_data.growth.congression_mask.shape[-1] == new_batch_size
