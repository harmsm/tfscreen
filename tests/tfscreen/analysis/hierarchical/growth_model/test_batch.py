import pytest
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# --- Import Module Under Test (MUT) ---
from tfscreen.analysis.hierarchical.growth_model.batch import get_batch

# --- Mock Data Structures ---
# Must be proper stdlib dataclasses so dataclasses.replace() works inside get_batch.

@dataclass
class MockGrowthData:
    batch_size: int
    batch_idx: torch.Tensor
    scale_vector: torch.Tensor
    ln_cfu: torch.Tensor
    ln_cfu_std: torch.Tensor
    t_pre: torch.Tensor
    t_sel: torch.Tensor
    map_condition_pre: torch.Tensor
    map_condition_sel: torch.Tensor
    good_mask: torch.Tensor
    congression_mask: torch.Tensor
    geno_theta_idx: Optional[torch.Tensor] = None

@dataclass
class MockDataClass:
    growth: MockGrowthData


@pytest.fixture
def full_data_setup():
    """
    Creates a 'full' dataset with known values for testing slicing.
    Dimensions:
      - Genotypes (Total): 10
      - Other dims (Rep, Time, etc.): 1
    """
    total_size = 10

    scale_vector = torch.arange(total_size, dtype=torch.float64)
    ln_cfu = torch.arange(total_size, dtype=torch.float64) * 10.0
    ln_cfu_std = torch.arange(total_size, dtype=torch.float64) * 0.1
    t_pre = torch.arange(total_size, dtype=torch.float64) + 100.0
    t_sel = torch.arange(total_size, dtype=torch.float64) + 200.0

    map_condition_pre = torch.arange(total_size, dtype=torch.int64)
    map_condition_sel = torch.arange(total_size, dtype=torch.int64)
    good_mask = torch.ones(total_size, dtype=torch.bool)
    congression_mask = torch.ones(total_size, dtype=torch.bool)

    batch_idx = torch.arange(total_size, dtype=torch.int64)

    growth = MockGrowthData(
        batch_size=total_size,
        batch_idx=batch_idx,
        scale_vector=scale_vector,
        ln_cfu=ln_cfu,
        ln_cfu_std=ln_cfu_std,
        t_pre=t_pre,
        t_sel=t_sel,
        map_condition_pre=map_condition_pre,
        map_condition_sel=map_condition_sel,
        good_mask=good_mask,
        congression_mask=congression_mask,
        geno_theta_idx=torch.arange(total_size, dtype=torch.int32),
    )

    return MockDataClass(growth=growth)


def test_get_batch_slicing(full_data_setup):
    """
    Tests that get_batch correctly slices data based on the index array.
    """
    full_data = full_data_setup

    indices = torch.tensor([2, 5, 8], dtype=torch.int64)

    batch_data = get_batch(full_data, indices)

    # 1. Check Metadata Updates
    assert batch_data.growth.batch_size == 3
    assert torch.equal(batch_data.growth.batch_idx, indices)

    # 2. Check Data Slicing
    expected_scale = full_data.growth.scale_vector[indices]
    assert torch.equal(batch_data.growth.scale_vector, expected_scale)

    expected_ln_cfu = torch.tensor([20.0, 50.0, 80.0], dtype=torch.float64)
    assert torch.allclose(batch_data.growth.ln_cfu, expected_ln_cfu)

    expected_t_pre = torch.tensor([102.0, 105.0, 108.0], dtype=torch.float64)
    assert torch.allclose(batch_data.growth.t_pre, expected_t_pre)

    assert torch.equal(batch_data.growth.map_condition_pre, indices)


def test_get_batch_ordering(full_data_setup):
    """
    Tests that the returned batch respects the *order* of the provided indices,
    even if they are not sorted.
    """
    full_data = full_data_setup

    indices = torch.tensor([8, 0, 5], dtype=torch.int64)

    batch_data = get_batch(full_data, indices)

    expected_ln_cfu = torch.tensor([80.0, 0.0, 50.0], dtype=torch.float64)

    assert torch.allclose(batch_data.growth.ln_cfu, expected_ln_cfu)
    assert torch.equal(batch_data.growth.batch_idx, indices)


def test_get_batch_full_multidimensional_support():
    """
    Tests get_batch with actual multi-dimensional arrays to ensure the
    ellipsis (...) slicing works as intended.
    """
    data_shape = (2, 2, 5)
    full_array = torch.arange(20).reshape(data_shape)

    indices = torch.tensor([1, 3], dtype=torch.int64)
    expected = full_array[..., indices]

    assert expected.shape == (2, 2, 2)
    assert torch.equal(expected[:, :, 0], full_array[:, :, 1])
    assert torch.equal(expected[:, :, 1], full_array[:, :, 3])
