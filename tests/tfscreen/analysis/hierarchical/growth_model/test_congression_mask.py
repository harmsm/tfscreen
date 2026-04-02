import pytest
import torch
from tfscreen.analysis.hierarchical.growth_model.components.transformation import _congression as transformation_congression

def test_update_thetas_with_mask():
    """Verify that the mask correctly selects between corrected and uncorrected thetas."""

    # Setup data
    num_geno = 4
    theta = torch.ones((2, num_geno)) * 0.5

    # params: lam=1.0, mu=0.0, sigma=1.0 (some correction will happen)
    params = (1.0, 0.0, 1.0)

    # Mask: genotypes 0 and 2 are corrected (True), 1 and 3 are NOT (False)
    mask = torch.tensor([True, False, True, False])

    # Get results
    # First, get what it would be without mask (all corrected)
    res_no_mask = transformation_congression.update_thetas(theta, params, mask=None)

    # Now, with mask
    res_with_mask = transformation_congression.update_thetas(theta, params, mask=mask)

    # Check shape
    assert res_with_mask.shape == (2, num_geno)

    # Genotypes 0 and 2 should match res_no_mask
    assert torch.allclose(res_with_mask[:, 0], res_no_mask[:, 0])
    assert torch.allclose(res_with_mask[:, 2], res_no_mask[:, 2])

    # Genotypes 1 and 3 should match original theta
    assert torch.allclose(res_with_mask[:, 1], theta[:, 1])
    assert torch.allclose(res_with_mask[:, 3], theta[:, 3])

    # Sanity check: verify that correction actually did something
    # For mu=0, sigma=1, lam=1, F(0.5) = 0.5
    assert not torch.allclose(res_no_mask[:, 0], theta[:, 0])
    # Expected value should be > 0.5 (shifts towards the mean of 0.5)
    assert torch.all(res_no_mask > 0.5)
    assert torch.all(res_no_mask < 0.8)

def test_update_thetas_no_mask_is_default():
    """Verify that passing mask=None is the same as all True."""
    num_geno = 3
    theta = torch.ones((1, num_geno)) * 0.4
    params = (1.0, 0.0, 1.0)

    res1 = transformation_congression.update_thetas(theta, params, mask=None)
    res2 = transformation_congression.update_thetas(theta, params, mask=torch.ones(num_geno, dtype=torch.bool))

    assert torch.allclose(res1, res2)
