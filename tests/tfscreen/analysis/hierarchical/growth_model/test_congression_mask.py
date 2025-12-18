import pytest
import jax.numpy as jnp
from tfscreen.analysis.hierarchical.growth_model.components import transformation_congression

def test_update_thetas_with_mask():
    """Verify that the mask correctly selects between corrected and uncorrected thetas."""
    
    # Setup data
    num_geno = 4
    theta = jnp.ones((2, num_geno)) * 0.5
    
    # params: lam=1.0, a=1.0, b=1.0 (some correction will happen)
    params = (1.0, 1.0, 1.0)
    
    # Mask: genotypes 0 and 2 are corrected (True), 1 and 3 are NOT (False)
    mask = jnp.array([True, False, True, False])
    
    # Get results
    # First, get what it would be without mask (all corrected)
    res_no_mask = transformation_congression.update_thetas(theta, params, mask=None)
    
    # Now, with mask
    res_with_mask = transformation_congression.update_thetas(theta, params, mask=mask)
    
    # Check shape
    assert res_with_mask.shape == (2, num_geno)
    
    # Genotypes 0 and 2 should match res_no_mask
    assert jnp.allclose(res_with_mask[:, 0], res_no_mask[:, 0])
    assert jnp.allclose(res_with_mask[:, 2], res_no_mask[:, 2])
    
    # Genotypes 1 and 3 should match original theta
    assert jnp.allclose(res_with_mask[:, 1], theta[:, 1])
    assert jnp.allclose(res_with_mask[:, 3], theta[:, 3])
    
    # Sanity check: verify that correction actually did something
    # For a=1, b=1, lam=1, Fx = x, prob_is_max = exp(lam*(x-1)) = exp(x-1)
    # integral part: integral(m * lam * exp(lam*(m-1)) dm, m=x..1)
    # = [m * exp(m-1) - exp(m-1)] from x to 1
    # = (1*1 - 1) - (x*exp(x-1) - exp(x-1)) = exp(x-1) - x*exp(x-1)
    # total = x*exp(x-1) + exp(x-1) - x*exp(x-1) = exp(x-1)
    # For x=0.5, exp(-0.5) approx 0.606
    assert not jnp.allclose(res_no_mask[:, 0], theta[:, 0])
    assert jnp.isclose(res_no_mask[0, 0], jnp.exp(-0.5), atol=1e-3)

def test_update_thetas_no_mask_is_default():
    """Verify that passing mask=None is the same as all True."""
    num_geno = 3
    theta = jnp.ones((1, num_geno)) * 0.4
    params = (1.0, 2.0, 2.0)
    
    res1 = transformation_congression.update_thetas(theta, params, mask=None)
    res2 = transformation_congression.update_thetas(theta, params, mask=jnp.ones(num_geno, dtype=bool))
    
    assert jnp.allclose(res1, res2)

if __name__ == "__main__":
    test_update_thetas_with_mask()
    test_update_thetas_no_mask_is_default()
    print("Mask tests passed!")
