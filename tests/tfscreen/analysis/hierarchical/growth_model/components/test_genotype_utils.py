import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro
import pytest
from flax.struct import dataclass
from tfscreen.analysis.hierarchical.growth_model.components.genotype_utils import (
    sample_genotype_parameter,
    sample_genotype_parameter_guide,
    get_genotype_parameter_guesses
)

@dataclass
class MockData:
    batch_idx: jnp.ndarray = None
    batch_size: int = 0
    num_genotype: int = 0
    epistasis_mode: str = "genotype"
    num_mutation: int = 0
    map_genotype_to_mutations: jnp.ndarray = None
    genotype_num_mutations: jnp.ndarray = None
    scale_vector: jnp.ndarray = None

def test_get_genotype_parameter_guesses():
    # Setup dummy data
    data = MockData(
        batch_idx=jnp.arange(10),
        batch_size=10,
        num_genotype=10,
        epistasis_mode="genotype",
        num_mutation=5,
        map_genotype_to_mutations=jnp.zeros((10, 5), dtype=jnp.int32),
        genotype_num_mutations=jnp.array([1]*10)
    )

    def guess_fn(name, size):
        return {f"{name}_val": jnp.zeros(size)}

    # Genotype mode
    guesses = get_genotype_parameter_guesses("test", data, guess_fn)
    assert "test_val" in guesses
    assert guesses["test_val"].shape == (10,)

    # Horseshoe mode
    data = data.replace(epistasis_mode="horseshoe")
    guesses = get_genotype_parameter_guesses("test", data, guess_fn)
    assert "test_mut_val" in guesses
    assert guesses["test_mut_val"].shape == (5,)
    assert "test_epi_tau" in guesses
    assert "test_epi_lambda" in guesses
    assert "test_epi_z" in guesses
    assert guesses["test_epi_lambda"].shape == (10,)
    assert guesses["test_epi_z"].shape == (10,)

    # Spikeslab mode
    data = data.replace(epistasis_mode="spikeslab")
    guesses = get_genotype_parameter_guesses("test", data, guess_fn)
    assert "test_mut_val" in guesses
    assert "test_epi_prob" in guesses
    assert "test_epi_w" in guesses
    assert "test_epi_z" in guesses

    # None mode
    data = data.replace(epistasis_mode="none")
    guesses = get_genotype_parameter_guesses("test", data, guess_fn)
    assert "test_mut_val" in guesses
    assert "test_epi_tau" not in guesses

def test_sampling_epistasis_mode():
    # Setup dummy data
    data = MockData(
        batch_idx=jnp.arange(5),
        batch_size=5,
        num_genotype=10,
        epistasis_mode="genotype",
        num_mutation=0,
        map_genotype_to_mutations=None,
        genotype_num_mutations=None,
        scale_vector=jnp.ones(5)
    )

    def sample_fn(name, size):
        return numpyro.sample(name, dist.Normal(jnp.zeros(size), 1.0))

    with numpyro.handlers.seed(rng_seed=0):
        val = sample_genotype_parameter("test", data, sample_fn)
    
    assert val.shape == (5,)

def test_sampling_horseshoe_mode():
    # Setup dummy data
    # 3 genotypes, 2 mutations
    # G0: [M0], G1: [M1], G2: [M0, M1]
    map_gt_mt = jnp.array([
        [1, 0],
        [2, 0],
        [1, 2]
    ], dtype=jnp.int32)
    num_muts = jnp.array([1, 1, 2])
    
    data = MockData(
        batch_idx=jnp.array([0, 1, 2]),
        batch_size=3,
        num_genotype=3,
        epistasis_mode="horseshoe",
        num_mutation=2,
        map_genotype_to_mutations=map_gt_mt,
        genotype_num_mutations=num_muts,
        scale_vector=jnp.ones(3)
    )

    def sample_fn(name, size):
        return numpyro.sample(name, dist.Normal(jnp.zeros(size), 1.0))

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.substitute(data={
            "test_mut": jnp.array([1.0, 10.0]),
            "test_epi_tau": 0.0, # Disable epistasis
        }):
            val = sample_genotype_parameter("test", data, sample_fn)
    
    expected = jnp.array([1.0, 10.0, 11.0])
    assert jnp.allclose(val, expected)

def test_sampling_none_mode():
    map_gt_mt = jnp.array([
        [1, 0],
        [2, 0],
        [1, 2]
    ], dtype=jnp.int32)
    num_muts = jnp.array([1, 1, 2])
    
    data = MockData(
        batch_idx=jnp.array([0, 1, 2]),
        batch_size=3,
        num_genotype=3,
        epistasis_mode="none",
        num_mutation=2,
        map_genotype_to_mutations=map_gt_mt,
        genotype_num_mutations=num_muts,
        scale_vector=jnp.ones(3)
    )

    def sample_fn(name, size):
        return numpyro.sample(name, dist.Normal(jnp.zeros(size), 1.0))

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.substitute(data={
            "test_mut": jnp.array([1.0, 10.0]),
        }):
            val = sample_genotype_parameter("test", data, sample_fn)
    
    expected = jnp.array([1.0, 10.0, 11.0])
    assert jnp.allclose(val, expected)

if __name__ == "__main__":
     pytest.main([__file__])
