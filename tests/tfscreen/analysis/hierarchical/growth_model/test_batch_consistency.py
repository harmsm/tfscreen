import pytest
import torch
import numpy as np
import pyro
import pyro.poutine as poutine
from tfscreen.analysis.hierarchical.growth_model.data_class import DataClass, GrowthData, BindingData
from tfscreen.analysis.hierarchical.growth_model.batch import get_batch
from tfscreen.analysis.hierarchical.growth_model.components.dk_geno import hierarchical as dk_geno_hierarchical
from tfscreen.analysis.hierarchical.growth_model.components.activity import horseshoe as activity_horseshoe
from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass


def _make_growth_data(batch_size, total_genotypes):
    return GrowthData(
        batch_size=batch_size,
        batch_idx=torch.arange(batch_size, dtype=torch.int64),
        scale_vector=torch.ones(batch_size),
        geno_theta_idx=torch.arange(batch_size, dtype=torch.int32),
        ln_cfu=torch.zeros((1, 1, 1, batch_size)),
        ln_cfu_std=torch.ones((1, 1, 1, batch_size)),
        t_pre=torch.zeros((1, 1, 1, batch_size)),
        t_sel=torch.zeros((1, 1, 1, batch_size)),
        good_mask=torch.ones((1, 1, 1, batch_size), dtype=torch.bool),
        congression_mask=torch.ones(batch_size, dtype=torch.bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=total_genotypes,
        num_condition_rep=1,
        map_condition_pre=torch.zeros(batch_size, dtype=torch.int64),
        map_condition_sel=torch.zeros(batch_size, dtype=torch.int64),
        titrant_conc=torch.tensor([1.0]),
        log_titrant_conc=torch.tensor([0.0]),
        wt_indexes=torch.tensor([0], dtype=torch.int64),
        scatter_theta=1,
    )


def test_batch_scaling_unbiased():
    """
    Test that get_batch correctly applies the mini-batch scaling factors.
    """
    total_genotypes = 100
    batch_size = 10
    num_binding = 5

    expected_scale = 19.0  # (100-5) / (10-5)

    full_scale_vector = torch.ones(total_genotypes)
    not_binding_idx = torch.arange(num_binding, total_genotypes, dtype=torch.int64)
    full_scale_vector[not_binding_idx] = expected_scale

    growth = GrowthData(
        batch_size=total_genotypes,
        batch_idx=torch.arange(total_genotypes, dtype=torch.int64),
        scale_vector=full_scale_vector,
        geno_theta_idx=torch.arange(total_genotypes, dtype=torch.int32),
        ln_cfu=torch.zeros((1, 1, 1, total_genotypes)),
        ln_cfu_std=torch.ones((1, 1, 1, total_genotypes)),
        t_pre=torch.zeros((1, 1, 1, total_genotypes)),
        t_sel=torch.zeros((1, 1, 1, total_genotypes)),
        good_mask=torch.ones((1, 1, 1, total_genotypes), dtype=torch.bool),
        congression_mask=torch.ones(total_genotypes, dtype=torch.bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=total_genotypes,
        num_condition_rep=1,
        map_condition_pre=torch.zeros(total_genotypes, dtype=torch.int64),
        map_condition_sel=torch.zeros(total_genotypes, dtype=torch.int64),
        titrant_conc=torch.tensor([1.0]),
        log_titrant_conc=torch.tensor([0.0]),
        wt_indexes=torch.tensor([0], dtype=torch.int64),
        scatter_theta=1,
    )

    full_data = DataClass(
        num_genotype=total_genotypes,
        batch_idx=torch.arange(total_genotypes, dtype=torch.int64),
        batch_size=total_genotypes,
        not_binding_idx=not_binding_idx,
        not_binding_batch_size=95,
        num_binding=num_binding,
        growth=growth,
        binding=None,
    )

    batch_idx = torch.tensor([0, 1, 2, 3, 4, 5, 20, 50, 80, 99], dtype=torch.int64)
    batch_data = get_batch(full_data, batch_idx)

    assert torch.all(batch_data.growth.scale_vector[:5] == 1.0)
    assert torch.all(batch_data.growth.scale_vector[5:] == expected_scale)


def test_component_shape_guards():
    """
    Test that component guides handle full-sized parameter substitution correctly.
    """
    total_genotypes = 100
    batch_size = 10

    growth = _make_growth_data(batch_size, total_genotypes)

    priors = dk_geno_hierarchical.get_priors()

    # Create full-sized substitution values (100 genotypes)
    substitutions = {
        "dk_geno_offset": torch.zeros(total_genotypes),
    }

    # Use poutine.do to substitute latent values
    pyro.clear_param_store()
    torch.manual_seed(0)
    with poutine.do(data=substitutions):
        dk_geno_hierarchical.guide("dk_geno", growth, priors)

    # Test activity_horseshoe
    priors_hs = activity_horseshoe.get_priors()
    substitutions_hs = {
        "activity_global_scale": torch.tensor(0.1),
        "activity_local_scale": torch.ones(total_genotypes),
        "activity_offset": torch.zeros(total_genotypes),
    }

    pyro.clear_param_store()
    torch.manual_seed(1)
    with poutine.do(data=substitutions_hs):
        activity_horseshoe.guide("activity", growth, priors_hs)


def test_num_genotype_preserved():
    """
    Test that get_batch preserves the total num_genotype while updating batch_size.
    """
    total_genotypes = 100
    batch_size = 10

    growth = GrowthData(
        batch_size=total_genotypes,
        batch_idx=torch.arange(total_genotypes, dtype=torch.int64),
        scale_vector=torch.ones(total_genotypes),
        geno_theta_idx=torch.arange(total_genotypes, dtype=torch.int32),
        ln_cfu=torch.zeros((1, 1, 1, total_genotypes)),
        ln_cfu_std=torch.ones((1, 1, 1, total_genotypes)),
        t_pre=torch.zeros((1, 1, 1, total_genotypes)),
        t_sel=torch.zeros((1, 1, 1, total_genotypes)),
        good_mask=torch.ones((1, 1, 1, total_genotypes), dtype=torch.bool),
        congression_mask=torch.ones(total_genotypes, dtype=torch.bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=total_genotypes,
        num_condition_rep=1,
        map_condition_pre=torch.zeros(total_genotypes, dtype=torch.int64),
        map_condition_sel=torch.zeros(total_genotypes, dtype=torch.int64),
        titrant_conc=torch.tensor([1.0]),
        log_titrant_conc=torch.tensor([0.0]),
        wt_indexes=torch.tensor([0], dtype=torch.int64),
        scatter_theta=1,
    )

    full_data = DataClass(
        num_genotype=total_genotypes,
        batch_idx=torch.arange(total_genotypes, dtype=torch.int64),
        batch_size=total_genotypes,
        not_binding_idx=torch.arange(total_genotypes, dtype=torch.int64),
        not_binding_batch_size=total_genotypes,
        num_binding=0,
        growth=growth,
        binding=None,
    )

    batch_idx = torch.arange(batch_size, dtype=torch.int64)
    batch_data = get_batch(full_data, batch_idx)

    assert batch_data.growth.batch_size == batch_size
    assert batch_data.growth.num_genotype == total_genotypes
    assert batch_data.num_genotype == total_genotypes
