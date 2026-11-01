"""
Unit tests for tfscreen.tfmodel.data_class.

The JIT-stability behaviour is covered in test_data_class_jit.py.  This
file focuses on field defaults, the replace() contract, and dataclass
construction for all four dataclasses in the module.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock

from tfscreen.tfmodel.data_class import (
    GrowthData,
    BindingData,
    PreSplitData,
    BaseGrowthData,
    DataClass,
    GrowthPriors,
    GrowthObsPriors,
    BaseGrowthPriors,
    BindingPriors,
    PriorsClass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_growth_data(num_genotype=4, **overrides):
    """Return a minimal valid GrowthData for testing."""
    G = num_genotype
    shape = (1, 1, 1, 1, 1, 1, G)
    kwargs = dict(
        batch_size=G,
        batch_idx=jnp.arange(G, dtype=jnp.int32),
        scale_vector=jnp.ones(G),
        geno_theta_idx=jnp.arange(G, dtype=jnp.int32),
        ln_cfu=jnp.zeros(shape),
        ln_cfu_std=jnp.ones(shape),
        t_pre=jnp.zeros(shape),
        t_sel=jnp.zeros(shape),
        good_mask=jnp.ones(shape, dtype=bool),
        num_replicate=1,
        num_time=1,
        num_condition_pre=1,
        num_condition_sel=1,
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=G,
        num_condition_rep=1,
        map_condition_pre=jnp.zeros(shape, dtype=jnp.int32),
        map_condition_sel=jnp.zeros(shape, dtype=jnp.int32),
        titrant_conc=jnp.array([0.0]),
        log_titrant_conc=jnp.array([0.0]),
        wt_indexes=jnp.array([0]),
        scatter_theta=0,
        ln_cfu0_spiked_mask=jnp.zeros(G, dtype=bool),
        ln_cfu0_wt_mask=jnp.zeros(G, dtype=bool),
    )
    kwargs.update(overrides)
    return GrowthData(**kwargs)


def _make_binding_data(num_genotype=4, **overrides):
    """Return a minimal valid BindingData for testing."""
    G = num_genotype
    kwargs = dict(
        batch_size=G,
        batch_idx=jnp.arange(G, dtype=jnp.int32),
        scale_vector=jnp.ones(G),
        geno_theta_idx=jnp.arange(G, dtype=jnp.int32),
        theta_obs=jnp.zeros((1, 1, G)),
        theta_std=jnp.ones((1, 1, G)),
        good_mask=jnp.ones((1, 1, G), dtype=bool),
        num_titrant_name=1,
        num_titrant_conc=1,
        num_genotype=G,
        titrant_conc=jnp.array([0.0]),
        log_titrant_conc=jnp.array([0.0]),
        scatter_theta=0,
    )
    kwargs.update(overrides)
    return BindingData(**kwargs)


def _make_data_class(num_genotype=4):
    """Return a minimal DataClass with both growth and binding."""
    G = num_genotype
    return DataClass(
        num_genotype=G,
        batch_idx=jnp.arange(G, dtype=jnp.int32),
        batch_size=G,
        not_binding_idx=jnp.arange(G, dtype=jnp.int32),
        not_binding_batch_size=G,
        num_binding=0,
        growth=_make_growth_data(G),
        binding=_make_binding_data(G),
    )


# ---------------------------------------------------------------------------
# GrowthData
# ---------------------------------------------------------------------------

class TestGrowthData:
    def test_construction(self):
        gd = _make_growth_data()
        assert gd.num_genotype == 4

    def test_required_fields_accessible(self):
        gd = _make_growth_data()
        assert gd.ln_cfu.shape[-1] == 4
        assert gd.batch_size == 4

    def test_default_ln_cfu0_library_masks_is_none(self):
        gd = _make_growth_data()
        assert gd.ln_cfu0_library_masks is None

    def test_default_num_ln_cfu0_library_classes(self):
        gd = _make_growth_data()
        assert gd.num_ln_cfu0_library_classes == 1

    def test_default_growth_shares_replicates(self):
        gd = _make_growth_data()
        assert gd.growth_shares_replicates is False

    def test_default_mutation_fields(self):
        gd = _make_growth_data()
        assert gd.num_mutation == 0
        assert gd.num_pair == 0
        assert gd.mut_geno_matrix is None

    def test_default_struct_fields(self):
        gd = _make_growth_data()
        assert gd.num_struct == 0
        assert gd.struct_names is None
        assert gd.struct_features is None

    def test_replace_updates_batch_size(self):
        gd = _make_growth_data(num_genotype=4)
        gd2 = gd.replace(batch_size=2)
        assert gd2.batch_size == 2
        assert gd.batch_size == 4  # original unchanged

    def test_replace_updates_ln_cfu(self):
        gd = _make_growth_data(num_genotype=4)
        new_ln_cfu = jnp.ones_like(gd.ln_cfu) * 99.0
        gd2 = gd.replace(ln_cfu=new_ln_cfu)
        assert float(gd2.ln_cfu.max()) == pytest.approx(99.0)
        assert float(gd.ln_cfu.max()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BindingData
# ---------------------------------------------------------------------------

class TestBindingData:
    def test_construction(self):
        bd = _make_binding_data()
        assert bd.num_genotype == 4

    def test_theta_obs_shape(self):
        bd = _make_binding_data()
        assert bd.theta_obs.shape == (1, 1, 4)

    def test_default_mutation_fields(self):
        bd = _make_binding_data()
        assert bd.num_mutation == 0
        assert bd.mut_geno_matrix is None

    def test_default_struct_fields(self):
        bd = _make_binding_data()
        assert bd.num_struct == 0
        assert bd.struct_names is None

    def test_replace_updates_batch_size(self):
        bd = _make_binding_data()
        bd2 = bd.replace(batch_size=2)
        assert bd2.batch_size == 2
        assert bd.batch_size == 4


# ---------------------------------------------------------------------------
# PreSplitData
# ---------------------------------------------------------------------------

class TestPreSplitData:
    def test_construction(self):
        G = 3
        ps = PreSplitData(
            ln_cfu_t0=jnp.zeros((1, 1, G)),
            ln_cfu_t0_std=jnp.ones((1, 1, G)),
            good_mask=jnp.ones((1, 1, G), dtype=bool),
            num_replicate=1,
            num_condition_pre=1,
            num_genotype=G,
        )
        assert ps.num_genotype == G
        assert ps.ln_cfu_t0.shape == (1, 1, G)

    def test_replace_works(self):
        G = 3
        ps = PreSplitData(
            ln_cfu_t0=jnp.zeros((1, 1, G)),
            ln_cfu_t0_std=jnp.ones((1, 1, G)),
            good_mask=jnp.ones((1, 1, G), dtype=bool),
            num_replicate=1,
            num_condition_pre=1,
            num_genotype=G,
        )
        ps2 = ps.replace(num_genotype=10)
        assert ps2.num_genotype == 10
        assert ps.num_genotype == G


# ---------------------------------------------------------------------------
# BaseGrowthData
# ---------------------------------------------------------------------------

class TestBaseGrowthData:
    def test_construction(self):
        G = 3
        bg = BaseGrowthData(
            rate_obs=jnp.zeros(G),
            rate_std=jnp.ones(G),
            good_mask=jnp.ones(G, dtype=bool),
            num_genotype=G,
        )
        assert bg.num_genotype == G
        assert bg.rate_obs.shape == (G,)
        assert bg.rate_std.shape == (G,)
        assert bg.good_mask.shape == (G,)

    def test_replace_works(self):
        G = 3
        bg = BaseGrowthData(
            rate_obs=jnp.zeros(G),
            rate_std=jnp.ones(G),
            good_mask=jnp.ones(G, dtype=bool),
            num_genotype=G,
        )
        bg2 = bg.replace(num_genotype=10)
        assert bg2.num_genotype == 10
        assert bg.num_genotype == G

    def test_replace_updates_rate_obs(self):
        G = 3
        bg = BaseGrowthData(
            rate_obs=jnp.zeros(G),
            rate_std=jnp.ones(G),
            good_mask=jnp.ones(G, dtype=bool),
            num_genotype=G,
        )
        new_rate = jnp.array([0.01, 0.02, 0.03])
        bg2 = bg.replace(rate_obs=new_rate)
        assert jnp.array_equal(bg2.rate_obs, new_rate)
        assert jnp.array_equal(bg.rate_obs, jnp.zeros(G))


# ---------------------------------------------------------------------------
# DataClass
# ---------------------------------------------------------------------------

class TestDataClass:
    def test_construction_with_growth_and_binding(self):
        dc = _make_data_class()
        assert dc.num_genotype == 4
        assert dc.growth is not None
        assert dc.binding is not None

    def test_default_growth_is_none(self):
        # growth defaults to None
        dc = DataClass(
            num_genotype=4,
            batch_idx=jnp.arange(4, dtype=jnp.int32),
            batch_size=4,
            not_binding_idx=jnp.arange(4, dtype=jnp.int32),
            not_binding_batch_size=4,
            num_binding=4,
        )
        assert dc.growth is None
        assert dc.binding is None
        assert dc.presplit is None
        assert dc.base_growth is None

    def test_replace_growth(self):
        dc = _make_data_class()
        new_gd = _make_growth_data(num_genotype=8)
        dc2 = dc.replace(growth=new_gd)
        assert dc2.growth.num_genotype == 8
        assert dc.growth.num_genotype == 4

    def test_batch_idx_shape(self):
        dc = _make_data_class(num_genotype=6)
        assert dc.batch_idx.shape == (6,)

    def test_replace_base_growth(self):
        dc = _make_data_class(num_genotype=4)
        bg = BaseGrowthData(
            rate_obs=jnp.zeros(4),
            rate_std=jnp.ones(4),
            good_mask=jnp.ones(4, dtype=bool),
            num_genotype=4,
        )
        dc2 = dc.replace(base_growth=bg)
        assert dc2.base_growth is bg
        assert dc.base_growth is None


# ---------------------------------------------------------------------------
# PriorsClass
# ---------------------------------------------------------------------------

class TestPriorsClass:
    def test_construction(self):
        mock_any = MagicMock()
        gp = GrowthPriors(
            condition_growth=mock_any,
            growth_transition=mock_any,
            ln_cfu0=mock_any,
            dk_geno=mock_any,
            activity=mock_any,
            transformation=mock_any,
            theta_growth_noise=mock_any,
            growth_noise=mock_any,
            sample_offset=mock_any,
            growth_obs=mock_any,
        )
        bp = BindingPriors(theta_binding_noise=mock_any)
        pc = PriorsClass(theta=bp, growth=gp, binding=bp)
        assert pc.growth is gp
        assert pc.binding is bp
        assert pc.theta is bp

    def test_growth_priors_base_growth_defaults_to_none(self):
        mock_any = MagicMock()
        gp = GrowthPriors(
            condition_growth=mock_any,
            growth_transition=mock_any,
            ln_cfu0=mock_any,
            dk_geno=mock_any,
            activity=mock_any,
            transformation=mock_any,
            theta_growth_noise=mock_any,
            growth_noise=mock_any,
            sample_offset=mock_any,
            growth_obs=mock_any,
        )
        assert gp.base_growth is None

    def test_base_growth_priors_construction(self):
        bgp = BaseGrowthPriors(k_ref_loc=0.02, k_ref_scale=0.02)
        assert bgp.k_ref_loc == pytest.approx(0.02)
        assert bgp.k_ref_scale == pytest.approx(0.02)

    def test_growth_obs_priors_construction(self):
        gop = GrowthObsPriors(nu_concentration=2.0, nu_rate=0.1)
        assert gop.nu_concentration == pytest.approx(2.0)
        assert gop.nu_rate == pytest.approx(0.1)

    def test_growth_priors_with_base_growth_set(self):
        mock_any = MagicMock()
        bgp = BaseGrowthPriors(k_ref_loc=0.015, k_ref_scale=0.01)
        gp = GrowthPriors(
            condition_growth=mock_any,
            growth_transition=mock_any,
            ln_cfu0=mock_any,
            dk_geno=mock_any,
            activity=mock_any,
            transformation=mock_any,
            theta_growth_noise=mock_any,
            growth_noise=mock_any,
            sample_offset=mock_any,
            growth_obs=mock_any,
            base_growth=bgp,
        )
        assert gp.base_growth is bgp


# ---------------------------------------------------------------------------
# StaticArray
# ---------------------------------------------------------------------------

class TestStaticArray:
    from tfscreen.tfmodel.data_class import StaticArray as _SA

    def test_equality_and_hash_follow_contents(self):
        a = self._SA(np.array([1, 2, 3]))
        b = self._SA(np.array([1, 2, 3]))
        assert a == b and hash(a) == hash(b)
        assert a != self._SA(np.array([1, 2, 4]))
        assert a != self._SA(np.array([1, 2, 3], dtype=np.int8))
        assert a != self._SA(np.array([[1, 2, 3]]))

    def test_comparison_with_other_types_is_not_implemented(self):
        a = self._SA(np.array([1, 2]))
        assert a.__eq__(np.array([1, 2])) is NotImplemented
        assert (a == "x") is False

    def test_behaves_as_the_array(self):
        a = self._SA(np.array([[1, 2], [3, 4]], dtype=np.int32))
        assert a.shape == (2, 2)
        assert a.sum(axis=0).tolist() == [4, 6]
        assert a[1, 0] == 3
        assert len(a) == 2
        assert [row.tolist() for row in a] == [[1, 2], [3, 4]]
        np.testing.assert_array_equal(np.asarray(a), a.array)
        np.testing.assert_array_equal(jnp.array(a), a.array)
        assert jnp.array(a, dtype=jnp.float32).dtype == jnp.float32
        assert np.asarray(a, dtype=float).dtype == float
        assert "StaticArray" in repr(a)

    def test_copied_and_read_only(self):
        source = np.array([1.0, 2.0])
        a = self._SA(source)
        source[0] = 9.0
        assert a[0] == 1.0
        with pytest.raises(ValueError):
            a.array[0] = 5.0
        with pytest.raises(AttributeError):
            a.foo = 1

    def test_pickle_round_trip(self):
        import pickle
        import copy
        a = self._SA(np.array([1, 2, 3]))
        for b in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
            assert b == a and hash(b) == hash(a)
            assert b.shape == (3,)

    def test_unset_slot_raises_attribute_error(self):
        a = self._SA.__new__(self._SA)
        with pytest.raises(AttributeError):
            a.shape

    def test_wrapping_is_idempotent(self):
        from tfscreen.tfmodel.data_class import StaticArray
        gd = _make_growth_data(mut_nnz_mut_idx=np.array([0, 1]))
        held = gd.mut_nnz_mut_idx
        assert isinstance(held, StaticArray)
        assert gd.replace(batch_size=4).mut_nnz_mut_idx is held
