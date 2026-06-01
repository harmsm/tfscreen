import pytest
import numpy as np
import pandas as pd

from tfscreen.simulate.sim_data_class import SimData, build_sim_data, _ZERO_CONC_SENTINEL


@pytest.fixture
def simple_library_df():
    return pd.DataFrame({"genotype": ["wt", "A1B", "C2D"]})


@pytest.fixture
def simple_sample_df():
    return pd.DataFrame({
        "titrant_conc": [0.0, 1.0, 10.0],
        "condition_pre": ["M9", "M9", "M9"],
        "condition_sel": ["M9+Ab", "M9+Ab", "M9+Ab"],
    })


# ----------------------------------------------------------------------------
# build_sim_data — basic correctness
# ----------------------------------------------------------------------------

class TestBuildSimData:

    def test_returns_sim_data_instance(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert isinstance(sd, SimData)

    def test_num_genotype(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.num_genotype == 3

    def test_num_titrant_conc_unique(self, simple_library_df, simple_sample_df):
        # simple_sample_df has 3 unique concentrations
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.num_titrant_conc == 3

    def test_num_titrant_conc_deduplicates(self, simple_library_df):
        # Duplicate concentrations — should be deduplicated
        sample_df = pd.DataFrame({"titrant_conc": [1.0, 1.0, 10.0]})
        sd = build_sim_data(simple_library_df, sample_df)
        assert sd.num_titrant_conc == 2

    def test_num_titrant_name_always_one(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.num_titrant_name == 1

    def test_scatter_theta_always_zero(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.scatter_theta == 0

    def test_batch_size_equals_num_genotype(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.batch_size == sd.num_genotype

    def test_concentrations_sorted(self, simple_library_df):
        sample_df = pd.DataFrame({"titrant_conc": [10.0, 0.0, 1.0]})
        sd = build_sim_data(simple_library_df, sample_df)
        concs = np.array(sd.titrant_conc)
        assert np.all(concs[:-1] <= concs[1:])

    def test_zero_concentration_replaced_by_sentinel_in_log(self, simple_library_df):
        sample_df = pd.DataFrame({"titrant_conc": [0.0, 1.0]})
        sd = build_sim_data(simple_library_df, sample_df)
        log_concs = np.array(sd.log_titrant_conc)
        # The zero-conc entry should be log(sentinel), not -inf
        assert np.isfinite(log_concs[0])
        assert np.isclose(log_concs[0], np.log(_ZERO_CONC_SENTINEL))

    def test_titrant_conc_preserves_zero(self, simple_library_df):
        # The raw titrant_conc array keeps 0.0, only log version uses sentinel
        sample_df = pd.DataFrame({"titrant_conc": [0.0, 1.0]})
        sd = build_sim_data(simple_library_df, sample_df)
        concs = np.array(sd.titrant_conc)
        assert concs[0] == 0.0

    def test_batch_idx_is_arange(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        np.testing.assert_array_equal(np.array(sd.batch_idx), np.arange(3))

    def test_geno_theta_idx_is_arange(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        np.testing.assert_array_equal(np.array(sd.geno_theta_idx), np.arange(3))

    def test_num_mutation_positive(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.num_mutation > 0

    def test_no_struct_by_default(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.num_struct == 0
        assert sd.struct_names is None
        assert sd.struct_features is None

    def test_skip_pairs_sets_num_pair_zero(self, simple_library_df, simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df, skip_pairs=True)
        assert sd.num_pair == 0
