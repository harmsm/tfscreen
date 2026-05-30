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

    # ── Activity component support fields ─────────────────────────────────────

    def test_scale_vector_populated(self, simple_library_df, simple_sample_df):
        """build_sim_data must populate scale_vector."""
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.scale_vector is not None

    def test_scale_vector_all_ones(self, simple_library_df, simple_sample_df):
        """scale_vector should be all 1.0 for simulation (no subsampling)."""
        sd = build_sim_data(simple_library_df, simple_sample_df)
        np.testing.assert_allclose(np.array(sd.scale_vector), 1.0)

    def test_scale_vector_length_equals_num_genotype(self, simple_library_df,
                                                      simple_sample_df):
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert len(np.array(sd.scale_vector)) == sd.num_genotype

    def test_wt_indexes_populated(self, simple_library_df, simple_sample_df):
        """build_sim_data must populate wt_indexes."""
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert sd.wt_indexes is not None

    def test_wt_indexes_identifies_wt_genotype(self, simple_library_df,
                                                simple_sample_df):
        """wt_indexes must point to the 'wt' entry in library_df."""
        # simple_library_df: ["wt", "A1B", "C2D"]  →  wt is at index 0
        sd = build_sim_data(simple_library_df, simple_sample_df)
        wt_idx = np.array(sd.wt_indexes)
        assert 0 in wt_idx

    def test_wt_indexes_correct_count_single_wt(self, simple_library_df,
                                                 simple_sample_df):
        """One 'wt' genotype → wt_indexes has exactly one entry."""
        sd = build_sim_data(simple_library_df, simple_sample_df)
        assert len(np.array(sd.wt_indexes)) == 1

    def test_wt_indexes_empty_when_no_wt(self, simple_sample_df):
        """Library with no 'wt' genotype → wt_indexes is empty."""
        library_df = pd.DataFrame({"genotype": ["A1B", "C2D", "A1B/C2D"]})
        sd = build_sim_data(library_df, simple_sample_df)
        assert len(np.array(sd.wt_indexes)) == 0

    def test_wt_indexes_correct_when_wt_not_first(self, simple_sample_df):
        """wt_indexes should reflect the actual position of 'wt' in library_df."""
        library_df = pd.DataFrame({"genotype": ["A1B", "wt", "C2D"]})
        sd = build_sim_data(library_df, simple_sample_df)
        wt_idx = np.array(sd.wt_indexes)
        assert len(wt_idx) == 1
        assert wt_idx[0] == 1  # wt is at index 1
