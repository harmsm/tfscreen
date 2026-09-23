import doctest

import numpy as np
import pandas as pd
import pytest

import tfscreen.genetics.library_design as library_design
from tfscreen.genetics import (
    estimate_library_mixture,
    expected_library_composition,
    scale_library_design,
)


def _library(rows):
    """rows: (origin, genotype, degeneracy) -> build_library_df-like frame."""
    df = pd.DataFrame(rows, columns=["library_origin", "genotype", "degeneracy"])
    df["weight"] = df["degeneracy"] / df.groupby("library_origin")["degeneracy"].transform("sum")
    return df


@pytest.fixture
def reference_library():
    # single-1: 10 sequences (wt x2, M42I x1, others), double: 100 sequences,
    # spiked: wt + M42I + H74A/K84L (not encoded by the bulk).
    return _library([
        ("single-1", "wt", 2),
        ("single-1", "M42I", 1),
        ("single-1", "M42V", 7),
        ("double-1-2", "wt", 4),
        ("double-1-2", "M42I", 6),
        ("double-1-2", "M42V/H74A", 90),
        ("spiked", "wt", 1),
        ("spiked", "M42I", 1),
        ("spiked", "H74A/K84L", 1),
    ])


def test_doctests():
    assert doctest.testmod(library_design).failed == 0


class TestExpectedLibraryComposition:

    def test_bulk_fraction_by_hand(self, reference_library):
        mixture = {"single-1": 10, "double-1-2": 100, "spiked": 3}
        out = expected_library_composition(reference_library, mixture).set_index("genotype")

        # masses: single-1 = 10*deg/10 = deg; double = 100*deg/100 = deg;
        # spiked = 3 * 1/3 = 1 per spiked genotype
        wt_bulk, wt_spike = 2 + 4, 1
        m42i_bulk, m42i_spike = 1 + 6, 1
        assert out.loc["wt", "bulk_fraction"] == pytest.approx(wt_bulk / (wt_bulk + wt_spike))
        assert out.loc["M42I", "bulk_fraction"] == pytest.approx(m42i_bulk / (m42i_bulk + m42i_spike))
        assert out.loc["H74A/K84L", "bulk_fraction"] == 0.0
        assert out.loc["M42V", "bulk_fraction"] == 1.0
        assert out.loc["M42V/H74A", "bulk_fraction"] == 1.0

        total = 10 + 100 + 3
        assert out.loc["wt", "pool_fraction"] == pytest.approx((wt_bulk + wt_spike) / total)
        assert out["pool_fraction"].sum() == pytest.approx(1.0)

    def test_only_ratios_matter(self, reference_library):
        a = expected_library_composition(reference_library,
                                         {"single-1": 1, "double-1-2": 2, "spiked": 3})
        b = expected_library_composition(reference_library,
                                         {"single-1": 10, "double-1-2": 20, "spiked": 30})
        pd.testing.assert_frame_equal(a, b)

    def test_zero_spike_mixture_gives_pure_bulk(self, reference_library):
        out = expected_library_composition(
            reference_library, {"single-1": 1, "double-1-2": 1, "spiked": 0}
        ).set_index("genotype")
        assert out.loc["wt", "bulk_fraction"] == 1.0
        # A genotype with no mass at all has an undefined bulk fraction.
        assert np.isnan(out.loc["H74A/K84L", "bulk_fraction"])
        assert out.loc["H74A/K84L", "pool_fraction"] == 0.0

    def test_unnormalized_weights_are_normalized(self, reference_library):
        mixture = {"single-1": 10, "double-1-2": 100, "spiked": 3}
        scaled = reference_library.assign(weight=reference_library["weight"] * 7.0)
        pd.testing.assert_frame_equal(
            expected_library_composition(reference_library, mixture),
            expected_library_composition(scaled, mixture),
        )

    def test_custom_spiked_origin(self, reference_library):
        df = reference_library.replace({"library_origin": {"spiked": "controls"}})
        out = expected_library_composition(
            df, {"single-1": 10, "double-1-2": 100, "controls": 3},
            spiked_origin="controls",
        ).set_index("genotype")
        assert out.loc["H74A/K84L", "bulk_fraction"] == 0.0

    def test_mixture_keys_must_match(self, reference_library):
        with pytest.raises(ValueError, match="Missing"):
            expected_library_composition(reference_library, {"single-1": 1, "spiked": 1})
        with pytest.raises(ValueError, match="Not in the library"):
            expected_library_composition(
                reference_library,
                {"single-1": 1, "double-1-2": 1, "spiked": 1, "single-2": 1},
            )

    def test_bad_mixture_values(self, reference_library):
        with pytest.raises(ValueError, match="non-negative"):
            expected_library_composition(
                reference_library, {"single-1": -1, "double-1-2": 1, "spiked": 1})
        with pytest.raises(ValueError, match="non-negative"):
            expected_library_composition(
                reference_library, {"single-1": 0, "double-1-2": 0, "spiked": 0})

    def test_missing_column(self, reference_library):
        with pytest.raises(ValueError, match="weight"):
            expected_library_composition(reference_library.drop(columns="weight"),
                                         {"single-1": 1, "double-1-2": 1, "spiked": 1})


class TestScaleLibraryDesign:

    @pytest.fixture
    def target_library(self):
        # Same degeneracy patterns for the spikes, fewer bulk sequences.
        return _library([
            ("single-1", "wt", 1),
            ("single-1", "M42I", 1),
            ("single-1", "M42V", 3),
            ("double-1-2", "wt", 2),
            ("double-1-2", "M42I", 3),
            ("double-1-2", "M42V/H74A", 5),
            ("spiked", "wt", 1),
            ("spiked", "M42I", 1),
            ("spiked", "H74A/K84L", 1),
        ])

    def test_per_sequence_quantities_preserved(self, reference_library, target_library):
        mixture = {"single-1": 100, "double-1-2": 1000, "spiked": 1}
        sizes = {"single-1": 100_000, "double-1-2": 300_000, "spiked": 1_000}
        out = scale_library_design(reference_library, target_library, mixture, sizes,
                                   cfu0=6.5e7, total_num_reads=1_000_000_000)

        ref_seqs = {"single-1": 10, "double-1-2": 100, "spiked": 3}
        tgt_seqs = {"single-1": 5, "double-1-2": 10, "spiked": 3}

        ref_total = sum(mixture.values())
        tgt_total = sum(out["library_mixture"].values())
        ref_spike_seq = mixture["spiked"] / ref_seqs["spiked"]
        tgt_spike_seq = out["library_mixture"]["spiked"] / tgt_seqs["spiked"]
        for k in mixture:
            # abundance of one sequence relative to one spiked sequence
            assert (out["library_mixture"][k] / tgt_seqs[k] / tgt_spike_seq
                    == pytest.approx(mixture[k] / ref_seqs[k] / ref_spike_seq))
            # cells per sequence
            assert (out["cfu0"] * out["library_mixture"][k] / tgt_total / tgt_seqs[k]
                    == pytest.approx(6.5e7 * mixture[k] / ref_total / ref_seqs[k]))
            # transformants per sequence (to rounding)
            assert (out["transform_sizes"][k] / tgt_seqs[k]
                    == pytest.approx(sizes[k] / ref_seqs[k], rel=1e-3))

        # Unchanged sequence count keeps the reference value.
        assert out["library_mixture"]["spiked"] == 1.0
        assert out["transform_sizes"]["spiked"] == 1_000
        assert isinstance(out["transform_sizes"]["double-1-2"], int)
        assert isinstance(out["total_num_reads"], int)

    def test_bulk_fraction_carries_over_with_matching_degeneracy(self, reference_library):
        # Target library: each bulk origin has half the reference's sequences,
        # but wt and M42I keep the same number of encoding sequences. Scaling
        # preserves mass per sequence, so their bulk fractions carry over.
        target = _library([
            ("single-1", "wt", 2),
            ("single-1", "M42I", 1),
            ("single-1", "M42V", 2),        # 5 sequences (ref 10)
            ("double-1-2", "wt", 4),
            ("double-1-2", "M42I", 6),
            ("double-1-2", "M42V/H74A", 40),  # 50 sequences (ref 100)
            ("spiked", "wt", 1),
            ("spiked", "M42I", 1),
            ("spiked", "H74A/K84L", 1),
        ])
        mixture = {"single-1": 100, "double-1-2": 1000, "spiked": 1}
        sizes = {"single-1": 1, "double-1-2": 1, "spiked": 1}
        scaled = scale_library_design(reference_library, target, mixture, sizes)

        ref = expected_library_composition(reference_library, mixture).set_index("genotype")
        tgt = expected_library_composition(target, scaled["library_mixture"]).set_index("genotype")

        for g in ("wt", "M42I", "H74A/K84L"):
            assert tgt.loc[g, "bulk_fraction"] == pytest.approx(ref.loc[g, "bulk_fraction"])

        # Copying the reference mixture unscaled would not preserve them.
        naive = expected_library_composition(target, mixture).set_index("genotype")
        assert naive.loc["M42I", "bulk_fraction"] != pytest.approx(ref.loc["M42I", "bulk_fraction"])

    def test_optional_totals(self, reference_library, target_library):
        out = scale_library_design(reference_library, target_library,
                                   {"single-1": 1, "double-1-2": 1, "spiked": 1},
                                   {"single-1": 1, "double-1-2": 1, "spiked": 1})
        assert out["cfu0"] is None
        assert out["total_num_reads"] is None
        # Tiny scaled transform sizes are floored at one transformant.
        assert min(out["transform_sizes"].values()) >= 1

    def test_origins_must_match(self, reference_library, target_library):
        bad = target_library[target_library["library_origin"] != "double-1-2"]
        with pytest.raises(ValueError, match="same origins"):
            scale_library_design(reference_library, bad,
                                 {"single-1": 1, "double-1-2": 1, "spiked": 1},
                                 {"single-1": 1, "double-1-2": 1, "spiked": 1})
        with pytest.raises(ValueError, match="transform_sizes"):
            scale_library_design(reference_library, target_library,
                                 {"single-1": 1, "double-1-2": 1, "spiked": 1},
                                 {"single-1": 1, "spiked": 1})

    def test_needs_degeneracy(self, reference_library, target_library):
        with pytest.raises(ValueError, match="degeneracy"):
            scale_library_design(reference_library.drop(columns="degeneracy"),
                                 target_library,
                                 {"single-1": 1, "double-1-2": 1, "spiked": 1},
                                 {"single-1": 1, "double-1-2": 1, "spiked": 1})


class TestEstimateLibraryMixture:

    @pytest.fixture
    def library(self):
        # Two tiles: singles in each, cross-tile doubles, and spikes.
        return _library([
            ("single-1", "wt", 2), ("single-1", "A1V", 3), ("single-1", "A1G", 5),
            ("single-2", "wt", 1), ("single-2", "B2V", 4), ("single-2", "B2G", 5),
            ("double-1-2", "wt", 2), ("double-1-2", "A1V", 3), ("double-1-2", "B2V", 4),
            ("double-1-2", "A1V/B2V", 6), ("double-1-2", "A1G/B2G", 5),
            ("spiked", "wt", 1), ("spiked", "A1V", 1), ("spiked", "C3D", 1),
        ])

    def _expected_abundance(self, library, mixture, total=1e6):
        comp = expected_library_composition(library, mixture)
        return pd.Series(comp["pool_fraction"].to_numpy() * total,
                         index=comp["genotype"].to_numpy())

    def test_recovers_mixture_from_expected_abundance(self, library):
        truth = {"single-1": 400, "single-2": 550, "double-1-2": 1000, "spiked": 20}
        observed = self._expected_abundance(library, truth)
        mixture, groups, unexplained = estimate_library_mixture(library, observed)
        total = sum(truth.values())
        for k, v in truth.items():
            assert mixture[k] == pytest.approx(v / total, rel=1e-6)
        assert unexplained == 0.0
        assert groups["fitted_share"].to_numpy() == pytest.approx(
            groups["observed_share"].to_numpy())

    def test_wt_excess_is_ignored_by_default(self, library):
        truth = {"single-1": 1, "single-2": 1, "double-1-2": 2, "spiked": 0.5}
        observed = self._expected_abundance(library, truth)
        observed["wt"] *= 50.0
        mixture, _, _ = estimate_library_mixture(library, observed)
        total = sum(truth.values())
        for k, v in truth.items():
            assert mixture[k] == pytest.approx(v / total, rel=1e-6)
        # Including wt distorts the estimate.
        biased, _, _ = estimate_library_mixture(library, observed, exclude_genotypes=())
        assert biased["single-1"] != pytest.approx(truth["single-1"] / total, rel=1e-3)

    def test_undesigned_genotypes_reported_not_fit(self, library):
        truth = {"single-1": 1, "single-2": 1, "double-1-2": 2, "spiked": 0.5}
        observed = self._expected_abundance(library, truth)
        design_total = observed.sum()
        observed["A1V/A1G"] = design_total / 9.0   # 10% of the new total
        mixture, _, unexplained = estimate_library_mixture(library, observed)
        assert unexplained == pytest.approx(0.1)
        assert mixture["double-1-2"] == pytest.approx(2 / 4.5, rel=1e-6)

    def test_absent_genotypes_count_as_zero_and_dataframe_input(self, library):
        truth = {"single-1": 1, "single-2": 1, "double-1-2": 2, "spiked": 0.0}
        observed = self._expected_abundance(library, truth)
        observed = observed[observed > 0]   # spike-only genotype missing
        df = observed.rename("abundance").rename_axis("genotype").reset_index()
        mixture, _, _ = estimate_library_mixture(library, df)
        assert mixture["spiked"] == pytest.approx(0.0, abs=1e-12)
        assert mixture["double-1-2"] == pytest.approx(0.5, rel=1e-6)

    def test_validation(self, library):
        good = pd.Series({"A1V": 1.0, "B2V": 1.0})
        with pytest.raises(ValueError, match="weight"):
            estimate_library_mixture(library.drop(columns="weight"), good)
        with pytest.raises(ValueError, match="abundance"):
            estimate_library_mixture(library, pd.DataFrame({"genotype": ["A1V"]}))
        with pytest.raises(ValueError, match="non-negative"):
            estimate_library_mixture(library, pd.Series({"A1V": -1.0}))
        with pytest.raises(ValueError, match="duplicate"):
            estimate_library_mixture(library, pd.Series([1.0, 2.0], index=["A1V", "A1V"]))
        with pytest.raises(ValueError, match="sum to zero"):
            estimate_library_mixture(library, pd.Series({"A1V": 0.0}))
        with pytest.raises(ValueError, match="No observed abundance"):
            estimate_library_mixture(library, pd.Series({"wt": 5.0}))
