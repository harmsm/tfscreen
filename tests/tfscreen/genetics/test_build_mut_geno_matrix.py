import pytest
import numpy as np

from tfscreen.genetics import build_mut_geno_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(genotypes):
    return build_mut_geno_matrix(genotypes)


def _coo_to_dense(pair_nnz_pair_idx, pair_nnz_geno_idx, num_pair, num_genotype):
    """Reconstruct dense (num_pair, num_genotype) matrix from COO indices."""
    P = np.zeros((num_pair, num_genotype), dtype=np.float32)
    for p, g in zip(pair_nnz_pair_idx, pair_nnz_geno_idx):
        P[p, g] = 1.0
    return P


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestBuildMutGenoMatrix:

    # --- Return types and basic structure ---

    def test_returns_five_items(self):
        ml, pl, M, pi, gi = _call(["wt"])
        assert isinstance(ml, list)
        assert isinstance(pl, list)
        assert isinstance(M, np.ndarray)
        assert isinstance(pi, np.ndarray)   # pair_nnz_pair_idx
        assert isinstance(gi, np.ndarray)   # pair_nnz_geno_idx

    def test_wt_only_gives_empty_matrices(self):
        ml, pl, M, pi, gi = _call(["wt"])
        assert ml == []
        assert pl == []
        assert M.shape == (0, 1)
        assert pi.shape == (0,)
        assert gi.shape == (0,)

    def test_dtype_float32_and_int32(self):
        _, _, M, pi, gi = _call(["wt", "M42I"])
        assert M.dtype == np.float32
        assert pi.dtype == np.int32
        assert gi.dtype == np.int32

    # --- Single mutants ---

    def test_single_mutant_one_mutation(self):
        ml, pl, M, pi, gi = _call(["wt", "M42I"])
        assert ml == ["M42I"]
        assert pl == []
        assert M.shape == (1, 2)   # (num_mutation, num_genotype)
        assert len(pi) == 0        # no pairs
        assert len(gi) == 0

    def test_single_mutant_wt_column_is_zero(self):
        _, _, M, _, _ = _call(["wt", "M42I"])
        # Column 0 is wt
        assert M[0, 0] == 0.0
        assert M[0, 1] == 1.0

    def test_multiple_singles_column_order(self):
        genotypes = ["wt", "M42I", "K84L"]
        ml, pl, M, pi, gi = _call(genotypes)
        # mut_labels in first-seen order: M42I, K84L
        assert ml == ["M42I", "K84L"]
        assert pl == []
        assert M.shape == (2, 3)
        expected = np.array([[0, 1, 0],   # M42I
                              [0, 0, 1]], dtype=np.float32)  # K84L
        np.testing.assert_array_equal(M, expected)

    # --- Doubles ---

    def test_double_mutant_produces_pair(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        ml, pl, M, pi, gi = _call(genotypes)
        assert len(pl) == 1
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        assert P.shape == (1, 4)

    def test_double_pair_label_is_alphabetically_sorted(self):
        # M42I comes before K84L alphabetically? No: K < M, so sorted gives K84L/M42I
        _, pl, _, _, _ = _call(["wt", "M42I", "K84L", "M42I/K84L"])
        assert pl == ["K84L/M42I"]

    def test_double_pair_matrix_values(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        _, pl, M, pi, gi = _call(genotypes)
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        # Only the double mutant (column 3) has the pair
        expected_P = np.array([[0, 0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(P, expected_P)

    def test_double_mut_geno_matrix_values(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        _, _, M, _, _ = _call(genotypes)
        expected_M = np.array([[0, 1, 0, 1],   # M42I
                               [0, 0, 1, 1]], dtype=np.float32)  # K84L
        np.testing.assert_array_equal(M, expected_M)

    def test_observed_pair_only_pair_in_data_gets_entry(self):
        # M42I/K84L and M42I/V10L are two different doubles
        genotypes = ["wt", "M42I", "K84L", "V10L", "M42I/K84L", "M42I/V10L"]
        _, pl, _, pi, gi = _call(genotypes)
        assert len(pl) == 2
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        assert P.shape == (2, 6)

    # --- Triple mutants ---

    def test_triple_mutant_produces_three_pairs(self):
        genotypes = ["wt", "A1B", "C2D", "E3F", "A1B/C2D/E3F"]
        _, pl, _, pi, gi = _call(genotypes)
        # C(3,2) = 3 pairs
        assert len(pl) == 3
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        assert P.shape == (3, 5)

    def test_triple_mutant_pair_matrix_correct_column(self):
        genotypes = ["wt", "A1B", "C2D", "E3F", "A1B/C2D/E3F"]
        _, pl, _, pi, gi = _call(genotypes)
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        # Triple mutant is column 4; all three pairs should appear there
        triple_col = P[:, 4]
        assert all(triple_col == 1.0), "All pairs must be present in the triple mutant column"

    def test_triple_mutant_singles_do_not_have_pairs(self):
        genotypes = ["wt", "A1B", "C2D", "E3F", "A1B/C2D/E3F"]
        _, pl, _, pi, gi = _call(genotypes)
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        # Columns 0-3 are wt and the three singles — none should have any pair
        for col in range(4):
            assert all(P[:, col] == 0.0), f"Singles/wt col {col} should have no pairs"

    # --- COO-specific invariants ---

    def test_coo_indices_same_length(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        _, _, _, pi, gi = _call(genotypes)
        assert len(pi) == len(gi)

    def test_coo_nnz_equals_total_pair_occurrences(self):
        # One double: 1 pair occurrence; one triple: 3 pair occurrences
        genotypes = ["wt", "A1B", "C2D", "E3F", "A1B/C2D", "A1B/C2D/E3F"]
        _, pl, _, pi, gi = _call(genotypes)
        # A1B/C2D double contributes 1; triple contributes 3
        assert len(pi) == 1 + 3

    def test_skip_pairs_returns_empty_coo(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        ml, pl, M, pi, gi = build_mut_geno_matrix(genotypes, skip_pairs=True)
        assert pl == []
        assert pi.shape == (0,)
        assert gi.shape == (0,)
        # mut_geno_matrix should still be correct
        assert M.shape == (2, 4)

    # --- First-seen ordering of mutation labels ---

    def test_mut_labels_first_seen_order(self):
        # Introduce K84L before M42I by way of a double
        genotypes = ["wt", "K84L/M42I", "M42I"]
        ml, _, _, _, _ = _call(genotypes)
        # K84L is seen first (inside the double), but let's check the actual order
        # "K84L/M42I".split("/") → ["K84L", "M42I"] → K84L first
        assert ml[0] == "K84L"
        assert ml[1] == "M42I"

    # --- wt label handling ---

    def test_wt_label_case_insensitive(self):
        ml, _, M, _, _ = _call(["WT", "M42I"])
        # "WT" should be treated as wildtype (no mutations)
        assert ml == ["M42I"]
        assert M[0, 0] == 0.0   # WT column is 0

    def test_only_doubles_no_singles(self):
        """Library with doubles but no corresponding singles still works."""
        genotypes = ["wt", "A1B/C2D"]
        ml, pl, M, pi, gi = _call(genotypes)
        assert ml == ["A1B", "C2D"]
        assert len(pl) == 1
        assert M.shape == (2, 2)
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        assert P.shape == (1, 2)

    # --- Column and row counts match genotype/mutation counts ---

    def test_num_columns_M_equals_num_genotypes(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        _, pl, M, pi, gi = _call(genotypes)
        assert M.shape[1] == len(genotypes)
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        assert P.shape[1] == len(genotypes)

    def test_num_rows_M_equals_num_unique_mutations(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        ml, _, M, _, _ = _call(genotypes)
        assert M.shape[0] == len(ml) == 2

    def test_num_rows_P_equals_num_unique_pairs(self):
        genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
        _, pl, _, pi, gi = _call(genotypes)
        assert len(pl) == 1
        P = _coo_to_dense(pi, gi, len(pl), len(genotypes))
        assert P.shape[0] == len(pl)
