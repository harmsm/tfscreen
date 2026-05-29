"""
Tests for subset_genotypes_cli.py.

Synthetic data uses genotypes of the form XsiteY (e.g. A1B) to avoid any
dependency on the real tfscreen test data files.  The fixture `growth_df`
contains:
    wt          – 1 genotype, 2 rows
    Singles:    A1B, A1C (site 1)   B2C, B2D (site 2)   C3D, C3E (site 3)
    Doubles:    A1B/B2C, A1B/B2D, A1C/B2C, A1C/B2D   (site-1 × site-2)
                A1B/C3D, A1C/C3D                        (site-1 × site-3)
                B2C/C3D                                 (site-2 × site-3)
"""

import os

import pandas as pd
import pytest

from tfscreen.growth_model.scripts.subset_genotypes_cli import (
    _mutations_in_genotype,
    _reconcile_doubles_and_singles,
    subset_genotypes,
)


# ---------------------------------------------------------------------------
# Second synthetic dataset for whitelist-seeding tests
# ---------------------------------------------------------------------------
# W1B is the whitelist single.  Its only double partner is W1B/B2C.
# B2D has no double partner at all (no double contains B2D).
# The expansion loop would never naturally add W1B/B2C because W1B is not
# in the reconciled doubles that the loop stumbles across — unless seeding
# explicitly finds it.
WHITELIST_SINGLES_SEED = ["W1B"]
WHITELIST_DOUBLES_SEED = ["W1B/B2C"]  # only double involving W1B


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_growth_df(genotypes, rows_per_geno=2):
    """Return a minimal DataFrame with a 'genotype' column and dummy data."""
    rows = []
    for g in genotypes:
        for _ in range(rows_per_geno):
            rows.append({"genotype": g, "ln_cfu": 10.0, "replicate": 1})
    return pd.DataFrame(rows)


ALL_SINGLES = ["A1B", "A1C", "B2C", "B2D", "C3D", "C3E"]
DOUBLES_SITE12 = ["A1B/B2C", "A1B/B2D", "A1C/B2C", "A1C/B2D"]
DOUBLES_SITE13 = ["A1B/C3D", "A1C/C3D"]
DOUBLES_SITE23 = ["B2C/C3D"]
ALL_DOUBLES = DOUBLES_SITE12 + DOUBLES_SITE13 + DOUBLES_SITE23


@pytest.fixture
def growth_csv(tmp_path):
    """Write the synthetic growth DataFrame to a CSV and return its path."""
    df = _build_growth_df(["wt"] + ALL_SINGLES + ALL_DOUBLES)
    path = tmp_path / "growth.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def whitelist_file(tmp_path):
    """A whitelist containing the two site-1 singles."""
    p = tmp_path / "whitelist.txt"
    p.write_text("A1B\nA1C\n")
    return str(p)


@pytest.fixture
def blacklist_file(tmp_path):
    p = tmp_path / "blacklist.txt"
    p.write_text("C3D\nC3E\n")
    return str(p)


def _read_leftout(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip() and not l.startswith("#")]


def _output_pairs(out_dir):
    """Return sorted list of (growth_csv, leftout_txt) path pairs."""
    files = os.listdir(out_dir)
    growth = sorted(f for f in files if f.endswith("_growth.csv"))
    leftout = sorted(f for f in files if f.endswith("_leftout.txt"))
    assert len(growth) == len(leftout)
    return list(zip(
        [os.path.join(out_dir, f) for f in growth],
        [os.path.join(out_dir, f) for f in leftout],
    ))


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_mutations_in_genotype_wt():
    assert _mutations_in_genotype("wt") == frozenset()


def test_mutations_in_genotype_single():
    assert _mutations_in_genotype("A1B") == frozenset({"A1B"})


def test_mutations_in_genotype_double():
    assert _mutations_in_genotype("A1B/B2C") == frozenset({"A1B", "B2C"})


# ---------------------------------------------------------------------------
# Unit tests for _reconcile_doubles_and_singles
# ---------------------------------------------------------------------------

def test_reconcile_keeps_complete_cycles():
    """Doubles whose both singles are present are kept."""
    singles = {"A1B", "B2C"}
    doubles = {"A1B/B2C"}
    kept, pool = _reconcile_doubles_and_singles(doubles, singles, blacklist=set())
    assert set(kept) == {"A1B/B2C"}
    assert pool == {"A1B", "B2C"}


def test_reconcile_drops_double_with_missing_single():
    """Doubles with a missing constituent single are dropped."""
    singles = {"A1B"}          # B2C is absent
    doubles = {"A1B/B2C"}
    kept, pool = _reconcile_doubles_and_singles(doubles, singles, blacklist=set())
    assert kept == []
    assert pool == set()


def test_reconcile_drops_double_with_blacklisted_single():
    """Doubles containing a blacklisted single are dropped."""
    singles = {"A1B", "B2C"}
    doubles = {"A1B/B2C"}
    kept, pool = _reconcile_doubles_and_singles(doubles, singles, blacklist={"B2C"})
    assert kept == []
    assert pool == set()


def test_reconcile_mixed_doubles():
    """Only doubles with both singles present and not blacklisted are kept."""
    singles = {"A1B", "A1C", "B2C"}   # B2D absent; C3D absent
    doubles = {"A1B/B2C", "A1C/B2C", "A1B/B2D", "B2C/C3D"}
    kept, pool = _reconcile_doubles_and_singles(doubles, singles, blacklist=set())
    assert set(kept) == {"A1B/B2C", "A1C/B2C"}
    assert pool == {"A1B", "A1C", "B2C"}


def test_reconcile_singles_pool_excludes_isolated_singles():
    """Singles that don't participate in any kept double are not in the pool."""
    singles = {"A1B", "A1C", "Z9Z"}   # Z9Z has no double partner
    doubles = {"A1B/A1C"}
    kept, pool = _reconcile_doubles_and_singles(doubles, singles, blacklist=set())
    assert "Z9Z" not in pool


# ---------------------------------------------------------------------------
# Core output-pair tests
# ---------------------------------------------------------------------------

def test_step_count_and_filenames(growth_csv, tmp_path):
    """n_steps output pairs are generated (or fewer if double universe is small)."""
    out = str(tmp_path / "out" / "cv")
    os.makedirs(os.path.dirname(out))
    subset_genotypes(growth_csv, n_singles=6, n_steps=5, out_prefix=out, random_seed=0)
    pairs = _output_pairs(os.path.dirname(out))
    # 7 doubles total, 5 linspace steps → 5 unique counts (0,1,3,5,7) → 5 pairs
    assert len(pairs) == 5


def test_filenames_encode_counts(growth_csv, tmp_path):
    """File names contain the correct singles/doubles counts."""
    out = str(tmp_path / "cv")
    subset_genotypes(growth_csv, n_singles=6, n_steps=3, out_prefix=out, random_seed=0)
    files = sorted(os.listdir(tmp_path))
    growth_files = [f for f in files if "_growth.csv" in f]
    # Each growth filename must contain hyphenated singles and doubles counts
    for f in growth_files:
        assert "-singles_" in f
        assert "-doubles_" in f
    # The last step should include all 7 doubles
    last = sorted(growth_files)[-1]
    assert "7-doubles" in last


def test_wt_always_in_training(growth_csv, tmp_path):
    """wt genotype appears in every training CSV."""
    out = str(tmp_path / "cv")
    subset_genotypes(growth_csv, n_singles=4, n_steps=4, out_prefix=out, random_seed=1)
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        assert "wt" in df["genotype"].values


def test_first_step_has_zero_doubles(growth_csv, tmp_path):
    """The first output pair always has 0 doubles in the training set."""
    out = str(tmp_path / "cv")
    subset_genotypes(growth_csv, n_singles=6, n_steps=4, out_prefix=out, random_seed=0)
    pairs = _output_pairs(tmp_path)
    first_csv, _ = pairs[0]
    df = pd.read_csv(first_csv)
    assert all("/" not in g for g in df["genotype"].unique())


def test_last_step_has_all_doubles(growth_csv, tmp_path):
    """The last step includes all constructible doubles; left-out file is empty."""
    out = str(tmp_path / "cv")
    subset_genotypes(growth_csv, n_singles=6, n_steps=4, out_prefix=out, random_seed=0)
    pairs = _output_pairs(tmp_path)
    last_csv, last_lo = pairs[-1]
    df = pd.read_csv(last_csv)
    doubles_in_training = [g for g in df["genotype"].unique() if "/" in g]
    assert len(doubles_in_training) == len(ALL_DOUBLES)
    assert _read_leftout(last_lo) == []


def test_training_plus_leftout_equals_universe(growth_csv, tmp_path):
    """For every step, training doubles + left-out doubles = the full double universe."""
    out = str(tmp_path / "cv")
    subset_genotypes(growth_csv, n_singles=6, n_steps=5, out_prefix=out, random_seed=0)
    for gcsv, lout in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        train_doubles = set(g for g in df["genotype"].unique() if "/" in g)
        leftout = set(_read_leftout(lout))
        assert train_doubles & leftout == set(), "training and left-out overlap"
        assert train_doubles | leftout == set(ALL_DOUBLES)


def test_only_constructible_doubles_in_training(growth_csv, tmp_path):
    """Doubles in the training CSV can only come from the selected singles."""
    out = str(tmp_path / "cv")
    subset_genotypes(growth_csv, n_singles=4, n_steps=3, out_prefix=out, random_seed=7)
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        singles = set(g for g in df["genotype"].unique() if "/" not in g and g != "wt")
        single_muts = set()
        for s in singles:
            single_muts.add(s)
        for g in df["genotype"].unique():
            if "/" in g:
                parts = set(g.split("/"))
                assert parts <= single_muts, f"{g} uses mutations not in selected singles"


# ---------------------------------------------------------------------------
# Whitelist / blacklist
# ---------------------------------------------------------------------------

def test_whitelist_singles_always_selected(growth_csv, whitelist_file, tmp_path):
    """Whitelisted singles appear in every training CSV."""
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=4, n_steps=3,
        out_prefix=out, whitelist_file=whitelist_file, random_seed=0,
    )
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "A1B" in genos
        assert "A1C" in genos


def test_blacklist_singles_never_selected(growth_csv, blacklist_file, tmp_path):
    """Blacklisted genotypes never appear in any training CSV."""
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=4, n_steps=3,
        out_prefix=out, blacklist_file=blacklist_file, random_seed=0,
    )
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "C3D" not in genos
        assert "C3E" not in genos


def test_blacklist_removes_dependent_doubles(growth_csv, blacklist_file, tmp_path):
    """Doubles that require a blacklisted single are excluded from the universe."""
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=4, n_steps=3,
        out_prefix=out, blacklist_file=blacklist_file, random_seed=0,
    )
    for gcsv, lout in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        all_genos = set(df["genotype"].unique()) | set(_read_leftout(lout))
        # Any double involving C3D or C3E should not appear at all
        for g in all_genos:
            if "/" in g:
                assert "C3D" not in g.split("/")
                assert "C3E" not in g.split("/")


# ---------------------------------------------------------------------------
# Reconciliation integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def growth_csv_orphan_doubles(tmp_path):
    """
    Dataset where some doubles are missing one constituent single.

    Present singles: A1B, B2C only.
    Doubles in data:
        A1B/B2C   — complete cycle (both singles present)  → kept
        A1C/B2C   — A1C absent                             → dropped
        A1B/C3D   — C3D absent                             → dropped
    """
    genotypes = ["wt", "A1B", "B2C", "A1B/B2C", "A1C/B2C", "A1B/C3D"]
    df = _build_growth_df(genotypes)
    path = tmp_path / "orphan_growth.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_reconciliation_drops_orphan_doubles(growth_csv_orphan_doubles, tmp_path):
    """Doubles with a missing single never appear in any output file."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = str(out_dir / "cv")
    subset_genotypes(
        growth_csv_orphan_doubles, n_singles=2, n_steps=3, out_prefix=out, random_seed=0
    )
    for gcsv, lout in _output_pairs(out_dir):
        all_genos = (
            set(pd.read_csv(gcsv)["genotype"].unique()) | set(_read_leftout(lout))
        )
        assert "A1C/B2C" not in all_genos
        assert "A1B/C3D" not in all_genos


def test_reconciliation_singles_pool_derived_from_doubles(
    growth_csv_orphan_doubles, tmp_path
):
    """
    The singles pool is derived from the reconciled doubles, so a single that
    only appears in a dropped double (here A1C, C3D) is never selected.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = str(out_dir / "cv")
    subset_genotypes(
        growth_csv_orphan_doubles, n_singles=2, n_steps=3, out_prefix=out, random_seed=0
    )
    for gcsv, _ in _output_pairs(out_dir):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "A1C" not in genos
        assert "C3D" not in genos


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_step_deduplication_when_few_doubles(growth_csv, tmp_path):
    """When n_steps > double universe size, output is deduplicated to unique counts."""
    out = str(tmp_path / "cv")
    # 7 doubles, request 20 steps — should collapse to at most 8 unique steps (0..7)
    subset_genotypes(growth_csv, n_singles=6, n_steps=20, out_prefix=out, random_seed=0)
    pairs = _output_pairs(tmp_path)
    assert len(pairs) <= 8


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_error_whitelist_single_not_in_data(growth_csv, tmp_path):
    """Raises if a whitelisted single is absent from the growth data."""
    wl = tmp_path / "wl.txt"
    wl.write_text("NOTREAL\n")
    with pytest.raises(ValueError, match="Whitelist singles not present"):
        subset_genotypes(
            growth_csv, n_singles=3, n_steps=2,
            out_prefix=str(tmp_path / "cv"), whitelist_file=str(wl),
        )


def test_error_whitelist_double_not_in_data(growth_csv, tmp_path):
    """Raises if a whitelisted double is absent from the growth data."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B/Z9Z\n")  # Z9Z is not in data
    with pytest.raises(ValueError, match="Whitelist doubles not present"):
        subset_genotypes(
            growth_csv, n_singles=3, n_steps=2,
            out_prefix=str(tmp_path / "cv"), whitelist_file=str(wl),
        )


def test_error_whitelist_too_many_mutations(growth_csv, tmp_path):
    """Raises if a whitelisted genotype has more than 2 mutations."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B/B2C/C3D\n")
    with pytest.raises(ValueError, match="3 mutations"):
        subset_genotypes(
            growth_csv, n_singles=3, n_steps=2,
            out_prefix=str(tmp_path / "cv"), whitelist_file=str(wl),
        )


def test_error_whitelist_double_constituent_single_blacklisted(growth_csv, tmp_path):
    """Raises if a constituent single of a whitelisted double is blacklisted."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B/B2C\n")
    bl = tmp_path / "bl.txt"
    bl.write_text("B2C\n")
    with pytest.raises(ValueError, match="require singles that are blacklisted"):
        subset_genotypes(
            growth_csv, n_singles=3, n_steps=2,
            out_prefix=str(tmp_path / "cv"),
            whitelist_file=str(wl), blacklist_file=str(bl),
        )


def test_error_whitelist_blacklist_overlap(growth_csv, tmp_path):
    """Raises if a genotype appears in both whitelist and blacklist."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B\n")
    bl = tmp_path / "bl.txt"
    bl.write_text("A1B\n")
    with pytest.raises(ValueError, match="both whitelist and blacklist"):
        subset_genotypes(
            growth_csv, n_singles=3, n_steps=2,
            out_prefix=str(tmp_path / "cv"),
            whitelist_file=str(wl), blacklist_file=str(bl),
        )


def test_whitelist_exceeds_n_singles_uses_whitelist_size(growth_csv, tmp_path):
    """When whitelist > n_singles, the effective target is len(whitelist), not n_singles."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B\nA1C\nB2C\n")  # 3 whitelist singles
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=2, n_steps=2,
        out_prefix=out, whitelist_file=str(wl),
    )
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "A1B" in genos
        assert "A1C" in genos
        assert "B2C" in genos


def test_trim_respects_n_singles(growth_csv, tmp_path):
    """
    When seeding balloons selected_singles above n_singles, the trim phase
    brings it back to exactly n_singles (or whitelist size if larger).

    The full synthetic dataset has 6 singles.  With n_singles=2 and no
    whitelist, the expansion loop adds one double (2 singles) and stops.
    With n_singles=2 and whitelist=[A1B] whose seeding pulls in additional
    partners, the trim must cut back to exactly 2.
    """
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B\n")  # A1B pairs with B2C, B2D, C3D via doubles
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=2, n_steps=3,
        out_prefix=out, whitelist_file=str(wl), random_seed=0,
    )
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        singles = [g for g in df["genotype"].unique() if "/" not in g and g != "wt"]
        assert len(singles) == 2
        assert "A1B" in singles  # whitelist single always present


def test_wt_in_whitelist_is_silently_ignored(growth_csv, tmp_path):
    """wt in the whitelist does not cause an error and is always in training."""
    wl = tmp_path / "wl.txt"
    wl.write_text("wt\nA1B\n")
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=3, n_steps=2,
        out_prefix=out, whitelist_file=str(wl),
    )
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "wt" in genos
        assert "A1B" in genos


def test_whitelist_cycles_seeded_before_expansion(tmp_path):
    """
    Whitelist singles and all their double partners are always in the
    training set, even when the random expansion loop would not have
    reached them.

    Dataset: W1B and B2C are linked by the double W1B/B2C.  A1B/A1C is
    a separate pair.  With n_singles=2 and whitelist=[W1B], seeding must
    add B2C (the partner of W1B) before the expansion loop runs — so both
    W1B and B2C appear in every output.
    """
    genotypes = ["wt", "W1B", "B2C", "A1B", "A1C", "W1B/B2C", "A1B/A1C"]
    df = _build_growth_df(genotypes)
    csv_path = tmp_path / "g.csv"
    df.to_csv(csv_path, index=False)

    wl = tmp_path / "wl.txt"
    wl.write_text("W1B\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    subset_genotypes(
        str(csv_path), n_singles=2, n_steps=3,
        out_prefix=str(out_dir / "cv"),
        whitelist_file=str(wl), random_seed=0,
    )
    for gcsv, _ in _output_pairs(out_dir):
        df_out = pd.read_csv(gcsv)
        genos = set(df_out["genotype"].unique())
        assert "W1B" in genos, "whitelist single missing"
        assert "B2C" in genos, "whitelist single's double partner missing"


def test_whitelist_double_always_in_training_never_leftout(growth_csv, tmp_path):
    """A whitelisted double appears in every training CSV and never in any left-out file."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B/B2C\n")
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=4, n_steps=4,
        out_prefix=out, whitelist_file=str(wl), random_seed=0,
    )
    for gcsv, lout in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        assert "A1B/B2C" in df["genotype"].values, "whitelist double missing from training"
        assert "A1B/B2C" not in _read_leftout(lout), "whitelist double appeared in left-out"


def test_whitelist_double_in_first_step(growth_csv, tmp_path):
    """Whitelisted doubles appear even in step 0 (zero titrated doubles)."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B/B2C\n")
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=4, n_steps=4,
        out_prefix=out, whitelist_file=str(wl), random_seed=0,
    )
    first_csv, _ = _output_pairs(tmp_path)[0]
    df = pd.read_csv(first_csv)
    assert "A1B/B2C" in df["genotype"].values


def test_whitelist_double_constituent_singles_forced(growth_csv, tmp_path):
    """Constituent singles of a whitelisted double are always selected."""
    wl = tmp_path / "wl.txt"
    wl.write_text("A1B/B2C\n")
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=2, n_steps=3,
        out_prefix=out, whitelist_file=str(wl), random_seed=0,
    )
    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "A1B" in genos, "constituent single A1B missing"
        assert "B2C" in genos, "constituent single B2C missing"


def test_whitelist_double_excluded_from_titration_universe(growth_csv, tmp_path):
    """
    The titrated double universe excludes whitelisted doubles.  So for every
    step, training_doubles ∪ leftout == ALL_DOUBLES − whitelist_doubles.
    """
    wl_double = "A1B/B2C"
    wl = tmp_path / "wl.txt"
    wl.write_text(f"{wl_double}\n")
    out = str(tmp_path / "cv")
    subset_genotypes(
        growth_csv, n_singles=6, n_steps=4,
        out_prefix=out, whitelist_file=str(wl), random_seed=0,
    )
    expected_universe = set(ALL_DOUBLES) - {wl_double}
    for gcsv, lout in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        train_doubles = set(g for g in df["genotype"].unique() if "/" in g and g != wl_double)
        leftout = set(_read_leftout(lout))
        assert train_doubles | leftout == expected_universe


def test_error_missing_genotype_column(tmp_path):
    """Raises if the growth CSV has no 'genotype' column."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ln_cfu": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="no 'genotype' column"):
        subset_genotypes(str(bad), n_singles=2, n_steps=2,
                           out_prefix=str(tmp_path / "cv"))


# ---------------------------------------------------------------------------
# Smoke test on real test data
# ---------------------------------------------------------------------------

_REAL_GROWTH = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../notebooks/create-test-data/growth.csv",
)


@pytest.mark.slow
def test_smoke_real_data(tmp_path):
    """
    Runs subset_growth_data on the real test growth CSV end-to-end.

    The real data has 965 singles but only 3 doubles (M42I/H74A, M42I/K84L,
    H74A/K84L).  Reconciliation should automatically restrict the singles pool
    to {M42I, H74A, K84L} — no whitelist required.
    """
    real_path = os.path.normpath(_REAL_GROWTH)
    if not os.path.exists(real_path):
        pytest.skip("Real growth CSV not found")

    out = str(tmp_path / "smoke")
    subset_genotypes(real_path, n_singles=6, n_steps=5, out_prefix=out, random_seed=42)

    pairs = _output_pairs(tmp_path)
    assert len(pairs) >= 1

    # Every training CSV must contain wt
    for gcsv, _ in pairs:
        df = pd.read_csv(gcsv)
        assert "wt" in df["genotype"].values

    # Last step: all 3 doubles in training, empty left-out
    last_csv, last_lo = pairs[-1]
    df = pd.read_csv(last_csv)
    doubles_in_last = [g for g in df["genotype"].unique() if "/" in g]
    assert len(doubles_in_last) == 3
    assert _read_leftout(last_lo) == []


@pytest.mark.slow
def test_smoke_real_data_whitelist(tmp_path):
    """
    Whitelist forces specific singles to be included even when not required
    by reconciliation.  Uses the real test growth CSV.
    """
    real_path = os.path.normpath(_REAL_GROWTH)
    if not os.path.exists(real_path):
        pytest.skip("Real growth CSV not found")

    wl = tmp_path / "wl.txt"
    wl.write_text("M42I\nH74A\nK84L\n")

    out = str(tmp_path / "smoke_wl")
    subset_genotypes(
        real_path, n_singles=3, n_steps=4,
        out_prefix=out, whitelist_file=str(wl), random_seed=0,
    )

    for gcsv, _ in _output_pairs(tmp_path):
        df = pd.read_csv(gcsv)
        genos = set(df["genotype"].unique())
        assert "M42I" in genos
        assert "H74A" in genos
        assert "K84L" in genos
