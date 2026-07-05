"""
Tests for tfscreen.simulate.presplit_data.generate_presplit_df.

Coverage:
  - Output columns and shape for a simple two-genotype, two-condition case
  - Genotypes absent from the transformation pool (ln_cfu_0 = -inf) → NaN ln_cfu
  - extra_noise parameter inflates ln_cfu_std
  - Output is deterministic given the same rng seed
"""
import numpy as np
import pandas as pd

from tfscreen.simulate.presplit_data import generate_presplit_df


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _make_inputs(
    replicates=(1, 2),
    condition_pres=("kanR", "pheS"),
    genotypes=("wt", "A1V", "A2V"),
    ln_cfu_0_val=10.0,
):
    """Build minimal combined_sample_df and combined_counts_df."""
    sample_rows = []
    counts_rows = []
    sample_id = 0

    for rep in replicates:
        for cp in condition_pres:
            # Two selection samples per (rep, cp) (different t_sel)
            for t_sel in (60.0, 90.0):
                sample_rows.append(
                    {"sample": sample_id, "replicate": rep, "library": "lib",
                     "condition_pre": cp, "t_sel": t_sel,
                     "sample_cfu": 1e8, "sample_cfu_std": 5e6}
                )
                for geno in genotypes:
                    counts_rows.append(
                        {"sample": sample_id, "genotype": geno,
                         "ln_cfu_0": ln_cfu_0_val, "counts": 100}
                    )
                sample_id += 1

    sample_df = pd.DataFrame(sample_rows).set_index("sample")
    counts_df = pd.DataFrame(counts_rows)
    return sample_df, counts_df


def _minimal_cf(noise=0.0, extra_keys=None):
    cf = {
        "cfu0": 1e8,
        "total_num_reads": 30_000_000,
        "prob_index_hop": None,
        "presplit_data": {"noise": noise},
    }
    if extra_keys:
        cf.update(extra_keys)
    return cf


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------

def test_output_columns():
    sample_df, counts_df = _make_inputs()
    result = generate_presplit_df(sample_df, counts_df, _minimal_cf(),
                                   np.random.default_rng(0))
    for col in ["library", "replicate", "condition_pre", "genotype",
                "ln_cfu", "ln_cfu_std", "ln_cfu_0_true"]:
        assert col in result.columns, f"missing column: {col}"


def test_output_row_count():
    """One row per (replicate, condition_pre, genotype)."""
    replicates = (1, 2)
    cps = ("kanR", "pheS")
    genos = ("wt", "A1V", "A2V")
    sample_df, counts_df = _make_inputs(replicates=replicates,
                                         condition_pres=cps,
                                         genotypes=genos)
    result = generate_presplit_df(sample_df, counts_df, _minimal_cf(),
                                   np.random.default_rng(0))
    expected = len(replicates) * len(cps) * len(genos)
    assert len(result) == expected


def test_ln_cfu_is_finite_for_present_genotypes():
    sample_df, counts_df = _make_inputs()
    result = generate_presplit_df(sample_df, counts_df, _minimal_cf(),
                                   np.random.default_rng(0))
    # All genotypes have finite ln_cfu_0 so ln_cfu should be finite
    assert result["ln_cfu"].notna().all()


def test_ln_cfu_std_is_positive():
    sample_df, counts_df = _make_inputs()
    result = generate_presplit_df(sample_df, counts_df, _minimal_cf(),
                                   np.random.default_rng(0))
    assert (result["ln_cfu_std"] > 0).all()


# ---------------------------------------------------------------------------
# Absent genotypes (ln_cfu_0 = -inf)
# ---------------------------------------------------------------------------

def test_absent_genotype_lower_lncfu():
    """A genotype with ln_cfu_0 = -inf (never transformed) gets zero counts
    from the multinomial draw.  After applying the pseudocount it still gets a
    finite (but very low) ln_cfu estimate — the same behaviour as
    counts_to_lncfu — and its ln_cfu is substantially lower than a present
    genotype."""
    sample_df, counts_df = _make_inputs()
    # Set one genotype's ln_cfu_0 to -inf across all samples
    counts_df.loc[counts_df["genotype"] == "A2V", "ln_cfu_0"] = -np.inf
    result = generate_presplit_df(sample_df, counts_df, _minimal_cf(),
                                   np.random.default_rng(0))
    absent_ln  = result.loc[result["genotype"] == "A2V", "ln_cfu"].mean()
    present_ln = result.loc[result["genotype"] == "wt",  "ln_cfu"].mean()
    # Absent genotype should be much lower than a normally-present genotype
    assert absent_ln < present_ln - 5.0


# ---------------------------------------------------------------------------
# extra_noise parameter
# ---------------------------------------------------------------------------

def test_extra_noise_inflates_ln_cfu_std():
    """Adding extra noise should increase the mean ln_cfu_std."""
    sample_df, counts_df = _make_inputs()
    rng0 = np.random.default_rng(42)
    rng1 = np.random.default_rng(42)

    result_no_noise    = generate_presplit_df(sample_df, counts_df,
                                               _minimal_cf(noise=0.0), rng0)
    result_with_noise  = generate_presplit_df(sample_df, counts_df,
                                               _minimal_cf(noise=0.5), rng1)

    mean_std_no_noise   = result_no_noise["ln_cfu_std"].mean()
    mean_std_with_noise = result_with_noise["ln_cfu_std"].mean()
    assert mean_std_with_noise > mean_std_no_noise


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Same rng seed → identical output."""
    sample_df, counts_df = _make_inputs()
    r1 = generate_presplit_df(sample_df.copy(), counts_df.copy(), _minimal_cf(),
                               np.random.default_rng(7))
    r2 = generate_presplit_df(sample_df.copy(), counts_df.copy(), _minimal_cf(),
                               np.random.default_rng(7))
    pd.testing.assert_frame_equal(r1.reset_index(drop=True),
                                   r2.reset_index(drop=True))
