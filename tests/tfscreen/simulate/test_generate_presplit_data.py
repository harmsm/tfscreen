"""
Tests for _generate_presplit_data in run_simulation_cli.

Coverage:
  - Output columns and shape for a simple two-genotype, two-condition case
  - Genotypes absent from the transformation pool (ln_cfu_0 = -inf) → NaN ln_cfu
  - extra_noise parameter inflates ln_cfu_std
  - Output is deterministic given the same rng seed
  - run_simulation_from_config writes a presplit CSV when presplit_data is in cf
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from tfscreen.simulate.scripts.run_simulation_cli import _generate_presplit_data


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
                    {"sample": sample_id, "replicate": rep,
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
        "final_cfu_pct_err": 0.05,
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
    result = _generate_presplit_data(sample_df, counts_df, _minimal_cf(),
                                      np.random.default_rng(0))
    for col in ["replicate", "condition_pre", "genotype", "ln_cfu", "ln_cfu_std",
                "ln_cfu_0_true"]:
        assert col in result.columns, f"missing column: {col}"


def test_output_row_count():
    """One row per (replicate, condition_pre, genotype)."""
    replicates = (1, 2)
    cps = ("kanR", "pheS")
    genos = ("wt", "A1V", "A2V")
    sample_df, counts_df = _make_inputs(replicates=replicates,
                                         condition_pres=cps,
                                         genotypes=genos)
    result = _generate_presplit_data(sample_df, counts_df, _minimal_cf(),
                                      np.random.default_rng(0))
    expected = len(replicates) * len(cps) * len(genos)
    assert len(result) == expected


def test_ln_cfu_is_finite_for_present_genotypes():
    sample_df, counts_df = _make_inputs()
    result = _generate_presplit_data(sample_df, counts_df, _minimal_cf(),
                                      np.random.default_rng(0))
    # All genotypes have finite ln_cfu_0 so ln_cfu should be finite
    assert result["ln_cfu"].notna().all()


def test_ln_cfu_std_is_positive():
    sample_df, counts_df = _make_inputs()
    result = _generate_presplit_data(sample_df, counts_df, _minimal_cf(),
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
    result = _generate_presplit_data(sample_df, counts_df, _minimal_cf(),
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

    result_no_noise    = _generate_presplit_data(sample_df, counts_df,
                                                  _minimal_cf(noise=0.0), rng0)
    result_with_noise  = _generate_presplit_data(sample_df, counts_df,
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
    r1 = _generate_presplit_data(sample_df.copy(), counts_df.copy(), _minimal_cf(),
                                  np.random.default_rng(7))
    r2 = _generate_presplit_data(sample_df.copy(), counts_df.copy(), _minimal_cf(),
                                  np.random.default_rng(7))
    pd.testing.assert_frame_equal(r1.reset_index(drop=True),
                                   r2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Integration: run_simulation_from_config writes presplit CSV
# ---------------------------------------------------------------------------

def test_run_simulation_writes_presplit_csv(tmp_path):
    """presplit CSV is written when presplit_data block is in the config."""
    from tfscreen.simulate.scripts.run_simulation_cli import run_simulation_from_config

    lib_df   = pd.DataFrame({"genotype": ["wt", "A1V"]})
    pheno_df = pd.DataFrame({"genotype": ["wt", "A1V"]})
    theta_df = pd.DataFrame({"genotype": ["wt", "A1V"]})
    params_df = pd.DataFrame({"genotype": ["wt", "A1V"],
                               "dk_geno": [0.0, 0.0], "activity": [1.0, 1.0]})

    # Sample/counts for one (replicate, condition_pre, t_sel) combo
    # Real _simulate_library_group returns sample_df with "sample" as a
    # regular column and an unnamed integer index.
    sample_df = pd.DataFrame([{
        "sample": 0, "replicate": 1, "condition_pre": "kanR",
        "t_sel": 60.0, "sample_cfu": 1e8, "sample_cfu_std": 5e6,
    }], index=[0])
    counts_df = pd.DataFrame([
        {"sample": 0, "genotype": "wt",  "counts": 500, "ln_cfu_0": 10.0},
        {"sample": 0, "genotype": "A1V", "counts": 500, "ln_cfu_0": 10.0},
    ])
    growth_df = pd.DataFrame({"genotype": ["wt", "A1V"], "ln_cfu": [10.0, 9.9]})

    cf = {
        "random_seed": 1,
        "cfu0": 1e8,
        "total_num_reads": 10_000_000,
        "final_cfu_pct_err": 0.05,
        "prob_index_hop": None,
        "presplit_data": {"noise": 0.0},
    }

    with patch("tfscreen.util.read_yaml", return_value=cf), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.counts_to_lncfu",
               return_value=growth_df):
        run_simulation_from_config("fake_config.yaml", str(tmp_path))

    presplit_path = tmp_path / "tfs_sim_presplit.csv"
    assert presplit_path.exists(), "presplit CSV was not written"
    presplit_df = pd.read_csv(presplit_path)
    for col in ["replicate", "condition_pre", "genotype", "ln_cfu", "ln_cfu_std"]:
        assert col in presplit_df.columns


def test_run_simulation_no_presplit_without_config(tmp_path):
    """presplit CSV is NOT written when presplit_data is absent from config."""
    from tfscreen.simulate.scripts.run_simulation_cli import run_simulation_from_config

    lib_df   = pd.DataFrame({"genotype": ["wt"]})
    pheno_df = pd.DataFrame({"genotype": ["wt"]})
    theta_df = pd.DataFrame({"genotype": ["wt"]})
    params_df = pd.DataFrame({"genotype": ["wt"], "dk_geno": [0.0], "activity": [1.0]})
    sample_df = pd.DataFrame([{"sample": 0, "replicate": 1,
                                "condition_pre": "kanR", "t_sel": 60.0,
                                "sample_cfu": 1e8, "sample_cfu_std": 5e6}],
                              index=[0])
    counts_df = pd.DataFrame([{"sample": 0, "genotype": "wt",
                                "counts": 1000, "ln_cfu_0": 10.0}])
    growth_df = pd.DataFrame({"genotype": ["wt"], "ln_cfu": [10.0]})

    cf = {"random_seed": 1, "cfu0": 1e8, "total_num_reads": 1_000_000,
          "final_cfu_pct_err": 0.05, "prob_index_hop": None}

    with patch("tfscreen.util.read_yaml", return_value=cf), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.library_prediction",
               return_value=(lib_df, pheno_df, theta_df, params_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.selection_experiment",
               return_value=(sample_df, counts_df)), \
         patch("tfscreen.simulate.scripts.run_simulation_cli.counts_to_lncfu",
               return_value=growth_df):
        run_simulation_from_config("fake_config.yaml", str(tmp_path))

    presplit_path = tmp_path / "tfs_sim_presplit.csv"
    assert not presplit_path.exists(), "presplit CSV should not be written without config block"
