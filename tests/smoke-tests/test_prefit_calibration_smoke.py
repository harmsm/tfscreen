"""
Smoke tests for tfs-prefit-calibration writing per-condition growth priors.

These run a genuine (tiny) configure -> prefit for each condition_growth
component and assert that the pre-fit rewrites the baseline term's prior as
per-condition *indexed* rows (labeled by condition_rep) which then load back
as a per-condition array.  This is the end-to-end check that the generic
per-condition-prior path works uniformly across linear / power / saturation
(not just the linear component it was first built for).
"""

import os

import numpy as np
import pandas as pd
import pytest

from tfscreen.tfmodel.scripts.configure_model_cli import configure_model
from tfscreen.tfmodel.scripts.prefit_calibration_cli import run_prefit_calibration
from tfscreen.tfmodel.scripts.fit_model_cli import fit_model
from tfscreen.tfmodel.configuration_io import read_configuration


def _write_inputs(tmpdir):
    """Two genotypes x three conditions x two titrant concs x two timepoints."""
    genotypes = ["wt", "A123B"]
    conc = [0.1, 1.0]
    # condition_pre is a control ('-'); two distinct selection ('+') conditions
    # give three distinct condition_reps: C1-, C2+, C3+.
    sel_conditions = ["C2+", "C3+"]

    rows = []
    for g in genotypes:
        for cs in sel_conditions:
            for c in conc:
                for t_sel in [0.0, 10.0]:
                    rows.append({
                        "library": "lib",
                        "replicate": "R1",
                        "condition_pre": "C1-",
                        "condition_sel": cs,
                        "genotype": g,
                        "t_sel": t_sel,
                        "t_pre": 12.0,
                        # mild genotype/condition-dependent slope so the MAP has signal
                        "ln_cfu": 1.0 + 0.05 * t_sel + (0.2 if g == "A123B" else 0.0),
                        "ln_cfu_std": 0.1,
                        "titrant_name": "T1",
                        "titrant_conc": c,
                    })
    growth_df = pd.DataFrame(rows)

    binding_df = pd.DataFrame({
        "genotype": ["wt", "wt", "A123B", "A123B"],
        "titrant_name": ["T1"] * 4,
        "titrant_conc": [0.1, 1.0, 0.1, 1.0],
        "theta_obs": [0.8, 0.3, 0.7, 0.2],
        "theta_std": [0.05, 0.05, 0.05, 0.05],
    })

    growth_path = os.path.join(tmpdir, "growth.csv")
    binding_path = os.path.join(tmpdir, "binding.csv")
    growth_df.to_csv(growth_path, index=False)
    binding_df.to_csv(binding_path, index=False)
    return growth_path, binding_path


@pytest.mark.slow
@pytest.mark.parametrize("cg_model,baseline_loc", [
    ("linear", "k_loc"),
    ("power", "k_loc"),
    ("saturation", "min_loc"),
])
def test_prefit_writes_per_condition_baseline_prior(tmpdir, cg_model, baseline_loc):
    growth_path, binding_path = _write_inputs(tmpdir)
    out_prefix = os.path.join(tmpdir, f"tfs_{cg_model}")

    configure_model(binding_path,
                    growth_df=growth_path,
                    condition_growth_model=cg_model,
                    theta_model="categorical_geno",
                    out_prefix=out_prefix)

    config_file = f"{out_prefix}_config.yaml"
    priors_file = f"{out_prefix}_priors.csv"

    # Before the prefit the baseline prior is a single scalar row.
    pre = pd.read_csv(priors_file)
    baseline_param = f"growth.condition_growth.{baseline_loc}"
    assert (pre["parameter"] == baseline_param).sum() == 1
    assert "flat_index" not in pre.columns or pre["flat_index"].isna().all()

    run_prefit_calibration(config_file=config_file,
                           seed=1,
                           max_num_epochs=3,
                           convergence_check_interval=1,
                           patience=1,
                           hessian_chunk_size=8)

    # After the prefit the baseline prior is per-condition indexed rows.
    post = pd.read_csv(priors_file)
    base_rows = post[post["parameter"] == baseline_param]
    assert len(base_rows) >= 2, "baseline prior should be expanded per condition"
    assert base_rows["flat_index"].notna().all()
    assert "condition_rep" in post.columns
    assert base_rows["condition_rep"].notna().all()
    # One row per distinct condition
    assert base_rows["condition_rep"].nunique() == len(base_rows)

    # And it loads back as a per-condition array on the ModelPriors.
    orch, _ = read_configuration(config_file)
    cg = orch.priors.growth.condition_growth
    loaded = np.asarray(getattr(cg, baseline_loc))
    assert loaded.shape == (len(base_rows),)


@pytest.mark.slow
def test_configure_prefit_fit_full_loop_with_per_condition_priors(tmpdir):
    """The full configure -> prefit -> fit loop runs once the priors CSV holds
    per-condition (array) baseline priors — i.e. the SVI actually consumes
    them without a tracing/shape error."""
    growth_path, binding_path = _write_inputs(tmpdir)
    out_prefix = os.path.join(tmpdir, "tfs_full")

    configure_model(binding_path,
                    growth_df=growth_path,
                    condition_growth_model="linear",
                    theta_model="categorical_geno",
                    out_prefix=out_prefix)
    config_file = f"{out_prefix}_config.yaml"

    run_prefit_calibration(config_file=config_file, seed=1,
                           max_num_epochs=3, convergence_check_interval=1,
                           patience=1, hessian_chunk_size=8)

    # Sanity: the priors CSV now carries per-condition k_loc rows.
    post = pd.read_csv(f"{out_prefix}_priors.csv")
    assert (post["parameter"] == "growth.condition_growth.k_loc").sum() >= 2

    fit_out = os.path.join(tmpdir, "tfs_full_fit")
    fit_model(config_file=config_file, seed=42, max_num_epochs=1,
              out_prefix=fit_out)
    assert os.path.exists(f"{fit_out}_checkpoint.pkl")
