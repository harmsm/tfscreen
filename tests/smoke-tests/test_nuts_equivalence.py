"""
Tier 3 statistical equivalence test: Pyro NUTS vs NumPyro NUTS reference.

This test is slow (NUTS on 9 genotypes, 500+500 samples) and is skipped by
default.  Run with:

    pytest tests/smoke-tests/test_nuts_equivalence.py --runnuts

Prerequisites
-------------
The reference file tests/fixtures/nuts_gold_standard_reference.json must
exist.  Generate it once with (NumPyro environment):

    NUMBA_DISABLE_JIT=1 python tests/fixtures/generate_nuts_reference.py

Equivalence criteria
--------------------
For each extracted parameter (activity, dk_geno, theta_theta_low,
theta_theta_high, theta_log_hill_K, theta_hill_n):

  1. **Coverage**: ≥ 75% of entries have the NumPyro reference median
     inside the Pyro 95% CI.
  2. **Sign agreement** (dk_geno, activity): ≥ 80% of entries agree on sign
     with the reference median.
  3. **Divergence rate**: < 5% of total NUTS samples are divergences.

These tolerances are deliberately loose: NUTS with 500 samples has
non-trivial Monte Carlo variance, and the two frameworks use different
implementations of the leapfrog integrator and step-size adaptation.
"""

import json
import os
import pytest
import numpy as np
import torch
import pyro
import pandas as pd

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(REPO_ROOT, "tests", "fixtures")
REFERENCE    = os.path.join(REPO_ROOT, "tests", "fixtures",
                             "nuts_gold_standard_reference.json")
GROWTH_CSV   = os.path.join(DATA_DIR, "growth_gold-standared.csv")
BINDING_CSV  = os.path.join(DATA_DIR, "binding_gold-standared.csv")

NUTS_SEED       = 1243   # different from reference seed to probe variance
NUTS_WARMUP     = 500
NUTS_SAMPLES    = 500
NUTS_CHAINS     = 1
MAX_DIVERGENCE_FRAC = 0.05
MIN_COVERAGE_FRAC   = 0.75
MIN_SIGN_AGREE_FRAC = 0.80


@pytest.fixture(scope="module")
def reference():
    if not os.path.exists(REFERENCE):
        pytest.skip(
            f"Reference file not found: {REFERENCE}\n"
            "Generate it with: NUMBA_DISABLE_JIT=1 python "
            "tests/fixtures/generate_nuts_reference.py"
        )
    with open(REFERENCE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pyro_params(reference):
    """Run Pyro NUTS on gold-standard data and return extracted parameters."""
    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    from tfscreen.analysis.hierarchical.run_inference import RunInference
    from tfscreen.analysis.hierarchical.growth_model.extraction import extract_parameters
    from pyro.infer import Predictive

    pyro.clear_param_store()

    growth_df  = pd.read_csv(GROWTH_CSV)
    binding_df = pd.read_csv(BINDING_CSV)

    # Use the same model configuration as the reference
    cfg = reference["metadata"]["model_config"]
    model = ModelClass(
        growth_df=growth_df,
        binding_df=binding_df,
        condition_growth=cfg["condition_growth"],
        dk_geno=cfg["dk_geno"],
        activity=cfg["activity"],
        theta=cfg["theta"],
        transformation=cfg["transformation"],
        theta_growth_noise=cfg["theta_growth_noise"],
        theta_binding_noise=cfg["theta_binding_noise"],
    )

    ri = RunInference(model=model, seed=NUTS_SEED)

    mcmc = ri.run_nuts(
        num_warmup=NUTS_WARMUP,
        num_samples=NUTS_SAMPLES,
        num_chains=NUTS_CHAINS,
        jit_compile=True,
    )

    # Count divergences
    if hasattr(mcmc, "_diagnostics") and mcmc._diagnostics:
        n_div = sum(
            len(mcmc._diagnostics[i].get("divergences", []))
            for i in range(NUTS_CHAINS)
        )
    else:
        n_div = 0
    mcmc._tfscreen_num_divergences = n_div

    # MCMC.get_samples() returns only latent (sample) sites, not
    # pyro.deterministic sites like `activity` or `dk_geno`.
    # Run a forward Predictive pass to compute those deterministic quantities.
    latent_samples = mcmc.get_samples()

    all_indices = torch.arange(model.data.num_genotype, dtype=torch.long)
    full_data = model.get_batch(model.data, all_indices)

    # No return_sites restriction — Predictive returns all non-param sites,
    # including every pyro.deterministic needed by extract_parameters
    # (activity, dk_geno, condition_growth_m/k, ln_cfu0, transformation_lam, etc.)
    predictive = Predictive(model.pyro_model,
                            posterior_samples=latent_samples)
    det_samples = predictive(priors=model.priors, data=full_data)

    # Merge; extract_parameters reshapes via (num_samples, -1) so extra
    # singleton plate dims from Predictive broadcasting are harmless.
    full_posteriors = {}
    for k, v in latent_samples.items():
        full_posteriors[k] = np.asarray(
            v.detach().cpu() if isinstance(v, torch.Tensor) else v)
    for k, v in det_samples.items():
        full_posteriors[k] = np.asarray(
            v.detach().cpu() if isinstance(v, torch.Tensor) else v)

    params = extract_parameters(
        model, full_posteriors,
        q_to_get={"median": 0.5, "lower_95": 0.025, "upper_95": 0.975}
    )

    return mcmc, params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.nuts
def test_nuts_divergence_rate(pyro_params):
    """Divergence fraction must be below threshold."""
    mcmc, _ = pyro_params
    n_div   = mcmc._tfscreen_num_divergences
    total   = NUTS_SAMPLES * NUTS_CHAINS
    frac    = n_div / total
    assert frac < MAX_DIVERGENCE_FRAC, (
        f"Too many NUTS divergences: {n_div}/{total} = {frac:.1%} "
        f"(threshold {MAX_DIVERGENCE_FRAC:.0%})"
    )


@pytest.mark.nuts
@pytest.mark.parametrize("param_name", [
    "activity",
    "dk_geno",
    "theta_theta_low",
    "theta_theta_high",
    "theta_log_hill_K",
    "theta_hill_n",
])
def test_nuts_coverage(pyro_params, reference, param_name):
    """NumPyro reference median should lie inside Pyro 95% CI for ≥ 75% of entries."""
    _, pyro_params_dict = pyro_params

    if param_name not in reference["parameters"]:
        pytest.skip(f"{param_name} not in reference")
    if param_name not in pyro_params_dict:
        pytest.skip(f"{param_name} not in Pyro posteriors")

    ref_entries  = reference["parameters"][param_name]
    pyro_df      = pyro_params_dict[param_name]

    # Build a lookup from key → (lower_95, upper_95) for Pyro
    def _make_key(row):
        if "titrant_name" in pyro_df.columns:
            return f"{row['genotype']}|{row['titrant_name']}"
        return str(row["genotype"])

    pyro_lookup = {}
    for _, row in pyro_df.iterrows():
        pyro_lookup[_make_key(row)] = (row["lower_95"], row["upper_95"])

    n_covered = 0
    n_compared = 0
    misses = []
    for key, ref_vals in ref_entries.items():
        if key not in pyro_lookup:
            continue
        ref_med          = ref_vals["median"]
        pyro_lo, pyro_hi = pyro_lookup[key]
        n_compared += 1
        if pyro_lo <= ref_med <= pyro_hi:
            n_covered += 1
        else:
            misses.append((key, ref_med, pyro_lo, pyro_hi))

    if n_compared == 0:
        pytest.skip(f"No common keys between reference and Pyro for {param_name}")

    coverage = n_covered / n_compared
    miss_detail = "\n  ".join(
        f"{k}: ref_med={rm:.4f}  pyro_CI=[{lo:.4f}, {hi:.4f}]"
        for k, rm, lo, hi in misses
    )
    assert coverage >= MIN_COVERAGE_FRAC, (
        f"{param_name}: only {n_covered}/{n_compared} ({coverage:.1%}) reference medians "
        f"fall inside Pyro 95% CI (threshold {MIN_COVERAGE_FRAC:.0%})\n"
        f"Misses:\n  {miss_detail}"
    )


@pytest.mark.nuts
@pytest.mark.parametrize("param_name", ["activity", "dk_geno"])
def test_nuts_sign_agreement(pyro_params, reference, param_name):
    """Posterior median sign should agree between frameworks for ≥ 80% of entries."""
    _, pyro_params_dict = pyro_params

    if param_name not in reference["parameters"]:
        pytest.skip(f"{param_name} not in reference")
    if param_name not in pyro_params_dict:
        pytest.skip(f"{param_name} not in Pyro posteriors")

    ref_entries = reference["parameters"][param_name]
    pyro_df     = pyro_params_dict[param_name]

    def _make_key(row):
        return str(row["genotype"])

    pyro_medians = {_make_key(row): row["median"] for _, row in pyro_df.iterrows()}

    n_agree = 0
    n_compared = 0
    for key, ref_vals in ref_entries.items():
        if key not in pyro_medians:
            continue
        n_compared += 1
        ref_sign  = np.sign(ref_vals["median"])
        pyro_sign = np.sign(pyro_medians[key])
        if ref_sign == 0 or pyro_sign == 0:
            n_agree += 1  # zero is compatible with either sign
        elif ref_sign == pyro_sign:
            n_agree += 1

    if n_compared == 0:
        pytest.skip(f"No common genotypes between reference and Pyro for {param_name}")

    frac = n_agree / n_compared
    assert frac >= MIN_SIGN_AGREE_FRAC, (
        f"{param_name}: only {n_agree}/{n_compared} ({frac:.1%}) sign agreements "
        f"(threshold {MIN_SIGN_AGREE_FRAC:.0%})"
    )
