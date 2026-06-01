"""
Regression tests for the _needs_mut gate in ModelOrchestrator.

Covers the bug where *_unfolded_lnK_mut theta variants were omitted from
the _needs_mut tuple in both the growth path and the binding_only path,
causing num_mutation to stay 0 and per-mutation offset guesses to be
zero-length arrays that failed the guesses-CSV validation.

Each test checks that after ModelOrchestrator construction:
  - mut_labels is non-empty
  - data.{growth,binding}.num_mutation > 0
"""

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Minimal synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_GENOTYPES = ["wt", "M42I", "K84L", "M42I/K84L"]


def _make_growth_csv(tmp_path, genotypes=_GENOTYPES):
    rows = []
    for geno in genotypes:
        parts = geno.split("/") if geno != "wt" else []
        wt_aa_1  = parts[0][0]        if len(parts) > 0 else ""
        resid_1  = int(parts[0][1:-1]) if len(parts) > 0 else ""
        mut_aa_1 = parts[0][-1]       if len(parts) > 0 else ""
        wt_aa_2  = parts[1][0]        if len(parts) > 1 else ""
        resid_2  = int(parts[1][1:-1]) if len(parts) > 1 else ""
        mut_aa_2 = parts[1][-1]       if len(parts) > 1 else ""
        num_muts = len(parts)

        for rep in [1.0, 2.0]:
            for conc in [0.0, 100.0]:
                rows.append({
                    "sample":            f"s-{geno}-{rep}-{conc}",
                    "genotype":          geno,
                    "wt_aa_1":           wt_aa_1,
                    "resid_1":           resid_1,
                    "mut_aa_1":          mut_aa_1,
                    "wt_aa_2":           wt_aa_2,
                    "resid_2":           resid_2,
                    "mut_aa_2":          mut_aa_2,
                    "wt_aa_3":           "",
                    "resid_3":           "",
                    "mut_aa_3":          "",
                    "num_muts":          num_muts,
                    "counts":            1000,
                    "replicate":         rep,
                    "condition_pre":     "pre",
                    "t_pre":             24.0,
                    "condition_sel":     "sel",
                    "t_sel":             48.0,
                    "titrant_name":      "iptg",
                    "titrant_conc":      conc,
                    "barcode_i7":        "AAAA",
                    "barcode_i5":        "TTTT",
                    "seq_run":           1,
                    "od600":             0.3,
                    "library":           "lib",
                    "obs_file":          "dummy.csv",
                    "sample_cfu":        1e7,
                    "sample_cfu_std":    1e5,
                    "sample_cfu_detectable": True,
                    "adjusted_counts":   1001,
                    "frequency":         0.1,
                    "cfu":               1e6,
                    "cfu_var":           1e8,
                    "ln_cfu_var":        0.01,
                    "ln_cfu":            14.0 + 0.1 * np.random.randn(),
                })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "growth.csv")
    df.to_csv(path, index=False)
    return path


def _make_binding_csv(tmp_path, genotypes=_GENOTYPES):
    rows = []
    for geno in genotypes:
        for conc in [1e-6, 1e-4]:
            rows.append({
                "genotype":     geno,
                "titrant_name": "iptg",
                "titrant_conc": conc,
                "theta_obs":    0.5,
                "theta_std":    0.05,
            })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "binding.csv")
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Growth-path _needs_mut tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theta", [
    "thermo.O2_C4_K3_U1_a.PK",
    "thermo.O2_C12_K5_U1_a.PK",
])
def test_needs_mut_growth_path(tmp_path, theta):
    """
    Growth-path _needs_mut must fire for *_unfolded_lnK_mut variants so that
    the mutation decomposition matrices are built and num_mutation > 0.
    """
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)

    mc = ModelOrchestrator(growth_path, binding_path, theta=theta)

    assert len(mc.mut_labels) > 0, \
        f"{theta} (growth): mut_labels is empty — _needs_mut did not fire"
    assert mc.data.growth.num_mutation > 0, \
        f"{theta} (growth): num_mutation == 0 — mutation matrices not built"


# ──────────────────────────────────────────────────────────────────────────────
# Binding-only _needs_mut tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theta", [
    "thermo.O2_C4_K3_U1_a.PK",
    "thermo.O2_C12_K5_U1_a.PK",
])
def test_needs_mut_binding_only_path(tmp_path, theta):
    """
    Binding-only _needs_mut must fire for *_unfolded_lnK_mut variants so that
    mutation matrices are built and per-mutation offset guesses have the right
    shape (not zero-length).
    """
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

    binding_path = _make_binding_csv(tmp_path)

    mc = ModelOrchestrator(None, binding_path, binding_only=True, theta=theta)

    assert len(mc.mut_labels) > 0, \
        f"{theta} (binding_only): mut_labels is empty — _needs_mut did not fire"
    assert mc.data.binding.num_mutation > 0, \
        f"{theta} (binding_only): num_mutation == 0 — mutation matrices not built"


# ──────────────────────────────────────────────────────────────────────────────
# Guesses shape regression: offset arrays must match num_mutation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theta", [
    "thermo.O2_C4_K3_U1_a.PK",
    "thermo.O2_C12_K5_U1_a.PK",
])
def test_offset_guesses_have_correct_shape_binding_only(tmp_path, theta):
    """
    The per-mutation offset initial guesses must have length == num_mutation,
    not zero.  A zero-length array indicates _needs_mut did not fire.
    """
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

    binding_path = _make_binding_csv(tmp_path)
    mc = ModelOrchestrator(None, binding_path, binding_only=True, theta=theta)

    M = mc.data.binding.num_mutation
    assert M > 0

    # Every *_offset key in init_params must have length M (or M as a dimension)
    for key, val in mc.init_params.items():
        if "offset" in key and hasattr(val, "shape") and val.shape != ():
            assert val.shape[0] == M or val.shape[-1] == M, \
                f"{key} has shape {val.shape} but expected M={M} on at least one axis"
