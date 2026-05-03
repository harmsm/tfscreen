"""
Tests for Phase 5 — struct ensemble wiring in ModelClass.

Covers:
  1. Constructor parameter and settings property
  2. Error handling when struct_ensemble_paths is missing
  3. Full integration: GrowthData.struct_* fields populated correctly
  4. _needs_mut activated for lac_dimer_lnK_nn_prior
  5. Registry entry is present
"""

import io
import numpy as np
import pandas as pd
import pytest

STRUCTURE_NAMES = ('H', 'HD', 'L', 'LE2')

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal synthetic data
# ──────────────────────────────────────────────────────────────────────────────

# Residue numbers needed to cover genotypes in the minimal growth data below
_RESNUMS = [42, 84]

# Genotypes: wt + two single mutants + one double mutant (needed for pair detection)
_GENOTYPES = ["wt", "M42I", "K84L", "M42I/K84L"]


def _write_struct_npz(path, resnums, seed=0):
    """Write a minimal valid structure NPZ file for the given residue numbers."""
    rng = np.random.RandomState(seed)
    L = len(resnums)
    logP      = rng.randn(L, 20).astype(np.float32)
    dist_mat  = np.abs(rng.randn(L, L)).astype(np.float32)
    np.fill_diagonal(dist_mat, 0.0)
    np.savez(
        path,
        logP=logP,
        residue_nums=np.asarray(resnums, dtype=np.int32),
        dist_matrix=dist_mat,
        n_chains_bearing_mut=np.int32(2),
    )


def _make_struct_ensemble_paths(tmp_path, resnums=_RESNUMS):
    """Create one NPZ per structure and return a {name: path} dict."""
    paths = {}
    for i, name in enumerate(STRUCTURE_NAMES):
        p = str(tmp_path / f"{name}.npz")
        _write_struct_npz(p, resnums, seed=i)
        paths[name] = p
    return paths


def _make_growth_csv(tmp_path, genotypes=_GENOTYPES):
    """
    Write a minimal growth CSV with the given genotypes.
    Two replicates, two IPTG concentrations, one selection condition.
    """
    rows = []
    for geno in genotypes:
        parts = geno.split("/") if geno != "wt" else []
        wt_aa_1 = parts[0][0]       if len(parts) > 0 else ""
        resid_1  = int(parts[0][1:-1]) if len(parts) > 0 else ""
        mut_aa_1 = parts[0][-1]     if len(parts) > 0 else ""
        wt_aa_2 = parts[1][0]       if len(parts) > 1 else ""
        resid_2  = int(parts[1][1:-1]) if len(parts) > 1 else ""
        mut_aa_2 = parts[1][-1]     if len(parts) > 1 else ""
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
    """Write a minimal binding CSV."""
    rows = []
    for geno in genotypes:
        for conc in [1e-6, 1e-4]:
            rows.append({
                "genotype":    geno,
                "titrant_name": "iptg",
                "titrant_conc": conc,
                "theta_obs":    0.5,
                "theta_std":    0.05,
                "tx":           0.5,
                "pred":         0.5,
            })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "binding.csv")
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

def test_registry_entry():
    from tfscreen.analysis.hierarchical.growth_model.registry import model_registry
    assert "lac_dimer_lnK_nn_prior" in model_registry["theta"]


# ──────────────────────────────────────────────────────────────────────────────
# Constructor / settings (no data loading needed)
# ──────────────────────────────────────────────────────────────────────────────

def test_settings_includes_struct_ensemble_paths(tmp_path):
    """settings property must expose struct_ensemble_paths."""
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)
    paths = _make_struct_ensemble_paths(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    mc = ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_paths=paths,
    )
    assert "struct_ensemble_paths" in mc.settings
    assert mc.settings["struct_ensemble_paths"] == paths


def test_default_struct_ensemble_paths_is_none(tmp_path):
    """struct_ensemble_paths defaults to None and appears as None in settings."""
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    mc = ModelClass(growth_path, binding_path, theta="hill")
    assert mc.settings["struct_ensemble_paths"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_struct_paths_raises(tmp_path):
    """lac_dimer_lnK_nn_prior without struct_ensemble_paths must raise ValueError."""
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    with pytest.raises(ValueError, match="struct_ensemble_paths"):
        ModelClass(growth_path, binding_path, theta="lac_dimer_lnK_nn_prior")


# ──────────────────────────────────────────────────────────────────────────────
# Integration — struct fields populated on GrowthData
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_mc(tmp_path_factory):
    """
    Build a ModelClass with lac_dimer_lnK_nn_prior and synthetic NPZ files.
    Expensive to construct, so scoped to the module.
    """
    tmp_path = tmp_path_factory.mktemp("struct_wiring")
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)
    paths        = _make_struct_ensemble_paths(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    return ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_paths=paths,
    )


class TestStructFieldsOnGrowthData:
    def test_struct_names_is_correct_tuple(self, fitted_mc):
        assert fitted_mc.data.growth.struct_names == STRUCTURE_NAMES

    def test_num_struct_is_four(self, fitted_mc):
        assert fitted_mc.data.growth.num_struct == 4

    def test_struct_features_shape(self, fitted_mc):
        M = fitted_mc.data.growth.num_mutation
        S = fitted_mc.data.growth.num_struct
        feat = fitted_mc.data.growth.struct_features
        assert feat.shape == (M, S, 60)

    def test_struct_n_chains_shape(self, fitted_mc):
        S = fitted_mc.data.growth.num_struct
        assert fitted_mc.data.growth.struct_n_chains.shape == (S,)

    def test_struct_n_chains_value(self, fitted_mc):
        # All structures were written with n_chains_bearing_mut=2
        import jax.numpy as jnp
        n = fitted_mc.data.growth.struct_n_chains
        assert int(n[0]) == 2

    def test_no_contact_distances_without_epistasis(self, fitted_mc):
        # Fixture was built without epistasis=True, so no contact arrays
        assert fitted_mc.data.growth.struct_contact_distances is None
        assert fitted_mc.data.growth.struct_contact_pair_idx  is None

    def test_mut_labels_populated(self, fitted_mc):
        # _needs_mut must have fired for lac_dimer_lnK_nn_prior
        assert len(fitted_mc.mut_labels) > 0

    def test_num_mutation_nonzero(self, fitted_mc):
        assert fitted_mc.data.growth.num_mutation > 0


class TestStructFieldsWithEpistasis:
    @pytest.fixture(scope="class")
    def mc_epi(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("struct_epi")
        growth_path  = _make_growth_csv(tmp_path)
        binding_path = _make_binding_csv(tmp_path)
        paths        = _make_struct_ensemble_paths(tmp_path)

        from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
        return ModelClass(
            growth_path, binding_path,
            theta="lac_dimer_lnK_nn_prior",
            struct_ensemble_paths=paths,
            epistasis=True,
        )

    def test_contact_distances_present(self, mc_epi):
        dist = mc_epi.data.growth.struct_contact_distances
        assert dist is not None

    def test_contact_distances_shape(self, mc_epi):
        P = mc_epi.data.growth.num_pair
        S = mc_epi.data.growth.num_struct
        if P > 0:
            assert mc_epi.data.growth.struct_contact_distances.shape == (P, S)

    def test_pair_labels_populated(self, mc_epi):
        assert isinstance(mc_epi.pair_labels, list)


# ──────────────────────────────────────────────────────────────────────────────
# Slow SVI smoke test — runs a handful of gradient steps end-to-end
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_svi_runs_without_error(tmp_path):
    """
    Five steps of SVI with lac_dimer_lnK_nn_prior should not raise and should
    return a non-None svi_state and a params dict.
    """
    import os
    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    from tfscreen.analysis.hierarchical.run_inference import RunInference

    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)
    paths        = _make_struct_ensemble_paths(tmp_path)

    mc = ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_paths=paths,
    )

    out_root = str(tmp_path / "svi_out")
    inference = RunInference(model=mc, seed=0)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_root=out_root,
        checkpoint_interval=100,
    )

    assert svi_state is not None
    assert isinstance(params, dict)
    assert len(params) > 0


@pytest.mark.slow
def test_svi_with_epistasis_runs_without_error(tmp_path):
    """Same as above but with epistasis=True (activates horseshoe terms)."""
    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    from tfscreen.analysis.hierarchical.run_inference import RunInference

    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)
    paths        = _make_struct_ensemble_paths(tmp_path)

    mc = ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_paths=paths,
        epistasis=True,
    )

    out_root = str(tmp_path / "svi_epi_out")
    inference = RunInference(model=mc, seed=0)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_root=out_root,
        checkpoint_interval=100,
    )

    assert svi_state is not None
    assert isinstance(params, dict)
