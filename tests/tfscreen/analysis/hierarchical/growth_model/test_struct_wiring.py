"""
Tests for struct ensemble wiring in ModelClass.

Covers:
  1. Constructor parameter and settings property
  2. Error handling when struct_ensemble_path is missing
  3. Full integration: GrowthData.struct_* fields populated correctly
  4. _needs_mut activated for lac_dimer_lnK_nn_prior
  5. Registry entry is present
"""

import numpy as np
import pandas as pd
import pytest
import h5py

STRUCTURE_NAMES = ('H', 'HD', 'L', 'LE2')

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal synthetic data
# ──────────────────────────────────────────────────────────────────────────────

# Residue numbers needed to cover genotypes in the minimal growth data below
_RESNUMS = [42, 84]

# Genotypes: wt + two single mutants + one double mutant (needed for pair detection)
_GENOTYPES = ["wt", "M42I", "K84L", "M42I/K84L"]


def _make_struct_ensemble_h5(tmp_path, resnums=_RESNUMS):
    """Write a minimal valid HDF5 ensemble file covering all four structures."""
    path = str(tmp_path / "ensemble.h5")
    L = len(resnums)
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('structure_names',
                          data=np.array(list(STRUCTURE_NAMES), dtype=h5py.string_dtype()))
        for seed_i, name in enumerate(STRUCTURE_NAMES):
            rng = np.random.RandomState(seed_i)
            grp = hf.create_group(name)
            logP     = rng.randn(L, 20).astype(np.float32)
            dist_mat = np.abs(rng.randn(L, L)).astype(np.float32)
            np.fill_diagonal(dist_mat, 0.0)
            grp.create_dataset('logP',                 data=logP)
            grp.create_dataset('residue_nums',         data=np.asarray(resnums, dtype=np.int32))
            grp.create_dataset('dist_matrix',          data=dist_mat)
            grp.create_dataset('n_chains_bearing_mut', data=np.int32(2))
    return path


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

def test_settings_includes_struct_ensemble_path(tmp_path):
    """settings property must expose struct_ensemble_path."""
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)
    h5_path      = _make_struct_ensemble_h5(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    mc = ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_path=h5_path,
    )
    assert "struct_ensemble_path" in mc.settings
    assert mc.settings["struct_ensemble_path"] == h5_path


def test_default_struct_ensemble_path_is_none(tmp_path):
    """struct_ensemble_path defaults to None and appears as None in settings."""
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    mc = ModelClass(growth_path, binding_path, theta="hill")
    assert mc.settings["struct_ensemble_path"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_struct_path_raises(tmp_path):
    """lac_dimer_lnK_nn_prior without struct_ensemble_path must raise ValueError."""
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    with pytest.raises(ValueError, match="struct_ensemble_path"):
        ModelClass(growth_path, binding_path, theta="lac_dimer_lnK_nn_prior")


# ──────────────────────────────────────────────────────────────────────────────
# Integration — struct fields populated on GrowthData
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_mc(tmp_path_factory):
    """
    Build a ModelClass with lac_dimer_lnK_nn_prior and a synthetic HDF5 file.
    Expensive to construct, so scoped to the module.
    """
    tmp_path = tmp_path_factory.mktemp("struct_wiring")
    growth_path  = _make_growth_csv(tmp_path)
    binding_path = _make_binding_csv(tmp_path)
    h5_path      = _make_struct_ensemble_h5(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    return ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_path=h5_path,
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
        h5_path      = _make_struct_ensemble_h5(tmp_path)

        from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
        return ModelClass(
            growth_path, binding_path,
            theta="lac_dimer_lnK_nn_prior",
            struct_ensemble_path=h5_path,
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
    h5_path      = _make_struct_ensemble_h5(tmp_path)

    mc = ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_path=h5_path,
    )

    out_prefix = str(tmp_path / "svi_out")
    inference = RunInference(model=mc, seed=0)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
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
    h5_path      = _make_struct_ensemble_h5(tmp_path)

    mc = ModelClass(
        growth_path, binding_path,
        theta="lac_dimer_lnK_nn_prior",
        struct_ensemble_path=h5_path,
        epistasis=True,
    )

    out_prefix = str(tmp_path / "svi_epi_out")
    inference = RunInference(model=mc, seed=0)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=100,
    )

    assert svi_state is not None
    assert isinstance(params, dict)


# ──────────────────────────────────────────────────────────────────────────────
# binding_only + struct models
# ──────────────────────────────────────────────────────────────────────────────

_MWC_STRUCTURE_NAMES = ('H', 'HO', 'L', 'LO', 'HE2', 'LE2')


def _make_ddG_prior_csv(tmp_path, genotypes=_GENOTYPES):
    """Write a minimal ddG prior CSV for mwc_dimer_lnK_ddG_prior."""
    mut_labels = sorted({
        part
        for g in genotypes if g != "wt"
        for part in g.split("/")
    })
    rows = [{"mut": m, **{s: 0.0 for s in _MWC_STRUCTURE_NAMES}} for m in mut_labels]
    df = pd.DataFrame(rows)
    path = str(tmp_path / "ddg_prior.csv")
    df.to_csv(path, index=False)
    return path


def _make_binding_only_csv(tmp_path, genotypes=_GENOTYPES):
    """Write a minimal binding CSV (no growth data — binding_only mode)."""
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
    path = str(tmp_path / "binding_only.csv")
    df.to_csv(path, index=False)
    return path


def test_binding_only_struct_raises_without_struct_path(tmp_path):
    """mwc_dimer_lnK_ddG_prior in binding_only mode must raise if struct_ensemble_path is absent."""
    binding_path = _make_binding_only_csv(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    with pytest.raises(ValueError, match="struct_ensemble_path"):
        ModelClass(None, binding_path,
                   binding_only=True,
                   theta="mwc_dimer_lnK_ddG_prior")


@pytest.fixture(scope="module")
def binding_only_ddG_mc(tmp_path_factory):
    """
    ModelClass in binding_only mode with mwc_dimer_lnK_ddG_prior.
    Scoped to module so it is built once for all tests below.
    """
    tmp_path = tmp_path_factory.mktemp("binding_only_ddG")
    binding_path = _make_binding_only_csv(tmp_path)
    ddg_path     = _make_ddG_prior_csv(tmp_path)

    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    return ModelClass(
        None, binding_path,
        binding_only=True,
        theta="mwc_dimer_lnK_ddG_prior",
        struct_ensemble_path=ddg_path,
    )


class TestBindingOnlyDdGStructFields:
    """BindingData must carry the struct_* fields when using mwc_dimer_lnK_ddG_prior."""

    def test_struct_names_is_correct_tuple(self, binding_only_ddG_mc):
        assert set(binding_only_ddG_mc.data.binding.struct_names) == set(_MWC_STRUCTURE_NAMES)

    def test_num_struct_is_six(self, binding_only_ddG_mc):
        assert binding_only_ddG_mc.data.binding.num_struct == 6

    def test_struct_features_shape(self, binding_only_ddG_mc):
        M = binding_only_ddG_mc.data.binding.num_mutation
        S = binding_only_ddG_mc.data.binding.num_struct
        feat = binding_only_ddG_mc.data.binding.struct_features
        assert feat.shape == (M, S)

    def test_mut_labels_populated(self, binding_only_ddG_mc):
        assert len(binding_only_ddG_mc.mut_labels) > 0

    def test_num_mutation_nonzero(self, binding_only_ddG_mc):
        assert binding_only_ddG_mc.data.binding.num_mutation > 0

    def test_growth_data_is_none(self, binding_only_ddG_mc):
        assert binding_only_ddG_mc.data.growth is None

    def test_struct_n_chains_is_none_for_ddG_csv(self, binding_only_ddG_mc):
        # ddG CSV loader does not provide n_chains (it's None)
        assert binding_only_ddG_mc.data.binding.struct_n_chains is None


@pytest.mark.slow
def test_binding_only_ddG_svi_runs(tmp_path):
    """Five SVI steps with mwc_dimer_lnK_ddG_prior in binding_only mode must not raise."""
    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    from tfscreen.analysis.hierarchical.run_inference import RunInference

    binding_path = _make_binding_only_csv(tmp_path)
    ddg_path     = _make_ddG_prior_csv(tmp_path)

    mc = ModelClass(
        None, binding_path,
        binding_only=True,
        theta="mwc_dimer_lnK_ddG_prior",
        struct_ensemble_path=ddg_path,
    )

    out_prefix = str(tmp_path / "svi_binding_only_ddg")
    inference = RunInference(model=mc, seed=0)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=100,
    )

    assert svi_state is not None
    assert isinstance(params, dict)
    assert len(params) > 0


# ──────────────────────────────────────────────────────────────────────────────
# binding_only mini-batching (batch_size < num_genotypes)
# ──────────────────────────────────────────────────────────────────────────────

def test_binding_only_minibatch_get_random_idx_does_not_crash(tmp_path):
    """
    get_random_idx must not crash when batch_size < num_genotypes in binding-only mode.
    Previously the batching code tried to pin all N binding genotypes into a
    batch_size-sized array, causing an IndexError.
    """
    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass

    # 4 genotypes, batch_size=2 — forces the mini-batch path
    big_genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
    binding_path = _make_binding_only_csv(tmp_path, genotypes=big_genotypes)
    ddg_path = _make_ddG_prior_csv(tmp_path, genotypes=big_genotypes)

    mc = ModelClass(
        None, binding_path,
        binding_only=True,
        theta="mwc_dimer_lnK_ddG_prior",
        struct_ensemble_path=ddg_path,
        batch_size=2,
    )

    # Should not raise
    idx = mc.get_random_idx(batch_key=42, num_batches=1)
    assert len(idx) == 2

    # All genotypes are subsamplable; none are pinned
    assert mc.data.num_binding == 0
    assert len(mc.data.not_binding_idx) == 4


@pytest.mark.slow
def test_binding_only_minibatch_svi_runs(tmp_path):
    """Five SVI steps with batch_size < num_genotypes in binding-only mode must not raise."""
    from tfscreen.analysis.hierarchical.growth_model.model_class import ModelClass
    from tfscreen.analysis.hierarchical.run_inference import RunInference

    big_genotypes = ["wt", "M42I", "K84L", "M42I/K84L"]
    binding_path = _make_binding_only_csv(tmp_path, genotypes=big_genotypes)
    ddg_path = _make_ddG_prior_csv(tmp_path, genotypes=big_genotypes)

    mc = ModelClass(
        None, binding_path,
        binding_only=True,
        theta="mwc_dimer_lnK_ddG_prior",
        struct_ensemble_path=ddg_path,
        batch_size=2,
    )

    out_prefix = str(tmp_path / "svi_binding_only_minibatch")
    inference = RunInference(model=mc, seed=0)
    svi = inference.setup_svi(adam_step_size=1e-3)
    svi_state, params, converged = inference.run_optimization(
        svi=svi,
        max_num_epochs=5,
        out_prefix=out_prefix,
        checkpoint_interval=100,
    )

    assert svi_state is not None
    assert isinstance(params, dict)
    assert len(params) > 0
