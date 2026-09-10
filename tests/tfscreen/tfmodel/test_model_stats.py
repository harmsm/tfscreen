"""
Tests for the pre-fit parameter/observation accounting in tfmodel.model_stats.

Coverage:
  - site table: exact per-site parameter counts against hand-computed values
  - classification: scope / entity / level / kind, incl. mutation-level models
  - observation counts: the direct good_mask sums must equal the masks a
    concrete trace actually applies (drift guard on the observers)
  - per-genotype coverage: matches a direct reduction of the mask
  - ragged data: partially-observed genotypes are counted, padding is not
  - summary/warnings/writers
"""

import json

import numpy as np
import pandas as pd
import pytest

import jax
import jax.numpy as jnp
from numpyro.handlers import trace, seed

from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
from tfscreen.tfmodel import model_stats
from tfscreen.tfmodel.model_stats import (
    ModelStats,
    count_model_dimensions,
    format_model_stats,
    write_model_stats,
    _is_scale_like,
    _site_component,
    _site_scope,
    _observation_counts,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GENOTYPES = ["wt", "A1V", "A2V", "A1V/A2V"]
REPLICATES = [1, 2]
CONDITIONS = ["kanR-cond", "pheS-cond"]
CONCS = [0.0, 0.01, 0.1, 1.0]
TIMES = [60.0, 90.0]


@pytest.fixture
def growth_df():
    """Fully crossed growth data: 4 genotypes x 2 reps x 2 cond x 4 conc x 2 t."""
    rows = []
    for rep in REPLICATES:
        for cond in CONDITIONS:
            for conc in CONCS:
                for geno in GENOTYPES:
                    for t_sel in TIMES:
                        rows.append({
                            "library": "lib",
                            "replicate": rep,
                            "condition_pre": cond,
                            "condition_sel": cond,
                            "titrant_name": "iptg",
                            "titrant_conc": conc,
                            "t_pre": 30.0,
                            "t_sel": t_sel,
                            "genotype": geno,
                            "ln_cfu": 10.0,
                            "ln_cfu_std": 0.5,
                        })
    return pd.DataFrame(rows)


@pytest.fixture
def binding_df():
    """Binding curves for two of the four genotypes."""
    rows = []
    for geno in ["wt", "A1V"]:
        for conc in CONCS:
            rows.append({
                "genotype": geno,
                "titrant_name": "iptg",
                "titrant_conc": conc,
                "theta_obs": 0.5,
                "theta_std": 0.1,
            })
    return pd.DataFrame(rows)


# ModelOrchestrator defaults theta_growth_noise to "logit_normal", but
# configure_model -- the caller this tool actually runs under -- defaults it to
# "zero". Pin it so the base fixture reflects the CLI default; the
# logit_normal case gets its own test.
BASE_KWARGS = {"batch_size": None, "theta_growth_noise": "zero"}


@pytest.fixture
def orchestrator(growth_df, binding_df):
    return ModelOrchestrator(growth_df, binding_df, **BASE_KWARGS)


@pytest.fixture
def stats(orchestrator):
    return count_model_dimensions(orchestrator)


@pytest.fixture
def presplit_df():
    """Pre-split data covering every (replicate, condition_pre, genotype)."""
    return pd.DataFrame([
        {"library": "lib", "replicate": rep, "condition_pre": cond,
         "genotype": geno, "ln_cfu": 9.5, "ln_cfu_std": 0.4}
        for rep in REPLICATES for cond in CONDITIONS for geno in GENOTYPES
    ])


@pytest.fixture
def base_growth_df():
    """A single direct reference-condition growth-rate measurement."""
    return pd.DataFrame([{"genotype": "wt", "rate": 0.02, "rate_std": 0.001}])


@pytest.fixture
def all_channel_orchestrator(growth_df, binding_df, presplit_df, base_growth_df):
    """All four likelihood channels wired up at once."""
    return ModelOrchestrator(growth_df, binding_df, presplit_df=presplit_df,
                             base_growth_df=base_growth_df, **BASE_KWARGS)


def _full_data(orchestrator):
    """The full-batch data pytree, as count_model_dimensions builds it."""
    data = jax.device_put(orchestrator.data)
    return orchestrator.get_batch(
        data, jnp.arange(orchestrator.data.num_genotype)
    )


# ---------------------------------------------------------------------------
# Site table
# ---------------------------------------------------------------------------

def test_site_table_shape(stats):
    """Every latent gets exactly one row; observed sites are excluded."""
    assert isinstance(stats, ModelStats)
    assert len(stats.sites) > 0
    assert stats.sites["site"].is_unique
    assert not any(name.endswith("_obs") for name in stats.sites["site"])


def test_per_genotype_site_counts(stats, orchestrator):
    """
    hill_geno carries four per-genotype latents, each of size num_genotype
    (times num_titrant_name, which is 1 here).
    """
    n_geno = orchestrator.data.num_genotype
    sites = stats.sites.set_index("site")

    for suffix in ("logit_low", "logit_delta", "log_hill_K", "log_hill_n"):
        row = sites.loc[f"theta_{suffix}_offset"]
        assert row["n_param"] == n_geno
        assert row["scope"] == "per_genotype"
        assert row["entity"] == "genotype"
        assert row["level"] == "individual"


def test_condition_level_site_counts(stats, orchestrator):
    """linear growth carries one k and one m per condition_rep."""
    n_cond = len(orchestrator.growth_tm.map_groups["condition_rep"])
    sites = stats.sites.set_index("site")

    for name in ("condition_growth_k", "condition_growth_m"):
        row = sites.loc[name]
        assert row["n_param"] == n_cond
        assert row["scope"] == "per_condition"
        assert not row["scales_with_library"]


def test_hyperparameters_are_labelled(stats):
    """
    Sites above the entity plate in a component that has entity-level latents
    are hyperparameters; a component with no entity latents has none.
    """
    sites = stats.sites.set_index("site")

    assert sites.loc["theta_logit_low_hyper_loc", "level"] == "hyperparameter"
    assert sites.loc["dk_geno_hyper_scale", "level"] == "hyperparameter"

    # condition_growth has no entity-indexed latents at all
    cg = stats.sites[stats.sites["component"] == "condition_growth"]
    assert (cg["level"] == "individual").all()


def test_scale_like_detection(stats):
    """Positive-support latents are tagged 'scale'; real-support ones are not."""
    sites = stats.sites.set_index("site")

    assert sites.loc["theta_logit_low_hyper_scale", "kind"] == "scale"
    assert sites.loc["theta_logit_low_hyper_loc", "kind"] == "location"
    assert sites.loc["activity_global_scale", "kind"] == "scale"
    assert sites.loc["activity_local_scale", "kind"] == "scale"


def test_total_matches_site_sum(stats):
    """The headline total is just the sum of the table."""
    assert stats.summary["parameters"]["n_param_total"] == \
        int(stats.sites["n_param"].sum())


def test_component_attribution_is_longest_prefix():
    """theta_growth_noise_* must not be attributed to 'theta' or 'growth'."""
    assert _site_component("theta_growth_noise_epsilon") == "theta_growth_noise"
    assert _site_component("theta_logit_low_offset") == "theta"
    assert _site_component("growth_noise_sigma_k") == "growth_noise"
    assert _site_component("growth_nu") == "growth"
    assert _site_component("base_growth_k_ref") == "base_growth"


# ---------------------------------------------------------------------------
# Scope classification
# ---------------------------------------------------------------------------

class _Frame:
    def __init__(self, name, size):
        self.name = name
        self.size = size


def _scope(*frames):
    return _site_scope([(n, s) for n, s in frames])


def test_scope_global_when_no_plates():
    assert _scope() == ("global", None, False)


def test_scope_ignores_size_one_design_plates():
    """A size-1 plate is not a real axis."""
    assert _scope(("foo_titrant_name", 1)) == ("global", None, False)


def test_scope_entity_priority():
    """pair beats mutation beats genotype."""
    scope, entity, _ = _scope(("x_mutation_plate", 5), ("x_pair_plate", 3))
    assert (scope, entity) == ("per_pair", "pair")

    scope, entity, _ = _scope(("x_genotype_plate", 9), ("x_mutation_plate", 5))
    assert (scope, entity) == ("per_mutation", "mutation")


def test_scope_per_datum_requires_entity_and_measurement_axis():
    """
    genotype x titrant_conc grows with the data; genotype x titrant_name (a
    phenotype axis) does not.
    """
    scope, _, per_datum = _scope(("x_genotype", 9), ("x_titrant_conc", 8))
    assert per_datum
    assert scope == "per_genotype_datum"

    scope, _, per_datum = _scope(("x_genotype", 9), ("x_titrant_name", 2))
    assert not per_datum
    assert scope == "per_genotype"


def test_scope_design_axes_without_entity():
    assert _scope(("x_condition_parameters", 8))[0] == "per_condition"
    assert _scope(("x_tubes", 12))[0] == "per_sample"


def test_is_scale_like_handles_missing_support():
    class _NoSupport:
        pass
    assert not _is_scale_like(_NoSupport())


# ---------------------------------------------------------------------------
# Observation counting -- drift guard against the observers
# ---------------------------------------------------------------------------

def _masks_from_concrete_trace(orchestrator):
    """Recover the mask each observer actually applied, by tracing for real."""
    full_data = _full_data(orchestrator)
    tr = trace(seed(orchestrator.jax_model, rng_seed=0)).get_trace(
        data=full_data, priors=orchestrator.priors
    )

    masks = {}
    for name, site in tr.items():
        if site["type"] != "sample" or not site.get("is_observed", False):
            continue
        mask = getattr(site["fn"], "_mask", None)
        assert mask is not None, f"{name} is not masked; update _observation_counts"
        masks[name[:-len("_obs")]] = int(np.asarray(mask).sum())

    return masks


def test_observation_counts_match_applied_masks(orchestrator):
    """
    _observation_counts reads good_mask directly rather than from the trace.
    That shortcut is only valid while it reproduces the masks the observers
    apply -- this test fails if an observer changes how it masks.
    """
    direct = _observation_counts(_full_data(orchestrator))
    traced = _masks_from_concrete_trace(orchestrator)

    assert set(direct) == set(traced)
    for channel, counts in direct.items():
        assert counts["n_obs"] == traced[channel], channel


def test_observation_counts_match_applied_masks_all_channels(
        all_channel_orchestrator):
    """
    Same drift guard with presplit and base_growth wired up. Those two
    observers index their masks by the growth genotype batch, which is the
    part of _observation_counts most likely to go stale.
    """
    direct = _observation_counts(_full_data(all_channel_orchestrator))
    traced = _masks_from_concrete_trace(all_channel_orchestrator)

    assert set(direct) == {"growth", "binding", "presplit", "base_growth"}
    assert set(direct) == set(traced)
    for channel, counts in direct.items():
        assert counts["n_obs"] == traced[channel], channel


def test_all_channels_counted(all_channel_orchestrator):
    """Every channel shows up in the census, with its own anchor counts."""
    stats = count_model_dimensions(all_channel_orchestrator)

    channels = stats.summary["observations"]["by_channel"]
    assert channels["presplit"]["n_obs"] == \
        len(REPLICATES) * len(CONDITIONS) * len(GENOTYPES)
    assert channels["base_growth"]["n_obs"] == 1

    anchors = stats.summary["anchors"]
    assert anchors["n_presplit_genotype"] == len(GENOTYPES)
    assert anchors["n_base_growth_genotype"] == 1

    coverage = stats.per_genotype.set_index("genotype")
    assert coverage.loc["wt", "n_obs_base_growth"] == 1
    assert coverage.loc["A1V", "n_obs_base_growth"] == 0

    # base_growth owns the k_ref latent; it must be counted.
    assert "base_growth_k_ref" in set(stats.sites["site"])

    # Both anchor warnings are silent now that the anchors exist.
    assert not any("base_growth" in w for w in stats.warnings)


def test_observation_counts_exclude_padding(orchestrator, stats):
    """
    Binding covers 2 of 4 genotypes, so the binding channel is dense over its
    own tensor while the growth tensor is fully crossed and unpadded.
    """
    obs = stats.summary["observations"]["by_channel"]
    assert obs["growth"]["n_obs"] == len(REPLICATES) * len(CONDITIONS) * \
        len(CONCS) * len(GENOTYPES) * len(TIMES)
    assert obs["binding"]["n_obs"] == 2 * len(CONCS)


def test_ragged_growth_data_is_counted_not_padded(growth_df, binding_df):
    """Dropping rows lowers n_obs and shows up as padding, not as data."""
    trimmed = growth_df[~((growth_df["genotype"] == "A2V") &
                          (growth_df["titrant_conc"] == 1.0))].copy()

    full = count_model_dimensions(
        ModelOrchestrator(growth_df, binding_df, **BASE_KWARGS))
    ragged = count_model_dimensions(
        ModelOrchestrator(trimmed, binding_df, **BASE_KWARGS))

    n_dropped = len(growth_df) - len(trimmed)
    assert (full.summary["observations"]["by_channel"]["growth"]["n_obs"]
            - ragged.summary["observations"]["by_channel"]["growth"]["n_obs"]
            == n_dropped)
    assert ragged.summary["observations"]["padding_fraction"] > 0


# ---------------------------------------------------------------------------
# Per-genotype coverage
# ---------------------------------------------------------------------------

def test_coverage_one_row_per_genotype(stats, orchestrator):
    coverage = stats.per_genotype
    assert len(coverage) == orchestrator.data.num_genotype
    assert set(coverage["genotype"]) == set(GENOTYPES)


def test_coverage_matches_direct_mask_reduction(stats, orchestrator):
    """n_obs_growth must equal the mask summed over every non-genotype axis."""
    mask = np.asarray(_full_data(orchestrator).growth.good_mask).astype(bool)
    expected = mask.reshape(-1, mask.shape[-1]).sum(axis=0)

    assert list(stats.per_genotype["n_obs_growth"]) == list(expected)


def test_coverage_distinct_levels(stats):
    """The fully-crossed fixture gives every genotype the full design."""
    coverage = stats.per_genotype
    assert (coverage["n_titrant_conc"] == len(CONCS)).all()
    assert (coverage["n_replicate"] == len(REPLICATES)).all()
    assert (coverage["n_timepoint"] == len(TIMES)).all()
    assert (coverage["n_condition"] == len(CONDITIONS)).all()


def test_coverage_ragged_titration(growth_df, binding_df):
    """A genotype measured at fewer concentrations is reported as such."""
    trimmed = growth_df[~((growth_df["genotype"] == "A2V") &
                          (growth_df["titrant_conc"].isin([0.1, 1.0])))].copy()
    stats = count_model_dimensions(
        ModelOrchestrator(trimmed, binding_df, **BASE_KWARGS))

    coverage = stats.per_genotype.set_index("genotype")
    assert coverage.loc["A2V", "n_titrant_conc"] == len(CONCS) - 2
    assert coverage.loc["wt", "n_titrant_conc"] == len(CONCS)


def test_coverage_binding_joined_by_label(stats):
    """Binding counts land on the right genotypes, not on tensor positions."""
    coverage = stats.per_genotype.set_index("genotype")
    assert coverage.loc["wt", "n_obs_binding"] == len(CONCS)
    assert coverage.loc["A1V", "n_obs_binding"] == len(CONCS)
    assert coverage.loc["A2V", "n_obs_binding"] == 0


def test_theta_under_determined_flag(growth_df, binding_df):
    """
    Fewer distinct concentrations than theta parameters means the theta curve
    is not identified from growth data alone.
    """
    trimmed = growth_df[growth_df["titrant_conc"].isin([0.0, 1.0])].copy()
    stats = count_model_dimensions(
        ModelOrchestrator(trimmed, binding_df, **BASE_KWARGS))

    cov = stats.summary["coverage"]
    assert stats.summary["library"]["theta_params_per_genotype"] == 4
    assert cov["n_genotype_theta_under_determined"] == len(GENOTYPES)
    assert any("distinct titrant" in w for w in stats.warnings)


# ---------------------------------------------------------------------------
# Effective-parameter bracket
# ---------------------------------------------------------------------------

def test_effective_parameter_bracket_is_ordered(stats):
    ep = stats.summary["effective_parameters"]
    assert 0 < ep["p_eff_lower"] <= ep["p_eff_upper"]
    assert ep["obs_per_param_lower_bound"] <= ep["obs_per_param_upper_bound"]


def test_bracket_excludes_per_datum_latents(growth_df, binding_df):
    """
    logit_normal noise adds one latent per genotype per concentration. Those
    are random effects, so they are counted and flagged but kept out of the
    bracket and out of params_per_genotype.
    """
    stats = count_model_dimensions(
        ModelOrchestrator(growth_df, binding_df, batch_size=None,
                          theta_growth_noise="logit_normal"))

    n_per_datum = stats.summary["parameters"]["n_param_per_datum"]
    assert n_per_datum == len(GENOTYPES) * len(CONCS)

    ep = stats.summary["effective_parameters"]
    assert ep["p_eff_upper"] == \
        stats.summary["parameters"]["n_param_total"] - n_per_datum
    assert any("per observation" in w for w in stats.warnings)


def test_no_per_datum_latents_by_default(stats):
    assert stats.summary["parameters"]["n_param_per_datum"] == 0


# ---------------------------------------------------------------------------
# Mutation-level models
# ---------------------------------------------------------------------------

def test_mutation_level_model_scales_with_mutations(growth_df, binding_df):
    """
    hill_mut parameters are indexed by mutation, not genotype -- the whole
    point of the decomposition, and something a hand-maintained parameter
    table would get wrong.
    """
    stats = count_model_dimensions(
        ModelOrchestrator(growth_df, binding_df, theta="hill_mut",
                          **BASE_KWARGS))

    theta_sites = stats.sites[stats.sites["component"] == "theta"]
    assert (theta_sites["entity"] == "mutation").any()
    assert not (theta_sites["entity"] == "genotype").any()

    assert stats.summary["library"]["params_per_mutation"] > 0
    assert stats.summary["library"]["theta_params_per_genotype"] == 0


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def test_anchor_inventory(stats):
    anchors = stats.summary["anchors"]
    assert anchors["n_genotype"] == len(GENOTYPES)
    assert anchors["n_binding_genotype"] == 2
    assert anchors["n_base_growth_genotype"] == 0
    assert anchors["n_presplit_genotype"] == 0
    assert any("base_growth" in w for w in stats.warnings)


def test_spiked_genotypes_split_binding_anchors(growth_df, binding_df):
    stats = count_model_dimensions(
        ModelOrchestrator(growth_df, binding_df, spiked_genotypes=["wt"],
                          **BASE_KWARGS))

    anchors = stats.summary["anchors"]
    assert anchors["n_binding_genotype"] == 2
    assert anchors["n_binding_genotype_spiked"] == 1
    assert anchors["n_binding_genotype_in_library"] == 1


# ---------------------------------------------------------------------------
# Batching must not shrink the counts
# ---------------------------------------------------------------------------

def test_minibatching_does_not_change_counts(growth_df, binding_df):
    """
    The trace runs on the full genotype batch. Tracing a mini-batch would
    report per-genotype counts as the batch size instead of the library size.
    """
    full = count_model_dimensions(
        ModelOrchestrator(growth_df, binding_df, **BASE_KWARGS))
    batched = count_model_dimensions(
        ModelOrchestrator(growth_df, binding_df, batch_size=2,
                          theta_growth_noise="zero"))

    assert full.summary["parameters"]["n_param_total"] == \
        batched.summary["parameters"]["n_param_total"]
    assert full.summary["observations"]["n_obs_total"] == \
        batched.summary["observations"]["n_obs_total"]


# ---------------------------------------------------------------------------
# Binding-only
# ---------------------------------------------------------------------------

def test_binding_only_model(binding_df):
    """A binding-only model has no growth channel and no coverage table."""
    stats = count_model_dimensions(
        ModelOrchestrator(None, binding_df, binding_only=True,
                          **BASE_KWARGS))

    channels = stats.summary["observations"]["by_channel"]
    assert "binding" in channels
    assert "growth" not in channels
    assert len(stats.per_genotype) == 0
    assert stats.summary["parameters"]["n_param_total"] > 0


# ---------------------------------------------------------------------------
# Tracing mode
# ---------------------------------------------------------------------------

def test_trace_is_abstract(stats):
    """
    The default path must stay free. If this flips to 'concrete', a component
    started doing something abstract evaluation cannot handle -- correct, but
    it now costs a full forward pass on the real library.
    """
    assert stats.summary["trace_mode"] == "abstract"


def test_falls_back_to_concrete_trace(orchestrator, monkeypatch):
    """When eval_shape fails, the counts still come out."""
    def _boom(*args, **kwargs):
        raise RuntimeError("no abstract evaluation for you")

    monkeypatch.setattr(model_stats.jax, "eval_shape", _boom)
    stats = count_model_dimensions(orchestrator)

    assert stats.summary["trace_mode"] == "concrete"
    assert stats.summary["parameters"]["n_param_total"] > 0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def test_write_model_stats(stats, tmp_path):
    prefix = str(tmp_path / "run")
    csv_path, json_path = write_model_stats(stats, prefix)

    sites = pd.read_csv(csv_path)
    assert len(sites) == len(stats.sites)
    assert "n_param" in sites.columns

    with open(json_path) as f:
        payload = json.load(f)
    assert payload["parameters"]["n_param_total"] == \
        stats.summary["parameters"]["n_param_total"]
    assert payload["warnings"] == stats.warnings


def test_format_model_stats_is_text(stats):
    text = format_model_stats(stats)
    assert "MODEL DIMENSION SUMMARY" in text
    assert "Observations" in text
    assert "Anchors" in text
    assert str(stats.summary["parameters"]["n_param_total"]) in \
        text.replace(",", "")
