"""
Predictions on a genotype subset must equal the full prediction.

predict() rebuilds the model on the requested genotypes (copy_orchestrator).
For components with per-mutation and per-pair parameters (hill_mut, the
*_mut activity components, the thermodynamic theta models) the rebuilt model
used to derive its mutation and pair labels from the subset alone, so the
fitted per-mutation and per-pair posterior arrays were applied to the wrong
mutations and pairs. tfs-predict-growth always chunks a full-size library,
so every chunk was silently wrong.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro.handlers import seed, trace

from tfscreen.genetics import build_mut_geno_matrix
from tfscreen.tfmodel.analysis.prediction import copy_orchestrator, predict
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_MUTS = ["A1C", "D2E", "F3G", "H4I", "K5L", "M6N"]

# Doubles first, in reverse, so any subset sees its mutations and pairs in a
# different first-seen order than the full library does.
_GENOTYPES = (["wt"]
              + ["/".join(p) for p in itertools.combinations(_MUTS, 2)][::-1]
              + _MUTS)

_KEY = ["genotype", "titrant_conc", "t_sel"]

_MODELS = {
    "hill_mut": dict(theta="hill_mut", epistasis=True),
    "activity_hierarchical_mut": dict(theta="hill_geno",
                                      activity="hierarchical_mut",
                                      epistasis=True),
    "thermo_C4_U0_PK": dict(theta="thermo.O2_C4_K3_U0_a.PK",
                            epistasis=True),
}


def _growth_df():
    rng = np.random.default_rng(0)
    rows = []
    for g in _GENOTYPES:
        for c in [0.0, 0.01, 0.1, 1.0, 10.0]:
            for t in [0.0, 10.0]:
                rows.append(dict(library="lib", genotype=g,
                                 titrant_name="iptg", titrant_conc=c,
                                 condition_pre="pre", condition_sel="sel+",
                                 t_pre=10.0, t_sel=t,
                                 ln_cfu=10 + rng.normal(), ln_cfu_std=0.1,
                                 replicate=1))
    return pd.DataFrame(rows)


def _prior_posteriors(orch, num_draws=3):
    """Several prior draws of every latent, as a posterior dict."""
    data = orch.get_batch(orch.data, jnp.arange(orch.data.num_genotype))
    draws = {}
    for s in range(num_draws):
        tr = trace(seed(orch.jax_model, rng_seed=s)).get_trace(
            data=data, priors=orch.priors)
        for name, site in tr.items():
            if site["type"] == "sample" and not site.get("is_observed", False):
                draws.setdefault(name, []).append(np.asarray(site["value"]))
    return {k: np.stack(v) for k, v in draws.items()}


def _predict(orch, posteriors, genotypes=None):
    df = predict(orch, posteriors, q_to_get=[0.5], num_samples=None,
                 genotypes=genotypes)
    return df.set_index(_KEY)["q0.5"]


def _max_chunk_error(orch, posteriors, batch_sizes):
    full = _predict(orch, posteriors)
    genotypes = list(orch.growth_tm.tensor_dim_labels[-1])
    worst = 0.0
    for bs in batch_sizes:
        for i in range(0, len(genotypes), bs):
            sub = _predict(orch, posteriors, genotypes[i:i + bs])
            worst = max(worst,
                        float(np.max(np.abs(sub - full.loc[sub.index]))))
    return worst


@pytest.fixture(scope="module")
def growth_df():
    return _growth_df()


@pytest.mark.parametrize("model", list(_MODELS))
def test_chunked_prediction_matches_full(growth_df, model):
    """Every chunking (pairs split across chunks included) equals full."""
    orch = ModelOrchestrator(growth_df, None, **_MODELS[model])
    assert len(orch.pair_labels) > 0
    posteriors = _prior_posteriors(orch)
    assert _max_chunk_error(orch, posteriors, [2, 3, 5, 7]) < 1e-5


@pytest.mark.parametrize("model", list(_MODELS))
def test_genotype_subset_prediction_matches_full(growth_df, model):
    """A --genotypes_file style subset (a scattered selection) equals full."""
    orch = ModelOrchestrator(growth_df, None, **_MODELS[model])
    posteriors = _prior_posteriors(orch)
    full = _predict(orch, posteriors)
    subset = ["M6N", "D2E/K5L", "A1C", "H4I/M6N", "wt", "F3G/H4I"]
    sub = _predict(orch, posteriors, subset)
    assert set(sub.index.get_level_values(0)) == set(subset)
    np.testing.assert_allclose(sub.values, full.loc[sub.index].values,
                               atol=1e-5)


def test_subset_orchestrator_keeps_fitted_mutation_labels(growth_df):
    """The rebuilt model indexes mutations and pairs as the fit did."""
    orch = ModelOrchestrator(growth_df, None, **_MODELS["hill_mut"])
    new = copy_orchestrator(orch, genotypes=["M6N", "D2E/K5L"])
    assert new.mut_labels == orch.mut_labels
    assert new.pair_labels == orch.pair_labels
    assert new.data.growth.num_mutation == len(orch.mut_labels)
    assert new.data.growth.num_pair == len(orch.pair_labels)
    # Settings stay label-free, so a written configuration does not carry them.
    assert "mut_labels" not in new.settings


def test_map_fit_chunked_prediction_matches_full(growth_df, tmp_path):
    """A short MAP fit's posterior file predicts the same in chunks."""
    from tfscreen.tfmodel.inference.run_inference import RunInference

    orch = ModelOrchestrator(growth_df, None, **_MODELS["hill_mut"])
    ri = RunInference(model=orch, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-2, guide_type="delta")
    svi_state, _, _ = ri.run_optimization(
        svi=svi, max_num_epochs=50, out_prefix=str(tmp_path / "fit"),
        checkpoint_interval=1000)
    map_params = svi.optim.get_params(svi_state.optim_state)
    ri.get_map_posteriors(map_params, out_prefix=str(tmp_path / "map"))
    posterior_file = str(tmp_path / "map_posterior.h5")
    assert _max_chunk_error(orch, posterior_file, [3, 7]) < 1e-5


def test_fixed_labels_index_space():
    """Fixed labels set the row order; absent mutations get zero rows."""
    muts, pairs, mat, p_idx, g_idx = build_mut_geno_matrix(
        ["K5L/M6N", "M6N"], mut_labels=_MUTS,
        pair_labels=["A1C/D2E", "K5L/M6N"])
    assert muts == _MUTS
    assert pairs == ["A1C/D2E", "K5L/M6N"]
    np.testing.assert_array_equal(mat[_MUTS.index("M6N")], [1, 1])
    np.testing.assert_array_equal(mat[_MUTS.index("K5L")], [1, 0])
    assert mat[:4].sum() == 0
    np.testing.assert_array_equal(p_idx, [1])
    np.testing.assert_array_equal(g_idx, [0])


def test_fixed_labels_reject_unknown_mutation_or_pair():
    with pytest.raises(ValueError, match="mutation\\(s\\) not in"):
        build_mut_geno_matrix(["A1C/Q9R"], mut_labels=["A1C"],
                              pair_labels=[])
    with pytest.raises(ValueError, match="pair\\(s\\) not in"):
        build_mut_geno_matrix(["A1C/D2E"], mut_labels=["A1C", "D2E"],
                              pair_labels=[])
