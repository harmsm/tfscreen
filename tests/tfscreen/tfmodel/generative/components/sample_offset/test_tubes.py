"""Tube labels for the per-tube sample offsets (extraction)."""
import os

import numpy as np
import pytest
from numpyro import handlers

from tfscreen.tfmodel.analysis.extraction import extract_parameters
from tfscreen.tfmodel.generative.components.sample_offset import level, normal
from tfscreen.tfmodel.generative.components.sample_offset._tubes import (
    TUBE_DIMS,
    tube_index,
)
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator

_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")

KEY = ["replicate", "condition_pre", "condition_sel", "titrant_name",
       "titrant_conc", "t_sel"]


@pytest.fixture(scope="module", params=["level", "normal"])
def model(request):
    o = ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                          batch_size=6, sample_offset=request.param)
    return request.param, o


def _site(kind):
    return ("sample_offset_offset", "sample_offset_sigma") if kind == "level" \
        else ("sample_offset_delta_k", "sample_offset_sigma_env")


def test_extracted_offsets_match_the_model_tubes(model):
    """
    Each extracted row carries the value the model adds to that tube's
    growth rows, so the labels follow the component's own reshape.
    """
    kind, o = model
    site, sigma_site = _site(kind)
    g = o.data.growth
    shape = (g.num_replicate, g.num_time, g.num_condition_pre,
             g.num_condition_sel, g.num_titrant_name, g.num_titrant_conc)
    values = np.random.default_rng(0).normal(size=int(np.prod(shape)))

    module = level if kind == "level" else normal
    with handlers.seed(rng_seed=0):
        sub = handlers.substitute(module.define_model,
                                  data={site: values, sigma_site: 0.3})
        out = np.asarray(sub("sample_offset", g, module.get_priors()))
    if kind == "normal":
        out = values.reshape(*shape, 1)

    posteriors = {site: values[None, :], sigma_site: np.array([[0.3]])}
    params = extract_parameters(o, posteriors, q_to_get=[0.5])
    ext = params[site]
    assert not ext.duplicated(KEY).any()
    assert params[sigma_site]["q0.5"].iloc[0] == pytest.approx(0.3)

    df = o.training_tm.df
    idx = tuple(df[f"{d}_idx"].to_numpy(dtype=int) for d in TUBE_DIMS)
    expected = df[KEY].assign(expected=out[idx + (0,)])
    merged = ext.merge(expected.drop_duplicates(KEY), on=KEY, how="left")
    assert merged["expected"].notna().all()
    np.testing.assert_allclose(merged["q0.5"], merged["expected"])
    # one row per tube that holds growth data
    assert len(ext) == len(np.unique(tube_index(o.training_tm)))


def test_zero_offset_extracts_nothing():
    from tfscreen.tfmodel.generative.components.sample_offset import zero
    assert zero.get_extract_specs(None) == []


def test_tube_index_refuses_rows_without_tube_columns(model):
    _, o = model
    tm = o.training_tm
    from types import SimpleNamespace
    stub = SimpleNamespace(df=tm.df.drop(columns=["time_idx", "titrant_conc_idx"]),
                           tensor_dim_names=tm.tensor_dim_names,
                           tensor_dim_labels=tm.tensor_dim_labels)
    with pytest.raises(ValueError, match="lack the tube index columns") as e:
        tube_index(stub)
    assert "time_idx" in str(e.value) and "titrant_conc_idx" in str(e.value)
    assert "replicate_idx" not in str(e.value)
