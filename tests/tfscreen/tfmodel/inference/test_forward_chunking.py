"""
A posterior file must not depend on how the forward pass is chunked.

get_posteriors / get_map_posteriors run the model over genotype chunks of
``forward_batch_size``. Library-sized latents have to reach the model whole
(components slice them with ``batch_idx``); slicing them to the chunk first
made hill_geno / thermo / categorical_geno read past the sliced arrays for
every chunk after the first (JAX clamps the index), silently corrupting the
deterministic sites of any library larger than one chunk.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")

_SITES = ("theta_growth_pred", "growth_pred", "dk_geno", "activity")


def _read(path):
    with h5py.File(path) as f:
        return {k: f[k][()] for k in _SITES if k in f}


def _fit(orchestrator, guide_type="component"):
    inference = RunInference(model=orchestrator, seed=1)
    svi = inference.setup_svi(adam_step_size=1e-3, guide_type=guide_type)
    state, params, _ = inference.run_optimization(
        svi=svi, max_num_epochs=1,
        out_prefix=os.path.join(tempfile.mkdtemp(), "fit"))
    return svi, state, params


def _assert_same(a, b):
    assert set(a) == set(b) and a
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-5, atol=1e-6,
                                   err_msg=f"site '{k}'")


@pytest.mark.slow
@pytest.mark.parametrize("theta", ["hill_geno", "categorical_geno",
                                   "thermo.O2_C4_K3_U0_a.PK"])
def test_get_posteriors_independent_of_chunk_size(theta):
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV, theta=theta)
    assert orchestrator.data.num_genotype > 2
    svi, state, _ = _fit(orchestrator)

    out = {}
    for forward_batch_size in (512, 2):
        prefix = os.path.join(tempfile.mkdtemp(), "p")
        RunInference(model=orchestrator, seed=7).get_posteriors(
            svi=svi, svi_state=state, out_prefix=prefix,
            num_posterior_samples=3, forward_batch_size=forward_batch_size)
        out[forward_batch_size] = _read(f"{prefix}_posterior.h5")

    _assert_same(out[512], out[2])


@pytest.mark.slow
def test_get_map_posteriors_independent_of_chunk_size():
    orchestrator = ModelOrchestrator(growth_df=_GROWTH_CSV,
                                     binding_df=_BINDING_CSV,
                                     theta="hill_geno")
    _, _, params = _fit(orchestrator, guide_type="delta")

    out = {}
    for forward_batch_size in (512, 2):
        prefix = os.path.join(tempfile.mkdtemp(), "m")
        RunInference(model=orchestrator, seed=7).get_map_posteriors(
            params, out_prefix=prefix, forward_batch_size=forward_batch_size)
        out[forward_batch_size] = _read(f"{prefix}_posterior.h5")

    _assert_same(out[512], out[2])
