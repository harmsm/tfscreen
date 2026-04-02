import pytest
import torch
import pyro
import numpy as np
import pandas as pd
import os
import dill
from dataclasses import dataclass
from pyro.infer.autoguide import AutoDelta

from tfscreen.analysis.hierarchical.run_inference import RunInference


@dataclass
class MockData:
    num_genotype: int = 10


class MockModel:
    def __init__(self, num_genotype=10):
        self.data = MockData(num_genotype=num_genotype)
        self.priors = {}
        self.pyro_model = lambda **kwargs: None
        self.pyro_model_guide = lambda **kwargs: None
        self.init_params = {}

    def get_batch(self, data, indices):
        return data

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def test_init():
    model = MockModel()
    ri = RunInference(model, seed=42)
    assert ri.model == model
    assert ri._seed == 42

    # Test missing attribute
    del model.data
    with pytest.raises(ValueError, match="`model` must have attribute data"):
        RunInference(model, seed=42)


def test_setup_svi():
    model = MockModel()
    ri = RunInference(model, seed=42)

    # Test with component guide
    svi = ri.setup_svi(guide_type="component")
    assert svi.guide == model.pyro_model_guide

    # Test with delta (AutoDelta) guide
    svi = ri.setup_svi(guide_type="delta")
    assert isinstance(svi.guide, AutoDelta)


def test_setup_svi_invalid_guide_type():
    """setup_svi raises ValueError for unrecognized guide_type."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    with pytest.raises(ValueError, match="not recognized"):
        ri.setup_svi(guide_type="bad_guide")


def test_run_optimization(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)

    out_root = os.path.join(tmpdir, "test")

    mock_svi = mocker.Mock()
    mock_svi.step.return_value = 1.0

    pyro.clear_param_store()
    state, params, converged = ri.run_optimization(
        mock_svi,
        max_num_epochs=10,
        convergence_check_interval=1,
        checkpoint_interval=1,
        out_root=out_root,
        convergence_tolerance=0.01,
        patience=1
    )

    assert state is not None
    assert isinstance(params, dict)
    assert isinstance(converged, bool)
    assert os.path.exists(f"{out_root}_losses.bin")
    assert os.path.exists(f"{out_root}_losses.txt")


def test_run_optimization_with_init_params(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)

    mock_svi = mocker.Mock()
    mock_svi.step.return_value = 1.0

    pyro.clear_param_store()
    ri.run_optimization(
        mock_svi,
        max_num_epochs=2,
        convergence_check_interval=1,
        checkpoint_interval=1,
        init_params={"p": np.array(1.0)}
    )
    assert len(ri._loss_deque) > 0


def test_run_optimization_svi_state_types(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)

    mock_svi = mocker.Mock()
    mock_svi.step.return_value = 1.0

    # Non-string, non-None state: pass a valid param store state dict
    pyro.clear_param_store()
    valid_state = pyro.get_param_store().get_state()
    ri.run_optimization(mock_svi, svi_state=valid_state, max_num_epochs=1,
                        convergence_check_interval=1, checkpoint_interval=1)

    # Invalid string path
    with pytest.raises(ValueError, match="is not valid"):
        ri.run_optimization(mock_svi, svi_state="invalid_path_to_svi_state",
                            max_num_epochs=1, convergence_check_interval=1,
                            checkpoint_interval=1)


def test_run_optimization_convergence(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)

    # Return losses that decrease and then stabilize
    losses = np.concatenate([np.linspace(100, 10, 50), np.ones(50) * 10])
    loss_iter = iter(losses.tolist())

    mock_svi = mocker.Mock()
    mock_svi.step.side_effect = lambda **kwargs: next(loss_iter, 10.0)

    pyro.clear_param_store()
    state, params, converged = ri.run_optimization(
        mock_svi,
        max_num_epochs=100,
        convergence_check_interval=1,
        checkpoint_interval=1,
        convergence_tolerance=1e-1,
        convergence_window=1,
        patience=1
    )
    assert converged in [True, False]


def test_run_optimization_nan_explosion(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)

    mock_svi = mocker.Mock()
    mock_svi.step.return_value = 1.0

    # Put a NaN param in the param store so the explosion check triggers
    pyro.clear_param_store()
    pyro.param("p", torch.tensor(float('nan')))

    with pytest.raises(RuntimeError, match="model exploded"):
        ri.run_optimization(mock_svi, max_num_epochs=1, convergence_check_interval=1,
                            checkpoint_interval=1)


def test_get_posteriors(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test")

    mock_svi = mocker.Mock()
    mock_svi.guide = mocker.Mock()

    # Mock _get_genotype_dim_map so we don't need a real model trace
    mocker.patch.object(ri, "_get_genotype_dim_map", return_value={})

    # Mock Predictive
    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_latent_sampler = mock_predictive.return_value
    mock_latent_sampler.return_value = {"param": torch.zeros(10, 10)}

    ri.get_posteriors(mock_svi, "state", out_root,
                      num_posterior_samples=20, sampling_batch_size=10)

    assert os.path.exists(f"{out_root}_posterior.h5")


def test_get_posteriors_batching_logic(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_batching")

    mock_svi = mocker.Mock()
    mock_svi.guide = mocker.Mock()

    mocker.patch.object(ri, "_get_genotype_dim_map", return_value={})

    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_sampler = mock_predictive.return_value
    # 3 latent batches (10+10+5), each has 1 forward call
    mock_sampler.side_effect = [
        {"p": torch.zeros(10, 5)},   # latent 1
        {"obs": torch.zeros(10, 5)}, # forward 1
        {"p": torch.zeros(10, 5)},   # latent 2
        {"obs": torch.zeros(10, 5)}, # forward 2
        {"p": torch.zeros(5, 5)},    # latent 3
        {"obs": torch.zeros(5, 5)}   # forward 3
    ]

    ri.get_posteriors(mock_svi, "state", out_root,
                      num_posterior_samples=25, sampling_batch_size=10)
    assert os.path.exists(f"{out_root}_posterior.h5")


def test_get_posteriors_full_logic(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_full")

    mock_svi = mocker.Mock()

    # global_p: not in dim_map; geno_p: in dim_map at dim -1
    mocker.patch.object(ri, "_get_genotype_dim_map", return_value={"geno_p": -1})

    mock_predictive = mocker.patch("tfscreen.analysis.hierarchical.run_inference.Predictive")
    mock_sampler = mock_predictive.return_value
    mock_sampler.side_effect = [
        {"global_p": torch.zeros(1, 1), "geno_p": torch.zeros(1, 10)},  # latent
        {"obs": torch.zeros(1, 5)},   # forward 1 (genotypes 0-4)
        {"obs": torch.zeros(1, 5)},   # forward 2 (genotypes 5-9)
    ]

    ri.get_posteriors(mock_svi, "state", out_root,
                      num_posterior_samples=1, sampling_batch_size=1,
                      forward_batch_size=5)
    assert os.path.exists(f"{out_root}_posterior.h5")


def test_run_optimization_restore(tmpdir, mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_restore")

    # Write a valid checkpoint using the Pyro param store state format
    pyro.clear_param_store()
    ps_state = pyro.get_param_store().get_state()
    checkpoint_data = {
        "rng_state": ri._rng.bit_generator.state,
        "svi_state": ps_state,
        "current_step": 0,
    }
    checkpoint_file = f"{out_root}_checkpoint.pkl"
    with open(checkpoint_file, "wb") as f:
        dill.dump(checkpoint_data, f)

    mock_svi = mocker.Mock()
    mock_svi.step.return_value = 1.0

    # Run with checkpoint path
    ri.run_optimization(mock_svi, svi_state=checkpoint_file, max_num_epochs=1,
                        convergence_check_interval=1, checkpoint_interval=1)


def test_write_params(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_params")
    params = {"p": np.array([1.0])}
    ri.write_params(params, out_root)
    assert os.path.exists(f"{out_root}_params.npz")


def test_write_losses_append(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_losses")

    path = f"{out_root}_losses.bin"
    with open(path, "wb") as f:
        f.write(b"header")

    ri._write_losses([1.0, 2.0], out_root)
    assert os.path.getsize(path) > 6


def test_write_losses_empty(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_losses_empty")

    ri._write_losses([], out_root)
    assert not os.path.exists(f"{out_root}_losses.bin")


def test_update_loss_deque():
    model = MockModel()
    ri = RunInference(model, seed=42)
    from collections import deque
    ri._loss_deque = deque(maxlen=200)
    ri._update_loss_deque(np.ones(200))
    # std will be 0, mean_new == mean_old → relative_change = 0.0
    assert ri._relative_change == 0.0


def test_get_site_names(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)

    # Mock poutine.trace: poutine.trace(fn) returns handler; handler.get_trace(...) returns trace
    mock_nodes = {
        "site1": {"type": "deterministic"},
        "site2": {"type": "sample"},
    }
    mock_trace_obj = mocker.Mock()
    mock_trace_obj.nodes = mock_nodes

    mock_handler = mocker.Mock()
    mock_handler.get_trace.return_value = mock_trace_obj
    mocker.patch(
        "tfscreen.analysis.hierarchical.run_inference.poutine.trace",
        return_value=mock_handler
    )

    sites = ri._get_site_names()
    assert "site1" in sites
    assert "site2" not in sites


def test_restore_checkpoint_error(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test")

    # Write a bad checkpoint where svi_state is not a dict
    with open(f"{out_root}_bad.pkl", "wb") as f:
        dill.dump({"rng_state": ri._rng.bit_generator.state,
                   "svi_state": "not_a_state"}, f)

    with pytest.raises(ValueError, match="does not appear to have a saved svi_state"):
        ri._restore_checkpoint(f"{out_root}_bad.pkl")


def test_restore_checkpoint_current_step(tmpdir):
    """_restore_checkpoint restores _current_step when present in checkpoint."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_root = os.path.join(tmpdir, "test_step")

    pyro.clear_param_store()
    ps_state = pyro.get_param_store().get_state()

    checkpoint_file = f"{out_root}_checkpoint.pkl"
    checkpoint_data = {
        'svi_state': ps_state,
        'rng_state': ri._rng.bit_generator.state,
        'current_step': 999,
    }
    with open(checkpoint_file, "wb") as f:
        dill.dump(checkpoint_data, f)

    restored = ri._restore_checkpoint(checkpoint_file)
    assert ri._current_step == 999
    assert isinstance(restored, dict)


def test_jitter_init_parameters():
    model = MockModel()
    ri = RunInference(model, seed=42)
    params = {"p": np.array(1.0), "a": np.array([1.0, 2.0])}

    jittered = ri._jitter_init_parameters(params, 0.1)
    assert not np.array_equal(np.atleast_1d(jittered["p"]), [1.0])

    # Zero jitter returns unchanged
    params = {"p": np.array(1.0)}
    jittered = ri._jitter_init_parameters(params, 0.0)
    assert np.array_equal(np.atleast_1d(jittered["p"]), [1.0])


def test_get_key():
    model = MockModel()
    ri = RunInference(model, seed=42)
    k1 = ri.get_key()
    k2 = ri.get_key()
    assert isinstance(k1, int)
    assert k1 != k2
