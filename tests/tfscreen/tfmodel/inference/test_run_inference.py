import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
import h5py
from tfscreen.tfmodel.inference.run_inference import (
    LAPLACE_MIN_EIGENVALUE,
    RunInference,
    laplace_eigenvalue_floor,
    resolve_guide_type,
)
import os
import dill
import optax
from numpyro.infer.autoguide import AutoDelta, AutoLaplaceApproximation
from numpyro.infer.svi import SVIState

from numpyro.infer.svi import SVIState
from flax import struct

@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    # Many tests below call run_optimization / _write_checkpoint with the
    # default out_prefix ("tfs"), which writes checkpoint/loss files into the
    # cwd. Run every test from its own tmp dir so nothing leaks into the repo.
    monkeypatch.chdir(tmp_path)

@struct.dataclass
class MockData:
    num_genotype: int = 10

class MockModel:
    def __init__(self, num_genotype=10):
        self.data = MockData(num_genotype=num_genotype)
        self.priors = {}
        self.jax_model = lambda **kwargs: None
        self.jax_model_guide = lambda **kwargs: None
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
    
    # Test with default guide_type
    svi = ri.setup_svi(guide_type="component")
    assert svi.guide == model.jax_model_guide
    
    svi = ri.setup_svi(guide_type="delta")
    assert isinstance(svi.guide, AutoDelta)
    
# =============================================================================
# get_posteriors — real model + component guide, no Predictive mocking
# =============================================================================
#
# get_posteriors drives its genotype-batch forward pass with a JIT-compiled
# per-chunk function (see `_build_genotype_chunk_scanner` /
# `_genotype_chunk_indices` in run_inference.py) so that `Predictive` is
# traced only once and GPU memory holds one chunk at a time.  Mocking
# `Predictive` with a fixed call-count `side_effect` list (the old test
# strategy) no longer reflects how the code calls it, so these tests run a
# real tiny numpyro model + component guide end-to-end and check actual
# numeric/index correctness instead.

def _component_jax_model(data, priors):
    """One global + one per-genotype parameter, plus a deterministic site
    that echoes the genotype index — used to catch any genotype-order
    scrambling introduced by batching/stitching."""
    global_p = numpyro.sample("global_p", dist.Normal(0., 1.))
    with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
        numpyro.sample("geno_p", dist.Normal(global_p, 1.))
        numpyro.deterministic("geno_idx_check", data.batch_idx.astype(jnp.float32))


def _component_jax_model_guide(data, priors):
    """Mean-field guide matching `_component_jax_model`."""
    global_loc = numpyro.param("global_loc", 0.0)
    global_scale = numpyro.param("global_scale", 1.0,
                                 constraint=dist.constraints.positive)
    numpyro.sample("global_p", dist.Normal(global_loc, global_scale))

    with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
        geno_loc = numpyro.param("geno_loc", jnp.zeros(data.num_genotype))
        geno_scale = numpyro.param("geno_scale", jnp.ones(data.num_genotype),
                                   constraint=dist.constraints.positive)
        numpyro.sample("geno_p", dist.Normal(geno_loc, geno_scale))


class ComponentModel:
    """Minimal RunInference-compatible model wrapper with both a generative
    model and a component (mean-field) guide, for exercising get_posteriors
    end-to-end."""

    def __init__(self, num_genotype=10):
        self.data = LaplaceData(num_genotype=num_genotype,
                                batch_idx=jnp.arange(num_genotype))
        self.priors = {}
        self.jax_model = _component_jax_model
        self.jax_model_guide = _component_jax_model_guide

    def get_batch(self, data, indices):
        return LaplaceData(num_genotype=len(indices), batch_idx=indices)

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def _component_svi(model, seed=0):
    """Initialize (but don't train) a component-guide SVI for `model`."""
    ri = RunInference(model, seed=seed)
    svi = ri.setup_svi(guide_type="component")
    svi_state = svi.init(ri.get_key(), priors=model.priors, data=model.data)
    return ri, svi, svi_state


def test_get_posteriors_creates_h5(tmpdir):
    model = ComponentModel(num_genotype=5)
    ri, svi, svi_state = _component_svi(model)
    out_prefix = str(tmpdir.join("post_create"))

    ri.get_posteriors(svi, svi_state, out_prefix,
                      num_posterior_samples=4, sampling_batch_size=4)

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_posteriors_output_shapes(tmpdir):
    num_genotype = 6
    num_samples = 10
    model = ComponentModel(num_genotype=num_genotype)
    ri, svi, svi_state = _component_svi(model)
    out_prefix = str(tmpdir.join("post_shapes"))

    ri.get_posteriors(svi, svi_state, out_prefix,
                      num_posterior_samples=num_samples, sampling_batch_size=4,
                      forward_batch_size=num_genotype)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)
        assert hf.attrs["num_samples"] == num_samples


def test_get_posteriors_sampling_batch_uneven(tmpdir):
    """num_posterior_samples not evenly divisible by sampling_batch_size."""
    num_genotype = 4
    num_samples = 7  # 7 = 3 + 3 + 1
    model = ComponentModel(num_genotype=num_genotype)
    ri, svi, svi_state = _component_svi(model)
    out_prefix = str(tmpdir.join("post_uneven"))

    ri.get_posteriors(svi, svi_state, out_prefix,
                      num_posterior_samples=num_samples, sampling_batch_size=3,
                      forward_batch_size=num_genotype)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == num_samples
        assert hf["geno_p"].shape[0] == num_samples


def test_get_posteriors_sites_to_save(tmpdir):
    model = ComponentModel(num_genotype=5)
    ri, svi, svi_state = _component_svi(model)
    out_prefix = str(tmpdir.join("post_filtered"))

    ri.get_posteriors(svi, svi_state, out_prefix,
                      num_posterior_samples=3, sampling_batch_size=3,
                      sites_to_save=["geno_p"])

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf


def test_get_posteriors_genotype_order_preserved_full_batch(tmpdir):
    """forward_batch_size == num_genotype (single, unpadded chunk)."""
    num_genotype = 7
    model = ComponentModel(num_genotype=num_genotype)
    ri, svi, svi_state = _component_svi(model)
    out_prefix = str(tmpdir.join("post_full"))

    ri.get_posteriors(svi, svi_state, out_prefix,
                      num_posterior_samples=4, sampling_batch_size=4,
                      forward_batch_size=num_genotype)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        idx_check = hf["geno_idx_check"][:]

    expected = np.tile(np.arange(num_genotype), (4, 1))
    np.testing.assert_array_equal(idx_check, expected)


def test_get_posteriors_genotype_order_preserved_ragged_batches(tmpdir):
    """forward_batch_size that doesn't evenly divide num_genotype must still
    produce correctly-ordered, unpadded output.

    Regression test for past genotype-index scrambling bugs in the
    batching/stitching logic: 7 genotypes with forward_batch_size=3 forces
    chunks of (3, 3, 1), internally padded to (3, 3, 3) and trimmed back by
    `_merge_scan_chunks`.
    """
    num_genotype = 7
    model = ComponentModel(num_genotype=num_genotype)
    ri, svi, svi_state = _component_svi(model)
    out_prefix = str(tmpdir.join("post_ragged"))

    ri.get_posteriors(svi, svi_state, out_prefix,
                      num_posterior_samples=3, sampling_batch_size=3,
                      forward_batch_size=3)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        idx_check = hf["geno_idx_check"][:]
        geno_p = hf["geno_p"][:]

    expected = np.tile(np.arange(num_genotype), (3, 1))
    np.testing.assert_array_equal(idx_check, expected)
    assert geno_p.shape == (3, num_genotype)


def test_get_posteriors_chunking_matches_unchunked_latents(tmpdir):
    """Sampled latent values (geno_p, global_p) must be identical regardless
    of forward_batch_size: latents are drawn once on the full dataset before
    genotype chunking even starts, and (since RNG consumption inside
    get_posteriors no longer depends on forward_batch_size) the two runs
    draw from identical PRNG streams."""
    num_genotype = 9

    model_a = ComponentModel(num_genotype=num_genotype)
    ri_a, svi_a, state_a = _component_svi(model_a, seed=11)
    out_a = str(tmpdir.join("post_a"))
    ri_a.get_posteriors(svi_a, state_a, out_a,
                        num_posterior_samples=2, sampling_batch_size=2,
                        forward_batch_size=num_genotype)

    model_b = ComponentModel(num_genotype=num_genotype)
    ri_b, svi_b, state_b = _component_svi(model_b, seed=11)
    out_b = str(tmpdir.join("post_b"))
    ri_b.get_posteriors(svi_b, state_b, out_b,
                        num_posterior_samples=2, sampling_batch_size=2,
                        forward_batch_size=4)  # ragged: chunks of 4, 4, 1

    with h5py.File(f"{out_a}_posterior.h5", "r") as hf_a, \
         h5py.File(f"{out_b}_posterior.h5", "r") as hf_b:
        np.testing.assert_allclose(hf_a["geno_p"][:], hf_b["geno_p"][:], rtol=1e-5)
        np.testing.assert_allclose(hf_a["global_p"][:], hf_b["global_p"][:], rtol=1e-5)


# =============================================================================
# _genotype_chunk_indices / _concat_genotype_chunks — pure indexing-math tests
# =============================================================================

def test_genotype_chunk_indices_exact_divisor():
    idx = np.asarray(RunInference._genotype_chunk_indices(6, 3))
    np.testing.assert_array_equal(idx, [[0, 1, 2], [3, 4, 5]])


def test_genotype_chunk_indices_padding_repeats_last_valid_index():
    idx = np.asarray(RunInference._genotype_chunk_indices(7, 3))
    np.testing.assert_array_equal(idx, [[0, 1, 2], [3, 4, 5], [6, 6, 6]])


def test_genotype_chunk_indices_single_chunk_smaller_than_batch():
    idx = np.asarray(RunInference._genotype_chunk_indices(2, 5))
    np.testing.assert_array_equal(idx, [[0, 1, 1, 1, 1]])


def test_concat_genotype_chunks_positive_axis_no_padding():
    # 2 chunks, per-chunk shape (samples=2, genotypes=3), genotype axis=1
    chunk0 = np.array([[0., 1., 2.], [10., 11., 12.]])
    chunk1 = np.array([[3., 4., 5.], [13., 14., 15.]])
    merged = RunInference._concat_genotype_chunks([chunk0, chunk1], axis=1, total_size=6)
    expected = np.array([[0., 1., 2., 3., 4., 5.],
                         [10., 11., 12., 13., 14., 15.]])
    np.testing.assert_array_equal(merged, expected)


def test_concat_genotype_chunks_trims_padding():
    # total real genotypes = 5, forward_batch_size=3 -> chunks (0,1,2),(3,4,4)
    chunk0 = np.array([[0., 1., 2.]])
    chunk1 = np.array([[3., 4., 4.]])  # last entry is padding (duplicate of idx 4)
    merged = RunInference._concat_genotype_chunks([chunk0, chunk1], axis=1, total_size=5)
    expected = np.array([[0., 1., 2., 3., 4.]])
    np.testing.assert_array_equal(merged, expected)


def test_concat_genotype_chunks_negative_axis():
    # genotype axis = -1 (last axis) of a per-chunk shape (samples, genotypes)
    chunk0 = np.array([[0., 1.], [10., 11.]])
    chunk1 = np.array([[2., 3.], [12., 13.]])
    merged = RunInference._concat_genotype_chunks([chunk0, chunk1], axis=-1, total_size=4)
    expected = np.array([[0., 1., 2., 3.], [10., 11., 12., 13.]])
    np.testing.assert_array_equal(merged, expected)


def test_write_params(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_params")
    params = {"p": jnp.array([1.0])}
    ri.write_params(params, out_prefix)
    assert os.path.exists(f"{out_prefix}_params.npz")

def test_write_losses_append(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_losses")
    # Pre-existing binary file is ignored when current_step != 0
    path = f"{out_prefix}_losses.bin"
    with open(path, "wb") as f:
        f.write(b"existing")

    ri._write_losses([1.0, 2.0], out_prefix)
    assert os.path.getsize(path) > 8


def test_write_losses_txt_has_header(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_hdr")

    # Step 0: header is written even when losses list is empty
    ri._write_losses(np.array([]), out_prefix)
    txt_path = f"{out_prefix}_losses.txt"
    assert os.path.exists(txt_path)
    with open(txt_path) as fh:
        first_line = fh.readline().strip()
    assert first_line == "epoch,loss,step,step_size"


def test_write_losses_txt_format(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_fmt")

    # Init (step 0)
    ri._write_losses(np.array([]), out_prefix)

    # Simulate one interval having completed
    ri._current_step = ri._iterations_per_epoch  # epoch 1
    ri._step_size = 1e-3
    ri._write_losses(np.array([42.0, 43.0, 50.0]), out_prefix)

    with open(f"{out_prefix}_losses.txt") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    assert lines[0] == "epoch,loss,step,step_size"
    # Second line: epoch=1, loss=median of the block (43.0), step, step size
    parts = lines[1].split(",")
    assert int(parts[0]) == 1
    assert float(parts[1]) == pytest.approx(43.0)
    assert int(parts[2]) == ri._iterations_per_epoch
    assert float(parts[3]) == pytest.approx(1e-3)


def test_get_site_names(mocker):
    model = MockModel()
    ri = RunInference(model, seed=42)
    
    # Mock trace and seed
    mock_trace = mocker.patch("tfscreen.tfmodel.inference.run_inference.trace")
    mock_seed = mocker.patch("tfscreen.tfmodel.inference.run_inference.seed")
    
    # Mock the trace object
    mock_t = mocker.Mock()
    mock_t.items.return_value = [
        ("site1", {"type": "deterministic"}),
        ("site2", {"type": "sample"})
    ]
    mock_trace.return_value.get_trace.return_value = mock_t
    
    sites = ri._get_site_names()
    assert "site1" in sites
    assert "site2" not in sites

def test_restore_checkpoint_error(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test")
    
    # Write a bad checkpoint
    with open(f"{out_prefix}_bad.pkl", "wb") as f:
        dill.dump({"main_key": 0, "svi_state": "not_a_state"}, f)
    
    with pytest.raises(ValueError, match="does not appear to have a saved svi_state"):
        ri._restore_checkpoint(f"{out_prefix}_bad.pkl")

def test_jitter_init_parameters():
    model = MockModel()
    ri = RunInference(model, seed=42)
    params = {"p": jnp.array(1.0), "a": jnp.array([1.0, 2.0])}
    
    jittered = ri._jitter_init_parameters(params, 0.1)
    assert not jnp.array_equal(jittered["p"], 1.0)
    
    # Zero jitter
    params = {"p": jnp.array(1.0)}
    jittered = ri._jitter_init_parameters(params, 0.0)
    assert jittered["p"] == 1.0



def test_write_losses_empty(tmpdir):
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_losses_empty")

    # 707: empty losses returns early
    ri._write_losses([], out_prefix)
    assert not os.path.exists(f"{out_prefix}_losses.bin")

def test_setup_svi_invalid_guide_type():
    """setup_svi raises ValueError for unrecognized guide_type."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    with pytest.raises(ValueError, match="not recognized"):
        ri.setup_svi(guide_type="bad_guide")


# -----------------------------------------------------------------------------
# Autoguide batch-safety guard in run_optimization
# -----------------------------------------------------------------------------

_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "smoke-tests", "test_data",
)


def _smoke_orchestrator(**kwargs):
    from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator
    return ModelOrchestrator(
        growth_df=os.path.join(_SMOKE_DATA, "growth-smoke.csv"),
        binding_df=os.path.join(_SMOKE_DATA, "binding-smoke.csv"),
        batch_size=6,
        **kwargs,
    )


def _svi_with_guide(ri, guide):
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import ClippedAdam
    return SVI(ri.model.jax_model, guide, ClippedAdam(step_size=1e-3),
               loss=Trace_ELBO())


def test_run_optimization_refuses_any_autoguide(tmpdir):
    """The guard keys off AutoGuide, not a particular guide class."""
    from numpyro.infer.autoguide import AutoNormal
    ri = RunInference(_smoke_orchestrator(theta_growth_noise="beta"), seed=0)
    svi = _svi_with_guide(ri, AutoNormal(ri.model.jax_model))
    with pytest.raises(ValueError, match="'AutoNormal' cannot fit"):
        ri.run_optimization(svi, max_num_epochs=1,
                            out_prefix=os.path.join(tmpdir, "refused"))


def test_check_autoguide_reports_dense_guide_size(capsys):
    from numpyro.infer.autoguide import AutoMultivariateNormal
    ri = RunInference(_smoke_orchestrator(), seed=0)
    ri._check_autoguide(AutoMultivariateNormal(ri.model.jax_model))
    assert "latent dimensions" in capsys.readouterr().out


def test_check_autoguide_warns_on_large_dense_guide(monkeypatch):
    from numpyro.infer.autoguide import AutoMultivariateNormal
    from tfscreen.tfmodel.inference import run_inference
    monkeypatch.setattr(run_inference, "_DENSE_GUIDE_WARN_GB", 0.0)
    ri = RunInference(_smoke_orchestrator(), seed=0)
    with pytest.warns(UserWarning, match="auto_low_rank_multivariate_normal"):
        ri._check_autoguide(AutoMultivariateNormal(ri.model.jax_model))


@pytest.mark.parametrize("guide_type,cls_name", [
    ("delta", "AutoDelta"),
    ("auto_normal", "AutoNormal"),
    ("auto_diagonal_normal", "AutoDiagonalNormal"),
    ("auto_multivariate_normal", "AutoMultivariateNormal"),
    ("auto_low_rank_multivariate_normal", "AutoLowRankMultivariateNormal"),
])
def test_setup_svi_builds_autoguide(guide_type, cls_name):
    from numpyro.infer import autoguide
    ri = RunInference(MockModel(), seed=42)
    svi = ri.setup_svi(guide_type=guide_type)
    assert isinstance(svi.guide, getattr(autoguide, cls_name))
    assert ri._guide_type == guide_type


@pytest.mark.parametrize("alias,canonical", [
    ("AutoNormal", "auto_normal"),
    ("AUTO_NORMAL", "auto_normal"),
    ("AutoDelta", "delta"),
    ("auto_delta", "delta"),
    ("autolowrankmultivariatenormal", "auto_low_rank_multivariate_normal"),
    ("Component", "component"),
])
def test_resolve_guide_type_aliases(alias, canonical):
    assert resolve_guide_type(alias) == canonical


def test_setup_svi_forwards_guide_kwargs():
    ri = RunInference(MockModel(), seed=42)
    svi = ri.setup_svi(guide_type="auto_low_rank_multivariate_normal",
                       guide_kwargs={"rank": 3, "init_scale": 0.05})
    assert svi.guide.rank == 3
    assert svi.guide._init_scale == 0.05
    assert ri._guide_kwargs == {"rank": 3, "init_scale": 0.05}


@pytest.mark.parametrize("guide_type,kwargs", [
    ("auto_normal", {"rank": 3}),
    ("delta", {"init_scale": 0.1}),
    ("component", {"init_scale": 0.1}),
])
def test_setup_svi_rejects_unaccepted_kwargs(guide_type, kwargs):
    ri = RunInference(MockModel(), seed=42)
    with pytest.raises(ValueError, match="not accepted"):
        ri.setup_svi(guide_type=guide_type, guide_kwargs=kwargs)


def test_setup_svi_component_rejects_init_values():
    ri = RunInference(MockModel(), seed=42)
    with pytest.raises(ValueError, match="init_values"):
        ri.setup_svi(guide_type="component", init_values={"a": 1.0})


def test_setup_svi_init_values_become_init_loc_fn():
    ri = RunInference(MockModel(), seed=42)
    values = {"global_p": 1.0}
    svi = ri.setup_svi(guide_type="auto_normal", init_values=values)
    assert svi.guide.init_loc_fn.keywords["values"] is values


def test_checkpoint_records_guide(tmpdir):
    ri = RunInference(MockModel(), seed=42)
    ri.setup_svi(guide_type="auto_normal", guide_kwargs={"init_scale": 0.2})
    out_prefix = os.path.join(tmpdir, "guide_meta")
    ri._write_checkpoint({"dummy": np.zeros(2)}, out_prefix)
    with open(f"{out_prefix}_checkpoint.pkl", "rb") as f:
        saved = dill.load(f)
    assert saved["guide_type"] == "auto_normal"
    assert saved["guide_kwargs"] == {"init_scale": 0.2}

def test_restore_checkpoint_current_step(tmpdir):
    """_restore_checkpoint restores _current_step when present in checkpoint."""
    model = MockModel()
    ri = RunInference(model, seed=42)
    out_prefix = os.path.join(tmpdir, "test_step")

    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    # Write checkpoint that includes 'current_step'
    checkpoint_file = f"{out_prefix}_checkpoint.pkl"
    checkpoint_data = {
        'svi_state': state,
        'main_key': ri._main_key,
        'current_step': 999,
    }
    import dill as _dill
    with open(checkpoint_file, "wb") as f:
        _dill.dump(checkpoint_data, f)

    restored = ri._restore_checkpoint(checkpoint_file)
    assert ri._current_step == 999
    assert isinstance(restored, SVIState)


# =============================================================================
# _map_params_to_constrained
# =============================================================================

def test_map_params_to_constrained_strips_auto_loc():
    """Returned dict has site names without the _auto_loc suffix."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    constrained = ri._map_params_to_constrained(map_params)

    assert all(not k.endswith("_auto_loc") for k in constrained)
    assert "global_p" in constrained
    assert "geno_p" in constrained


def test_map_params_to_constrained_ignores_non_auto_loc_keys():
    """Extra keys without _auto_loc are not included in the output."""
    model = LaplaceModel(num_genotype=3)
    ri, map_params = _laplace_map_params(model)

    poisoned = dict(map_params, some_extra_key=jnp.array(99.0))
    constrained = ri._map_params_to_constrained(poisoned)

    assert "some_extra_key" not in constrained


def test_map_params_to_constrained_shapes():
    """Output shapes match the model site shapes."""
    num_genotype = 5
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    constrained = ri._map_params_to_constrained(map_params)

    assert constrained["global_p"].shape == ()
    assert constrained["geno_p"].shape == (num_genotype,)


def test_map_params_to_constrained_values_finite():
    """Constrained values should be finite (no NaN/Inf from bijection)."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    constrained = ri._map_params_to_constrained(map_params)

    for k, v in constrained.items():
        assert np.all(np.isfinite(np.asarray(v))), f"Non-finite value in '{k}'"


# =============================================================================
# get_map_posteriors
# =============================================================================

def _batch_sized_map_params(model, batch_size):
    """MAP params whose genotype latent is batch-sized (pre-fix checkpoint)."""
    ri, map_params = _laplace_map_params(model)
    aliased = dict(map_params)
    aliased["geno_p_auto_loc"] = map_params["geno_p_auto_loc"][..., :batch_size]
    return ri, aliased


def test_get_map_posteriors_rejects_batch_sized_latent(tmpdir):
    """A genotype latent shorter than the library is refused, not clipped."""
    ri, aliased = _batch_sized_map_params(LaplaceModel(num_genotype=5), 3)
    with pytest.raises(ValueError, match="mini-batch position"):
        ri.get_map_posteriors(aliased, out_prefix=str(tmpdir.join("aliased")))


def test_get_laplace_posteriors_rejects_batch_sized_latent(tmpdir):
    """The Laplace path refuses it too, before computing the Hessian."""
    ri, aliased = _batch_sized_map_params(LaplaceModel(num_genotype=5), 3)
    with pytest.raises(ValueError, match="mini-batch position"):
        ri.get_laplace_posteriors(aliased,
                                  out_prefix=str(tmpdir.join("aliased")),
                                  num_posterior_samples=2)


def test_get_map_posteriors_creates_h5(tmpdir):
    """get_map_posteriors writes an HDF5 posterior file."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_map_posteriors_num_samples_is_one(tmpdir):
    """The HDF5 file contains exactly 1 sample."""
    num_genotype = 5
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_shapes"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == 1
        assert hf["global_p"].shape == (1,)
        assert hf["geno_p"].shape == (1, num_genotype)


def test_get_map_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype still produces correct output shapes."""
    num_genotype = 6
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_fwd"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix, forward_batch_size=2)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (1,)
        assert hf["geno_p"].shape == (1, num_genotype)


def test_get_map_posteriors_sites_to_save(tmpdir):
    """sites_to_save restricts which sites appear in the output."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_filtered"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix,
                          sites_to_save=["geno_p"])

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf
        assert hf["geno_p"].shape == (1, 4)


def test_get_map_posteriors_compression(tmpdir):
    """HDF5 datasets are gzip-compressed."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_compressed"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert hf[k].compression == "gzip", f"'{k}' should be gzip-compressed"


def test_get_map_posteriors_values_finite(tmpdir):
    """All values in the MAP posterior should be finite."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_finite"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert not np.any(np.isnan(hf[k][:])), f"NaN found in '{k}'"


def test_get_map_posteriors_matches_constrained(tmpdir):
    """The geno_p values in the h5 file match _map_params_to_constrained output."""
    num_genotype = 4
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("map_match"))
    ri.get_map_posteriors(map_params, out_prefix=out_prefix)

    constrained = ri._map_params_to_constrained(map_params)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        np.testing.assert_allclose(
            hf["global_p"][0], np.asarray(constrained["global_p"]), rtol=1e-5
        )
        np.testing.assert_allclose(
            hf["geno_p"][0], np.asarray(constrained["geno_p"]), rtol=1e-5
        )


# =============================================================================
# get_laplace_posteriors
# =============================================================================

# Shared data class and model for Laplace tests.  Uses a real numpyro model
# so that potential_energy, trace, and biject_to work end-to-end on a tiny
# problem (small D ≈ 1 + num_genotype parameters).

@struct.dataclass
class LaplaceData:
    num_genotype: int
    batch_idx: jnp.ndarray


def _laplace_jax_model(data, priors):
    """One global + one per-genotype parameter, no observations.

    Also emits a deterministic site that echoes the genotype index, so
    tests can directly verify genotype order survives forward-batching
    (rather than just checking output shapes).
    """
    global_p = numpyro.sample("global_p", dist.Normal(0., 1.))
    with numpyro.plate("shared_genotype_plate", data.num_genotype, dim=-1):
        numpyro.sample("geno_p", dist.Normal(global_p, 1.))
        numpyro.deterministic("geno_idx_check", data.batch_idx.astype(jnp.float32))


class LaplaceModel:
    """Minimal model wrapper compatible with RunInference."""

    def __init__(self, num_genotype=4):
        self.data = LaplaceData(num_genotype=num_genotype,
                                batch_idx=jnp.arange(num_genotype))
        self.priors = {}
        self.jax_model = _laplace_jax_model
        self.jax_model_guide = lambda data, priors: None  # not used in Laplace

    def get_batch(self, data, indices):
        return LaplaceData(num_genotype=len(indices), batch_idx=indices)

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def _laplace_map_params(model, seed=0):
    """Return AutoDelta MAP params (at prior initialisation, no optimisation)."""
    ri = RunInference(model, seed=seed)
    svi = ri.setup_svi(guide_type="delta")
    svi_state = svi.init(ri.get_key(), data=model.data, priors=model.priors)
    return ri, svi.get_params(svi_state)


def test_get_laplace_posteriors_creates_h5(tmpdir):
    """get_laplace_posteriors produces an HDF5 posterior file."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=10,
        sampling_batch_size=5,
        forward_batch_size=4,
    )

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_laplace_posteriors_output_shapes(tmpdir):
    """Output datasets have the expected shapes (num_samples, *param_shape)."""
    num_genotype = 5
    num_samples = 20
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_shapes"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=10,
        forward_batch_size=num_genotype,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)
        assert hf.attrs["num_samples"] == num_samples


def test_get_laplace_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype produces the same shapes as full-batch."""
    num_genotype = 6
    num_samples = 10
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_fwd"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=5,
        forward_batch_size=2,   # forces multiple forward batches
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_laplace_posteriors_genotype_order_preserved_ragged_batches(tmpdir):
    """forward_batch_size that doesn't evenly divide num_genotype must still
    produce correctly-ordered, unpadded output.

    Regression test for the genotype-batch lax.scan stitching logic shared
    with get_posteriors: 5 genotypes with forward_batch_size=2 forces chunks
    of (2, 2, 1), internally padded to (2, 2, 2) and trimmed back.
    """
    num_genotype = 5
    num_samples = 6
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_order"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=num_samples,
        forward_batch_size=2,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        idx_check = hf["geno_idx_check"][:]

    expected = np.tile(np.arange(num_genotype), (num_samples, 1))
    np.testing.assert_array_equal(idx_check, expected)


def test_get_laplace_posteriors_sampling_batching(tmpdir):
    """num_posterior_samples not evenly divisible by sampling_batch_size works."""
    num_genotype = 4
    num_samples = 7        # 7 = 3 + 3 + 1
    model = LaplaceModel(num_genotype=num_genotype)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_sbatch"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=num_samples,
        sampling_batch_size=3,
        forward_batch_size=num_genotype,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == num_samples
        assert hf["global_p"].shape[0] == num_samples
        assert hf["geno_p"].shape[0] == num_samples


def test_get_laplace_posteriors_non_auto_loc_keys_ignored(tmpdir):
    """Keys without _auto_loc suffix are silently excluded from the Hessian."""
    model = LaplaceModel(num_genotype=3)
    ri, map_params = _laplace_map_params(model)

    # Inject a key without the expected suffix
    poisoned = dict(map_params, some_other_key=jnp.array(99.0))

    out_prefix = str(tmpdir.join("laplace_extra"))
    # Should not raise; the extra key is ignored
    ri.get_laplace_posteriors(
        map_params=poisoned,
        out_prefix=out_prefix,
        num_posterior_samples=6,
        sampling_batch_size=6,
        forward_batch_size=3,
    )

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_laplace_posteriors_negative_eigenvalues_no_nan(tmpdir, mocker):
    """An indefinite Hessian (negative eigenvalues) must not produce NaN posteriors.

    Regression test: a MAP solution at a saddle point has negative Hessian
    eigenvalues.  The old code (jitter + jnp.linalg.inv) left the covariance
    non-PD in float32, causing jax.random.multivariate_normal to return all-NaN.
    The fix uses numpy float64 eigendecomposition, clamps negative eigenvalues
    to 1e-3, and samples via z @ L^T instead of multivariate_normal.

    LaplaceModel(num_genotype=2) has D=3 unconstrained parameters
    (1 global_p + 2 geno_p), so a 3x3 Hessian is injected.
    """
    model = LaplaceModel(num_genotype=2)
    ri, map_params = _laplace_map_params(model)

    # 3x3 symmetric matrix with one clearly negative eigenvalue (-5).
    bad_H = jnp.array([[10., 0., 0.],
                        [ 0., -5., 0.],
                        [ 0.,  0.,  8.]])

    # jax.hessian(fn)(x): patch so that jax.hessian(pe_fn) returns a callable
    # that ignores x and returns bad_H regardless of fn/x.
    mocker.patch("jax.hessian", return_value=lambda x: bad_H)

    out_prefix = str(tmpdir.join("laplace_negeig"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=20,
        sampling_batch_size=10,
        forward_batch_size=2,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert not np.any(np.isnan(hf[k][:])), f"NaN found in posterior key '{k}'"


# =============================================================================
# get_nuts_posteriors
# =============================================================================

def _fake_mcmc_samples(num_samples, num_genotype):
    """Return a samples dict in the shape expected by LaplaceModel."""
    return {
        "global_p": jnp.zeros((num_samples,)),
        "geno_p": jnp.zeros((num_samples, num_genotype)),
    }


def test_get_nuts_posteriors_creates_h5(tmpdir):
    """get_nuts_posteriors writes an HDF5 posterior file."""
    num_genotype = 4
    num_samples = 8
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix)

    assert os.path.exists(f"{out_prefix}_posterior.h5")


def test_get_nuts_posteriors_output_shapes(tmpdir):
    """Output datasets have the expected shapes (num_samples, *site_shape)."""
    num_genotype = 5
    num_samples = 12
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_shapes"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           forward_batch_size=num_genotype)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)
        assert hf.attrs["num_samples"] == num_samples


def test_get_nuts_posteriors_forward_batching(tmpdir):
    """forward_batch_size < num_genotype produces correct shapes."""
    num_genotype = 6
    num_samples = 8
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_fwd"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           forward_batch_size=2)  # forces 3 forward batches

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf["global_p"].shape == (num_samples,)
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_nuts_posteriors_genotype_order_preserved_ragged_batches(tmpdir):
    """forward_batch_size that doesn't evenly divide num_genotype must still
    produce correctly-ordered, unpadded output (same lax.scan stitching path
    as get_posteriors / get_laplace_posteriors)."""
    num_genotype = 7
    num_samples = 5
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_order"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           forward_batch_size=3)  # chunks of 3, 3, 1

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        idx_check = hf["geno_idx_check"][:]

    expected = np.tile(np.arange(num_genotype), (num_samples, 1))
    np.testing.assert_array_equal(idx_check, expected)


def test_get_nuts_posteriors_num_samples_in_attrs(tmpdir):
    """num_samples attribute in HDF5 reflects the actual sample count."""
    num_genotype = 3
    num_samples = 7
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_attr"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert hf.attrs["num_samples"] == num_samples


def test_get_nuts_posteriors_sites_to_save(tmpdir):
    """sites_to_save restricts which sites appear in the HDF5 file."""
    num_genotype = 5
    num_samples = 8
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_filtered"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix,
                           sites_to_save=["geno_p"])

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf
        assert hf["geno_p"].shape == (num_samples, num_genotype)


def test_get_nuts_posteriors_compression(tmpdir):
    """HDF5 datasets written by get_nuts_posteriors are gzip-compressed."""
    num_genotype = 4
    num_samples = 6
    model = LaplaceModel(num_genotype=num_genotype)
    ri = RunInference(model, seed=0)
    samples = _fake_mcmc_samples(num_samples, num_genotype)

    out_prefix = str(tmpdir.join("nuts_compressed"))
    ri.get_nuts_posteriors(samples, out_prefix=out_prefix)

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert hf[k].compression == "gzip", (
                f"dataset '{k}' should be gzip-compressed"
            )


def test_get_laplace_posteriors_sites_to_save(tmpdir):
    """sites_to_save restricts which sites appear in the Laplace HDF5 file."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_filtered"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=8,
        sampling_batch_size=4,
        forward_batch_size=4,
        sites_to_save=["geno_p"],
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert "geno_p" in hf
        assert "global_p" not in hf
        assert hf["geno_p"].shape == (8, 4)


def test_get_laplace_posteriors_compression(tmpdir):
    """HDF5 datasets written by get_laplace_posteriors are gzip-compressed."""
    model = LaplaceModel(num_genotype=4)
    ri, map_params = _laplace_map_params(model)

    out_prefix = str(tmpdir.join("laplace_compressed"))
    ri.get_laplace_posteriors(
        map_params=map_params,
        out_prefix=out_prefix,
        num_posterior_samples=8,
        sampling_batch_size=8,
        forward_batch_size=4,
    )

    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        for k in hf.keys():
            assert hf[k].compression == "gzip", (
                f"dataset '{k}' should be gzip-compressed"
            )


# =============================================================================
# _write_epoch_checkpoint
# =============================================================================

def _make_ri_with_epoch_dir(tmpdir):
    """Return a RunInference with _epoch_checkpoints_dir already set."""
    model = MockModel()
    ri = RunInference(model, seed=0)
    epoch_dir = str(tmpdir.join("checkpoints"))
    os.makedirs(epoch_dir, exist_ok=True)
    ri._epoch_checkpoints_dir = epoch_dir
    return ri


def test_write_epoch_checkpoint_creates_file(tmpdir):
    """_write_epoch_checkpoint writes a pkl file with the correct zero-padded name."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=42)

    expected = os.path.join(ri._epoch_checkpoints_dir, "0000042_checkpoint.pkl")
    assert os.path.exists(expected)


def test_write_epoch_checkpoint_file_format(tmpdir):
    """Epoch number is zero-padded to 7 digits."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=1)
    ri._write_epoch_checkpoint(state, epoch=999999)

    assert os.path.exists(os.path.join(ri._epoch_checkpoints_dir, "0000001_checkpoint.pkl"))
    assert os.path.exists(os.path.join(ri._epoch_checkpoints_dir, "0999999_checkpoint.pkl"))


def test_write_epoch_checkpoint_raises_if_exists(tmpdir):
    """_write_epoch_checkpoint raises FileExistsError if the target file already exists."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=7)

    with pytest.raises(FileExistsError, match="0000007_checkpoint.pkl"):
        ri._write_epoch_checkpoint(state, epoch=7)


def test_write_epoch_checkpoint_saves_correct_fields(tmpdir):
    """Checkpoint file contains main_key, svi_state, and current_step."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._current_step = 1234
    ri._write_epoch_checkpoint(state, epoch=5)

    path = os.path.join(ri._epoch_checkpoints_dir, "0000005_checkpoint.pkl")
    with open(path, "rb") as f:
        data = dill.load(f)

    assert "main_key" in data
    assert "svi_state" in data
    assert "current_step" in data
    assert data["current_step"] == 1234


def test_write_epoch_checkpoint_atomic_no_tmp_left(tmpdir):
    """No .tmp file is left behind after a successful write."""
    ri = _make_ri_with_epoch_dir(tmpdir)
    try:
        state = SVIState(None, None, None)
    except Exception:
        state = SVIState(None, None)

    ri._write_epoch_checkpoint(state, epoch=3)

    tmp_path = os.path.join(ri._epoch_checkpoints_dir, "0000003_checkpoint.pkl.tmp")
    assert not os.path.exists(tmp_path)


# =============================================================================
# compute_hessian_sigmas: takes constrained MAP values (svi.get_params)
# =============================================================================

def test_compute_hessian_sigmas_takes_constrained_values():
    """
    run_optimization returns svi.get_params(), which AutoDelta constrains.
    compute_hessian_sigmas must map those back to unconstrained space; it
    used to take them as unconstrained, so a positive site was evaluated at
    exp(value) (the pre-fit's k_scale came out NaN on the count likelihood,
    2026-09-27). With s ~ LogNormal(1, 0.5), log s is Normal(1, 0.5): at the
    MAP s = e the unconstrained sigma is 0.5 and the constrained one e / 2.
    """
    def model(data, priors):
        numpyro.sample("s", dist.LogNormal(1.0, 0.5))
        numpyro.sample("x", dist.Normal(0.0, 2.0))

    m = MockModel(num_genotype=1)
    m.jax_model = model
    ri = RunInference(m, seed=0)

    # confirm what AutoDelta's get_params hands over
    svi = ri.setup_svi(guide_type="delta",
                       init_values={"s": float(np.e), "x": 0.0})
    state = svi.init(ri.get_key(), data=m.data, priors=m.priors)
    params = svi.get_params(state)
    assert float(params["s_auto_loc"]) == pytest.approx(np.e, rel=1e-5)

    out = ri.compute_hessian_sigmas(params)
    assert float(out["s"]["map"]) == pytest.approx(np.e, rel=1e-5)
    assert float(out["s"]["sigma"]) == pytest.approx(np.e * 0.5, rel=1e-3)
    assert float(out["x"]["sigma"]) == pytest.approx(2.0, rel=1e-3)


def test_run_nuts_starts_at_given_values(mocker):
    """
    run_nuts starts every chain at the given constrained site values and
    falls back to prior medians (it once handed initialize_model's ParamInfo
    tuple to init_to_value, which matched no site).
    """
    from unittest.mock import MagicMock

    import numpyro.distributions as dist

    ri = RunInference.__new__(RunInference)
    ri._seed = 0
    ri.model = MagicMock()
    ri.model.data.num_genotype = 3
    nuts = mocker.patch("numpyro.infer.NUTS")
    mcmc = mocker.patch("numpyro.infer.MCMC")
    mcmc.return_value.get_extra_fields.return_value = {}
    ri.run_nuts(num_warmup=1, num_samples=1, init_values={"a": 2.5},
                dense_mass=True)
    assert nuts.call_args.kwargs["dense_mass"] is True
    strategy = nuts.call_args.kwargs["init_strategy"]
    site = {"type": "sample", "is_observed": False, "name": "a",
            "fn": dist.Normal(0.0, 1.0), "value": None, "kwargs": {}}
    assert float(strategy(site)) == 2.5
    # the full data set goes in as one batch
    ri.model.get_batch.assert_called_once()


def test_report_nuts_diagnostics(capsys):
    """Worst split R-hat and least n_eff, skipping constant sites."""
    from unittest.mock import MagicMock

    rng = np.random.default_rng(0)
    mixed = rng.normal(size=(2, 200))
    stuck = np.stack([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    mcmc = MagicMock()
    mcmc.get_samples.return_value = {"good": mixed, "bad": stuck,
                                     "fixed": np.zeros((2, 200))}
    RunInference._report_nuts_diagnostics(mcmc)
    out = capsys.readouterr().out
    assert "max split R-hat" in out and "(bad)" in out
    assert "inf" not in out


# ---------------------------------------------------------------------------
# laplace_eigenvalue_floor
# ---------------------------------------------------------------------------

def test_laplace_eigenvalue_floor_uses_prior_curvature():
    """A direction flatter than the prior (or negative) gets the prior's curvature."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    prior_precision = np.array([4.0, 100.0, 1.0])
    curvature = np.einsum("ij,i,ij->j", q, prior_precision, q)
    eigenvalues = np.array([-50.0, 1e6, 0.5 * curvature[2]])
    floored, n = laplace_eigenvalue_floor(eigenvalues, q, prior_precision)
    assert n == 2
    assert floored[0] == pytest.approx(curvature[0])
    assert floored[1] == pytest.approx(1e6)          # well-determined: kept
    assert floored[2] == pytest.approx(curvature[2])


def test_laplace_eigenvalue_floor_bounds_leak_into_other_elements():
    """
    A negative eigenvalue no longer inflates every element it touches.

    The old flat floor of 1e-3 gave such a direction a variance of 1000, so
    an element with a 3% share of it (the growth slope m, seed 8 of the
    two-stage grid) had an SD near 1.
    """
    theta = 0.03
    v = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    eigenvalues = np.array([-10.0, 1e6])     # direction 0 mostly element 0
    prior_precision = np.array([1.0, 1e4])
    floored, _ = laplace_eigenvalue_floor(eigenvalues, v, prior_precision)
    cov = v @ np.diag(1.0 / floored) @ v.T
    old = v @ np.diag(1.0 / np.maximum(eigenvalues, 1e-3)) @ v.T
    assert np.sqrt(old[1, 1]) > 0.5
    assert np.sqrt(cov[1, 1]) < 0.05


def test_laplace_eigenvalue_floor_minimum():
    floored, n = laplace_eigenvalue_floor(np.array([-1.0]), np.eye(1),
                                          np.array([0.0]))
    assert floored[0] == pytest.approx(LAPLACE_MIN_EIGENVALUE)
    assert n == 1


# =============================================================================
# Block (per-genotype) Laplace
# =============================================================================

@struct.dataclass
class BlockData:
    batch_idx: jnp.ndarray
    y: jnp.ndarray            # (num_genotype, 2) observations
    num_genotype: int = struct.field(pytree_node=False, default=0)
    coupled: bool = struct.field(pytree_node=False, default=False)


def _block_jax_model(data, priors):
    """Shared mu; per-genotype (a, b) in a library-sized plate, sliced by
    batch_idx; two observations per genotype that correlate a and b. With
    ``coupled`` each observation also sees the library mean of a, so the
    genotype blocks are not separable."""
    mu = numpyro.sample("mu", dist.Normal(0., 1.))
    with numpyro.plate("block_genotype_plate", data.num_genotype, dim=-1):
        a = numpyro.sample("a", dist.Normal(mu, 1.))
        b = numpyro.sample("b", dist.Normal(0., 1.))
    a_b, b_b = a[data.batch_idx], b[data.batch_idx]
    if data.coupled:
        a_b = a_b + jnp.mean(a)
    y = data.y[data.batch_idx]
    with numpyro.plate("obs_genotype_plate", data.batch_idx.shape[0], dim=-1):
        numpyro.sample("y1", dist.Normal(a_b + b_b, 0.3), obs=y[:, 0])
        numpyro.sample("y2", dist.Normal(a_b - 2.0 * b_b, 0.2), obs=y[:, 1])


class BlockModel:
    def __init__(self, num_genotype=6, coupled=False, seed=3):
        rng = np.random.default_rng(seed)
        self.data = BlockData(num_genotype=num_genotype,
                              batch_idx=jnp.arange(num_genotype),
                              y=jnp.asarray(rng.normal(size=(num_genotype, 2))),
                              coupled=coupled)
        self.priors = {}
        self.jax_model = _block_jax_model
        self.jax_model_guide = lambda data, priors: None

    def get_batch(self, data, indices):
        return data.replace(batch_idx=jnp.asarray(indices))

    def get_random_idx(self, key=None, num_batches=1):
        if num_batches == 1:
            return np.array([0])
        return np.zeros((num_batches, 1), dtype=int)


def _block_setup(model):
    import jax.flatten_util
    ri, map_params = _laplace_map_params(model)
    unc = {k[:-len("_auto_loc")]: jnp.asarray(v) for k, v in map_params.items()
           if k.endswith("_auto_loc")}
    flat, unravel = jax.flatten_util.ravel_pytree(unc)
    G = model.data.num_genotype
    kw = {"priors": model.priors,
          "data": ri._unscaled_batch(model.data, jnp.arange(G))}
    return ri, map_params, unc, flat, unravel, kw


def test_block_laplace_factors_match_full_hessian_blocks():
    """Each block's covariance is the inverse of the matching diagonal block
    of the full Hessian: given mu, genotypes do not couple."""
    from numpyro.infer.util import potential_energy
    model = BlockModel()
    ri, _, unc, flat, unravel, kw = _block_setup(model)
    dim_map = ri._get_genotype_dim_map()
    elem_idx, L = ri.block_laplace_factors(unc, dim_map, model.data, flat,
                                           unravel, kw)
    assert elem_idx.shape == (6, 2) and L.shape == (6, 2, 2)
    pe = lambda f: potential_energy(model.jax_model, [], kw, unravel(f))
    H = ri._chunked_hessian(pe, flat, 64)
    for g in range(6):
        hb = H[np.ix_(elem_idx[g], elem_idx[g])]
        np.testing.assert_allclose(L[g] @ L[g].T, np.linalg.inv(hb),
                                   rtol=1e-4, atol=1e-6)
        # the observations correlate a and b, so the blocks are not diagonal
        assert abs(hb[0, 1]) > 1.0


def test_block_laplace_factors_chunking_matches():
    model = BlockModel(num_genotype=7)
    ri, _, unc, flat, unravel, kw = _block_setup(model)
    dim_map = ri._get_genotype_dim_map()
    e1, L1 = ri.block_laplace_factors(unc, dim_map, model.data, flat, unravel, kw)
    e2, L2 = ri.block_laplace_factors(unc, dim_map, model.data, flat, unravel, kw,
                                      genotype_chunk_size=3)
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_allclose(np.einsum("gij,gkj->gik", L1, L1),
                               np.einsum("gij,gkj->gik", L2, L2), rtol=1e-5)


def test_block_laplace_factors_refuse_coupled_genotypes():
    model = BlockModel(coupled=True)
    ri, _, unc, flat, unravel, kw = _block_setup(model)
    with pytest.raises(ValueError, match="coupled"):
        ri.block_laplace_factors(unc, ri._get_genotype_dim_map(), model.data,
                                 flat, unravel, kw)


def test_get_laplace_posteriors_block_holds_shared_and_varies_genotypes(tmpdir):
    model = BlockModel()
    ri, map_params, *_ = _block_setup(model)
    out = str(tmpdir.join("block"))
    ri.get_laplace_posteriors(map_params, out_prefix=out,
                              num_posterior_samples=400,
                              sampling_batch_size=100,
                              forward_batch_size=6,
                              block_genotypes=True)
    with h5py.File(f"{out}_posterior.h5", "r") as hf:
        mu, a = hf["mu"][...], hf["a"][...]
    np.testing.assert_allclose(mu, float(map_params["mu_auto_loc"]), atol=1e-6)
    assert a.shape == (400, 6)
    assert np.all(a.std(axis=0) > 0.01)
    np.testing.assert_allclose(a.mean(axis=0),
                               np.asarray(map_params["a_auto_loc"]), atol=0.1)


def _arrowhead_covariance(elem_idx, L_blocks, shared_idx, L_shared, response):
    """Dense covariance implied by the arrowhead factors."""
    G, B = elem_idx.shape
    D = G * B + shared_idx.size
    cov = np.zeros((D, D))
    css = L_shared @ L_shared.T
    cov[np.ix_(shared_idx, shared_idx)] = css
    for g in range(G):
        Mg = np.asarray(response[g], dtype=float)
        cov[np.ix_(elem_idx[g], shared_idx)] = -Mg @ css
        cov[np.ix_(shared_idx, elem_idx[g])] = -(Mg @ css).T
        for h in range(G):
            Mh = np.asarray(response[h], dtype=float)
            block = Mg @ css @ Mh.T
            if g == h:
                block = block + L_blocks[g] @ L_blocks[g].T
            cov[np.ix_(elem_idx[g], elem_idx[h])] = block
    return cov


@pytest.mark.parametrize("chunk", [None, 3])
def test_arrowhead_laplace_matches_full_laplace(chunk):
    """Genotype blocks plus their coupling to the shared mu reproduce the
    inverse of the full Hessian, chunked or not (the chunked sum counts the
    priors once per chunk and must take the surplus back out)."""
    from numpyro.infer.util import potential_energy
    model = BlockModel(num_genotype=7)
    ri, _, unc, flat, unravel, kw = _block_setup(model)
    factors = ri.arrowhead_laplace_factors(unc, ri._get_genotype_dim_map(),
                                           model.data, flat, unravel, kw,
                                           genotype_chunk_size=chunk)
    elem_idx, L_blocks, shared_idx, L_shared, response = factors
    assert shared_idx.size == 1 and response.shape == (7, 2, 1)
    pe = lambda f: potential_energy(model.jax_model, [], kw, unravel(f))
    H = ri._chunked_hessian(pe, flat, 64)
    np.testing.assert_allclose(_arrowhead_covariance(*factors),
                               np.linalg.inv(H), rtol=1e-4, atol=1e-6)


def test_get_laplace_posteriors_block_shared_varies_shared(tmpdir):
    model = BlockModel()
    ri, map_params, unc, flat, unravel, kw = _block_setup(model)
    from numpyro.infer.util import potential_energy
    pe = lambda f: potential_energy(model.jax_model, [], kw, unravel(f))
    cov = np.linalg.inv(ri._chunked_hessian(pe, flat, 64))
    out = str(tmpdir.join("arrow"))
    ri.get_laplace_posteriors(map_params, out_prefix=out,
                              num_posterior_samples=4000,
                              sampling_batch_size=500,
                              forward_batch_size=6,
                              block_genotypes=True, block_shared=True)
    with h5py.File(f"{out}_posterior.h5", "r") as hf:
        mu, a = hf["mu"][...], hf["a"][...]
    # ravel_pytree orders sites a, b, mu
    i_mu, i_a = 12, np.arange(6)
    assert mu.std() == pytest.approx(np.sqrt(cov[i_mu, i_mu]), rel=0.1)
    # a follows mu: its draws correlate with mu's as the full Laplace says
    sd_a = np.sqrt(np.diag(cov)[i_a])
    want = cov[i_mu, i_a] / (np.sqrt(cov[i_mu, i_mu]) * sd_a)
    got = [np.corrcoef(mu, a[:, g])[0, 1] for g in range(6)]
    np.testing.assert_allclose(got, want, atol=0.08)


def test_block_shared_needs_block_genotypes(tmpdir):
    model = BlockModel()
    ri, map_params, *_ = _block_setup(model)
    with pytest.raises(ValueError, match="block_genotypes"):
        ri.get_laplace_posteriors(map_params, out_prefix=str(tmpdir.join("x")),
                                  block_shared=True)


def test_unscaled_batch_removes_minibatch_scale():
    """A model configured with batch_size below the library weights each
    growth likelihood by num_genotype / batch_size even at full batch;
    the Laplace must see each observation once."""
    orch = _smoke_orchestrator()
    ri = RunInference(orch, seed=0)
    G = orch.data.num_genotype
    data = jax.device_put(orch.data)
    scaled = orch.get_batch(data, jnp.arange(G))
    assert float(jnp.max(scaled.growth.scale_vector)) > 1.0
    unscaled = ri._unscaled_batch(data, jnp.arange(G))
    np.testing.assert_array_equal(np.asarray(unscaled.growth.scale_vector), 1.0)
