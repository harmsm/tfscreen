"""
Guide selection against the real model (smoke data): autoguide refusal for
batch-dependent latents, and fit -> checkpoint -> restore -> posterior
round trips for each autoguide.
"""
import os

import dill
import h5py
import pytest

from tfscreen.tfmodel.inference.run_inference import RunInference
from tfscreen.tfmodel.model_orchestrator import ModelOrchestrator


_SMOKE_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "smoke-tests", "test_data",
)
_GROWTH_CSV = os.path.join(_SMOKE_DATA, "growth-smoke.csv")
_BINDING_CSV = os.path.join(_SMOKE_DATA, "binding-smoke.csv")


def _orchestrator(**kwargs):
    return ModelOrchestrator(growth_df=_GROWTH_CSV, binding_df=_BINDING_CSV,
                             batch_size=6, **kwargs)


@pytest.mark.parametrize("guide_type", ["delta", "auto_normal",
                                        "auto_multivariate_normal"])
def test_autoguide_refused_for_batch_dependent_latent(tmpdir, guide_type):
    """Fitting is refused before the first step; building the guide is not."""
    ri = RunInference(_orchestrator(theta_growth_noise="beta"), seed=0)
    svi = ri.setup_svi(guide_type=guide_type)
    with pytest.raises(ValueError, match="theta_growth_noise_dist"):
        ri.run_optimization(svi, max_num_epochs=1,
                            out_prefix=os.path.join(tmpdir, "refused"))


def test_component_guide_allowed_for_batch_dependent_latent(tmpdir):
    """The component guide slices library-sized params itself."""
    ri = RunInference(_orchestrator(theta_growth_noise="beta"), seed=0)
    svi = ri.setup_svi(adam_step_size=1e-3, guide_type="component")
    ri.run_optimization(svi, max_num_epochs=1, init_param_jitter=0.0,
                        out_prefix=os.path.join(tmpdir, "component"),
                        epoch_checkpoint_interval=None)


@pytest.mark.slow
@pytest.mark.parametrize("guide_type,guide_kwargs", [
    ("auto_normal", None),
    ("auto_diagonal_normal", None),
    ("auto_multivariate_normal", None),
    ("auto_low_rank_multivariate_normal", {"rank": 2}),
])
def test_autoguide_round_trip(tmpdir, guide_type, guide_kwargs):
    """Fit a few minibatched epochs, restore from the checkpoint, sample."""
    orchestrator = _orchestrator()
    num_genotype = orchestrator.data.num_genotype
    out_prefix = os.path.join(tmpdir, guide_type)

    ri = RunInference(orchestrator, seed=0)
    svi = ri.setup_svi(adam_step_size=1e-3, guide_type=guide_type,
                       guide_kwargs=guide_kwargs)
    _, params, _ = ri.run_optimization(svi, max_num_epochs=2,
                                       out_prefix=out_prefix,
                                       init_param_jitter=0.0,
                                       epoch_checkpoint_interval=None)

    with open(f"{out_prefix}_checkpoint.pkl", "rb") as f:
        assert dill.load(f)["guide_type"] == guide_type

    ri2 = RunInference(_orchestrator(), seed=1)
    svi2, state2 = ri2.restore_svi_from_checkpoint(f"{out_prefix}_checkpoint.pkl")
    assert type(svi2.guide) is type(svi.guide)
    restored = svi2.get_params(state2)
    assert set(restored) == set(params)

    ri2.get_posteriors(svi2, state2, out_prefix, num_posterior_samples=4,
                       sampling_batch_size=4, forward_batch_size=4)
    with h5py.File(f"{out_prefix}_posterior.h5", "r") as hf:
        assert not any(k.startswith("_") for k in hf)
        assert hf["dk_geno_offset"].shape == (4, num_genotype)
