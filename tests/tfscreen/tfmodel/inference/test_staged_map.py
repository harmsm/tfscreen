"""Tests for tfscreen.tfmodel.inference.staged_map (the staged level-offset MAP)."""
import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from tfscreen.tfmodel.inference import staged_map as S

OFF = "sample_offset_offset"
SIG = "sample_offset_sigma"


# ---------------------------------------------------------------------------
# HeldModel
# ---------------------------------------------------------------------------

class _Toy:
    """A minimal model object: two latents, an attribute to delegate."""

    data = "the data"
    priors = "the priors"
    jax_model_guide = "the guide"

    @staticmethod
    def jax_model():
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([3]).to_event(1))
        return a, b


def test_held_model_conditions_sites():
    held = S.HeldModel(_Toy(), {"b": np.array([1.0, 2.0, 3.0])})
    tr = trace(seed(held.jax_model, 0)).get_trace()
    assert not tr["a"]["is_observed"]
    assert tr["b"]["is_observed"]
    assert np.allclose(tr["b"]["value"], [1.0, 2.0, 3.0])
    # held values become jax arrays (the model indexes them with tracers)
    assert isinstance(held.held["b"], jnp.ndarray)


def test_held_model_delegates():
    held = S.HeldModel(_Toy(), {"a": 0.0})
    assert held.data == "the data"
    assert held.priors == "the priors"
    assert held.jax_model_guide == "the guide"


# ---------------------------------------------------------------------------
# use_stages
# ---------------------------------------------------------------------------

def _orch(sample_offset="level", sigma_fixed=0.0, sigma_prior_scale=0.2):
    so = SimpleNamespace(sigma_fixed=sigma_fixed, sigma_prior_scale=sigma_prior_scale)
    growth = SimpleNamespace(num_replicate=1, num_time=2, num_condition_pre=1,
                             num_condition_sel=2, num_titrant_name=1,
                             num_titrant_conc=3)
    return SimpleNamespace(settings={"sample_offset": sample_offset},
                           jax_model=_Toy.jax_model,
                           priors=SimpleNamespace(growth=SimpleNamespace(sample_offset=so)),
                           data=SimpleNamespace(growth=growth))


@pytest.mark.parametrize("kwargs,expected", [
    (dict(stage_offsets="auto", analysis_method="map"), True),
    (dict(stage_offsets="auto", analysis_method="svi"), False),
    (dict(stage_offsets="auto", analysis_method="map", checkpoint_file="c.pkl"), False),
    (dict(stage_offsets="auto", analysis_method="map", init_from="p.npz"), False),
    (dict(stage_offsets="off", analysis_method="map"), False),
    (dict(stage_offsets="on", analysis_method="map"), True),
])
def test_use_stages(kwargs, expected):
    stage = kwargs.pop("stage_offsets")
    method = kwargs.pop("analysis_method")
    assert S.use_stages(stage, _orch(), method, **kwargs) is expected


def test_use_stages_auto_needs_level():
    assert S.use_stages("auto", _orch(sample_offset="zero"), "map") is False


@pytest.mark.parametrize("orch,method,kwargs,match", [
    (_orch(sample_offset="zero"), "map", {}, "level"),
    (_orch(), "svi", {}, "analysis_method"),
    (_orch(), "map", {"init_from": "p.npz"}, "init_from"),
])
def test_use_stages_on_refuses(orch, method, kwargs, match):
    with pytest.raises(ValueError, match=match):
        S.use_stages("on", orch, method, **kwargs)


def test_use_stages_bad_choice():
    with pytest.raises(ValueError, match="stage_offsets"):
        S.use_stages("maybe", _orch(), "map")


def test_offset_sites_sigma_hold():
    assert S.offset_sites(_orch())[2] == 0.2
    assert S.offset_sites(_orch(sigma_fixed=0.17))[2] is None


# ---------------------------------------------------------------------------
# run_staged_map, with a fake fit
# ---------------------------------------------------------------------------

class _FakeRI:
    def __init__(self, model):
        self.model = model

    def site_values(self, values):
        return {k[:-len("_auto_loc")] if k.endswith("_auto_loc") else k: v
                for k, v in values.items()}


class _FakeMap:
    """Stands in for _run_map: records each call and writes a params file."""

    def __init__(self):
        self.calls = []

    def __call__(self, ri, init_values=None, checkpoint_file=None,
                 out_prefix=None, label=None, **kwargs):
        held = ri.model.held if isinstance(ri.model, S.HeldModel) else {}
        self.calls.append(dict(held=held, init_values=init_values,
                               checkpoint_file=checkpoint_file,
                               out_prefix=out_prefix, kwargs=kwargs))
        if OFF in held:                       # stage 1: everything but offsets
            out = {"k_auto_loc": np.array(0.01), "theta_auto_loc": np.ones(4)}
        elif "k" in held:                     # stage 2: offsets only
            out = {f"{OFF}_auto_loc": np.linspace(-0.1, 0.1, 6)}
        else:                                 # stage 3: everything
            out = {"k_auto_loc": np.array(0.02), f"{OFF}_auto_loc": np.zeros(6)}
        np.savez(f"{out_prefix}_params.npz", **out)
        return "state", out, True


def _run(tmp_path, orch=None, **kw):
    fake = _FakeMap()
    result = S.run_staged_map(orch or _orch(), make_ri=_FakeRI, run_map=fake,
                              guesses={"k": 0.0}, out_prefix=str(tmp_path / "fit"),
                              map_kwargs={"adam_step_size": 1e-3, "patience": 3},
                              **kw)
    return fake, result


def test_three_stages(tmp_path):
    fake, result = _run(tmp_path)
    s1, s2, s3 = fake.calls
    # stage 1: offsets held at 0, learned sigma held at its prior scale
    assert set(s1["held"]) == {OFF, SIG}
    assert np.allclose(s1["held"][OFF], 0.0) and s1["held"][OFF].shape == (12,)
    assert float(s1["held"][SIG]) == pytest.approx(0.2)
    assert s1["out_prefix"].endswith("fit_stage1")
    # stage 2: everything from stage 1 held, offsets free
    assert set(s2["held"]) == {"k", "theta", SIG}
    assert s2["out_prefix"].endswith("fit_stage2")
    # stage 3: no holds, starts at stage 1 + stage 2's offsets, small step
    assert s3["held"] == {}
    assert s3["out_prefix"].endswith("fit")
    assert s3["kwargs"]["adam_step_size"] == S.DEFAULT_STAGED_STEP_SIZE
    assert float(s3["init_values"]["k"]) == 0.01
    assert np.allclose(s3["init_values"][OFF], np.linspace(-0.1, 0.1, 6))
    assert result[2] is True


def test_fixed_sigma_not_held(tmp_path):
    fake, _ = _run(tmp_path, orch=_orch(sigma_fixed=0.17))
    assert set(fake.calls[0]["held"]) == {OFF}
    assert SIG not in fake.calls[1]["held"]


def test_staged_step_size_and_stage_kwargs(tmp_path):
    fake = _FakeMap()
    S.run_staged_map(_orch(), _FakeRI, fake, {}, str(tmp_path / "fit"),
                     map_kwargs={"adam_step_size": 1e-3, "patience": 3},
                     stage_map_kwargs={"adam_step_size": 1e-2, "patience": 1},
                     staged_step_size=5e-5)
    assert fake.calls[0]["kwargs"] == {"adam_step_size": 1e-2, "patience": 1}
    assert fake.calls[2]["kwargs"] == {"adam_step_size": 5e-5, "patience": 3}


def test_finished_stage_reused(tmp_path):
    _run(tmp_path)
    os.remove(tmp_path / "fit_params.npz")
    fake, _ = _run(tmp_path)
    # stages 1 and 2 are read back; only the joint stage runs
    assert len(fake.calls) == 1
    assert fake.calls[0]["held"] == {}


def test_interrupted_stage_resumes(tmp_path):
    open(tmp_path / "fit_stage1_checkpoint.pkl", "w").close()
    fake, _ = _run(tmp_path)
    assert fake.calls[0]["checkpoint_file"] == str(tmp_path / "fit_stage1_checkpoint.pkl")
    assert fake.calls[0]["init_values"] is None
    assert fake.calls[1]["checkpoint_file"] is None
