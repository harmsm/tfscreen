"""Tests for tfscreen.simulate.toy_thermo.sampling."""

import numpy as np
import pytest

from tfscreen.simulate.toy_thermo.sampling import sample_effects, _draw
from tfscreen.simulate.toy_thermo.genotypes import MutationEffects, STATES


def test_draw_sd_only_and_mean_sd():
    rng = np.random.default_rng(0)
    out = _draw(rng, {"HD": 1.0, "LE": (5.0, 0.0)})
    assert set(out) == set(STATES)
    assert out["LE"] == 5.0          # sd == 0 -> exactly the mean
    assert out["H"] == 0.0           # unspecified -> 0
    assert out["L"] == 0.0


def test_draw_zero_sd_is_deterministic():
    rng = np.random.default_rng(0)
    out = _draw(rng, {"HD": 0.0})
    assert out["HD"] == 0.0


def test_sample_effects_returns_catalog_with_all_mutations():
    eff = sample_effects(["A", "B", "C"],
                         effect_sd={"HD": 1.0, "LE": 1.0},
                         rng=np.random.default_rng(0))
    assert isinstance(eff, MutationEffects)
    assert set(eff.mutations) == {"A", "B", "C"}


def test_sample_effects_reproducible():
    kw = dict(mutations=["A", "B"],
              effect_sd={"HD": 1.0, "H": 0.5, "L": 0.5, "LE": 1.0},
              epistasis_sd={"HD": 0.3})
    a = sample_effects(rng=np.random.default_rng(42), **kw)
    b = sample_effects(rng=np.random.default_rng(42), **kw)
    c = sample_effects(rng=np.random.default_rng(7), **kw)
    assert a.ddg_for("A/B") == b.ddg_for("A/B")
    assert a.ddg_for("A/B") != c.ddg_for("A/B")


def test_no_epistasis_by_default():
    eff = sample_effects(["A", "B"], effect_sd={"HD": 1.0},
                         rng=np.random.default_rng(0))
    # With no epistasis_sd, A/B ddG is exactly the sum of the two singles.
    d = eff.ddg_for("A/B")
    da, db = eff.ddg_for("A"), eff.ddg_for("B")
    for s in STATES:
        assert d[s] == pytest.approx(da[s] + db[s])


def test_epistasis_only_on_requested_pairs():
    eff = sample_effects(["A", "B", "C"], effect_sd={"HD": 1.0},
                         epistasis_sd={"HD": 0.5}, pairs=[("A", "B")],
                         rng=np.random.default_rng(0))
    # A/B has an epistasis term; A/C does not (sum of singles only).
    ab = eff.ddg_for("A/B")["HD"]
    a, b = eff.ddg_for("A")["HD"], eff.ddg_for("B")["HD"]
    assert ab != pytest.approx(a + b)
    ac = eff.ddg_for("A/C")["HD"]
    a, c = eff.ddg_for("A")["HD"], eff.ddg_for("C")["HD"]
    assert ac == pytest.approx(a + c)


def test_mean_shift_biases_draws():
    """A large positive mean pushes draws positive."""
    eff = sample_effects([f"m{i}" for i in range(200)],
                         effect_sd={"HD": (5.0, 0.1)},
                         rng=np.random.default_rng(0))
    vals = [eff.ddg_for(m)["HD"] for m in eff.mutations]
    assert np.mean(vals) == pytest.approx(5.0, abs=0.1)
