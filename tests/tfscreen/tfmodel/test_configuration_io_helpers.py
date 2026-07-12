"""
Unit tests for the internal helpers in tfscreen.tfmodel.configuration_io.

_update_dataclass is tested in test_configure_and_run.py; this file covers
_extract_scalars and _gather_dict_field which have no prior test coverage.
"""

import dataclasses
import numpy as np
import pytest

import pandas as pd
import jax.numpy as jnp

from tfscreen.tfmodel.configuration_io import (
    _extract_scalars,
    _gather_dict_field,
    _extract_prior_arrays,
    _assemble_condition_array,
    _read_priors_flat,
    _update_dataclass,
)
from tfscreen.tfmodel.generative.components.growth.linear import (
    get_priors as _linear_get_priors,
)


# Minimal dataclasses mirroring the growth.condition_growth prior nesting so
# _extract_prior_arrays produces the real dotted keys the loader expects.
@dataclasses.dataclass
class _CondGrowthPriors:
    k_loc: object = dataclasses.field(default_factory=lambda: np.array([0.011, 0.021, 0.029]))
    k_scale: float = 0.1
    m_is_selection: tuple = (True, False, True)  # static bool tuple: must be ignored


@dataclasses.dataclass
class _GrowthPriors:
    condition_growth: _CondGrowthPriors = dataclasses.field(default_factory=_CondGrowthPriors)


@dataclasses.dataclass
class _RootPriors:
    growth: _GrowthPriors = dataclasses.field(default_factory=_GrowthPriors)


def _cond_rep_map():
    return pd.DataFrame({
        "map_condition_rep": [0, 1, 2],
        "condition_rep": ["kanR+kan", "kanR-kan", "pheS+4CP"],
    })


# ---------------------------------------------------------------------------
# Minimal dataclass helpers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Leaf:
    x: float = 1.0
    y: float = 2.5


@dataclasses.dataclass
class Nested:
    sub: Leaf = dataclasses.field(default_factory=Leaf)
    z: float = 3.0


@dataclasses.dataclass
class WithDict:
    d: dict = dataclasses.field(default_factory=lambda: {"a": 1.0, "b": 2.0})
    scalar: float = 5.0


@dataclasses.dataclass
class WithArray:
    arr: object = dataclasses.field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
    scalar: float = 7.0


@dataclasses.dataclass
class WithString:
    label: str = "hello"
    value: float = 9.0


# ---------------------------------------------------------------------------
# _extract_scalars
# ---------------------------------------------------------------------------

class TestExtractScalars:
    def test_flat_scalar_fields_extracted(self):
        out = _extract_scalars(Leaf())
        assert out["x"] == pytest.approx(1.0)
        assert out["y"] == pytest.approx(2.5)

    def test_flat_values_are_float(self):
        out = _extract_scalars(Leaf())
        for v in out.values():
            assert isinstance(v, (int, float))

    def test_nested_dataclass_prefixed(self):
        out = _extract_scalars(Nested())
        assert "sub.x" in out
        assert "sub.y" in out
        assert out["sub.x"] == pytest.approx(1.0)
        assert out["z"] == pytest.approx(3.0)

    def test_nested_flat_key_absent(self):
        # The nested object should NOT appear as a raw key
        out = _extract_scalars(Nested())
        assert "sub" not in out

    def test_dict_field_expanded_with_dot_key(self):
        out = _extract_scalars(WithDict())
        assert "d.a" in out
        assert "d.b" in out
        assert out["d.a"] == pytest.approx(1.0)
        assert out["d.b"] == pytest.approx(2.0)

    def test_dict_scalar_sibling_also_extracted(self):
        out = _extract_scalars(WithDict())
        assert "scalar" in out
        assert out["scalar"] == pytest.approx(5.0)

    def test_array_field_skipped(self):
        """1-D arrays should not appear in output (non-scalar shape)."""
        out = _extract_scalars(WithArray())
        assert "arr" not in out

    def test_scalar_alongside_array_extracted(self):
        out = _extract_scalars(WithArray())
        assert "scalar" in out
        assert out["scalar"] == pytest.approx(7.0)

    def test_string_field_stored_as_str(self):
        out = _extract_scalars(WithString())
        assert "label" in out
        assert isinstance(out["label"], str)
        assert out["label"] == "hello"

    def test_numeric_field_alongside_string(self):
        out = _extract_scalars(WithString())
        assert out["value"] == pytest.approx(9.0)

    def test_prefix_prepended(self):
        out = _extract_scalars(Leaf(), prefix="outer.")
        assert "outer.x" in out
        assert "outer.y" in out

    def test_empty_object_returns_empty_dict(self):
        @dataclasses.dataclass
        class Empty:
            pass
        out = _extract_scalars(Empty())
        assert isinstance(out, dict)

    def test_dict_array_value_skipped(self):
        """Arrays stored as dict values should be skipped."""
        @dataclasses.dataclass
        class WithDictArr:
            d: dict = dataclasses.field(
                default_factory=lambda: {"arr": np.array([1.0, 2.0]), "val": 3.0}
            )
        out = _extract_scalars(WithDictArr())
        assert "d.arr" not in out
        assert "d.val" in out


# ---------------------------------------------------------------------------
# _gather_dict_field
# ---------------------------------------------------------------------------

class TestGatherDictField:
    def test_basic_collection(self):
        flat = {"a.x": 1.0, "a.y": 2.0}
        result = _gather_dict_field("a", flat)
        assert result == {"x": 1.0, "y": 2.0}

    def test_ignores_unrelated_keys(self):
        flat = {"a.x": 1.0, "b.x": 9.0, "c": 99.0}
        result = _gather_dict_field("a", flat)
        assert "x" in result
        assert len(result) == 1

    def test_skips_deeper_nesting(self):
        """Keys with two-level suffixes (a.x.sub) must be ignored."""
        flat = {"a.x": 1.0, "a.y.deep": 5.0}
        result = _gather_dict_field("a", flat)
        assert "x" in result
        assert "y" not in result
        assert "y.deep" not in result

    def test_returns_empty_when_no_match(self):
        flat = {"b.x": 1.0, "c.y": 2.0}
        result = _gather_dict_field("a", flat)
        assert result == {}

    def test_coerces_to_float(self):
        flat = {"a.x": "3.14"}
        result = _gather_dict_field("a", flat)
        assert result["x"] == pytest.approx(3.14)
        assert isinstance(result["x"], float)

    def test_non_numeric_string_preserved(self):
        flat = {"a.label": "hello"}
        result = _gather_dict_field("a", flat)
        assert result["label"] == "hello"

    def test_integer_coerced_to_float(self):
        flat = {"a.n": 42}
        result = _gather_dict_field("a", flat)
        assert result["n"] == pytest.approx(42.0)

    def test_prefix_must_match_exactly(self):
        """'ab.x' must not be collected when full_key is 'a'."""
        flat = {"ab.x": 99.0, "a.x": 1.0}
        result = _gather_dict_field("a", flat)
        assert "x" in result
        assert result["x"] == pytest.approx(1.0)
        assert len(result) == 1

    def test_multiple_suffixes_collected(self):
        flat = {f"p.k{i}": float(i) for i in range(5)}
        result = _gather_dict_field("p", flat)
        assert len(result) == 5
        for i in range(5):
            assert result[f"k{i}"] == pytest.approx(float(i))


# ---------------------------------------------------------------------------
# _extract_prior_arrays
# ---------------------------------------------------------------------------

class TestExtractPriorArrays:

    def test_float_array_extracted_with_dotted_key(self):
        out = _extract_prior_arrays(_RootPriors())
        assert "growth.condition_growth.k_loc" in out
        assert np.allclose(out["growth.condition_growth.k_loc"],
                           [0.011, 0.021, 0.029])

    def test_scalar_field_not_extracted(self):
        out = _extract_prior_arrays(_RootPriors())
        assert "growth.condition_growth.k_scale" not in out

    def test_bool_tuple_not_extracted(self):
        """Static config tuples (m_is_selection) must never look like a prior."""
        out = _extract_prior_arrays(_RootPriors())
        assert "growth.condition_growth.m_is_selection" not in out


# ---------------------------------------------------------------------------
# _assemble_condition_array — name-join + fail-fast
# ---------------------------------------------------------------------------

class TestAssembleConditionArray:

    def _grp(self, rows):
        return pd.DataFrame(rows)

    def test_name_join_orders_by_map_not_row_order(self):
        # Rows deliberately shuffled; flat_index does NOT match map order.
        grp = self._grp([
            {"value": 0.029, "flat_index": 0, "condition_rep": "pheS+4CP"},
            {"value": 0.011, "flat_index": 1, "condition_rep": "kanR+kan"},
            {"value": 0.021, "flat_index": 2, "condition_rep": "kanR-kan"},
        ])
        arr = _assemble_condition_array("k_loc", grp, _cond_rep_map())
        assert np.allclose(arr, [0.011, 0.021, 0.029])

    def test_unknown_condition_raises(self):
        grp = self._grp([
            {"value": 0.011, "flat_index": 0, "condition_rep": "kanR+kan"},
            {"value": 0.021, "flat_index": 1, "condition_rep": "kanR-kan"},
            {"value": 0.029, "flat_index": 2, "condition_rep": "MYSTERY"},
        ])
        with pytest.raises(ValueError, match="not one of the model's conditions"):
            _assemble_condition_array("k_loc", grp, _cond_rep_map())

    def test_missing_condition_raises(self):
        grp = self._grp([
            {"value": 0.011, "flat_index": 0, "condition_rep": "kanR+kan"},
            {"value": 0.021, "flat_index": 1, "condition_rep": "kanR-kan"},
        ])
        with pytest.raises(ValueError, match="missing per-condition"):
            _assemble_condition_array("k_loc", grp, _cond_rep_map())

    def test_fallback_to_flat_index_without_map(self):
        grp = self._grp([
            {"value": 0.029, "flat_index": 2},
            {"value": 0.011, "flat_index": 0},
            {"value": 0.021, "flat_index": 1},
        ])
        arr = _assemble_condition_array("k_loc", grp, None)
        assert np.allclose(arr, [0.011, 0.021, 0.029])


# ---------------------------------------------------------------------------
# _read_priors_flat — scalar / mixed / legacy
# ---------------------------------------------------------------------------

class TestReadPriorsFlat:

    def test_legacy_two_column_scalar_only(self):
        df = pd.DataFrame({
            "parameter": ["growth.condition_growth.k_scale", "growth.x.y"],
            "value": [0.1, 2.0],
        })
        flat = _read_priors_flat(df, _cond_rep_map())
        assert flat["growth.condition_growth.k_scale"] == pytest.approx(0.1)
        assert flat["growth.x.y"] == pytest.approx(2.0)

    def test_mixed_scalar_and_indexed(self):
        df = pd.DataFrame([
            {"parameter": "growth.condition_growth.m_scale_plus",
             "value": 0.01, "flat_index": np.nan, "condition_rep": np.nan},
            {"parameter": "growth.condition_growth.k_loc",
             "value": 0.021, "flat_index": 1, "condition_rep": "kanR-kan"},
            {"parameter": "growth.condition_growth.k_loc",
             "value": 0.011, "flat_index": 0, "condition_rep": "kanR+kan"},
            {"parameter": "growth.condition_growth.k_loc",
             "value": 0.029, "flat_index": 2, "condition_rep": "pheS+4CP"},
        ])
        flat = _read_priors_flat(df, _cond_rep_map())
        assert flat["growth.condition_growth.m_scale_plus"] == pytest.approx(0.01)
        assert np.allclose(np.asarray(flat["growth.condition_growth.k_loc"]),
                           [0.011, 0.021, 0.029])

    def test_indexed_array_is_jax(self):
        df = pd.DataFrame([
            {"parameter": "growth.condition_growth.k_loc",
             "value": v, "flat_index": i, "condition_rep": c}
            for i, (v, c) in enumerate([(0.011, "kanR+kan"),
                                        (0.021, "kanR-kan"),
                                        (0.029, "pheS+4CP")])
        ])
        flat = _read_priors_flat(df, _cond_rep_map())
        assert isinstance(flat["growth.condition_growth.k_loc"], jnp.ndarray)


# ---------------------------------------------------------------------------
# write→read round trip at the helper level (no orchestrator)
# ---------------------------------------------------------------------------

class TestPriorArrayRoundTrip:

    def test_extract_then_read_preserves_array(self):
        # Emulate write_configuration's per-condition prior emission.
        arrays = _extract_prior_arrays(_RootPriors())
        key = "growth.condition_growth.k_loc"
        arr = arrays[key]
        cond_rep_map = _cond_rep_map()
        sorted_map = cond_rep_map.sort_values("map_condition_rep").reset_index(drop=True)
        df = pd.DataFrame({
            "parameter": key,
            "value": np.asarray(arr).flatten(),
            "flat_index": range(len(arr)),
            "condition_rep": sorted_map["condition_rep"].values,
        })
        flat = _read_priors_flat(df, cond_rep_map)
        assert np.allclose(np.asarray(flat[key]), [0.011, 0.021, 0.029])


class TestUpdateDataclassStaticFields:
    """_update_dataclass must not corrupt static tuple fields, and must apply
    scalar (incl. bool-as-float) overrides."""

    def test_tuple_field_not_overwritten_by_csv_string(self):
        # m_is_selection is serialised by _extract_scalars as a string; the
        # freshly-derived tuple must survive the round-trip unchanged.
        priors = _linear_get_priors(is_selection=[True, False, True])
        assert priors.m_is_selection == (True, False, True)

        flat = {
            "m_is_selection": "(True, False, True)",  # the corrupting string
            "m_scale_plus": 0.5,                        # a real scalar override
        }
        updated = _update_dataclass(priors, "", flat)

        assert updated.m_is_selection == (True, False, True)  # tuple preserved
        assert updated.m_scale_plus == 0.5                     # scalar applied

    def test_m_pinned_bool_roundtrips_truthy(self):
        priors = _linear_get_priors()
        assert not priors.m_pinned
        updated = _update_dataclass(priors, "", {"m_pinned": 1.0})
        assert bool(updated.m_pinned)
