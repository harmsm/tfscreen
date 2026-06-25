"""
Unit tests for the internal helpers in tfscreen.tfmodel.configuration_io.

_update_dataclass is tested in test_configure_and_run.py; this file covers
_extract_scalars and _gather_dict_field which have no prior test coverage.
"""

import dataclasses
import numpy as np
import pytest

from tfscreen.tfmodel.configuration_io import (
    _extract_scalars,
    _gather_dict_field,
)


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
