"""
Tests for tfscreen.util.parallel.resolve_workers.
"""
from unittest.mock import patch

from tfscreen.util.parallel import resolve_workers
from tfscreen.util import resolve_workers as resolve_workers_reexport


def test_none_is_serial():
    assert resolve_workers(None) == 1


def test_one_is_serial():
    assert resolve_workers(1) == 1


def test_explicit_count_passes_through():
    assert resolve_workers(4) == 4


def test_negative_uses_cpu_count_minus_one():
    with patch("tfscreen.util.parallel.os.cpu_count", return_value=8):
        assert resolve_workers(-1) == 7


def test_negative_other_values_also_cpu_count_minus_one():
    with patch("tfscreen.util.parallel.os.cpu_count", return_value=8):
        assert resolve_workers(-4) == 7


def test_negative_floors_at_one():
    with patch("tfscreen.util.parallel.os.cpu_count", return_value=1):
        assert resolve_workers(-1) == 1


def test_cpu_count_none_falls_back():
    # os.cpu_count() can return None; fallback keeps it >= 1.
    with patch("tfscreen.util.parallel.os.cpu_count", return_value=None):
        assert resolve_workers(-1) == 1


def test_reexported_from_util_package():
    assert resolve_workers_reexport is resolve_workers
