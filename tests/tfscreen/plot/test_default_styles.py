import pytest
from tfscreen.plot.default_styles import (
    DEFAULT_SCATTER_KWARGS,
    DEFAULT_FIT_LINE_KWARGS,
    DEFAULT_EXPT_SCATTER_KWARGS,
    DEFAULT_EXPT_ERROR_KWARGS,
    DEFAULT_HMAP_GRID_KWARGS,
    DEFAULT_HMAP_PATCH_KWARGS,
    DEFAULT_HMAP_AA_AXIS_KWARGS,
    DEFAULT_HMAP_TITRANT_AXIS_KWARGS,
    DEFAULT_HMAP_SITE_AXIS_KWARGS
)

def test_constants_exist():
    assert isinstance(DEFAULT_SCATTER_KWARGS, dict)
    assert isinstance(DEFAULT_FIT_LINE_KWARGS, dict)
    assert isinstance(DEFAULT_EXPT_SCATTER_KWARGS, dict)
    assert isinstance(DEFAULT_EXPT_ERROR_KWARGS, dict)
    assert isinstance(DEFAULT_HMAP_GRID_KWARGS, dict)
    assert isinstance(DEFAULT_HMAP_PATCH_KWARGS, dict)
    assert isinstance(DEFAULT_HMAP_AA_AXIS_KWARGS, dict)
    assert isinstance(DEFAULT_HMAP_TITRANT_AXIS_KWARGS, dict)
    assert isinstance(DEFAULT_HMAP_SITE_AXIS_KWARGS, dict)
