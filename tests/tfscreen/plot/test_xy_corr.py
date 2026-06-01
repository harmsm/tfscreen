
import pytest
import numpy as np
from matplotlib import pyplot as plt
from tfscreen.plot.xy_corr import xy_corr
from tfscreen.plot.default_styles import DEFAULT_HEXBIN_KWARGS

def test_xy_corr_basic():
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.1, 100)
    
    # Close figures
    plt.close('all')
    
    ax = xy_corr(x, y, percentile=0.0, pad_by=0.1)
    
    assert ax is not None
    assert len(ax.collections) >= 1 # scatter plot should create a collection
    
    # Check limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Since we pad by 0.1, limits should be slightly larger than data range
    assert xlim[0] < np.min(x)
    assert xlim[1] > np.max(x)

def test_xy_corr_with_ax():
    fig, ax = plt.subplots()
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    
    ret_ax = xy_corr(x, y, ax=ax)
    assert ret_ax is ax
    assert len(ax.collections) > 0 # scatter plot

def test_xy_corr_subsample():
    x = np.random.normal(0, 1, 100)
    y = x
    
    ax = xy_corr(x, y, subsample=10)
    # Scatter plot data source should be subsampled? 
    # Hard to check exact points plotted without deeper inspection, but 
    # we can check it runs.
    # Actually, we can check the offsets of the collection
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 10

def test_xy_corr_custom_kwargs():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])

    ax = xy_corr(x, y, scatter_kwargs={'color': 'red', 's': 50})
    # Check if kwargs were applied (mocking plt/ax might be better for this,
    # but introspection works too)

    # For scatter, facecolor might be complex (array), but edgecolors should be default or overridden.
    # DEFAULT_SCATTER_KWARGS has edgecolor royalblue.
    # helper copies it. But we didn't override edgecolor, we made color=red which might set facecolor?
    # Actually matplotlib scatter 'color' sets both unless specified.
    pass


# ---------------------------------------------------------------------------
# hexbin mode
# ---------------------------------------------------------------------------

def test_hexbin_returns_axes():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 0.2, 200)
    plt.close("all")
    ax = xy_corr(x, y, as_hexbin=True)
    assert ax is not None
    plt.close("all")


def test_hexbin_adds_colorbar():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 0.2, 200)
    plt.close("all")
    ax = xy_corr(x, y, as_hexbin=True)
    # colorbar is a new axes attached to the figure
    assert len(ax.get_figure().axes) == 2
    plt.close("all")


def test_hexbin_no_scatter_collection():
    """In hexbin mode, ax.scatter should not be called — no PathCollection."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 0.2, 200)
    plt.close("all")
    ax = xy_corr(x, y, as_hexbin=True)
    from matplotlib.collections import PathCollection
    path_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(path_collections) == 0
    plt.close("all")


def test_hexbin_default_kwargs_applied():
    """Default hexbin kwargs (log bins, plasma cmap) must be used."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 0.2, 200)
    plt.close("all")
    ax = xy_corr(x, y, as_hexbin=True)
    from matplotlib.collections import PolyCollection
    hb = next(c for c in ax.collections if isinstance(c, PolyCollection))
    assert hb.cmap.name == DEFAULT_HEXBIN_KWARGS["cmap"]
    plt.close("all")


def test_hexbin_kwargs_override_defaults():
    """User hexbin_kwargs must override DEFAULT_HEXBIN_KWARGS."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 0.2, 200)
    plt.close("all")
    ax = xy_corr(x, y, as_hexbin=True, hexbin_kwargs={"cmap": "viridis"})
    from matplotlib.collections import PolyCollection
    hb = next(c for c in ax.collections if isinstance(c, PolyCollection))
    assert hb.cmap.name == "viridis"
    plt.close("all")


def test_hexbin_colorbar_not_added_in_scatter_mode():
    """Scatter mode must not add a colorbar axes."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    plt.close("all")
    ax = xy_corr(x, y, as_hexbin=False)
    assert len(ax.get_figure().axes) == 1
    plt.close("all")


def test_hexbin_with_custom_ax():
    """Colorbar is added to the figure that owns the user-supplied ax."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0, 0.2, 200)
    plt.close("all")
    fig, ax = plt.subplots()
    ret = xy_corr(x, y, as_hexbin=True, ax=ax)
    assert ret is ax
    assert len(fig.axes) == 2
    plt.close("all")

