import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.collections import PatchCollection

from unittest.mock import MagicMock, ANY

# Import the function to be tested
from tfscreen.plot.heatmap.heatmap_core import _build_heatmap_collection
from tfscreen.plot.heatmap.heatmap_core import _build_color_mapper
from tfscreen.plot.heatmap.heatmap_core import _get_fig_dim
from tfscreen.plot.heatmap.heatmap_core import _get_ticks
from tfscreen.plot.heatmap.heatmap_core import _format_axis
from tfscreen.plot.heatmap.heatmap_core import heatmap


# Import defaults for comparison
from tfscreen.plot.default_styles import (
    DEFAULT_HMAP_PATCH_KWARGS,
    DEFAULT_HMAP_MISSING_VALUE_COLOR,
    DEFAULT_HMAP_FIG_HEIGHT
)

from tfscreen.plot.helper import get_ax_limits 


# --- Reusable Fixtures -------------------------------------------------------

@pytest.fixture
def heatmap_value_matrix() -> np.ndarray:
    """Provides a simple 2x3 NumPy array for heatmap data, including a NaN."""
    return np.array([[1.0, 2.0, 3.0],
                     [4.0, np.nan, 6.0]])

@pytest.fixture
def color_mapper(heatmap_value_matrix: np.ndarray) -> cm.ScalarMappable:
    """Provides a ScalarMappable object based on the heatmap_value_matrix."""
    norm = mcolors.Normalize(vmin=np.nanmin(heatmap_value_matrix),
                             vmax=np.nanmax(heatmap_value_matrix))
    cmap = plt.get_cmap("viridis")
    return cm.ScalarMappable(norm=norm, cmap=cmap)

@pytest.fixture
def diverging_values() -> np.ndarray:
    """Provides a NumPy array with values that span zero."""
    return np.array([-10., -5., 0., 5., 20.])

@pytest.fixture
def custom_color_func_rgb() -> callable:
    """A sample callable that returns an RGB tuple."""
    # A simple black-to-red gradient
    return lambda x: (x, 0., 0.)

@pytest.fixture
def custom_color_func_rgba() -> callable:
    """A sample callable that returns an RGBA tuple."""
    # A simple black-to-green gradient with changing alpha
    return lambda x: (0., x, 0., 0.5 + x/2)

@pytest.fixture
def prebuilt_mappable() -> cm.ScalarMappable:
    """A fixture that returns a pre-configured ScalarMappable."""
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("cool")
    return cm.ScalarMappable(norm=norm, cmap=cmap)

# ----------------------------------------------
# tests for _build_heatmap_collection
# ----------------------------------------------

class TestBuildHeatmapCollection:

    def test_happy_path_defaults(self, heatmap_value_matrix, color_mapper):
        """
        GIVEN a valid value matrix and color mapper
        WHEN _build_heatmap_collection is called with default arguments
        THEN it should return a correctly configured PatchCollection.
        """
        # ACT
        result = _build_heatmap_collection(heatmap_value_matrix, color_mapper)

        # ASSERT
        assert isinstance(result, PatchCollection)
        # The number of patches should equal the number of cells in the matrix
        assert len(result.get_paths()) == heatmap_value_matrix.size

        # Check that the NaN value was correctly colored
        colors = result.get_facecolor()
        nan_patch_color = colors[4] # NaN is at row 1, col 1 -> index (1*3 + 1) = 4
        expected_missing_color = mcolors.to_rgba(DEFAULT_HMAP_MISSING_VALUE_COLOR)
        assert np.array_equal(nan_patch_color, expected_missing_color)

        # Check that a data value was correctly colored
        first_patch_color = colors[0]
        expected_first_color = color_mapper.to_rgba(heatmap_value_matrix[0, 0])
        assert np.array_equal(first_patch_color, expected_first_color)

    @pytest.mark.parametrize("kwargs, match_str", [
        ({"value_matrix": [1, 2, 3]}, "value_matrix must be a 2D numpy array"),
        ({"color_mapper": "not_a_mapper"}, "`to_rgba` method"),
        ({"btwn_square_space": 1.1}, "btwn_square_space.*not valid"),
        ({"x_values": np.arange(2)}, "x_values should be.*long"),
        ({"y_values": np.arange(2)}, "y_values should be.*long"),
    ])
    def test_validation_errors(self, heatmap_value_matrix, color_mapper, kwargs, match_str):
        """
        GIVEN invalid inputs for various parameters
        WHEN _build_heatmap_collection is called
        THEN it should raise a ValueError with an informative message.
        """
        # ARRANGE
        all_args = {
            "value_matrix": heatmap_value_matrix,
            "color_mapper": color_mapper
        }
        # Overwrite default args with the specific invalid one for this test case
        all_args.update(kwargs)
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match=match_str):
            _build_heatmap_collection(**all_args)

    def test_custom_coordinates_and_spacing(self, heatmap_value_matrix, color_mapper):
        """
        GIVEN custom x/y values and spacing
        WHEN _build_heatmap_collection is called
        THEN the generated polygons should have the correct coordinates.
        """
        # ARRANGE
        x_vals = np.array([10, 20, 30])      # For 2 rows of data
        y_vals = np.array([100, 110, 120, 130]) # For 3 columns of data
        spacing = 0.2
        buffer = spacing / 2
        
        # ACT
        result = _build_heatmap_collection(
            heatmap_value_matrix, color_mapper, 
            x_values=x_vals, y_values=y_vals, 
            btwn_square_space=spacing
        )
        
        # ASSERT
        # Check the vertices of the first polygon (top-left square)
        first_patch = result.get_paths()[0]
        vertices = first_patch.vertices
        
        expected_left = x_vals[0] + buffer     # 10.1
        expected_right = x_vals[1] - buffer    # 19.9
        expected_bottom = y_vals[0] + buffer   # 100.1
        expected_top = y_vals[1] - buffer      # 109.9
        
        expected_vertices = np.array([
            [expected_left, expected_bottom],
            [expected_left, expected_top],
            [expected_right, expected_top],
            [expected_right, expected_bottom],
            [expected_left, expected_bottom],
        ])
        
        assert np.allclose(vertices, expected_vertices)

    def test_patch_kwargs_and_rasterized(self, heatmap_value_matrix, color_mapper):
        """
        GIVEN custom patch_kwargs and heatmap_as_img=True
        WHEN _build_heatmap_collection is called
        THEN the resulting collection should have the correct properties.
        """
        # ARRANGE
        custom_kwargs = {"edgecolor": "red", "linewidth": 5}
        
        # ACT
        result = _build_heatmap_collection(
            heatmap_value_matrix, color_mapper, 
            patch_kwargs=custom_kwargs,
            heatmap_as_img=True
        )
        
        # ASSERT
        assert result.get_rasterized() is True

        # FIX: Check that all returned edgecolors match the expected color.
        red_rgba = mcolors.to_rgba("red")
        returned_edgecolors = result.get_edgecolor()
        assert all(np.allclose(color, red_rgba) for color in returned_edgecolors)
        
        assert result.get_linewidth()[0] == 5

    def test_custom_missing_value_color(self, heatmap_value_matrix, color_mapper):
        """
        GIVEN a custom color for missing values
        WHEN _build_heatmap_collection is called
        THEN the NaN patch should be set to that color.
        """
        # ARRANGE
        custom_color = "magenta"
        
        # ACT
        result = _build_heatmap_collection(
            heatmap_value_matrix, color_mapper, missing_value_color=custom_color
        )
        
        # ASSERT
        colors = result.get_facecolor()
        nan_patch_color = colors[4] # NaN is at index 4
        expected_color = mcolors.to_rgba(custom_color)
        
        assert np.array_equal(nan_patch_color, expected_color)


# ----------------------------------------------
# tests for _build_color_mapper
# ----------------------------------------------

class TestBuildColorMapper:

    def test_from_string_and_values(self, heatmap_value_matrix):
        """
        GIVEN a colormap string and a data array
        WHEN _build_color_mapper is called
        THEN it should create a mappable with an auto-detected linear scale.
        """
        # ARRANGE
        values = heatmap_value_matrix
        
        # ACT
        result = _build_color_mapper(color_fcn="viridis", values=values)
        
        # ASSERT
        assert isinstance(result, cm.ScalarMappable)
        assert result.cmap.name == "viridis"
        assert isinstance(result.norm, mcolors.Normalize)
        assert np.isclose(result.norm.vmin, 1.0,atol=0.1)
        assert np.isclose(result.norm.vmax, 6.0,atol=0.1)

    def test_with_explicit_vlim(self, heatmap_value_matrix):
        """
        GIVEN an explicit vlim tuple
        WHEN _build_color_mapper is called
        THEN it should use the specified vlim, ignoring the data in `values`.
        """
        result = _build_color_mapper("viridis", values=heatmap_value_matrix, vlim=(-10, 10))
        assert result.norm.vmin == -10
        assert result.norm.vmax == 10

    def test_with_log_scale(self, heatmap_value_matrix):
        """
        GIVEN scale="log"
        WHEN _build_color_mapper is called
        THEN it should return a mappable with a LogNorm.
        """
        result = _build_color_mapper("viridis", values=heatmap_value_matrix, scale="log")
        assert isinstance(result.norm, mcolors.LogNorm)
        
    def test_auto_centers_diverging_data(self, diverging_values, mocker):
        """
        GIVEN data that spans zero
        WHEN _build_color_mapper is called without a vlim
        THEN it should create a symmetric color scale centered on zero.
        """
        # ARRANGE: Mock the helper to isolate the logic being tested
        mock_get_ax_limits = mocker.patch(
            "tfscreen.plot.heatmap.heatmap_core.get_ax_limits",
            side_effect=lambda vals, center_on_zero: (-20, 20) if center_on_zero else (vals.min(), vals.max())
        )
        

        # ACT
        result = _build_color_mapper("coolwarm", values=diverging_values)
        
        # ASSERT
        assert result.norm.vmin == -20
        assert result.norm.vmax == 20
        # Verify it was the center_on_zero=True path that was taken
        mock_get_ax_limits.assert_called_with(diverging_values, center_on_zero=True)

    @pytest.mark.parametrize("color_func_fixture", ["custom_color_func_rgb", "custom_color_func_rgba"])
    def test_from_callable(self, color_func_fixture, request):
        """
        GIVEN a callable that returns RGB or RGBA tuples
        WHEN _build_color_mapper is called
        THEN it should create a custom ListedColormap.
        """
        # ARRANGE
        color_func = request.getfixturevalue(color_func_fixture)
        
        # ACT
        result = _build_color_mapper(color_fcn=color_func, vlim=(0, 1))
        
        # ASSERT
        assert isinstance(result.cmap, mcolors.ListedColormap)
        # Check that the middle of the colormap is the expected color
        mid_color = result.to_rgba(0.5) # Corresponds to input x=0.5
        expected_mid_color = color_func(0.5)
        # Pad with 1.0 if the function was RGB
        if len(expected_mid_color) == 3:
            expected_mid_color = (*expected_mid_color, 1.0)
            
        assert np.allclose(mid_color, expected_mid_color,atol=0.01)

    def test_passthrough_mappable(self, prebuilt_mappable):
        """
        GIVEN a pre-built ScalarMappable object
        WHEN _build_color_mapper is called
        THEN it should return the exact same object.
        """
        result = _build_color_mapper(prebuilt_mappable)
        assert result is prebuilt_mappable # Check for object identity

    @pytest.mark.parametrize("kwargs, match_str", [
        ({"values": None, "vlim": None}, "either values or vlim must be defined"),
        ({"color_fcn": "not_a_real_colormap", "vlim": (0,1)}, "color_fcn value of"),
        ({"color_fcn": lambda x: (x, x, -0.5), "vlim": (0,1)}, "color_fcn should be a function"),
        ({"scale": "invalid", "vlim": (0,1)}, "scale must be 'linear' or 'log'"),
    ])
    def test_validation_errors(self, kwargs, match_str):
        """
        GIVEN various invalid inputs
        WHEN _build_color_mapper is called
        THEN it should raise a ValueError.
        """
        # ARRANGE: Add a valid color_fcn if not part of the test case
        if "color_fcn" not in kwargs:
            kwargs["color_fcn"] = "viridis"
            
        # ACT & ASSERT
        with pytest.raises(ValueError, match=match_str):
            _build_color_mapper(**kwargs)


# ----------------------------------------------
# tests for _get_fig_dim
# ----------------------------------------------

class TestGetFigDim:

    def test_neither_dimension_provided(self):
        """
        GIVEN no height or width
        WHEN _get_fig_dim is called
        THEN it should use the default height and calculate width.
        """
        # ARRANGE
        num_x, num_y = 20, 10
        expected_height = DEFAULT_HMAP_FIG_HEIGHT # 6
        expected_width = (num_x / num_y) * expected_height 
        
        # ACT
        width, height = _get_fig_dim(num_x, num_y)
        
        # ASSERT
        assert height == expected_height
        assert np.isclose(width, expected_width)

    def test_height_provided(self):
        """
        GIVEN only a height
        WHEN _get_fig_dim is called
        THEN it should use that height and calculate width.
        """
        num_x, num_y = 30, 10
        width, height = _get_fig_dim(num_x, num_y, height=5.0)
        assert height == 5.0
        assert np.isclose(width, (30/10) * 5.0) # 15.0

    def test_width_provided(self):
        """
        GIVEN only a width
        WHEN _get_fig_dim is called
        THEN it should use that width and calculate height.
        """
        num_x, num_y = 30, 10
        width, height = _get_fig_dim(num_x, num_y, width=9.0)
        assert width == 9.0
        assert np.isclose(height, (10/30) * 9.0) # 3.0

    def test_both_dimensions_provided(self):
        """
        GIVEN both height and width
        WHEN _get_fig_dim is called
        THEN it should return them unchanged.
        """
        width, height = _get_fig_dim(num_x=10, num_y=10, height=5.0, width=7.0)
        assert height == 5.0
        assert width == 7.0
        
    @pytest.mark.parametrize("x, y", [(0, 10), (10, 0), (-5, 10)])
    def test_raises_error_on_non_positive_dimension(self, x, y):
        """
        GIVEN a zero or negative num_x or num_y
        WHEN _get_fig_dim is called
        THEN it should raise a ValueError.
        """
        with pytest.raises(ValueError, match="must be positive"):
            _get_fig_dim(x, y)


# ----------------------------------------------
# tests for _get_ticks
# ----------------------------------------------

@pytest.fixture
def tick_data() -> tuple:
    """Provides a fixture with 20 labels and 21 boundary values."""
    labels = [f"Label_{i}" for i in range(20)]
    values = np.arange(21)
    return np.array(labels), values

class TestGetTicks:

    def test_no_downsampling_needed(self, tick_data):
        """
        GIVEN max_num_ticks is greater than the number of labels
        WHEN _get_ticks is called
        THEN it should return all labels and their calculated center positions.
        """
        # ARRANGE
        labels, values = tick_data
        
        # ACT
        result_labels, result_values = _get_ticks(labels, values, max_num_ticks=30)
        
        # ASSERT
        assert len(result_labels) == 20
        assert len(result_values) == 20
        # The center of the first tick (between 0 and 1) should be 0.5
        assert np.isclose(result_values[0], 0.5)
        # The last label should be the last original label
        assert result_labels[-1] == "Label_19"

    def test_no_max_num_ticks_provided(self, tick_data):
        """
        GIVEN max_num_ticks is None
        WHEN _get_ticks is called
        THEN it should return all labels and their positions.
        """
        labels, values = tick_data
        result_labels, result_values = _get_ticks(labels, values, max_num_ticks=None)
        assert len(result_labels) == 20
        assert np.array_equal(result_labels, labels)

    def test_downsampling_logic(self, tick_data):
        """
        GIVEN max_num_ticks that requires an even downsampling
        WHEN _get_ticks is called
        THEN it should return the correctly stepped subset of ticks.
        """
        # ARRANGE
        labels, values = tick_data # 20 labels
        
        # ACT
        # step should be ceil(20 / 5) = 4
        result_labels, result_values = _get_ticks(labels, values, max_num_ticks=5)
        
        # ASSERT
        assert len(result_labels) == 5
        expected_labels = ["Label_0", "Label_4", "Label_8", "Label_12", "Label_16"]
        assert np.array_equal(result_labels, expected_labels)
        # The value for "Label_4" is the center of boundaries 4 and 5 -> 4.5
        assert np.isclose(result_values[1], 4.5)

    def test_downsampling_with_uneven_division(self, tick_data):
        """
        GIVEN max_num_ticks that requires an uneven downsampling
        WHEN _get_ticks is called
        THEN it should calculate the ceiling of the step size correctly.
        """
        # ARRANGE
        labels, values = tick_data # 20 labels
        
        # ACT
        # step should be ceil(20 / 7) = 3
        result_labels, result_values = _get_ticks(labels, values, max_num_ticks=7)
        
        # ASSERT
        # The number of ticks will be ceil(20 / 3) = 7
        assert len(result_labels) == 7
        expected_labels = ["Label_0", "Label_3", "Label_6", "Label_9", 
                           "Label_12", "Label_15", "Label_18"]
        assert np.array_equal(result_labels, expected_labels)

    @pytest.mark.parametrize("max_ticks", [0, -1])
    def test_zero_or_negative_max_ticks(self, tick_data, max_ticks):
        """
        GIVEN max_num_ticks is zero or negative
        WHEN _get_ticks is called
        THEN it should return no ticks.
        """
        labels, values = tick_data
        result_labels, result_values = _get_ticks(labels, values, max_num_ticks=max_ticks)
        assert len(result_labels) == 0
        assert len(result_values) == 0
        
    def test_empty_input(self):
        """
        GIVEN an empty list of labels
        WHEN _get_ticks is called
        THEN it should return empty lists/arrays.
        """
        result_labels, result_values = _get_ticks([], np.array([0]))
        assert len(result_labels) == 0
        assert len(result_values) == 0



# --- Fixture for a clean Axes object ---

@pytest.fixture
def ax() -> plt.Axes:
    """Provides a clean matplotlib Axes object for each test."""
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)

# ----------------------------------------------
# tests for _format_axis
# ----------------------------------------------

class TestFormatAxis:

    def test_formats_x_axis(self, ax, tick_data):
        """
        GIVEN an Axes object and x-axis parameters
        WHEN _format_axis is called with axis_key='x'
        THEN the x-axis should be correctly formatted.
        """
        # ARRANGE
        labels, values = tick_data
        
        # ACT
        _format_axis(ax, labels, values, axis_key='x', label_font_size=15)
        
        # ASSERT
        # Check that ticks were placed at the center of the grid lines
        expected_centers = (values[1:] + values[:-1]) / 2
        assert np.allclose(ax.get_xticks(), expected_centers)
        
        # Check that labels were set correctly
        result_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert np.array_equal(result_labels, labels)
        
        # Check a font property
        assert ax.get_xticklabels()[0].get_fontsize() == 15

    def test_formats_y_axis(self, ax, tick_data):
        """
        GIVEN an Axes object and y-axis parameters
        WHEN _format_axis is called with axis_key='y'
        THEN the y-axis should be correctly formatted.
        """
        labels, values = tick_data
        _format_axis(ax, labels, values, axis_key='y', label_horizontal_alignment='right')

        assert np.allclose(ax.get_yticks(), (values[1:] + values[:-1]) / 2)
        assert [t.get_text() for t in ax.get_yticklabels()] == list(labels)
        assert ax.get_yticklabels()[0].get_horizontalalignment() == 'right'

    def test_downsamples_ticks(self, ax, tick_data):
        """
        GIVEN a max_num_ticks value
        WHEN _format_axis is called
        THEN it should display a downsampled number of ticks.
        """
        labels, values = tick_data # 20 labels
        _format_axis(ax, labels, values, axis_key='x', max_num_ticks=5)
        
        # step = ceil(20/5) = 4. Number of ticks will be 5.
        assert len(ax.get_xticks()) == 5
        assert ax.get_xticklabels()[1].get_text() == "Label_4"

    def test_sets_tick_length(self, ax, tick_data, mocker):
        """
        GIVEN a tick_length value
        WHEN _format_axis is called
        THEN it should call set_tick_params with the correct length.
        """
        # ARRANGE
        labels, values = tick_data
        # Mock the method on the axis object that will be retrieved by getattr
        mock_set_params = mocker.patch.object(ax.xaxis, 'set_tick_params')
        
        # ACT
        _format_axis(ax, labels, values, axis_key='x', tick_length=10.0)
        
        # ASSERT
        mock_set_params.assert_called_once_with(length=10.0)
        
    def test_handles_no_ticks(self, ax, tick_data):
        """
        GIVEN max_num_ticks=0
        WHEN _format_axis is called
        THEN no ticks or labels should be set.
        """
        labels, values = tick_data
        _format_axis(ax, labels, values, axis_key='x', max_num_ticks=0)
        
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_xticklabels()) == 0

    def test_raises_error_on_invalid_axis_key(self, ax, tick_data):
        """
        GIVEN an invalid axis_key (not 'x' or 'y')
        WHEN _format_axis is called
        THEN it should raise a ValueError.
        """
        labels, values = tick_data
        with pytest.raises(ValueError, match="Invalid axis_key: 'z'"):
            _format_axis(ax, labels, values, axis_key='z')


# --- Fixture for a wide-format DataFrame ---

@pytest.fixture
def plot_df(heatmap_value_matrix) -> pd.DataFrame:
    """Provides a 2x3 wide-format DataFrame for plotting."""
    return pd.DataFrame(
        heatmap_value_matrix,
        index=[f"Site_{i}" for i in range(2)],
        columns=[f"AA_{i}" for i in range(3)]
    )

# ----------------------------------------------
# tests for heatmap
# ----------------------------------------------

class TestHeatmap:

    @pytest.fixture(autouse=True)
    def mock_helpers(self, mocker) -> dict:
        """Auto-mock all helper functions called by heatmap."""
        mocks = {
            "build_color_mapper": mocker.patch("tfscreen.plot.heatmap.heatmap_core._build_color_mapper"),
            "get_fig_dim": mocker.patch("tfscreen.plot.heatmap.heatmap_core._get_fig_dim", return_value=(10, 8)),
            "build_heatmap_collection": mocker.patch("tfscreen.plot.heatmap.heatmap_core._build_heatmap_collection"),
            "format_axis": mocker.patch("tfscreen.plot.heatmap.heatmap_core._format_axis"),
            "colorbar": mocker.patch("matplotlib.figure.Figure.colorbar")
        }
        # The collection needs to be a mock that can be added to an ax
        mock_p_collection = MagicMock(spec=PatchCollection)
        mock_p_collection.get_transform().contains_branch_seperately.return_value = (True, True)
        mock_p_collection.get_offset_transform().contains_branch_seperately.return_value = (True, True)
        mocks["build_heatmap_collection"].return_value = mock_p_collection
        return mocks

    def test_happy_path_creates_new_ax(self, plot_df, mock_helpers, mocker):
        """
        GIVEN a valid plot_df and default arguments
        WHEN heatmap is called without an `ax`
        THEN it should create a new figure and call all helpers correctly.
        """
        # ARRANGE: Mock the creation of the figure and axes
        mock_ax = MagicMock(spec=Axes)
        mock_fig = MagicMock(spec=Figure)

        mock_spines = {
            'top': MagicMock(), 'bottom': MagicMock(),
            'left': MagicMock(), 'right': MagicMock()
        }
        mock_ax.spines = mock_spines

        mocker.patch("tfscreen.plot.heatmap.heatmap_core.plt.subplots",
                    return_value=(mock_fig, mock_ax))

        # ACT
        fig, ax = heatmap(plot_df)

        # ASSERT
        assert fig is mock_fig
        assert ax is mock_ax

        # Verify helpers were called
        mock_helpers["get_fig_dim"].assert_called_once()
        mock_helpers["build_heatmap_collection"].assert_called_once()
        
        mock_ax.add_collection.assert_called_once_with(
            mock_helpers["build_heatmap_collection"].return_value
        )
        
        assert mock_helpers["format_axis"].call_count == 2
        mock_ax.set_aspect.assert_called_with("equal")
        
        # FIX: Assert that the colorbar method was called on the mock_fig instance itself.
        mock_fig.colorbar.assert_called_once()
        
        mock_spines['right'].set_visible.assert_called_once_with(False)

    def test_draws_on_existing_ax(self, plot_df, mock_helpers):
        """
        GIVEN a pre-existing Axes object
        WHEN heatmap is called with that `ax`
        THEN it should draw on it and not apply final formatting.
        """
        # ARRANGE
        fig, existing_ax = plt.subplots()
        # Mock the `get_figure` method on the provided ax
        existing_ax.get_figure = MagicMock(return_value=fig)
        
        # ACT
        ret_fig, ret_ax = heatmap(plot_df, ax=existing_ax)
        
        # ASSERT
        assert ret_fig is fig
        assert ret_ax is existing_ax
        
        # Helpers that create/size the figure should NOT be called
        mock_helpers["get_fig_dim"].assert_not_called()
        
        # The function should NOT apply its own final formatting
        assert ret_ax.get_aspect() != "equal" # Should remain 'auto'
        plt.close(fig)

    def test_grid_drawing_respects_offsets(self, plot_df, mocker):
        """
        GIVEN grid=True and non-zero offsets
        WHEN heatmap is called
        THEN it should draw the grid at the correct offset coordinates.
        """
        # ARRANGE
        fig, ax = plt.subplots()
        mocker.patch.object(ax, 'plot')
        
        # ACT
        heatmap(plot_df, ax=ax, grid=True, x_offset=10, y_offset=20)
        
        # ASSERT
        assert ax.plot.call_count == (plot_df.shape[0] + 1) + (plot_df.shape[1] + 1)

        # FIX: Manually check the call arguments list for a matching call.
        # This avoids the invalid `**ANY` syntax.
        
        # Expected x-range is [10, 10 + num_rows = 12].
        # Expected y-range is [20, 20 + num_cols = 23].
        expected_x_coords = [10, 12]
        
        # Check if any call matches a horizontal line: plot([10, 12], [y, y], ...)
        found_match = any(
            np.array_equal(call.args[0], expected_x_coords)
            for call in ax.plot.call_args_list
        )
        
        assert found_match, "ax.plot was not called with the expected x-coordinates"
        plt.close(fig)

    def test_colorbar_can_be_disabled(self, plot_df, mock_helpers):
        """
        GIVEN plot_scale=False
        WHEN heatmap is called
        THEN no colorbar should be drawn.
        """
        heatmap(plot_df, plot_scale=False)
        mock_helpers["colorbar"].assert_not_called()

    def test_custom_kwargs_are_passed_to_helpers(self, plot_df, mock_helpers):
        """
        GIVEN various custom kwargs
        WHEN heatmap is called
        THEN they should be passed correctly to the relevant helper functions.
        """
        # ARRANGE
        x_kwargs = {"max_num_ticks": 5}
        patch_kwargs = {"edgecolor": "purple"}
        
        # ACT
        heatmap(plot_df, x_axis_kwargs=x_kwargs, heatmap_patch_kwargs=patch_kwargs)
        
        # ASSERT
        # Check that x_axis_kwargs were passed to _format_axis
        # The second call to _format_axis is for the y-axis
        x_axis_call = mock_helpers["format_axis"].call_args_list[0]
        assert x_axis_call.kwargs['max_num_ticks'] == 5
        
        # Check that patch_kwargs were passed to _build_heatmap_collection
        mock_helpers["build_heatmap_collection"].assert_called_with(
            patch_kwargs=patch_kwargs,
            value_matrix=ANY, color_mapper=ANY, x_values=ANY, y_values=ANY,
            missing_value_color=ANY, btwn_square_space=ANY, heatmap_as_img=ANY
        )

    @pytest.mark.parametrize("bad_input, arg_name, match_str", [
        ( "not_a_df", "plot_df", "plot_df should be a pandas DataFrame"),
        ( pd.DataFrame([['a', 'b'], ['c', 'd']]), "plot_df", "must all be numbers"),
        ( "not_an_ax", "ax", "ax must be a matplotlib Axes instance"),
        ( "not_a_type", "x_axis_type", "x_axis_type.*not recognized"),
    ])
    def test_validation_errors(self, plot_df, bad_input, arg_name, match_str):
        """
        GIVEN various invalid inputs
        WHEN heatmap is called
        THEN it should raise a ValueError.
        """
        # ARRANGE
        kwargs = {"plot_df": plot_df}
        kwargs[arg_name] = bad_input
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match=match_str):
            heatmap(**kwargs)