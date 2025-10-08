from tfscreen.plot.helper import get_ax_limits

import pandas as pd
import numpy as np

import matplotlib 
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable

from typing import Optional, Union, Literal, Callable, Tuple, Sequence

from tfscreen.plot.default_styles import (
    DEFAULT_HMAP_FIG_HEIGHT,
    DEFAULT_HMAP_GRID_KWARGS,
    DEFAULT_HMAP_PATCH_KWARGS,
    DEFAULT_HMAP_MISSING_VALUE_COLOR,
    DEFAULT_HMAP_SITE_AXIS_KWARGS,
    DEFAULT_HMAP_AA_AXIS_KWARGS,
    DEFAULT_HMAP_TITRANT_AXIS_KWARGS
)

def _build_heatmap_collection(
    value_matrix: np.ndarray,
    color_mapper: ScalarMappable,
    x_values: Optional[np.ndarray] = None,
    y_values: Optional[np.ndarray] = None,
    patch_kwargs: Optional[dict] = None,
    missing_value_color: Optional[Union[str, tuple]] = None,
    btwn_square_space: float = 0,
    heatmap_as_img: bool = False,
) -> PatchCollection:
    """
    Create a Matplotlib PatchCollection for a heatmap.

    This function takes a 2D matrix of values, a color mapper, and grid
    coordinates to generate a collection of colored polygons. It is the
    core visual-element generator for the heatmap function.

    Parameters
    ----------
    value_matrix : numpy.ndarray
        A 2D array of the values to be plotted.
    color_mapper : matplotlib.cm.ScalarMappable
        A mappable object (like one from `_build_color_mapper`) that has a
        `.to_rgba()` method to convert values to colors.
    x_values : numpy.ndarray, optional
        An array of boundaries for the heatmap squares along the x-axis.
        Its length must be `value_matrix.shape[0] + 1`. If not provided,
        integer coordinates are generated automatically.
    y_values : numpy.ndarray, optional
        An array of boundaries for the heatmap squares along the y-axis.
        Its length must be `value_matrix.shape[1] + 1`. If not provided,
        integer coordinates are generated automatically.
    patch_kwargs : dict, optional
        A dictionary of keyword arguments to be passed to each
        `matplotlib.patches.Polygon` (e.g., `edgecolor`, `linewidth`).
        Note that `facecolor` will be overwritten.
    missing_value_color : color-like, optional
        The color to use for cells where the value in `value_matrix` is NaN.
        If not specified, a default light gray is used.
    btwn_square_space : float, default 0
        The fraction of space between squares, ranging from 0 (touching) to
        1 (no visible squares).
    heatmap_as_img : bool, default False
        If True, the entire patch collection is rasterized upon rendering.
        This is useful for creating clean vector graphics (PDF, SVG) when
        the heatmap has a very large number of squares.

    Returns
    -------
    matplotlib.collections.PatchCollection
        A collection of colored `Polygon` objects that can be added to a
        Matplotlib Axes object.

    Raises
    ------
    ValueError
        If inputs are not of the correct type or shape, or if
        `btwn_square_space` is not between 0 and 1.
    """

    # --- Do some sanity checking --- 

    if not isinstance(value_matrix,np.ndarray) or len(value_matrix.shape) != 2:
        raise ValueError("value_matrix must be a 2D numpy array")

    if not hasattr(color_mapper,"to_rgba"):
        raise ValueError(
            "color_mapper should be an object with an `to_rgba` method that "
            "returns an RGBA value for the values in value_matrix. Usually "
            "this is a matplotlib.cm.ScalarMappable object."
        )

    if btwn_square_space < 0 or btwn_square_space > 1:
        raise ValueError(
            f"btwn_square_space '{btwn_square_space}' is not valid. It must ",
            "be between 0 and 1."
        )

    if x_values is None:
        x_values = np.arange(value_matrix.shape[0] + 1,dtype=int)

    if y_values is None:
        y_values = np.arange(value_matrix.shape[1] + 1,dtype=int)

    if len(x_values) != value_matrix.shape[0] + 1:
        raise ValueError(
            f"x_values should be (value_matrix.shape[0] + 1) long. "
            f"x_values length is {len(x_values)} and value_matrix shape is "
            f"{value_matrix.shape}"
        )

    if len(y_values) != value_matrix.shape[1] + 1:
        raise ValueError(
            f"y_values should be (value_matrix.shape[1] + 1) long. "
            f"y_values length is {len(y_values)} and value_matrix shape is "
            f"{value_matrix.shape}"
        )
    
    # --- Get polygon formatting and colors --- 

    # get patch kwargs by union of default and user-specified kwargs
    patch_kwargs = DEFAULT_HMAP_PATCH_KWARGS | (patch_kwargs or {}) 
    missing_value_color = missing_value_color or DEFAULT_HMAP_MISSING_VALUE_COLOR

    # Figure out how to treat patch collection (img or vector)
    if heatmap_as_img:
        rasterized = True
    else:
        rasterized = False

    # calculate color values, replacing nans with missing_value_color
    rgba_array = color_mapper.to_rgba(value_matrix)
    rgba_array[np.isnan(value_matrix)] = mcolors.to_rgba(missing_value_color)

    # --- Build the heat map --- 
    
    # Build the polygons
    buffer = btwn_square_space/2
    square_list = []
    for i in range(value_matrix.shape[0]):

        left  = x_values[i  ] + buffer
        right = x_values[i+1] - buffer
            
        for j in range(value_matrix.shape[1]):

            bottom = y_values[j  ] + buffer
            top    = y_values[j+1] - buffer
                
            pts = np.array([[left ,bottom],
                            [left ,top   ],
                            [right,top   ],
                            [right,bottom]])

            square_list.append(Polygon(pts,facecolor=rgba_array[i,j],
                                       **patch_kwargs))

    # Build the patch collection
    heat_map = PatchCollection(square_list,
                               match_original=True,
                               rasterized=rasterized)

    return heat_map

def _build_color_mapper(
    color_fcn: Union[str, Callable, cm.ScalarMappable],
    values: Optional[np.ndarray] = None,
    vlim: Optional[Tuple[float, float]] = None,
    scale: Literal["linear", "log"] = "linear"
) -> cm.ScalarMappable:
    """
    Create a Matplotlib ScalarMappable object to map values to colors.

    This is a flexible factory for creating `ScalarMappable` objects. It can
    automatically determine color limits from data, handle diverging data by
    centering the color scale on zero, and build colormaps from either a
    string name, a custom color function, or a pre-existing mappable object.

    Parameters
    ----------
    color_fcn : str or callable or matplotlib.cm.ScalarMappable
        The colormap to use.
        - If already a `ScalarMappable`, it is returned directly.
        - If a string, it's used to look up a built-in Matplotlib colormap.
        - If a callable, it's a function that maps a float in `[0, 1]` to an
          RGB or RGBA tuple. It will be used to build a custom colormap.
    values : numpy.ndarray, optional
        An array of data values used to automatically determine the color
        limits (`vlim`). This is required if `vlim` is not provided.
    vlim : tuple of (float, float), optional
        A tuple specifying the `(vmin, vmax)` for the color scale. If
        provided, it overrides any calculation from `values`.
    scale : {"linear", "log"}, default "linear"
        The type of color scale normalization to apply.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        The configured mappable object, which can be used to convert data
        values to RGBA colors or to create a colorbar.

    Raises
    ------
    ValueError
        If both `vlim` and `values` are `None`, if `color_fcn` is an invalid
        string, or if `scale` is not one of the allowed values.
    """

    # pass through if already built
    if isinstance(color_fcn,cm.ScalarMappable):
        return color_fcn

    # Make sure at least vlim or values is defined
    if (vlim is None) and (values is None):
        raise ValueError(
            "either values or vlim must be defined to construct a color map."
        )

    # Get vlim from values if only values are defined
    if vlim is None:

        if np.nanmin(values) < 0 and np.nanmax(values) > 0:
            center_on_zero = True
        else:
            center_on_zero = False
        vlim = get_ax_limits(values,
                             center_on_zero=center_on_zero)

    # Build a custom cmap if the user passed a function
    if callable(color_fcn):

        try:
            color_list = [color_fcn(i/255.0) for i in range(256)]
            color_array = np.ones((256,4),dtype=float)
            tmp_color_array = np.array(color_list)
            color_array[:,:tmp_color_array.shape[1]] = tmp_color_array
            if np.min(color_array) < 0 or np.max(color_array) > 1:
                raise ValueError(
                    "RGB/A channel values must be between 0 and 1."
                )
        except Exception as e:
            raise ValueError(
                "color_fcn should be a function that takes a single value or "
                "1D numpy array as input and returns 3 or 4 floats between "
                "[0,1] for every input value. These are interpreted as the "
                "red, green, blue, and (optionally) alpha channels."
            ) from e

        # Create the colormap object
        cmap = mcolors.ListedColormap(color_array, name='_custom_cmap')

    else:
        try:
            cmap = plt.get_cmap(color_fcn)
        except ValueError as e:
            raise ValueError(
                f"color_fcn value of '{color_fcn}' is invalid. It should "
                "either be a function that takes a single value and returns "
                "RGB or RGBA values or a string for looking up matplotlib "
                "colormaps."
            ) from e

    if scale == "linear":
        norm = mcolors.Normalize(vmin=vlim[0], vmax=vlim[1])
    elif scale == "log":
        norm = mcolors.LogNorm(vmin=vlim[0], vmax=vlim[1])
    else:
        raise ValueError("scale must be 'linear' or 'log'")
    
    color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    return color_mapper
    
def _get_fig_dim(
    num_x: int,
    num_y: int,
    height: Optional[float] = None,
    width: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate figure dimensions for a heatmap while preserving aspect ratio.

    This function determines a sensible (width, height) for a figure based on
    the number of cells in the x and y dimensions. The logic is as follows:
    - If both height and width are provided, they are returned directly.
    - If only height is provided, width is calculated to match the data's aspect ratio.
    - If only width is provided, height is calculated to match the data's aspect ratio.
    - If neither is provided, a default height is used to calculate the width.

    Parameters
    ----------
    num_x : int
        The number of cells along the x-axis (determines width).
    num_y : int
        The number of cells along the y-axis (determines height).
    height : float, optional
        The desired figure height in inches.
    width : float, optional
        The desired figure width in inches.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the calculated `(width, height)`.

    Raises
    ------
    ValueError
        If `num_x` or `num_y` are not positive.
    """
    if num_x <= 0 or num_y <= 0:
        raise ValueError("num_x and num_y must be positive.")

    if height is not None and width is not None:
        # Both are specified, return them as is.
        return width, height
    elif height is not None:
        # Height is specified, calculate width based on aspect ratio.
        width = (float(num_x) / num_y) * height
        return width, height
    elif width is not None:
        # Width is specified, calculate height based on aspect ratio.
        height = (float(num_y) / num_x) * width
        return width, height
    else:
        # Neither is specified, use default height to calculate width.
        height = DEFAULT_HMAP_FIG_HEIGHT
        width = (float(num_x) / num_y) * height
        return width, height

def _get_ticks(
    tick_labels: Sequence[str],
    tick_values: np.ndarray,
    max_num_ticks: Optional[int] = None
) -> Tuple[Sequence[str], np.ndarray]:
    """
    Select a subset of ticks and labels to avoid overcrowding an axis.

    This function takes a complete list of labels and their corresponding grid
    boundaries, calculates the center position for each label, and then
    returns a downsampled set of labels and positions if the total number
    exceeds `max_num_ticks`.

    Parameters
    ----------
    tick_labels : Sequence[str]
        The complete list of labels for the axis (e.g., from a DataFrame index).
    tick_values : numpy.ndarray
        The grid boundary coordinates corresponding to the labels. Its length
        must be `len(tick_labels) + 1`.
    max_num_ticks : int, optional
        The maximum number of ticks to display. If `None` or greater than the
        number of labels, all ticks are returned. If <= 0, no ticks are
        returned.

    Returns
    -------
    tuple of (Sequence[str], numpy.ndarray)
        A tuple containing the downsampled `(labels, positions)`.
    """
    num_labels = len(tick_labels)

    # Handle cases where no labels should be returned.
    if num_labels == 0 or (max_num_ticks is not None and max_num_ticks <= 0):
        # Return empty arrays, preserving type if possible.
        empty_labels = tick_labels[0:0] 
        return empty_labels, np.array([])
    
    # Calculate the center position for every possible tick.
    centers = (tick_values[1:] + tick_values[:-1]) / 2

    # If no downsampling is needed, return all labels and their centers.
    if max_num_ticks is None or max_num_ticks >= num_labels:
        return tick_labels, centers
    
    # Calculate the step size and return the downsampled slice.
    step = int(np.ceil(num_labels / max_num_ticks))
    
    return tick_labels[::step], centers[::step]

def _format_axis(
    ax: Axes,
    tick_labels: Sequence[str],
    tick_values: np.ndarray,
    axis_key: Literal["x", "y"],
    max_num_ticks: Optional[int] = None,
    tick_length: Optional[float] = None,
    label_font: str = "Courier New",
    label_font_size: int = 20,
    label_horizontal_alignment: str = "center"
) -> None:
    """
    Apply tick and label formatting to a specified Matplotlib axis.

    This function serves as a generic wrapper to format either the x- or y-axis
    of a heatmap. It uses the `_get_ticks` helper to select an appropriate
    number of ticks to display and then applies various font and style
    properties to the axis labels and ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to format.
    tick_labels : Sequence[str]
        The complete list of labels for the axis.
    tick_values : numpy.ndarray
        The grid boundary coordinates corresponding to the labels.
    axis_key : {"x", "y"}
        A string specifying which axis to format.
    max_num_ticks : int, optional
        The maximum number of ticks to display. Passed to `_get_ticks`.
    tick_length : float, optional
        The desired length of the tick marks in points.
    label_font : str, default "Courier New"
        The font name for the tick labels.
    label_font_size : int, default 20
        The font size for the tick labels.
    label_horizontal_alignment : str, default "center"
        The horizontal alignment for the tick labels.

    Raises
    ------
    ValueError
        If `axis_key` is not 'x' or 'y'.
    """

    try:
        axis_obj = getattr(ax, f'{axis_key}axis')
    except AttributeError:
        raise ValueError(f"Invalid axis_key: '{axis_key}'. Must be 'x' or 'y'.")

    # Get labels and values at each tick
    labels, values = _get_ticks(tick_labels,
                                tick_values,
                                max_num_ticks=max_num_ticks)
    
    # Set styling on ticks
    axis_obj.set_ticks(values)
    if tick_length is not None:
        axis_obj.set_tick_params(length=tick_length)
    
    # Set styling on tick labels
    axis_obj.set_ticklabels(labels,
                            font=label_font,
                            size=label_font_size)
    for label in axis_obj.get_ticklabels():
        label.set_horizontalalignment(label_horizontal_alignment)

        
def heatmap(
    plot_df: pd.DataFrame,
    color_fcn: Union[str, Callable, ScalarMappable] = "pink_r",
    vlim: Optional[Tuple[float, float]] = None,
    cmap_scale: Literal["linear", "log"] = "linear",
    height: Optional[float] = None,
    width: Optional[float] = None,
    btwn_square_space: float = 0,
    grid: bool = False,
    grid_kwargs: Optional[dict] = None,
    plot_scale: bool = True,
    heatmap_patch_kwargs: Optional[dict] = None,
    missing_value_color: Optional[Union[str, tuple]] = None,
    heatmap_as_img: bool = False,
    x_axis_type: str = "site",
    y_axis_type: str = "aa",
    x_axis_kwargs: Optional[dict] = None,
    y_axis_kwargs: Optional[dict] = None,
    x_offset: float = 0,
    y_offset: float = 0,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create a heatmap from a matrix-like DataFrame.

    This function serves as a core plotter for visualizing a 2D matrix of
    values. It is built on a `PatchCollection` for maximum flexibility and
    clean vector graphics output. Key features include flexible color
    mapping, automatic axis formatting with presets, the ability to draw on
    existing axes, and various aesthetic customizations.

    Parameters
    ----------
    plot_df : pandas.DataFrame
        A 2D, matrix-like DataFrame where the index will be used for the
        x-axis labels, the columns for the y-axis labels, and the values
        will be mapped to the heatmap colors.
    color_fcn : str or callable or ScalarMappable, default "pink_r"
        The colormap to use. Can be a Matplotlib colormap name (str), a
        custom function, or a pre-built `ScalarMappable` object.
    vlim : tuple of (float, float), optional
        A tuple specifying the `(vmin, vmax)` for the color scale. If not
        provided, limits are inferred from the data.
    cmap_scale : {"linear", "log"}, default "linear"
        The scale to use for the colormap normalization.
    height : float, optional
        The height of the figure in inches. If not specified, it is
        calculated from the width or a default.
    width : float, optional
        The width of the figure in inches. If not specified, it is
        calculated from the height to match the data's aspect ratio.
    btwn_square_space : float, default 0
        The fraction of space between heatmap cells, from 0 (touching) to 1 
        (cells would be invisible)
    grid : bool, default False
        If True, draws grid lines around the heatmap cells.
    grid_kwargs : dict, optional
        A dictionary of keyword arguments to pass to `ax.plot` for styling
        the grid lines (e.g., `{"color": "white", "linewidth": 0.5}`).
    plot_scale : bool, default True
        If True, a colorbar is added to the plot.
    heatmap_patch_kwargs : dict, optional
        A dictionary of keyword arguments for styling the heatmap squares,
        passed to `matplotlib.patches.Polygon` (e.g., `{"edgecolor": "black"}`).
    missing_value_color : color-like, optional
        The color to use for `NaN` values in `plot_df`.
    heatmap_as_img : bool, default False
        If True, the heatmap patch collection is rasterized. This is highly
        recommended for vector outputs (PDF, SVG) with large heatmaps to
        reduce file size and improve rendering speed.
    x_axis_type : str, default "site"
        The name of a formatting preset for the x-axis. Allowed values are 
        "site", "aa", or "titrant".
    y_axis_type : str, default "aa"
        The name of a formatting preset for the y-axis. Allowed values are 
        "site", "aa", or "titrant".
    x_axis_kwargs : dict, optional
        A dictionary of keyword arguments to override the x-axis preset.
    y_axis_kwargs : dict, optional
        A dictionary of keyword arguments to override the y-axis preset.
    x_offset : float, default 0
        A value to shift the entire heatmap along the x-axis.
    y_offset : float, default 0
        A value to shift the entire heatmap along the y-axis.
    ax : matplotlib.axes.Axes, optional
        A pre-existing `Axes` object to draw the heatmap on. If `None`, a new
        figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object for the plot.
    ax : matplotlib.axes.Axes
        The Axes object containing the heatmap.

    Raises
    ------
    ValueError
        If inputs are invalid, such as a non-DataFrame `plot_df`, a
        non-numeric `plot_df`, an invalid `ax` object, or an unrecognized
        `x_axis_type` or `y_axis_type`.
    """

    # Load the dataframe
    if not isinstance(plot_df,pd.DataFrame):
        raise ValueError(
            "plot_df should be a pandas DataFrame with the desired x-axis "
            "as rows and the y-axis as the columns."
        )

    # Grab values for generating the heat map from the dataframe
    try:
        value_matrix = np.asarray(plot_df.to_numpy(),dtype=float)
    except Exception as e:
        raise ValueError(
            "The values in plot_df must all be numbers (coercible to float)"
        )

    # Get color mapper
    color_mapper = _build_color_mapper(color_fcn,
                                       values=value_matrix.flatten(),
                                       vlim=vlim,
                                       scale=cmap_scale)
    
    # Grab the dimensions of the plot
    num_x = plot_df.shape[0] + 1
    num_y = plot_df.shape[1] + 1

    # Get x- and y-values. (... xx customize in future)
    x_values = np.arange(num_x) + x_offset
    y_values = np.arange(num_y) + y_offset

    # Get fig/ax pair. Either build from scratch or grab fig from existing 
    # ax to allow a consistent return. 
    if ax is None:
        figsize = _get_fig_dim(num_x,num_y,height=height,width=width)
        fig, ax = plt.subplots(1,figsize=figsize)
        is_new_ax = True
    else:
        if not isinstance(ax,matplotlib.axes.Axes):
            raise ValueError("ax must be a matplotlib Axes instance")
        fig = ax.get_figure()
        is_new_ax = False

    # Get heat map squares
    heatmap_collection = _build_heatmap_collection(
        value_matrix=value_matrix,
        color_mapper=color_mapper,
        x_values=x_values,
        y_values=y_values,
        patch_kwargs=heatmap_patch_kwargs,
        missing_value_color=missing_value_color,
        btwn_square_space=btwn_square_space,
        heatmap_as_img=heatmap_as_img,
    )
                                                   
    # Append heatmap to plot
    ax.add_collection(heatmap_collection)
        
    # grid
    if grid:

        # Figure out grid kwargs by union of whatever the user passed in and
        # defaults. 
        grid_kwargs = DEFAULT_HMAP_GRID_KWARGS | (grid_kwargs or {}) 
        
        xmin, xmax = x_values[0], x_values[-1]
        ymin, ymax = y_values[0], y_values[-1]
        
        # Draw vertical lines
        for x in x_values:
            ax.plot([x, x], [ymin, ymax], **grid_kwargs)
        
        # Draw horizontal lines
        for y in y_values:
            ax.plot([xmin, xmax], [y, y], **grid_kwargs)

    # -- Format axes --

    # Grab default axis formatting 
    default_axis_kwargs = {"site":DEFAULT_HMAP_SITE_AXIS_KWARGS,
                           "aa":DEFAULT_HMAP_AA_AXIS_KWARGS,
                           "titrant":DEFAULT_HMAP_TITRANT_AXIS_KWARGS}

    # Go through x and y and generate final kwargs
    kwargs = []
    for var_name, ax_type, ax_kwargs in zip(["x_axis_type","y_axis_type"],
                                            [x_axis_type,y_axis_type],
                                            [x_axis_kwargs,y_axis_kwargs]):

        # x_axis_type or y_axis_type is not one of the recognized types 
        if ax_type not in default_axis_kwargs:
            raise ValueError(
                f"{var_name} '{ax_type}' not recognized. Should be one "
                f"of: {list(default_axis_kwargs.keys())}."
            )

        # Record kwargs
        kwargs.append(default_axis_kwargs[ax_type] | (ax_kwargs or {}))

    # Unpack kwargs
    x_axis_kwargs, y_axis_kwargs = kwargs

    # Format axes
    _format_axis(ax,plot_df.index,x_values,axis_key="x",**x_axis_kwargs)
    _format_axis(ax,plot_df.columns,y_values,axis_key="y",**y_axis_kwargs)

    # Plot a color scale bar
    if plot_scale:
        fig.colorbar(mappable=color_mapper,ax=ax,shrink=0.85)

    # Further format axes (if this is new axis object). If an ax is passed in,
    # assume other function(s) are responsible for these formatting details.
    if is_new_ax:
        ax.set_xlim(np.min(x_values) - 0.5,np.max(x_values) + 0.5)
        ax.set_ylim(np.min(y_values) - 0.5,np.max(y_values) + 0.5)
        ax.set_aspect("equal")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.tight_layout()
    
    return fig, ax