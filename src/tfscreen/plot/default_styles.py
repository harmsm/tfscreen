
from matplotlib import pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


DEFAULT_SCATTER_KWARGS = {
    "s":10,
    "alpha":0.1,
    "edgecolor":"royalblue",
    "facecolor":"none"
}

DEFAULT_FIT_LINE_KWARGS = {
    "lw":2,
    "color":"firebrick"
}

DEFAULT_EXPT_SCATTER_KWARGS = {
    "s":70,
    "edgecolor":"black",
    "facecolor":"none"
}

DEFAULT_EXPT_ERROR_KWARGS = {
    "color":"black",
    "lw":0,
    "elinewidth":1,
    "capsize":10
}

# Heat map styling

DEFAULT_HMAP_FIG_HEIGHT = 6
DEFAULT_HMAP_GRID_KWARGS = {
    "lw":0.5,
    "color":"gray",
    "zorder":20 # should put it on top for most plots
}
DEFAULT_HMAP_PATCH_KWARGS = {"edgecolor":"none"}
DEFAULT_HMAP_MISSING_VALUE_COLOR = (0.7,0.7,0.7,1.0)

DEFAULT_HMAP_AA_AXIS_KWARGS = {
    "tick_length":0,
    "label_font":"Courier New", # use a monospace font
    "label_font_size":20,
    "max_num_ticks":None,       # this will label every amino acid
    "label_horizontal_alignment":"center"
}

DEFAULT_HMAP_TITRANT_AXIS_KWARGS = {
    "tick_length":None,  # use default matplotlib tick length
    "label_font":"Arial",
    "label_font_size":20,
    "max_num_ticks":10,
    "label_horizontal_alignment":"center"
}

# Site axis is usually residue numbering
DEFAULT_HMAP_SITE_AXIS_KWARGS = {
    "tick_length":0, 
    "label_font":"Arial",
    "label_font_size":20,
    "max_num_ticks":26,
    "label_horizontal_alignment":"center"
}