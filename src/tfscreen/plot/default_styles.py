
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


DEFAULT_SCATTER_KWARGS = {"s":10,
                          "alpha":0.1,
                          "edgecolor":"royalblue",
                          "facecolor":"none"}

DEFAULT_FIT_LINE_KWARGS = {"lw":2,
                           "color":"firebrick"}

DEFAULT_EXPT_SCATTER_KWARGS = {"s":70,
                               "edgecolor":"black",
                               "facecolor":"none"}

DEFAULT_EXPT_ERROR_KWARGS = {"color":"black",
                             "lw":0,
                             "elinewidth":1,
                             "capsize":10}

