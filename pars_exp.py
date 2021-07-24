#%%
###############################################################################
# Import libraries
###############################################################################

from brian2 import *
import numpy as np
from pars import *
import matplotlib
import matplotlib.pyplot as plt

#%%
###############################################################################
# Simulation
###############################################################################

### Time parameters
seeds_num = 20  # number of seeds for most experiments
seeds_num_1 = 30  # number of seeds for persistence and spont. act. experiments
growth_time = 400 * second  # growth time
test_time = 100 * second  # test time
test_time_afper = 300 * second  # test time after learning for persistence experiment
learn_time = 200 * second  # training/learning time
relax_time = 10 * second  # relaxation time

### Cluster parameters
clu_start = np.array([375, 500]) * umeter  # position of first cluster
clu_end = np.array([2125, 500]) * umeter  # position of last cluster
clu_shift = np.array([0, 150]) * umeter  # shift of clusters/sequence from default position
clu_num = 8  # number of clusters/elements
clu_r = 100 * umeter  # radius of clusters

### Light spot
seq_break = 2000 * ms  # break between trials
offset_time = 199.9 * ms  # time after full second at which input starts
spot_start = clu_start - np.array([clu_r, 0]) * meter  # start position of spot
spot_end = clu_end + np.array([clu_r, 0]) * meter  # end position of spot
spot_mid = 1 / 2 * (spot_end + spot_start)  # mid position of spot
spot_dist = np.linalg.norm(spot_end - spot_start) * meter  # distance of spot travel
w_ff = 0.04  # weight of input connections
N_aff = 100  # number of input spiketrains
spot_flash = 100 * ms  # duration of flash during testing
bar_flash = 400 * ms  # duration of bar flash during learning
spot_peak = 50 * Hz  # max rate of input spike trains
spot_width = 150 * umeter  # scale of light spot
spot_v = 4 * umeter/ms  # default speed of light spot
spot_vs = np.linspace(4, 20, 5)  * umeter/ms  # speeds of light spot

# input rate
def spot_rate(x, u):
    g = spot_peak * np.exp(-((np.linalg.norm(x - u, axis=1) * meter / spot_width) ** 4))
    return g

### Light bar

# Function to determine distance from a point to a line
def pnt2line(pnt, start, end):
    line_vec = end - start
    pnt_vec = pnt - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = 1 / line_len * line_vec
    pnt_vec_scaled = 1 / line_len * pnt_vec
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = t * line_vec
    dist = np.linalg.norm(pnt_vec - nearest)
    return dist


# Exponential power function describing the bar stimulus
def bar_rate(x):
    g = spot_peak * np.exp(-((pnt2line(x, spot_start, spot_end) * meter / spot_width) ** 4))
    return g

#%%
###############################################################################
# Analysis and plotting
###############################################################################

### Trial specifications
trial_dur = 500 * ms  # trial duration
trials_n = int(test_time/seq_break)  # nmber of trials per test phase

### Pairwise Cross-Correlogram parameters
cc_binsize = 5 * ms  # bin size
cc_delta = 50 * ms  # maximal time difference 
cc_edge = cc_delta+cc_binsize/2  # outer edge of bins
cc_bins = int(2*cc_edge/cc_binsize)  # number of bins

### Rate
rate_dt = 1 * ms  # time step for computation of firing rate
rate_tau = 50 * ms  # time scale of gaussian filter kernel

### Helper object to create custom colormap
### Adapted from "http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib"
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 1025)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 513, endpoint=False), 
        np.linspace(midpoint, 1.0, 512, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < 5e-3:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0) # midpoint of seismic cmap is not white...
        else:
            r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
