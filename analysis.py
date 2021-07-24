# %%
###############################################################################
# Import libraries
###############################################################################

import pickle
from brian2 import *
import numpy as np
from scipy.signal import argrelmax
from scipy.stats import spearmanr
from pars import *
from pars_exp import *

# %%
###############################################################################
# Data loading and preparation
###############################################################################

# Load data
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Returns cluster neurons sorted according to postion along spot path
def get_cluster_neurons(X_e):
    clu_neur = []
    for clu in range(clu_num):
        clu_center = clu_start + (clu_end - clu_start) * clu / (clu_num-1)
        clu_neur.append(
            np.arange(N_e)[(np.linalg.norm(X_e-clu_center, axis=1) * meter < clu_r)])
        clu_neur[-1] = clu_neur[-1][np.argsort(X_e[clu_neur[-1], 0])]
    return clu_neur

# Returns spikes of (cluster) neurons for each trial
# in the test phase starting after t_start
def get_spikes(clu_neurs, spikes_dic, t_start):
    spikes = np.concatenate([spikes_dic[neur] for neur in clu_neurs]) * second
    spikes_list = []
    for trial in range(trials_n):
        trial_start = t_start + offset_time + trial * seq_break
        spikes_list.append(spikes[(trial_start < spikes) & (
            spikes <= trial_start + trial_dur)])
    return spikes_list


# %%
###############################################################################
# Cross-correlation
###############################################################################

# Cross-correlation between spikes of two clusters
def cc_clu(spikes_a, spikes_b):

    bins = np.linspace(-cc_edge, cc_edge, cc_bins + 1)
    ccs = np.zeros(cc_bins)
    spikes_bins = np.linspace(0*ms, trial_dur, int(trial_dur/cc_binsize) + 1)
    spikes_count_a = []
    spikes_count_b = []
    for s_a, s_b in zip(spikes_a, spikes_b):
        spikes_count_a.append(np.histogram((s_a-offset_time)%seq_break, spikes_bins)[0])
        spikes_count_b.append(np.histogram((s_b-offset_time)%seq_break, spikes_bins)[0])
        for t_a in s_a:
            ccs += np.histogram(s_b - t_a, bins)[0] / len(spikes_a)
    spikes_count_a = np.array([spikes_count_a])
    spikes_count_b = np.array([spikes_count_b])
    spikes_a_std = np.std([spikes_count_a])
    spikes_b_std = np.std([spikes_count_b])
    ccs = ((ccs - spikes_count_a.mean() * spikes_count_b.mean()) 
           / (spikes_count_a.std()*spikes_count_b.std()))

    return ccs


# %%
###############################################################################
# Rate, firing times, Spearman CC
###############################################################################

# Firing rate and times
def rates_ftimes(spikes, t_start,  clu_neur_n, tau=rate_tau):
    rates = []
    firing_times = []
    for trial, s in enumerate(spikes):
        trial_start = t_start + offset_time + trial * seq_break
        times = np.linspace(trial_start, trial_start +
                            trial_dur, int(trial_dur/rate_dt)+1)
        if len(s) > 1:
            rates.append(np.sum([gaussian(s_i, times, tau) for s_i in s], 0))
        else:
            rates.append(np.zeros_like(times))
        rates[-1] /= clu_neur_n
        firing_idx = argrelmax(rates[-1])[0][0] if len(
            argrelmax(rates[-1])[0]) > 0 else int(1/2 * trial_dur / rate_dt)
        firing_times.append(times[firing_idx])
    return rates, firing_times

# Spearman CC
def sp_corr(firing_times, clu_seq=np.arange(clu_num)):
    spCCs = []
    for ftimes in firing_times:
        spCCs.append(spearmanr(ftimes, clu_seq)[0])
    return spCCs