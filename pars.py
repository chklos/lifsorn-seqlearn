#%%
###############################################################################
# Import libraries
###############################################################################

from brian2 import *
import numpy as np
from scipy.special import erf

#%%
###############################################################################
# Parameters and helper functions
###############################################################################

# Factor for adjusting total weight according to position
def posTOa(x):
    return (
        1 / 4
        * (
            erf((length_T - x[:, 0]) / np.sqrt(2 * width_T ** 2))
            - erf((-x[:, 0]) / np.sqrt(2 * width_T ** 2))
        )
        * (
            erf((height_T - x[:, 1]) / np.sqrt(2 * width_T ** 2))
            - erf((-x[:, 1]) / np.sqrt(2 * width_T ** 2))
        )
    )

# Gaussian 
def gaussian(x, u, s):
    g = 1 / np.sqrt(2 * np.pi * s * s) * np.exp(-(x - u) * (x - u) / (2 * s * s))
    return g

# Determines connectivity
def init_con(P, sparse, rng):
    N_pre, N_post = np.shape(P)
    n_new = int(round(N_pre * N_post * sparse))
    p_val = P / rng.random((N_pre, N_post))
    C_idx = np.nonzero((p_val > np.sort(p_val, axis=None)[-n_new - 1]))
    return C_idx

# Determines connection probability
def conprob(X_pre, X_post, intra_group=False):
    N_pre, N_post = X_pre.shape[0], X_post.shape[0]
    D = np.zeros((N_pre, N_post)) * meter
    for i in range(N_pre):
        Dx = X_pre[i, 0] - X_post[:, 0]
        Dy = X_pre[i, 1] - X_post[:, 1]
        D[i, :] = np.sqrt(Dx ** 2 + Dy ** 2)
    P = 2 * gaussian(D, 0 * meter, width_T) * umeter
    if intra_group:
        np.fill_diagonal(P, 0)
    return P

### Base parameters
length_T = 2500 * umeter  # sheet length
height_T = 1000 * umeter  # sheet height
size_T = np.array([length_T, height_T]) * meter  # sheet size
N_e = 1000  # excitatory population size
N_i = int(0.2 * N_e)  # inhibitory population size

### Neuron parameters
sigma_noise = 16 * mV  # noise amplitude
tau = 20 * ms  # membrane time constant
Vr_e = -70 * mV  # excitatory reset potential
Vr_i = -60 * mV  # inhibitory reset potential
El = -60 * mV  # resting potential
Vti = 30 * mV  # minus maximum initial threshold voltage
Vtvar = 5 * mV  # maximum initial threshold voltage swing
Vvi = 50 * mV  # minus maximum initial voltage
Vvar = 20 * mV  # maximum initial voltage swing
Vvi_i = 50 * mV  # minus maximum initial inh. voltage
Vvar_i = 20 * mV  # maximum initial voltage swing
Vt_i = -48 * mV  # threshold of inhibitory neurons

### Synapse parameters
width_T = 200 * umeter  # growth radius
sparse_eTOe = 0.1  # target recurrent excitatory sparseness
sparse_iTOe = 0.1  # inhibitory to excitatory sparseness
sparse_eTOi = 0.1  # excitatory to inhibitory  sparseness
sparse_iTOi = 0.5  # inhibitory to inhibitory sparseness
wi_eTOe = 0.8  # target e->e weight
wi_eTOi = 0.15  # initial e->i weight
wi_iTOe = 0.4  # initial i->e weight
wi_iTOi = 0.4  # initial i->i weight
delay_eTOe = 3 * ms  # e->e latency
delay_eTOi = 1 * ms  # e->i latency
delay_iTOe = 2 * ms  # i->e latency
delay_iTOi = 2 * ms  # i->i latency
tau_e = 3 * ms  # EPSP time constant
tau_i = 5 * ms  # IPSP time constant
Ee = 0 * mV  # reversal potential excitation
Ei = -80 * mV  # reversal potential inhibition

### STDP parameters
taupre = 15 * ms  # pre-before-post STDP time constant
taupost = taupre * 2.0  # post-before-pre STDP time constant
Ap = 4.8e-1  # potentiating STDP learning rate
Ad = -Ap * 0.5  # depressing STDP learning rate

### STP parameters
U = 0.04  # faciliation increment
tauf = 2000 * ms  # faciliation time constant
taud = 500 * ms  # depression time constant

### Intrinsic plasticity parameters
h_ip = 3 * Hz  # target rate
eta_ip = 0.1 * mV  # IP learning rate

### Synaptic normalization parameters
total_in_eTOe = N_e * sparse_eTOe * wi_eTOe  # total e->e synaptic input
total_in_iTOe = N_i * sparse_iTOe * wi_iTOe  # total i->e synaptic input
total_in_eTOi = N_e * sparse_eTOi * wi_eTOi  # total e->i synaptic input
total_in_iTOi = N_i * sparse_iTOi * wi_iTOi  # total i->i synaptic input

### Structural plasticity parameters
sp_initial = 1e-3  # initial weight for newly created synapses
zero_cut = 1e-4  # zero pruning cutoff
sp_rate = 6000  # stochastic rate of new synapse production