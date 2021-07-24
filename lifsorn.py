#%%
###############################################################################
# Import libraries
###############################################################################

from brian2 import *
import numpy as np
from pars import *

#%%
###############################################################################
# Network creation
###############################################################################

def create_LIFSORN(rng):

    ###########################   Prolog   ####################################
    ### Extra clocks
    slow_time = 1 * second
    slow_clock = Clock(slow_time, name="slow_clock")

    ###########################   Neurons   ###################################
    print("Initialize neurons")

    ### LIF neuron with adjustable threshold and white noise
    noisylif = """
        dV / dt =  (El - V + ge*(Ee-V) + gi*(Ei-V))/tau + sigma_noise * xi / (tau **.5): volt 
        dge /dt = -ge/tau_e : 1   # conductance exc. synapses
        dgi /dt = -gi/tau_i : 1   # conductance inh. synapses
        a : 1 (constant)          # factor for adjusting total weight according to position
        x : meter (constant)      # position in x-direction
        y : meter (constant)      # position in y-direction
        """
    noisylif_e = Equations(noisylif + "\n" + "dVt / dt = -eta_ip * h_ip : volt")
    noisylif_i = Equations(noisylif)
    reset_e = """
        V = Vr_e
        Vt += eta_ip
        """

    ### Neuron Groups
    G_e = NeuronGroup(
        N_e, noisylif_e, threshold="V > Vt", reset=reset_e, method="euler", name="G_e"
    )  # excitatory group
    G_i = NeuronGroup(
        N_i,
        noisylif_i,
        threshold="V > Vt_i",
        reset="V = Vr_i",
        method="euler",
        name="G_i",
    )  # inhibitory group

    ### Randomize initial voltages
    G_e.V = -(Vvi + rng.random(N_e) * Vvar)
    G_i.V = -(Vvi + rng.random(N_i) * Vvar)

    ### Randomize initial excitatory thresholds
    G_e.Vt = -(Vti + rng.random(N_e) * Vtvar)

    ### Topology
    X_e = size_T * rng.random((N_e, 2))  # excitatory neuron positions
    G_e.x = X_e[:, 0]
    G_e.y = X_e[:, 1]
    G_e.a = posTOa(X_e)
    X_i = size_T * rng.random((N_i, 2))  # inhibitory neuron positions
    G_i.x = X_i[:, 0]
    G_i.y = X_i[:, 1]
    G_i.a = posTOa(X_i)
    
    ###########################   Synapses   ##################################
    print("Initialize synapses")
    
    ### Connection probability
    P_eTOe = conprob(X_e, X_e, intra_group=True)
    P_eTOi = conprob(X_e, X_i)
    P_iTOe = conprob(X_i, X_e)
    P_iTOi = conprob(X_i, X_i, intra_group=True)

    ### Synapses exc. to exc.
    
    # Synaptic equations executed at every timestep
    syn_eTOe = """
        p : 1                                           # connection probability
        w : 1                                           # weight
        c : 1                                           # connection variable (1 (0) if synapse (non-)existant)
        dapre/dt = -apre/taupre : 1 (event-driven)      # STDP trace (potentiation)
        dapost/dt = -apost/taupost : 1 (event-driven)   # STDP trace (depression)
        du/dt = (U-u)/tauf : 1 (event-driven)           # STP trace (facilitation)
        dx_STP/dt = (1-x_STP)/taud : 1 (event-driven)   # STP trace (depression)
        """
    
    # Synaptic equations executed upon a presynaptic spike
    synpre_eTOe = """
        ge_post += u * x_STP * w                        # increment exc. conductance
        x_STP -= c * x_STP * u                          # increment STD variable
        u += c * U * (1-u)                              # increment STF variable
        apre = c * Ap                                   # nearest neighbor STDP
        w = c * clip(w + apost, 0, total_in_eTOe)       # nearest neighbor STDP
        """
    
    # Synaptic equations executed upon a postsynaptic spike
    synpost_eTOe = """
        apost = c * Ad                                  # nearest neighbor STDP
        w = c * clip(w + apre, 0, total_in_eTOe)        # nearest neighbor STDP
        """
    
    # Synapses
    S_eTOe = Synapses(
        G_e,
        G_e,
        syn_eTOe,
        on_pre=synpre_eTOe,
        on_post=synpost_eTOe,
        delay=delay_eTOe,
        name="S_eTOe",
    )
    
    # Intialize synaptic parameters
    S_eTOe.connect(condition="i!=j")
    S_eTOe.p = P_eTOe[~np.eye(N_e, dtype=bool)]
    S_eTOe.u = U
    S_eTOe.x_STP = 1
    S_eTOe.c = 0
    S_eTOe.w = 0

    # Synaptic normalization
    def synnorm():
        W_ee = np.zeros((N_e, N_e))
        W_ee[S_eTOe.i[:], S_eTOe.j[:]] = S_eTOe.w[:]
        W_e_in = W_ee[:, :].sum(0) / G_e.a
        W_e_ins = np.tile(W_e_in[:], (N_e, 1))[~np.eye(N_e, dtype=bool)]
        W_e_ins[W_e_ins == 0] = 1
        S_eTOe.w = S_eTOe.w * total_in_eTOe / W_e_ins

    @network_operation(when="end", order=0, clock=slow_clock, name="synnorm")
    def synnorm_op():
        synnorm()

    # Structural pruning
    struct_prune = S_eTOe.run_regularly(
        "c = int(w>zero_cut); w*=c",
        clock=slow_clock,
        when="end",
        order=1,
        name="struct_prune",
    )

    # Structural growth
    @network_operation(when="end", order=2, clock=slow_clock, name="struct_growth")
    def struct_growth():
        C = np.zeros_like(P_eTOe)
        C[S_eTOe.i[:], S_eTOe.j[:]] = S_eTOe.c[:]
        n_new = round(rng.normal(sp_rate * slow_time, np.sqrt(sp_rate * slow_time)))
        p_val = (1 - C) * P_eTOe / rng.random((N_e, N_e))
        p_val = p_val[~np.eye(N_e, dtype=bool)]
        Syn_idx = np.nonzero(p_val > np.sort(p_val)[-n_new - 1])[0]
        S_eTOe.c[Syn_idx] = 1
        S_eTOe.u[Syn_idx] = U
        S_eTOe.x_STP[Syn_idx] = 1
        S_eTOe.apre[Syn_idx] = 0
        S_eTOe.apost[Syn_idx] = 0
        S_eTOe.w[Syn_idx] = sp_initial
        synnorm()

    ### Other synapses
    
    # Synaptic equations executed at every timestep
    syn_eTOi = """
    w : 1 (constant)    # weight
    """
    syn_iTOx = """
    w : 1 (constant)    # weight
    """
    
    # Synaptic equations executed upon a presynaptic spike
    synpre_eTOi = """
    ge_post += w        # increment exc. conductance
    """
    synpre_iTOx = """
    gi_post += w        # increment inh. conductance
    """
    
    # Synapses and intialization of synaptic parameters
    S_eTOi = Synapses(
        G_e, G_i, syn_eTOi, on_pre=synpre_eTOi, delay=delay_eTOi, name="S_eTOi"
    )
    Cidx_eTOi = init_con(P_eTOi, sparse_eTOi, rng)
    S_eTOi.connect(i=Cidx_eTOi[0], j=Cidx_eTOi[1])
    S_eTOi.w = "a_post * total_in_eTOi / N_incoming"

    S_iTOe = Synapses(
        G_i, G_e, syn_iTOx, on_pre=synpre_iTOx, delay=delay_iTOe, name="S_iTOe"
    )
    Cidx_iTOe = init_con(P_iTOe, sparse_iTOe, rng)
    S_iTOe.connect(i=Cidx_iTOe[0], j=Cidx_iTOe[1])
    S_iTOe.w = "a_post * total_in_iTOe / N_incoming"

    S_iTOi = Synapses(
        G_i, G_i, syn_iTOx, on_pre=synpre_iTOx, delay=delay_iTOi, name="S_iTOi"
    )
    Cidx_iTOi = init_con(P_iTOi, sparse_iTOi, rng)
    S_iTOi.connect(i=Cidx_iTOi[0], j=Cidx_iTOi[1])
    S_iTOi.w = "a_post * total_in_iTOi / N_incoming"
    
    ###########################   Network   ###################################
    LIFSORN = Network(
        G_e,
        G_i,
        S_eTOe,
        S_eTOi,
        S_iTOe,
        S_iTOi,
        synnorm_op,
        struct_prune,
        struct_growth,
        name="LIFSORN",
    )

    return LIFSORN
