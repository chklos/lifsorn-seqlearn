#%%
###############################################################################
# Import libraries
###############################################################################

import pickle
from brian2 import *
import numpy as np
from pars import *
from pars_exp import *
from lifsorn import *

#%%
###############################################################################
# Free dynamics (without input)
###############################################################################

def run_free(seedi=0, report="text", report_period=60*second):

    ### Set seed
    seed(seedi)
    rng = np.random.default_rng(seedi)

    ### Network
    slow_clock = Clock(1 * second, name="slow_clock")
    LIFSORN = create_LIFSORN(rng)
    G_e, G_i, S_eTOe = LIFSORN.sorted_objects[:3]

    ### Recording
    
    # Spikes
    MS_e = SpikeMonitor(G_e, name="MS_e")  # exc. spikes
    MS_i = SpikeMonitor(G_i, name="MS_i")  # inh. spikes

    # Exc. to exc. weights
    highp_con = np.where(S_eTOe.p >= np.sort(S_eTOe.p)[-int(1e3)])[0]
    MW_eTOe = StateMonitor(S_eTOe, "w", record=highp_con, when="end", 
                           order=10, clock=slow_clock, name="MW_eTOe")
    
    # Connection fraction
    CF_eTOe = []
    tot_con_eTOe = N_e * (N_e - 1)
    @network_operation(when="end", order=10, clock=slow_clock, name="record_CF_eTOe")
    def record_CF_eTOe():
        CF_eTOe.append(sum(S_eTOe.c) / tot_con_eTOe)

    ### Simulation
    print("Growth phase")
    LIFSORN.add(record_CF_eTOe)
    LIFSORN.run(growth_time, report=report, report_period=report_period)
    print("Analysis phase")
    LIFSORN.add(MS_e, MS_i, MW_eTOe)
    LIFSORN.run(test_time, report=report, report_period=report_period)
    print("Done")

    return LIFSORN, np.array(CF_eTOe)


#%%
###############################################################################
# Sequence learning
###############################################################################


def run_seqlearn(seedi=0, cue="s", spot_v=spot_v, test_time_af=test_time, 
                 record_weights=False, report="text", report_period=100*second):

    ### Set seed
    seed(seedi)
    rng = np.random.default_rng(seedi)

    ### Network
    LIFSORN = create_LIFSORN(rng)
    G_e, G_i, S_eTOe = LIFSORN.sorted_objects[:3]

    ### Flashing spot
    X_e = np.array([G_e.x[:], G_e.y[:]]).T * meter
    if cue == "s":
        flash_p_spike = defaultclock.dt * spot_rate(X_e, spot_start)
    elif cue == "m":
        flash_p_spike = defaultclock.dt * spot_rate(X_e, spot_mid)
    elif cue == "g":
        flash_p_spike = defaultclock.dt * spot_rate(X_e, spot_end)

    @network_operation(when="end", clock=defaultclock, name='flash')
    def flash(t):
        t_rel = t % seq_break - offset_time
        if 0 * ms <= t_rel < spot_flash:
            G_e.ge = G_e.ge + w_ff * np.sum(
                rng.random((N_aff, N_e)) < np.tile(flash_p_spike, (N_aff, 1)), 0
            )

    ### Moving spot
    spot_dur = spot_dist / spot_v
    @network_operation(when="end", clock=defaultclock, name='sequence')
    def sequence(t):
        t_rel = t % seq_break - offset_time
        if 0 * ms <= t_rel < spot_dur:
            spot_center = spot_start + (spot_end-spot_start) * (t_rel/spot_dur)
            seq_p_spike = defaultclock.dt * spot_rate(X_e, spot_center)
            G_e.ge = G_e.ge + w_ff * np.sum(
                rng.random((N_aff, N_e)) < np.tile(seq_p_spike, (N_aff, 1)), 0
            )

    ### Recording
    
    # Spikes
    MS_e = SpikeMonitor(G_e, name="MS_e")
    
    # Exc. to exc. weights
    if record_weights:
        very_slow_clock = Clock(10 * second, name="very_slow_clock")
        MW_eTOe = StateMonitor(S_eTOe, "w", record=True, when="end", order=10, 
                               name="MW_eTOe", clock=very_slow_clock)
    else:
        MW_eTOe = StateMonitor(S_eTOe, "w", record=True, when="end", order=10, 
                               name="MW_eTOe", dt=100_000*second)

    ### Simulation
    print("Growth phase")
    LIFSORN.run(growth_time, report=report, report_period=report_period)
    print("Prelearning phase")
    LIFSORN.add(MS_e, MW_eTOe)
    if not cue=="0":
        LIFSORN.add(flash)
    LIFSORN.run(test_time, report=report, report_period=report_period)
    if not cue=="0":
        flash.active = False
    LIFSORN.run(relax_time, report=report, report_period=report_period)
    print("Learning phase")
    if not record_weights:
        MW_eTOe.record_single_timestep()
    LIFSORN.add(sequence)
    LIFSORN.run(learn_time, report=report, report_period=report_period)
    LIFSORN.remove(sequence)
    if not record_weights:
        MW_eTOe.record_single_timestep()
    print("Postlearning phase")
    LIFSORN.run(relax_time, report=report, report_period=report_period)
    if not cue=="0":
        flash.active = True
    LIFSORN.run(test_time_af, report=report, report_period=report_period)
    if not cue=="0":
        LIFSORN.remove(flash)
    print("Done")

    return LIFSORN


# Wrapper to run simulation and save data relevant for analysis
def run_seqlearn_exp(seedi, cue='s', spot_v=spot_v, persistence=False):
    if persistence:
        LIFSORN = run_seqlearn(seedi, record_weights=True, test_time_af=test_time_afper)
    else:
        LIFSORN = run_seqlearn(seedi, cue=cue, spot_v=spot_v)
    G_e = LIFSORN.sorted_objects[0]
    MS_e = LIFSORN.sorted_objects[10]
    MW_eTOe = LIFSORN.sorted_objects[-1]
    weights = MW_eTOe.w[:]
    X_e = np.array([G_e.x[:], G_e.y[:]]).T * meter
    spikes = MS_e.spike_trains()
    dic = {'X_e' : X_e, 'spikes' : spikes, 'weights' : weights}
    if persistence:
        with open("data/seqlearn_cues_per_seed{}.pickle".format(seedi), 'wb') as f:
            pickle.dump(dic, f)
    else:
        with open("data/seqlearn_cue{}_v{}_seed{}.pickle".format(cue, int(spot_v*ms/umeter), seedi), 'wb') as f:
            pickle.dump(dic, f)