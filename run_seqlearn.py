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
from experiments import *

#%%
###############################################################################
# Run experiments
###############################################################################

# This script runs all sequence learning experiments. Running them sequentially 
# (as done here) takes a long time. So it is advisable to parallelize the code. 
# Hence, this script is more like a blueprint to get the data underlying the 
# figures shown in the notebooks. Further note that for each random seed the 
# persistence experiments creates almost 500MB of data.
seeds_run = 0 # takes track of used random seeds

### Cue at M
for seedi in range(seeds_run, seeds_run + seeds_num):
    run_seqlearn_exp(seedi, cue='m')
    seeds_run += 1
    
### Cue at G
for seedi in range(seeds_run, seeds_run + seeds_num):
    run_seqlearn_exp(seedi, cue='g')
    seeds_run += 1

### Cue at S for different speeds
for v in spot_vs:
    for seedi in range(seeds_run, seeds_run + seeds_num):
        run_seqlearn_exp(seedi, spot_v=v)
        seeds_run += 1
    
### No cue
for seedi in range(seeds_run, seeds_run + seeds_num_1):
    run_seqlearn_exp(seedi, cue='0')
    seeds_run += 1
        
### Persistence
for seedi in range(seeds_run, seeds_run + seeds_num_1):
    run_seqlearn_exp(seedi, persistence=True)
    seeds_run += 1