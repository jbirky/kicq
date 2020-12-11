"""
Rup 147 (Torres+2018) MCMC run
"""

"""
Rup 147 (Torres+2018) MCMC run
"""

import os
import sys
sys.path.append('/gscratch/astro/jbirky/projects/kicq')
sys.path.append('/astro/users/jbirky/packages/vplanet')
#from kicq import pool
from multiprocessing import Pool
#from schwimmbad import MPIPool
from kicq import mcmcUtils
import config

PATH = "infile"

# Open a pool, and let it rip!
#with pool.Pool(pool='MPIPool') as pp:
with Pool() as pp:

    kwargs = config.kwargs
    kwargs["nsteps"]   = int(1e4)
    kwargs["nwalk"]    = 45
    kwargs["pool"]     = pp
    kwargs["restart"]  = True 
    kwargs["PATH"]     = PATH
    kwargs["backend"]  = "results/Rup147_emcee_node1.h5"
    kwargs["progress"] = True

    # Check for output dir, make it if it doesn't already exist
    if not os.path.exists(os.path.join(PATH, "output")):
        os.makedirs(os.path.join(PATH, "output"))

    # Run
    mcmcUtils.RunMCMC(**kwargs)

# Done!
