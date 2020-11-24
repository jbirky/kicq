"""
Rup 147 (Torres+2018) MCMC run
"""

"""
Rup 147 (Torres+2018) MCMC run
"""

import os
from kicq import pool
from kicq import mcmcUtils
import config


# Open a pool, and let it rip!
with pool.Pool(pool='MPIPool') as pp:

    kwargs = config.kwargs
    kwargs["nsteps"]   = int(1e4)
    kwargs["nwalk"]    = 90
    kwargs["pool"]     = pp
    kwargs["restart"]  = False
    kwargs["PATH"]     = PATH
    kwargs["backend"]  = "results/Rup147_emcee.h5"
    kwargs["progress"] = True

    # Check for output dir, make it if it doesn't already exist
    if not os.path.exists(os.path.join(PATH, "output")):
        os.makedirs(os.path.join(PATH, "output"))

    # Run
    mcmcUtils.RunMCMC(**kwargs)

# Done!
