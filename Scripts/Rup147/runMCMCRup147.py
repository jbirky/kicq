#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rup 147 (Torres+2018) MCMC run
"""

import os
# from kicq import pool
from multiprocessing import Pool
from kicq import rup147, mcmcUtils

os.nice(10)

# Define run parameters
ndim = 9
nwalk = 90
nsteps = 1000
# nsamples = 0
restart = False
backend = "Rup147_emcee.h5"
pool = Pool()

# # Open a pool, and let it rip!
# with pool.Pool(pool='SerialPool') as pool:

# Options
kwargs = rup147.kwargsRUP147
kwargs["nsteps"] = nsteps
# kwargs["nsamples"] = nsamples
kwargs["nwalk"] = nwalk
kwargs["pool"] = pool
kwargs["restart"] = restart
kwargs["LnPrior"] = rup147.LnPriorRUP147
kwargs["PriorSample"] = rup147.samplePriorRUP147
PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH
kwargs["backend"] = backend
kwargs["progress"] = True

# Check for output dir, make it if it doesn't already exist
if not os.path.exists(os.path.join(PATH, "output")):
    os.makedirs(os.path.join(PATH, "output"))

# Run
mcmcUtils.RunMCMC(**kwargs)

# Done!
