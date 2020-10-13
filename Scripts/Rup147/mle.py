#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
from datetime import datetime
from kicq import ad3814, mcmcUtils as kicmc


nRestarts = 2

# Prior bounds
bounds = ((0.35, 0.45),
          (0.15, 0.25),
          (0.25, 15),
          (0.25, 15),
          (-3, 2),
          (-3, 2),
          (6.015717, 11),
          (0, 0.6),
          (0.625, 0.675))

# Loglikelihood function setup
# All the AD 3814 system constraints
kwargs = ad3814.kwargsAD3814
kwargs["PATH"] = os.path.dirname(os.path.abspath(__file__))

# Get the input files, save them as strings
with open(os.path.join(kwargs["PATH"], "primary.in"), 'r') as f:
    primary_in = f.read()
    kwargs["PRIMARYIN"] = primary_in
with open(os.path.join(kwargs["PATH"], "secondary.in"), 'r') as f:
    secondary_in = f.read()
    kwargs["SECONDARYIN"] = secondary_in
with open(os.path.join(kwargs["PATH"], "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in

# Optimize!

# Maximize the lnlike by minimizing the -lnlike
def fn(x):
    return -(kicmc.LnLike(x, **kwargs)[0] + ad3814.LnPriorAD3814(x, **kwargs))

def minimizeFn(x):
    seed = int(os.getpid()*datetime.now().microsecond/100)
    np.random.seed(seed)
    p0 = ad3814.samplePriorAD3814(1)
    print("Initial state:", p0)
    res = minimize(fn, p0, bounds=bounds,
                   method="l-bfgs-b")

    # Save initial condition in results dictionary
    res["init"] = p0
    return res
# end function


# Create a pool to distribution runs with as many cores as possible
pool = Pool()

init = list()
soln = list()
lnlike = list()

# Run them!
results = [pool.apply_async(minimizeFn, args=(ad3814.samplePriorAD3814(1),)) for x in range(nRestarts)]
output = [p.get() for p in results]

for ii, out in enumerate(output):
    init.append(out["init"])
    soln.append(out["x"])
    lnlike.append(out["fun"])
    print("Run %d init, soln, lnlike:", out["init"], out["x"], out["fun"])

bestInd = np.argmax(lnlike)

print("Best:")
print(init[bestInd])
print(soln[bestInd])
print(lnlike[bestInd])

# Cache results
np.savez("minSoln.npz", soln=soln, lnlike=lnlike, init=init)

# Done!
