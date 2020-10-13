#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np

# Load results
res = np.load("../../Data/minSoln.npz")

soln = np.array(res["soln"])
lnlike = np.array(res["lnlike"])

mask = np.isfinite(lnlike)

ind = np.nanargmin(lnlike[mask])

print(soln[mask][ind])

# Done!
