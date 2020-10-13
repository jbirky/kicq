#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rup 147 (Torres+2018) MCMC run
"""

import os
import tqdm
from multiprocessing import Pool
from kicq import rup147, mcmcUtils as kicmc
from approxposterior import approx
import emcee, corner
import numpy as np
import george
from functools import partial
import time

os.nice(10)

# Define algorithm parameters
ndim = 9                          # Dimensionality of the problem
m0 = 1000                         # Initial size of training set
m = 500                           # Number of new points to find each iteration
nmax = 20                         # Maximum number of iterations
kmax = 5                          # Number of consecutive iterations for convergence check to pass before successfully ending algorithm
nGPRestarts = 1                   # Number of times to restart GP hyperparameter optimization
nMinObjRestarts = 5               # Number of times to restart objective fn minimization
optGPEveryN = 10                  # Optimize GP hyperparameters even this many iterations

bounds = rup147.bounds
algorithm = "bape"               # Use the Kandasamy et al. (2015) formalism
trainSimCache = "apRunAPFModelCache.npz"

# for MCMC parallelization
ncpu = os.cpu_count()
pool = Pool()

# emcee.EnsembleSampler parameters
samplerKwargs = {"nwalkers" : 90}

# emcee.EnsembleSampler.run_mcmc parameters
mcmcKwargs = {"iterations" : int(2.0e4),
              "pool" : pool, 
              "progress" : True}
# gmmKwargs = {"reg_covar" : 1.0e-4}

# Loglikelihood function setup
kwargs = rup147.kwargsRUP147            # All the Rup 147 system constraints
PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH

# Get the input files, save them as strings
with open(os.path.join(PATH, "primary.in"), 'r') as f:
    primary_in = f.read()
    kwargs["PRIMARYIN"] = primary_in
with open(os.path.join(PATH, "secondary.in"), 'r') as f:
    secondary_in = f.read()
    kwargs["SECONDARYIN"] = secondary_in
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in

# =====================
# Generate initial GP training samples
# =====================

if not os.path.exists(trainSimCache):
    y = np.zeros(m0)
    theta = np.zeros((m0, ndim))

    t0 = time.time()
    for ii in tqdm.tqdm(range(m0)):
        theta[ii,:] = rup147.samplePriorRUP147()
        # y[ii] = kicmc.LnLike(theta[ii], **kwargs)[0] + rup147.LnPriorRUP147(theta[ii], **kwargs)

    lnp = partial(rup147.LnProbRUP147, **kwargs)

    with Pool(ncpu) as pp:
      y = pp.map(lnp, theta)

      # for result in tqdm(pp.imap(func=lnp, iterable=theta), total=len(theta)):
      #     y.append(result)

    np.savez(trainSimCache, theta=theta, y=y)
    print('Finished running {} vplanet sims:  {}'.format(m0, time.time() - t0))

else:
    print("Loading in cached simulations...")
    sims = np.load(trainSimCache)
    theta = sims["theta"]
    y = sims["y"]

print(theta.shape)

# =====================
# Initialize GP
# =====================

# Guess initial metric, or scale length of the covariances in loglikelihood space
initialMetric = np.nanmedian(theta**2, axis=0)/10.0

# Create kernel: We'll model coverianges in loglikelihood space using a
# Squared Expoential Kernel as we anticipate Gaussian-ish posterior
# distributions in our 2-dimensional parameter space
kernel = george.kernels.ExpSquaredKernel(initialMetric, ndim=ndim)

# Guess initial mean function
mean = np.nanmedian(y)

# Create GP and compute the kernel
gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
gp.compute(theta)

# =====================
# Run approxposterior
# =====================

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=rup147.LnPriorRUP147,
                            lnlike=kicmc.LnLike,
                            priorSample=rup147.samplePriorRUP147,
                            algorithm=algorithm,
                            bounds=bounds)

# ap.run(m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, bounds=bounds,  estBurnin=True,
#        nKLSamples=nKLSamples, mcmcKwargs=mcmcKwargs, maxComp=12, thinChains=True,
#        samplerKwargs=samplerKwargs, verbose=True, gmmKwargs=gmmKwargs, **kwargs)

ap.run(m=m, nmax=nmax, kmax=kmax, mcmcKwargs=mcmcKwargs, samplerKwargs=samplerKwargs, 
       nGPRestarts=nGPRestarts, nMinObjRestarts=nMinObjRestarts, optGPEveryN=optGPEveryN, eps=0.1,
       thinChains=True, estBurnin=True, verbose=True, cache=True, convergenceCheck=True, **kwargs)


# =====================
# Plot posterior
# =====================

# Load in chain from last iteration
reader = emcee.backends.HDFBackend(ap.backends[-1], read_only=True)
samples = reader.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    labels=[r'$M_1$', r'$M_2$', r'$Prot_1$', r'$Prot_2$', r'$\tau_1$', r'$\tau_2$', r'$P_{orb}$', r'$e$', r'$age$'],
                    scale_hist=True, plot_contours=True)

fig.savefig("apFinalPosterior.png", bbox_inches="tight") # Uncomment to save

np.save('ap_final_samples.npy', samples)
