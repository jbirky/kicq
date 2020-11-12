from approxposterior import approx
import emcee, corner
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import numpy as np
import george
import os
import config 
from kicq import priors
from kicq import mcmcUtils as kicmc
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

# os.nice(10)


# ===============================================
# Load samples
# ===============================================

runName = 'emceeRun'

# Load vplanet sims
sims = np.load('test0/apRunAPFModelCache.npz')
theta = sims['theta']
y = sims['y']
print(theta.shape)

# Load optimized GP parameters
gpPar = np.load('test0/apRunAPGP.npz')['gpParamValues'][-1]

# emcee.EnsembleSampler parameters
samplerKwargs = {"nwalkers" : 90,
                 "pool" : Pool()}

# emcee.EnsembleSampler.run_mcmc parameters
mcmcKwargs = {"iterations" : int(5.0e4), 
              "progress" : True}

# ===============================================
# Load config parameters
# ===============================================

kwargs = config.kwargs              # All the Rup 147 system constraints
bounds = config.bounds              # Prior bounds

PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH


# ===============================================
# Load vplanet input files, save them as strings
# ===============================================

with open(os.path.join(PATH, "primary.in"), 'r') as f:
    primary_in = f.read()
    kwargs["PRIMARYIN"] = primary_in
with open(os.path.join(PATH, "secondary.in"), 'r') as f:
    secondary_in = f.read()
    kwargs["SECONDARYIN"] = secondary_in
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in


# ===============================================
# Initialize approxposterior
# ===============================================

gp = approx.gpUtils.defaultGP(theta, y, white_noise=-15, fitAmp=False)
gp.set_parameter_vector(gpPar)

ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=kwargs["LnPrior"],
                            lnlike=kicmc.LnLike,
                            priorSample=kwargs["PriorSample"],
                            algorithm="bape",
                            bounds=bounds)

# # t0 = [1.08, 1.06, 8.2, 5.4, 2., 2., 8., .27, 2.5]
# # sol = ap.findMAP(theta0=t0)

labels = [r'$M_1$', r'$M_2$', r'$P_{rot1}$', r'$P_{rot2}$', r'$\tau_1$', r'$\tau_2$', r'$P_{orb}$', r'$e$', r'$age$']

# ===============================================
# MCMC with emcee
# ===============================================

# ap.runMCMC(samplerKwargs=samplerKwargs, mcmcKwargs=mcmcKwargs, runName=runName, cache=True, \
# 	estBurnin=True, thinChains=True, verbose=False)


# burn = int(1e3)
# thin = int(1e2)

# reader = emcee.backends.HDFBackend(runName + '.h5')
# chain = reader.get_chain(flat=True, discard=burn, thin=thin)
# # chain = reader.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# fig = corner.corner(chain, labels=labels, plot_contours=True,
# 			plot_density=True, plot_points=True, quantiles=[0.16, 0.5, 0.84], 
# 			show_titles=True, title_kwargs={"fontsize": 15})

# fig.savefig(runName + '.png')


# ===============================================
# MCMC with dynesty 
# ===============================================

def prior_transform(u, bounds=bounds):

    pt = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        pt[i] = (b[1] - b[0]) * u[i] + b[0]

    return pt


def lnlike(theta):

	return ap.gp.log_likelihood(y, theta)


ndim = theta.shape[1]

sampler = NestedSampler(lnlike, prior_transform, ndim, nlive=1000)
sampler.run_nested()
res = sampler.results

print(res.summary())

# fig, axes = dyplot.runplot(res)
# fig.savefig('dynestyRun.png')

# fig, axes = dyplot.traceplot(res, show_titles=True, trace_cmap='plasma', quantiles=None)
# fig.savefig('dynestyTrace.png')

# samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
# np.save('dynesty_samples.npy', samples)
# np.save('dynesty_weights.npy', weights)

fig, ax = dyplot.cornerplot(res, color='black', plot_contours=True,
							plot_density=True, plot_points=True,
							quantiles=[0.16, 0.5, 0.84], 
							show_titles=True, title_kwargs={"fontsize": 18}, 
							label_kwargs={"fontsize": 22})
fig.savefig('dynestyCorner.png')