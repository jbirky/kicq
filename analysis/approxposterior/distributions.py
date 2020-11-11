import numpy as np
import h5py
import emcee
import corner
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
import matplotlib as mpl
import time
import os
import glob


# =============================================
# Plot setup
# =============================================

INPATH  = '../../Scripts/Rup147_CTL/results'
plotDir = '../../Scripts/Rup147_CTL/plots'
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

labels = [r"$M_1$", r"$M_2$", r'$P_{\rm rot1}$', r'$P_{\rm rot2}$', r'$\tau_1$', r'$\tau_2$', r'$P_{\rm orb}$', r'$e$', r'$\rm age$']
m0 = 1000

# =============================================
# MCMC corner plot
# =============================================

def plotPosterior(file, **kwargs):
    reader = emcee.backends.HDFBackend(file)
    burn = kwargs.get('burn', None)
    thin = kwargs.get('thin', None)
    
    chain = reader.get_chain(flat=True, discard=burn, thin=thin)
    print(chain.shape)

    range = kwargs.get('range', None)

    warnings.simplefilter("ignore")
    fig = corner.corner(chain, labels=labels, range=range, plot_contours=True,
                        plot_density=True, plot_points=True,
                        quantiles=[0.16, 0.5, 0.84], 
                        show_titles=True, title_kwargs={"fontsize": 18}, 
                        label_kwargs={"fontsize": 22})
    return chain, fig
    
# ----------------------------

# get hdf5 file of last MCMC iteration
hdFiles = glob.glob(os.path.join(INPATH, 'apRun*.h5'))
lastF = max([f.split('apRun')[1].split('.h5')[0] for f in hdFiles])

chain, fig = plotPosterior(os.path.join(INPATH, 'apRun%s.h5'%(lastF)), burn=int(1e3), thin=100)
fig.savefig(os.path.join(plotDir, 'emcee_corner.png'))


# =============================================
# lnP vs approxposterior iteration
# =============================================

sims = np.load(os.path.join(INPATH, 'apRunAPFModelCache.npz'))
y = -sims['y']

plt.figure(figsize=[10,8])
itr = np.arange(len(y))
plt.scatter(itr, y)
plt.yscale('log')
plt.xlabel('iteration', fontsize=20)
plt.ylabel(r'$-\ln P$', fontsize=20)
plt.savefig(os.path.join(plotDir, 'bape_lnp_iter.png'))
plt.show()


# =============================================
# Dist of training samples - density plot
# =============================================

fig = corner.corner(sims['theta'][0:m0], labels=labels, plot_contours=True,
                    plot_density=True, plot_points=True,
                    quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, title_kwargs={"fontsize": 18}, 
                    label_kwargs={"fontsize": 22})
fig.savefig(os.path.join(plotDir, 'init_samples.png'))


fig = corner.corner(sims['theta'][m0:], labels=labels, plot_contours=True,
                    plot_density=True, plot_points=True,
                    quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, title_kwargs={"fontsize": 18}, 
                    label_kwargs={"fontsize": 22},
                    truths=sims['theta'][np.argmax(sims['y'])])
fig.savefig(os.path.join(plotDir, 'bape_samples.png'))


# =============================================
# Dist of training samples - colored by lnP value
# =============================================

def plotCornerLnp(tt, yy):

    fig = corner.corner(tt, c=yy, labels=labels, 
                  plot_datapoints=False, plot_density=False, plot_contours=False,
                  show_titles=True, title_kwargs={"fontsize": 18}, 
                  label_kwargs={"fontsize": 22})

    ndim = tt.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            im = ax.scatter(tt.T[xi], tt.T[yi], c=yy, s=2, cmap='coolwarm', norm=colors.LogNorm(vmin=yy.min(), vmax=yy.max()))

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=.98, pad=.1)
    cb.set_label(r'$-\ln P$', fontsize=20)
    cb.ax.tick_params(labelsize=18)
    
    return fig

# ----------------------------

fig = plotCornerLnp(sims['theta'], -sims['y'])
fig.savefig(os.path.join(plotDir, 'all_samples_lnp.png'))

fig = plotCornerLnp(sims['theta'][m0:], -sims['y'][m0:])
fig.savefig(os.path.join(plotDir, 'bape_samples_lnp.png'))

fig = plotCornerLnp(sims['theta'][:m0], -sims['y'][:m0])
fig.savefig(os.path.join(plotDir, 'init_samples_lnp.png'))