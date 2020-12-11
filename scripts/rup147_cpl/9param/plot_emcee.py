import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)

labels = [r"$M_1$", r"$M_2$", r'$P_{\rm rot1}$', r'$P_{\rm rot2}$', r'$\tau_1$', r'$\tau_2$', r'$P_{\rm orb}$', r'$e$', r'$\rm age$']

burn = 500
thin = 100
range = None

r1 = emcee.backends.HDFBackend('results/Rup147_emcee_node1.h5', read_only=True)
r2 = emcee.backends.HDFBackend('results/Rup147_emcee_node2.h5', read_only=True)

chain1 = r1.get_chain(flat=True, discard=burn, thin=thin)
chain2 = r2.get_chain(flat=True, discard=burn, thin=thin)
# chain1 = arr1.reshape(-1, arr1.shape[-1])
# chain2 = arr2.reshape(-1, arr2.shape[-1])
print(chain1.shape)
print(chain2.shape)

chain = np.concatenate((chain1, chain2), axis=0)

fig = corner.corner(chain, labels=labels, range=range, plot_contours=True,
                    plot_density=True, plot_points=True,
                    quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, title_kwargs={"fontsize": 18}, 
                    label_kwargs={"fontsize": 22})

fig.savefig('plots/emcee_hyak.png')
