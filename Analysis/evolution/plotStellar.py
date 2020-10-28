import numpy as np
import vplot as vpl
import os
import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import sys
# sys.path.append('../../Scripts/Rup147_CTL')
sys.path.append('../../Scripts/Rup147_CPL')
import config
from config import kwargs
from config import m1, m1Sig, m2, m2Sig, \
				   r1, r1Sig, r2, r2Sig, \
				   teff1, teff1Sig, teff2, teff2Sig, \
				   lum1, lum1Sig, lum2, lum2Sig, \
				   prot1, prot1Sig, porb, porbSig, \
				   ecc, eccSig, age, ageSig

plotName = 'primary_stellar_evolution.png'
sysName = 'Rup 147'

# ===================================================
# Load sim files
# ===================================================

PATH = os.path.dirname(os.path.abspath(__file__))
OUTPATH = 'stellar'

# Time -Radius Temperature -Luminosity -RotPer -LostAngMom
pri_file = glob.glob(os.path.join(OUTPATH, "*.primary.forward"))[0]
print('loading', pri_file)
pri = np.genfromtxt(pri_file)

output = vpl.GetOutput(OUTPATH)


# ===================================================
# Plot
# ===================================================

lbls = [r'Radius [$R_{\odot}$]', 
        r'$T_{\rm eff}$ [K]', 
        r'$L$ [$L_{\odot}$]',
        r'$P_{\rm rot}$']
nplots = len(lbls)

lw = 1.5 

fig, axs = plt.subplots(4, 1, figsize=[9, 20], sharex=True)
plt.subplots_adjust(hspace=0.1, right=.8)

for i in range(4):
    axs[i].set_ylabel(lbls[i], fontsize=20)

# Radius
axs[0].plot(pri.T[0], pri.T[1], linewidth=lw, color='b')

# Teff
axs[1].plot(pri.T[0], pri.T[2], linewidth=lw, color='b')

# Luminosity
axs[2].plot(pri.T[0], pri.T[3], linewidth=lw, color='b')

# Prot
axs[3].plot(pri.T[0], pri.T[4], linewidth=lw, color='b')

# observed values
# axs[0].axhline(r1, color='b', linestyle='--')
# axs[0].fill_between(pri.T[0], y1=r1-r1Sig, y2=r1+r1Sig, color='b', alpha=.05)
# axs[0].fill_between(pri.T[0], y1=r2-r2Sig, y2=r2+r2Sig, color='r', alpha=.05)

# axs[1].axhline(teff1, color='b', linestyle='--')
# axs[1].fill_between(pri.T[0], y1=teff1-teff1Sig, y2=teff1+teff1Sig, color='b', alpha=.05)
# axs[1].fill_between(pri.T[0], y1=teff2-teff2Sig, y2=teff2+teff2Sig, color='r', alpha=.05)

# axs[2].axhline(lum1, color='b', linestyle='--')
# axs[2].fill_between(pri.T[0], y1=lum1-lum1Sig, y2=lum1+lum1Sig, color='b', alpha=.05)
# axs[2].fill_between(pri.T[0], y1=lum2-lum2Sig, y2=lum2+lum2Sig, color='r', alpha=.05)

axs[-1].set_xlabel('Time [yr]', fontsize=20)
# axs[0].legend(loc='upper left')
# plt.suptitle(f"{sysName} {kwargs['MODEL']} Model", fontsize=25)
plt.xscale('log')
plt.xlim(1e6, max(pri.T[0]))
plt.minorticks_on()
# plt.tight_layout()
plt.savefig(plotName)
plt.show()