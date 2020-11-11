import numpy as np
import vplot as vpl
import os
import re
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


# ===================================================
# Load sim files
# ===================================================

# Use infiles from the following INPATH
INPATH = '../../Scripts/Rup147_CTL'
# INPATH = '../../Scripts/Rup147_CPL'
OUTPATH = 'sims'

plotName = 'plots/evolution_ctl.png'
sysName = 'Rup147'

import sys
sys.path.append(INPATH)
import config
from config import kwargs
from config import m1, m1Sig, m2, m2Sig, \
				   r1, r1Sig, r2, r2Sig, \
				   teff1, teff1Sig, teff2, teff2Sig, \
				   lum1, lum1Sig, lum2, lum2Sig, \
				   prot1, prot1Sig, porb, porbSig, \
				   ecc, eccSig, age, ageSig

# ===================================================

kwargs['PATH'] = INPATH
kwargs['OUTPATH'] = OUTPATH

# remove all old files from OUTPATH
files = glob.glob(os.path.join(OUTPATH, '*'))
for f in files:
    os.remove(f)

with open(os.path.join(INPATH, "primary.in"), 'r') as f:
    primary_in = f.read()
    kwargs["PRIMARYIN"] = primary_in
with open(os.path.join(INPATH, "secondary.in"), 'r') as f:
    secondary_in = f.read()
    kwargs["SECONDARYIN"] = secondary_in
with open(os.path.join(INPATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    # save values for full evolution
    vpl_in = re.sub("%s(.*?)#" % "dOutputTime", "%s %.6e #" % ("dOutputTime", 1e6), vpl_in)
    kwargs["VPLIN"] = vpl_in

# ===================================================
# Run simulation
# ===================================================

theta = np.array([ 1.08,  1.07,  6.2,  6.2, -0.5, -0.5,  6.8,  0.2,  2.45])

kwargs['remove'] = False
kwargs['dOutputTime'] = 1e6
lnP = config.LnProb(theta, **kwargs)

# Time -Radius Temperature -Luminosity -RotPer -LostAngMom -SurfEnFluxTotal
pri = np.genfromtxt(glob.glob(os.path.join(OUTPATH, "*.primary.forward"))[0])

# Time -Radius Temperature -Luminosity -TotEn -TotAngMom -Semim -RotPer Ecce -OrbPer -SurfEnFluxTotal
sec = np.genfromtxt(glob.glob(os.path.join(OUTPATH, "*.secondary.forward"))[0])

output = vpl.GetOutput(OUTPATH)

# ===================================================

YRSEC = 3600 * 24 * 365
pri_tlock = float(output.log.final.primary.LockTime) / YRSEC
sec_tlock = float(output.log.final.secondary.LockTime) / YRSEC

tpms1 = 6e7 * (m1)**-2.5
tpms2 = 6e7 * (m2)**-2.5

pms_ind1 = np.argmin(np.abs(pri.T[0] - tpms1))
pms_ind2 = np.argmin(np.abs(sec.T[0] - tpms2))


summary =  "Input parameters: \n" 
summary += r"M$_1$ = %.3f M$_{\odot}$"%(theta[0])
summary += "\n"
summary += r"M$_2$ = %.3f M$_{\odot}$"%(theta[1])
summary += "\n"
summary += r"P$_{\rm rot1}$ = %.3f d"%(theta[2])
summary += "\n"
summary += r"P$_{\rm rot2}$ = %.3f d"%(theta[3])
summary += "\n"
summary += r"$\log\tau_1$ = %.3f"%(theta[4])
summary += "\n"
summary += r"$\log\tau_2$ = %.3f"%(theta[5])
summary += "\n"
summary += r"P$_{\rm orb}$ = %.3f d"%(theta[6])
summary += "\n"
summary += "e = %.3f"%(theta[7])
summary += "\n"
summary += "age = %.3f Gyr"%(theta[8])

output  = "Output: \n"
output += r"$\ln$P = %.3e"%(lnP)


# ===================================================
# Plot
# ===================================================

lbls = [[r'$P_{\rm rot}$ [d]', r'Radius [$R_{\odot}$]'], 
        [r'$P_{\rm orb}$ [d]', r'$T_{\rm eff}$ [K]'], 
        [r'$e$', r'$L$ [$L_{\odot}$]']]
nplots = len(lbls)

lw = 1.5 

fig, axs = plt.subplots(3, 2, figsize=[18, 15], sharex=True)
plt.subplots_adjust(hspace=0.1, right=.8)

for i in range(3):
    for j in range(2):
        axs[i][j].set_ylabel(lbls[i][j], fontsize=20)
        axs[i][j].axvline(pri_tlock, color='b', linestyle='--')
        axs[i][j].axvline(sec_tlock, color='r', linestyle='--')
        
# Prot
axs[0][0].plot(pri.T[0], pri.T[4], color='b', linewidth=lw, label='Pri')
axs[0][0].plot(sec.T[0], sec.T[7], color='r', linewidth=lw, label='Sec')
axs[0][0].axhline(prot1, color='m', linestyle='--')
axs[0][0].fill_between(pri.T[0], y1=prot1-prot1Sig, y2=prot1+prot1Sig, color='m', alpha=.05)
axs[0][0].scatter(pri.T[0][pms_ind1], pri.T[4][pms_ind1], color='b')
axs[0][0].scatter(sec.T[0][pms_ind2], sec.T[7][pms_ind2], color='r')

# Radius
axs[0][1].plot(pri.T[0], pri.T[1], linewidth=lw, color='b')
axs[0][1].plot(sec.T[0], sec.T[1], linewidth=lw, color='r')
axs[0][1].axhline(r1, color='b', linestyle='--')
axs[0][1].axhline(r2, color='r', linestyle='--')
axs[0][1].fill_between(pri.T[0], y1=r1-r1Sig, y2=r1+r1Sig, color='b', alpha=.05)
axs[0][1].fill_between(pri.T[0], y1=r2-r2Sig, y2=r2+r2Sig, color='r', alpha=.05)
axs[0][1].scatter(pri.T[0][pms_ind1], pri.T[1][pms_ind1], color='b')
axs[0][1].scatter(sec.T[0][pms_ind2], sec.T[1][pms_ind2], color='r')

# Porb
axs[1][0].plot(sec.T[0], sec.T[9], linewidth=lw, color='m')
axs[1][0].axhline(porb, color='m', linestyle='--')
axs[1][0].fill_between(pri.T[0], y1=porb-porbSig, y2=porb+porbSig, color='m', alpha=.05)
axs[1][0].scatter(sec.T[0][pms_ind2], sec.T[9][pms_ind2], color='m')

# Teff
axs[1][1].plot(pri.T[0], pri.T[2], linewidth=lw, color='b')
axs[1][1].plot(sec.T[0], sec.T[2], linewidth=lw, color='r')
axs[1][1].axhline(teff1, color='b', linestyle='--')
axs[1][1].axhline(teff2, color='r', linestyle='--')
axs[1][1].fill_between(pri.T[0], y1=teff1-teff1Sig, y2=teff1+teff1Sig, color='b', alpha=.05)
axs[1][1].fill_between(pri.T[0], y1=teff2-teff2Sig, y2=teff2+teff2Sig, color='r', alpha=.05)
axs[1][1].scatter(pri.T[0][pms_ind1], pri.T[2][pms_ind1], color='b')
axs[1][1].scatter(sec.T[0][pms_ind2], sec.T[2][pms_ind2], color='r')

# ecc
axs[2][0].plot(sec.T[0], sec.T[8], linewidth=lw, color='m')
axs[2][0].axhline(ecc, color='m', linestyle='--')
axs[2][0].fill_between(pri.T[0], y1=ecc-eccSig, y2=ecc+eccSig, color='m', alpha=.05)
axs[2][0].scatter(sec.T[0][pms_ind2], sec.T[8][pms_ind2], color='m')

# Luminosity
axs[2][1].plot(pri.T[0], pri.T[3], linewidth=lw, color='b')
axs[2][1].plot(sec.T[0], sec.T[3], linewidth=lw, color='r')
axs[2][1].axhline(lum1, color='b', linestyle='--')
axs[2][1].axhline(lum2, color='r', linestyle='--')
axs[2][1].fill_between(pri.T[0], y1=lum1-lum1Sig, y2=lum1+lum1Sig, color='b', alpha=.05)
axs[2][1].fill_between(pri.T[0], y1=lum2-lum2Sig, y2=lum2+lum2Sig, color='r', alpha=.05)
axs[2][1].scatter(pri.T[0][pms_ind1], pri.T[3][pms_ind1], color='b')
axs[2][1].scatter(sec.T[0][pms_ind2], sec.T[3][pms_ind2], color='r')

plt.gcf().text(.82, .9, summary, fontsize=22, va='top')
plt.gcf().text(.82, .6, output, fontsize=22, va='top')

axs[2][0].set_xlabel('Time [yr]', fontsize=20)
axs[2][1].set_xlabel('Time [yr]', fontsize=20)
axs[0][0].legend(loc='upper left')
plt.suptitle("{} {} Model".format(sysName, kwargs['MODEL']), fontsize=25)
plt.xscale('log')
plt.xlim(1e6, max(sec.T[0]))
plt.minorticks_on()
# plt.tight_layout()
plt.savefig(plotName)
plt.show()