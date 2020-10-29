import numpy as np
import h5py
import emcee
import corner
import warnings
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
from config import kwargs, bounds
from kicq import mcmcUtils as kicmc


# ============================================
# Load simulation samples
# ============================================

sims = np.load('test0/apRunAPFModelCache.npz')

nTrain = 600
tTrain = sims['theta'][0:nTrain]
yTrain = sims['y'][0:nTrain]

tTest = sims['theta'][nTrain:]
yTest = sims['y'][nTrain:]


# ============================================
# Initialize GP
# ============================================

fitAmp = True

gp = gpUtils.defaultGP(tTrain, yTrain, white_noise=-12, fitAmp=fitAmp)
gp.compute(tTrain)

gpPar = np.zeros(gp.get_parameter_vector().shape[0])
testScales = np.arange(-4,4,.5)

trainErr = []
testErr  = []

for scale in testScales:
	if fitAmp == True:
	    gpPar[0] = np.mean(yTrain)
	    gpPar[1] = np.var(yTrain)
	    gpPar[2:] = scale
	else:
		gpPar[0] = np.mean(yTrain)
	    gpPar[1:] = scale

    gp.set_parameter_vector(gpPar)

    yTrainPred = gp.predict(yTrain, tTrain)[0]
    mseTrain = np.sum((yTrain - yTrainPred)**2) / len(yTrain)
    trainErr.append(mseTrain)

    yTestPred = gp.predict(yTrain, tTest)[0]
    mseTest = np.sum((yTest - yTestPred)**2) / len(yTest)
    testErr.append(mseTest)


# ============================================
# Plot results
# ============================================

plt.plot(testScales, trainErr, label='Trainining error', color='b')
plt.plot(testScales, testErr, label='Test error', color='g')
plt.legend(loc='upper right')
plt.xlabel('log scale')
plt.ylabel('MSE')
plt.yscale('log')
plt.xlim(max(testScales), min(testScales))
if fitAmp == True:
	plt.savefig('gp_bias_variance_amp.png')
else:
	plt.savefig('gp_bias_variance_noamp.png')
plt.close()