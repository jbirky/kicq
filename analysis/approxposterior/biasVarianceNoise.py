import numpy as np
import h5py
import emcee
import corner
import warnings
import time
import tqdm
import random
import os
from sklearn.metrics import mean_squared_error as mse

from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
# from config import kwargs, bounds
from kicq import mcmcUtils as kicmc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# from matplotlib import rc
# plt.style.use('classic')
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
# rc('figure', facecolor='w')
# rc('xtick', labelsize=20)
# rc('ytick', labelsize=20)


# ============================================
# Plot setup
# ============================================

BASE = '../../scripts/rup147_ctl/4param/'
INPATH  = BASE + 'results/'
plotDir = BASE + 'plots/'
if not os.path.exists(plotDir):
    os.mkdir(plotDir)
    
fitAmp = False
nErrSamples = 10
nTrain = 300
nSamp  = 400
testNoise = np.arange(-20,0,1)
computeSamples = True

simsFile = INPATH + 'apRunAPFModelCache.npz' 
gpFile   = INPATH + 'apRunAPGP.npz'
mseFile  = INPATH + 'mse_samples_noise.npz'

# load optimized GP parameters, last iteration
gpSaved = np.load(gpFile)['gpParamValues'][-1]
# white_noise = -15


# ============================================
# Load simulation samples
# ============================================


def loadSamples(nTrain=nTrain, simsFile=simsFile):

	sims = np.load(simsFile)
	print(len(sims['y']), 'total samples')

	train_ind = np.random.choice(nSamp, nTrain, replace=False)
	test_ind  = np.array(list(set(np.arange(nSamp)) - set(train_ind)))

	tTrain = sims['theta'][train_ind]
	yTrain = sims['y'][train_ind]

	tTest = sims['theta'][test_ind]
	yTest = sims['y'][test_ind]

	return tTrain, yTrain, tTest, yTest


# ============================================
# Compute train and test errors
# for varying GP white noise values
# ============================================

if computeSamples == True:

	trainErrSamples = []
	testErrSamples = []

	trainErrOpt = []
	testErrOpt = []

	for j in range(nErrSamples):

		tTrain, yTrain, tTest, yTest = loadSamples(nTrain=nTrain, simsFile=simsFile)

		trainErr = []
		testErr  = []

		for noise in tqdm.tqdm(testNoise):
			gp = gpUtils.defaultGP(tTrain, yTrain, white_noise=noise, fitAmp=fitAmp)
			if gpSaved is not None:
				gp.set_parameter_vector(gpSaved)
			gp.compute(tTrain)
		
			# if fitAmp == True:
			# 	gpPar[0] = np.mean(yTrain)
			# 	gpPar[1] = np.var(yTrain)
			# 	gpPar[2:] = scale
			# else:
			# 	gpPar[0] = np.mean(yTrain)
			# 	gpPar[1:] = scale

			# gp.set_parameter_vector(gpPar)
			# gp.recompute()

			yTrainPred = gp.predict(yTrain, tTrain)[0]
			mseTrain = mse(yTrain, yTrainPred)
			trainErr.append(mseTrain)

			yTestPred = gp.predict(yTrain, tTest)[0]
			mseTest = mse(yTest, yTestPred)
			testErr.append(mseTest)

		trainErrSamples.append(trainErr)
		testErrSamples.append(testErr)

	np.savez(mseFile, testNoise=testNoise, trainErrSamples=trainErrSamples, \
		testErrSamples=testErrSamples)


# ============================================
# Load results
# ============================================

# tTrain, yTrain, tTest, yTest = loadSamples(nTrain=nTrain, simsFile=simsFile)

# gp = gpUtils.defaultGP(tTrain, yTrain, white_noise=white_noise, fitAmp=fitAmp)
# gp.compute(tTrain)

# gp.set_parameter_vector(gpSaved)
# gp.recompute()

# yTrainOpt = gp.predict(yTrain, tTrain)[0]
# mseTrainOpt = mse(yTrain, yTrainOpt)

# yTestOpt = gp.predict(yTrain, tTest)[0]
# mseTestOpt = mse(yTest, yTestOpt)

# print('Opt GP train error: %.2e'%(mseTrainOpt))
# print('Opt GP test error: %.2e'%(mseTestOpt))

# mse_samples = np.load(mseFile)
# testScales = mse_samples['testScales']
# trainErrSamples = mse_samples['trainErrSamples']
# testErrSamples = mse_samples['testErrSamples']


# ============================================
# Plot results
# ============================================

fig, axs = plt.subplots(2, 1, figsize=[10,12], sharex=True)

for j in range(nErrSamples):
	axs[0].plot(testNoise, testErrSamples[j], color='g', linewidth=1, alpha=.1)
	axs[1].plot(testNoise, trainErrSamples[j], color='b', linewidth=1, alpha=.1)
axs[0].plot(testNoise, np.mean(testErrSamples, axis=0), label='Test error (N=%s)'%(nSamp - nTrain), color='g', linewidth=2)
axs[1].plot(testNoise, np.mean(trainErrSamples, axis=0), label='Train error (N=%s)'%(nTrain), color='b', linewidth=2)

# if gpSaved is not None:
	# cmap = mpl.cm.Spectral
	# nscales = len(gpSaved) - 1
	# colors = [cmap(i/nscales) for i in range(nscales)]
	# random.shuffle(colors)

	# for i in range(1, len(gpSaved)):
	# 	axs[0].axvline(gpSaved[i], color='r', linewidth=.5, linestyle='--')
	# 	axs[1].axvline(gpSaved[i], color='r', linewidth=.5, linestyle='--')

	# axs[0].axhline(mseTestOpt, color='r', linestyle='--', label='Opt GP test error: %.2e'%(mseTestOpt))
	# axs[1].axhline(mseTrainOpt, color='r', linestyle='--', label='Opt GP train error: %.2e'%(mseTrainOpt))

axs[0].axhline(np.var(yTest), color='k', linestyle='--', label='var(yTest)')
axs[1].axhline(np.var(yTrain), color='k', linestyle='--', label='var(yTrain)')

axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].set_ylabel(r'MSE', fontsize=20)
axs[1].set_ylabel(r'MSE', fontsize=20)
plt.xlabel(r'GP log noise parameter', fontsize=20)
axs[0].set_title('CTL model', fontsize=25)
plt.xlim(max(testNoise), min(testNoise))
plt.minorticks_on()
plt.tight_layout()
if fitAmp == True:
	plt.savefig(os.path.join(plotDir, 'gp_bias_variance_noise_amp.png'))
else:
	plt.savefig(os.path.join(plotDir, 'gp_bias_variance_noise_noamp.png'))
plt.close()