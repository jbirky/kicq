import numpy as np
import h5py
import emcee
import corner
import warnings
import time
import tqdm
import random
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
# Load simulation samples
# ============================================


def loadSamples(nTrain=500, simsFile='apRunAPFModelCache.npz'):

	sims = np.load(simsFile)
	nSamp = len(sims['y'])

	train_ind = np.random.choice(nSamp, nTrain, replace=False)
	test_ind  = np.array(list(set(np.arange(nSamp)) - set(train_ind)))

	tTrain = sims['theta'][train_ind]
	yTrain = sims['y'][train_ind]

	tTest = sims['theta'][test_ind]
	yTest = sims['y'][test_ind]

	return tTrain, yTrain, tTest, yTest


# ============================================
# Compute train and test errors
# for varying GP scale lengths
# ============================================

fitAmp = False
nErrSamples = 100
nTrain = 2000
nSamp  = 3000
testScales = np.arange(-11,16,.5)
computeSamples = True
simsFile = 'simsCache.npz'
gpSaved = np.load('test0/apRunAPGP.npz')['gpParamValues'][-1]


if computeSamples == True:

	trainErrSamples = []
	testErrSamples = []

	trainErrOpt = []
	testErrOpt = []

	for j in range(nErrSamples):

		tTrain, yTrain, tTest, yTest = loadSamples(nTrain=nTrain, simsFile=simsFile)

		gp = gpUtils.defaultGP(tTrain, yTrain, white_noise=-12, fitAmp=fitAmp)
		gp.compute(tTrain)

		gpPar = np.zeros(gp.get_parameter_vector().shape[0])

		trainErr = []
		testErr  = []

		for scale in tqdm.tqdm(testScales):
			if fitAmp == True:
				gpPar[0] = np.mean(yTrain)
				gpPar[1] = 10 #np.var(yTrain)
				gpPar[2:] = scale
			else:
				gpPar[0] = np.mean(yTrain)
				gpPar[1:] = scale

			gp.set_parameter_vector(gpPar)
			gp.recompute()

			yTrainPred = gp.predict(yTrain, tTrain)[0]
			mseTrain = mse(yTrain, yTrainPred)
			trainErr.append(mseTrain)

			yTestPred = gp.predict(yTrain, tTest)[0]
			mseTest = mse(yTest, yTestPred)
			testErr.append(mseTest)

		trainErrSamples.append(trainErr)
		testErrSamples.append(testErr)

		# Compute train/test error for optimized gp scale parameters
		if gpSaved is not None:

			gp.set_parameter_vector(gpSaved)
			gp.recompute()

			yTrainOpt = gp.predict(yTrain, tTrain)[0]
			mseTrainOpt = mse(yTrain, yTrainOpt)
			trainErrOpt.append(mseTrainOpt)

			yTestOpt = gp.predict(yTest, tTest)[0]
			mseTestOpt = mse(yTest, yTestOpt)
			testErrOpt.append(mseTestOpt)

	np.savez('mse_samples.npz', testScales=testScales, trainErrSamples=trainErrSamples, \
		testErrSamples=testErrSamples, trainErrOpt=trainErrOpt, testErrOpt=testErrOpt)


# ============================================
# Load results
# ============================================

tTrain, yTrain, tTest, yTest = loadSamples(nTrain=nTrain)
nSamp = len(yTrain) + len(yTest)

gp = gpUtils.defaultGP(tTrain, yTrain, white_noise=-12, fitAmp=fitAmp)
gp.compute(tTrain)

gp.set_parameter_vector(gpSaved)
gp.recompute()

yTrainOpt = gp.predict(yTrain, tTrain)[0]
mseTrainOpt = mse(yTrain, yTrainOpt)

yTestOpt = gp.predict(yTrain, tTest)[0]
mseTestOpt = mse(yTest, yTestOpt)


mse_samples = np.load('mse_samples.npz')
testScales = mse_samples['testScales']
trainErrSamples = mse_samples['trainErrSamples']
testErrSamples = mse_samples['testErrSamples']


# ============================================
# Plot results
# ============================================

fig, axs = plt.subplots(2, 1, figsize=[10,12], sharex=True)

for j in range(nErrSamples):
	axs[0].plot(testScales, testErrSamples[j], color='g', linewidth=1, alpha=.1)
	axs[1].plot(testScales, trainErrSamples[j], color='b', linewidth=1, alpha=.1)
axs[0].plot(testScales, np.mean(testErrSamples, axis=0), label='Validation error (N=%s)'%(nSamp - nTrain), color='g', linewidth=2)
axs[1].plot(testScales, np.mean(trainErrSamples, axis=0), label='Trainining error (N=%s)'%(nTrain), color='b', linewidth=2)

if gpSaved is not None:
	# cmap = mpl.cm.Spectral
	# nscales = len(gpSaved) - 1
	# colors = [cmap(i/nscales) for i in range(nscales)]
	# random.shuffle(colors)

	for i in range(nscales):
		# plt.axvline(gpSaved[i], color=colors[i], linewidth=1)
		axs[0].axvline(gpSaved[i], color='r', linewidth=.5, linestyle='--')
		axs[1].axvline(gpSaved[i], color='r', linewidth=.5, linestyle='--')

	axs[0].axhline(mseTestOpt, color='r', linestyle='--', label='Opt GP scale validation error')
	axs[1].axhline(mseTrainOpt, color='r', linestyle='--', label='Opt GP scale training error')

axs[0].axhline(np.var(yTest), color='k', linestyle='--', label='var(yTest)')
axs[1].axhline(np.var(yTrain), color='k', linestyle='--', label='var(yTrain)')

axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].set_ylabel(r'MSE', fontsize=20)
axs[1].set_ylabel(r'MSE', fontsize=20)
plt.xlabel(r'GP log scale parameter', fontsize=20)
plt.xlim(max(testScales), min(testScales))
plt.minorticks_on()
plt.tight_layout()
if fitAmp == True:
	plt.savefig('plots/gp_bias_variance_amp.png')
else:
	plt.savefig('plots/gp_bias_variance_noamp.png')
plt.close()