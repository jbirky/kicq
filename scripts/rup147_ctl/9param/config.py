import numpy as np
from scipy.stats import norm
from kicq import priors
from kicq.mcmcUtils import LnLike


model = "CTL"

# =========================================
# RUP 147 constraints from Torres et al. (2018)
# =========================================

# Stellar properties
m1 = 1.0782
m1Sig = 0.0019
m2 = 1.0661
m2Sig = 0.0027

# Radii [Rsun]
r1 = 1.055
r1Sig = 0.011
r2 = 1.042
r2Sig = 0.012

# Teffs [K]
teff1 = 5930.0
teff1Sig = 100.0
teff2 = 5880.0
teff2Sig = 100.0

# Luminosities [Lsun]
lum1 = 1.233
lum1Sig = 0.098
lum2 = 1.165
lum2Sig = 0.090

# Primary star rotation period [d]
prot1 = 6.89
prot1Sig = 0.27

# Orbital Properties

# Orbital period [d]
porb = 6.527139
porbSig = 0.000013

# Eccentricity -> wide constraint as Torres+2018 constrains e to within
# [0.00012,0.0023] with 99% confidence
ecc = 0.00121
eccSig = 0.0005

# Age [Gyr] constraint: 2.48 +/- 0.43 but this comes from stellar evolution
# models, so I'll use a broad prior: 1-4 Gyr
age = 2.48
ageSig = 0.43

# =========================================
# Set up priors
# =========================================

m1Bounds    = (1.060, 1.090)
m2Bounds    = (1.050, 1.080)
prot1Bounds = (4, 10)
prot2Bounds = (4, 10)
tau1Bounds  = (-3, 2)
tau2Bounds  = (-3, 2)
porbBounds  = (4, 10)
eccBounds   = (0, 0.6)
ageBounds   = (1.0, 4.0)

bounds = (m1Bounds, m2Bounds, prot1Bounds, prot2Bounds,
          tau1Bounds, tau2Bounds, porbBounds, eccBounds, ageBounds) 


def LnPriorObject(x, **kwargs):
    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x

    # Prot prior [d]
    if (dProt1 < prot1Bounds[0]) or (dProt1 > prot1Bounds[1]):
        return -np.inf
    if (dProt2 < prot2Bounds[0]) or (dProt2 > prot2Bounds[1]):
        return -np.inf

    # Tidal tau prior (note that tau is log10(tau [s]) in chain)
    if (dTau1 < tau1Bounds[0]) or (dTau1 > tau1Bounds[1]):
        return -np.inf
    if (dTau2 < tau2Bounds[0]) or (dTau2 > tau2Bounds[1]):
        return -np.inf

    # Eccentricity prior
    if (dEcc < eccBounds[0]) or (dEcc > eccBounds[1]):
        return -np.inf

    # Age prior: Instead of Gaussian on quote age, use wide flat prior over
    if (dAge < ageBounds[0]) or (dAge > ageBounds[1]):
        return -np.inf

    # Orbital period prior [d]
    if (dPorb < porbBounds[0]) or (dPorb > porbBounds[1]):
        return -np.inf

    # Generous mass prior range
    if (dMass1 < m1Bounds[0]) or (dMass1 > m1Bounds[1]) or (dMass2 < m2Bounds[0]) or (dMass2 > m2Bounds[1]):
        return -np.inf

    # Mass priors
    lnprior = norm.logpdf(dMass1, m1, m1Sig)
    lnprior += norm.logpdf(dMass2, m2, m2Sig)

    # Age prior
    lnprior += norm.logpdf(dAge, age, ageSig)

    return lnprior


def samplePriorObject(size=1, **kwargs):
    ret = []
    for ii in range(size):
        while True:
            guess = [norm.rvs(loc=m1, scale=m1Sig, size=1)[0],
                     norm.rvs(loc=m2, scale=m2Sig, size=1)[0],
                     np.random.uniform(low=prot1Bounds[0], high=prot1Bounds[1]),
                     np.random.uniform(low=prot2Bounds[0], high=prot2Bounds[1]),
                     np.random.uniform(low=tau1Bounds[0], high=tau1Bounds[1]),
                     np.random.uniform(low=tau2Bounds[0], high=tau2Bounds[1]),
                     np.random.uniform(low=porbBounds[0], high=porbBounds[1]),
                     np.random.uniform(low=eccBounds[0], high=eccBounds[1]),
                     norm.rvs(loc=age, scale=ageSig, size=1)[0]]
            if not np.isinf(LnPriorObject(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]


def LnProb(theta, **kwargs):

    return LnLike(theta, **kwargs)[0] + LnPriorObject(theta, **kwargs)


# =========================================
# Dict to hold all constraints
# =========================================

kwargs = {"PATH" : ".",
          "R1" : r1,
          "R1SIG" : r1Sig,
          "R2" : r2,
          "R2SIG" : r2Sig,
          "LUM1" : lum1,
          "LUM1SIG" : lum1Sig,
          "LUM2" : lum2,
          "LUM2SIG" : lum2Sig,
          "PROT1" : prot1,
          "PROT1SIG" : prot1Sig,
          "PORB" : porb,
          "PORBSIG" : porbSig,
          "ECC" : ecc,
          "ECCSIG" : eccSig,
          "LnPrior" : LnPriorObject,
          "PriorSample" : samplePriorObject, 
          "MODEL" : model}