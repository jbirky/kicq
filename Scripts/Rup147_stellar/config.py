import numpy as np
from scipy.stats import norm
from kicq.mcmcUtilsStellar import LnLike


model = "Stellar"

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

m1Bounds    = (0.9, 1.1)
m2Bounds    = (0.9, 1.1)
ageBounds   = (0.5, 4.5)

bounds = (m1Bounds, m2Bounds, ageBounds) 


def LnPriorObject(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass1, dMass2, dAge  = x

#     bounds   = kwargs.get('bounds')
    m1Obs    = kwargs.get('m1Obs')
    m1ObsSig = kwargs.get('m1ObsSig')
    m2Obs    = kwargs.get('m2Obs')
    m2ObsSig = kwargs.get('m2ObsSig')

    m1Bounds, m2Bounds, ageBounds = bounds   
    
    if (dAge < ageBounds[0]) or (dAge > ageBounds[1]):
        return -np.inf

    # Generous mass prior range
    if (dMass1 < m1Bounds[0]) or (dMass1 > m1Bounds[1]) or (dMass2 < m2Bounds[0]) or (dMass2 > m2Bounds[1]):
        return -np.inf

    # Mass priors
    lnprior = norm.logpdf(dMass1, m1Obs, m1ObsSig)
    lnprior += norm.logpdf(dMass2, m2Obs, m2ObsSig)

    return lnprior


def samplePriorObject(size=1, **kwargs):
    """
    """
#     bounds   = kwargs.get('bounds')
    m1Obs    = kwargs.get('m1Obs')
    m1ObsSig = kwargs.get('m1ObsSig')
    m2Obs    = kwargs.get('m2Obs')
    m2ObsSig = kwargs.get('m2ObsSig')

    print('sample prior bounds', bounds)
    m1Bounds, m2Bounds, ageBounds = bounds 

    ret = []
    for ii in range(size):
        while True:
            guess = [norm.rvs(loc=m1Obs, scale=m1ObsSig, size=1)[0],
                     norm.rvs(loc=m2Obs, scale=m2ObsSig, size=1)[0],
                     np.random.uniform(low=ageBounds[0], high=ageBounds[1])]
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
          "LnPrior" : LnPriorObject,
          "PriorSample" : samplePriorObject,
          "MODEL" : model}