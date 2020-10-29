import numpy as np
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

m1Bounds    = (0.9, 1.1)
m2Bounds    = (0.9, 1.1)
prot1Bounds = (0.25, 15)
prot2Bounds = (0.25, 15)
tau1Bounds  = (-3, 2)
tau2Bounds  = (-3, 2)
porbBounds  = (3, 11)
eccBounds   = (0, 0.6)
ageBounds   = (0.5, 4.5)

bounds = (m1Bounds, m2Bounds, prot1Bounds, prot2Bounds,
          tau1Bounds, tau2Bounds, porbBounds, eccBounds, ageBounds) 


def LnPriorObject(x, **kwargs):
    lnp = priors.LnPrior(x, bounds=bounds, 
                            m1Obs=m1,
                            m1ObsSig=m1Sig,
                            m2Obs=m2,
                            m2ObsSig=m2Sig)
    return lnp


def samplePriorObject(size=1, **kwargs):
    slnp = priors.samplePrior(size=size,
                              bounds=bounds, 
                              m1Obs=m1,
                              m1ObsSig=m1Sig,
                              m2Obs=m2,
                              m2ObsSig=m2Sig)
    return slnp


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