#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data and priors for Rup 147 stellar binary with data from Torres et al. (2018)
"""

import numpy as np
from scipy.stats import norm
from . import utils, LnLike


__all__ = ["kwargsRUP147", "LnPriorRUP147", "samplePriorRUP147", "LnProbRUP147", "bounds"]

# Observational constraints from Torres et al. (2018)

# Stellar properties
m1RUP147 = 1.0782
m1RUP147Sig = 0.0019
m2RUP147 = 1.0661
m2RUP147Sig = 0.0027

# Radii [Rsun]
r1RUP147 = 1.055
r1RUP147Sig = 0.011
r2RUP147 = 1.042
r2RUP147Sig = 0.012

# Teffs [K]
teff1RUP147 = 5930.0
teff1RUP147Sig = 100.0
teff2RUP147 = 5880.0
teff2RUP147Sig = 100.0

# Luminosities [Lsun]
lum1RUP147 = 1.233
lum1RUP147Sig = 0.098
lum2RUP147 = 1.165
lum2RUP147Sig = 0.090

# Primary star rotation period [d]
prot1RUP147 = 6.89
prot1RUP147Sig = 0.27

# Orbital Properties

# Orbital period [d]
porbRUP147 = 6.527139
porbRUP147Sig = 0.000013

# Eccentricity -> wide constraint as Torres+2018 constrains e to within
# [0.00012,0.0023] with 99% confidence
eccRUP147 = 0.00121
eccRUP147Sig = 0.0005

# Age [Gyr] constraint: 2.48 +/- 0.43 but this comes from stellar evolution
# models, so I'll use a broad prior: 1-4 Gyr
ageRUP147 = 2.48
ageRUP147Sig = 0.43

### Prior, likelihood, MCMC functions ###

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


def LnPriorRUP147(x, **kwargs):
    """
    log prior
    """

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
    # [0.5, 4.5] Gyr
    if (dAge < ageBounds[0]) or (dAge > ageBounds[1]):
        return -np.inf

    # Orbital period prior [d]
    if (dPorb < porbBounds[0]) or (dPorb > porbBounds[1]):
        return -np.inf

    # Generous mass prior range
    if (dMass1 < m1Bounds[0]) or (dMass1 > m1Bounds[1]) or (dMass2 < m2Bounds[0]) or (dMass2 > m2Bounds[1]):
        return -np.inf

    # Mass priors
    lnprior = norm.logpdf(dMass1, m1RUP147, m1RUP147Sig)
    lnprior += norm.logpdf(dMass2, m2RUP147, m2RUP147Sig)

    return lnprior


def samplePriorRUP147(size=1, **kwargs):
    """
    """
    ret = []
    for ii in range(size):
        while True:
            guess = [norm.rvs(loc=m1RUP147, scale=m1RUP147Sig, size=1)[0],
                     norm.rvs(loc=m2RUP147, scale=m2RUP147Sig, size=1)[0],
                     np.random.uniform(low=prot1Bounds[0], high=prot1Bounds[1]),
                     np.random.uniform(low=prot2Bounds[0], high=prot2Bounds[1]),
                     np.random.uniform(low=tau1Bounds[0], high=tau1Bounds[1]),
                     np.random.uniform(low=tau2Bounds[0], high=tau2Bounds[1]),
                     np.random.uniform(low=porbBounds[0], high=porbBounds[1]),
                     np.random.uniform(low=eccBounds[0], high=eccBounds[1]),
                     np.random.uniform(low=ageBounds[0], high=ageBounds[1])]
            if not np.isinf(LnPriorRUP147(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]


def LnProbRUP147(theta, **kwargs):
    lnP = -np.inf
    while not np.isfinite(lnP):
        lnP = LnLike(theta, **kwargs)[0] + LnPriorRUP147(theta, **kwargs)

    return lnP


# Dict to hold all constraints
kwargsRUP147 = {"PATH" : ".",
                "R1" : r1RUP147,
                "R1SIG" : r1RUP147Sig,
                "R2" : r2RUP147,
                "R2SIG" : r2RUP147Sig,
                "LUM1" : lum1RUP147,
                "LUM1SIG" : lum1RUP147Sig,
                "LUM2" : lum2RUP147,
                "LUM2SIG" : lum2RUP147Sig,
                "PROT1" : prot1RUP147,
                "PROT1SIG" : prot1RUP147Sig,
                "PORB" : porbRUP147,
                "PORBSIG" : porbRUP147Sig,
                "ECC" : eccRUP147,
                "ECCSIG" : eccRUP147Sig,
                "LnPrior" : LnPriorRUP147,
                "PriorSample" : samplePriorRUP147}
