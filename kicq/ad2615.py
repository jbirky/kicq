#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data and priors for the ~650 Myr old AD 2615 stellar binary in Praesepe
"""

import numpy as np
from scipy.stats import norm
from . import utils

__all__ = ["kwargsAD2615", "LnPriorAD2615", "samplePriorAD2615"]


# Stellar properties
m1AD2615 = 0.212
m1AD2615Sig = 0.0012
m2AD2615 = 0.255
m2AD2615Sig = 0.0013

# Radii [Rsun]
r1AD2615 = 0.233
r1AD2615Sig = 0.0013
r2AD2615 = 0.267
r2AD2615Sig = 0.0014

# Primary star rotation period [d]
prot1AD2615 = 12.150
prot1AD2615Sig = 0.074

# Orbital Properties

# Orbital period [d]
porbAD2615 = 11.615254
porbAD2615Sig = 0.000073

# Eccentricity
eccAD2615 = 0.00254
eccAD2615Sig = 0.00406

# Age [Gyr] constraint
ageAD2615 = 0.65

### Prior, likelihood, MCMC functions ###


def LnPriorAD2615(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x

    # Prot prior [d]
    if (dProt1 < 0.25) or (dProt1 > 16):
        return -np.inf
    if (dProt2 < 0.25) or (dProt2 > 16):
        return -np.inf

    # Tidal tau prior (note that tau is log10(tau [s]) in chain)
    if (dTau1 < -3.0) or (dTau1 > 2.0):
        return -np.inf
    if (dTau2 < -3.0) or (dTau2 > 2.0):
        return -np.inf

    # Eccentricity prior
    if (dEcc < 0) or (dEcc > 0.6):
        return -np.inf

    # Orbital period prior [d]
    if (dPorb < 3) or (dPorb > 16):
        return -np.inf

    # Generous mass prior range
    if (dMass1 < 0.15) or (dMass1 > 0.25) or (dMass2 < 0.20) or (dMass2 > 0.30):
        return -np.inf

    # Mass priors
    lnprior = norm.logpdf(dMass1, m1AD2615, m1AD2615Sig)
    lnprior += norm.logpdf(dMass2, m2AD2615, m2AD2615Sig)

    return lnprior
# end function


def samplePriorAD2615(size=1, **kwargs):
    """
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x
    """
    ret = []
    for ii in range(size):
        while True:
            guess = [norm.rvs(loc=m1AD2615, scale=m1AD2615Sig, size=1)[0],
                     norm.rvs(loc=m2AD2615, scale=m2AD2615Sig, size=1)[0],
                     np.random.uniform(low=0.25, high=16.0),
                     np.random.uniform(low=0.25, high=16.0),
                     np.random.uniform(low=-3, high=2),
                     np.random.uniform(low=-3, high=2),
                     np.random.uniform(low=3.0, high=16.0),
                     np.random.uniform(low=0.0, high=0.6),
                     np.random.uniform(low=0.4, high=1.0)]
            if not np.isinf(LnPriorAD2615(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function

# Dict to hold all constraints
kwargsAD2615 = {"PATH" : ".",
                "R1" : r1AD2615,
                "R1SIG" : r1AD2615Sig,
                "R2" : r2AD2615,
                "R2SIG" : r2AD2615Sig,
                "PROT1" : prot1AD2615,
                "PROT1SIG" : prot1AD2615Sig,
                "PORB" : porbAD2615,
                "PORBSIG" : porbAD2615Sig,
                "ECC" : eccAD2615,
                "ECCSIG" : eccAD2615Sig,
                "LnPrior" : LnPriorAD2615,
                "PriorSample" : samplePriorAD2615}
