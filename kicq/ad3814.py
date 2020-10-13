#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data and priors for the ~650 Myr old AD 3814 stellar binary in Praesepe
"""

import numpy as np
from scipy.stats import norm
from . import utils

__all__ = ["kwargsAD3814", "LnPriorAD3814", "samplePriorAD3814"]


# Stellar properties
m1AD3814 = 0.3813
m1AD3814Sig = 0.0074
m2AD3814 = 0.2022
m2AD3814Sig = 0.0045

# Radii [Rsun]
r1AD3814 = 0.3610
r1AD3814Sig = 0.0033
r2AD3814 = 0.2256
r2AD3814Sig = 0.0063

# Primary star rotation period [d]
prot1AD3814 = 7.494
prot1AD3814Sig = 0.004

# Orbital Properties

# Orbital period [d]
porbAD3814 = 6.015717
porbAD3814Sig = 0.000001

# Eccentricity
eccAD3814 = 0.00194
eccAD3814Sig = 0.00253

# Age [Gyr] constraint
ageAD3814 = 0.65

### Prior, likelihood, MCMC functions ###


def LnPriorAD3814(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x

    # Prot prior [d]
    if (dProt1 < 0.25) or (dProt1 > 15):
        return -np.inf
    if (dProt2 < 0.25) or (dProt2 > 15):
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
    if (dPorb < 3) or (dPorb > 11):
        return -np.inf

    # Generous mass prior range
    if (dMass1 < 0.35) or (dMass1 > 0.45) or (dMass2 < 0.15) or (dMass2 > 0.25):
        return -np.inf

    # Mass priors
    lnprior = norm.logpdf(dMass1, m1AD3814, m1AD3814Sig)
    lnprior += norm.logpdf(dMass2, m2AD3814, m2AD3814Sig)

    return lnprior
# end function


def samplePriorAD3814(size=1, **kwargs):
    """
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x
    """
    ret = []
    for ii in range(size):
        while True:
            guess = [norm.rvs(loc=m1AD3814, scale=m1AD3814Sig, size=1)[0],
                     norm.rvs(loc=m2AD3814, scale=m2AD3814Sig, size=1)[0],
                     np.random.uniform(low=0.25, high=15.0),
                     np.random.uniform(low=0.25, high=15.0),
                     np.random.uniform(low=-3, high=2),
                     np.random.uniform(low=-3, high=2),
                     np.random.uniform(low=3.0, high=11.0),
                     np.random.uniform(low=0.0, high=0.6),
                     np.random.uniform(low=0.4, high=1.0)]
            if not np.isinf(LnPriorAD3814(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function

# Dict to hold all constraints
kwargsAD3814 = {"PATH" : ".",
                "R1" : r1AD3814,
                "R1SIG" : r1AD3814Sig,
                "R2" : r2AD3814,
                "R2SIG" : r2AD3814Sig,
                "PROT1" : prot1AD3814,
                "PROT1SIG" : prot1AD3814Sig,
                "PORB" : porbAD3814,
                "PORBSIG" : porbAD3814Sig,
                "ECC" : eccAD3814,
                "ECCSIG" : eccAD3814Sig,
                "LnPrior" : LnPriorAD3814,
                "PriorSample" : samplePriorAD3814}
