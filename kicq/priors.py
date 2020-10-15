#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data and priors for Rup 147 stellar binary with data from Torres et al. (2018)
"""

import numpy as np
from scipy.stats import norm
from . import LnLike


__all__ = ["LnPrior", "samplePrior"]


def LnPrior(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x

    bounds   = kwargs.get('bounds')
    m1Obs    = kwargs.get('m1Obs')
    m1ObsSig = kwargs.get('m1ObsSig')
    m2Obs    = kwargs.get('m2Obs')
    m2ObsSig = kwargs.get('m2ObsSig')

    m1Bounds, m2Bounds, prot1Bounds, prot2Bounds, \
          tau1Bounds, tau2Bounds, porbBounds, eccBounds, ageBounds = bounds   

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
    lnprior = norm.logpdf(dMass1, m1Obs, m1ObsSig)
    lnprior += norm.logpdf(dMass2, m2Obs, m2ObsSig)

    return lnprior


def samplePrior(size=1, **kwargs):
    """
    """
    bounds   = kwargs.get('bounds')
    m1Obs    = kwargs.get('m1Obs')
    m1ObsSig = kwargs.get('m1ObsSig')
    m2Obs    = kwargs.get('m2Obs')
    m2ObsSig = kwargs.get('m2ObsSig')

    m1Bounds, m2Bounds, prot1Bounds, prot2Bounds, \
          tau1Bounds, tau2Bounds, porbBounds, eccBounds, ageBounds = bounds

    ret = []
    for ii in range(size):
        while True:
            guess = [norm.rvs(loc=m1Obs, scale=m1ObsSig, size=1)[0],
                     norm.rvs(loc=m2Obs, scale=m2ObsSig, size=1)[0],
                     np.random.uniform(low=prot1Bounds[0], high=prot1Bounds[1]),
                     np.random.uniform(low=prot2Bounds[0], high=prot2Bounds[1]),
                     np.random.uniform(low=tau1Bounds[0], high=tau1Bounds[1]),
                     np.random.uniform(low=tau2Bounds[0], high=tau2Bounds[1]),
                     np.random.uniform(low=porbBounds[0], high=porbBounds[1]),
                     np.random.uniform(low=eccBounds[0], high=eccBounds[1]),
                     np.random.uniform(low=ageBounds[0], high=ageBounds[1])]
            if not np.isinf(LnPrior(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
