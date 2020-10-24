#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
docs
"""

import vplot as vpl
import numpy as np
import emcee
import subprocess
import re
import os, sys
import random
import argparse
from . import utils


__all__ = ["FunctionWrapper", "LnLike", "GetEvol", "RunMCMC"]

class FunctionWrapper(object):
    """"
    A simple function wrapper class. Stores :py:obj:`args` and :py:obj:`kwargs` and
    allows an arbitrary function to be called with a single parameter :py:obj:`x`
    """

    def __init__(self, f, *args, **kwargs):
        """
        """

        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        """
        """

        return self.f(x, *self.args, **self.kwargs)
# end class

def LnLike(x, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation!
    """

    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x

    # Unlog tau, convect to yr
    if kwargs['MODEL'] == 'CTL':    
        dTau1 = (10 ** dTau1) / utils.YEARSEC
        dTau2 = (10 ** dTau2) / utils.YEARSEC
    else:
        dTau1 = (10 ** dTau1) 
        dTau2 = (10 ** dTau2) 

    # Convert from Gyr to yr then set stop time, output time to age of system
    dStopTime = dAge * 1.0e9
    dOutputTime = kwargs.get('dOutputTime', dStopTime)

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)

    # If not finite, invalid initial conditions
    if np.isinf(lnprior):
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Get constraints: if they don't exist, set them to None
    # Assume we have radius constraints for both stars
    r1 = kwargs.get("R1")
    r1Sig = kwargs.get("R1SIG")
    r2 = kwargs.get("R2")
    r2Sig = kwargs.get("R2SIG")
    try:
        teff1 = kwargs.get("TEFF1")
        teff1Sig = kwargs.get("TEFF1SIG")
    except KeyError:
        teff1 = None
        teff1Sig = None
    try:
        teff2 = kwargs.get("TEFF2")
        teff2Sig = kwargs.get("TEFF2SIG")
    except KeyError:
        teff2 = None
        teff2Sig = None
    try:
        lum1 = kwargs.get("LUM1")
        lum1Sig = kwargs.get("LUM1SIG")
    except KeyError:
        lum1 = None
        lum1Sig = None
    try:
        lum2 = kwargs.get("LUM2")
        lum2Sig = kwargs.get("LUM2SIG")
    except KeyError:
        lum2 = None
        lum2Sig = None
    try:
        porb = kwargs.get("PORB")
        porbSig = kwargs.get("PORBSIG")
    except KeyError:
        porb = None
        porbSig = None
    try:
        ecc = kwargs.get("ECC")
        eccSig = kwargs.get("ECCSIG")
    except KeyError:
        ecc = None
        eccSig = None
    try:
        prot1 = kwargs.get("PROT1")
        prot1Sig = kwargs.get("PROT1SIG")
    except KeyError:
        prot1 = None
        prot1Sig = None
    try:
        prot2 = kwargs.get("PROT2")
        prot2Sig = kwargs.get("PROT2SIG")
    except KeyError:
        prot2 = None
        prot2Sig = None

    # Get strings containing VPLanet input filex (they must be provided!)
    try:
        primary_in = kwargs.get("PRIMARYIN")
        secondary_in = kwargs.get("SECONDARYIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply PRIMARYIN, SECONDARYIN, VPLIN.")
        raise

    # Get input PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Get output path
    OUTPATH = kwargs.get('OUTPATH', PATH + "/output")
    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    primaryName = 'pri%012x' % random.randrange(16**12)
    secondaryName = 'sec%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    primaryFile = primaryName + '.in'
    secondaryFile = secondaryName + '.in'
    logfile = sysName + '.log'
    primaryFwFile = '%s.primary.forward' % sysName
    secondaryFwFile = '%s.secondary.forward' % sysName

    # Populate the primary input file (periods negative to make units days in VPLanet)
    primary_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass1), primary_in)
    primary_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt1), primary_in)
    if kwargs.get('MODEL') == "CPL":
        primary_in = re.sub("%s(.*?)#" % "dTidalQ", "%s %.6e #" % ("dTidalQ", dTau1), primary_in)
    else: #otherwise use CTL model
        primary_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau1), primary_in)

    with open(os.path.join(OUTPATH, primaryFile), 'w') as f:
        print(primary_in, file = f)

    # Populate the secondary input file
    secondary_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass2), secondary_in)
    secondary_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt2), secondary_in)
    if kwargs.get('MODEL') == "CPL":
        secondary_in = re.sub("%s(.*?)#" % "dTidalQ", "%s %.6e #" % ("dTidalQ", dTau2), secondary_in)
    else:
        secondary_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau2), secondary_in)
    secondary_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -dPorb), secondary_in)
    secondary_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", dEcc), secondary_in)
    with open(os.path.join(OUTPATH, secondaryFile), 'w') as f:
        print(secondary_in, file = f)

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (primaryFile, secondaryFile), vpl_in)
    with open(os.path.join(OUTPATH, sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(OUTPATH))
    output = vpl.GetOutput(os.path.join(OUTPATH), logfile = logfile)

    try:
        if kwargs.get('remove', True) == True:
            os.remove(os.path.join(OUTPATH, primaryFile))
            os.remove(os.path.join(OUTPATH, secondaryFile))
            os.remove(os.path.join(OUTPATH, sysFile))
            os.remove(os.path.join(OUTPATH, primaryFwFile))
            os.remove(os.path.join(OUTPATH, secondaryFwFile))
            os.remove(os.path.join(OUTPATH, logfile))
    except FileNotFoundError:
        # Run failed!
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age >= dStopTime:
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Get output parameters
    dRad1 = float(output.log.final.primary.Radius)
    dRad2 = float(output.log.final.secondary.Radius)
    dTeff1 = float(output.log.final.primary.Temperature)
    dTeff2 = float(output.log.final.secondary.Temperature)
    dLum1 = float(output.log.final.secondary.Luminosity)
    dLum2 = float(output.log.final.secondary.Luminosity)
    dPorb = float(output.log.final.secondary.OrbPeriod)
    dEcc = float(output.log.final.secondary.Eccentricity)
    dProt1 = float(output.log.final.primary.RotPer)
    dProt2 = float(output.log.final.secondary.RotPer)

    # Compute the likelihood using provided constraints, assuming we have
    # radius constraints for both stars
    
    lnlike = ((dRad1 - r1) / r1Sig) ** 2
    lnlike += ((dRad2 - r2) / r2Sig) ** 2
    if teff1 is not None:
        lnlike += ((dTeff1 - teff1) / teff1Sig) ** 2
    if teff2 is not None:
        lnlike += ((dTeff2 - teff2) / teff2Sig) ** 2
    if lum1 is not None:
        lnlike += ((dLum1 - lum1) / lum1Sig) ** 2
    if lum2 is not None:
        lnlike += ((dLum2 - lum2) / lum2Sig) ** 2
    if porb is not None:
        lnlike += ((dPorb - porb) / porbSig) ** 2
    if ecc is not None:
        lnlike += ((dEcc - ecc) / eccSig) ** 2
    if prot1 is not None:
        lnlike += ((dProt1 - prot1) / prot1Sig) ** 2
    if prot2 is not None:
        lnlike += ((dProt2 - prot2) / prot2Sig) ** 2
    # lnlike = -0.5 * lnlike + lnprior
    lnlike = -0.5 * lnlike 

    # Return likelihood and blobs
    return lnlike, dProt1, dProt2, dPorb, dEcc, dRad1, dRad2, dLum1, dLum2, dTeff1, dTeff2
# end function


def GetEvol(x, **kwargs):
    """
    Run a VPLanet simulation for this initial condition vector, x
    """

    # Get the current vector
    dMass1, dMass2, dProt1, dProt2, dTau1, dTau2, dPorb, dEcc, dAge = x    

    # Unlog tau, convect to yr
    if kwargs['MODEL'] == 'CTL':    
        dTau1 = (10 ** dTau1) / utils.YEARSEC
        dTau2 = (10 ** dTau2) / utils.YEARSEC
    else:
        dTau1 = (10 ** dTau1) 
        dTau2 = (10 ** dTau2) 

    # Convert age to yr from Gyr, set stop time, output time to age of system
    dStopTime = dAge * 1.0e9
    dOutputTime = kwargs.get('dOutputTime', dStopTime)

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)

    # If not finite, invalid initial conditions
    if np.isinf(lnprior):
        return None

    # Get strings containing VPLanet input filex (they must be provided!)
    try:
        primary_in = kwargs.get("PRIMARYIN")
        secondary_in = kwargs.get("SECONDARYIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: must supply PRIMARYIN, SECONDARYIN, VPLIN.")
        raise

    OUTPATH = kwargs.get('OUTPATH', PATH + "/output")
    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    primaryName = 'pri%012x' % random.randrange(16**12)
    secondaryName = 'sec%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    primaryFile = primaryName + '.in'
    secondaryFile = secondaryName + '.in'
    logfile = sysName + '.log'
    primaryFwFile = '%s.primary.forward' % sysName
    secondaryFwFile = '%s.secondary.forward' % sysName

    # Populate the primary input file (periods negative to make units days in VPLanet)
    primary_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass1), primary_in)
    primary_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt1), primary_in)
    if kwargs.get('MODEL') == "CPL":
        primary_in = re.sub("%s(.*?)#" % "dTidalQ", "%s %.6e #" % ("dTidalQ", dTau1), primary_in)
    else: #otherwise use CTL model
        primary_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau1), primary_in)

    with open(os.path.join(OUTPATH, primaryFile), 'w') as f:
        print(primary_in, file = f)

    # Populate the secondary input file
    secondary_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass2), secondary_in)
    secondary_in = re.sub("%s(.*?)#" % "dRotPeriod", "%s %.6e #" % ("dRotPeriod", -dProt2), secondary_in)
    if kwargs.get('MODEL') == "CPL":
        secondary_in = re.sub("%s(.*?)#" % "dTidalQ", "%s %.6e #" % ("dTidalQ", dTau2), secondary_in)
    else:
        secondary_in = re.sub("%s(.*?)#" % "dTidalTau", "%s %.6e #" % ("dTidalTau", dTau2), secondary_in)
    secondary_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -dPorb), secondary_in)
    secondary_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", dEcc), secondary_in)
    with open(os.path.join(OUTPATH, secondaryFile), 'w') as f:
        print(secondary_in, file = f)

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (primaryFile, secondaryFile), vpl_in)
    with open(os.path.join(OUTPATH, sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(OUTPATH))
    output = vpl.GetOutput(os.path.join(OUTPATH), logfile = logfile)

    try:
        if kwargs.get('remove', True) == True:
            os.remove(os.path.join(OUTPATH, primaryFile))
            os.remove(os.path.join(OUTPATH, secondaryFile))
            os.remove(os.path.join(OUTPATH, sysFile))
            os.remove(os.path.join(OUTPATH, primaryFwFile))
            os.remove(os.path.join(OUTPATH, secondaryFwFile))
            os.remove(os.path.join(OUTPATH, logfile))
    except FileNotFoundError:
        # Run failed!
        return None

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age >= dStopTime:
        return None

    # Return output
    return output
# end function


def RunMCMC(x0=None, ndim=9, nwalk=90, nsteps=1000, pool=None, backend=None,
            restart=False, **kwargs):
    """
    """

    # Ensure LnPrior, prior sample are in kwargs
    try:
        kwargs["PriorSample"]
    except KeyError as err:
        print("ERROR: Must supply PriorSample function!")
        raise
    try:
        kwargs["LnPrior"]
    except KeyError as err:
        print("ERROR: Must supply LnPrior function!")
        raise

    # Extract path
    PATH = kwargs["PATH"]

    print("Running MCMC...")

    # Get the input files, save them as strings
    with open(os.path.join(PATH, "primary.in"), 'r') as f:
        primary_in = f.read()
        kwargs["PRIMARYIN"] = primary_in
    with open(os.path.join(PATH, "secondary.in"), 'r') as f:
        secondary_in = f.read()
        kwargs["SECONDARYIN"] = secondary_in
    with open(os.path.join(PATH, "vpl.in"), 'r') as f:
        vpl_in = f.read()
        kwargs["VPLIN"] = vpl_in

    # Set up backend to save results
    if backend is not None:
        # Set up the backend
        handler = emcee.backends.HDFBackend(backend)

        # If restarting from a previous interation, initialize backend
        if not restart:
            handler.reset(nwalk, ndim)

    # Populate initial conditions for walkers using random samples over prior
    if not restart:
        # If MCMC isn't initialized, just sample from the prior
        if x0 is None:
            x0 = np.array([kwargs["PriorSample"](**kwargs) for w in range(nwalk)])

    ### Run MCMC ###

    # Define blobs, blob data types
    dtype = [("dProt1F", np.float64), ("dProt2F", np.float64), ("dPorbF", np.float64),
             ("dEccF", np.float64), ("dRad1F", np.float64), ("dRad2F", np.float64),
             ("dLum1F", np.float64), ("dLum2F", np.float64), ("dTeff1F", np.float64),
             ("dTeff2F", np.float64)]

    # Initialize the sampler object
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike, kwargs=kwargs, pool=pool,
                                    blobs_dtype=dtype, backend=handler)

    # Actually run the MCMC
    if restart:
        sampler.run_mcmc(None, nsteps)
    else:
        for ii, result in enumerate(sampler.sample(x0, iterations=nsteps)):
            print("MCMC: %d/%d..." % (ii + 1, nsteps))

    print("Done!")
# end function
