# -*- coding: iso-8859-1 -*-
"""
    Created on November 24 2021
    
    Description: Routine to model the telluric transmission spectrum in the IGRINS data
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/Science/IGRINS/igrins_telluric_model.py --input=SDCHK_20211004_0557e.fits
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import telluric_lib
import igrinslib
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import leastsq

from copy import deepcopy
igrins_dir = os.path.dirname(__file__)
telluric_grid_path = os.path.join(igrins_dir, 'TelluricGrids/')
#telluric_grid_path = os.path.join("/Volumes/EDERIAP/","TelluricGrids/")


def get_probes() :
    co2_probes = [[1531.62,1543.89],
                  [1568.04,1585.97],
                  [1597.71,1616.48],
                  [2045.52,2077.13]]

    h2o_probes = [[1491.72,1506.01],
                  [1731.31,1763.70],
                  [1979.04,1995.14],
                  [2096.21,2129.79]]

    ch4_probes = [[1629.07,1689.62],
                  [2219.00,2360.10]]

    return co2_probes + h2o_probes + ch4_probes


def fit_spectrum(x, y, yerr, xm, ym) :

    def specmodel (spars, wl):
        
        wl_copy = deepcopy(wl)
        wl_copy *= (1.0 - spars[0] / (constants.c / 1000.))

        if wl_copy[0] > xm[0] and  wl_copy[-1] < xm[-1] :
            fout = spars[1] + spars[2] * telluric_lib.interp_spectrum(wl_copy, xm, ym) + spars[3] * wl_copy * wl_copy
        else :
            fout = np.ones_like(wl_copy)
        return fout
        
    def errfunc (coeffs, xx, yy, yyerr):
        model = specmodel(coeffs, xx)
        return (yy - model) / yyerr


    pars = [0., 0.0, 1.0, 0.]
    pfit, pcov, infodict, errmsg, success = leastsq(errfunc, pars, args=(x, y, yerr), full_output=1)

    if (len(fluxdata) > len(vars)) and pcov is not None:
        s_sq = (residual(pfit, wldata, fluxdata, errordata)**2).sum()/(len(fluxdata)-len(vars))
        pcov = pcov * s_sq
    else:
        pcov = np.inf
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    perr = np.array(error)
    #print(pfit,perr)
    return pfit, perr



def fit_h2o(pars, wl, flux, plot=False) :

    def h2o_trans_model (spars):
        """
            pars[0] = radial velocity shift
            pars[1] = constant shift
            pars[2] = relative humidity
         """
        
        wl_copy = deepcopy(wl)
        wl_copy *= (1.0 - spars[0] / (constants.c / 1000.))
        
        pars_copy = deepcopy(pars)
        pars_copy['humidity'] = spars[2]
        flux_model = telluric_lib.calculate_spec_model(wl_copy, pars_copy, molecule='h2o')

        fout = spars[1] + flux_model["telluric_model"]

        return fout

    def h2o_errfunc (coeffs, yy):
        model = h2o_trans_model(coeffs)
        return (yy - model)

    guess_spars = [0., 0.0, 50.]
    pfit, pcov, infodict, errmsg, success = leastsq(h2o_errfunc, guess_spars, args=(flux), full_output=1)
    
    fit_h2o_trans = h2o_trans_model(pfit)
    
    if plot :
        plt.plot(wl, flux,'o', label="data")
        plt.plot(wl, fit_h2o_trans, 'g-', label="fit model")
    
        plt.legend()
        plt.xlabel("wavelength [nm]")
        plt.ylabel("transmission")
        plt.show()

    pars["humidity"] = pfit[2]
    return pars, fit_h2o_trans

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input spectrum file",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output telluric model FITS file",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with igrins_telluric_model.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input spectrum file: ', options.input)
    print('Output telluric model FITS file: ', options.output)

# Load SPIRou reduced FITS file
spectrum = igrinslib.load_spectrum(options.input, standard=True)

wl, flux, fluxerr = np.array([]), np.array([]), np.array([])
for order in range(54) :
    finite = np.isfinite(spectrum["flux"][order])
    wl = np.append(wl,spectrum["wl"][order][finite])
    flux = np.append(flux,spectrum["MODEL_TELTRANS"][order][finite] * spectrum["SPEC_FLATTENED"][order][finite])
    fluxerr = np.append(fluxerr,spectrum["variance"][order][finite])

    plt.plot(spectrum["wl"][order][finite],spectrum["MODEL_TELTRANS"][order][finite]*spectrum["SPEC_FLATTENED"][order][finite],'.',color='darkgreen', alpha=0.4)
    #plt.plot(spectrum["wl"][order][finite],spectrum["MODEL_TELTRANS"][order][finite],'-',color='darkblue', alpha=0.4)

sorted = np.argsort(wl)
wl, flux, fluxerr = wl[sorted], flux[sorted], fluxerr[sorted]
wl1, wl2 = np.min(wl), np.max(wl)



airmasses = [1.0,1.5,2.0,2.5,3.0]
pwvs = ["005","010","015","025","035","050","075","100","200"]
pwvs_colors = ["b","r","g","orange","brown","cyan","magenta","purple","k"]
#for am in airmasses :
am = 1.5

for r in [[1491.72,1506.01],[1731.31,1763.70],[1979.04,1995.14],[2096.21,2129.79]] :
    for i in range(len(pwvs)) :
        tell = telluric_lib.get_telluric_model_from_grid(telluric_grid_path, airmass=am, pwv=pwvs[i], wl1=r[0], wl2=r[1], to_resolution=40000)
        plt.plot(tell["wl"],tell["trans"], color=pwvs_colors[i], label="AM={:.1f} PWVV={}".format(am,pwvs[i]))
    
#plt.legend()
plt.xlabel("wavelength [nm]")
plt.ylabel("transmission")
plt.show()




