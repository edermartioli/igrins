"""
    Created on Nov 21 2023
    
    Description: This routine process e2ds data into a 1d specrtum
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil
    Institut d'Astrophysique de Paris, France
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_process_e2ds_to_s1d.py --inputA=/Volumes/Samsung_T5/IGRINS-DATA/GS-2023B-Q-201/20230730/analysis/SDCHK_20230730_0150e.fits --inputB=/Volumes/Samsung_T5/IGRINS-DATA/GS-2023B-Q-201/20230730/analysis/SDCHK_20230730_0151e.fits --stdA=/Volumes/Samsung_T5/IGRINS-DATA/GS-2023B-Q-201/20230730/analysis/SDCHK_20230730_0152e.fits --stdB=/Volumes/Samsung_T5/IGRINS-DATA/GS-2023B-Q-201/20230730/analysis/SDCHK_20230730_0153e.fits --output=HD12392_20230730_s1d.fits

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob
import numpy as np

import reduc_lib
import igrinslib
import telluric_lib

import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import constants


from scipy.interpolate import interp1d


def write_spectrum_to_fits(filename, wave, flux, fluxerr, header=None):
    """
        Description: function to save the spectrum to a fits file
        """
    
    if header is None :
        header = fits.Header()

    header.set('TTYPE1', "WAVE")
    header.set('TUNIT1', "NM")
    header.set('TTYPE2', "FLUXES")
    header.set('TUNIT2', "COUNTS")
    header.set('TTYPE2', "FLUXERR")
    header.set('TUNIT2', "COUNTS")

    primary_hdu = fits.PrimaryHDU(header=header)
    hdu_wl = fits.ImageHDU(data=wave, name="WAVE")
    hdu_flux = fits.ImageHDU(data=flux, name="FLUX")
    hdu_fluxerr = fits.ImageHDU(data=fluxerr, name="FLUXERR")

    listofhuds = [primary_hdu, hdu_wl, hdu_flux, hdu_fluxerr]

    mef_hdu = fits.HDUList(listofhuds)

    mef_hdu.writeto(filename, overwrite=True)


igrins_dir = os.path.dirname(__file__)

#-- end of spirou_ccf routine
parser = OptionParser()
parser.add_option("-A", "--inputA", dest="inputA", help="Input A e.fits spectrum data file",type='string',default="")
parser.add_option("-B", "--inputB", dest="inputB", help="Input B e.fits spectrum data file",type='string',default="")
parser.add_option("-a", "--stdA", dest="stdA", help="Input A e.fits standard spectrum data file",type='string',default="")
parser.add_option("-b", "--stdB", dest="stdB", help="Input B e.fits standard spectrum data file",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output s1d spectrum data file",type='string',default="")
parser.add_option("-t", action="store_true", dest="correct_tellurics", help="correct tellurics", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h igrins_process_e2ds_to_s1d.py")
    sys.exit(1)

if options.verbose:
    print('Input A e.fits spectrum: ', options.inputA)
    print('Input B e.fits spectrum: ', options.inputB)
    print('Input A standard e.fits spectrum: ', options.stdA)
    print('Input B standard e.fits spectrum: ', options.stdB)
    print('Output s1d spectrum: ', options.output)


# make list of tfits data files
if options.verbose:
    print("Creating list of e.fits spectrum files...")

spcA = igrinslib.load_spectrum(options.inputA, standard=False)
spcB = igrinslib.load_spectrum(options.inputB, standard=False)

stdA = igrinslib.load_spectrum(options.stdA, standard=True)
stdB = igrinslib.load_spectrum(options.stdB, standard=True)

#speed_of_light_in_kps = constants.c / 1000.
#tell_wl = tell_spectrum["wl"] * (1.0 + spectra[i]['BERV'] / speed_of_light_in_kps)

wave = np.array([])
fluxes = np.array([])
fluxerrs = np.array([])

refit_continuum = True
use_model_teltrans = False

correct_tellurics = options.correct_tellurics

plot_continuum = False

good_windows = []

for order in range(spcA["norders"]) :
    wl = spcA["wl"][order]
    flux = spcA["flux"][order] + spcB["flux"][order]
    fluxerr = np.sqrt(spcA["variance"][order] + spcB["variance"][order])
    keep = np.isfinite(flux)

    swl = stdA["wl"][order]
    sflux = stdA["flux"][order] + stdB["flux"][order]
    sfluxerr = np.sqrt(stdA["variance"][order] + stdB["variance"][order])
    skeep = np.isfinite(sflux)

    fitcont = stdA["FITTED_CONTINUUM"][order]
    obstrans = (stdA["SPEC_FLATTENED"][order] + stdB["SPEC_FLATTENED"][order])/2
    mtrans = (stdA["MODEL_TELTRANS"][order] + stdB["MODEL_TELTRANS"][order])/2

    if len(swl[skeep]) > 100  and len(wl[keep]) > 100 :

        print(order, swl[skeep][0], swl[skeep][-1], np.nanmedian(np.abs(wl[1:]-wl[:-1])))
        #smflux = reduc_lib.interp_spectrum(wl[keep], swl[skeep], sflux[skeep], good_windows=[[swl[skeep][0],swl[skeep][-1]]], kind='cubic')

        mfitcont = np.full_like(wl[keep],1.0)
        mobstrans = np.full_like(wl[keep],1.0)

        if correct_tellurics :
            if use_model_teltrans :
                mobstrans = reduc_lib.interp_spectrum(wl[keep], swl[skeep], mtrans[skeep], good_windows=[[swl[skeep][0],swl[skeep][-1]]], kind='cubic')
            else :
                mobstrans = reduc_lib.interp_spectrum(wl[keep], swl[skeep], obstrans[skeep], good_windows=[[swl[skeep][0],swl[skeep][-1]]], kind='cubic')

        if refit_continuum :
            continuum = np.full_like(wl,np.nan)
            if len(wl[keep]) > 300 :
                continuum[keep] = reduc_lib.fit_continuum(wl[keep], flux[keep], function='polynomial', order=4, nit=10, rej_low=1.0, rej_high=4.0, grow=1, med_filt=1, percentile_low=0., percentile_high=100.,min_points=100, xlabel="wavelength", ylabel="flux", plot_fit=False, silent=True)
            else :
                continue
            mfitcont = continuum[keep]
        else :
            mfitcont = reduc_lib.interp_spectrum(wl[keep], swl[skeep], fitcont[skeep], good_windows=[[swl[skeep][0],swl[skeep][-1]]], kind='cubic')

        wave = np.append(wave, wl[keep])
        fluxes = np.append(fluxes, flux[keep] / (mfitcont * mobstrans))
        fluxerrs = np.append(fluxerrs, fluxerr[keep] / (mfitcont * mobstrans))

        good_windows.append([wl[keep][0],wl[keep][-1]])

        if plot_continuum :
            plt.plot(wl[keep], flux[keep], '-', color="grey", alpha=0.5, zorder=1.2)
            plt.plot(wl[keep], mfitcont, '-', color="green", lw=2, zorder=2)
            plt.plot(wl[keep], flux[keep] / mobstrans, '-', color="darkblue", alpha=0.5, zorder=1)
        else :
            plt.plot(wl[keep], flux[keep] / (mfitcont * mobstrans), '-', color="k", alpha=0.5, zorder=1.2)
            plt.plot(wl[keep], flux[keep] / mfitcont, '-', color="darkblue", alpha=0.5, zorder=1)
            plt.plot(swl, mtrans, '-', color="darkred", alpha=0.5, zorder=1)

plt.xlabel("wavelength [nm]",fontsize=18)
plt.ylabel("flux",fontsize=18)
plt.legend(fontsize=18)
plt.show()


if options.output != "" :

    sortedwl = np.argsort(wave)
    
    write_spectrum_to_fits(options.output, wave[sortedwl], fluxes[sortedwl], fluxerrs[sortedwl], header=spcA["header"])

