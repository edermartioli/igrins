"""
    Created on Nov 17 2021
    
    Description: This routine extracts IGRINS spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil
    Institut d'Astrophysique de Paris, France
    
    Simple usage example:
    
    python /Users/eder/Science/IGRINS/igrins_extract_spectra.py --recipe_file=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_B/20210803.recipes --raw_data_dir=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_B/raw/  --reduced_data_dir=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_B/reduced/ --output_dir=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_B/analysis/
    
    python /Users/eder/Science/IGRINS/igrins_extract_spectra.py --recipe_file=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_C/20211004.recipes --raw_data_dir=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_C/raw/  --reduced_data_dir=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_C/reduced/ --output_dir=/Volumes/EDERIAP/IGRINS-DATA/AUMIC_C/analysis/

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob
import numpy as np
import astropy.io.fits as fits
#import reduc_lib
import igrinslib

import matplotlib.pyplot as plt
from scipy import optimize
from copy import deepcopy
import warnings

from PyAstronomy import pyasl

igrins_dir = os.path.dirname(__file__)


def read_recipe_file(recipe_file) :

    recipes = {}
    
    recipes['DATE'] = os.path.basename(recipe_file)[:8]

    recipes['OBJNAME'] = []
    recipes['OBJTYPE'] = []
    recipes['GROUP1'] = []
    recipes['GROUP2'] = []
    recipes['EXPTIME'] = []
    recipes['RECIPE'] = []
    recipes['OBSIDS'] = []
    recipes['FRAMETYPES'] = []

    f = open(recipe_file, 'r')
    
    for line in f:
        if line[0] != '#':
            cols = line.replace("\n","").split(",")
            if cols[0] != 'OBJNAME':
                recipes['OBJNAME'].append(cols[0].replace(" ",""))
                recipes['OBJTYPE'].append(cols[1].replace(" ",""))
                recipes['GROUP1'].append(cols[2].replace(" ",""))
                recipes['GROUP2'].append(cols[3].replace(" ",""))
                recipes['EXPTIME'].append(cols[4].replace(" ",""))
                recipes['RECIPE'].append(cols[5].replace(" ",""))
                obsids = np.array(cols[6].split(),dtype="int")
                frametypes = np.array(cols[7].split())
                recipes['OBSIDS'].append(obsids)
                recipes['FRAMETYPES'].append(frametypes)
    f.close()
    
    recipes['OBJNAME'] = np.array(recipes['OBJNAME'])
    recipes['OBJTYPE'] = np.array(recipes['OBJTYPE'])
    recipes['GROUP1'] = np.array(recipes['GROUP1'])
    recipes['GROUP2'] = np.array(recipes['GROUP2'])
    recipes['EXPTIME'] = np.array(recipes['EXPTIME'])
    recipes['RECIPE'] = np.array(recipes['RECIPE'])
    recipes['OBSIDS'] = recipes['OBSIDS']
    recipes['FRAMETYPES'] = recipes['FRAMETYPES']

    return recipes


def get_wavelengths(reduced_data_dir) :
    stds = recipes['OBJTYPE'] == 'STD'
    
    h_pattern = reduced_data_dir + "/SDCH_*wave.fits"
    k_pattern = reduced_data_dir + "/SDCK_*wave.fits"

    h_wave_files = glob.glob(h_pattern)
    k_wave_files = glob.glob(k_pattern)

    waves = []
    
    for i in range(len(h_wave_files)) :
        
        h_data = fits.getdata(h_wave_files[i],0)
        k_data = fits.getdata(k_wave_files[i],0)
        
        if i == 0 :
            for j in range(28) :
                waves.append(h_data[27-j])
            for j in range(26) :
                waves.append(k_data[25-j])
        else :
            for j in range(28) :
                waves[j] += h_data[27-j]
            for j in range(26) :
                waves[j+28] += k_data[25-j]

    for order in range(54) :
        waves[order] /= len(h_wave_files)

    return waves


def extraction_files(date, red_id, id, positive_flux, frametype, objtype, raw_data_dir, reduced_data_dir, outdir) :

    ### Set file existence flags :
    has_required_files = np.array([False,False,False,False,False,False])
    has_reduced_files = np.array([False,False,False,False,False,False])

    loc = {}
    
    ### Basenames :
    h_basename = "SDCH_{}_{:04d}".format(date,id)
    k_basename = "SDCK_{}_{:04d}".format(date,id)

    h_red_basename = "SDCH_{}_{:04d}".format(date,red_id)
    k_red_basename = "SDCK_{}_{:04d}".format(date,red_id)

    loc["H_BASENAME"] = h_basename
    loc["K_BASENAME"] = k_basename

    loc["H_RED_BASENAME"] = h_red_basename
    loc["K_RED_BASENAME"] = k_red_basename

    ### Raw files:
    h_raw_file = raw_data_dir + h_basename + ".fits"
    if os.path.exists(h_raw_file) : has_required_files[0] = True
    else : print("WARNING: missing required file: {}".format(h_raw_file))
                
    k_raw_file = raw_data_dir + k_basename + ".fits"
    if os.path.exists(k_raw_file) : has_required_files[1] = True
    else : print("WARNING: missing required file: {}".format(k_raw_file))
    
    loc["H_RAW"] = h_raw_file
    loc["K_RAW"] = k_raw_file

    ### Rectified spectral files: spec2d, var2d
    h_spec2d_file = reduced_data_dir  + h_red_basename + ".spec2d.fits"
    if os.path.exists(h_spec2d_file) : has_required_files[2] = True
    else : print("WARNING: missing required file: {}".format(h_spec2d_file))

    k_spec2d_file = reduced_data_dir  + k_red_basename + ".spec2d.fits"
    if os.path.exists(k_spec2d_file) : has_required_files[3] = True
    else : print("WARNING: missing required file: {}".format(k_spec2d_file))

    h_var2d_file = reduced_data_dir  + h_red_basename + ".var2d.fits"
    if os.path.exists(h_var2d_file) : has_required_files[4] = True
    else : print("WARNING: missing required file: {}".format(h_var2d_file))
                
    k_var2d_file = reduced_data_dir  + k_red_basename + ".var2d.fits"
    if os.path.exists(k_var2d_file) : has_required_files[5] = True
    else : print("WARNING: missing required file: {}".format(k_var2d_file))

    loc["H_SPEC2D"] = h_spec2d_file
    loc["K_SPEC2D"] = k_spec2d_file

    loc["H_VAR2D"] = h_var2d_file
    loc["K_VAR2D"] = k_var2d_file

    loc["HAS_REQUIRED_FILES"] = has_required_files

    ### Reduced files: spec, variance, sn
    h_spec_file = reduced_data_dir  + h_red_basename + ".spec.fits"
    if os.path.exists(h_spec_file) : has_reduced_files[0] = True
    else : print("WARNING: missing required file: {}".format(h_spec_file))

    k_spec_file = reduced_data_dir  + k_red_basename + ".spec.fits"
    if os.path.exists(k_spec_file) : has_reduced_files[1] = True
    else : print("WARNING: missing required file: {}".format(k_spec2d_file))

    h_variance_file = reduced_data_dir  + h_red_basename + ".variance.fits"
    if os.path.exists(h_variance_file) : has_reduced_files[2] = True
    else : print("WARNING: missing required file: {}".format(h_variance_file))
                
    k_variance_file = reduced_data_dir  + k_red_basename + ".variance.fits"
    if os.path.exists(k_variance_file) : has_reduced_files[3] = True
    else : print("WARNING: missing required file: {}".format(k_variance_file))

    h_sn_file = reduced_data_dir  + h_red_basename + ".sn.fits"
    if os.path.exists(h_sn_file) : has_reduced_files[4] = True
    else : print("WARNING: missing required file: {}".format(h_sn_file))
                
    k_sn_file = reduced_data_dir  + k_red_basename + ".sn.fits"
    if os.path.exists(k_sn_file) : has_reduced_files[5] = True
    else : print("WARNING: missing required file: {}".format(k_sn_file))

    loc["H_SPEC"] = h_spec_file
    loc["K_SPEC"] = k_spec_file

    loc["H_VARIANCE"] = h_variance_file
    loc["K_VARIANCE"] = k_variance_file
    
    loc["H_SN"] = h_sn_file
    loc["K_SN"] = k_sn_file

    loc["HAS_REDUCED_FILES"] = has_reduced_files

    if objtype == 'STD' :
        has_flattened_files = np.array([False,False])

        h_spec_flattened_file = reduced_data_dir  + h_red_basename + ".spec_flattened.fits"
        if os.path.exists(h_spec_flattened_file) : has_flattened_files[0] = True
        else : print("WARNING: missing flattened file: {}".format(h_spec_flattened_file))

        k_spec_flattened_file = reduced_data_dir  + k_red_basename + ".spec_flattened.fits"
        if os.path.exists(k_spec_flattened_file) : has_flattened_files[1] = True
        else : print("WARNING: missing flattened file: {}".format(k_spec_flattened_file))

        loc["H_SPEC_FLATTENED"] = h_spec_flattened_file
        loc["K_SPEC_FLATTENED"] = k_spec_flattened_file
        
        loc["HAS_FLATTENED_FILES"] = has_flattened_files


    loc["POSITIVE_FLUX"] = positive_flux
    loc["FRAMETYPE"] = frametype
    loc["OBJTYPE"] = objtype
    
    loc["OUTPUT_EFILE"] = outdir + "SDCHK_{}_{:04d}e.fits".format(date,id)
    loc["OUTPUT_TFILE"] = outdir + "SDCHK_{}_{:04d}t.fits".format(date,id)

    return loc


def check_duplication(efiles, list_of_efiles) :
    
    keys = ["H_RAW","K_RAW"]
    duplicated_entry = False
    
    for i in range(len(list_of_efiles)) :
        if efiles["H_RAW"] == list_of_efiles[i]["H_RAW"] and efiles["K_RAW"] == list_of_efiles[i]["K_RAW"] :
            duplicated_entry = True

    return duplicated_entry


def get_reduction_list(recipes, reduced_data_dir, raw_data_dir, outdir, sequence_length=0) :

    """
        Description: generate a list of files for extraction
        
        recipes : Dict, cotainer of information in the IGRINS recipe file
        reduced_data_dir: str, path to reduced data
        raw_data_dir: str, path to raw data
        outdir: str, path to output
        sequence_length: int, length of ..AB..BA.. sequence, default is 0 for all lenghts
    """

    loc = {}

    for i in range(len(recipes['OBJNAME'])) :
        
        red_id = int(recipes['GROUP1'][i])
        positive_flux = True
        objtype = recipes['OBJTYPE'][i]
        objname = recipes['OBJNAME'][i]
        
        if (objtype == 'TAR' and objname != 'SKY') and (len(recipes['OBSIDS'][i]) == sequence_length or sequence_length == 0) :

            for j in range(len(recipes['OBSIDS'][i])) :
                
                id = recipes['OBSIDS'][i][j]
                frametype = recipes['FRAMETYPES'][i][j]
                positive_flux = j % 2 == 0

                efiles = extraction_files(recipes['DATE'], red_id, id, positive_flux, frametype, objtype, raw_data_dir, reduced_data_dir, outdir)
                
                if np.all(efiles["HAS_REQUIRED_FILES"]) :
                    if objname not in loc.keys() :
                        loc[objname] = []
                        
                    if check_duplication(efiles, loc[objname]) :
                        print("WARNING: duplicated entry H_BASENAME={} K_BASENAME={}, skipping ... ".format(efiles["H_BASENAME"],efiles["K_BASENAME"]))
                    else :
                        loc[objname].append(efiles)

        elif (objtype == 'STD' and objname != 'SKY'):

            id = recipes['OBSIDS'][i][0]
            frametype = recipes['FRAMETYPES'][i][0]
            pos_efiles = extraction_files(recipes['DATE'], red_id, id, True, frametype, objtype, raw_data_dir, reduced_data_dir, outdir)
            
            id = recipes['OBSIDS'][i][1]
            frametype = recipes['FRAMETYPES'][i][1]
            neg_efiles = extraction_files(recipes['DATE'], red_id, id, False, frametype, objtype, raw_data_dir, reduced_data_dir, outdir)

            if np.all(pos_efiles["HAS_REQUIRED_FILES"]) and np.all(neg_efiles["HAS_REQUIRED_FILES"]) :
                if objname not in loc.keys() :
                    loc[objname] = []
                loc[objname].append(pos_efiles)
                loc[objname].append(neg_efiles)

    return loc


def get_profile(spec2d_data, plot=False) :

    loc = {}

    norders, nprof, nspix = len(spec2d_data), len(spec2d_data[0]), len(spec2d_data[0][0])

    midpoint = np.floor(nprof / 2)
    xpix = np.linspace(0,nprof,nprof,endpoint=False)
    profile = np.zeros(nprof)
    profile_orders = []
    for order in range(norders) :
        img = spec2d_data[order]
        with warnings.catch_warnings(record=True) as _:
            tmp_profile = np.nanmean(img,axis=1)
            nans = np.isnan(tmp_profile)
            tmp_profile[nans] = 0.
            profile_orders.append(tmp_profile)
            profile += tmp_profile

    diff = np.gradient(profile)
    midrange = xpix > midpoint - 5
    midrange &= xpix < midpoint + 5
    
    mpinmidrange = np.nanargmin(np.abs(diff)[midrange])
    midpoint = xpix[midrange][mpinmidrange]

    xmax = np.nanargmax(profile)
    xmin = np.nanargmin(profile)

    if xmax < midpoint and xmin > midpoint :
        pos_mask = xpix <= midpoint
        neg_mask = xpix >= midpoint
    elif xmax > midpoint and xmin < midpoint :
        neg_mask = xpix <= midpoint
        pos_mask = xpix >= midpoint
    else :
        print("ERROR: ymin={} and ymax={} fall at the same side of profile!".format(xmin, xmax))
        exit()
    
    if plot :
        nprofile = profile/np.nanmax(profile)
        plt.plot(np.array([xmin, xmax]), np.array([nprofile[xmin],nprofile[xmax]]), 'o', color='g')
        plt.plot(xpix, nprofile, ':', color='k')
        plt.plot(xpix[pos_mask],nprofile[pos_mask], '--', color='b', label="Positive")
        plt.plot(xpix[neg_mask],nprofile[neg_mask], '--', color='r', label="Negative")
        plt.legend()
        plt.xlabel("x-coord (pixel)")
        plt.ylabel("normalized flux")
        plt.show()
    
    loc["x"] = xpix
    loc["y"] = profile
    loc["pos_mask"] = pos_mask
    loc["neg_mask"] = neg_mask
    loc["profile_orders"] = profile_orders

    return loc


def fit_profile(img, var, f, prof, function='polynomial', fitorder=2, plot=False) :
    
    nprof, nspix = len(prof), len(f)
    
    x = np.linspace(0,nspix,nspix,endpoint=False)
    ydata, yvar = deepcopy(img), deepcopy(var)
        
    for i in range(nspix) :
        good = ydata[:,i] >= 0

        with warnings.catch_warnings(record=True) as _:
            median = np.nanmedian(ydata[good,i])
            medsig = np.nanmedian(np.abs(ydata[good,i] - median)) / 0.67449

            good &= np.sqrt(yvar[:,i]) < 2.0*medsig
            ydata[~good,i] *= np.nan

            ydata[:,i] /= f[i]
            yvar[:,i] /= f[i]*f[i]
    
    prof2D = np.zeros_like(img)

    for i in range(nprof) :
    
        if len(x[np.isfinite(ydata[i])]) :
            with warnings.catch_warnings(record=True) as _:
                # First fit a low order polynomial fora first iteration
                prof2D[i] = igrinslib.fit_continuum(x, ydata[i], function=function, order=fitorder, nit=5, rej_low=2.5, rej_high=2.5, grow=3, med_filt=1, percentile_low=0.,percentile_high=100.,min_points=300, xlabel="spectral pixels", ylabel="flux fraction", plot_fit=False, silent=True)
                medsig = np.nanmedian(np.abs(ydata[i] - prof2D[i])) / 0.67449
                good = ydata[i] > prof2D[i] - 5*medsig
                good &= ydata[i] < prof2D[i] + 5*medsig
                good &= np.sqrt(yvar[i]) < 2.0*medsig
        
                ydata[i,~good] *= np.nan
            
                if len(x[np.isfinite(ydata[i])]) == 0 :
                    prof2D[i] *= np.nan

        if len(x[np.isfinite(ydata[i])]) :
            with warnings.catch_warnings(record=True) as _:

                prof2D[i] = igrinslib.fit_continuum(x, ydata[i], function=function, order=fitorder, nit=2, rej_low=3.0, rej_high=3.0, grow=3, med_filt=1, percentile_low=0.,percentile_high=100.,min_points=300, xlabel="spectral pixels", ylabel="flux fraction", plot_fit=False, silent=True)

                sig = np.nanstd(ydata[i] - prof2D[i])
                good = ydata[i] > prof2D[i] - 3*sig
                good &= ydata[i] < prof2D[i] + 3*sig
                ydata[i,~good] *= np.nan
            
                if len(x[np.isfinite(ydata[i])]) == 0 :
                    prof2D[i] *= np.nan
                
        first_j, last_j = 0, nspix
        
        for j in range(nspix) :
            if np.isfinite(ydata[i][j]) and  ydata[i][j] > 0 :
                first_j = j
                break
        for j in range(nspix) :
            if np.isfinite(ydata[i][nspix-j-1]) and  ydata[i][nspix-j-1] > 0 :
                last_j = nspix-j-1
                break
        
        with warnings.catch_warnings(record=True) as _:
            prof2D[i][:first_j] *= np.nan
            prof2D[i][last_j:] *= np.nan

        if plot :
            plt.errorbar(x+i*nspix, ydata[i], yerr=np.sqrt(yvar[i]), fmt='.', alpha=0.3)
            plt.plot(x+i*nspix, prof2D[i],'k-', lw=2)
            
    if plot :
        plt.show()
    
    return prof2D



def extract_flux(spec2d_data, var2d_data, profile, optimal_extraction=True, positive_flux=True, niter=3, nsigclip=3, plot_fit_profile=False, plot=False, verbose=False) :

    norders, nspix = len(spec2d_data), len(spec2d_data[0][0])

    if positive_flux :
        pixmask = profile["pos_mask"]
        mult = 1.
    else :
        pixmask = profile["neg_mask"]
        mult = -1.
    
    flux, variance = [], []
    spix = np.linspace(0,nspix,nspix,endpoint=False)
    
    for order in range(norders) :
        #prof = mult * profile["profile_orders"][order][pixmask]
        prof = mult * profile["y"][pixmask]
        prof[prof<0] = 0.
        prof = prof / np.nansum(prof)
    
        img = mult * spec2d_data[order][pixmask]
        var = var2d_data[order][pixmask]

        f = np.nansum(img,axis=0)
        v = np.nansum(var,axis=0)
        if plot :
            plt.errorbar(spix+order*nspix, f, yerr=np.sqrt(v), fmt='o', color='darkgrey', alpha=0.5)

        # First fit to profile
        try :
            prof2D = fit_profile(img, var, f, prof, function='polynomial', fitorder=2, plot=False)
        except :
            prof2D = np.zeros_like(img)
            for i in range(nspix) :
                prof2D[:,i] = prof
        
        if optimal_extraction :
            for iter in range(niter) :
                if verbose :
                    print("Optimal extraction of order={} iter={}".format(order,iter))
                
                try :
                    if iter == niter-1 :
                        prof2D = fit_profile(img, var, f, prof, function='polynomial', fitorder=5, plot=plot_fit_profile)
                    else :
                        prof2D = fit_profile(img, var, f, prof, function='polynomial', fitorder=3, plot=False)
                except :
                    print("WARNING: could not fit profile, skipping ...")
                    
                num, den = np.zeros_like(spix), np.zeros_like(spix)
            
                for i in range(len(prof)) :
                    mask = (img[i] - f * prof2D[i])**2 < nsigclip * nsigclip * v

                    num[mask] += img[i][mask] * prof2D[i][mask] / var[i][mask]
                    den[mask] += prof2D[i][mask] * prof2D[i][mask] / var[i][mask]
    
                den[den==0] = np.nan

                f = num / den
                v = 1. / den
            if plot :
                plt.errorbar(spix+order*nspix, f, yerr=np.sqrt(v), fmt='.', color='darkgreen')

        flux.append(f)
        variance.append(v)

    if plot :
        plt.ylabel("flux")
        plt.xlabel("pixel + npix x order number")
        plt.show()
        
    return flux, variance


def extract_spectrum (efiles, wave, optimal_extraction=True, output="", plot=False, verbose=False) :
    loc = {}
    
    loc["efiles"] = efiles
    
    header = fits.getheader(efiles["H_RAW"],0)
    header["BAND"] = 'HK'
    header["OBJTYPE"] = efiles["OBJTYPE"]
    
    # Coordinates of Gemini South
    longitude = 289.2633067
    latitude = 19.8238
    altitude = 2722.
    
    # Coordinates of HD 12345 (J2000)
    ra2000 = header["OBJRA"]
    dec2000 = header["OBJDEC"]
    objepoch = header["OBJEPOCH"]
    
    # (Mid-)Time of observation
    jd = (header["JD-END"] + header["JD-OBS"]) / 2

    # Calculate barycentric correction (debug=True show
    # various intermediate results)
    corr, hjd = pyasl.helcorr(longitude, latitude, altitude, ra2000, dec2000, jd, debug=False)

    #print("Barycentric correction [km/s]: ", corr)
    #print("Heliocentric Julian day: ", hjd)

    header.set('OBSLAT', latitude, 'Obs latitude [deg]')
    header.set('OBSLONG', longitude, 'Obs longitude [deg, E is positive]')
    header.set('OBSALT', altitude, 'Observatory altitude [m]')
    header.set('JD', jd, 'Average julian date')
    header.set('BERV', corr, 'Barycentric velocity correction [km/s]')
    header.set('HJD', hjd, 'Heliocentric Julian date')
    header.set('AIRMASS', (header["AMSTART"] + header["AMEND"]) / 2, 'Average airmass over exposure')

    # EXTRACT H-Band data
    h_spec2d_data = fits.getdata(efiles["H_SPEC2D"],0)
    h_var2d_data = fits.getdata(efiles["H_VAR2D"],0)
    h_profile = get_profile(h_spec2d_data, plot=plot)
    h_flux, h_variance = extract_flux(h_spec2d_data, h_var2d_data, h_profile, optimal_extraction=optimal_extraction, positive_flux=efiles["POSITIVE_FLUX"], plot_fit_profile=False, plot=plot, verbose=verbose)
    
    # EXTRACT K-Band data
    k_spec2d_data = fits.getdata(efiles["K_SPEC2D"],0)
    k_var2d_data = fits.getdata(efiles["K_VAR2D"],0)
    k_profile = get_profile(k_spec2d_data, plot=plot)
    k_flux, k_variance = extract_flux(k_spec2d_data, k_var2d_data, k_profile, optimal_extraction=optimal_extraction, positive_flux=efiles["POSITIVE_FLUX"], plot_fit_profile=False, plot=plot, verbose=verbose)

    flux, variance = [], []
    for j in range(28) :
        flux.append(h_flux[27-j])
        variance.append(h_variance[27-j])
    for j in range(26) :
        flux.append(k_flux[25-j])
        variance.append(k_variance[25-j])

    flux = np.array(flux, dtype='float')
    variance = np.array(variance, dtype='float')

    loc["HEADER"] = header
    loc["WAVE"] = wave
    loc["FLUX"] = flux
    loc["VARIANCE"] = wave

    # same primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)
    
    hdu_wave = fits.ImageHDU(data=wave, header=header, name='wave')
    hdu_flux = fits.ImageHDU(data=flux, header=header, name='flux')
    hdu_variance = fits.ImageHDU(data=variance, header=header, name='variance')

    hdus = [primary_hdu, hdu_wave, hdu_flux, hdu_variance]

    if efiles["OBJTYPE"] == "STD" and np.all(efiles["HAS_FLATTENED_FILES"]) :
    
        h_spec_flattened = fits.getdata(efiles["H_SPEC_FLATTENED"],0)
        h_fitted_continuum = fits.getdata(efiles["H_SPEC_FLATTENED"],2)
        h_mask = fits.getdata(efiles["H_SPEC_FLATTENED"],3)
        h_a0v_norm = fits.getdata(efiles["H_SPEC_FLATTENED"],4)
        h_model_teltrans = fits.getdata(efiles["H_SPEC_FLATTENED"],5)
        
        k_spec_flattened = fits.getdata(efiles["K_SPEC_FLATTENED"],0)
        k_fitted_continuum = fits.getdata(efiles["K_SPEC_FLATTENED"],2)
        k_mask = fits.getdata(efiles["K_SPEC_FLATTENED"],3)
        k_a0v_norm = fits.getdata(efiles["K_SPEC_FLATTENED"],4)
        k_model_teltrans = fits.getdata(efiles["K_SPEC_FLATTENED"],5)
        
        spec_flattened, fitted_continuum= [], []
        a0v_norm, model_teltrans= [], []
        
        for j in range(28) :
            spec_flattened.append(h_spec_flattened[27-j])
            fitted_continuum.append(h_fitted_continuum[27-j])
            a0v_norm.append(h_a0v_norm[27-j])
            model_teltrans.append(h_model_teltrans[27-j])
        for j in range(26) :
            spec_flattened.append(k_spec_flattened[25-j])
            fitted_continuum.append(k_fitted_continuum[25-j])
            a0v_norm.append(k_a0v_norm[25-j])
            model_teltrans.append(k_model_teltrans[25-j])
            
        hdu_spec_flattened = fits.ImageHDU(data=spec_flattened, header=header, name='SPEC_FLATTENED')
        hdu_fitted_continuum = fits.ImageHDU(data=fitted_continuum, header=header, name='FITTED_CONTINUUM')
        hdu_a0v_norm = fits.ImageHDU(data=a0v_norm, header=header, name='A0V_NORM')
        hdu_model_teltrans = fits.ImageHDU(data=model_teltrans, header=header, name='MODEL_TELTRANS')

        hdus.append(hdu_spec_flattened)
        hdus.append(hdu_fitted_continuum)
        hdus.append(hdu_a0v_norm)
        hdus.append(hdu_model_teltrans)

    hdu_list = fits.HDUList(hdus)
    
    if verbose :
        print("Saving H+K spectrum to file:",efiles["OUTPUT_EFILE"])
        
    hdu_list.writeto(efiles["OUTPUT_EFILE"], overwrite=True)

    return loc


def extract_spectra(recipes, reduced_data_dir, raw_data_dir, outdir=".", optimal_extraction=False, skip=False) :

    wave = get_wavelengths(reduced_data_dir)

    extraction_queue = get_reduction_list(recipes, reduced_data_dir, raw_data_dir, outdir, sequence_length=2)

    for target in extraction_queue.keys() :
        
        nspectra = len(extraction_queue[target])
    
        print("Starting extraction for target {} with N={} spectra".format(target,nspectra))
    
        for i in range(nspectra) :
            efiles = extraction_queue[target][i]

            print("Extracting H+K spectrum {}/{} raw H file:{} frametype={} positive={} objtype={}".format(i+1,nspectra,efiles["H_RAW"],efiles["FRAMETYPE"],efiles["POSITIVE_FLUX"],efiles["OBJTYPE"]))

            if os.path.exists(efiles["OUTPUT_EFILE"]) and skip :
                print("File: {} already exists, skipping ... ".format(efiles["OUTPUT_EFILE"]))
                continue
            try :
                spectrum = extract_spectrum(efiles, wave, optimal_extraction=optimal_extraction, plot=False, verbose=False)
            except :
                print("WARNING: could not extract spectrum, skipping ...")
                continue


#-- end of spirou_ccf routine
parser = OptionParser()
parser.add_option("-r", "--recipe_file", dest="recipe_file", help="Recipe file name",type='string',default="")
parser.add_option("-w", "--raw_data_dir", dest="raw_data_dir", help="Raw data directory",type='string',default="")
parser.add_option("-d", "--reduced_data_dir", dest="reduced_data_dir", help="Reduced data directory",type='string',default="")
parser.add_option("-o", "--output_dir", dest="output_dir", help="Output directory",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h igrins_extract_spectra.py")
    sys.exit(1)

if options.verbose:
    print('Recipe file name: ', options.recipe_file)
    print('Raw data directory: ', options.raw_data_dir)
    print('Reduced data directory: ', options.reduced_data_dir)
    print('Output directory: ', options.output_dir)

recipes = read_recipe_file(options.recipe_file)

extract_spectra(recipes, options.reduced_data_dir, options.raw_data_dir, outdir = options.output_dir, optimal_extraction=True, skip=True)
