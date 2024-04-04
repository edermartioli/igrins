# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 16 2021
    
    Description: utilities for spirou data reduction
    
    @authors:  Eder Martioli <emartioli@lna.br>, <martioli@iap.fr>
    
    Laboratorio Nacional de Astrofisica, Brazil
    Institut d'Astrophysique de Paris, France

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
import numpy as np

from copy import deepcopy
import matplotlib.pyplot as plt

import astropy.io.fits as fits
from scipy import constants

import igrinslib

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy.interpolate import interp1d
from scipy import optimize
import scipy.interpolate as sint
import scipy.signal as sig
import ccf_lib, ccf2rv

from astropy.convolution import convolve, Gaussian1DKernel

import telluric_lib

from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms

NORDERS = 54

def load_array_of_igrins_spectra(inputdata, rvfile="", object_name="None", apply_berv=True, silent=True, convolve_spectra=False, to_resolution = 30000, plot_diagnostics=False, plot=False, verbose=False) :

    loc = {}
    loc["input"] = inputdata

    if silent :
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    if rvfile == "":
        #print("WARNING: RV file has not been provided ...")
        #print("*** Setting all source RVs equal to zero ***")
        rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
    else :
        loc["rvfile"] = rvfile

        if verbose :
            print("Reading RVs from file:",rvfile)
        
        rvbjds, rvs, rverrs = read_rvdata(rvfile)
        
        if len(rvs) != len(inputdata):
            if verbose :
                print("WARNING: size of RVs is different than number of input *t.fits files")
                print("*** Ignoring input RVs and setting all source RVs equal to zero ***")
            rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
        else :
            for i in range(len(rvs)):
                hdr = fits.getheader(inputdata[i],1)
                print(i, inputdata[i], hdr['BJD'], rvbjds[i], rvs[i], rverrs[i])
    #---
    loc["source_rv"] = np.nanmedian(rvs)

    spectra = []
    speed_of_light_in_kps = constants.c / 1000.

    if plot_diagnostics :
        bjd, snr, airmass, berv = [], [], [], []

    for i in range(len(inputdata)) :
        
        spectrum = igrinslib.load_spectrum(inputdata[i])
        
        # set source RVs
        spectrum['FILENAME'] = inputdata[i]
        spectrum["source_rv"] = rvs[i]
        spectrum["rvfile"] = rvfile
        spectrum['RV'] = rvs[i]
        spectrum['RVERR'] = rverrs[i]

        out_wl, out_flux, out_fluxerr, out_order = [], [], [], []
        out_blaze = [], []
        wl_sf, vels = [], []

        hdr = spectrum["header"]

        if object_name not in hdr["OBJECT"] :
            print("Object name {} does not match object={}, skipping spectrum ...".format(object_name,hdr["OBJECT"]))
            continue

        # get DETECTOR GAIN and READ NOISE from header
        gain, rdnoise = hdr['GAIN'], hdr['RDNOISE']

        # Coordinates of Gemini South
        #longitude = hdr['OBSLONG']
        #latitude = hdr['OBSLAT']
        #altitude = hdr['OBSALT']
        
        # E. Martioli 25/02/2022 - The fix below is because the extracted images have inverted LAT<->LON, this bug has been fixed in igrins_extract_spectra, but needs to run again on AUMIC data
        longitude = 289.2633067
        latitude = 19.8238
        altitude = 2722.
    
        gemini_south = EarthLocation.from_geodetic(lat=latitude*u.deg, lon=longitude*u.deg, height=altitude*u.m)
        
        obj_ra, obj_dec = hdr['OBJRA'], hdr['OBJDEC']
        sc = SkyCoord(ra=obj_ra*u.deg, dec=obj_dec*u.deg)
        
        spectrum['JD_mid'] = hdr['JD'] + (hdr['JD-END'] - hdr['JD-OBS']) / 2.
        spectrum['MJDMID'] = hdr['MJD-OBS'] + (hdr['JD-END'] - hdr['JD-OBS']) / 2.

        obstime=Time(spectrum['JD_mid'], format='jd', scale='utc', location=gemini_south)
        barycorr = sc.radial_velocity_correction(obstime=obstime)

        hdr['BJD_MID'] = (obstime.tcb.jd,"Barycentric Julian Date")
        spectrum['DATE'] = hdr['DATE-OBS']
        spectrum['BJD_mid'] = obstime.tcb.jd
        spectrum['BERV'] = barycorr.to(u.km/u.s).value
        spectrum['airmass'] = (hdr['AMSTART'] + hdr['AMEND']) / 2
        spectrum['exptime'] = hdr['EXPTIME']

        wave = deepcopy(spectrum["wl"])
        fluxes = deepcopy(spectrum["flux"])
        fluxvar = deepcopy(spectrum["variance"])

        # Estimate signal-to-noise
        max_flux = []
        for order in range(len(wave)) :
            finite = np.isfinite(fluxes[order])
            if len(fluxes[order][finite]) :
                max_flux.append(np.percentile(fluxes[order][finite],99))
        mean_flux = np.nanmean(np.array(max_flux))
        maxsnr = mean_flux / np.sqrt(mean_flux + (rdnoise * rdnoise / gain * gain))

        spectrum['SNR'] = maxsnr
        hdr['SNR'] = (maxsnr,"Maximum signal-to-noise ratio")
        
        if plot_diagnostics :
            if i == 0 :
                objectname = hdr['OBJECT']
            bjd.append(spectrum['BJD_mid'])
            snr.append(maxsnr)
            airmass.append(spectrum['airmass'])
            berv.append(spectrum['BERV'])

        if verbose :
            print("Spectrum ({0}/{1}): {2} OBJ={3} BJD={4:.6f} SNR={5:.1f} EXPTIME={6:.0f}s BERV={7:.3f} km/s".format(i+1,len(inputdata)-1,inputdata[i],hdr['OBJECT'],spectrum['BJD_mid'],maxsnr,hdr['EXPTIME'],spectrum['BERV']))

        for order in range(len(wave)) :
            
            wl = deepcopy(wave[order])
            flux = deepcopy(fluxes[order])
            fluxerr = deepcopy(np.sqrt(fluxvar[order]))

            wlc = 0.5 * (wl[0] + wl[-1])

            if convolve_spectra :
                wl, flux, fluxerr = convolve_spectrum(wl, flux, fluxerr, to_resolution, from_resolution=None)
                zeros = flux == 0
                fluxerr[zeros] = np.nan
                flux[zeros] = np.nan
                
            if plot :
                p = plt.plot(wl, flux)
                color = p[0].get_color()
                plt.plot(wl, flux, color=color, lw=0.3, alpha=0.6)
                
            order_vec = np.full_like(wl,float(order))

            if apply_berv :
                vel_shift = spectrum['RV'] - spectrum['BERV']
            else :
                vel_shift = spectrum['RV']

            # relativistic calculation
            wl_stellar_frame = wl * np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
            #wl_stellar_frame = wl / (1.0 + vel_shift / speed_of_light_in_kps)
            vel = speed_of_light_in_kps * ( wl_stellar_frame / wlc - 1.)

            out_wl.append(wl)
            out_flux.append(flux)
            out_fluxerr.append(fluxerr)
            out_order.append(order_vec)
            wl_sf.append(wl_stellar_frame)
            vels.append(vel)
                
        if plot :
            plt.xlabel(r"wavelength [nm]")
            plt.xlabel(r"flux")
            plt.show()
            exit()

        spectrum['wl_sf'] = wl_sf
        spectrum['vels'] = vels

        spectrum['order'] = out_order
        spectrum['wl'] = out_wl
        spectrum['flux'] = out_flux
        spectrum['fluxerr'] = out_fluxerr

        spectra.append(spectrum)

    loc["spectra"] = spectra

    if plot_diagnostics :
        bjd = np.array(bjd)
        snr = np.array(snr)
        airmass = np.array(airmass)
        berv = np.array(berv)
        
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('{} spectra of {}'.format(len(inputdata), objectname))
        axs[0].plot(bjd, snr, '-', color="orange",label="SNR")
        axs[0].set_ylabel('SNR')
        axs[0].legend()

        axs[1].plot(bjd, airmass, '--', color="olive",label="Airmass")
        axs[1].set_ylabel('Airmass')
        axs[1].legend()

        axs[2].plot(bjd, berv, ':', color="darkblue",label="BERV")
        axs[2].set_xlabel('BJD')
        axs[2].set_ylabel('BERV [km/s]')
        axs[2].legend()
        
        plt.show()
    
    return loc


def get_wlmin_wlmax(spectra, edge_size=100) :

    speed_of_light_in_kps = constants.c / 1000.
    
    # find minimum and maximum wavelength for valid (not NaN) data
    wlmin, wlmax = np.full(NORDERS,-1e20), np.full(NORDERS,+1e20)

    for order in range(NORDERS) :
        for i in range(spectra['nspectra']) :
            minwl_sf = np.nanmin(spectra["waves_sf"][order][i])
            maxwl_sf = np.nanmax(spectra["waves_sf"][order][i])

            if minwl_sf > wlmin[order] :
                wlmin[order] = minwl_sf
            
            if maxwl_sf < wlmax[order] :
                wlmax[order] = maxwl_sf

    spectra["wlmin"] = wlmin
    spectra["wlmax"] = wlmax

    return spectra


def get_spectral_data(array_of_spectra, ref_index=0, edge_size=100, verbose=False) :
    """
        Description: this function loads an array of spectra
        Input:
            array_of_spectra : list, [spectrum1, spectrum2, ...]
            ref_index : int, index of reference spectrum
            edge_size : int, size of edge to be ignored in the data
        
        Output:
            loc : dict, container for the following data:
            loc['nspectra'] : int, number of spectra in the input array
            
            loc["header"] : fits header, header of base spectra
            loc["filenames"] : list, file names of spectra

            loc["bjds"] : float array (1 x nspectra), barycentric julian dates of spectra
            loc["airmasses"] : float array  (1 x nspectra), airmasses of spectra
            loc["rvs"] : float array (1 x nspectra), radial velocities of spectra
            loc["rverrs"] : float array (1 x nspectra), RV errors of spectra
            loc["bervs"] : float array (1 x nspectra), barycentric earth RV of spectra

            loc["snrs"] : 2D float array (norders x nspectra), SNR
            loc["orders"] : 2D float array (norders x nspectra), order number
            loc["waves"] : 2D float array (norders x nspectra), wavelength in nm
            loc["waves_sf"] : 2D float array (norders x nspectra), star frame wavelength in nm
            loc["vels"]  : 2D float array (norders x nspectra), velocity in km/s

            loc["fluxes"] : 2D float array (norders x nspectra), flux
            loc["fluxerrs"] : 2D float array (norders x nspectra), flux error
            
            loc["wlmin"] : float, minimum wavelength in nm
            loc["wlmax"] : float, maximum wavelength in nm

    """
    if verbose :
        print("Loading data")
    
    loc = {}

    spectra = array_of_spectra["spectra"]

    filenames, dates = [], []
    bjds, airmasses, rvs, rverrs, bervs = [], [], [], [], []

    ref_spectrum = spectra[ref_index]

    nspectra = len(spectra)
    loc['nspectra'] = nspectra
    snrs = []
    waves, waves_sf, vels = [], [], []
    fluxes, fluxerrs, orders = [], [], []
    hdr = []

    for order in range(NORDERS) :
        snrs.append([])
        orders.append([])
        waves.append([])
        waves_sf.append([])
        vels.append([])
        fluxes.append([])
        fluxerrs.append([])

    for i in range(nspectra) :
        
        spectrum = spectra[i]

        if verbose:
            print("Loading input spectrum {0}/{1} : {2}".format(i,nspectra-1,spectrum['FILENAME']))
            
        filenames.append(spectrum['FILENAME'])
        hdr.append(spectrum['header'])
        dates.append(spectrum['DATE'])
            
        bjds.append(spectrum['BJD_mid'])
        airmasses.append(spectrum['airmass'])
        rvs.append(spectrum['RV'])
        rverrs.append(spectrum['RVERR'])
        bervs.append(spectrum['BERV'])
                    
        for order in range(len(spectrum['wl'])) :
            mean_snr = np.nanmean(spectrum['flux'][order] / spectrum['fluxerr'][order])
            
            snrs[order].append(mean_snr)

            orders[order].append(spectrum['order'][order])

            waves[order].append(spectrum['wl'][order])
            waves_sf[order].append(spectrum['wl_sf'][order])
            vels[order].append(spectrum['vels'][order])

            fluxes[order].append(spectrum['flux'][order])
            fluxerrs[order].append(spectrum['fluxerr'][order])

    bjds  = np.array(bjds, dtype=float)
    airmasses  = np.array(airmasses, dtype=float)
    rvs  = np.array(rvs, dtype=float)
    rverrs  = np.array(rverrs, dtype=float)
    bervs  = np.array(bervs, dtype=float)

    for order in range(NORDERS) :
        snrs[order] = np.array(snrs[order], dtype=float)
            
        orders[order]  = np.array(orders[order], dtype=float)

        waves[order]  = np.array(waves[order], dtype=float)
        waves_sf[order]  = np.array(waves_sf[order], dtype=float)
        vels[order]  = np.array(vels[order], dtype=float)

        fluxes[order]  = np.array(fluxes[order], dtype=float)
        fluxerrs[order]  = np.array(fluxerrs[order], dtype=float)

    loc["header"] = hdr
    loc["filenames"] = filenames

    loc["bjds"] = bjds
    loc["airmasses"] = airmasses
    loc["rvs"] = rvs
    loc["rverrs"] = rverrs
    loc["bervs"] = bervs

    loc["snrs"] = snrs
    loc["orders"] = orders
    loc["waves"] = waves
    loc["waves_sf"] = waves_sf
    loc["vels"] = vels

    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs

    loc = get_wlmin_wlmax(loc, edge_size=edge_size)
    
    return loc


def get_gapfree_windows(spectra, max_vel_distance=3.0, min_window_size=120., fluxkey="fluxes", velkey="vels", wavekey="waves", verbose=False) :
    
    windows = []
    
    for order in range(NORDERS) :
        windows.append([])
    
    for order in range(NORDERS) :
        if verbose :
            print("Calculating windows with size > {0:.0f} km/s and with gaps < {1:.1f} km/s for order={2}".format(min_window_size,max_vel_distance, order))

        for i in range(spectra['nspectra']) :
        
            nanmask = np.isfinite(spectra[fluxkey][order][i])
            vels = spectra[velkey][order][i]
            wl = spectra[wavekey][order][i]

            if len(vels[nanmask]) > min_window_size / 4.0 :

                dv = np.abs(vels[nanmask][1:] - vels[nanmask][:-1])
            
                gaps = dv > max_vel_distance
            
                window_v_ends = np.append(vels[nanmask][:-1][gaps],vels[nanmask][-1])
                window_v_starts = np.append(vels[nanmask][0],vels[nanmask][1:][gaps])

                window_size = np.abs(window_v_ends - window_v_starts)
                good_windows = window_size > min_window_size

                window_wl_ends = np.append(wl[nanmask][:-1][gaps],wl[nanmask][-1])
                window_wl_starts = np.append(wl[nanmask][0],wl[nanmask][1:][gaps])

                loc_windows = np.array((window_wl_starts[good_windows],window_wl_ends[good_windows])).T
            else :
                loc_windows = np.array([])

            windows[order].append(loc_windows)

    # save window function
    spectra["windows"] = windows

    return spectra


def set_common_wl_grid(spectra, vel_sampling=2.0, verbose=False) :

    if "wlmin" not in spectra.keys() or "wlmax" not in spectra.keys():
        print("ERROR: function set_common_wl_grid() requires keywords wlmin and wlmax in input spectra, exiting.. ")
        exit()
    
    common_wl, common_vel = [], []
    speed_of_light_in_kps = constants.c / 1000.
    drv = 1.0 + vel_sampling / speed_of_light_in_kps
    drv_neg = 1.0 - vel_sampling / speed_of_light_in_kps

    np_min = 1e50
    
    for order in range(NORDERS) :
        
        wlmin = spectra["wlmin"][order]
        wlmax = spectra["wlmax"][order]
        
        wl_array = []
        wl = wlmin
        while wl < wlmax * drv_neg :
            wl *= drv
            wl_array.append(wl)
        wl_array = np.array(wl_array)
        
        wlc = (wl_array[0]+wl_array[-1])/2

        vels = speed_of_light_in_kps * ( wl_array / wlc - 1.)
        
        common_vel.append(vels)
        common_wl.append(wl_array)
    
        if len(wl_array) < np_min :
            np_min = len(wl_array)

    for order in range(NORDERS) :
        diff_size = len(common_wl[order]) - np_min
        half_diff_size = int(diff_size/2)
        
        if diff_size :
            common_vel[order] = common_vel[order][half_diff_size:np_min+half_diff_size]
            common_wl[order] = common_wl[order][half_diff_size:np_min+half_diff_size]

    spectra["common_vel"] = np.array(common_vel, dtype=float)
    spectra["common_wl"] = np.array(common_wl, dtype=float)

    return spectra


# function to interpolate spectrum
def interp_spectrum(wl_out, wl_in, flux_in, good_windows, kind='cubic') :

    flux_out = np.full_like(wl_out, np.nan)

    for w in good_windows :

        mask = wl_in >= w[0]
        mask &= wl_in <= w[1]

        wl_in_copy = deepcopy(wl_in)
        flux_in_copy = deepcopy(flux_in)

        # create interpolation function for input data
        f = interp1d(wl_in_copy[mask], flux_in_copy[mask], kind=kind)

        wl1, wl2 = w[0], w[1]

        if wl1 < wl_in[mask][0] :
            wl1 = wl_in[mask][0]
        if wl2 > wl_in[mask][-1] :
            wl2 = wl_in[mask][-1]

        out_mask = wl_out > wl1
        out_mask &= wl_out < wl2

        # interpolate data
        flux_out[out_mask] = f(wl_out[out_mask])

    return flux_out


def resample_and_align_spectra(spectra, use_gp=True, interp_kind='cubic', plot=False, verbose=False) :

    if "common_wl" not in spectra.keys() :
        print("ERROR: function resample_and_align_spectra() requires keyword common_wl in input spectra, exiting.. ")
        exit()
    
    aligned_waves = []
    
    sf_fluxes, sf_fluxerrs = [], []
    rest_fluxes, rest_fluxerrs = [], []

    for order in range(NORDERS) :
        aligned_waves.append([])

        sf_fluxes.append([])
        sf_fluxerrs.append([])
        rest_fluxes.append([])
        rest_fluxerrs.append([])

    for order in range(NORDERS) :
        if verbose :
            print("Aligning all spectra to a common wavelength grid for order=", order)

        common_wl = spectra['common_wl'][order]
        
        for i in range(spectra['nspectra']) :
            if "windows" in spectra.keys() :
                windows = spectra["windows"][order][i]
            else :
                windows = [[common_wl[0],common_wl[-1]]]
            keep = np.isfinite(spectra["fluxes"][order][i])

            flux = spectra["fluxes"][order][i][keep]
            fluxerr = spectra["fluxerrs"][order][i][keep]
            
            wl_sf = spectra["waves_sf"][order][i][keep]
            wl_rest = spectra["waves"][order][i][keep]

            if use_gp :
                sf_flux, sf_fluxerr = interp_spectrum_using_gp(common_wl, wl_sf, flux, fluxerr, windows, verbose=False, plot=False)
                rest_flux, rest_fluxerr = interp_spectrum_using_gp(common_wl, wl_rest, flux, fluxerr, windows, verbose=False, plot=False)
            else :
                sf_flux = interp_spectrum(common_wl, wl_sf, flux, windows, kind=interp_kind)
                sf_fluxerr = interp_spectrum(common_wl, wl_sf, fluxerr, windows, kind=interp_kind)
                rest_flux = interp_spectrum(common_wl, wl_rest, flux, windows, kind=interp_kind)
                rest_fluxerr = interp_spectrum(common_wl, wl_rest, fluxerr, windows, kind=interp_kind)

            aligned_waves[order].append(common_wl)

            sf_fluxes[order].append(sf_flux)
            sf_fluxerrs[order].append(sf_fluxerr)
            rest_fluxes[order].append(rest_flux)
            rest_fluxerrs[order].append(rest_fluxerr)

            if plot :
                p_rest = plt.errorbar(wl_rest, flux, yerr=fluxerr, fmt=".", lw=0.3, alpha=0.6)
                color_rest = p_rest[0].get_color()
                plt.fill_between(common_wl, rest_flux+rest_fluxerr, rest_flux-rest_fluxerr, color=color_rest, alpha=0.3, edgecolor="none")
                
                p_sf = plt.errorbar(wl_sf, flux, yerr=fluxerr, fmt=".", lw=0.3, alpha=0.6)
                color_sf = p_sf[0].get_color()
                plt.fill_between(common_wl, sf_flux+sf_fluxerr, sf_flux-sf_fluxerr, color=color_sf, alpha=0.3, edgecolor="none")

                for w in windows:
                    plt.vlines(w, [np.min(flux),np.min(flux)], [np.max(flux),np.max(flux)], color = "r", ls="--")
    if plot :
        plt.show()

    spectra["aligned_waves"] = aligned_waves

    spectra["sf_fluxes"] = sf_fluxes
    spectra["sf_fluxerrs"] = sf_fluxerrs
    spectra["rest_fluxes"] = rest_fluxes
    spectra["rest_fluxerrs"] = rest_fluxerrs

    return spectra


def reduce_spectra(spectra, nsig_clip=0.0, combine_by_median=False, subtract=True,  output="", fluxkey="fluxes", fluxerrkey="fluxerrs", wavekey="wl", update_spectra=False, plot=False, verbose=False) :
    
    signals, ref_snrs, noises,  orders = [], [], [], []
    rel_noises = []
    snrs, snrs_err = [], []
    template = []

    if subtract :
        sub_flux_base = 0.0
    else :
        sub_flux_base = 1.0
    
    for order in range(NORDERS) :
    #for order in range(30,31) :

        if verbose:
            print("Reducing spectra for order {0} / {1} ...".format(order,NORDERS))

        # get mean signal before applying flux corrections
        median_signals = []
        for i in range(spectra['nspectra']) :
            median_signals.append(np.nanmedian(spectra[fluxkey][order][i]))
        median_signals = np.array(median_signals)

        # 1st pass - to build template for each order and subtract out all spectra by template
        order_template = calculate_template(spectra[fluxkey][order], wl=spectra[wavekey][order], fit=True, median=combine_by_median, subtract=True, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # Recover fluxes already shifted and re-scaled to match the template
        fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base

        # 2nd pass - to build template from calibrated fluxes
        order_template = calculate_template(fluxes, wl=spectra[wavekey][order], fit=True, median=combine_by_median, subtract=subtract, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # apply sigma-clip using template and median dispersion in time as clipping criteria
        # bad values can either be replaced by the template values, by interpolated values or by NaNs
        if nsig_clip > 0 :
            order_template = sigma_clip(order_template, nsig=nsig_clip, interpolate=False, replace_by_model=False, sub_flux_base=sub_flux_base, plot=False)
            #order_template = sigma_clip_remove_bad_columns(order_template, nsig=nsig_clip, plot=False)

        # Recover fluxes already shifted and re-scaled to match the template
        if subtract :
            fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base
        else:
            fluxes = order_template["flux_arr_sub"] * order_template["flux"]

        # 3rd pass - Calculate a final template combined by the mean
        order_template = calculate_template(fluxes, wl=spectra[wavekey][order], fit=True, median=combine_by_median, subtract=subtract, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # save number of spectra in the time series
        order_template['nspectra'] = spectra['nspectra']

        # save order flag
        order_template['order'] = order

        rel_noise, noise = [], []
        for i in range(spectra['nspectra']) :
            noise.append(np.nanstd(order_template["flux_residuals"][i]))
            rel_noise.append(np.nanstd(order_template["flux_arr_sub"][i]))

        noise = np.array(noise)
        rel_noise = np.array(rel_noise)
        
        m_signal = np.nanmedian(median_signals)
        m_ref_snr = np.nanmean(spectra["snrs"][order])
        #m_noise = np.nanmean(noise)
        m_noise = np.nanmean(order_template["fluxerr"])

        #m_snr = np.nanmean(median_signals/noise)
        m_snr = m_signal/m_noise
        sig_snr = np.nanstd(median_signals/noise)
        m_rel_noise = np.nanmedian(rel_noise)

        #if verbose :
        #    print("Order {0}: median flux = {1:.2f}; median noise = {2:.2f};  SNR={3:.2f}".format(order, m_signal, m_noise, m_signal/m_noise))

        signals.append(m_signal)
        noises.append(m_noise)
        ref_snrs.append(m_ref_snr)
        snrs.append(m_snr)
        snrs_err.append(sig_snr)
        orders.append(order)

        order_template["median_signal"] = median_signals
        order_template["ref_snr"] = spectra["snrs"][order]
        order_template["noise"] = noise
        order_template["mean_snr"] = median_signals/noise
        order_template["rel_noise"] = m_rel_noise

        template.append(order_template)
    
        if update_spectra :
            # Recover fluxes already shifted and re-scaled to match the template
            if subtract :
                fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base
            else:
                fluxes = order_template["flux_arr_sub"] * order_template["flux"]
                
            fluxerr = np.zeros_like(order_template["flux_arr_sub"]) + order_template["fluxerr"]

            for i in range(spectra['nspectra']) :
                spectra[fluxkey][order][i] = fluxes[i]
                spectra[fluxerrkey][order][i] = fluxerr[i]

    signals = np.array(signals)
    ref_snrs = np.array(ref_snrs)
    noises = np.array(noises)
    snrs, snrs_err = np.array(snrs), np.array(snrs_err)
    orders = np.array(orders)

    if plot :
        plt.errorbar(orders, snrs, yerr=snrs_err, fmt='o', color='k', label="Measured noise")
        plt.plot(orders, snrs, 'k-', lw=0.5)
        plt.plot(orders, ref_snrs, '--', label="Photon noise")
        plt.xlabel(r"Spectral order")
        plt.ylabel(r"Signal-to-noise ratio (SNR)")
        plt.legend()
        plt.show()

    if output != "":
        np.save(output, template)

    return template


def reduce_spectra_wavecal(spectra, combine_by_median=False, subtract=True,  output="", fluxkey="fluxes", fluxerrkey="fluxerrs", wavekey="wl", update_spectra=False, plot=False, verbose=False) :
    
    template = []

    if subtract :
        sub_flux_base = 0.0
    else :
        sub_flux_base = 1.0
    
    for order in range(NORDERS) :
    #for order in range(7,8) :
    
        if verbose:
            print("Reducing spectra w/ wavecal for order {0} / {1} ...".format(order,NORDERS))
            
        # 1st pass - to build template for each order and subtract out all spectra by template
        order_template = calculate_template_wavecal(spectra[fluxkey][order], spectra[fluxerrkey][order], wl=spectra[wavekey][order], fit=True, median=combine_by_median, subtract=True, sub_flux_base=sub_flux_base, verbose=False, plot=False)

        # save number of spectra in the time series
        order_template['nspectra'] = spectra['nspectra']
        # save order flag
        order_template['order'] = order
        template.append(order_template)

        if update_spectra :
            fluxes = fluxerr = None
            # Recover fluxes already shifted and re-scaled to match the template
            if subtract :
                fluxes = order_template["flux_arr_sub"] + order_template["flux"] - sub_flux_base
            else:
                fluxes = order_template["flux_arr_sub"] * order_template["flux"]
                
            fluxerr = np.zeros_like(order_template["flux_arr_sub"]) + order_template["fluxerr"]

            for i in range(spectra['nspectra']) :
                spectra[fluxkey][order][i] = fluxes[i]
                spectra[fluxerrkey][order][i] = fluxerr[i]

    return template

import emcee, corner


#################################################################################################
def calculate_template_wavecal(flux_arr, fluxerr_arr, wl=[], fit=False, median=True, subtract=False, sub_flux_base=1.0, min_npoints=100, verbose=False, plot=False, run_mcmc=False, pfilename=""):
    """
        Compute the mean/median template spectrum along the time axis and divide/subtract
        each exposure by the mean/median
        
        Inputs:
        - flux_arr: 2D flux matrix (N_exposures, N_wavelengths)
        - wl: 1D wavelength array (N_wavelengths)
        - fit: boolean to fit median spectrum to each observation before normalizing it
        - median: boolean to calculate median instead of mean
        - subtract: boolean to subtract instead of dividing out spectra by the mean/median template

        Outputs:
        - loc: python dict containing all products
    """
    
    speed_of_light_in_kps = constants.c / 1000.
    speed_of_light_in_mps = constants.c

    loc = {}

    loc["fit"] = fit
    loc["median"] = median
    loc["subtract"] = subtract
    loc["pfilename"] = pfilename

    if len(wl) == 0:
        x = np.arange(len(flux_arr[0]))
    else :
        x = wl

    if verbose :
        print("Calculating template out of {0} input spectra".format(len(flux_arr)))
    
    if median :
        # median combine
        flux_template = np.nanmedian(flux_arr,axis=0)
        flux_template_err = np.nanmedian(np.abs(flux_arr - flux_template),axis=0) / 0.67449
    else :
        # mean combine
        flux_template = np.nanmean(flux_arr,axis=0)
        flux_template_err = np.nanstd(flux_arr,axis=0)

    min_vshift, max_vshift = -20000,+20000  # in m/s
    min_shift, max_shift = -0.5,+0.5
    min_scale, max_scale = 0.3,1.5
    edge_size_kps = 20.0 # edge size in km/s
    xmean = np.nanmean(x)
    edgecut = edge_size_kps * xmean / speed_of_light_in_kps

    # set mask to consider only the spectral range with a valid template
    mask = (np.isfinite(flux_template)) & (np.isfinite(flux_template_err))

    if fit and len(x[mask]) > min_npoints:
    
        # set x limits
        x0, xf = x[mask][0] + edgecut, x[mask][-1] - edgecut
    
        # Set up the GP model
        kernel = terms.RealTerm(log_a=np.log(np.var(flux_template[mask])), log_c=-np.log(10.0))
        gp = celerite.GP(kernel, mean=np.nanmean(flux_template[mask]), fit_mean=True)
        gp.compute(x[mask], flux_template_err[mask])
        # Fit for the maximum likelihood parameters
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", bounds=bounds, args=(flux_template[mask], gp))
        gp.set_parameter_vector(soln.x)

        sx = np.linspace(x0 - 2*edgecut, xf + 2*edgecut, num=len(x)+2*min_npoints)
        gp_template, var = gp.predict(flux_template[mask], sx, return_var=True)
        gp_template_err = np.sqrt(var)
        
        template = interp1d(sx, gp_template, kind='cubic', bounds_error=False, fill_value=np.nan)
        template_err = interp1d(sx, gp_template_err, kind='cubic', bounds_error=False, fill_value=np.nan)

        if plot :
            # Plot the data
            color = "#ff7f0e"
            plt.errorbar(x, flux_template, yerr=flux_template_err, fmt=".k", capsize=0)
            plt.plot(sx, gp_template, color=color)
            plt.fill_between(sx, gp_template-gp_template_err, gp_template+gp_template_err, color=color, alpha=0.3, edgecolor="none")
            plt.ylabel(r"Flux")
            plt.xlabel(r"wavelength [nm]")
            plt.title("GP interpolation")
            plt.show()


        flux_calib = []
        flux_fit = []
        
        vshift_arr = []
        scale_arr, shift_arr = [], []

        def flux_model (theta, wave) :
            #new_wave = (scale * (wave - xmean) + xmean) * (1.0 + shift / speed_of_light_in_mps)
            new_wave = wave * (1.0 + theta[0] / speed_of_light_in_mps)
            outmodel = template(new_wave) * theta[2] + theta[1]
            #outmodel, var = gp.predict(flux_template[mask], new_wave, return_var=True)
            return outmodel

        def log_likelihood(theta, xx, yy, yyerr):
            #shift, scale = theta
            model = flux_model(theta, xx)
            sigma2 = yyerr ** 2 + model ** 2
            return -0.5 * np.sum((yy - model) ** 2 / sigma2 + np.log(sigma2))

        def log_prior(theta):
            #shift, scale = theta
            #if min_shift < shift < max_shift and min_scale < scale < max_scale :
            if min_vshift < theta[0] < max_vshift and min_shift < theta[1] < max_shift and min_scale < theta[2] < max_scale :
                return 0.0
            return -np.inf

        def log_probability(theta, xx, yy, yyerr):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            logprob = lp + log_likelihood(theta, xx, yy, yyerr)
            if np.isnan(logprob) :
                return -np.inf
            else :
                return logprob
    
        for i in range(len(flux_arr)):
            if verbose :
                print("Calibrating wavelength of exposure # {}/{}".format(i+1,len(flux_arr)))
            
            keep = (np.isfinite(flux_arr[i])) & (np.isfinite(fluxerr_arr[i]))
            keep &= (x > x0) & (x < xf)

            wl, flux, fluxerr = x[keep], flux_arr[i][keep], fluxerr_arr[i][keep]

            initial = np.array([0.001,0.001,1.])
            #initial = np.array([0.001])
            
            if run_mcmc :
                niter = 3000
                nbins = 30
                use_mode = True
                pos = initial + 1e-5 * np.random.randn(32, 3)
                nwalkers, ndim = pos.shape

                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(wl, flux, fluxerr))
            
                sampler.run_mcmc(pos, niter, progress=True);

                #tau = sampler.get_autocorr_time()
                #print("Autocorrelation time:",tau)
            
                flat_samples = sampler.get_chain(discard=600, flat=True)
                
                pfit = efitlow = efithigh = np.array([])
                for j in range(ndim):
                    mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
                    q = np.diff(mcmc)
                    
                    hist, bin_edges = np.histogram(flat_samples[:,j], bins=nbins, range=(mcmc[1]-5*q[0],mcmc[1]+5*q[1]), density=True)
                    xcen = (bin_edges[:-1] + bin_edges[1:])/2
                    mode = xcen[np.argmax(hist)]
                    
                    if use_mode :
                        pfit = np.append(pfit,mode)
                    else :
                        pfit = np.append(pfit,mcmc[1])

                    efitlow = np.append(efitlow,q[0])
                    efithigh = np.append(efithigh,q[1])

                #print("Wavelength scaled by {:.3f}+{:.3f}-{:.3f} and shifted by {:.3f}+{:.3f}-{:.3f} m/s".format(pfit[1],efitlow[1],efithigh[1],pfit[0],efitlow[0],efithigh[0]))
                #fig = corner.corner(flat_samples, labels=["shift","scale"], quantiles=[0.16, 0.5, 0.84], truths=pfit, show_titles=True)
                if verbose :
                    print("Wavelength shifted by {:.3f}+{:.3f}-{:.3f} m/s".format(pfit[0],efitlow[0],efithigh[0]))
                    print("Flux shifted by {:.3f}+{:.3f}-{:.3f} ".format(pfit[1],efitlow[1],efithigh[1]))
                    print("Flux scaled by {:.3f}+{:.3f}-{:.3f}".format(pfit[2],efitlow[2],efithigh[2]))
                if plot :
                    fig = corner.corner(flat_samples, labels=["vshift","shift","scale"], quantiles=[0.16, 0.5, 0.84], truths=pfit, show_titles=True)
                    plt.show()
            else :
                # maximum likelihood estimation
                nll = lambda *args: -log_likelihood(*args)
                soln = minimize(nll, initial, args=(wl, flux, fluxerr))
                pfit = soln.x
                if verbose :
                    #print("Wavelength scaled by {} and shifted by {} m/s".format(pfit[1],pfit[0]))
                    print("Wavelength shifted by {:.3f} m/s".format(pfit[0]))
                    print("Flux shifted by {:.3f} ".format(pfit[1]))
                    print("Flux scaled by {:.3f} ".format(pfit[2]))

            flux_template_fit = flux_model(pfit, x)
            flux_fit.append(flux_template_fit)

            vshift_arr.append(pfit[0])
            shift_arr.append(pfit[1])
            scale_arr.append(pfit[2])
            
            #new_x = (pfit[1] * (x - xmean) + xmean) * (1.0 + pfit[0] / speed_of_light_in_mps)
            new_x = x * (1.0 + pfit[0] / speed_of_light_in_mps)
            finite = (np.isfinite(flux_arr[i])) & (np.isfinite(fluxerr_arr[i]))
            flux_calib_loc, fluxerr_calib_loc = np.full_like(x, np.nan), np.full_like(x, np.nan)
            
            flux_calib_loc[keep], fluxerr_calib_loc[keep] = interp_spectrum_using_gp(x[keep], new_x[finite], flux_arr[i][finite], fluxerr_arr[i][finite], [[x[0],x[-1]]], verbose=False, plot=False)

            flux_calib_loc[keep] = (flux_calib_loc[keep] - pfit[1]) / pfit[2]

            if plot :
                stemplate, var = gp.predict(flux_template[mask], x, return_var=True)
                stemplate_err = np.sqrt(var)
                # Plot the data
                color = "#ff7f0e"
                plt.errorbar(x[finite], flux_arr[i][finite], yerr=fluxerr_arr[i][finite], fmt='.', color='k', alpha=0.8, label="uncorrected flux")
                plt.errorbar(x, flux_calib_loc, yerr=fluxerr_calib_loc, fmt='.', color='r', alpha=0.8, label="corrected flux")
                plt.plot(x, stemplate, color=color, label="Template")
                plt.fill_between(x, stemplate-stemplate_err, stemplate+stemplate_err, color=color, alpha=0.3, edgecolor="none")
                plt.ylabel(r"Flux",fontsize=16)
                plt.xlabel(r"wavelength [nm]",fontsize=16)
                plt.legend(fontsize=16)
                plt.show()

            flux_calib.append(flux_calib_loc)

        loc["vshift"] = np.array(vshift_arr, dtype=float)
        loc["shift"] = np.array(shift_arr, dtype=float)
        loc["scale"] = np.array(scale_arr, dtype=float)

        flux_calib = np.array(flux_calib, dtype=float)
        flux_fit = np.array(flux_fit, dtype=float)

        # Compute median on all spectra along the time axis
        if median :
            flux_template_new = np.nanmedian(flux_calib,axis=0)
        else :
            flux_template_new = np.nanmean(flux_calib,axis=0)
            #flux_template_new = np.average(flux_calib,axis=0, weights=weights)

        flux_template = flux_template_new
        if subtract :
            flux_arr_sub = flux_calib - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_calib / flux_template

        residuals = flux_calib - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_calib
    else :
        # Divide or subtract each ccf by ccf_med
        if subtract :
            flux_arr_sub = flux_arr - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_arr / flux_template

        residuals = flux_arr - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_arr

    loc["flux"] = flux_template
    loc["fluxerr"] = flux_template_medsig
    loc["wl"] = x
    loc["flux_arr_sub"] = flux_arr_sub
    loc["flux_residuals"] = residuals
    loc["snr"] = flux_arr / flux_template_medsig

    loc["template_source"] = "data"
    
    template_nanmask = ~np.isnan(flux_template)
    template_nanmask &= ~np.isnan(flux_template_medsig)
    
    if len(flux_template_medsig[template_nanmask]) :
        loc["fluxerr_model"] = fit_continuum(x, flux_template_medsig, function='polynomial', order=5, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux error", plot_fit=False,silent=True)
    else :
        loc["fluxerr_model"] = np.full_like(x,np.nan)

    if plot :
        plot_template_products(loc, pfilename=pfilename)
    
    return loc



#################################################################################################
def calculate_template(flux_arr, wl=[], fit=False, median=True, subtract=False, sub_flux_base=1.0, min_npoints=100, verbose=False, plot=False, pfilename=""):
    """
        Compute the mean/median template spectrum along the time axis and divide/subtract
        each exposure by the mean/median
        
        Inputs:
        - flux_arr: 2D flux matrix (N_exposures, N_wavelengths)
        - wl: 1D wavelength array (N_wavelengths)
        - fit: boolean to fit median spectrum to each observation before normalizing it
        - median: boolean to calculate median instead of mean
        - subtract: boolean to subtract instead of dividing out spectra by the mean/median template

        Outputs:
        - loc: python dict containing all products
    """
    
    loc = {}

    loc["fit"] = fit
    loc["median"] = median
    loc["subtract"] = subtract
    loc["pfilename"] = pfilename

    if len(wl) == 0:
        x = np.arange(len(flux_arr[0]))
    else :
        x = wl

    if verbose :
        print("Calculating template out of {0} input spectra".format(len(flux_arr)))
    
    if median :
        # median combine
        flux_template = np.nanmedian(flux_arr,axis=0)
    else :
        # mean combine
        flux_template = np.nanmean(flux_arr,axis=0)
        #flux_template = np.average(flux_arr,axis=0, weights=weights)

    if fit :
        flux_calib = []
        flux_fit = []
        
        shift_arr = []
        scale_arr = []
        quadratic_arr = []

        def flux_model (coeffs, template, wave):
            outmodel = coeffs[2] * wave * wave + coeffs[1] * template + coeffs[0]
            return outmodel
        
        def errfunc (coeffs, fluxes, xx) :
            nanmask = ~np.isnan(fluxes)
            residuals = fluxes[nanmask] - flux_model (coeffs, flux_template[nanmask], xx[nanmask])
            return residuals

        for i in range(len(flux_arr)):
            
            nanmask = ~np.isnan(flux_arr[i])
            
            if len(flux_arr[i][nanmask]) > min_npoints :
                #guess = [0.0001, 1.001]
                guess = [0.0001, 1.001, 0.0000001]
                pfit, success = optimize.leastsq(errfunc, guess, args=(flux_arr[i], x))
            else :
                pfit = [0.,1.,0.]

            flux_template_fit = flux_model(pfit, flux_template, x)
            flux_fit.append(flux_template_fit)

            shift_arr.append(pfit[0])
            scale_arr.append(pfit[1])
            quadratic_arr.append(pfit[2])

            #flux_calib_loc = (flux_arr[i] - pfit[0]) / pfit[1]
            flux_calib_loc = (flux_arr[i] - pfit[2] * x * x - pfit[0]) / pfit[1]
            flux_calib.append(flux_calib_loc)

        loc["shift"] = np.array(shift_arr, dtype=float)
        loc["scale"] = np.array(scale_arr, dtype=float)
        loc["quadratic"] = np.array(quadratic_arr, dtype=float)

        flux_calib = np.array(flux_calib, dtype=float)
        flux_fit = np.array(flux_fit, dtype=float)

        # Compute median on all spectra along the time axis
        if median :
            flux_template_new = np.nanmedian(flux_calib,axis=0)
        else :
            flux_template_new = np.nanmean(flux_calib,axis=0)
            #flux_template_new = np.average(flux_calib,axis=0, weights=weights)

        flux_template = flux_template_new
        if subtract :
            flux_arr_sub = flux_calib - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_calib / flux_template

        residuals = flux_calib - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_calib
    else :
        # Divide or subtract each ccf by ccf_med
        if subtract :
            flux_arr_sub = flux_arr - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_arr / flux_template

        residuals = flux_arr - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_arr

    loc["flux"] = flux_template
    loc["fluxerr"] = flux_template_medsig
    loc["wl"] = x
    loc["flux_arr_sub"] = flux_arr_sub
    loc["flux_residuals"] = residuals
    loc["snr"] = flux_arr / flux_template_medsig

    loc["template_source"] = "data"
    
    template_nanmask = ~np.isnan(flux_template)
    template_nanmask &= ~np.isnan(flux_template_medsig)
    
    if len(flux_template_medsig[template_nanmask]) :
        loc["fluxerr_model"] = fit_continuum(x, flux_template_medsig, function='polynomial', order=5, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux error", plot_fit=False,silent=True)
    else :
        loc["fluxerr_model"] = np.full_like(x,np.nan)

    if plot :
        plot_template_products(loc, pfilename=pfilename)
    
    return loc


def sigma_clip(template, nsig=3.0, interpolate=False, replace_by_model=True, sub_flux_base=1.0, plot=False) :
    
    out_flux_arr = np.full_like(template["flux_arr"], np.nan)
    out_flux_arr_sub = np.full_like(template["flux_arr_sub"], np.nan)

    for i in range(len(template["flux_arr"])) :
        sigclipmask = np.abs(template["flux_residuals"][i]) > (nsig * template["fluxerr_model"])
        if plot :
            plt.plot(template["wl"], template["flux_residuals"][i], alpha=0.3)
            if len(template["flux_residuals"][i][sigclipmask]) :
                plt.plot(template["wl"][sigclipmask], template["flux_residuals"][i][sigclipmask], "bo")
    
        # set good values first
        out_flux_arr[i][~sigclipmask] = template["flux_arr"][i][~sigclipmask]
        out_flux_arr_sub[i][~sigclipmask] = template["flux_arr_sub"][i][~sigclipmask]
    
        # now decide what to do with outliers
        if interpolate :
            if i > 0 and i < len(template["flux_arr"]) - 1 :
                out_flux_arr[i][sigclipmask] = (template["flux_arr"][i-1][sigclipmask] + template["flux_arr"][i+1][sigclipmask]) / 2.
                out_flux_arr_sub[i][sigclipmask] = (template["flux_arr_sub"][i-1][sigclipmask] + template["flux_arr_sub"][i+1][sigclipmask]) / 2.
            elif i == 0 :
                out_flux_arr[i][sigclipmask] = template["flux_arr"][i+1][sigclipmask]
                out_flux_arr_sub[i][sigclipmask] = template["flux_arr_sub"][i+1][sigclipmask]
            elif i == len(template["flux_arr"]) - 1 :
                out_flux_arr[i][sigclipmask] = template["flux_arr"][i-1][sigclipmask]
                out_flux_arr_sub[i][sigclipmask] = template["flux_arr_sub"][i-1][sigclipmask]
        
        if replace_by_model :
            out_flux_arr[i][sigclipmask] = template["flux"][sigclipmask]
            out_flux_arr_sub[i][sigclipmask] = sub_flux_base

        #if plot :
        #    plt.plot(template["wl"][sigclipmask],out_flux_arr[i][sigclipmask],'b.')

    if plot :
        plt.plot(template["wl"], nsig * template["fluxerr_model"], 'r--', lw=2)
        plt.plot(template["wl"], -nsig * template["fluxerr_model"], 'r--', lw=2)
        plt.show()
    
    template["flux_arr"] = out_flux_arr
    template["flux_arr_sub"] = out_flux_arr_sub

    return template


def plot_template_products(template, pfilename="") :

    wl = template["wl"]

    for i in range(len(template["flux_arr"])) :
        
        flux = template["flux_arr"][i]
        resids = template["flux_residuals"][i]

        if i == len(template["flux_arr"]) - 1 :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="SPIRou data")
            plt.plot(wl, resids,"-", color='#8c564b', lw=0.6, alpha=0.5, label="Residuals")
        else :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6)
            plt.plot(wl, resids,"-", color='#8c564b', lw=0.6, alpha=0.5)

    plt.plot(template["wl"], template["flux"],"-", color="red", lw=2, label="Template spectrum")

    sig_clip = 3.0
    plt.plot(template["wl"], sig_clip * template["fluxerr"],"--", color="olive", lw=0.8)
    plt.plot(template["wl"], sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8)

    plt.plot(template["wl"],-sig_clip * template["fluxerr"],"--", color="olive", lw=0.8, label=r"{0:.0f}$\sigma$ (MAD)".format(sig_clip))
    plt.plot(template["wl"],-sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8, label="{0:.0f}$\sigma$ fit model".format(sig_clip))

    plt.legend(fontsize=20)
    plt.xlabel(r"$\lambda$ [nm]", fontsize=26)
    plt.ylabel(r"Flux", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    if pfilename != "" :
        plt.savefig(pfilename, format='png')
    else :
        plt.show()
    plt.clf()
    plt.close()



def fit_continuum(wav, spec, function='polynomial', order=3, nit=5, rej_low=2.0,
    rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,
                  min_points=10, xlabel="", ylabel="", plot_fit=True, verbose=False, silent=False):
    """
    Continuum fitting re-implemented from IRAF's 'continuum' function
    in non-interactive mode only but with additional options.

    :Parameters:
    
    wav: array(float)
        abscissa values (wavelengths, velocities, ...)

    spec: array(float)
        spectrum values

    function: str
        function to fit to the continuum among 'polynomial', 'spline3'

    order: int
        fit function order:
        'polynomial': degree (not number of parameters as in IRAF)
        'spline3': number of knots

    nit: int
        number of iteractions of non-continuum points
        see also 'min_points' parameter

    rej_low: float
        rejection threshold in unit of residul standard deviation for point
        below the continuum

    rej_high: float
        same as rej_low for point above the continuum

    grow: int
        number of neighboring points to reject

    med_filt: int
        median filter the spectrum on 'med_filt' pixels prior to fit
        improvement over IRAF function
        'med_filt' must be an odd integer

    percentile_low: float
        reject point below below 'percentile_low' percentile prior to fit
        improvement over IRAF function
        "percentile_low' must be a float between 0. and 100.

    percentile_high: float
        same as percentile_low but reject points in percentile above
        'percentile_high'
        
    min_points: int
        stop rejection iterations when the number of points to fit is less than
        'min_points'

    plot_fit: bool
        if true display two plots:
            1. spectrum, fit function, rejected points
            2. residual, rejected points

    verbose: bool
        if true fit information is printed on STDOUT:
            * number of fit points
            * RMS residual
    """
    if silent :
        import warnings
        warnings.simplefilter('ignore', np.RankWarning)
    
    mspec = np.ma.masked_array(spec, mask=np.zeros_like(spec))
    # mask 1st and last point: avoid error when no point is masked
    # [not in IRAF]
    mspec.mask[0] = True
    mspec.mask[-1] = True
    
    mspec = np.ma.masked_where(np.isnan(spec), mspec)
    
    # apply median filtering prior to fit
    # [opt] [not in IRAF]
    if int(med_filt):
        fspec = sig.medfilt(spec, kernel_size=med_filt)
    else:
        fspec = spec
    # consider only a fraction of the points within percentile range
    # [opt] [not in IRAF]
    mspec = np.ma.masked_where(fspec < np.percentile(fspec, percentile_low),
        mspec)
    mspec = np.ma.masked_where(fspec > np.percentile(fspec, percentile_high),
        mspec)
    # perform 1st fit
    if function == 'polynomial':
        coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
        cont = np.poly1d(coeff)(wav)
    elif function == 'spline3':
        knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
        spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
        cont = sint.splev(wav, spl)
    else:
        raise(AttributeError)
    # iteration loop: reject outliers and fit again
    if nit > 0:
        for it in range(nit):
            res = fspec-cont
            sigm = np.std(res[~mspec.mask])
            # mask outliers
            mspec1 = np.ma.masked_where(res < -rej_low*sigm, mspec)
            mspec1 = np.ma.masked_where(res > rej_high*sigm, mspec1)
            # exlude neighbors cf IRAF's continuum parameter 'grow'
            if grow > 0:
                for sl in np.ma.clump_masked(mspec1):
                    for ii in range(sl.start-grow, sl.start):
                        if ii >= 0:
                            mspec1.mask[ii] = True
                    for ii in range(sl.stop+1, sl.stop+grow+1):
                        if ii < len(mspec1):
                            mspec1.mask[ii] = True
            # stop rejection process when min_points is reached
            # [opt] [not in IRAF]
            if np.ma.count(mspec1) < min_points:
                if verbose:
                    print("  min_points %d reached" % min_points)
                break
            mspec = mspec1
            if function == 'polynomial':
                coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
                cont = np.poly1d(coeff)(wav)
            elif function == 'spline3':
                knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
                spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
                cont = sint.splev(wav, spl)
            else:
                raise(AttributeError)
    # compute residual and rms
    res = fspec-cont
    sigm = np.std(res[~mspec.mask])
    if verbose:
        print("  nfit=%d/%d" %  (np.ma.count(mspec), len(mspec)))
        print("  fit rms=%.3e" %  sigm)
    # compute residual and rms between original spectrum and model
    # different from above when median filtering is applied
    ores = spec-cont
    osigm = np.std(ores[~mspec.mask])
    if int(med_filt) and verbose:
        print("  unfiltered rms=%.3e" %  osigm)
    # plot fit results
    if plot_fit:
        # overplot spectrum and model + mark rejected points
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(wav[~mspec.mask], spec[~mspec.mask],
            c='tab:blue', lw=1.0)
        # overplot median filtered spectrum
        if int(med_filt):
            ax1.plot(wav[~mspec.mask], fspec[~mspec.mask],
                c='tab:cyan', lw=1.0)
        ax1.scatter(wav[mspec.mask], spec[mspec.mask], s=20., marker='d',
        edgecolors='tab:gray', facecolors='none', lw=0.5)
        ax1.plot(wav, cont, ls='--', c='tab:orange')
        if nit > 0:
            # plot residuals and rejection thresholds
            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.axhline(0., ls='--', c='tab:orange', lw=1.)
            ax2.axhline(-rej_low*sigm, ls=':')
            ax2.axhline(rej_high*sigm, ls=':')
            ax2.scatter(wav[mspec.mask], res[mspec.mask],
                s=20., marker='d', edgecolors='tab:gray', facecolors='none',
                lw=0.5)
            ax2.scatter(wav[~mspec.mask], ores[~mspec.mask],
                marker='o', s=10., edgecolors='tab:blue', facecolors='none',
                lw=.5)
            # overplot median filtered spectrum
            if int(med_filt):
                ax2.scatter(wav[~mspec.mask], res[~mspec.mask],
                    marker='s', s=5., edgecolors='tab:cyan', facecolors='none',
                    lw=.2)
        if xlabel != "" :
            plt.xlabel(xlabel)
        if ylabel != "" :
            plt.ylabel(ylabel)
        plt.show()
    return cont



def normalize_spectra(spectra, template, fluxkey="fluxes", fluxerrkey="fluxerrs", cont_function='polynomial', polyn_order=4, med_filt=1, plot=False) :
    
    continuum_fluxes = []
    
    for order in range(len(template)) :
        order_template = template[order]

        wl = order_template["wl"]
        flux = order_template["flux"]
        fluxerr = order_template["fluxerr"]
        
        nanmask = np.isfinite(flux)
        nanmask &= np.isfinite(wl)

        continuum = np.full_like(wl, np.nan)
        
        if len(flux[nanmask]) > 10 :
            continuum[nanmask] = fit_continuum(wl[nanmask], flux[nanmask], function=cont_function, order=polyn_order, nit=10, rej_low=1.0, rej_high=4, grow=1, med_filt=med_filt, percentile_low=0., percentile_high=100.,min_points=100, xlabel="wavelength", ylabel="flux", plot_fit=False, silent=True)
        if plot :
            plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', lw=0.3, alpha=0.3, zorder=1)
            #plt.scatter(wl, flux, marker='o', s=10., edgecolors='tab:blue', facecolors='none', lw=.5)
            plt.plot(wl, continuum, '-', lw=2, zorder=2)
    
        for i in range(spectra['nspectra']) :
            spectra[fluxkey][order][i] /= continuum
            spectra[fluxerrkey][order][i] /= continuum
           
        template[order]["continuum"] = continuum
           
        template[order]["flux"] /= continuum
        template[order]["fluxerr"] /= continuum
        template[order]["fluxerr_model"] /= continuum
        
        for j in range(len(template[order]["flux_arr"])) :
            template[order]["flux_arr"][j] /= continuum
            template[order]["flux_residuals"][j] /= continuum

        continuum_fluxes.append(continuum)

    if plot :
        plt.show()

    spectra["continuum_{}".format(fluxkey)] = np.array(continuum_fluxes, dtype=float)

    return spectra, template



def recover_continuum(spectra, template, fluxkey="fluxes", fluxerrkey="fluxerrs") :

    for order in range(len(template)) :
    
        continuum_flux = spectra["continuum_{}".format(fluxkey)][order]
    
        for i in range(spectra['nspectra']) :
            spectra[fluxkey][order][i] *= continuum_flux
            spectra[fluxerrkey][order][i] *= continuum_flux
                
        template[order]["flux"] *= continuum_flux
        template[order]["fluxerr"] *= continuum_flux
        template[order]["fluxerr_model"] *= continuum_flux
        
        for j in range(len(template[order]["flux_arr"])) :
            template[order]["flux_arr"][j] *= continuum_flux
            template[order]["flux_residuals"][j] *= continuum_flux

    return spectra, template


def calculate_weights(spectra, template, normalize_weights=True, use_err_model=True, plot=False) :

    weights = []

    normfactor = []

    for order in range(len(template)) :
        order_template = template[order]
        
        wl = order_template["wl"]
        flux = order_template["flux"]

        if use_err_model :
            fluxerr = order_template["fluxerr_model"]
        else :
            fluxerr = order_template["fluxerr"]

        nanmask = np.isfinite(flux)
        nanmask &= np.isfinite(fluxerr)
        nanmask &= flux > 0
            
        loc_weights = np.full_like(fluxerr, np.nan)
        
        loc_weights[nanmask] = 1. / (fluxerr[nanmask] * fluxerr[nanmask])
        weights.append(loc_weights)
        
        if len(loc_weights[nanmask]) :
            normfactor.append(np.nanmedian(loc_weights[nanmask]))
        else :
            normfactor.append(np.nan)

    normfactor = np.array(normfactor)
    
    if normalize_weights :
        median_norm_factor = np.nanmax(normfactor[np.isfinite(normfactor)])
    else :
        median_norm_factor = 1.0
    
    for order in range(len(template)) :
        
        order_template = template[order]
        
        wl = order_template["wl"]
        flux = order_template["flux"]

        weights[order] /= median_norm_factor
        
        if plot :
            
            plt.ylim(-0.5,3.0)
            
            plt.scatter(wl, weights[order], marker='o', s=10., edgecolors='tab:red', facecolors='none', lw=.5)
            
            plt.plot(wl, flux, '-')

    if plot :
        plt.show()

    spectra["weights"] = np.array(weights, dtype=float)

    return spectra


def plot_template_products_with_CCF_mask(template, ccfmask, source_rv=0, pfilename="") :

    wl = template["wl"]

    for i in range(len(template["flux_arr"])) :
        
        flux = template["flux_arr"][i]
        resids = template["flux_residuals"][i]

        if i == len(template["flux_arr"]) - 1 :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="SPIRou data", zorder=1)
            plt.plot(wl, resids,".", color='#8c564b', lw=0.2, alpha=0.2, label="Residuals", zorder=1)
        else :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, zorder=1)
            plt.plot(wl, resids,".", color='#8c564b', lw=0.2, alpha=0.2, zorder=1)

    # Plot CCF mask
    lines_in_order = ccfmask["orders"] == template['order']
    order_lines_wlc = ccfmask["centers"][lines_in_order]
    order_lines_wei = ccfmask["weights"][lines_in_order]
    speed_of_light_in_kps = constants.c / 1000.
    wlc_starframe = order_lines_wlc * (1.0 + source_rv / speed_of_light_in_kps)
    median_flux = np.nanmedian(template["flux"])
    plt.vlines(wlc_starframe, median_flux - order_lines_wei / np.nanmax(order_lines_wei), median_flux,ls="--", lw=0.7, label="CCF lines", zorder=2)
    #---------------
    
    plt.plot(template["wl"], template["flux"],"-", color="red", lw=2, label="Template spectrum", zorder=1.5)

    sig_clip = 3.0
    plt.plot(template["wl"], sig_clip * template["fluxerr"],"-", color="darkgreen", lw=2, zorder=1.1)
    #plt.plot(template["wl"], sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8, zorder=2)

    plt.plot(template["wl"],-sig_clip * template["fluxerr"],"-", color="darkgreen", lw=2, label=r"{0:.0f}$\sigma$".format(sig_clip), zorder=1.1)
    #plt.plot(template["wl"],-sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8, label="{0:.0f}$\sigma$ fit model".format(sig_clip), zorder=2)

    plt.legend(fontsize=16)
    plt.xlabel(r"$\lambda$ [nm]", fontsize=26)
    plt.ylabel(r"Relative flux", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    #plt.xlim(1573,1581)
    if pfilename != "" :
        plt.savefig(pfilename, format='png')
    else :
        plt.show()
    plt.clf()
    plt.close()


def calculate_template_and_normalize(spectra, nsig_clip=3, cont_function='polynomial', polyn_order=4, med_filt=1, update_spectra=True, verbose=False) :

    # Calculate template in the rest frame
    template_rest = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", wavekey="common_wl", update_spectra=update_spectra, plot=False, verbose=verbose)
    
    # Calculate continuum and normalize spectra
    spectra, template_rest = normalize_spectra(spectra, template_rest, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", cont_function=cont_function, polyn_order=polyn_order, med_filt=med_filt, plot=False)
    #spectra["rest_fluxes"] / spectra["continuum_rest_fluxes"]
    
    # Calculate template in the stellar frame
    template_sf = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=update_spectra, plot=False, verbose=verbose)
    # Calculate continuum and normalize spectra
    spectra, template_sf = normalize_spectra(spectra, template_sf, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", cont_function=cont_function, polyn_order=polyn_order, med_filt=med_filt, plot=False)
    #spectra["sf_fluxes"] / spectra["continuum_sf_fluxes"]

    return template_rest, template_sf, spectra


def reduce_timeseries_of_spectra(inputdata, ccf_mask, object_name="", stellar_spectrum_file="", source_rv=0., max_gap_size=8.0, min_window_size=200., align_spectra=True, vel_sampling=1.8, nsig_clip = 3.0, ccf_width=150, convolve_spectra=False, to_resolution=30000, output_template="", tel_mask="", h2o_mask="", telluric_rv=False, verbose=False) :
    
    """
        Description: function to process a series of IGRINS spectra. The processing consist of
        the following steps:
        """
    
    if verbose :
        print("******************************")
        print("STEP: Loading IGRINS data ...")
        print("******************************")
 
    # First load spectra into a container
    array_of_spectra = load_array_of_igrins_spectra(inputdata, object_name=object_name, convolve_spectra=convolve_spectra, to_resolution=to_resolution, plot_diagnostics=False, plot=False, verbose=True)

    # Then load data into order vectors -- it is more efficient to work the reduction order-by-order
    spectra = get_spectral_data(array_of_spectra, verbose=True)

    # Select regions with saturated telluric lines to be masked out
    saturated_tellurics = [[1820,1823],[1897.7,1907.2],[1911.046,1916.747],[1951.25,1959.714],[1998.01,2014.61],[2434.979,2439.186],[2450.8,2451.64],[2479.381,2480.0]]

    for order in range(NORDERS) :
        for i in range(spectra['nspectra']) :
            for r in saturated_tellurics :
                mask = (spectra["waves"][order][i] > r[0]) & (spectra["waves"][order][i] < r[1])
                spectra["fluxes"][order][i][mask] = np.nan
                spectra["fluxerrs"][order][i][mask] = np.nan
   
    # Detect gaps in the data and create a gap function for good windows without gaps
    # Use wide values to avoid too much clipping at this point. This will improve the noise model
    spectra = get_gapfree_windows(spectra, max_vel_distance=max_gap_size, min_window_size=min_window_size, fluxkey="fluxes", wavekey="waves_sf", verbose=False)

    # Set a common wavelength grid for all input spectra
    spectra = set_common_wl_grid(spectra, vel_sampling=vel_sampling)
    
    # Interpolate all spectra to a common wavelength grid
    spectra = resample_and_align_spectra(spectra, use_gp=False, verbose=verbose, plot=False)
    #spectra["aligned_waves"]
    #spectra["sf_fluxes"],spectra["sf_fluxerrs"]
    #spectra["rest_fluxes"], spectra["rest_fluxerrs"]

    # Calculate stellar spectrum from an input spirou template
    spectra = calculate_stellar_spectra_from_spirou_template(spectra, stellar_spectrum_file)
    #spectra["stellar_sf_fluxes"], spectra["stellar_sf_fluxerrs"]
    #spectra["stellar_rest_fluxes"], spectra["stellar_rest_fluxerrs"]

    # Remove stellar contribution from input spectra
    spectra = remove_stellar_spectra(spectra)
    #spectra["sf_fluxes"] / spectra["stellar_sf_fluxes"]
    #spectra["rest_fluxes"] / spectra["stellar_rest_fluxes"]

    # calculate templates in both ref frames and normalize spectra by the continuum
    template_rest, template_sf, spectra = calculate_template_and_normalize(spectra, nsig_clip=nsig_clip, update_spectra=True, verbose=verbose)

    # Calculate stellar spectrum from an input template
    spectra, template_rest = calculate_telluric_spectra(spectra, template_rest, plot=False)
    #spectra["telluric_sf_fluxes"], spectra["telluric_sf_fluxerrs"]
    #spectra["telluric_rest_fluxes"], spectra["telluric_rest_fluxerrs"]

    # remove tellurics
    spectra, template_rest = remove_telluric_spectra(spectra, template_rest)
    #spectra["sf_fluxes"] / spectra["telluric_sf_fluxes"]
    #spectra["rest_fluxes"] / spectra["telluric_rest_fluxes"]

    # recover stellar spectra
    spectra = recover_stellar_spectra(spectra)
    #spectra["rest_fluxes"] * spectra["stellar_rest_fluxes"]
    #spectra["sf_fluxes"] * spectra["stellar_sf_fluxes"]

    # Calculate again the template in the stellar frame
    template_sf = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=verbose)
    
    """
    for order in range(NORDERS) :
        stellar_flux = template_sf[order]["flux"]
        # plot stella flux template
        plt.plot(template_sf[order]["wl"],stellar_flux,'r-',alpha=0.5)
        for i in range(spectra['nspectra']) :
            sflux = template_sf[order]["flux_residuals"][i] + stellar_flux
            plt.plot(template_sf[order]["wl"],sflux,'-',alpha=0.5)
    """

    # recover continuum
    #spectra, template_rest = recover_continuum(spectra, template_rest, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs")
    #spectra, template_sf = recover_continuum(spectra, template_sf, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs")

    # recalculate stellar spectra from template in the stellar frame
    spectra, template_sf = calculate_stellar_spectra(spectra, template_sf, plot=False)
    #spectra["stellar_sf_fluxes"], spectra["stellar_sf_fluxerrs"]
    #spectra["stellar_rest_fluxes"], spectra["stellar_rest_fluxerrs"]

    # Remove stellar contribution from input spectra
    spectra, template_sf = remove_stellar_spectra(spectra, template_sf)
    #spectra["sf_fluxes"] / spectra["stellar_sf_fluxes"]
    #spectra["rest_fluxes"] / spectra["stellar_rest_fluxes"]

    # recover telluric spectra
    spectra, template_rest = recover_telluric_spectra(spectra, template_rest)
    #spectra["rest_fluxes"] * spectra["telluric_rest_fluxes"]
    #spectra["sf_fluxes"] * spectra["telluric_sf_fluxes"]
    
    # Calculate again the template in the stellar frame
    template_rest = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=verbose)
    
    # Calculate template in the rest frame
    #template_rest = reduce_spectra_wavecal(spectra, combine_by_median=True, subtract=True, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=verbose)
    # Here, if line above is uncommented, one should apply the wave shifts per order to the stellar spectra
    # as well, and it would already work as a correction to possible wavelength calibration problems.
    
    # Calculate stellar spectrum from an input template
    spectra, template_rest = calculate_telluric_spectra(spectra, template_rest, plot=False)
    #spectra["telluric_sf_fluxes"], spectra["telluric_sf_fluxerrs"]
    #spectra["telluric_rest_fluxes"], spectra["telluric_rest_fluxerrs"]

    if telluric_rv :
        if verbose :
            print("Calculating template of telluric (recon) spectra ...")
        telluric_template = deepcopy(template_rest)
    else :
        telluric_template = None

    #order_subset_for_mean_ccf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    order_subset_for_mean_ccf = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 34, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    """
    for order in range(NORDERS) :
        if order in order_subset_for_mean_ccf :
            telluric_rest_flux = template_rest[order]["telluric_flux"]
            # plot telluric flux
            plt.plot(template_rest[order]["wl"],telluric_rest_flux,'r-',alpha=0.5)
     """
    # remove tellurics
    spectra, template_rest = remove_telluric_spectra(spectra, template_rest)
    #spectra["sf_fluxes"] / spectra["telluric_sf_fluxes"]
    #spectra["rest_fluxes"] / spectra["telluric_rest_fluxes"]
    
    # calculate templates in both ref frames and normalize spectra by the continuum
    template_rest, template_sf, spectra = calculate_template_and_normalize(spectra, nsig_clip=nsig_clip, update_spectra=True, verbose=verbose)

    # recover stellar spectra
    spectra = recover_stellar_spectra(spectra)
    #spectra["rest_fluxes"] * spectra["stellar_rest_fluxes"]
    #spectra["sf_fluxes"] * spectra["stellar_sf_fluxes"]

    # Calculate again the template in the stellar frame
    template_sf = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=verbose)

    # Calculate template in the rest frame
    #template_rest = reduce_spectra_wavecal(spectra, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=verbose)

    # recalculate stellar spectra from template in the stellar frame
    spectra, template_sf = calculate_stellar_spectra(spectra, template_sf, plot=False)
    #spectra["stellar_sf_fluxes"], spectra["stellar_sf_fluxerrs"]
    #spectra["stellar_rest_fluxes"], spectra["stellar_rest_fluxerrs"]


    """
    for order in range(NORDERS) :
    
        if order in order_subset_for_mean_ccf :

            #telluric_rest_flux = template[order]["telluric_flux"]
            #template_continuum_flux = template[order]["continuum"]
        
            color = [order/NORDERS,1-order/NORDERS,1-order/NORDERS]
        
            common_wl = spectra["common_wl"][order]
                
            flux = spectra["rest_fluxes"][order][0]
            fluxerr = spectra["rest_fluxerrs"][order][0]

            contflux = np.ones_like(common_wl) * np.nan
            keep = np.isfinite(flux)
            #if len(flux[keep]) > 300 :
            #    contflux[keep] = fit_continuum(common_wl[keep], flux[keep], function='spline3', order=6, nit=5, rej_low=1.5, rej_high=3.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux error", plot_fit=True,silent=True)
        
            #plot data
            plt.errorbar(common_wl, flux, yerr=fluxerr, fmt='.', color='k', alpha=0.3,label="IGRINS observations")
        
            stellar_rest_flux = spectra["stellar_rest_fluxes"][order][0]
            # plot stellar flux
            plt.plot(common_wl, stellar_rest_flux, 'b-',label="Stellar template")
        
            # plot telluric flux
            telluric_rest_flux = spectra["telluric_fluxes"][order][0]
            plt.plot(common_wl,telluric_rest_flux,'r-',alpha=0.5, label="Tellurics")
        
            # plot residuals
            plt.errorbar(common_wl, (flux-stellar_rest_flux) +0.7, yerr=fluxerr, fmt='.', color='k', alpha=0.3, label="Residuals")

            # plot continuum flux
            #plt.plot(template[order]["wl"],template_continuum_flux,'g-',alpha=0.5, lw=2)

    plt.xlabel("wavelength [nm]",fontsize=20)
    plt.ylabel("flux",fontsize=20)
    plt.show()
    """
    
    # From here on it runs the CCF routines
    template = template_sf
    
    # Calculate statistical weights based on the time series dispersion 1/sig^2
    spectra = calculate_weights(spectra, template, use_err_model=False, plot=False)

    # Start dealing with CCF related parameters and construction of a weighted mask
    # load science CCF parameters
    ccf_params = ccf_lib.set_ccf_params(ccf_mask)

    # update ccf width with input value
    ccf_params["CCF_WIDTH"] = float(ccf_width)

    templ_fluxes, templ_fluxerrs, templ_wave = [], [], []
    templ_tellwave, templ_tellfluxes, templ_tellfluxerrs = [], [], []

    for order in range(NORDERS) :
        order_template = template[order]
        templ_fluxes.append(order_template["flux"])
        templ_fluxerrs.append(order_template["fluxerr"])
        templ_wave.append(order_template["wl"])
        
        if telluric_rv :
            order_telltemplate = telluric_template[order]
            templ_tellfluxes.append(order_telltemplate["flux"])
            templ_tellfluxerrs.append(order_telltemplate["fluxerr"])
            templ_tellwave.append(order_telltemplate["wl"])

    templ_fluxes = np.array(templ_fluxes, dtype=float)
    templ_fluxerrs = np.array(templ_fluxerrs, dtype=float)
    templ_wave = np.array(templ_wave, dtype=float)
    
    templ_tellwave = np.array(templ_tellwave, dtype=float)
    templ_tellfluxes = np.array(templ_tellfluxes, dtype=float)
    templ_tellfluxerrs = np.array(templ_tellfluxerrs, dtype=float)

    ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, templ_wave, templ_fluxes, templ_fluxerrs, spectra["weights"], median=True, remove_lines_with_nans=True, source_rv=source_rv, verbose=False)
    
    base_header = deepcopy(array_of_spectra["spectra"][0]["header"])

    template_ccf = ccf_lib.run_ccf_eder(ccf_params, templ_wave, templ_fluxes, base_header, ccfmask, targetrv=source_rv, valid_orders=order_subset_for_mean_ccf, normalize_ccfs=True, output=False, plot=False, verbose=False)
    
    if output_template != "" :
        if verbose :
            print("Saving template spectrum to file: {0} ".format(output_template))
        #spectrumlib.write_spectrum_orders_to_fits(templ_wave, templ_fluxes, templ_fluxerrs, output_template, header=template_ccf["header"])

    source_rv = template_ccf["header"]['RV_OBJ']
    ccf_params["SOURCE_RV"] = source_rv
    ccf_params["CCF_WIDTH"] = 8 * template_ccf["header"]['CCFMFWHM']

    if verbose :
        print("Source RV={:.4f} km/s  CCF FWHM={:.2f} km/s CCF window size={:.2f} km/s".format(source_rv,template_ccf["header"]['CCFMFWHM'],ccf_params["CCF_WIDTH"]))
    # Apply weights to stellar CCF mask
    ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, templ_wave, templ_fluxes, templ_fluxerrs, spectra["weights"], median=True, remove_lines_with_nans=True, source_rv=source_rv, verbose=verbose)

    #order_to_plot = 7
    #plot_template_products_with_CCF_mask(template[order_to_plot], ccfmask, source_rv=ccf_params["SOURCE_RV"],pfilename="")
    # Uncomment below to check the match between telluric spectra and H2O CCF mask for every order
    """
    for order in range(NORDERS) :
        print("plotting order {}".format(order))
        try :
            plot_template_products_with_CCF_mask(template[order], ccfmask, source_rv=ccf_params["SOURCE_RV"],pfilename="")
        except :
            print("could not plot order {}".format(order))
            continue
    """
    
    if telluric_rv and h2o_mask != "" :
        if verbose :
            print("Applying weights to H2O CCF mask ...")
        h2o_ccf_params = ccf_lib.set_ccf_params(h2o_mask)
        h2o_ccf_params["CCF_WIDTH"] = float(ccf_width)
        h2o_ccfmask = ccf_lib.apply_weights_to_ccf_mask(h2o_ccf_params, templ_tellwave, templ_tellfluxes, templ_tellfluxerrs, np.full_like(templ_tellfluxes,1.0), median=False, verbose=verbose)
        # Uncomment below to check the match between telluric spectra and H2O CCF mask for every order
        """
        for order in range(NORDERS) :
            print("plotting order {}".format(order))
            try :
                plot_template_products_with_CCF_mask(telluric_template[order], h2o_ccfmask, source_rv=0.,pfilename="")
            except :
                print("could not plot order {}".format(order))
                continue
            """
    if telluric_rv and tel_mask != "" :
        if verbose :
            print("Applying weights to telluric CCF mask ...")
        #tell_ccf_params = ccf_lib.set_ccf_params(h2o_mask, telluric_masks=[tel_mask])
        tell_ccf_params = ccf_lib.set_ccf_params(tel_mask)
        tell_ccf_params["CCF_WIDTH"] = float(ccf_width)
        tell_ccfmask = ccf_lib.apply_weights_to_ccf_mask(tell_ccf_params, templ_tellwave, templ_tellfluxes, templ_tellfluxerrs, np.full_like(templ_tellfluxes,1.0), median=False, verbose=verbose)
        # Uncomment below to check the match between telluric spectra and telluric CCF mask for every order
        """
        for order in range(NORDERS) :
            print("plotting order {}".format(order))
            try :
                plot_template_products_with_CCF_mask(telluric_template[order], tell_ccfmask, source_rv=0., pfilename="")
            except :
                print("could not plot order {}".format(order))
                continue
            """

    loc = {}

    loc["array_of_spectra"] = array_of_spectra
    loc["spectra"] = spectra
    loc["template"] = template

    loc["ccf_params"] = ccf_params
    loc["ccfmask"] = ccfmask
    
    if telluric_rv :
        loc["tell_ccfmask"] = tell_ccfmask
        loc["tell_ccf_params"] = tell_ccf_params
        loc["h2o_ccfmask"] = h2o_ccfmask
        loc["h2o_ccf_params"] = h2o_ccf_params
    else :
        loc["tell_ccfmask"] = None
        loc["tell_ccf_params"] = None
        loc["h2o_ccfmask"] = None
        loc["h2o_ccf_params"] = None

    loc["fluxkey"], loc["fluxerrkey"] = "sf_fluxes", "sf_fluxerrs"
    loc["waveskey"], loc["wavekey"] =  "aligned_waves", "common_wl"

    return loc


def spec_model(template_wl, template_flux, outwl, scale, shift, a, b, return_template=False) :

    f = interp1d(template_wl, template_flux, kind='cubic')
        
    flux_model = f(outwl) * scale + shift + (a * outwl + b * outwl * outwl)

    if return_template :
        return flux_model, f(outwl)
    else :
        return flux_model


def fit_template(wl, flux, fluxerr, template_wl, template_flux, resolution=0, plot=False) :

    if resolution :
        template_fluxerr = np.zeros_like(template_flux)
        template_wl, template_flux, template_fluxerr = telluric_lib.__convolve_spectrum(template_wl, template_flux, template_fluxerr, resolution)

    loc = {}

    def fit_template_errfunc (pars, locwl, locflux, locfluxerr):
    
        scale, shift, a, b = pars[0], pars[1], pars[2], pars[3]
    
        flux_model = spec_model(template_wl, template_flux, locwl, scale, shift, a, b)
        
        residuals = (locflux - flux_model) / locfluxerr
        
        return residuals
    
    guess = [1.01, 0.01, 0.0001, 0.0001]
    
    pfit, success = optimize.leastsq(fit_template_errfunc, guess, args=(wl, flux, fluxerr))

    #print(pfit)

    yfit, template_obs = spec_model(template_wl, template_flux, wl, *pfit, return_template=True)

    continuum = yfit / template_obs

    if plot :
        plt.plot(wl, flux, 'o', label='data')
        plt.plot(wl, yfit, '-', label='template * cont fit')

        plt.plot(wl, yfit/template_obs,'--', label='continuum fit')
        plt.xlabel("flux", fontsize=20)
        plt.xlabel("wavelength [nm]", fontsize=20)
        plt.legend()
        plt.show()
        
    loc = {}
    loc["flux_model"] = yfit
    loc["template_obs"] = template_obs
    loc["flux_without_continuum"] = yfit / continuum
    loc["reduced_flux"] = loc["flux_without_continuum"] / loc["template_obs"]
    loc["residuals"] = flux - yfit
    loc["pfit"] = pfit

    return loc


def calculate_stellar_spectra_from_spirou_template(spectra, stellar_spectrum_file, template=None) :
    
    speed_of_light_in_kps = constants.c / 1000.

    wl1, wl2 = 1400, 2500
    
    stellar_spectrum = igrinslib.load_spirou_s1d_template(stellar_spectrum_file, wl1=wl1, wl2=wl2, to_resolution=0, normalize=True, plot=False)

    fobj = interp1d(stellar_spectrum['wl'], stellar_spectrum['flux'], kind='cubic')
    ferrobj = interp1d(stellar_spectrum['wl'], stellar_spectrum['fluxerr'], kind='cubic')

    stellar_sf_fluxes, stellar_sf_fluxerrs = [], []
    stellar_rest_fluxes, stellar_rest_fluxerrs = [], []

    for order in range(NORDERS) :
        stellar_sf_fluxes.append([])
        stellar_sf_fluxerrs.append([])
        stellar_rest_fluxes.append([])
        stellar_rest_fluxerrs.append([])

    for order in range(NORDERS) :
        if template :
            order_template = template[order]
            wl = order_template["wl"]
            template_stellar_flux = np.ones_like(wl)
            if wl[0] > wl1 and wl[-1] < wl2 :
                template_stellar_flux = fobj(wl)
            template[order]["stellar_flux"] = template_stellar_flux

        wave = spectra["common_wl"][order]
            
        for i in range(spectra['nspectra']) :

            stellar_sf_flux, stellar_sf_fluxerr = np.ones_like(wave), np.zeros_like(wave)
            if wave[0] > wl1 and wave[-1] < wl2 :
                stellar_sf_flux = fobj(wave)
                stellar_sf_fluxerr = ferrobj(wave)
            
            vel_shift = (spectra["rvs"][i] - spectra["bervs"][i])
            # undo relativistic calculation
            wave_rest = wave * np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
            
            stellar_rest_flux, stellar_rest_fluxerr = np.ones_like(wave_rest), np.zeros_like(wave_rest)
            if wave_rest[0] > wl1 and wave_rest[-1] < wl2 :
                stellar_rest_flux = fobj(wave_rest)
                stellar_rest_fluxerr = ferrobj(wave_rest)

            stellar_sf_fluxes[order].append(stellar_sf_flux)
            stellar_sf_fluxerrs[order].append(stellar_sf_fluxerr)
            stellar_rest_fluxes[order].append(stellar_rest_flux)
            stellar_rest_fluxerrs[order].append(stellar_rest_fluxerr)

    spectra["stellar_sf_fluxes"] = stellar_sf_fluxes
    spectra["stellar_sf_fluxerrs"] = stellar_sf_fluxerrs
    spectra["stellar_rest_fluxes"] = stellar_rest_fluxes
    spectra["stellar_rest_fluxerrs"] = stellar_rest_fluxerrs

    if template :
        return spectra, template
    else :
        return spectra



def remove_stellar_spectra(spectra, template=None) :

    for order in range(NORDERS) :
        if template :
            order_template = template[order]

            template_stellar_flux = template[order]["stellar_flux"]

            template[order]["flux"] /= template_stellar_flux
            template[order]["fluxerr"] /= template_stellar_flux
            template[order]["fluxerr_model"] /= template_stellar_flux

            for j in range(len(template[order]["flux_arr"])) :
                template[order]["flux_arr"][j] /= template_stellar_flux
                template[order]["flux_residuals"][j] /= template_stellar_flux

        for i in range(spectra['nspectra']) :

            spectra["rest_fluxes"][order][i] /= spectra["stellar_rest_fluxes"][order][i]
            spectra["rest_fluxerrs"][order][i] /= spectra["stellar_rest_fluxes"][order][i]
            spectra["sf_fluxes"][order][i] /= spectra["stellar_sf_fluxes"][order][i]
            spectra["sf_fluxerrs"][order][i] /= spectra["stellar_sf_fluxes"][order][i]

    if template :
        return spectra, template
    else :
        return spectra


def recover_stellar_spectra(spectra, template=None) :

    for order in range(NORDERS) :
        if template :
            order_template = template[order]

            template_stellar_flux = template[order]["stellar_flux"]

            template[order]["flux"] *= template_stellar_flux
            template[order]["fluxerr"] *= template_stellar_flux
            template[order]["fluxerr_model"] *= template_stellar_flux

            for j in range(len(template[order]["flux_arr"])) :
                template[order]["flux_arr"][j] *= template_stellar_flux
                template[order]["flux_residuals"][j] *= template_stellar_flux

        for i in range(spectra['nspectra']) :
            spectra["rest_fluxes"][order][i] *= spectra["stellar_rest_fluxes"][order][i]
            spectra["rest_fluxerrs"][order][i] *= spectra["stellar_rest_fluxes"][order][i]
            spectra["sf_fluxes"][order][i] *= spectra["stellar_sf_fluxes"][order][i]
            spectra["sf_fluxerrs"][order][i] *= spectra["stellar_sf_fluxes"][order][i]

    if template :
        return spectra, template
    else :
        return spectra


# Define a cost function
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]


# function to interpolate spectrum using GP
def interp_spectrum_using_gp(wl_out, wl_in, flux_in, fluxerr_in, good_windows, verbose=False, plot=False) :

    flux_out = np.full_like(wl_out, np.nan)
    fluxerr_out = np.full_like(wl_out, np.nan)

    for w in good_windows :

        mask = wl_in >= w[0]
        mask &= wl_in <= w[1]
        mask &= np.isfinite(flux_in)
        mask &= np.isfinite(fluxerr_in)
        # Set up the GP model
        kernel = terms.RealTerm(log_a=np.log(np.var(flux_in[mask])), log_c=-np.log(10.0))
        gp = celerite.GP(kernel, mean=np.nanmean(flux_in[mask]), fit_mean=True)
        gp.compute(wl_in[mask], fluxerr_in[mask])
        # Fit for the maximum likelihood parameters
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", bounds=bounds, args=(flux_in[mask], gp))
        gp.set_parameter_vector(soln.x)

        wl1, wl2 = w[0], w[1]

        if wl1 < wl_in[mask][0] :
            wl1 = wl_in[mask][0]
        if wl2 > wl_in[mask][-1] :
            wl2 = wl_in[mask][-1]

        out_mask = wl_out > wl1
        out_mask &= wl_out < wl2

        flux_out[out_mask], var = gp.predict(flux_in[mask], wl_out[out_mask], return_var=True)
        fluxerr_out[out_mask] = np.sqrt(var)

    if plot :
        # Plot the data
        color = "#ff7f0e"
        plt.errorbar(wl_in, flux_in, yerr=fluxerr_in, fmt=".k", capsize=0)
        plt.plot(wl_out, flux_out, color=color)
        plt.fill_between(wl_out, flux_out+fluxerr_out, flux_out-fluxerr_out, color=color, alpha=0.3, edgecolor="none")
        plt.ylabel(r"CCF")
        plt.xlabel(r"Velocity [km/s]")
        plt.title("maximum likelihood prediction")
        plt.show()

    return flux_out, fluxerr_out




def calculate_variances(spectra, template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs") :

    var_template, var_meas_template = 0., 0.
    var_global, var_meas_global = 0., 0.

    for order in range(NORDERS) :
        if template :
            order_template = template[order]
            keep = np.isfinite(template[order]["flux"])
            if len(template[order]["flux"][keep]) :
                sigma = np.nanstd(template[order]["flux"][keep])
                var_template += sigma*sigma
                var_meas_template += np.nanmean(template[order]["fluxerr"][keep]*template[order]["fluxerr"][keep]) / NORDERS

        for i in range(spectra['nspectra']) :
            keep = np.isfinite(spectra[fluxkey][order][i])
            if len(spectra[fluxkey][order][i][keep]) :
                sigma_i = np.nanstd(spectra[fluxkey][order][i][keep])
                var_global += sigma_i*sigma_i
                var_meas_global += np.nanmean(spectra[fluxerrkey][order][i][keep]*spectra[fluxerrkey][order][i][keep]) / NORDERS

    return var_template, var_meas_template, var_global, var_meas_global



def calculate_telluric_spectra(spectra, template, plot=False) :
    
    speed_of_light_in_kps = constants.c / 1000.

    telluric_sf_fluxes, telluric_sf_fluxerrs = [], []
    telluric_rest_fluxes, telluric_rest_fluxerrs = [], []
    telluric_fluxes, telluric_fluxerrs = [], []

    for order in range(NORDERS) :
        telluric_sf_fluxes.append([])
        telluric_sf_fluxerrs.append([])
        telluric_rest_fluxes.append([])
        telluric_rest_fluxerrs.append([])
        telluric_fluxes.append([])
        telluric_fluxerrs.append([])

    for order in range(NORDERS) :
        order_template = template[order]
        
        telluric_wl = order_template["wl"]
        telluric_flux = order_template["flux"]
        telluric_fluxerr = order_template["fluxerr"]
        
        keep = np.isfinite(telluric_wl) * np.isfinite(telluric_flux) * np.isfinite(telluric_fluxerr)
        
        if len(telluric_wl[keep]) > 300 :
            ftell = interp1d(telluric_wl[keep], telluric_flux[keep], kind='cubic', bounds_error=False, fill_value=np.nan)
            ferrtell = interp1d(telluric_wl[keep], telluric_fluxerr[keep], kind='cubic', bounds_error=False, fill_value=np.nan)
        else :
            ftell = lambda x: x*0. + np.nan
            ferrtell = lambda x: x*0. + np.nan
            
        template[order]["telluric_flux"] = ftell(telluric_wl)

        #plt.plot(telluric_wl, ftell(telluric_wl), 'k-')
       
        wave = spectra["common_wl"][order]
            
        for i in range(spectra['nspectra']) :

            if "windows" in spectra.keys() :
                windows = spectra["windows"][order][i]
            else :
                windows = [[common_wl[0],common_wl[-1]]]

            telluric_sf_flux, telluric_sf_fluxerr = np.ones_like(wave), np.zeros_like(wave)
            telluric_rest_flux, telluric_rest_fluxerr = np.ones_like(wave), np.zeros_like(wave)

            vel_shift = (spectra["rvs"][i] - spectra["bervs"][i])
            # undo relativistic calculation
            wave_sf = wave / np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
            
            telluric_sf_flux = interp_spectrum(wave_sf, telluric_wl[keep], telluric_flux[keep], windows, kind='cubic')
            telluric_sf_fluxerr = interp_spectrum(wave_sf, telluric_wl[keep], telluric_fluxerr[keep], windows, kind='cubic')

            telluric_rest_flux = interp_spectrum(wave, telluric_wl[keep], telluric_flux[keep], windows, kind='cubic')
            telluric_rest_fluxerr = interp_spectrum(wave, telluric_wl[keep], telluric_fluxerr[keep], windows, kind='cubic')
            
            
            #telluric_sf_flux = ftell(wave_sf)
            #telluric_sf_fluxerr = ferrtell(wave_sf)
            
            #telluric_rest_flux = ftell(wave)
            #telluric_rest_fluxerr = ferrtell(wave)
            if plot :
                plt.plot(wave,telluric_sf_flux,'.',alpha=0.5)

            telluric_sf_fluxes[order].append(telluric_sf_flux)
            telluric_sf_fluxerrs[order].append(telluric_sf_fluxerr)
            telluric_rest_fluxes[order].append(telluric_rest_flux)
            telluric_rest_fluxerrs[order].append(telluric_rest_fluxerr)
            
            telluric_i_flux = deepcopy(spectra["rest_fluxes"][order][i])
            telluric_i_fluxerr = deepcopy(spectra["rest_fluxerrs"][order][i])
            telluric_fluxes[order].append(telluric_i_flux)
            telluric_fluxerrs[order].append(telluric_i_fluxerr)

            
    if plot :
        plt.show()
        
    spectra["telluric_sf_fluxes"] = telluric_sf_fluxes
    spectra["telluric_sf_fluxerrs"] = telluric_sf_fluxerrs
    
    spectra["telluric_rest_fluxes"] = telluric_rest_fluxes
    spectra["telluric_rest_fluxerrs"] = telluric_rest_fluxerrs

    spectra["telluric_fluxes"] = telluric_fluxes
    spectra["telluric_fluxerrs"] = telluric_fluxerrs
    
    return spectra, template



def remove_telluric_spectra(spectra, template=None) :

    for order in range(NORDERS) :
        if template :
            order_template = template[order]

            template_telluric_flux = template[order]["telluric_flux"]

            template[order]["flux"] /= template_telluric_flux
            template[order]["fluxerr"] /= template_telluric_flux
            template[order]["fluxerr_model"] /= template_telluric_flux

            for j in range(len(template[order]["flux_arr"])) :
                template[order]["flux_arr"][j] /= template_telluric_flux
                template[order]["flux_residuals"][j] /= template_telluric_flux

        for i in range(spectra['nspectra']) :

            spectra["rest_fluxes"][order][i] /= spectra["telluric_rest_fluxes"][order][i]
            spectra["rest_fluxerrs"][order][i] /= spectra["telluric_rest_fluxes"][order][i]
            spectra["sf_fluxes"][order][i] /= spectra["telluric_sf_fluxes"][order][i]
            spectra["sf_fluxerrs"][order][i] /= spectra["telluric_sf_fluxes"][order][i]

    if template :
        return spectra, template
    else :
        return spectra


def recover_telluric_spectra(spectra, template=None) :

    for order in range(NORDERS) :
        if template :
            order_template = template[order]

            template_telluric_flux = template[order]["telluric_flux"]

            template[order]["flux"] *= template_telluric_flux
            template[order]["fluxerr"] *= template_telluric_flux
            template[order]["fluxerr_model"] *= template_telluric_flux

            for j in range(len(template[order]["flux_arr"])) :
                template[order]["flux_arr"][j] *= template_telluric_flux
                template[order]["flux_residuals"][j] *= template_telluric_flux

        for i in range(spectra['nspectra']) :
            spectra["rest_fluxes"][order][i] *= spectra["telluric_rest_fluxes"][order][i]
            spectra["rest_fluxerrs"][order][i] *= spectra["telluric_rest_fluxes"][order][i]
            spectra["sf_fluxes"][order][i] *= spectra["telluric_sf_fluxes"][order][i]
            spectra["sf_fluxerrs"][order][i] *= spectra["telluric_sf_fluxes"][order][i]

    if template :
        return spectra, template
    else :
        return spectra




def calculate_stellar_spectra(spectra, template, plot=False) :
    
    speed_of_light_in_kps = constants.c / 1000.

    stellar_sf_fluxes, stellar_sf_fluxerrs = [], []
    stellar_rest_fluxes, stellar_rest_fluxerrs = [], []

    for order in range(NORDERS) :
        stellar_sf_fluxes.append([])
        stellar_sf_fluxerrs.append([])
        stellar_rest_fluxes.append([])
        stellar_rest_fluxerrs.append([])

    for order in range(NORDERS) :
        order_template = template[order]

        stellar_wl = order_template["wl"]
        stellar_flux = order_template["flux"]
        stellar_fluxerr = order_template["fluxerr"]
        
        keep = np.isfinite(stellar_wl) * np.isfinite(stellar_flux) * np.isfinite(stellar_fluxerr)
        
        if len(stellar_wl[keep]) > 300 :
            fstellar = interp1d(stellar_wl[keep], stellar_flux[keep], kind='cubic', bounds_error=False, fill_value=np.nan)
            ferrstellar = interp1d(stellar_wl[keep], stellar_fluxerr[keep], kind='cubic', bounds_error=False, fill_value=np.nan)
        else :
            fstellar = lambda x: x*0. + np.nan
            ferrstellar = lambda x: x*0. + np.nan
            
        template[order]["stellar_flux"] = fstellar(stellar_wl)

        #plt.plot(stellar_wl, fstellar(stellar_wl), 'k-')
       
        wave_sf = spectra["common_wl"][order]
            
        for i in range(spectra['nspectra']) :

            if "windows" in spectra.keys() :
                windows = spectra["windows"][order][i]
            else :
                windows = [[common_wl[0],common_wl[-1]]]

            stellar_sf_flux, stellar_sf_fluxerr = np.ones_like(wave_sf), np.zeros_like(wave_sf)
            stellar_rest_flux, stellar_rest_fluxerr = np.ones_like(wave_sf), np.zeros_like(wave_sf)

            vel_shift = (spectra["rvs"][i] - spectra["bervs"][i])
            # undo relativistic calculation
            wave = wave_sf * np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
            
            stellar_sf_flux = interp_spectrum(wave_sf, stellar_wl[keep], stellar_flux[keep], windows, kind='cubic')
            stellar_sf_fluxerr = interp_spectrum(wave_sf, stellar_wl[keep], stellar_fluxerr[keep], windows, kind='cubic')

            stellar_rest_flux = interp_spectrum(wave, stellar_wl[keep], stellar_flux[keep], windows, kind='cubic')
            stellar_rest_fluxerr = interp_spectrum(wave, stellar_wl[keep], stellar_fluxerr[keep], windows, kind='cubic')
            
            if plot :
                plt.plot(wave, stellar_sf_flux, '.', alpha=0.5)

            stellar_sf_fluxes[order].append(stellar_sf_flux)
            stellar_sf_fluxerrs[order].append(stellar_sf_fluxerr)
            stellar_rest_fluxes[order].append(stellar_rest_flux)
            stellar_rest_fluxerrs[order].append(stellar_rest_fluxerr)
            
    if plot :
        plt.show()
        
    spectra["stellar_sf_fluxes"] = stellar_sf_fluxes
    spectra["stellar_sf_fluxerrs"] = stellar_sf_fluxerrs
    spectra["stellar_rest_fluxes"] = stellar_rest_fluxes
    spectra["stellar_rest_fluxerrs"] = stellar_rest_fluxerrs

    return spectra, template


def get_zero_drift_containers(inputdata) :
    
    drifts, drifts_err = np.zeros(len(inputdata)), np.zeros(len(inputdata))
    output = []
    for i in range(len(inputdata)) :
        hdr = fits.getheader(inputdata[i])
        loc = {}
        loc["FILENAME"] = inputdata[i] # Wavelength sol absolute CCF FP Drift [km/s]
        loc["WFPDRIFT"] = 'None' # Wavelength sol absolute CCF FP Drift [km/s]
        loc["RV_WAVFP"] = 'None' # RV measured from wave sol FP CCF [km/s]
        loc["RV_SIMFP"] = 'None' # RV measured from simultaneous FP CCF [km/s]
        loc["RV_DRIFT"] = drifts[i] # RV drift between wave sol and sim. FP CCF [km/s]
        loc["RV_DRIFTERR"] = drifts_err[i] # RV drift error between wave sol and sim. FP CCF [km/s]
        output.append(loc)
    
    return output


def run_igrins_ccf(reduced, ccf_mask, drifts, telluric_rv=False, normalize_ccfs=True, save_output=True, source_rv=0., ccf_width=100, vel_sampling=1.8, run_analysis=True, output_template="", tel_mask="", h2o_mask="", verbose=False, plot=False) :
    """
        Description: wrapper function to run an optimal CCF analysis of a time series of SPIRou spectra.
        This function run the following steps:
        1. Reduce a time series of SPIRou spectra
        2. Calculate CCF for the template spectrum
        3. Calculate the CCF for each reduced spectrum in the time series (including star, and tellurics)
        4. Run the CCF template matching analysis on the CCF time series data.
        """
    
    loc = {}

    fluxkey, fluxerrkey = reduced["fluxkey"], reduced["fluxerrkey"]
    waveskey, wavekey = reduced["waveskey"], reduced["wavekey"]
    
    array_of_spectra = reduced["array_of_spectra"]
    spectra, template = reduced["spectra"], reduced["template"]
    #telluric_template = reduced["telluric_template"]
    
    ccf_params, ccfmask = reduced["ccf_params"], reduced["ccfmask"]
    tell_ccf_params, tell_ccfmask = reduced["tell_ccf_params"], reduced["tell_ccfmask"]
    h2o_ccf_params, h2o_ccfmask = reduced["h2o_ccf_params"], reduced["h2o_ccfmask"]
    
    if verbose :
        print("******************************")
        print("STEP: calculating CCFs ...")
        print("******************************")

    order_subset_for_mean_ccf = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 34, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    if telluric_rv :
        order_subset_for_mean_h2occf = [1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 34, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52]
        #order_subset_for_mean_h2occf = [1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 32, 34, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, 51, 52]
        order_subset_for_mean_tellccf = [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        #order_subset_for_mean_tellccf = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    fluxes_tmpl, waves_sf_tmpl = [], []
    for order in range(NORDERS) :
        fluxes_tmpl.append(template[order]['flux'])
        waves_sf_tmpl.append(template[order]['wl'])
    fluxes_tmpl = np.array(fluxes_tmpl, dtype=float)
    waves_sf_tmpl = np.array(waves_sf_tmpl, dtype=float)
    base_header = deepcopy(array_of_spectra["spectra"][0]["header"])

    # run ccf on template
    template_ccf = ccf_lib.run_ccf_eder(reduced["ccf_params"], waves_sf_tmpl, fluxes_tmpl, base_header, reduced["ccfmask"], rv_drifts={}, targetrv=ccf_params["SOURCE_RV"], valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=False, plot=False, verbose=False)

    # If no source RV is provided then adopt the one measured from the template
    #if source_rv != 0 :
    #    if np.abs(source_rv - template_ccf["header"]['RV_OBJ']) > 2*vel_sampling :
    #        print("WARNING: measure RV on template is different than provided RV")

    ccf_params["SOURCE_RV"] = template_ccf["header"]['RV_OBJ']
    ccf_params["CCF_WIDTH"] = 7 * template_ccf["header"]['CCFMFWHM']
    if telluric_rv :
        tell_ccf_params["CCF_WIDTH"] = ccf_params["CCF_WIDTH"]
        h2o_ccf_params["CCF_WIDTH"] = ccf_params["CCF_WIDTH"]

    if verbose :
        print("Template RV={0:.5f} km/s CCF width={1:.0f} km/s".format(ccf_params["SOURCE_RV"], ccf_params["CCF_WIDTH"]))
    if plot :
        templ_legend = "Template of {}".format(template_ccf["header"]["OBJECT"].replace(" ",""))
        plt.plot(template_ccf['RV_CCF'], template_ccf['MEAN_CCF'], "-", color='green', lw=2, label=templ_legend, zorder=2)

    calib_rv, drift_rv,  = [], []
    tell_rv, h2o_rv = [], []
    mean_fwhm, mean_tell_fwhm, mean_h2o_fwhm = [], [], []
    sci_ccf_file_list = []
    tell_ccf_file_list, h2o_ccf_file_list = [], []

    for i in range(spectra['nspectra']) :
    
        if verbose :
            print("Running CCF on file {0}/{1} -> {2}".format(i,spectra['nspectra']-1,os.path.basename(spectra['filenames'][i])))

        rv_drifts = drifts[i]

        fluxes, waves_sf = [], []
        tellfluxes, waves = [], []
        for order in range(NORDERS) :
            fluxes.append(spectra[fluxkey][order][i])
            waves_sf.append(spectra[waveskey][order][i])
            tellfluxes.append(spectra["telluric_fluxes"][order][i])
            waves.append(spectra["common_wl"][order])

        fluxes = np.array(fluxes, dtype=float)
        waves_sf = np.array(waves_sf, dtype=float)
        tellfluxes = np.array(tellfluxes, dtype=float)
        waves = np.array(waves, dtype=float)

        # run main routine to process ccf on science fiber
        header = array_of_spectra["spectra"][i]["header"]

        # run an adpated version of the ccf codes using reduced spectra as input
        sci_ccf = ccf_lib.run_ccf_eder(ccf_params, waves_sf, fluxes, header, ccfmask, rv_drifts=rv_drifts, filename=spectra['filenames'][i], targetrv=ccf_params["SOURCE_RV"], valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

        sci_ccf_file_list.append(os.path.abspath(sci_ccf["file_path"]))

        calib_rv.append(sci_ccf["header"]['RV_OBJ'])
        mean_fwhm.append(sci_ccf["header"]['CCFMFWHM'])
        drift_rv.append(sci_ccf["header"]['RV_DRIFT'])
        
        if telluric_rv :
            tell_header = deepcopy(array_of_spectra["spectra"][i]["header"])
            h2o_header = deepcopy(array_of_spectra["spectra"][i]["header"])
            
            # run a adpated version fo the ccf codes using reduced spectra as input
            h2o_ccf = ccf_lib.run_ccf_eder(h2o_ccf_params, waves, tellfluxes, h2o_header, h2o_ccfmask, filename=spectra['filenames'][i],valid_orders=order_subset_for_mean_h2occf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

            tell_ccf = ccf_lib.run_ccf_eder(tell_ccf_params, waves, tellfluxes, tell_header, tell_ccfmask, filename=spectra['filenames'][i],valid_orders=order_subset_for_mean_tellccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

            tell_ccf_file_list.append(os.path.abspath(tell_ccf["file_path"]))
            h2o_ccf_file_list.append(os.path.abspath(h2o_ccf["file_path"]))
            
            tell_rv.append(tell_ccf["header"]['RV_OBJ'])
            h2o_rv.append(h2o_ccf["header"]['RV_OBJ'])
            
            mean_tell_fwhm.append(tell_ccf["header"]['CCFMFWHM'])
            mean_h2o_fwhm.append(h2o_ccf["header"]['CCFMFWHM'])
            
            if plot :
                if i == spectra['nspectra'] - 1 :
                    tellegend, h20legend = "Other tellurics", r"H$_2$O"
                else :
                    tellegend, h20legend = None, None
                plt.plot(tell_ccf['RV_CCF'],tell_ccf['MEAN_CCF'], "--", color='#d62728', label=tellegend)
                plt.plot(h2o_ccf['RV_CCF'],h2o_ccf['MEAN_CCF'], ":", color='#1f77b4', label=h20legend)
        else :
            tell_rv.append(np.nan)
            h2o_rv.append(np.nan)

        if verbose :
            print("Spectrum: {0} DATE={1} Sci_RV={2:.5f} km/s RV_DRIFT={3:.5f} km/s Tell_RV={4:.5f} km/s H2O_RV={5:.5f} km/s".format(os.path.basename(spectra['filenames'][i]), sci_ccf["header"]["DATE"], sci_ccf["header"]['RV_OBJ'], sci_ccf["header"]["RV_DRIFT"], tell_rv[i], h2o_rv[i]))
            
        if plot :
            if i == spectra['nspectra'] - 1 :
                scilegend = "{}".format(sci_ccf["header"]["OBJECT"].replace(" ",""))
            else :
                scilegend = None
            #plt.plot(esci_ccf['RV_CCF'],sci_ccf['MEAN_CCF']-esci_ccf['MEAN_CCF'], "--", label="spectrum")
            plt.plot(sci_ccf['RV_CCF'], sci_ccf['MEAN_CCF'], "-", color='#2ca02c', alpha=0.5, label=scilegend, zorder=1)

    mean_fwhm = np.array(mean_fwhm)
    velocity_window = 1.5*np.nanmedian(mean_fwhm)

    if telluric_rv :
        mean_tell_fwhm = np.array(mean_tell_fwhm)
        mean_h2o_fwhm = np.array(mean_h2o_fwhm)

    if plot :
        plt.xlabel('Velocity [km/s]')
        plt.ylabel('CCF')
        plt.legend()
        plt.show()

        calib_rv, median_rv = np.array(calib_rv), np.nanmedian(calib_rv)
        plt.plot(spectra["bjds"], (calib_rv  - median_rv), 'o', color='#2ca02c', label="Sci RV = {0:.4f} km/s".format(median_rv))
        plt.plot(spectra["bjds"], (mean_fwhm  - np.nanmean(mean_fwhm)), '--', color='#2ca02c', label="Sci FWHM = {0:.4f} km/s".format(np.nanmean(mean_fwhm)))
        
        drift_rv = np.array(drift_rv)
        
        mean_drift, sigma_drift = np.nanmedian(drift_rv), np.nanstd(drift_rv)
        plt.plot(spectra["bjds"], drift_rv, '.', color='#ff7f0e', label="Inst. FP drift = {0:.4f}+/-{1:.4f} km/s".format(mean_drift,sigma_drift))

        if telluric_rv :
            tell_rv = np.array(tell_rv)
            zero_telldrift, sigma_telldrift = np.nanmedian(tell_rv), np.nanstd(tell_rv)
            h2o_rv = np.array(h2o_rv)
            zero_h2odrift, sigma_h2odrift = np.nanmedian(h2o_rv), np.nanstd(h2o_rv)
            plt.plot(spectra["bjds"], (tell_rv  - zero_telldrift), '-', color='#d62728', label="Telluric drift = {0:.4f}+/-{1:.4f} km/s".format(zero_telldrift, sigma_telldrift))
            plt.plot(spectra["bjds"], (h2o_rv  - zero_h2odrift), '-', color='#1f77b4', label="H2O drift = {0:.4f}+/-{1:.4f} km/s".format(zero_h2odrift, sigma_h2odrift))
            plt.plot(spectra["bjds"], (mean_tell_fwhm  - np.nanmean(mean_tell_fwhm)), ':', color='#d62728', label="Telluric FWHM = {0:.4f} km/s".format(np.nanmean(mean_tell_fwhm)))
            plt.plot(spectra["bjds"], (mean_h2o_fwhm  - np.nanmean(mean_h2o_fwhm)), ':', color='#1f77b4', label="H2O FWHM = {0:.4f} km/s".format(np.nanmean(mean_h2o_fwhm)))
            
        plt.xlabel(r"BJD")
        plt.ylabel(r"Velocity [km/s]")
        plt.legend()
        plt.show()

    if run_analysis :
        if verbose :
            print("Running CCF analysis: velocity_window = {0:.3f} km/s".format(velocity_window))
        
        # exclude orders with strong telluric absorption
        #exclude_orders = [-1]  # to include all orders
        exclude_orders = [0,1,2,3,24,25,26,27,28,29,30,31,32,33,36,37,38,52,53]
        
        obj = sci_ccf["header"]["OBJECT"].replace(" ","")
        drs_version = "IGRINSv0.0"

        obj_ccf = ccf2rv.run_ccf_analysis(sci_ccf_file_list, ccf_mask, obj=obj, drs_version=drs_version, snr_min=10., velocity_window=velocity_window, pixel_size_in_kps=vel_sampling, dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=True, save_ccf_fitsfile=True, exclude_orders = exclude_orders, plot=plot, verbose=verbose)
        
        loc["OBJ_CCF"] = obj_ccf

        if telluric_rv and tel_mask != "":
            exclude_tell_orders = [0,1,2,3,4,5,6,7,23,25,26,27,28,29,30,31,33,52,53]
            #exclude_tell_orders = [-1]
            tell_velocity_window = 1.5*np.nanmedian(mean_tell_fwhm)
            tell_ccf = ccf2rv.run_ccf_analysis(tell_ccf_file_list, tel_mask, obj=obj, drs_version=drs_version, snr_min=10.,velocity_window=tell_velocity_window, pixel_size_in_kps=vel_sampling, dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, exclude_orders = exclude_tell_orders, plot=plot, verbose=verbose)
            loc["TELL_CCF"] = tell_ccf

        if telluric_rv and h2o_mask != "":
            exclude_h2_orders = [0,8,9,10,11,12,13,14,15,16,17,18,26,27,28,29,30,31,33,35,36,37,38,45,46,53]
            #exclude_h2_orders = [-1]
            h2o_velocity_window = 1.5*np.nanmedian(mean_h2o_fwhm)
            h2o_ccf = ccf2rv.run_ccf_analysis(h2o_ccf_file_list, h2o_mask, obj=obj, drs_version=drs_version, snr_min=10.,velocity_window=h2o_velocity_window, pixel_size_in_kps=vel_sampling, dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, exclude_orders = exclude_h2_orders, plot=plot, verbose=verbose)
            loc["H2O_CCF"] = h2o_ccf

    return loc


def convolve_spectrum(wl, flux, fluxerr, to_resolution, from_resolution=None):
    """
    Spectra resolution smoothness/degradation.

    If "from_resolution" is not specified or its equal to "to_resolution", then the spectrum
    is convolved with the instrumental gaussian defined by "to_resolution".

    If "from_resolution" is specified, the convolution is made with the difference of
    both resolutions in order to degrade the spectrum.
    """
    if from_resolution is not None and from_resolution <= to_resolution:
        raise Exception("This method cannot deal with final resolutions that are bigger than original")

    wl, flux, fluxerr = __convolve_spectrum(wl, flux, fluxerr, to_resolution, from_resolution=from_resolution)
    
    return wl, flux, fluxerr


def __fwhm_to_sigma(fwhm):
    """
    Calculate the sigma value from the FWHM.
    """
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    return sigma



def __convolve_spectrum(waveobs, flux, err, to_resolution, from_resolution=None):
    """
        Spectra resolution smoothness/degradation. Procedure:

        1) Define a bin per measure which marks the wavelength range that it covers.
        2) For each point, identify the window segment to convolve by using the bin widths and the FWHM.
        3) Build a gaussian using the sigma value and the wavelength values of the spectrum window.
        4) Convolve the spectrum window with the gaussian and save the convolved value.

        If "from_resolution" is not specified or its equal to "to_resolution", then the spectrum
        is convolved with the instrumental gaussian defined by "to_resolution".

        If "to_resolution" is specified, the convolution is made with the difference of
        both resolutions in order to degrade the spectrum.
    """
    if from_resolution is not None and from_resolution <= to_resolution:
        raise Exception("This method cannot deal with final resolutions that are bigger than original")

    total_points = len(waveobs)
    convolved_flux = np.zeros(total_points)
    convolved_err = np.zeros(total_points)

    last_reported_progress = -1

    # Consider the wavelength of the measurements as the center of the bins
    # Calculate the wavelength distance between the center of each bin
    wave_distance = waveobs[1:] - waveobs[:-1]
    # Define the edge of each bin as half the wavelength distance to the bin next to it
    edges_tmp = waveobs[:-1] + 0.5 * (wave_distance)
    # Define the edges for the first and last measure which where out of the previous calculations
    first_edge = waveobs[0] - 0.5*wave_distance[0]
    last_edge = waveobs[-1] + 0.5*wave_distance[-1]
    # Build the final edges array
    edges = np.array([first_edge] + edges_tmp.tolist() + [last_edge])

    # Bin width
    bin_width = edges[1:] - edges[:-1]          # width per pixel

    # FWHM of the gaussian for the given resolution
    if from_resolution is None:
        # Convolve using instrumental resolution (smooth but not degrade)
        fwhm = waveobs / to_resolution
    else:
        # Degrade resolution
        fwhm = __get_fwhm(waveobs, from_resolution, to_resolution)
    sigma = __fwhm_to_sigma(fwhm)
    # Convert from wavelength units to bins
    fwhm_bin = fwhm / bin_width

    # Round number of bins per FWHM
    nbins = np.ceil(fwhm_bin) #npixels

    # Number of measures
    nwaveobs = len(waveobs)

    # In theory, len(nbins) == len(waveobs)
    for i in np.arange(len(nbins)):
        current_nbins = 2 * nbins[i] # Each side
        current_center = waveobs[i] # Center
        current_sigma = sigma[i]

        # Find lower and uper index for the gaussian, taking care of the current spectrum limits
        lower_pos = int(max(0, i - current_nbins))
        upper_pos = int(min(nwaveobs, i + current_nbins + 1))

        # Select only the flux values for the segment that we are going to convolve
        flux_segment = flux[lower_pos:upper_pos+1]
        err_segment = err[lower_pos:upper_pos+1]
        waveobs_segment = waveobs[lower_pos:upper_pos+1]

        # Build the gaussian corresponding to the instrumental spread function
        gaussian = np.exp(- ((waveobs_segment - current_center)**2) / (2*current_sigma**2)) / np.sqrt(2*np.pi*current_sigma**2)
        gaussian = gaussian / np.sum(gaussian)

        # Convolve the current position by using the segment and the gaussian
        if flux[i] > 0:
            # Zero or negative values are considered as gaps in the spectrum
            only_positive_fluxes = flux_segment > 0
            weighted_flux = flux_segment[only_positive_fluxes] * gaussian[only_positive_fluxes]
            current_convolved_flux = weighted_flux.sum()
            convolved_flux[i] = current_convolved_flux
        else:
            convolved_err[i] = 0.0

        if err[i] > 0:
            # * Propagate error Only if the current value has a valid error value assigned
            #
            # Error propagation considering that measures are dependent (more conservative approach)
            # because it is common to find spectra with errors calculated from a SNR which
            # at the same time has been estimated from all the measurements in the same spectra
            #
            weighted_err = err_segment * gaussian
            current_convolved_err = weighted_err.sum()
            #current_convolved_err = np.sqrt(np.power(weighted_err, 2).sum()) # Case for independent errors
            convolved_err[i] = current_convolved_err
        else:
            convolved_err[i] = 0.0

    return waveobs, convolved_flux, convolved_err
