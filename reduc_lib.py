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
import ccf_lib

import telluric_lib

from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms


def load_array_of_igrins_spectra(inputdata, rvfile="", object_name="None", apply_berv=True, silent=True, convolve_spectra=False, plot_diagnostics=False, plot=False, verbose=False) :

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

        wl_mean = []
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
            
        if plot_diagnostics :
            if i == 0 :
                objectname = hdr['OBJECT']
            bjd.append(spectrum['BJD_mid'])
            snr.append(maxsnr)
            airmass.append(spectrum['airmass'])
            berv.append(spectrum['BERV'])

        if verbose :
            print("Spectrum ({0}/{1}): {2} OBJ={3} BJD={4:.6f} SNR={5:.1f} EXPTIME={6:.0f}s BERV={7:.3f} km/s".format(i,len(inputdata)-1,inputdata[i],hdr['OBJECT'],spectrum['BJD_mid'],maxsnr,hdr['EXPTIME'],spectrum['BERV']))

        for order in range(len(wave)) :
            
            wl = deepcopy(wave[order])

            wlc = 0.5 * (wl[0] + wl[-1])

            if convolve_spectra :
                flux = deepcopy(convolve(fluxes[order], gausskernel))
            else :
                flux = deepcopy(fluxes[order])

            fluxerr = np.sqrt(fluxvar[order])

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
            wl_mean.append(wlc)
                
        if plot :
            plt.xlabel(r"wavelength [nm]")
            plt.xlabel(r"flux")
            plt.show()
            exit()

        spectrum['wlmean'] = np.array(wl_mean)

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
    wlmin, wlmax = np.full(54,-1e20), np.full(54,+1e20)

    for order in range(54) :
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
    
    if verbose :
        print("Loading data")
    
    loc = {}

    spectra = array_of_spectra["spectra"]

    filenames, dates = [], []
    bjds, airmasses, rvs, rverrs, bervs = [], [], [], [], []
    wl_mean = []

    ref_spectrum = spectra[ref_index]

    nspectra = len(spectra)
    loc['nspectra'] = nspectra
    snrs = []
    waves, waves_sf, vels = [], [], []
    fluxes, fluxerrs, orders = [], [], []
    wl_out, wlsf_out, vel_out = [], [], []
    hdr = []

    for order in range(54) :
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
            
        wl_mean.append(spectrum['wlmean'])
        
        for order in range(len(spectrum['wl'])) :
            mean_snr = np.nanmean(spectrum['flux'][order] / spectrum['fluxerr'][order])
            
            snrs[order].append(mean_snr)

            orders[order].append(spectrum['order'][order])

            waves[order].append(spectrum['wl'][order])
            waves_sf[order].append(spectrum['wl_sf'][order])
            vels[order].append(spectrum['vels'][order])
            if i==0 :
                wl_out.append(spectrum['wl'][order])
                wlsf_out.append(spectrum['wl_sf'][order])
                vel_out.append(spectrum['vels'][order])
            
            fluxes[order].append(spectrum['flux'][order])
            fluxerrs[order].append(spectrum['fluxerr'][order])

    bjds  = np.array(bjds, dtype=float)
    airmasses  = np.array(airmasses, dtype=float)
    rvs  = np.array(rvs, dtype=float)
    rverrs  = np.array(rverrs, dtype=float)
    bervs  = np.array(bervs, dtype=float)
    wl_mean  = np.array(wl_mean, dtype=float)

    for order in range(54) :
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
    loc["wl_mean"] = wl_mean

    loc["snrs"] = snrs
    loc["orders"] = orders
    loc["waves"] = waves
    loc["waves_sf"] = waves_sf
    loc["vels"] = vels

    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs

    # set base wavelength from first spectrum
    loc["wl"] = np.array(wl_out)
    loc["wl_sf"] = np.array(wlsf_out)
    loc["vel"] = np.array(vel_out)

    loc = get_wlmin_wlmax(loc, edge_size=edge_size)
    
    return loc


def get_gapfree_windows(spectra, max_vel_distance=3.0, min_window_size=120., fluxkey="fluxes", velkey="vels", wavekey="waves", verbose=False) :
    
    windows = []
    
    for order in range(54) :
        windows.append([])
    
    for order in range(54) :
    #for order in range(38,39) :
        if verbose :
            print("Calculating windows with size > {0:.0f} km/s and with gaps < {1:.1f} km/s for order={2}".format(min_window_size,max_vel_distance, order))

        for i in range(spectra['nspectra']) :
        #for i in range(0,1) :
        
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
    
    for order in range(54) :
        
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

    for order in range(54) :
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

    for order in range(54) :
        aligned_waves.append([])

        sf_fluxes.append([])
        sf_fluxerrs.append([])
        rest_fluxes.append([])
        rest_fluxerrs.append([])

    for order in range(54) :
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
    
    for order in range(54) :
    #for order in range(30,31) :

        if verbose:
            print("Reducing spectra for order {0} / 48 ...".format(order))

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
                fluxerr = order_template["fluxerr"]
            else:
                fluxes = order_template["flux_arr_sub"] * order_template["flux"]
                fluxerr = order_template["flux_arr_sub"] * order_template["fluxerr"]

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


def reduce_timeseries_of_spectra(inputdata, ccf_mask, object_name="", stellar_spectrum_file="", source_rv=0., max_gap_size=8.0, min_window_size=200., align_spectra=True, vel_sampling=1.8, nsig_clip = 3.0, ccf_width=150, output_template="", verbose=False) :
    
    """
        Description: function to process a series of IGRINS spectra. The processing consist of
        the following steps:
        """
    
    if verbose :
        print("******************************")
        print("STEP: Loading IGRINS data ...")
        print("******************************")
 
    # First load spectra into a container
    array_of_spectra = load_array_of_igrins_spectra(inputdata, object_name=object_name, convolve_spectra=False, plot_diagnostics=False, plot=False, verbose=True)

    # Then load data into order vectors -- it is more efficient to work the reduction order-by-order
    spectra = get_spectral_data(array_of_spectra, verbose=True)
    
   
    saturated_tellurics = [[1820,1823],[1897.7,1907.2],[1911.046,1916.747],[1951.25,1959.714],[1998.01,2014.61],[2434.979,2439.186],[2450.8,2451.64],[2479.381,2480.0]]

    for order in range(54) :
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

    # Calculate stellar spectrum from an input template
    spectra = calculate_stellar_spectra(spectra, stellar_spectrum_file)

    # Remove stellar contribution from input spectra
    spectra = remove_stellar_spectra(spectra)

    # Calculate template in the rest frame
    template = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", wavekey="common_wl", update_spectra=False, plot=False, verbose=verbose)
    
    spectra, template = normalize_spectra(spectra, template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", plot=False)

    """
    for order in range(54) :
        continuum_flux = spectra["continuum_rest_fluxes"][order]
        color = [order/54,1-order/54,1-order/54]
        order_template = template[order]
        plt.plot(template[order]["wl"],continuum_flux,'-',alpha=0.5)
    """

    # Calculate stellar spectrum from an input template
    spectra, template = calculate_telluric_spectra(spectra, template)
    #spectra, template = remove_telluric_spectra(spectra, template)

    exit()

    # recover continuum
    spectra, template = recover_continuum(spectra, template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs")

    for order in range(54) :
        continuum_flux = spectra["continuum_rest_fluxes"][order]
        color = [order/54,1-order/54,1-order/54]
        order_template = template[order]
        telluric_flux = continuum_flux * order_template["telluric_flux"]
        plt.plot(order_template["wl"], telluric_flux, '-',alpha=0.5, lw=2)
    
    print("Continuum + Star + telluric removed:", calculate_variances(spectra, template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs")[0])

    #spectra, template = recover_telluric_spectra(spectra, template)

    ##### Calculate telluric spectrum from observed spectra
    #--> implement: spectra, template =  calculate_telluric_spectra(spectra, template=template)
    #--> implement: spectra, template =  remove_telluric_spectra(spectra, template=template)
    #--> implement: spectra, template =  recover_telluric_spectra(spectra, template=template)

    #--> implement: loop to iterate Continuum/Stellar/Telluric and minimize chi-square
    
    for order in range(54) :
        color = [order/54,1-order/54,1-order/54]
        order_template = template[order]
        plt.plot(template[order]["wl"],template[order]["flux"],'.',alpha=0.5)

    print("Continuum + Star removed:", calculate_variances(spectra, template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs")[0])

    # recover stellar
    spectra, template = calculate_stellar_spectra(spectra, stellar_spectrum_file, template=template)
    spectra, template = recover_stellar_spectra(spectra, template=template)

    for order in range(54) :
        color = [order/54,1-order/54,1-order/54]
        order_template = template[order]
        plt.plot(template[order]["wl"],template[order]["flux"],':',alpha=0.8)

    print("Continuum removed:", calculate_variances(spectra, template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs")[0])

    plt.show()
    
    exit()

    # Remove stellar contribution from input spectra
    spectra, template = remove_stellar_spectra(spectra, template=template)
    # Calculate template
    template = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=False, plot=False, verbose=verbose)
    spectra, template = calculate_stellar_spectra(spectra, stellar_spectrum_file, template=template)

    print("Template from star removed:", calculate_variances(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs")[0])
    # calculate continuum in the template and normalize all spectra by this continuum
    print("Continuum removed:", calculate_variances(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs")[0])

    # recover stellar
    spectra, template = recover_stellar_spectra(spectra, template=template)
    print("Stellar recovered:", calculate_variances(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs")[0])
    # recover continuum
    spectra, template = recover_continuum(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs")
    print("Continuum+Stellar recovered:", calculate_variances(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs")[0])

    # calculate continuum in the template and normalize all spectra by this continuum
    #spectra, template = normalize_spectra(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", plot=False)
    

    exit()



    exit()


    print("Calculating template spectra ...")
    # First run reduce routine to create template, calibrate all spectra to match template, and then
    # apply a sigma-clip
    template = reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=True, fluxkey=fluxkey, fluxerrkey=fluxerrkey, wavekey=wavekey, update_spectra=True, plot=False, verbose=verbose)




    if object_spectrum_file != "" :
        spectra, template = recover_stellar_spectra(spectra, template)

    # Uncomment below to recover continuum
    #spectra, template = recover_continuum(spectra, template, fluxkey=fluxkey, fluxerrkey=fluxerrkey)
    
    exit()

    # Calculate statistical weights based on the time series dispersion 1/sig^2
    spectra = calculate_weights(spectra, template, use_err_model=False, plot=False)

    # Start dealing with CCF related parameters and construction of a weighted mask
    # load science CCF parameters
    ccf_params = ccf_lib.set_ccf_params(ccf_mask)

    # update ccf width with input value
    ccf_params["CCF_WIDTH"] = float(ccf_width)

    templ_fluxes, templ_fluxerrs, templ_wave = [], [], []
    for order in range(54) :
        order_template = template[order]
        templ_fluxes.append(order_template["flux"])
        templ_fluxerrs.append(order_template["fluxerr"])
        templ_wave.append(order_template["wl"])
                
    templ_fluxes = np.array(templ_fluxes, dtype=float)
    templ_fluxerrs = np.array(templ_fluxerrs, dtype=float)
    templ_wave = np.array(templ_wave, dtype=float)

    ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, templ_wave, templ_fluxes, templ_fluxerrs, spectra["weights"], median=True, remove_lines_with_nans=True, source_rv=source_rv, verbose=False)

    order_subset_for_mean_ccf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

    base_header = deepcopy(array_of_spectra["spectra"][0]["header"])

    template_ccf = ccf_lib.run_ccf_eder(ccf_params, templ_wave, templ_fluxes, base_header, ccfmask, targetrv=source_rv, valid_orders=order_subset_for_mean_ccf, normalize_ccfs=True, output=False, plot=True, verbose=False)
    
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

    loc = {}

    loc["array_of_spectra"] = array_of_spectra
    loc["spectra"] = spectra
    loc["template"] = template

    loc["ccf_params"] = ccf_params
    loc["ccfmask"] = ccfmask
    
    loc["fluxkey"], loc["fluxerrkey"] = fluxkey, fluxerrkey
    loc["waveskey"], loc["wavekey"] =  waveskey, wavekey

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


def calculate_stellar_spectra(spectra, stellar_spectrum_file, template=None) :
    
    speed_of_light_in_kps = constants.c / 1000.

    wl1, wl2 = 1400, 2500
    
    stellar_spectrum = igrinslib.load_spirou_s1d_template(stellar_spectrum_file, wl1=wl1, wl2=wl2, to_resolution=0, normalize=True, plot=False)

    fobj = interp1d(stellar_spectrum['wl'], stellar_spectrum['flux'], kind='cubic')
    ferrobj = interp1d(stellar_spectrum['wl'], stellar_spectrum['fluxerr'], kind='cubic')

    stellar_sf_fluxes, stellar_sf_fluxerrs = [], []
    stellar_rest_fluxes, stellar_rest_fluxerrs = [], []

    for order in range(54) :
        stellar_sf_fluxes.append([])
        stellar_sf_fluxerrs.append([])
        stellar_rest_fluxes.append([])
        stellar_rest_fluxerrs.append([])

    for order in range(54) :
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

    for order in range(54) :
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

    for order in range(54) :
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

    for order in range(54) :
        if template :
            order_template = template[order]
            keep = np.isfinite(template[order]["flux"])
            if len(template[order]["flux"][keep]) :
                sigma = np.nanstd(template[order]["flux"][keep])
                var_template += sigma*sigma
                var_meas_template += np.nanmean(template[order]["fluxerr"][keep]*template[order]["fluxerr"][keep]) / 54

        for i in range(spectra['nspectra']) :
            keep = np.isfinite(spectra[fluxkey][order][i])
            if len(spectra[fluxkey][order][i][keep]) :
                sigma_i = np.nanstd(spectra[fluxkey][order][i][keep])
                var_global += sigma_i*sigma_i
                var_meas_global += np.nanmean(spectra[fluxerrkey][order][i][keep]*spectra[fluxerrkey][order][i][keep]) / 54

    return var_template, var_meas_template, var_global, var_meas_global



def calculate_telluric_spectra(spectra, template) :
    
    speed_of_light_in_kps = constants.c / 1000.

    telluric_sf_fluxes, telluric_sf_fluxerrs = [], []
    telluric_rest_fluxes, telluric_rest_fluxerrs = [], []

    for order in range(54) :
        telluric_sf_fluxes.append([])
        telluric_sf_fluxerrs.append([])
        telluric_rest_fluxes.append([])
        telluric_rest_fluxerrs.append([])

    for order in range(54) :
        order_template = template[order]
        
        telluric_wl = order_template["wl"]
        telluric_flux = order_template["flux"]
        telluric_fluxerr = order_template["fluxerr"]
        
        ftell = interp1d(telluric_wl, telluric_flux, kind='cubic')
        ferrtell = interp1d(telluric_wl, telluric_fluxerr, kind='cubic')

        template[order]["telluric_flux"] = ftell(telluric_wl)
        print(order)
        plt.plot(telluric_wl,telluric_flux)
        plt.show()
        wave = spectra["common_wl"][order]
            
        for i in range(spectra['nspectra']) :

            telluric_sf_flux, telluric_sf_fluxerr = np.ones_like(wave), np.zeros_like(wave)
            telluric_rest_flux, telluric_rest_fluxerr = np.ones_like(wave), np.zeros_like(wave)

            vel_shift = (spectra["rvs"][i] - spectra["bervs"][i])
            # undo relativistic calculation
            wave_sf = wave * np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
            
            keep = (wave_sf > wave[0]) & (wave_sf < wave[-1])
            telluric_sf_flux[keep] = ftell(wave_sf[keep])
            telluric_sf_fluxerr[keep] = ferrtell(wave_sf[keep])

            telluric_rest_flux = ftell(wave)
            telluric_rest_fluxerr = ferrtell(wave)

            telluric_sf_fluxes[order].append(telluric_sf_flux)
            telluric_sf_fluxerrs[order].append(telluric_sf_fluxerr)
            telluric_rest_fluxes[order].append(telluric_rest_flux)
            telluric_rest_fluxerrs[order].append(telluric_rest_fluxerr)

    spectra["telluric_sf_fluxes"] = telluric_sf_fluxes
    spectra["telluric_sf_fluxerrs"] = telluric_sf_fluxerrs
    spectra["telluric_rest_fluxes"] = telluric_rest_fluxes
    spectra["telluric_rest_fluxerrs"] = telluric_rest_fluxerrs

    return spectra, template



def remove_telluric_spectra(spectra, template=None) :

    for order in range(54) :
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

    for order in range(54) :
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
