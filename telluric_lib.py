# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 24 2021
    
    Description: library for telluric modeling and fitter
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """


import os, sys

from scipy import constants

import matplotlib.pyplot as plt
import numpy as np

from telfit import Modeler
from telfit import humidity_to_ppmv

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from copy import copy, deepcopy

import astropy.io.fits as fits

import time

def species_colors() :
    colors = {}
    colors["h2o"]='#1f77b4'
    colors["co2"]='#ff7f0e'
    colors["ch4"]='#2ca02c'
    colors["o2"]='#d62728'
    colors["co"]='#9467bd'
    colors["o3"]='#8c564b'
    colors["n2o"]='#e377c2'
    colors["no"]='#7f7f7f'
    colors["so2"]='#bcbd22'
    colors["no2"]='#17becf'
    colors["nh3"]='blue'
    colors["hno3"]='green'
    return colors


# function to interpolate spectrum
def interp_spectrum(wl_out, wl_in, flux_in, kind='cubic') :
    wl_in_copy = deepcopy(wl_in)
    
    # create interpolation function for input data
    f = interp1d(wl_in_copy, flux_in, kind=kind)
    
    # create mask for valid range of output vector
    mask = wl_out > wl_in[0]
    mask &= wl_out < wl_in[-1]
    
    flux_out = np.full_like(wl_out, np.nan)
    
    # interpolate data
    flux_out[mask] = f(wl_out[mask])
    return flux_out


def __get_fwhm(lambda_peak, from_resolution, to_resolution):
    """
    Calculate the FWHM of the gaussian needed to convert
    a spectrum from one resolution to another at a given wavelength point.
    """
    if from_resolution <= to_resolution:
        raise Exception("This method cannot deal with final resolutions that are equal or bigger than original")
    from_delta_lambda = (1.0*lambda_peak) / from_resolution
    to_delta_lambda = (1.0*lambda_peak) / to_resolution
    fwhm = np.sqrt(to_delta_lambda**2 - from_delta_lambda**2)
    return fwhm


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


def convolve_spectrum(spectrum, to_resolution, from_resolution=None):
    """
    Spectra resolution smoothness/degradation.

    If "from_resolution" is not specified or its equal to "to_resolution", then the spectrum
    is convolved with the instrumental gaussian defined by "to_resolution".

    If "from_resolution" is specified, the convolution is made with the difference of
    both resolutions in order to degrade the spectrum.
    """
    if from_resolution is not None and from_resolution <= to_resolution:
        raise Exception("This method cannot deal with final resolutions that are bigger than original")

    waveobs, flux, err = __convolve_spectrum(spectrum['wl'], spectrum['flux'], spectrum['fluxerr'], to_resolution, from_resolution=from_resolution)
    convolved_spectrum = {}
    convolved_spectrum['wl'] = waveobs
    convolved_spectrum['flux'] = flux
    convolved_spectrum['fluxerr'] = err
    return convolved_spectrum


# Function to fit telluric model using LBLRTM
def calibrate_telluric_model(spectrum, tell_spectrum, plot=False, verbose=False) :
    
    norders = len(spectrum['wl'])
    
    spectrum["telluric_model"] = np.empty_like(spectrum['wl'])
    spectrum["flux_model"] = np.empty_like(spectrum['wl'])
    spectrum["reduced_flux"] = np.empty_like(spectrum['wl'])
    spectrum["stellar_flux"] = np.empty_like(spectrum['wl'])

    spectrum["telluric_calib_pfit"] = []
    
    for order in range(norders) :
        if verbose :
            print("Calibrating telluric model for order=",order)
    
        mask = ~np.isnan(spectrum['flux'][order])
        mask &= spectrum['flux'][order] > 0

        wl = spectrum['wl'][order][mask]
        flux = spectrum['flux'][order][mask]

        # Interpolate input telluric model spectrum to the same wl grid as in observed data
        order_tell_model = {}
        order_tell_model['wl'] = deepcopy(wl)
        tell_nan_filter = ~np.isnan(tell_spectrum['telluric_model'])
        tell_model_wl = tell_spectrum['wl'][tell_nan_filter]
        tell_model_flux = tell_spectrum['telluric_model'][tell_nan_filter]
        order_tell_model['flux'] = interp_spectrum(wl, tell_model_wl, tell_model_flux, kind='cubic')
        order_tell_model['fluxerr'] = np.zeros_like(wl)
        
        tell_fit = fit_continuum(wl, flux, order_tell_model)
            
        spectrum["telluric_calib_pfit"].append(tell_fit['pfit'])
        
        spectrum["flux_model"][order] = tell_fit['flux_model']
        spectrum["telluric_model"][order] = tell_fit['telluric_model']
        spectrum["reduced_flux"][order] = tell_fit['reduced_flux']
        spectrum["stellar_flux"][order] = 1.0 + tell_fit['residuals']
        
        if plot :
            plt.plot(wl, flux,'-', label="Observed spectrum")
            plt.plot(wl, tell_fit['flux_model'],'-', label="Telluric model")
    
    if plot :
        plt.ylabel("flux")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        plt.show()
    
    # output spectrum dict
    return spectrum


# Function to fit telluric model using LBLRTM
def grid_search_fit_telluric_model(spectrum, plot=False, verbose=False) :

    # retrieve first guess
    pars = get_guess_params(spectrum)

    pars["humidity"] = 38.4
    pars["o2"] = 18800.0
    pars["co2"] = 29.9
    pars["ch4"] = 0.28

    # Tune parameters
    # Tune profile EditProfile("temperature" or "pressure", profile_height, profile_value)
    
    
    # grid search to fit humidity
    #pars = grid_search(pars, spectrum, parameter_name='humidity', order=12, wl0=1144.83, wlf=1146.11, ini=0., end=100., step=10., plot=False, verbose=True)
    #pars = grid_search(pars, spectrum, parameter_name='humidity', order=12, wl0=1144.83, wlf=1146.11, ini=pars['humidity']-10.0, end=pars['humidity']+10.0, step=2., plot=False, verbose=True)
    #pars = grid_search(pars, spectrum, parameter_name='humidity', order=12, wl0=1144.83, wlf=1146.11, ini=pars['humidity']-2.0, end=pars['humidity']+2.0, step=0.1, plot=True, verbose=True)
    
    # H2O probes:
    #h2o_probes = {12:[1144.83, 1146.11], 10:[1112.79, 1120.92]}
    h2o_probes = {12:[1144.83, 1146.11], 10:[1112.79, 1120.92]}
    #    h2o_probes = [[1491.72,1506.01],[1731.31,1763.70],[1979.04,1995.14],[2096.21,2129.79]]

    print("Initial humidity: ",pars['humidity'])
    pars = grid_search(pars, spectrum, parameter_name='humidity', probes=h2o_probes, ini=0., end=100., step=10., plot=False, verbose=True)
    pars = grid_search(pars, spectrum, parameter_name='humidity',probes=h2o_probes, ini=pars['humidity']-10.0, end=pars['humidity']+10.0, step=1., plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='humidity',probes=h2o_probes, ini=pars['humidity']-1.0, end=pars['humidity']+1.0, step=0.1, plot=plot, verbose=verbose)

    # O2 probe2
    o2_probes = {18:[1260.00, 1271.35]}
    pars = grid_search(pars, spectrum, parameter_name='o2', probes=o2_probes, ini=0., end=50000., step=10000., plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='o2', probes=o2_probes, ini=pars['o2']-10000., end=pars['o2']+10000., step=1000., plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='o2', probes=o2_probes, ini=pars['o2']-1000., end=pars['o2']+1000., step=100., plot=plot, verbose=verbose)

    # CO2 probe2
    co2_probes = {42:[2045.28, 2077.22]}
    pars = grid_search(pars, spectrum, parameter_name='co2', probes=co2_probes, ini=0., end=100., step=10., plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='co2', probes=o2_probes, ini=pars['co2']-10., end=pars['co2']+10., step=1., plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='co2', probes=o2_probes, ini=pars['co2']-1., end=pars['co2']+1., step=0.1, plot=plot, verbose=verbose)

    # CH4 probes
    ch4_probes = {45:[2275.12, 2280.97], 46:[2315.51, 2321.70]}
    pars = grid_search(pars, spectrum, parameter_name='ch4',probes=ch4_probes, ini=0., end=10., step=1.0, plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='ch4', probes=o2_probes, ini=pars['ch4']-1., end=pars['ch4']+1., step=0.1, plot=plot, verbose=verbose)
    pars = grid_search(pars, spectrum, parameter_name='ch4', probes=o2_probes, ini=pars['ch4']-0.1, end=pars['ch4']+0.1, step=0.01, plot=plot, verbose=verbose)

    #plot_species(pars, spectrum, 45, wl0=0., wlf=0.)

    # generate model and save results into spectrum dict
    #spectrum = calculate_telluric_model(pars, spectrum, fit_instr_calib=True, plot=False, verbose=True)

    # output spectrum dict
    return pars


def generate_telluric_model(spectrum, pars, wavekey="WAVE",rv_sampling=1.0, rv_overshoot=150., plot=False, verbose=False) :

    rv_step_forward = (1.0 + rv_sampling / (constants.c / 1000.))
    rv_step_backward = (1.0 - rv_sampling / (constants.c / 1000.))
    
    out_spectrum = {}

    out_spectrum["header0"] = spectrum["header"]
    out_spectrum["header1"] = spectrum["header"]
    
    out_spectrum["pars"] = pars
    
    norders = len(spectrum[wavekey])
    
    out_spectrum["wl"] = []
    out_spectrum["telluric_model"] = []
    
    for order in range(norders) :
        
        wave = spectrum['wl'][order]
        
        wl0 = wave[0] * (1.0 - rv_overshoot / (constants.c / 1000.))
        wlf = wave[-1] * (1.0 + rv_overshoot / (constants.c / 1000.))

        wlout = []
        wl = wl0 * rv_step_forward
        while wl < wlf * rv_step_backward :
            wlout.append(wl)
            wl *= rv_step_forward
        wlout = np.array(wlout)
        
        if verbose :
            print("Calculating telluric model for order=",order, " wl0=",wlout[0],"wlf=",wlout[-1], "npoints=", len(wlout))
    
        order_tell_model = telluric_model(pars, wl0, wlf, outwave=wlout)
        
        out_spectrum["wl"].append(order_tell_model['wl'])
        out_spectrum["telluric_model"].append(order_tell_model['flux'])
        if plot :
            plt.plot(order_tell_model['wl'], order_tell_model['flux'])

    if plot :
        plt.xlabel("wavelength (nm)")
        plt.ylabel("flux")
        plt.show()
    
    return out_spectrum


def plot_species(pars, spectrum, order, wl0=0., wlf=0., plot=False, verbose=True) :
    
    pars = instrumental_calibration(pars, spectrum, order, wl0=wl0, wlf=wlf, plot=False, verbose=verbose)

    # Load data
    wl, flux = get_chunk(spectrum, order, wl0, wlf)

    flux_without_continuum = (flux - (pars["shift"] + pars["a_calib"] * wl + pars["b_calib"] * wl * wl)) / pars["scale"]
    plt.plot(wl, flux_without_continuum, '-', lw=0.5, alpha=0.5, color='purple', label="Observed spectrum")
    
    h2o_model = calculate_spec_model (wl, pars, molecule='h2o')
    plt.plot(wl, h2o_model['telluric_model'], color=species_colors()['h2o'], alpha=0.5, label='H2O')
    
    co2_model = calculate_spec_model (wl, pars, molecule='co2')
    plt.plot(wl, co2_model['telluric_model'], color=species_colors()['co2'], alpha=0.5, label='CO2')
        
    ch4_model = calculate_spec_model (wl, pars, molecule='ch4')
    plt.plot(wl, ch4_model['telluric_model'], color=species_colors()['ch4'], alpha=0.5, label='CH4')
        
    o2_model = calculate_spec_model (wl, pars, molecule='o2')
    plt.plot(wl, o2_model['telluric_model'], color=species_colors()['o2'], alpha=0.5, label='O2')
        
    h2o_sum = np.sum(1.0 - h2o_model['telluric_model'])
    co2_sum = np.sum(1.0 - co2_model['telluric_model'])
    ch4_sum = np.sum(1.0 - ch4_model['telluric_model'])
    o2_sum = np.sum(1.0 - o2_model['telluric_model'])

    if verbose :
        print("order=", order, "R=",pars['resolution'], "H2O=", h2o_sum, "CO2=", co2_sum, "CH4=", ch4_sum, "O2=", o2_sum)

    plt.xlabel("wavelength (nm)")
    plt.ylabel("flux")
    plt.legend()
    plt.show()


# function to initilize parameters to calculate models
def get_guess_params(spectrum) :

    pars = {}
    
    pars["point_angle"] = 90 - spectrum['header']["ZDSTART"]
    pars["humidity"] = spectrum['header']["HUMIDITY"]
    pars["pressure"] = 731
    pars["temperature"] = 5 + 273.15
    pars["latitude"] = spectrum['header']["OBSLAT"]
    pars["altitude"] = spectrum['header']["OBSALT"]/1000.

    pars["co2"]=28.
    pars["o3"]=3.9e-2
    pars["n2o"]=0.32
    pars["co"]=0.14
    pars["ch4"]=1.8
    pars["o2"]=2.1e5
    pars["no"]=1.1e-19
    pars["so2"]=1e-4
    pars["no2"]=1e-4
    pars["nh3"]=1e-4
    pars["hno3"]=5.6e-4

    pars["resolution"] = 45000.

    pars["scale"] = 1.0
    pars["shift"] = 0.0
    pars["a_calib"] = 0.0
    pars["b_calib"] = 0.0

    pars["ppmv"] = humidity_to_ppmv(pars["humidity"], pars["temperature"], pars["pressure"])

    return pars


# main wrapper function to call LBLRTM to calculate telluric model
def telluric_model(pars, wl0, wlf, outwave=[]) :

    modeler = Modeler()

    tell_model = modeler.MakeModel(pressure=pars["pressure"], temperature=pars["temperature"], lowfreq=1e7/wlf, highfreq=1e7/wl0, angle=pars["point_angle"], humidity=pars["humidity"],co2=pars["co2"], o3=pars["o3"], n2o=pars["n2o"], co=pars["co"], ch4=pars["ch4"], o2=pars["o2"], no=pars["no"], so2=pars["so2"], no2=pars["no2"], nh3=pars["nh3"], hno3=pars["hno3"], lat=pars["latitude"], alt=pars["altitude"], vac2air=False)

    tell = {}
        
    if len(outwave) :
        tell['wl'] = outwave
        tell['flux'] = interp_spectrum(outwave, tell_model.x, tell_model.y, kind='cubic')
    else :
        tell['wl'], tell['flux'] = tell_model.x, tell_model.y

    tell['fluxerr'] = np.zeros_like(tell_model.y)

    return tell


# Function to populate SPIRou orders with telluric models calculated using input pars
def calculate_telluric_model(pars, spectrum, wavekey="WAVE", fit_instr_calib=True, plot=False, verbose=False) :

    norders = len(spectrum[wavekey])
    spectrum["telluric_model"] = np.empty_like(spectrum['wl'])
    spectrum["flux_model"] = np.empty_like(spectrum['wl'])
    spectrum["reduced_flux"] = np.empty_like(spectrum['wl'])
    spectrum["stellar_flux"] = np.empty_like(spectrum['wl'])

    spectrum["pars"] = []
    for order in range(norders) :
        spectrum["pars"].append(pars)
    
    for order in range(norders) :
    
        # make local copy of pars
        pars_copy = deepcopy(pars)
        
        if verbose :
            print("Calculating telluric model for order=",order)
    
        wave = spectrum['wl'][order]
        fluxes = spectrum['flux'][order]

        if fit_instr_calib :
            mask = ~np.isnan(spectrum['flux'][order])
            mask &= spectrum['flux'][order] > 0

            wl = spectrum['wl'][order][mask]
            flux = spectrum['flux'][order][mask]
        
            wl0, wlf = wl[0] - 0.1, wl[-1] + 0.1

            order_tell_model = telluric_model(pars_copy, wl0, wlf, outwave=wl)
            
            tell_fit = fit_continuum(wl, flux, order_tell_model)
            
            pars_copy = update_instrument_calib_pars(tell_fit['pfit'], pars_copy)
            
            spectrum["pars"][order] = pars_copy
        
        model = calculate_spec_model(wave, pars_copy)
            
        spectrum["flux_model"][order] = model['flux_model']

        spectrum["telluric_model"][order] = model['telluric_model']

        reduced_flux = (fluxes  - model['continuum']) / pars_copy["scale"]
        spectrum["reduced_flux"][order] = reduced_flux
        
        stellar_flux = 1.0 + (fluxes - model['flux_model'])

        spectrum["stellar_flux"][order] = stellar_flux
        
        if plot :
            plt.plot(wave, fluxes,'-', label="Observed spectrum")
            plt.plot(wave, model['flux_model'],'-', label="Telluric model")
        
    if plot :
        plt.ylabel("flux")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        plt.show()
    
    return spectrum


# function to calculate spectrum model using the instrumental calibration params
def calculate_spec_model (wl, pars, molecule=""):
    
    # make local copy of pars
    pars_copy = deepcopy(pars)
    
    # if molecule is specified then make model only for that particular molecule
    if molecule != "":
        if molecule == "h2o" :
            molecule = "humidity"
        pars_copy["humidity"] = 0.
        pars_copy["co2"] = 0.
        pars_copy["o3"] = 0.
        pars_copy["n2o"] = 0.
        pars_copy["co"] = 0.
        pars_copy["ch4"] = 0.
        pars_copy["o2"] = 0.
        pars_copy["no"] = 0.
        pars_copy["so2"] = 0.
        pars_copy["no2"] = 0.
        pars_copy["nh3"] = 0.
        pars_copy["hno3"] = 0.
        pars_copy[molecule] = pars[molecule]
    
    wl0, wlf = wl[0] - 0.1, wl[-1] + 0.1
    tell = telluric_model(pars_copy, wl0, wlf, outwave=wl)
    tell_conv = convolve_spectrum(tell, pars_copy['resolution'])
    continuum = pars_copy['shift']  + pars_copy['a_calib'] * wl + pars_copy['b_calib'] * wl * wl
    outmodel = tell_conv['flux'] * pars_copy['scale'] + continuum
    loc = {}
    loc["flux_model"] = outmodel
    loc["telluric_model"] = (outmodel - continuum) / pars_copy['scale']
    loc["continuum"] = continuum
    return loc


# function to calculate chi-square for grid-search fit algorithm
def chi_square(data, model) :
    
    mask = ~np.isnan(data)
    mask &= ~np.isnan(model)
    mask &= data != 0
    mask &= model != 0
    residuals = data[mask] - model[mask]
    chisqr = np.sum((residuals**2)/np.abs(model[mask]))
    return chisqr


# function to fit continuum, spectral resolution and scale factor.
def fit_continuum(wl, flux, tell, plot=False) :

    def spec_model (wave, resolution, scale, shift, a, b):
        tell_conv = convolve_spectrum(tell, resolution)
        outmodel = tell_conv['flux'] * scale + (shift + a * wave + b * wave * wave)
        return outmodel
    
    guess = [70000., 1.01, 0.01, 0.0001, 0.0001]
    
    pfit, pcov = curve_fit(spec_model, wl, flux, p0=guess)
    yfit = spec_model(wl, *pfit)
    
    if plot :
        plt.plot(wl, flux, 'o', label='data')
        plt.plot(wl, yfit, '-', label='telluric * cont fit')
        plt.plot(wl, yfit/tell_flux,'--', label='continuum fit')
        plt.legend()
        plt.show()
    loc = {}
    loc["flux_model"] = yfit
    loc["telluric_model"] = (yfit - (pfit[2] + pfit[3] * wl + pfit[4] * wl * wl)) / pfit[1]
    loc["flux_without_continuum"] = (flux - (pfit[2] + pfit[3] * wl + pfit[4] * wl * wl)) / pfit[1]
    loc["reduced_flux"] = loc["flux_without_continuum"] / loc["telluric_model"]
    loc["residuals"] = flux - yfit
    loc["pfit"] = pfit
    return loc


# function to update instrumental calibration parameters from fit
def update_instrument_calib_pars(pfit, pars) :
    pars['resolution'] = pfit[0]
    pars['scale'] = pfit[1]
    pars['shift'] = pfit[2]
    pars['a_calib'] = pfit[3]
    pars['b_calib'] = pfit[4]
    return pars


# function to perform instrumental calibration of spectrum
def instrumental_calibration(pars, spectrum, order, wl0=0., wlf=0., plot=False, verbose=False) :
    
    # Load data
    wl, flux = get_chunk(spectrum, order, wl0, wlf)

    # calculate telluric model
    tell = telluric_model(pars, wl[0] - 0.1, wl[-1] + 0.1, outwave=wl)

    # fit continuum and resolution
    cont_corr_conv_tell = fit_continuum(wl, flux, tell)

    pars = update_instrument_calib_pars(cont_corr_conv_tell["pfit"], pars)

    if verbose :
        print("pfit=", cont_corr_conv_tell["pfit"])

    if plot :
        #plt.plot(wl, flux, label="SPIRou spectrum")
        #plt.plot(wl, cont_corr_conv_tell["flux_model"], label="continuum corrected telluric")
        
        plt.plot(wl, cont_corr_conv_tell["flux_without_continuum"], label="flux without continuum")
        plt.plot(wl, cont_corr_conv_tell["telluric_model"], label="telluric model")

        plt.plot(wl, cont_corr_conv_tell["residuals"], label="residuals 2")

        plt.xlabel("wavelength (nm)")
        plt.ylabel("flux")

        plt.legend()
        plt.show()

    return pars


def get_chunk(spectrum, order, wl0=0., wlf=0.) :
    if wl0!=0 and wlf!=0 :
        wlmask = spectrum['wl'][order] > wl0
        wlmask &= spectrum['wl'][order] < wlf
        flux = spectrum['flux'][order][wlmask]
        wl = spectrum['wl'][order][wlmask]
    else :
        flux = spectrum['flux'][order]
        wl = spectrum['wl'][order]
        
    finite = np.isfinite(flux)
    return wl[finite], flux[finite]


# function to run grid search algorithm to fit parameter
def grid_search(pars, spectrum, parameter_name='humidity', probes={12:[1144.83,1146.11]}, ini=1.0, end=1.5, step=0.02, plot=False, verbose=False) :
    if verbose:
        print ("Running grid_search for parameter: ", parameter_name)
        iter=0
    
    #Initialize min_par, model, and chisqr values with current parameters
    min_chisqr = 0.
    best_param_value = pars[parameter_name]
    min_models = {}

    for order in probes.keys() :
        # Load data
        wl, flux = get_chunk(spectrum, order, probes[order][0], probes[order][1])
        specmodel = calculate_spec_model(wl, pars)
        min_models[order] = specmodel
        min_chisqr += chi_square(flux, specmodel["flux_model"])

    # make a working copy of pars
    pars_copy = deepcopy(pars)
    
    # initialize variable with ini value
    parameter_value = ini

    # loop over grid
    while parameter_value < end :
        
        # update parameter value in pars
        pars_copy[parameter_name] = parameter_value
        models_tmp = {}
        chisqr = 0.
        for order in probes.keys() :
            
            # Load data
            wl, flux = get_chunk(spectrum, order, probes[order][0], probes[order][1])
            
            # calculate model
            tell = telluric_model(pars_copy, wl[0] - 0.1, wl[-1] + 0.1, outwave=wl)
        
            # fit continuum, spectral resolution
            cont_inst_fit = fit_continuum(wl, flux, tell)
            
            # save current parameters for each probe
            models_tmp[order] = cont_inst_fit
            
            # increment chi-square
            chisqr += chi_square(flux, cont_inst_fit['flux_model'])
        
        if verbose :
            print("Iter=",iter, parameter_name, "=", parameter_value, "chisqr=", chisqr)
            iter +=1
    
        # if chisqr is better then update minimum values
        if chisqr < min_chisqr :
            min_chisqr, best_param_value = chisqr, parameter_value
            min_models = models_tmp

        # increment variable
        parameter_value += step
    
    if verbose :
        print("Final fit: param_value=", best_param_value, "min chisqr=", min_chisqr)
    
    # update par dictionary
    pars[parameter_name] = best_param_value

    for order in probes.keys() :
        pars = update_instrument_calib_pars(min_models[order]['pfit'], pars)
    
    # plot new model to see results
    if plot :
        for order in probes.keys() :
            wl, flux = get_chunk(spectrum, order, probes[order][0], probes[order][1])
            plt.plot(wl, flux)
            #plt.plot(wl, tell['flux'])
            plt.plot(wl, min_models[order]['flux_model'])
        plt.show()
    
    return pars


# Function to populate SPIRou orders with telluric models calculated using input pars
def plot_telluric_model(spectrum, wavekey="WAVE") :
    norders = len(spectrum[wavekey])
    
    for order in range(norders) :
        wave = spectrum['wl'][order]
        flux = spectrum['flux'][order]
        
        model = spectrum["flux_model"][order]
        telluric = spectrum["telluric_model"][order]
        reduced_flux = spectrum["reduced_flux"][order]
        stellar_flux = 1.0 - spectrum["stellar_flux"][order]

        plt.plot(wave, reduced_flux,'-', alpha=0.7)
        plt.plot(wave, telluric,'-', color='#ff7f0e', lw=1.5, alpha=0.7)
        plt.plot(wave, stellar_flux,'-')

    plt.ylabel("flux")
    plt.xlabel("Wavelength (nm)")
    plt.show()


def write_telluric_spectrum(spectrum, output, wavekey='wl', tell_modelkey='telluric_model') :

    header = spectrum["header"]
    telluric_header = fits.Header()
    
    pars = spectrum["pars"]
    telluric_header.set('h2o', pars["humidity"], "Humidity")
    telluric_header.set('co2', pars["co2"], "co2")
    telluric_header.set('o3', pars["o3"], "o3")
    telluric_header.set('n2o', pars["n2o"], "n2o")
    telluric_header.set('co', pars["co"], "co")
    telluric_header.set('ch4', pars["ch4"], "ch4")
    telluric_header.set('o2', pars["o2"], "o2")
    telluric_header.set('so2', pars["so2"], "so2")
    telluric_header.set('no2', pars["no2"], "no2")
    telluric_header.set('nh3', pars["nh3"], "nh3")
    telluric_header.set('hno3', pars["hno3"], "hno3")
    
    maxlen = 0
    for order in range(len(spectrum[wavekey])) :
        if len(spectrum[wavekey][order]) > maxlen :
            maxlen = len(spectrum[wavekey][order])

    wl_data = np.full((len(spectrum[wavekey]),maxlen), np.nan)
    telluric_model = np.full((len(spectrum[wavekey]),maxlen), np.nan)

    for order in range(len(spectrum[wavekey])) :
        for i in range(len(spectrum[wavekey][order])) :
            wl_data[order][i] = spectrum[wavekey][order][i]
            telluric_model[order][i] = spectrum[tell_modelkey][order][i]


    outhdulist = []
    
    primary_hdu = fits.PrimaryHDU(header=header)
    outhdulist.append(primary_hdu)

    hdu_wl = fits.ImageHDU(data=wl_data, name="WAVE", header=header)
    outhdulist.append(hdu_wl)

    hdu_telluric_model = fits.ImageHDU(data=telluric_model, name="TELLURIC_MODEL", header=telluric_header)
    outhdulist.append(hdu_telluric_model)

    mef_hdu = fits.HDUList(outhdulist)

    mef_hdu.writeto(output, overwrite=True)


def read_telluric_spectrum(input, clean_nans=True) :
    spectrum = {}
    
    hdu = fits.open(input)
    header = hdu[0].header
    header1 = hdu[1].header
    telluric_header = hdu['TELLURIC_MODEL'].header

    if clean_nans :
        spectrum['wl'], spectrum['telluric_model'] = [], []
        norders = len(hdu['WAVE'].data)
        for order in range(norders) :
            nanmask = ~np.isnan(hdu['TELLURIC_MODEL'].data[order])
            spectrum['wl'].append(hdu['WAVE'].data[order][nanmask])
            spectrum['telluric_model'].append(hdu['TELLURIC_MODEL'].data[order][nanmask])
    else :
        spectrum['wl'] = hdu['WAVE'].data
        spectrum['telluric_model'] = hdu['TELLURIC_MODEL'].data

    return spectrum


def get_telluric_model_from_grid(telluric_grid_path, airmass=1.5, pwv="050", wl1=0, wl2=3.e4, wl_in_nm=True, to_resolution=0) :
    
    airmass_dir = "pwv_R300k_airmass{:.1f}/".format(airmass)
    pwv_file = "LBL_A15_s0_w{}_R0300000_T.fits".format(pwv)

    tell_model_file = os.path.join(telluric_grid_path, airmass_dir+pwv_file)

    loc = {}
    
    if os.path.exists(tell_model_file) :
        hdu = fits.open(tell_model_file)
        
        factor = 1.0
        if wl_in_nm :
            factor = 1000.
        wl = hdu[1].data["lam"] * factor
        wlmask = wl > wl1
        wlmask &= wl < wl2
        
        loc["wl"] = wl[wlmask]
        loc["trans"] = hdu[1].data["trans"][wlmask]
        loc["mnstrans"] = hdu[1].data["mnstrans"][wlmask]
        loc["plstrans"] = hdu[1].data["plstrans"][wlmask]
        
        if to_resolution :
            err = np.zeros_like(loc["wl"])
            loc["wl"], loc["trans"], err = __convolve_spectrum(loc["wl"], loc["trans"], err, to_resolution)
            
        return loc
        
    else :
        print("ERROR: failed to read file:", tell_model_file)
        exit()
    
