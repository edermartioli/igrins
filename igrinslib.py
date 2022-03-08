"""
    Created on Nov 16 2021
    
    Description: espectro_ir (infrared) library to handle IGRINS data
    
    @author: Eder Martioli <emartioli@lna.br>, <martioli@iap.fr>
    
    Laboratorio Nacional de Astrofisica, Brazil
    Institut d'Astrophysique de Paris, France
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from copy import copy, deepcopy
from astropy.io import ascii
from astropy.io import fits
from scipy import signal
import scipy.interpolate as sint
from scipy.interpolate import interp1d

import telluric_lib, reduc_lib

def igrins_order_mask():
    order_mask = [[0, -1, -1, 'H'],
             [1, 1455.05, 1459.63, 'H'],
             [2, 1459.63, 1470.94,'H'],
             [3, 1470.94, 1482.43,'H'],
             [4, 1482.43, 1494.24,'H'],
             [5, 1494.24, 1506.42,'H'],
             [6, 1506.42, 1518.76,'H'],
             [7, 1518.76, 1531.13,'H'],
             [8, 1531.13, 1543.65,'H'],
             [9, 1543.65, 1556.34,'H'],
             [10, 1556.34, 1569.62,'H'],
             [11, 1569.62, 1582.70,'H'],
             [12, 1582.70, 1596.35,'H'],
             [13, 1596.35, 1610.29,'H'],
             [14, 1610.29, 1624.04,'H'],
             [15, 1624.04, 1638.60,'H'],
             [16, 1638.60, 1653.06,'H'],
             [17, 1653.06, 1667.87,'H'],
             [18, 1667.87, 1682.99,'H'],
             [19, 1682.99, 1698.46,'H'],
             [20, 1698.46, 1714.14,'H'],
             [21, 1714.14, 1730.25,'H'],
             [22, 1730.25, 1746.55,'H'],
             [23, 1746.55, 1763.23,'H'],
             [24, 1763.23, 1780.21,'H'],
             [25, 1780.21, 1797.41,'H'],
             [26, 1797.41, 1814.65,'H'],
             [27, 1814.65, 1825.22,'H'],
             [28, -1, -1,'K'],
             [29, 1885.33, 1888.99,'K'],
             [30, 1891.26, 1909.51,'K'],
             [31, 1909.51, 1929.77,'K'],
             [32, 1929.77, 1950.61,'K'],
             [33, 1950.61, 1971.99,'K'],
             [34, 1971.99, 1992.87,'K'],
             [35, 1992.87, 2014.69,'K'],
             [36, 2014.69, 2037.70,'K'],
             [37, 2037.70, 2060.72,'K'],
             [38, 2060.72, 2084.41, 'K'],
             [39, 2084.41, 2108.41,'K'],
             [40, 2108.41, 2133.22, 'K'],
             [41, 2133.22, 2158.49, 'K'],
             [42, 2158.49, 2184.51, 'K'],
             [43, 2184.51, 2211.46, 'K'],
             [44, 2211.46, 2238.60,'K'],
             [45, 2238.60, 2266.75,'K'],
             [46, 2266.75, 2295.44,'K'],
             [47, 2295.44, 2325.01,'K'],
             [48, 2325.01, 2355.15,'K'],
             [49, 2355.15, 2386.15,'K'],
             [50, 2386.15, 2417.93,'K'],
             [51, 2417.93, 2450.57,'K'],
             [52, 2451.05, 2480.26,'K'],
             [53, -1, -1,'K']]
             
    outorders, wl0, wlf, band = [], [], [], []
    for order in order_mask:
        outorders.append(order[0])
        wl0.append(order[1])
        wlf.append(order[2])
        band.append(order[3])
    
    loc = {}
    loc['orders'] = outorders
    loc['wl0'] = wl0
    loc['wlf'] = wlf
    loc['band'] = band
    return loc


def load_spectrum(filename, standard=False) :

    orders = igrins_order_mask()
    
    spectrum = {}
    
    hdu = fits.open(filename)
    
    spectrum["header"] = hdu[0].header
    
    spectrum["wl"] = hdu["WAVE"].data
    spectrum["flux"] = hdu["FLUX"].data
    spectrum["variance"] = hdu["VARIANCE"].data
    
    if standard :
        spectrum["SPEC_FLATTENED"] = hdu["SPEC_FLATTENED"].data
        spectrum["FITTED_CONTINUUM"] = hdu["FITTED_CONTINUUM"].data
        spectrum["A0V_NORM"] = hdu["A0V_NORM"].data
        spectrum["MODEL_TELTRANS"] = hdu["MODEL_TELTRANS"].data

    for order in range(len(orders['orders'])) :
    
        wl0, wlf = orders['wl0'][order], orders['wlf'][order]
        
        mask = hdu["WAVE"].data[order] >= wl0
        mask &= hdu["WAVE"].data[order] < wlf

        spectrum["flux"][order][~mask] = np.nan
        spectrum["variance"][order][~mask] = np.nan
    
        if standard :
            spectrum["SPEC_FLATTENED"][order][~mask] = np.nan
            spectrum["FITTED_CONTINUUM"][order][~mask] = np.nan
            spectrum["A0V_NORM"][order][~mask] = np.nan
            spectrum["MODEL_TELTRANS"][order][~mask] = np.nan

    return spectrum


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
        fspec = signal.medfilt(spec, kernel_size=med_filt)
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



def load_spirou_s1d_template(filename, use_rms_error=True, wl1=0, wl2=1e20, to_resolution=0, normalize=False, cont_function='spline3', order=20, plot=False) :

    object_spectrum = {}
    
    hdu_template = fits.open(filename)
    
    keep = np.isfinite(hdu_template[1].data['flux'])
    keep &= (hdu_template[1].data['wavelength'] > wl1) & (hdu_template[1].data['wavelength'] < wl2)
    
    object_spectrum["wl"] = hdu_template[1].data['wavelength'][keep]
    object_spectrum["flux"] = hdu_template[1].data['flux'][keep]
    if use_rms_error :
        object_spectrum["fluxerr"] = hdu_template[1].data['rms'][keep]
    else :
        object_spectrum["fluxerr"] = hdu_template[1].data['eflux'][keep]

    object_spectrum["continuum"] = reduc_lib.fit_continuum(object_spectrum["wl"], object_spectrum["flux"], function=cont_function, order=order, nit=10, rej_low=1., rej_high=4., grow=1, med_filt=1, percentile_low=0., percentile_high=100.,min_points=100, xlabel="wavelength", ylabel="flux", plot_fit=plot, silent=True)
        
    if normalize :
        object_spectrum["flux"] /= object_spectrum["continuum"]
        object_spectrum["fluxerr"] /= object_spectrum["continuum"]

    if plot :
        plt.errorbar(object_spectrum["wl"], object_spectrum["flux"], yerr=object_spectrum["fluxerr"], fmt='.', alpha=0.5, label="Template spectrum")

    if to_resolution :
        object_spectrum["wl"], object_spectrum["flux"], object_spectrum["fluxerr"] = telluric_lib.__convolve_spectrum(object_spectrum["wl"], object_spectrum["flux"], object_spectrum["fluxerr"], to_resolution)

        if plot :
            plt.errorbar(object_spectrum["wl"], object_spectrum["flux"], yerr=object_spectrum["fluxerr"], fmt='.', alpha=0.5, label="Convolved spectrum R={}".format(to_resolution))

    if plot :
        plt.xlabel("wavelength [nm]",fontsize=18)
        plt.ylabel("flux",fontsize=18)
        plt.legend(fontsize=18)
        plt.show()
        
    return object_spectrum
