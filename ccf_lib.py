# -*- coding: utf-8 -*-
"""
# CODE NAME HERE
# CODE DESCRIPTION HERE
Created on 2020-03-27 at 13:42
@author: cook (adapted by E. Martioli on 2020-04-28)
"""
from astropy.io import fits
from astropy.table import vstack,Table
from astropy import units as uu
import numpy as np
import warnings
import sys
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from astropy.io.fits.verify import VerifyWarning
import warnings
from scipy import constants
import glob

helpstr = """
----------------------------------------------------------------------------
new_ccf_code.py
----------------------------------------------------------------------------
This code takes no arguments - you must edit the "variables section"  of the
code
1. Finding the correct calibrations
    a) For your observation find the date
    b) Go to your /calibDB/ directory
    c) find the correct file (closest in time?):
        BLAZE:  *blaze_{fiber}.fits
        WAVE:   *wave_night_{fiber}.fits
2. Finding masks

    a) go to your apero-drs installation directory
    b) go to data/spirou/ccf sub-directory
    c) use one of these masks
3. Two options for where you put files
    a) copy all data to a directory of your choice ({path})
        i) copy this code there and set W1='' and W2=''
        ii) or set W1={path} and W2={path}
        
    b) set the paths for your data (W1) and your mask directory (W2)
    
    Then update the filenames:
        IN_FILE: the e2dsff_C or e2dsff_tcorr _AB file
        BLAZE_FILE: the blaze file from calibDB - get the fiber correct!
        WAVE_FILE: the wave file from calibDB - get the fiber correct!
        MASK_FILE: the mask file
        
    Note there are two cases (set CASE=1 or CASE=2)
    
    For case=1 we assume your IN_FILE is a OBJ
    For case=2 we assume your IN_FILe is a FP
----------------------------------------------------------------------------
"""


# =============================================================================
# Define variables
# =============================================================================
# constants
SPEED_OF_LIGHT = 299792.458    # [km/s]

# whether to plot (True or False)
PLOT = True

def set_ccf_params(maskfile, telluric_masks=[], science_channel=True) :
    
    loc = {}
    loc["MASK_FILE"] = maskfile
    loc["TELL_MASK_FILES"] = []
    if len(telluric_masks) :
        for m in telluric_masks :
            loc["TELL_MASK_FILES"].append(m)
    loc["SOURCE_RV"] = 0.0
    # CCF is a set of Dirac functions
    #loc["KERNEL"] = None
    # boxcar length expressed in km/s
    # loc["KERNEL"] = ['boxcar', 5]
    # gaussian with e-width in km/s
    # loc["KERNEL"] = ['gaussian', 3.5]
    # supergaussian e-width + exponent
    #loc["KERNEL"] = ['supergaussian', 3.5, 4]

    if science_channel :
        loc["MASK_COLS"] = ['ll_mask_s', 'll_mask_e', 'w_mask']
        if maskfile.endswith(".nmas") :
            loc["MASK_COLS"] = ['order_mask', 'll_mask_s', 'll_mask_e', 'w_mask']
        
        # variables
        # These values are taken from the constants file
        loc["MASK_WIDTH"] = 1.7                   # CCF_MASK_WIDTH
        loc["MASK_MIN_WEIGHT"] = 0.0              # CCF_MASK_MIN_WEIGHT
        loc["CCF_STEP"] = 0.5                     # CCF_DEFAULT_STEP (or user input)
        #loc["CCF_WIDTH"] = 300                    # CCF_DEFAULT_WIDTH (or user input)
        loc["CCF_WIDTH"] = 100                     # CCF_DEFAULT_WIDTH (or user input)
        #loc["CCF_WIDTH"] = 60                     # CCF_DEFAULT_WIDTH (or user input)
        loc["CCF_RV_NULL"] = -9999.99             # CCF_OBJRV_NULL_VAL
        loc["CCF_N_ORD_MAX"] = 48                 # CCF_N_ORD_MAX
        loc["BLAZE_NORM_PERCENTILE"] = 90         # CCF_BLAZE_NORM_PERCENTILE
        loc["BLAZE_THRESHOLD"] = 0.3              # WAVE_FP_BLAZE_THRES
        loc["IMAGE_PIXEL_SIZE"] = 2.28            # IMAGE_PIXEL_SIZE
        loc["NOISE_SIGDET"] = 8.0                 # CCF_NOISE_SIGDET
        loc["NOISE_SIZE"] = 12                    # CCF_NOISE_BOXSIZE
        loc["NOISE_THRES"] = 1.0e9                # CCF_NOISE_THRES
        loc["KERNEL"] = None
    else :
        # build file paths
        loc["MASK_COLS"] = ['ll_mask_s', 'll_mask_e', 'w_mask']
        # variables
        # These values are taken from the constants file
        loc["MASK_WIDTH"] = 1.7                   # CCF_MASK_WIDTH
        loc["MASK_MIN_WEIGHT"] = 0.0              # CCF_MASK_MIN_WEIGHT
        loc["CCF_STEP"] = 0.25                     # WAVE_CCF_STEP
        loc["CCF_WIDTH"] = 6.5                    # WAVE_CCF_WIDTH
        loc["CCF_RV_NULL"] = -9999.99             # CCF_OBJRV_NULL_VAL
        loc["CCF_N_ORD_MAX"] = 48                 # WAVE_CCF_N_ORD_MAX
        loc["BLAZE_NORM_PERCENTILE"] = 90         # CCF_BLAZE_NORM_PERCENTILE
        loc["BLAZE_THRESHOLD"] = 0.3              # WAVE_FP_BLAZE_THRES
        loc["IMAGE_PIXEL_SIZE"] = 2.28            # IMAGE_PIXEL_SIZE
        loc["NOISE_SIGDET"] = 8.0                 # WAVE_CCF_NOISE_SIGDET
        loc["NOISE_SIZE"] = 12                    # WAVE_CCF_NOISE_BOXSIZE
        loc["NOISE_THRES"] = 1.0e9                # WAVE_CCF_NOISE_THRES
        loc["KERNEL"] = ['gaussian', 1.4]

    return loc

# =============================================================================
# Define functions
# =============================================================================
def read_mask(mask_file, mask_cols):
    table = Table.read(mask_file, format='ascii')
    # get column names
    oldcols = list(table.colnames)
    # rename columns
    for c_it, col in enumerate(mask_cols):
        table[oldcols[c_it]].name = col
    # return table
    return table


def get_mask(table, mask_width, mask_min, mask_units='nm'):
    ll_mask_e = np.array(table['ll_mask_e']).astype(float)
    ll_mask_s = np.array(table['ll_mask_s']).astype(float)
    ll_mask_d = ll_mask_e - ll_mask_s
    ll_mask_ctr = ll_mask_s + ll_mask_d * 0.5
    if "order_mask" in table :
        order_mask = np.array(table['order_mask']).astype(float)
    # if mask_width > 0 ll_mask_d is multiplied by mask_width/c
    if mask_width > 0:
        ll_mask_d = mask_width * ll_mask_s / SPEED_OF_LIGHT
    # make w_mask an array
    w_mask = np.array(table['w_mask']).astype(float)
    # use w_min to select on w_mask or keep all if w_mask_min >= 1
    if mask_min < 1.0:
        mask = w_mask > mask_min
        ll_mask_d = ll_mask_d[mask]
        ll_mask_ctr = ll_mask_ctr[mask]
        w_mask = w_mask[mask]
        if "order_mask" in table :
            order_mask = order_mask[mask]

    # else set all w_mask to one (and use all lines in file)
    else:
        w_mask = np.ones(len(ll_mask_d))
    # ----------------------------------------------------------------------
    # deal with the units of ll_mask_d and ll_mask_ctr
    # must be returned in nanometers
    # ----------------------------------------------------------------------
    # get unit object from mask units string
    unit = getattr(uu, mask_units)
    # add units
    ll_mask_d = ll_mask_d * unit
    ll_mask_ctr = ll_mask_ctr * unit
    # convert to nanometers
    ll_mask_d = ll_mask_d.to(uu.nm).value
    ll_mask_ctr = ll_mask_ctr.to(uu.nm).value
    # ----------------------------------------------------------------------
    # return the size of each pixel, the central point of each pixel
    #    and the weight mask
    if "order_mask" in table :
        return order_mask, ll_mask_d, ll_mask_ctr, w_mask
    else :
        return ll_mask_d, ll_mask_ctr, w_mask


def relativistic_waveshift(dv, units='km/s'):
    """
    Relativistic offset in wavelength
    default is dv in km/s
    :param dv: float or numpy array, the dv values
    :param units: string or astropy units, the units of dv
    :return:
    """
    # get c in correct units
    # noinspection PyUnresolvedReferences
    if units == 'km/s' or units == uu.km/uu.s:
        c = SPEED_OF_LIGHT
    # noinspection PyUnresolvedReferences
    elif units == 'm/s' or units == uu.m/uu.s:
        c = SPEED_OF_LIGHT * 1000
    else:
        raise ValueError("Wrong units for dv ({0})".format(units))
    # work out correction
    corrv = np.sqrt((1 + dv / c) / (1 - dv / c))
    # return correction
    return corrv


def iuv_spline(x, y, **kwargs):
    # check whether weights are set
    w = kwargs.get('w', None)
    # copy x and y
    x, y = np.array(x), np.array(y)
    # find all NaN values
    nanmask = ~np.isfinite(y)

    if np.sum(~nanmask) < 2:
        y = np.zeros_like(x)
    elif np.sum(nanmask) == 0:
        pass
    else:
        # replace all NaN's with linear interpolation
        badspline = InterpolatedUnivariateSpline(x[~nanmask], y[~nanmask],
                                                 k=1, ext=1)
        y[nanmask] = badspline(x[nanmask])
    # return spline
    return InterpolatedUnivariateSpline(x, y, **kwargs)


def fit_ccf(rv, ccf, fit_type, verbose=False):
    """
    Fit the CCF to a guassian function
    :param rv: numpy array (1D), the radial velocities for the line
    :param ccf: numpy array (1D), the CCF values for the line
    :param fit_type: int, if "0" then we have an absorption line
                          if "1" then we have an emission line
    :return result: numpy array (1D), the fit parameters in the
                    following order:
                [amplitude, center, fwhm, offset from 0 (in y-direction)]
    :return ccf_fit: numpy array (1D), the fit values, i.e. the gaussian values
                     for the fit parameters in "result"
    """
    # deal with inconsistent lengths
    if len(rv) != len(ccf):
        print('\tERROR: RV AND CCF SHAPE DO NOT MATCH')
        sys.exit()

    # deal with all nans
    if np.sum(np.isnan(ccf)) == len(ccf):
        # log warning about all NaN ccf
        if verbose :
            print('\tWARNING: NANS in CCF')
        # return NaNs
        result = np.zeros(4) * np.nan
        ccf_fit = np.zeros_like(ccf) * np.nan
        return result, ccf_fit

    # get constants
    max_ccf, min_ccf = np.nanmax(ccf), np.nanmin(ccf)
    argmin, argmax = np.nanargmin(ccf), np.nanargmax(ccf)
    diff = max_ccf - min_ccf
    rvdiff = rv[1] - rv[0]
    # set up guess for gaussian fit
    # if fit_type == 0 then we have absorption lines
    if fit_type == 0:
        if np.nanmax(ccf) != 0:
            a = np.array([-diff / max_ccf, rv[argmin], 4 * rvdiff, 0])
        else:
            a = np.zeros(4)
    # else (fit_type == 1) then we have emission lines
    else:
        a = np.array([diff / max_ccf, rv[argmax], 4 * rvdiff, 1])
    # normalise y
    y = ccf / max_ccf - 1 + fit_type
    # x is just the RVs
    x = rv
    # uniform weights
    w = np.ones(len(ccf))
    # get gaussian fit
    nanmask = np.isfinite(y)
    y[~nanmask] = 0.0
    # fit the gaussian
    try:
        with warnings.catch_warnings(record=True) as _:
            result, fit = fitgaussian(x, y, weights=w, guess=a)
    except RuntimeError:
        result = np.repeat(np.nan, 4)
        fit = np.repeat(np.nan, len(x))

    # scale the ccf
    ccf_fit = (fit + 1 - fit_type) * max_ccf

    # return the best guess and the gaussian fit
    return result, ccf_fit


def gauss_function(x, a, x0, sigma, dc):
    """
    A standard 1D gaussian function (for fitting against)]=
    :param x: numpy array (1D), the x data points
    :param a: float, the amplitude
    :param x0: float, the mean of the gaussian
    :param sigma: float, the standard deviation (FWHM) of the gaussian
    :param dc: float, the constant level below the gaussian
    :return gauss: numpy array (1D), size = len(x), the output gaussian
    """
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + dc


def fitgaussian(x, y, weights=None, guess=None, return_fit=True,
                return_uncertainties=False):
    """
    Fit a single gaussian to the data "y" at positions "x", points can be
    weighted by "weights" and an initial guess for the gaussian parameters
    :param x: numpy array (1D), the x values for the gaussian
    :param y: numpy array (1D), the y values for the gaussian
    :param weights: numpy array (1D), the weights for each y value
    :param guess: list of floats, the initial guess for the guassian fit
                  parameters in the following order:
                  [amplitude, center, fwhm, offset from 0 (in y-direction)]
    :param return_fit: bool, if True also calculates the fit values for x
                       i.e. yfit = gauss_function(x, *pfit)
    :param return_uncertainties: bool, if True also calculates the uncertainties
                                 based on the covariance matrix (pcov)
                                 uncertainties = np.sqrt(np.diag(pcov))
    :return pfit: numpy array (1D), the fit parameters in the
                  following order:
                [amplitude, center, fwhm, offset from 0 (in y-direction)]
    :return yfit: numpy array (1D), the fit y values, i.e. the gaussian values
                  for the fit parameters, only returned if return_fit = True
    """

    # if we don't have weights set them to be all equally weighted
    if weights is None:
        weights = np.ones(len(x))
    weights = 1.0 / weights
    # if we aren't provided a guess, make one
    if guess is None:
        guess = [np.nanmax(y), np.nanmean(y), np.nanstd(y), 0]
    # calculate the fit using curve_fit to the function "gauss_function"
    with warnings.catch_warnings(record=True) as _:
        pfit, pcov = curve_fit(gauss_function, x, y, p0=guess, sigma=weights,
                               absolute_sigma=True)
    if return_fit and return_uncertainties:
        # calculate the fit parameters
        yfit = gauss_function(x, *pfit)
        # work out the normalisation constant
        chis, _ = chisquare(y, f_exp=yfit)
        norm = chis / (len(y) - len(guess))
        # calculate the fit uncertainties based on pcov
        efit = np.sqrt(np.diag(pcov)) * np.sqrt(norm)
        # return pfit, yfit and efit
        return pfit, yfit, efit
    # if just return fit
    elif return_fit:
        # calculate the fit parameters
        yfit = gauss_function(x, *pfit)
        # return pfit and yfit
        return pfit, yfit
    # if return uncertainties
    elif return_uncertainties:
        # calculate the fit parameters
        yfit = gauss_function(x, *pfit)
        # work out the normalisation constant
        chis, _ = chisquare(y, f_exp=yfit)
        norm = chis / (len(y) - len(guess))
        # calculate the fit uncertainties based on pcov
        efit = np.sqrt(np.diag(pcov)) * np.sqrt(norm)
        # return pfit and efit
        return pfit, efit
    # else just return the pfit
    else:
        # return pfit
        return pfit


def delta_v_rms_2d(spe, wave, sigdet, threshold, size):
    """
    Compute the photon noise uncertainty for all orders (for the 2D image)
    :param spe: numpy array (2D), the extracted spectrum
                size = (number of orders by number of columns (x-axis))
    :param wave: numpy array (2D), the wave solution for each pixel
    :param sigdet: float, the read noise (sigdet) for calculating the
                   noise array
    :param threshold: float, upper limit for pixel values, above this limit
                      pixels are regarded as saturated
    :param size: int, size (in pixels) around saturated pixels to also regard
                 as bad pixels
    :return dvrms2: numpy array (1D), the photon noise for each pixel (squared)
    :return weightedmean: float, weighted mean photon noise across all orders
    """
    # flag (saturated) fluxes above threshold as "bad pixels"
    with warnings.catch_warnings(record=True) as _:
        flag = spe < threshold
    # flag all fluxes around "bad pixels" (inside +/- size of the bad pixel)
    for i_it in range(1, 2 * size, 1):
        flag[:, size:-size] *= flag[:, i_it: i_it - 2 * size]
    # get the wavelength normalised to the wavelength spacing
    nwave = wave[:, 1:-1] / (wave[:, 2:] - wave[:, :-2])
    # get the flux + noise array
    sxn = (spe[:, 1:-1] + sigdet ** 2)
    # get the flux difference normalised to the flux + noise
    nspe = (spe[:, 2:] - spe[:, :-2]) / sxn
    # get the mask value
    maskv = flag[:, 2:] * flag[:, 1:-1] * flag[:, :-2]
    # get the total
    tot = np.nansum(sxn * ((nwave * nspe) ** 2) * maskv, axis=1)
    # convert to dvrms2
    with warnings.catch_warnings(record=True) as _:
        dvrms2 = ((SPEED_OF_LIGHT * 1000) ** 2) / abs(tot)
    # weighted mean of dvrms2 values
    weightedmean = 1. / np.sqrt(np.nansum(1.0 / dvrms2))
    # return dv rms and weighted mean
    return dvrms2, weightedmean


def fwhm(sigma=1.0):
    """
    Get the Full-width-half-maximum value from the sigma value (~2.3548)
    :param sigma: float, the sigma, default value is 1.0 (normalised gaussian)
    :return: 2 * sqrt(2 * log(2)) * sigma = 2.3548200450309493 * sigma
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fits2wave(file_or_header, npix=0):
    info = """
        Provide a fits header or a fits file
        and get the corresponding wavelength
        grid from the header.
        
        Usage :
        wave = fits2wave(hdr)
        or
        wave = fits2wave('my_e2ds.fits')
        
        Output has the same size as the input
        grid. This is derived from NAXIS
        values in the header
        """
    
    
    # check that we have either a fits file or an astropy header
    if type(file_or_header) == str:
        hdr = fits.getheader(file_or_header)
    elif str(type(file_or_header)) == "<class 'astropy.io.fits.header.Header'>":
        hdr = file_or_header
    else:
        print()
        print('~~~~ wrong type of input ~~~~')
        print()
        
        print(info)
        return []

    # get the keys with the wavelength polynomials
    wave_hdr = hdr['WAVE0*']
    # concatenate into a numpy array
    wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])
    
    # get the number of orders
    nord = hdr['WAVEORDN']
    
    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)
    
    # get the length of each order  if not provided (normally that's 4088 pix)
    if npix == 0 :
        if 'NAXIS1' in hdr.keys() :
            npix = hdr['NAXIS1']
        else :
            npix = 4088

    # project polynomial coefficiels
    wavesol = [np.polyval(wave_poly[i][::-1],np.arange(npix)) for i in range(nord) ]
    
    # return wave grid
    return np.array(wavesol)


def ccf_calculation(p, wave, image, blaze, targetrv, mask_centers, mask_weights,
                    berv, fit_type, mask_orders=[], verbose=False):
    # get rvmin and rvmax
    rvmin = targetrv - p['CCF_WIDTH']
    rvmax = targetrv + p['CCF_WIDTH'] + p['CCF_STEP']
    # get the dimensions
    nbo, nbpix = image.shape
    # create a rv ccf range
    rv_ccf = np.arange(rvmin, rvmax, p['CCF_STEP'])
    # storage of the ccf
    ccf_all = []
    ccf_noise_all = []
    ccf_all_fit = []
    ccf_all_results = []
    ccf_lines = []
    ccf_all_snr = []
    ccf_norm_all = []

    
    kernel = p['KERNEL']
    # if we have defined 'kernel', it must be a list
    # with the first element being the type of convolution
    # and subsequent arguments being parameters. For now,
    # we have :
    #
    #  --> boxcar convolution
    # ['boxcar', width]
    #
    # kernel = [1, 1, ...., 1, 1]
    #
    # --> gaussian convolution
    #
    # ['gaussian', e-width]
    # kernel = exp( -0.5*(x/ew)**2 )
    #
    # --> super gaussian
    #
    # ['supergaussian', e-width, beta]
    #
    # kernel = exp( -0.5*(x/ew)**beta )
    #
    # Other functions could be added below
    #
    if isinstance(kernel, list):
        if kernel[0] == 'boxcar':
            # ones with a length of kernel[1]
            ker = np.ones(int(np.round(kernel[1] / p['CCF_STEP'])))
        elif kernel[0] == 'gaussian':
            # width of the gaussian expressed in
            # steps of CCF
            ew = kernel[1] / p['CCF_STEP']
            index = np.arange(-4 * np.ceil(ew), 4 * np.ceil(ew) + 1)
            ker = np.exp(-0.5 * (index / ew) ** 2)
        elif kernel[0] == 'supergaussian':
            # width of the gaussian expressed in
            # steps of CCF. Exponents should be
            # between 0.1 and 10.. Values above
            # 10 are (nearly) the same as a boxcar.
            if (kernel[1] < 0.1) or (kernel[1] > 10):
                raise ValueError('CCF ERROR: kernel[1] is out of range.')

            ew = kernel[1] / p['CCF_STEP']

            index = np.arange(-4 * np.ceil(ew), 4 * np.ceil(ew) + 1)
            ker = np.exp(-0.5 * np.abs(index / ew) ** kernel[2])

        else:
                # kernel name is not known - generate error
            raise ValueError('CCF ERROR: name of kernel not accepted!')

        ker = ker / np.sum(ker)

        if len(ker) > (len(rv_ccf)-1):
            # TODO : give a proper error
            err_msg = """
            The size of your convolution kernel is too big for your
            CCF size. Please either increase the CCF_WIDTH value or
            decrease the width of your convolution kernel. In boxcar,
            this implies a length bigger than CCF_WIDTH/CCF_STEP, in
            gaussian and supergaussian, this means that
            CCF_WIDTH/CCF_STEP is >8*ew. The kernel has to run from
            -4 sigma to +4 sigma.
            """
            raise ValueError('CCF ERROR: {0}'.format(err_msg))
    
    # ----------------------------------------------------------------------
    # loop around the orders
    for order_num in range(nbo):
        # log the process
        #print('Order {0}'.format(order_num))
        # ------------------------------------------------------------------
        # get this orders values
        wa_ord = np.array(wave[order_num])
        sp_ord = np.array(image[order_num])
        bl_ord = np.array(blaze[order_num])

        # COMMENT EA, normalization moved before the masking
        #
        # normalize per-ord blaze to its peak value
        # this gets rid of the calibration lamp SED
        bl_ord /= np.nanpercentile(bl_ord, p['BLAZE_NORM_PERCENTILE'])
        # COMMENT EA, changing NaNs to 0 in the blaze
        bl_ord[np.isfinite(bl_ord) == 0] = 0
        # mask on the blaze
        with warnings.catch_warnings(record=True) as _:
            blazemask = bl_ord > p['BLAZE_THRESHOLD']
        # get order mask centers and mask weights
        min_ord_wav = np.nanmin(wa_ord[blazemask])
        max_ord_wav = np.nanmax(wa_ord[blazemask])
        # COMMENT EA there's a problem with the sign in the min/max
        min_ord_wav = min_ord_wav * (1 - rvmin / SPEED_OF_LIGHT)
        max_ord_wav = max_ord_wav * (1 - rvmax / SPEED_OF_LIGHT)
        # mask the ccf mask by the order length
        mask_wave_mask = (mask_centers > min_ord_wav)
        mask_wave_mask &= (mask_centers < max_ord_wav)
        if len(mask_orders) :
            mask_wave_mask &= mask_orders == order_num

        omask_centers = mask_centers[mask_wave_mask]
        omask_weights = mask_weights[mask_wave_mask]

        # ------------------------------------------------------------------
        # find any places in spectrum or blaze where pixel is NaN
        nanmask = np.isnan(sp_ord) | np.isnan(bl_ord)
        # ------------------------------------------------------------------
        # deal with no valid lines
        if np.sum(mask_wave_mask) == 0:
            if verbose:
                print('\tWARNING: MASK INVALID FOR WAVELENGTH RANGE --> NAN')
            # set all values to NaN
            ccf_all.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_all_fit.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_all_results.append(np.repeat(np.nan, 4))
            ccf_noise_all.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_lines.append(0)
            ccf_all_snr.append(np.nan)
            ccf_norm_all.append(np.nan)
            continue
        # ------------------------------------------------------------------
        # deal with orders where number of nans is too large (> 90% of order size)
        if np.sum(nanmask) > nbpix * 0.9:
            if verbose:
                print('\tWARNING: ALL SP OR BLZ NAN --> NAN')
            # set all values to NaN
            ccf_all.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_all_fit.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_all_results.append(np.repeat(np.nan, 4))
            ccf_noise_all.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_lines.append(0)
            ccf_all_snr.append(np.nan)
            ccf_norm_all.append(np.nan)
            continue
        # ------------------------------------------------------------------
        # set the spectrum or blaze NaN pixels to zero (dealt with by divide)
        sp_ord[nanmask] = 0
        bl_ord[nanmask] = 0
        # now every value that is zero is masked (we don't want to spline these)
        good = (sp_ord != 0) & (bl_ord != 0)
        # ------------------------------------------------------------------
        # spline the spectrum and the blaze
        spline_sp = iuv_spline(wa_ord[good], sp_ord[good], k=5, ext=1)
        spline_bl = iuv_spline(wa_ord[good], bl_ord[good], k=5, ext=1)
        # ------------------------------------------------------------------
        # set up the ccf for this order
        ccf_ord = np.zeros_like(rv_ccf)
        # ------------------------------------------------------------------
        # get the wavelength shift (dv) in relativistic way
        wave_shifts = relativistic_waveshift(rv_ccf - berv)
        # ------------------------------------------------------------------
        # set number of valid lines used to zero
        numlines = 0
        # loop around the rvs and calculate the CCF at this point
        part3 = spline_bl(omask_centers)
        for rv_element in range(len(rv_ccf)):
            wave_tmp = omask_centers * wave_shifts[rv_element]
            part1 = spline_sp(wave_tmp)
            part2 = spline_bl(wave_tmp)
            numlines = np.sum(spline_bl(wave_tmp) != 0)
            # CCF is the division of the sums
            with warnings.catch_warnings(record=True) as _:
                ccf_element = ((part1 * part3) / part2) * omask_weights
                ccf_ord[rv_element] = np.nansum(ccf_element)
        # ------------------------------------------------------------------
        # deal with NaNs in ccf
        if np.sum(np.isnan(ccf_ord)) > 0:
            # log all NaN
            if verbose:
                print('WARNING: CCF is NAN')
            # set all values to NaN
            ccf_all.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_all_fit.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_all_results.append(np.repeat(np.nan, 4))
            ccf_noise_all.append(np.repeat(np.nan, len(rv_ccf)))
            ccf_lines.append(0)
            ccf_all_snr.append(np.nan)
            ccf_norm_all.append(np.nan)
            continue
        # ------------------------------------------------------------------
        # Convolve by the appropriate CCF kernel, if any
        if type(kernel) == list:
            weight = np.convolve(np.ones(len(ccf_ord)), ker, mode='same')
            ccf_ord = np.convolve(ccf_ord, ker, mode='same') / weight
        # ------------------------------------------------------------------

        # normalise each orders CCF to median
        ccf_norm = np.nanmedian(ccf_ord)
        # ccf_ord = ccf_ord / ccf_norm
        # ------------------------------------------------------------------
        # fit the CCF with a gaussian
        fargs = [rv_ccf, ccf_ord, fit_type]
        ccf_coeffs_ord, ccf_fit_ord = fit_ccf(*fargs)
        # ------------------------------------------------------------------
        # calculate the residuals of the ccf fit
        res = ccf_ord - ccf_fit_ord
        # calculate the CCF noise per order
        ccf_noise = np.array(res)
        # calculate the snr for this order
        ccf_snr = np.abs(ccf_coeffs_ord[0] / np.nanmedian(np.abs(ccf_noise)))
        # ------------------------------------------------------------------
        # append ccf to storage
        ccf_all.append(ccf_ord)
        ccf_all_fit.append(ccf_fit_ord)
        ccf_all_results.append(ccf_coeffs_ord)
        ccf_noise_all.append(ccf_noise)
        ccf_lines.append(numlines)
        ccf_all_snr.append(ccf_snr)
        ccf_norm_all.append(ccf_norm)
    # store outputs in param dict
    props = dict()
    props['RV_CCF'] = rv_ccf
    props['CCF'] = np.array(ccf_all)
    props['CCF_LINES'] = np.array(ccf_lines)
    props['TOT_LINE'] = np.sum(ccf_lines)
    props['CCF_NOISE'] = np.array(ccf_noise_all)
    props['CCF_SNR'] = np.array(ccf_all_snr)
    props['CCF_FIT'] = np.array(ccf_all_fit)
    props['CCF_FIT_COEFFS'] = np.array(ccf_all_results)
    props['CCF_NORM'] = np.array(ccf_norm_all)

    # Return properties
    return props


def mean_ccf(p, props, targetrv, fit_type, valid_orders=None, normalize_ccfs=True, plot=False, verbose=False):
    
    allorders = [i for i in range(49)]
    
    if valid_orders == None :
        ccfs = props['CCF'][: p['CCF_N_ORD_MAX'] ]
        valid_orders = [i for i in range(len(ccfs))]
    else :
        ccfs = []
        for order in allorders :
            norm_ccf = (props['CCF'][order] / np.nanmedian(props['CCF'][order]))

            if order in valid_orders :
                if normalize_ccfs :
                    ccfs.append(norm_ccf)
                else :
                    ccfs.append(props['CCF'][order])

                if plot :
                    plt.plot(props['RV_CCF'],norm_ccf, 'c-', linewidth=0.6)
            else :
                if plot :
                    plt.plot(props['RV_CCF'],norm_ccf, 'm--', linewidth=0.3)
                pass
        ccfs = np.array(ccfs, dtype=float)

    # get the average ccf
    if normalize_ccfs :
        m_ccf = np.nanmedian(ccfs, axis=0)
    else :
        m_ccf = np.nanmean(ccfs, axis=0)


    # get the fit for the normalized average ccf
    mean_ccf_coeffs, mean_ccf_fit = fit_ccf(props['RV_CCF'],
                                            m_ccf, fit_type=fit_type)


    if plot :
        plt.plot(props['RV_CCF'], m_ccf / np.nanmedian(m_ccf), 'r--', linewidth=2, label="Mean CCF using {0} orders".format(len(valid_orders)))
        plt.plot(props['RV_CCF'], mean_ccf_fit / np.nanmedian(mean_ccf_fit), 'g-', linewidth=2, label="Model CCF")

        plt.xlabel(r"Velocity [km/s]")
        plt.ylabel(r"Relative flux")
        plt.legend()
        plt.show()

    # get the RV value from the normalised average ccf fit center location
    ccf_rv = float(mean_ccf_coeffs[1])
    # get the contrast (ccf fit amplitude)
    ccf_contrast = np.abs(100 * mean_ccf_coeffs[0])
    # get the FWHM value
    ccf_fwhm = mean_ccf_coeffs[2] * fwhm()
    # --------------------------------------------------------------------------
    #  CCF_NOISE uncertainty
    ccf_noise_tot = np.sqrt(np.nanmean(props['CCF_NOISE'] ** 2, axis=0))
    # Calculate the slope of the CCF
    average_ccf_diff = (m_ccf[2:] - m_ccf[:-2])
    rv_ccf_diff = (props['RV_CCF'][2:] - props['RV_CCF'][:-2])
    ccf_slope = average_ccf_diff / rv_ccf_diff
    # Calculate the CCF oversampling
    ccf_oversamp = p['IMAGE_PIXEL_SIZE'] / p['CCF_STEP']
    # create a list of indices based on the oversample grid size
    flist = np.arange(np.round(len(ccf_slope) / ccf_oversamp))
    indexlist = np.array(flist * ccf_oversamp, dtype=int)
    # we only want the unique pixels (not oversampled)
    indexlist = np.unique(indexlist)
    # get the rv noise from the sum of pixels for those points that are
    #     not oversampled
    keep_ccf_slope = ccf_slope[indexlist]
    keep_ccf_noise = ccf_noise_tot[1:-1][indexlist]
    rv_noise = np.nansum(keep_ccf_slope ** 2 / keep_ccf_noise ** 2) ** (-0.5)
    # --------------------------------------------------------------------------
    # log the stats
    wargs = [ccf_contrast, float(mean_ccf_coeffs[1]), rv_noise, ccf_fwhm]
    if verbose:
        print('MEAN CCF:')
        print('\tCorrelation: C={0:1f}[%] RV={1:.5f}[km/s] RV_NOISE={2:.5f}[km/s] '
          'FWHM={3:.4f}[km/s]'.format(*wargs))
    # --------------------------------------------------------------------------
    # add to output array
    props['MEAN_CCF'] = m_ccf
    props['MEAN_CCF_RES'] = mean_ccf_coeffs
    props['MEAN_CCF_FIT'] = mean_ccf_fit

    if np.isfinite(ccf_rv) :
        props['MEAN_RV'] = ccf_rv
        props['MEAN_CONTRAST'] = ccf_contrast
        props['MEAN_FWHM'] = ccf_fwhm
    else :
        props['MEAN_RV'] = -9999
        props['MEAN_CONTRAST'] = -9999
        props['MEAN_FWHM'] = -9999

    if np.isfinite(rv_noise) :
        props['MEAN_RV_NOISE'] = rv_noise
    else :
        props['MEAN_RV_NOISE'] = 0.
    # --------------------------------------------------------------------------
    # add constants to props
    props['CCF_MASK'] = os.path.basename(p['MASK_FILE'])
    props['CCF_STEP'] = p['CCF_STEP']
    props['CCF_WIDTH'] = p['CCF_WIDTH']
    props['TARGET_RV'] = targetrv
    props['CCF_SIGDET'] = p['NOISE_SIGDET']
    props['CCF_BOXSIZE'] = p['NOISE_SIZE']
    props['CCF_MAXFLUX'] = p['NOISE_THRES']
    props['CCF_NMAX'] = p['CCF_N_ORD_MAX']
    props['MASK_MIN'] = p['MASK_MIN_WEIGHT']
    props['MASK_WIDTH'] = p['MASK_WIDTH']
    props['MASK_UNITS'] = 'nm'
    # --------------------------------------------------------------------------
    return props


def plot_individual_ccf(props, nbo):
    # get the plot loop generator
    generator = plotloop(range(nbo))
    # loop around orders
    for order_num in generator:
        plt.close()
        fig, frame = plt.subplots(ncols=1, nrows=1)
        frame.plot(props['RV_CCF'], props['CCF'][order_num], color='b',
                   marker='+', ls='None', label='data')
        frame.plot(props['RV_CCF'], props['CCF_FIT'][order_num], color='r',)
        rvorder = props['CCF_FIT_COEFFS'][order_num][1]
        frame.set(title='Order {0}  RV = {1} km/s'.format(order_num, rvorder),
                  xlabel='RV [km/s]', ylabel='CCF')
        plt.show()
        plt.close()


def plot_mean_ccf(props):
    plt.close()
    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.plot(props['RV_CCF'], props['MEAN_CCF'], color='b', marker='+',
               ls='None')
    frame.plot(props['RV_CCF'], props['MEAN_CCF_FIT'], color='r')
    frame.set(title='Mean CCF   RV = {0} km/s'.format(props['MEAN_RV']),
              xlabel='RV [km/s]', ylabel='CCF')
    plt.show()
    plt.close()


def plotloop(looplist):
    # check that looplist is a valid list
    if not isinstance(looplist, list):
        # noinspection PyBroadException
        try:
            looplist = list(looplist)
        except Exception as _:
            print('PLOT ERROR: looplist must be a list')
    # define message to give to user
    message = ('Plot loop navigation: Go to \n\t [P]revious plot '
               '\n\t [N]ext plot \n\t [E]nd plotting '
               '\n\t Number from [0 to {0}]: \t')
    message = message.format(len(looplist) - 1)
    # start the iterator at zero
    it = 0
    first = True
    # loop around until we hit the length of the loop list
    while it < len(looplist):
        # deal with end of looplist
        if it == len(looplist):
            # break out of while
            break
        # if this is the first iteration do not print message
        if first:
            # yield the first iteration value
            yield looplist[it]
            # increase the iterator value
            it += 1
            first = False
        # else we need to ask to go to previous, next or end
        else:
            # get user input
            userinput = input(message)
            # try to cast into a integer
            # noinspection PyBroadException
            try:
                userinput = int(userinput)
            except Exception as _:
                userinput = str(userinput)
            # if 'p' in user input we assume they want to go to previous
            if 'P' in str(userinput).upper():
                yield looplist[it - 1]
                it -= 1
            # if 'n' in user input we assume they want to go to next
            elif 'N' in str(userinput).upper():
                yield looplist[it + 1]
                it += 1
            elif isinstance(userinput, int):
                it = userinput
                # deal with it too low
                if it < 0:
                    it = 0
                # deal with it too large
                elif it >= len(looplist):
                    it = len(looplist) - 1
                # yield the value of it
                yield looplist[it]
            # else we assume the loop is over and we want to exit
            else:
                # break out of while
                break


def construct_out_ccf_filename(infile, maskname) :
    # ----------------------------------------------------------------------
    # construct out file name
    inbasename = os.path.basename(infile).split('.')[0]
    maskbasename = os.path.basename(maskname).split('.')[0]
    inpath = os.path.dirname(infile)
    outfile = 'CCFTABLE_{0}_{1}.fits'.format(inbasename, maskbasename)
    outpath = os.path.join(inpath, outfile)
    # ----------------------------------------------------------------------
    return outpath


def write_file(props, infile, maskname, header, wheader, rv_drifts, save=True, verbose=False):

    warnings.simplefilter('ignore', category=VerifyWarning)

    outpath = construct_out_ccf_filename(infile, maskname)
    
    # produce CCF table
    table1 = Table()
    table1['RV'] = props['RV_CCF']
    for order_num in range(len(props['CCF'])):
        table1['ORDER{0:02d}'.format(order_num)] = props['CCF'][order_num]
    table1['COMBINED'] = props['MEAN_CCF']
    # ----------------------------------------------------------------------
    # produce stats table
    table2 = Table()
    table2['ORDERS'] = np.arange(len(props['CCF'])).astype(int)
    table2['NLINES'] = props['CCF_LINES']
    # get the coefficients
    coeffs = props['CCF_FIT_COEFFS']
    table2['CONTRAST'] = np.abs(100 * coeffs[:, 0])
    table2['RV'] = coeffs[:, 1]
    table2['FWHM'] = coeffs[:, 2]
    table2['DC'] = coeffs[:, 3]
    table2['SNR'] = props['CCF_SNR']
    table2['NORM'] = props['CCF_NORM']

    # ----------------------------------------------------------------------
    # add to the header
    # ----------------------------------------------------------------------
    # add results from the CCF
    header['CCFMNRV'] = (props['MEAN_RV'],
                         'Mean RV calc. from the mean CCF [km/s]')
    header['CCFMCONT'] = (props['MEAN_CONTRAST'],
                          'Mean contrast (depth of fit) from mean CCF')
    header['CCFMFWHM'] = (props['MEAN_FWHM'],
                          'Mean FWHM from mean CCF')
    header['CCFMRVNS'] = (props['MEAN_RV_NOISE'],
                          'Mean RV Noise from mean CCF')
    header['CCFTLINE'] = (props['TOT_LINE'],
                          'Total no. of mask lines used in CCF')
    # ----------------------------------------------------------------------
    # add constants used to process
    header['CCFMASK'] = (props['CCF_MASK'], 'CCF mask file used')
    header['CCFSTEP'] = (props['CCF_STEP'], 'CCF step used [km/s]')
    header['CCFWIDTH'] = (props['CCF_WIDTH'], 'CCF width used [km/s]')
    header['CCFTRGRV'] = (props['TARGET_RV'],
                          'CCF central RV used in CCF [km/s]')
    header['CCFSIGDT'] = (props['CCF_SIGDET'],
                          'Read noise used in photon noise calc. in CCF')
    header['CCFBOXSZ'] = (props['CCF_BOXSIZE'],
                          'Size of bad px used in photon noise calc. in CCF')
    header['CCFMAXFX'] = (props['CCF_MAXFLUX'],
                          'Flux thres for bad px in photon noise calc. in CCF')
    header['CCFORDMX'] = (props['CCF_NMAX'],
                          'Last order used in mean for mean CCF')
    header['CCFMSKMN'] = (props['MASK_MIN'],
                          'Minimum weight of lines used in the CCF mask')
    header['CCFMSKWD'] = (props['MASK_WIDTH'],
                          'Width of lines used in the CCF mask')
    header['CCFMUNIT'] = (props['MASK_UNITS'], 'Units used in CCF Mask')
    # ----------------------------------------------------------------------
    #header['RV_WAVFN'] = (os.path.basename(WAVE_FILE), 'RV wave file used')
    if "MJDMID" not in header :
        try :
            header['MJDMID'] = header['MJDATE'] + (header['MJDEND'] - header['MJDATE'])/2.
        except :
            header['MJDMID'] = header['MJD-OBS'] + (header['JD-END'] - header['JD-OBS']) / 2.

    if "MJDMID" not in wheader :
        wheader['MJDMID'] = header['MJDMID']

    header['RV_WAVTM'] = (wheader['MJDMID'],
                          'RV wave file time [mjd]')
    header['RV_WAVTD'] = (header['MJDMID'] - wheader['MJDMID'],
                          'RV timediff [days] btwn file and wave solution')
    header['WFPDRIFT'] = (rv_drifts['WFPDRIFT'], 'Wavelength sol absolute CCF FP Drift [km/s]')
    header['RV_WAVFP'] = (rv_drifts['RV_WAVFP'], 'RV measured from wave sol FP CCF [km/s]')
    header['RV_SIMFP'] = (rv_drifts['RV_SIMFP'], 'RV measured from simultaneous FP CCF [km/s]')
    header['RV_DRIFT'] = (rv_drifts['RV_DRIFT'],
                          'RV drift between wave sol and sim. FP CCF [km/s]')
    header.set('ERVDRIFT', rv_drifts['RV_DRIFTERR'], 'Error of RV_DRIFT [km/s]')
    header['RV_OBJ'] = (props['MEAN_RV'],
                        'RV calc in the object CCF (non corr.) [km/s]')
    if type(rv_drifts['RV_DRIFT']) == float :
        header['RV_CORR'] = (props['MEAN_RV']-rv_drifts['RV_DRIFT'], 'RV corrected for FP CCF drift [km/s]')
    else :
        header['RV_CORR'] = ('None', 'RV corrected for FP CCF drift [km/s]')
    # ----------------------------------------------------------------------

    # work around to make old data compatible:
    if "EXTSN035" not in header.keys() :
        if "SNR35" in header.keys():
            header["EXTSN035"] = header["SNR35"]
        #else :
        #   header["EXTSN035"] = ??
        
    if save :
        # log where we are writing the file to
        if verbose :
            print('Writing file to {0}'.format(outpath))
        
        # construct hdus
        hdu = fits.PrimaryHDU()
        t1 = fits.BinTableHDU(table1, header=header)
        t2 = fits.BinTableHDU(table2, header=header)
        # construct hdu list
        hdulist = fits.HDUList([hdu, t1, t2])
        # write hdulist
        hdulist.writeto(outpath, overwrite=True)
        props["file_path"] = outpath

    props["header"] = header
    props["RV_CCF"] = props['RV_CCF']
    props["MEAN_CCF"] = props['MEAN_CCF']

    return props


# =============================================================================
# main routine
# =============================================================================
def run_ccf_new(ccf_params, spectrum, rv_drifts, targetrv=0.0, valid_orders=None, normalize_ccfs=True, output=True, science_channel=True, plot=False, interactive_plot=False, verbose=False, merge_headers=False) :

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # get input telluric corrected file and header
    if science_channel :
        image = np.array(spectrum["FluxAB"], dtype=float)
        
        spectrum["header1"]["AIRMASS"] = spectrum["header0"]["AIRMASS"]
        
        if merge_headers :
            header = spectrum["header0"] + spectrum["header1"]
        else :
            header = spectrum["header0"]

        if "SPEMSNR" not in header.keys() :
            if "EXTSN035" in header.keys() :
                header["SPEMSNR"] = header["EXTSN035"]
            elif "SNR33" in header.keys() :
                header["SPEMSNR"] = header["SNR33"]
            else :
                header["SPEMSNR"] = 1

        blaze = np.array(spectrum["BlazeAB"], dtype=float)
    else :
        image = np.array(spectrum["FluxC"], dtype=float)
        if merge_headers :
            header = spectrum["header0"] + spectrum["headerC"]
        else :
            header = spectrum["headerC"]
        blaze = np.array(spectrum["BlazeC"], dtype=float)

    masktable = read_mask(ccf_params['MASK_FILE'], ccf_params['MASK_COLS'])
    # get the dimensions
    nbo, nbpix = image.shape

    if "WAVEORDN" not in header.keys() :
        header = make_wave_keywords(header)
    #wave = fits2wave(header, npix=nbpix)
    wheader = header
    wave = np.array(spectrum["WaveAB"])

    # --------------------------------------------------------------------------
    # get fiber typoe
    if 'FIBER' in header:
        fiber = header['FIBER']
    else:
        if science_channel :
            fiber = "AB"
        else :
            fiber = "C"
        #raise ValueError('HEADER ERROR: FIBER MISSING')
    # --------------------------------------------------------------------------
    # get dprtype
    #if 'DPRTYPE' in header:
    if fiber == 'AB':
        dprtype = "OBJ_FP".split('_')[0]
    else:
        dprtype = "OBJ_FP".split('_')[1]
    #else:
    #   raise ValueError('HEADER ERROR: DPRTYPE MISSING')
    # make sure dprtype is correct for fiber
    #   if dprtype not in ['OBJ', 'FP']:
    #    raise ValueError('HEADER ERROR: DPRTPYE must be OBJ or FP')
    # --------------------------------------------------------------------------
    # get berv from header
    if fiber == 'AB' and dprtype == 'OBJ':
        berv = header['BERV']
        # absorption features
        fit_type = 0
    else:
        berv = 0.0
        # emission features
        fit_type = 1

    #if berv == 'NaN     ':
    #        berv = 0.

    # --------------------------------------------------------------------------
    # get rv from header (or set to zero)
    if targetrv == 0. and ('OBJRV' in header) and dprtype == 'OBJ':
        targetrv = header['OBJRV']
        if np.isnan(targetrv) or targetrv == ccf_params["CCF_RV_NULL"]:
            targetrv = 0.0

    # --------------------------------------------------------------------------
    # get mask centers, and weights
    mask_orders = []
    if "order_mask" in masktable :
        mask_orders, mask_widths, mask_centers, mask_weights = get_mask(masktable, ccf_params["MASK_WIDTH"], ccf_params["MASK_MIN_WEIGHT"])
    else :
        _, mask_centers, mask_weights = get_mask(masktable, ccf_params["MASK_WIDTH"],
                                                 ccf_params["MASK_MIN_WEIGHT"])
    # --------------------------------------------------------------------------
    # Photon noise uncertainty
    # --------------------------------------------------------------------------
    dkwargs = dict(spe=image, wave=wave, sigdet=ccf_params["NOISE_SIGDET"],
                   size=ccf_params["NOISE_SIZE"], threshold=ccf_params["NOISE_THRES"])
    # run DeltaVrms2D
    dvrmsref, wmeanref = delta_v_rms_2d(**dkwargs)
    wmsg = 'On fiber {0} estimated RV uncertainty on spectrum is {1:.3f}'
    if verbose :
        print(wmsg.format(fiber, wmeanref))

    # Uncomment below to quickly check the input spectra to CCF routines
    #for i in range(49) :
    #    plt.plot(wave[i],image[i])
    #plt.show()
    # --------------------------------------------------------------------------
    # Calculate the CCF
    # --------------------------------------------------------------------------
    if verbose :
        print('\nRunning CCF calculation')
    props = ccf_calculation(ccf_params, wave, image, blaze, targetrv, mask_centers,
                            mask_weights, berv, fit_type, mask_orders=mask_orders)
    # --------------------------------------------------------------------------
    # Calculate the mean CCF
    # --------------------------------------------------------------------------
    if verbose :
        print('\nRunning Mean CCF')
    props = mean_ccf(ccf_params, props, targetrv, fit_type, valid_orders=valid_orders, normalize_ccfs=normalize_ccfs, plot=plot)

    # --------------------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------------------
    if interactive_plot :
        # plot individual CCFs
        if verbose :
            print('\n Plotting individual CCFs')
        plot_individual_ccf(props, nbo)
        # plot mean ccf and fit
        if verbose :
            print('\n Plotting Mean CCF')
        plot_mean_ccf(props)

    # --------------------------------------------------------------------------
    # Save file
    # --------------------------------------------------------------------------
    props = write_file(props, spectrum['filename'], ccf_params['MASK_FILE'], header, wheader, rv_drifts, save=output, verbose=verbose)

    return props


def make_wave_keywords(header) :
    """
        Description: function to create WAVE_* keywords for wavelength calibration
        as produced by previous versions of the DRS for backwards compatibility.
        """
    norders = header["TH_ORD_N"]
    degpoly = header["TH_LL_D"]

    header.set('WAVEORDN', norders, "nb orders in total")
    header.set('WAVEDEGN', degpoly, "degree of wave polyn fit")
    
    ncount = 0
    for order in range(norders) :
        for coeff in range(degpoly+1) :
            th_lc_key = "TH_LC{0}".format(ncount)
            wave_key = "WAVE{0:04d}".format(ncount)
            wave_comment = "Wavelength coefficients order={0} coeff={1}".format(order, coeff)
            header.set(wave_key, header[th_lc_key], wave_comment)
            ncount += 1
    return header


def select_best_ccf_mask(obj_temp, mask_repository) :
    """
        Description: function to select a CCF mask that best matches
        a given temperature.
        """

    teff = {'GJ1002_neg_depth.mas': 2900,
        'Gl905_neg_depth.mas': 2930,
        'Gl725B_neg_depth.mas': 3106,
        'Gl699_neg_depth.mas': 3224,
        'Gl725A_neg_depth.mas': 3398,
        'Gl412A_neg_depth.mas': 3549,
        'Gl846_neg_depth.mas': 3880,
        'HD189733_neg_depth.mas': 4870,
        'TauBoo_neg_depth.mas': 6309}
    
    default_mask = mask_repository + 'Gl412A_neg_depth.mas'

    pattern = mask_repository + '*.mas'

    list_of_masks = sorted(glob.glob(pattern))

    best_mask = default_mask
    temp_diff = 1e20

    for mask in list_of_masks :
        
        # get rid of full path before file name
        mask_basename = os.path.basename(mask)

        loc_teff_diff = np.abs(obj_temp - teff[mask_basename])

        if loc_teff_diff < temp_diff :
            temp_diff = loc_teff_diff
            best_mask = mask

    if temp_diff == 1e20 :
        print("WARNING: Could not identify any suitable mask, returning default mask: {}".format(best_mask))

    return best_mask


def apply_weights_to_ccf_mask(ccf_params, wave, fluxes, fluxerrs, weights, median=True, remove_lines_with_nans=True, ordertoplot=31, source_rv=0., verbose=False, plot=False) :
    
    sci_table = read_mask(ccf_params['MASK_FILE'], ccf_params['MASK_COLS'])
    
    if len(ccf_params['TELL_MASK_FILES']) :
        masktables = [sci_table]
        for m in ccf_params['TELL_MASK_FILES'] :
            masktables.append(read_mask(m, ccf_params['MASK_COLS']))
        masktable = vstack(masktables)
    else :
        masktable = sci_table

    # --------------------------------------------------------------------------
    # get mask centers, and weights
    lines_d, lines_wlc, lines_wei = get_mask(masktable, ccf_params["MASK_WIDTH"], ccf_params["MASK_MIN_WEIGHT"])

    lines_d = np.array(lines_d)
    lines_wlc = np.array(lines_wlc)
    lines_wei = np.array(lines_wei)

    mask_centers = np.array([])
    mask_weights = np.array([])
    mask_d = np.array([])
    mask_orders = np.array([])

    speed_of_light_in_kps = constants.c / 1000.
    edge_size = ccf_params["CCF_WIDTH"]
    #edge_size = 7. # +-7 km/s ~ instrumental resolution
    
    for order in range(49) :
        wl = wave[order]
        flux = fluxes[order]
        fluxerr = fluxerrs[order]
        weight = weights[order]
        
        median_flux = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - median_flux)) / 0.67449
        
        #print("order={0} median_flux={1:.4f} sigma={2:.4f}".format(order, median_flux, mad))
        
        minwl = wl[0] * (1.0 + edge_size / speed_of_light_in_kps)
        maxwl = wl[-1] * (1.0 - edge_size / speed_of_light_in_kps)
        
        order_lines = lines_wlc > minwl
        order_lines &= lines_wlc < maxwl

        order_lines_wlc = lines_wlc[order_lines]
        order_lines_wei = lines_wei[order_lines]
        order_lines_d = lines_d[order_lines]

        wl_ini = order_lines_wlc * (1.0 + (source_rv - edge_size) / speed_of_light_in_kps)
        wl_end = order_lines_wlc * (1.0 + (source_rv + edge_size) / speed_of_light_in_kps)

        nlinesinorder = 0
        
        if plot and order == ordertoplot :
            wlc_starframe = order_lines_wlc * (1.0 + source_rv / speed_of_light_in_kps)

            plt.vlines(wlc_starframe, median_flux - order_lines_wei / np.nanmax(order_lines_wei), median_flux,ls="--", lw=0.5, label="CCF mask lines")
            plt.plot(wl,flux,'-', lw=0.5, color="grey", label="Template spectrum")
            plt.plot(wl, 1.0 - fluxerr,'k:', lw=0.5, label="1-sigma noise threshold")

        for i in range(len(order_lines_wlc)) :
            inline = (wl >= wl_ini[i]) & (wl <= wl_end[i])
            #nanmask = ~np.isnan(flux[inline])
            #print(order, i, len(order_lines_wlc), wl_ini[i], wl_end[i], len(flux[inline]), len(flux[inline][nanmask]))
            if len(flux[inline]) == 0 :
                continue
            
            line_mean_flux = np.nanmean(flux[inline])
            line_mean_fluxerr = np.nanmean(fluxerr[inline])
            line_mean_wl = np.nanmean(wl[inline])
            
            #if line_mean_flux > (1.0 - line_mean_fluxerr) :
            #    continue

            #print("Line: ",i,wl_ini[i],wl_end[i],line_mean_flux)
            if plot and order == ordertoplot :
                p = plt.plot(wl[inline],flux[inline],'.',alpha=0.5)
                color = p[0].get_color()
                #plt.plot([np.mean(wl[inline])],[np.nanmean(weight[inline])],'o', color=color)
                plt.plot([line_mean_wl],[line_mean_flux],'o', color=color)
    
            if remove_lines_with_nans :
                
                if np.all(np.isfinite(flux[inline]) * np.isfinite(weight[inline])):
                    mask_centers = np.append(mask_centers, order_lines_wlc[i])
                    mask_d = np.append(mask_d, order_lines_d[i])
                    
                    if median :
                        mask_weights = np.append(mask_weights, order_lines_wei[i] * np.nanmedian(weight[inline]))
                    else :
                        mask_weights = np.append(mask_weights, order_lines_wei[i] * np.nanmean(weight[inline]))

                    mask_orders = np.append(mask_orders, float(order))
                    nlinesinorder += 1
            else :
                mask_centers = np.append(mask_centers, order_lines_wlc[i])
                mask_d = np.append(mask_d, order_lines_d[i])

                if np.isfinite(np.nanmedian(weight[inline])) :
                    if median :
                        mask_weights = np.append(mask_weights, order_lines_wei[i] * np.nanmedian(weight[inline]))
                    else :
                        mask_weights = np.append(mask_weights, order_lines_wei[i] * np.nanmean(weight[inline]))
                else :
                    mask_weights = np.append(mask_weights, order_lines_wei[i] * 0.)

                mask_orders = np.append(mask_orders, float(order))
                nlinesinorder += 1
        if verbose :
            print("Selected {0} lines in order {1} ".format(nlinesinorder,order))
        if plot and order == ordertoplot :
            plt.show()
    #mask_weights /= np.max(mask_weights)

    outmask = {}

    outmask["centers"] = mask_centers
    outmask["weights"] = mask_weights
    outmask["widths"] = mask_d
    outmask["orders"] = mask_orders

    return outmask

# =============================================================================
# main routine -- version that inputs the spectrum in different format that
# assumes a pre-processing.
# =============================================================================
def run_ccf_eder(ccf_params, wave, fluxes, header, ccfmask, rv_drifts={}, filename="", berv=0., targetrv=0.0, valid_orders=None, normalize_ccfs=True, fit_type=0, output=True, plot=False, verbose=False) :

    if rv_drifts == {} :
        rv_drifts["WFPDRIFT"] ='None'
        rv_drifts["RV_WAVFP"] = 'None'
        rv_drifts["RV_SIMFP"] = 'None'
        rv_drifts["RV_DRIFT"] = 0.
        rv_drifts["RV_DRIFTERR"] = 0.

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    blaze = np.ones_like(fluxes)
    
    # --------------------------------------------------------------------------
    # Photon noise uncertainty
    # --------------------------------------------------------------------------
    dkwargs = dict(spe=fluxes, wave=wave, sigdet=ccf_params["NOISE_SIGDET"],
                   size=ccf_params["NOISE_SIZE"], threshold=ccf_params["NOISE_THRES"])

    # run DeltaVrms2D
    dvrmsref, wmeanref = delta_v_rms_2d(**dkwargs)
    if verbose :
        print('Estimated RV uncertainty on spectrum is {0:.3f}'.format(wmeanref))

    # --------------------------------------------------------------------------
    # Calculate the CCF
    # --------------------------------------------------------------------------
    if verbose :
        print('\nRunning CCF calculation')

    props = ccf_calculation(ccf_params, wave, fluxes, blaze, targetrv, ccfmask["centers"],
                            ccfmask["weights"], berv, fit_type, mask_orders=ccfmask["orders"])

    # --------------------------------------------------------------------------
    # Calculate the mean CCF
    # --------------------------------------------------------------------------
    if verbose :
        print('\nRunning Mean CCF')
    props = mean_ccf(ccf_params, props, targetrv, fit_type, valid_orders=valid_orders, normalize_ccfs=normalize_ccfs, plot=plot)

    # --------------------------------------------------------------------------
    # Save file
    # --------------------------------------------------------------------------
    #if "WAVEORDN" not in header.keys() :
    #    header = make_wave_keywords(header)
    wheader = header

    props = write_file(props, filename, ccf_params['MASK_FILE'], header, wheader, rv_drifts, save=output, verbose=verbose)

    return props

# ==============================================================================
# End of code
# ==============================================================================


