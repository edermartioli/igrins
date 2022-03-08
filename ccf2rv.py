# -*- coding: iso-8859-1 -*-
"""
    Created on September 30 2020
    
    Description: Library to obtain radial velocity measurements from SPIRou CCFs
    
    @author: Etienne Artigau (UdM), Eder Martioli (IAP)
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import os, sys
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import warnings
#from check_blacklist import *
from astropy.time import Time
from collections import Counter
from astropy.io import ascii
import gp_lib

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

def run_ccf_analysis(ccf_files, mask_file, obj="", drs_version="", snr_min=20., dvmax_per_order=3.0, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, exclude_orders = [-1], velocity_window=10., pixel_size_in_kps=2.3, fpccf=False, outdir="", detailed_output=True, plot=False, verbose=False) :
    
    """
        Description: a wrapper to run the CCF template matching analysis
        """
    
    ##########################
    # Run CCF analysis to get object rvs using an optimized algorithm
    mask_basename = os.path.basename(mask_file)
    
    collection_key = "{}__{}__{}__{}".format(obj, mask_basename, sanit, drs_version)
    
    if verbose :
        print("Running CCF analysis on CCF data for collection: {}".format(collection_key))

    # set bandpass
    bandpass = "YJHK"

    if outdir == "":
        outdir = os.path.dirname(ccf_files[0])

    # form a unique batch name with mask, object and method
    batch_name = '{0}/{1}__{2}'.format(outdir, collection_key, bandpass)

    ccf = get_object_rv(ccf_files, collection_key=collection_key, method="all", exclude_orders=exclude_orders, snr_min=snr_min, bandpass=bandpass,velocity_window=velocity_window, dvmax_per_order=dvmax_per_order, pixel_size_in_kps=pixel_size_in_kps, save_rdb_timeseries=True, correct_rv_drift=correct_rv_drift, save_csv_table_of_results=True, save_ccf_cube=False, save_weight_table=False, doplot=plot, showplots=plot, save_ccf_fitsfile=save_ccf_fitsfile, saveplots=plot, detailed_output=detailed_output, verbose=verbose, fpccf=fpccf)

    return ccf
    ##########################


def get_object_rv(ccf_files,
                  collection_key='',
                  method = 'all',
                  exclude_orders = [-1],
                  weight_table = '',
                  snr_min = 0.0,
                  weight_type = '',
                  bandpass = 'YJHK',
                  velocity_window = 10.,
                  dvmax_per_order = 3.0,
                  pixel_size_in_kps = 2.3,
                  save_rdb_timeseries = True,
                  correct_rv_drift = True,
                  save_bisector_timeseries = True,
                  save_csv_table_of_results = True,
                  save_ccf_cube = False,
                  save_weight_table = False,
                  doplot = False,
                  showplots = False,
                  saveplots = False,
                  do_blacklist = False,
                  save_ccf_fitsfile = False,
                  detailed_output = False,
                  verbose = False,
                  fpccf = False
                  ):
    #
    # parameters :
    #
    # object -> name of the object to be analyzed, linked to the folder where the data should be. You need
    #           to use dispatch_object first to put the data where it belongs
    #
    #
    # mask = X -> name of the mask used for the CCF. You can have a number of different masks for the same object
    #           and these should be processed individually
    #
    # if you want to systematically exclude orders, then set exclude_orders = [a,b,c,d]. Leave to -1 if you want to
    #          keep all valid orders
    #
    # method = X -> Method to measured velocity. Add you own! For now we have---
    #
    #          bisector_N_M returns the velocity between N and Mth percentile of line depth.
    #
    #         gaussfit --> fits a gaussian to the mean CCF
    #
    # if you want to force a weight onto orders rather than let the program decide automatically, then provide a
    # value for weight_table (e.g., weigth_table = "my_weights.csv". It must have 49 rows, one column name
    # "WEIGHT" (all capitals) and be read as a proper numpy.table file
    #

    #if method == 'all':
    #    method = 'gaussian_template_bisector_20_80'

    if detailed_output:
        dict_ccf = dict()
    
    object, mask, sanit, drs_version = collection_key.split("__")

    outdir = os.path.dirname(ccf_files[0])
    
    if verbose :
        print("Saving products in directory:{}".format(outdir))

    # add excluded orders based on the bandpass
    exclude_orders = set_exclude_orders_by_bandpass(exclude_orders, ccf_files[0], bandpass)

    #if do_blacklist:
    #    ccf_files = check_blacklist(ccf_files)

    # form a unique batch name with mask, object and method
    batch_name = '{0}/{1}__{2}'.format(outdir, collection_key, bandpass)

    # build ccf cube
    ccf_cube, ccf_tbl, ccf_RV = build_ccf_cube(ccf_files, batch_name, exclude_orders=exclude_orders, save_ccf_cube=save_ccf_cube, verbose=verbose, fpccf=fpccf)

    # keywords from file headers to be added to the CSV table.
    keywords = ['MJDATE','BERV','BJD','RV_DRIFT','ERVDRIFT','EXTSN035','AIRMASS','TLPEH2O','TLPEOTR','RV_WAVFP','RV_SIMFP','DATE','MJDMID','DATE-OBS','EXPTIME','EXPNUM','MJDEND','SBCDEN_P','EXPTYPE','FILENAME','FIBER','WAVETIME']
    # set output csv table
    tbl = set_output_table(ccf_files, keywords)

    # we apply the SNR threshold
    tbl, ccf_cube, ccf_files = apply_snr_threshold(tbl, ccf_cube, ccf_files, snr_key='EXTSN035',snr_min=snr_min)

    # calculate the median CCF for all epochs for 49 orders
    with warnings.catch_warnings(record=True) as _:
        # some slices in the sum are NaNs, that's OK
        med_ccf = np.nanmedian(ccf_cube , axis = 2)

    # exclude orders full of NaNs
    exclude_orders = exclude_orders_full_of_nans(exclude_orders, med_ccf, verbose=verbose)

    # exclude orders with measured RVs offset by more than a pre-defined threshold
    exclude_orders = exclude_orders_with_large_rv_offsets(exclude_orders, med_ccf, ccf_RV, dvmax_per_order, verbose=verbose)

    # set updated excluded orders values in ccf_cube to NaN
    ccf_cube[exclude_orders, :, :] = np.nan

    # find minimum for CCF. This is used to fit a gaussian to each order and force velocity to zero
    id_min = np.nanargmin(np.nanmedian(med_ccf,axis=0))

    nsigcut=4
    maxfrac_of_spectra_to_keep_order=0.5
    if fpccf :
        nsigcut=20

    # measure CCF weights
    weights = measure_ccf_weights(ccf_cube, ccf_files, med_ccf, ccf_RV, id_min, velocity_window, nsigcut=nsigcut, maxfrac_of_spectra_to_keep_order=maxfrac_of_spectra_to_keep_order, exclude_orders=exclude_orders,  batch_name=batch_name, weight_table=weight_table, weight_type=weight_type, obj=object, mask=mask, save_weight_table=save_weight_table, doplot=doplot, saveplots=saveplots, showplots=showplots, verbose=verbose)

    if doplot:
        # plot median CCFs and residuals
        plot_median_ccfs_and_residuals(ccf_cube, med_ccf, ccf_RV, batch_name, saveplots=saveplots, showplots=showplots)
        # plot SNR
        plot_snr(tbl, batch_name, saveplots=saveplots, showplots=showplots)

    ccf_cube_norm, mean_ccf = apply_weights_to_ccf(ccf_cube, weights)

    if doplot:
        plot_weighted_mean_ccfs(ccf_files, ccf_RV, mean_ccf, batch_name, saveplots=saveplots, showplots=showplots)

    # if method == "bisector" or method == "all":
    # measure RVs using bisector method
    tbl = run_bisector_method(tbl, ccf_files, ccf_RV, mean_ccf, verbose=verbose)
    if doplot :
        plot_bisector_method(tbl, batch_name, saveplots=saveplots, showplots=showplots)

    #if method == "gaussian" or method == "all":
    # measure RVs using gaussian method
    tbl = run_gaussian_method(tbl, ccf_files, ccf_RV, mean_ccf)
    if doplot:
        plot_gaussian_method(tbl, batch_name, saveplots=saveplots, showplots=showplots)

    #if method == "template" or method == "all":
    # measure RVs using template method
    tbl, med_corr_ccf, corr_ccf = run_template_method(tbl, ccf_files, ccf_RV, mean_ccf, id_min, velocity_window, doplot=doplot, showplots=showplots, verbose=verbose)

    if doplot:
        plot_corr_ccf(ccf_files, ccf_RV, corr_ccf, batch_name, id_min, saveplots=saveplots, showplots=showplots)

    if fpccf :
        # Calculate systemic velocities using gaussian method -- why not from template?
        tbl = add_gaussian_systemic_velocity(tbl, ccf_RV, med_corr_ccf)
    else :
        # Calculate systemic velocities using bisector method -- why not from template?
        tbl = add_bisector_systemic_velocity(tbl, ccf_RV, med_corr_ccf, low_high_cut=0.3, object=object, saveplots=saveplots, showplots=showplots, doplot=doplot)
        
    # add a measurement of the STDDEV of each mean CCF relative to the median CCF after correcting for the measured velocity. If you are going to add 'methods', add them before this line
    med_corr_ccf, corr_ccf = add_stddev_to_ccf(ccf_files, tbl, ccf_RV, corr_ccf, id_min, doplot=False)

    tbl = calculate_resid_ccf_projections(ccf_files, tbl, ccf_RV, med_corr_ccf, corr_ccf, id_min, velocity_window, pixel_size_in_kps=pixel_size_in_kps)

    if doplot :
        plot_residual_ccf(ccf_files, ccf_RV, med_corr_ccf, corr_ccf, batch_name, saveplots=saveplots, showplots=showplots)

    #tbl = run_gp_gaussian_method(tbl, ccf_files, ccf_RV, corr_ccf, med_corr_ccf)
    #if doplot:
    #   plot_gaussian_method(tbl, batch_name, saveplots=saveplots, showplots=showplots)

    if save_ccf_fitsfile :
        output_ccf_fitsfile = '{0}.fits'.format(batch_name)
        if detailed_output:
            dict_ccf['CCF_FITS_FILE'] = output_ccf_fitsfile
    else :
        output_ccf_fitsfile = ""
        
    ccf_fits_hdu = save_ccf_to_fits(ccf_files, ccf_RV, med_corr_ccf, corr_ccf, output=output_ccf_fitsfile, plot=doplot)

    if doplot :
        plot_residual_timeseries(tbl, batch_name, saveplots=saveplots, showplots=showplots)

    # output to csv file
    if save_csv_table_of_results :
        tbl.write('{0}.csv'.format(batch_name),overwrite = True)

    if fpccf :
        if save_rdb_timeseries :
            fp_rdb_output = '{0}_fpdrift.rdb'.format(batch_name)
            if detailed_output:
                dict_ccf['FP_RDB_OUTPUT'] = fp_rdb_output
            save_drift_time_series_in_rdb_format(tbl, fp_rdb_output, rv_key='RV')

    else :
        if save_rdb_timeseries :
            if method == 'bisector' or method == 'all' :
                bisector_rv_output = '{0}_bis_rv.rdb'.format(batch_name)
                save_rv_time_series_in_rdb_format(tbl, bisector_rv_output, rv_key='RV_BIS', correct_rv_drift=correct_rv_drift)
            if method == 'gaussian' or method == 'all' :
                gaussian_rv_output = '{0}_gauss_rv.rdb'.format(batch_name)
                save_rv_time_series_in_rdb_format(tbl, gaussian_rv_output, rv_key='RV_GAUSS', correct_rv_drift=correct_rv_drift)
            if method == 'template' or method == 'all' :
                template_rv_output = '{0}_template_rv.rdb'.format(batch_name)
                save_rv_time_series_in_rdb_format(tbl, template_rv_output, rv_key='RV', correct_rv_drift=correct_rv_drift)
                    
        if save_bisector_timeseries :
            bisector_output = '{0}_bisector.rdb'.format(batch_name)
            save_bisector_time_series_in_rdb_format(tbl, bisector_output)
        
    if detailed_output :
        dict_ccf['TABLE_CCF'] = tbl
            
        dict_ccf['CCF_RV'] = ccf_RV
        dict_ccf['MEAN_CCF'] = mean_ccf
        dict_ccf['CCF_FILES'] = ccf_files
        dict_ccf['CORR_CCF'] = corr_ccf
        dict_ccf['MED_CORR_CCF'] = med_corr_ccf
        dict_ccf['CCF_MAD'] = ccf_fits_hdu["DATA"].data["CCFERR"]

        if correct_rv_drift and 'RV_DRIFT' in tbl.keys():
            dict_ccf['RV'] = tbl['RV'] - tbl['RV_DRIFT']
            if 'ERVDRIFT' in tbl.keys():
                dict_ccf['RVERR'] = np.sqrt(tbl['ERROR_RV']*tbl['ERROR_RV'] + tbl['ERVDRIFT']*tbl['ERVDRIFT'])
            else :
                dict_ccf['RVERR'] = tbl['ERROR_RV']
        else :
            dict_ccf['RV'] = tbl['RV']
            dict_ccf['RVERR'] = tbl['ERROR_RV']
            
        if fpccf :
            dict_ccf['BJD'] = tbl['MJDATE']
        else :
            #dict_ccf['BJD'] = tbl['BJD'] + (tbl['MJDEND'] - tbl['MJDATE']) / 2.
            dict_ccf['BJD'] = tbl['BJD']

        return dict_ccf
    else:
        return tbl



def set_output_table(ccf_files, keywords) :

    tbl = Table()# output table to be saved as CSV file with RV measurements
    tbl['FILES'] = ccf_files

    tbl['RV'] = np.zeros_like(ccf_files,dtype = float) # measured RV
    tbl['ERROR_RV'] = np.zeros_like(ccf_files,dtype = float) # measured RV error

    tbl = load_values_from_headers(ccf_files, tbl, keywords)
    
    return tbl


def fits2wave(file_or_header):
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
    if 'WAVEORDN' in hdr.keys() :
        nord = hdr['WAVEORDN']
    elif 'TH_ORD_N' in hdr.keys() :
        nord = hdr['TH_ORD_N']
    else :
        nord = 49

    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)

    # get the length of each order (normally that's 4088 pix)
    npix = hdr['NAXIS1']

    # project polynomial coefficiels
    wavesol = [np.polyval(wave_poly[i][::-1],np.arange(npix)) for i in range(nord) ]

    # return wave grid
    return np.array(wavesol)


def bisector(rv, ccf,  low_high_cut = 0.1, figure_title = '', doplot = False, ccf_plot_file = '', showplot=False):
    # use the props from the CCF determination code
    # Could be per-order or with the mean
    #rv = props['RV_CCF']
    #ccf = props['MEAN_CCF']

    # get minima
    imin = int(np.argmin(ccf))
    #print(imin,type(imin))

    # get point where the derivative changes sign at the edge of the line
    # the bisector is ambiguous passed this poind
    width_blue =  imin - np.max(np.where(np.gradient(ccf[:imin])>0))
    #print(width_blue)
    width_red = np.min(np.where(np.gradient(ccf[imin:])<0))
    #print(width_red)

    # get the width from the side of the center that reaches
    # that point first
    width = int(np.min([width_blue, width_red]))

    # set depth to zero
    ccf -= np.min(ccf)

    # set continuum to one
    ccf /= np.min( ccf[ [imin - width,imin + width] ])

    # interpolate each side of the ccf slope at a range of depths
    depth = np.arange(low_high_cut,1-low_high_cut,0.001)

    # blue and red side of line
    g1 = (ccf[imin:imin - width:-1]>low_high_cut) & (ccf[imin:imin - width:-1]<(1-low_high_cut))
    spline1 = ius(ccf[imin:imin - width:-1][g1],rv[imin:imin - width:-1 ][g1], k=2)

    g2 = (ccf[imin : imin + width]>low_high_cut) & (ccf[imin : imin + width]<(1-low_high_cut))
    spline2 = ius(ccf[imin : imin + width][g2],rv[imin : imin + width][g2], k=2)

    # get midpoint
    bisector_position = (spline2(depth)+spline1(depth))/2

    # get bisector widht
    width_ccf = (spline2(depth)-spline1(depth))

    if doplot:
        # some nice plots
        plt.plot(rv[imin - width : imin+ width],ccf[imin - width : imin+ width],label = 'ccf')
        plt.plot(bisector_position,depth,label = 'bisector')
        plt.plot((bisector_position-np.mean(bisector_position))*100+np.mean(bisector_position),depth, label = 'bisector * 100',
                 )
        plt.legend()
        plt.title(figure_title,fontsize=16)
        plt.xlabel('Velocity (km/s)',fontsize=16)
        plt.ylabel('Depth',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if ccf_plot_file !='':
            plt.savefig(ccf_plot_file)
        if showplot :
            plt.show()

    # define depth in the same way as Perryman, 0 is top, 1 is bottom
    return 1-depth, bisector_position, width_ccf


def gauss(v,v0,ew,zp,amp):
    # gaussian with a constant offset. As we know that the ccfs are negative structures, amp will be negative
    return zp+amp*np.exp( -0.5*(v-v0)**2/ew**2)


def set_exclude_orders_by_bandpass(exclude_orders, reffile, bandpass) :

    # definition of photometric bandpasses
    #http: // svo2.cab.inta - csic.es / svo / theory / fps3 / index.php?mode = browse & gname = CFHT & gname2 = Wircam & asttype =
    # see values here
    # Y = 938.600	1113.400
    # J = 1153.586	1354.422
    # H = 1462.897	1808.544
    # k = 1957.792	2400 # modified to get to the edge of the SPIRou domain

    # get typical wavelength solution from first file and central wavelength grid per order
    wave_middle = np.nanmean(fits2wave(fits.getheader(reffile, ext=1)), axis=1)
    
    keep_orders = np.zeros(49)
    if 'Y' in bandpass:
        keep_orders[(wave_middle>938)*(wave_middle<1113)] = True
    if 'J' in bandpass:
        keep_orders[(wave_middle>1113)*(wave_middle<1462)] = True
    if 'H' in bandpass:
        keep_orders[(wave_middle>1462)*(wave_middle<1957)] = True
    if 'K' in bandpass:
        keep_orders[(wave_middle>1957)*(wave_middle<2500)] = True

    for i in range(49):
        if i in exclude_orders:
            keep_orders[i] = False

    # update orders rejected because of domain as well as input requirements.
    exclude_orders= np.where(keep_orders == False)[0]

    return exclude_orders


def load_values_from_headers(ccf_files, tbl, keywords) :
    
    for i in (range(len(ccf_files))):
        hdr = fits.getheader(ccf_files[i],ext = 1)
        if i ==0:
            # now that we have a first header, we add the relevant columns to the CSV table
            for key in keywords:
                if key in hdr:

                    key_type = type(hdr[key])
                    # if we have a string, we set the table to accept long values (up to 99 characters)
                    if key_type == str:
                        key_type = '<U99'
                else:
                    # keyword not in header, we need to assume something. The safest is string
                    key_type = str

                # add the column to the CSV file
                tbl[key] = np.zeros_like(ccf_files,dtype = key_type)

        for key in keywords:
            if key in hdr:
                if hdr[key] == 'None':
                    tbl[key][i] = np.nan
                else :
                    tbl[key][i] = hdr[key]

    return tbl


def apply_snr_threshold(tbl, ccf_cube, ccf_files, snr_key='EXTSN035', snr_min=1) :
    
    keep = tbl[snr_key]>snr_min
    tbl = tbl[keep]
    ccf_cube = ccf_cube[:,:,keep]
    ccf_files = list(np.array(ccf_files)[keep])

    return tbl, ccf_cube, ccf_files


def build_ccf_cube(ccf_files,
                   batch_name,
                   exclude_orders=[-1],
                   save_ccf_cube=False,
                   verbose=False, fpccf=False):

    npy_file = '{}_ccf_cube.npy'.format(batch_name)

    if not os.path.isfile(npy_file):
        if verbose:
            print('we load all CCFs into one big cube')
        ccf_RV_previous = None
        for i in (range(len(ccf_files))):
            # we loop through all files
            ccf_tbl = fits.getdata(ccf_files[i])

            # ccf velocity offset, not to be confused with measured RV
            ccf_RV = ccf_tbl['RV']

            # We must absolutely have always the same RV grid for the CCF.
            # We must check that consecutive ccf_RV are identical
            if i != 0:
                if np.sum(ccf_RV != ccf_RV_previous):
                    print('We have a big problem!'
                          'The RV vector of CCF files are not all the same')
                    print(
                        'Files {0} and {1} have different RV vectors.'.format(
                            ccf_files[i-1], ccf_files[i]
                            )
                        )
                    sys.exit()
            ccf_RV_previous = np.array(ccf_RV)

            if verbose:
                print('V[min/max] {0:.1f} / {1:.1f} km/s, MJD={2:.6f} file {3}'.format(
                    np.min(ccf_RV), np.max(ccf_RV), fits.getheader(ccf_files[i],1)['MJDATE'], ccf_files[i]
                    ))

            # if this is the first file, we create a cube that contains all
            # CCFs for all orders for all files
            if i == 0:
                ccf_cube = np.zeros([49, len(ccf_tbl), len(ccf_files)])+np.nan

            # we input the CCFs in the CCF cube
            for j in range(49):
                tmp = ccf_tbl['ORDER'+str(j).zfill(2)]

                if fpccf :
                    tmp = - tmp + np.nanmax(tmp)
                
                if False not in np.isfinite(tmp):
                    # we normalize to a continuum of 1
                    tmp /= np.polyval(np.polyfit(ccf_RV, tmp, 1), ccf_RV)
                    ccf_cube[j, :, i] = tmp

        if save_ccf_cube:
            if verbose:
                print('We save {0}, this will speed things up '
                      'next time you run this code'.format(npy_file))
            np.save(npy_file, ccf_cube)

    else:
        if verbose:
            print('We load {0}, this is speedier'.format(npy_file))
        ccf_cube = np.load(npy_file)

        # we need to load the first file just to get the velocity grid
        ccf_tbl = fits.getdata(ccf_files[0])
        ccf_RV = ccf_tbl['RV']

    for j in range(49):
        # if we need to exlude orders, we do it here.
        if j in exclude_orders:
            ccf_cube[j, :, :] = np.nan

    return ccf_cube, ccf_tbl, ccf_RV


def exclude_orders_full_of_nans(exclude_orders, med_ccf, verbose=False) :
    
    for iord in range(49):
        if iord not in exclude_orders:
            if not np.isfinite(np.mean(med_ccf[iord,:])):
                if verbose :
                    print('Order {0} has a CCF full of NaNs. Added to the rejected orders '.format(iord))
                exclude_orders = np.append(exclude_orders,iord)

    return exclude_orders


def exclude_orders_with_large_rv_offsets(exclude_orders, med_ccf, ccf_RV, dvmax_per_order, verbose=False) :
    
    # find minimum for CCF. This is used to fit a gaussian to each order and force velocity to zero
    id_min = np.nanargmin(np.nanmedian(med_ccf,axis=0))

    # find if a given CCF is off by more than the pre-defined threshold
    dv_CCF_min = (ccf_RV[np.argmin(med_ccf,axis = 1)] - ccf_RV[id_min])
    bad_orders = dvmax_per_order < np.abs(dv_CCF_min)

    for iord in range(49):
        if iord not in exclude_orders:
            if bad_orders[iord]:
                if verbose :
                    print('The CCF of order {0} has its minima {1:.2f} km/s from median CCF, above threshold of +-{2:.2f} km/s'.
                      format(iord,dv_CCF_min[iord],dvmax_per_order))
                exclude_orders = np.append(exclude_orders,iord)
    
    return exclude_orders


def measure_ccf_weights(
        ccf_cube,
        ccf_files,
        med_ccf,
        ccf_RV,
        id_min,
        velocity_window,
        nsigcut = 3.,
        maxfrac_of_spectra_to_keep_order = 0.75,
        exclude_orders=[-1],
        batch_name='std_output',
        weight_table='',
        weight_type='',
        obj='',
        mask='',
        save_weight_table=False,
        doplot=False,
        saveplots=False,
        showplots=False,
        verbose=False,
        ):
    # Find valid pixels to measure CCF properties
    g = np.abs(ccf_RV - ccf_RV[id_min]) < velocity_window

    with warnings.catch_warnings(record=True) as _:
        # Some slices in the sum are NaNs, that's OK
        ccf_Q = np.nansum(np.gradient(med_ccf[:, g], axis=1)**2, axis=1)

    ccf_Q[ccf_Q == 0] = np.nan
    ccf_depth = 1-med_ccf[:, id_min]

    if weight_type == 'DVRMS_CC':
        raise ValueError('DVRMS_CC not available yet')
        # Commented out. We can implement this on separate branch
        # weights = 1/np.nanmedian(DVRMS_CC, axis=1)**2
        # weights[np.isfinite(weights) == False] = 0

        # for iord in range(49):
        #     if iord in exclude_orders:
        #         weights[iord] = 0

        # weights = weights/np.sum(weights)
    else:
        if weight_table == "" or (not os.path.isfile(weight_table)):
            # now we find the RMS of the Nth spectrum relative to the median
            rms = np.zeros([len(ccf_files), 49])

            for iter in range(5) :
                for i in range(len(ccf_files)):
                    with warnings.catch_warnings(record=True) as _:
                        # Some slices in the median are NaNs, that's OK
                        rms[i, :] = np.nanmedian(np.abs(ccf_cube[:, :, i] - med_ccf),axis=1)
                    rms[i, :] /= np.nanmedian(rms[i, :])

                rms[:, exclude_orders] = np.nan
                
                for iiter in range(5) :
                    nonzero = rms != 0
                    median_rms = np.nanmedian(rms[nonzero])
                    mad_rms = np.nanmedian(np.abs(rms[nonzero] - median_rms)) / 0.67449
                    cutcond = median_rms + nsigcut * mad_rms
                    #print("iter={} cutting off CCF data with RMS > {}".format(iter,cutcond))
                    rms[rms > cutcond] = 0.
            
                #for i in range(len(ccf_files)) :
                #   rms_in_time =
                # Exclude order if number of valid files is less than two thirds of initial sample
                for order in range(49) :
                    order_rms = np.array(rms[:,order])
                    nonzeroo = order_rms != 0
                    nonzeroo &= np.isfinite(order_rms)
                    #print("iter={} iiter={} order={} nvalid_files={}".format(iter, iiter,order,len(rms[:,order][nonzeroo])) )
                    if len(order_rms[nonzeroo]) < maxfrac_of_spectra_to_keep_order * len(ccf_files) :
                        rms[:, order] = 0.
                
                # First check if spectra is good compared to overall data:
                for order in range(49) :
                    for i in range(len(ccf_files)) :
                        if ~np.isfinite(rms[i,order]) or rms[i,order] == 0:
                            ccf_cube[order,:,i] = np.nan

                rms[:, exclude_orders] = np.nan
            
            if doplot:
                fig = plt.figure()
                vmin = np.nanpercentile(rms, 3)
                vmax = np.nanpercentile(rms, 97)
                plt.imshow(rms, aspect='auto', vmin=vmin, vmax=vmax)
                plt.xlabel('Nth order',fontsize=16)
                plt.ylabel('Nth frame',fontsize=16)
                plt.title('RMS of CCF relative to median',fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                if showplots:
                    plt.show()
                plt.close(fig)

            with warnings.catch_warnings(record=True) as _:
                # some slices in the sum are NaNs, that's OK
                # this is the typical noise from the ccf dispersion
                ccf_rms = np.nanmedian(rms, axis=0)

            # set to NaN values that are invalid
            ccf_rms[ccf_rms == 0] = np.nan

            # Assuming that the CCF has the same depth everywhere,
            # this is the correct weighting of orders
            weights = ccf_Q/ccf_rms**2
            weights[weights == 0] = np.nan
            weights[exclude_orders] = np.nan
            # we normalize the sum of the weights to one
            weights /= np.nansum(weights)

            if doplot:
                fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
                ax[0].plot(weights, 'go')
                ax[0].set(title='{0}, mask {1}'.format(obj, mask),
                          xlabel='Nth order', ylabel='Relative order weight')

                ax[1].plot(ccf_Q, 'go')
                ax[1].set(xlabel='Nth order', ylabel='ccf Q')

                ax[2].plot(1/ccf_rms**2, 'go')
                ax[2].set(xlabel='Nth order', ylabel='1/$\\sigma_{CCF}^2$')
                plt.tight_layout()
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                if saveplots:
                    plt.savefig('{0}_weights.pdf'.format(batch_name))
                if showplots:
                    plt.show()

            tbl_weights = Table()
            tbl_weights['order'] = np.arange(49)
            tbl_weights['weights'] = weights
            tbl_weights['ccf_depth'] = ccf_depth
            tbl_weights['ccf_Q'] = ccf_Q
            if save_weight_table:
                tbl_weights.write('{0}_weights.csv'.format(batch_name),
                                  overwrite=True)

        else:
            if verbose:
                print('You provided a weight file, we load it and'
                      'apply weights accordingly')
            tbl_weights = Table.read(weight_table)
            weights = np.array(tbl_weights['weights'], dtype=float)
            weights /= np.nansum(weights)

    return weights


def apply_weights_to_ccf(ccf_cube, weights) :
    
    ccf_cube_norm = np.zeros_like(ccf_cube)
    for i in range(49):
        if np.isfinite(weights[i]):
            ccf_cube_norm[i, :, :] = (ccf_cube[i,:,:] * weights[i])

    # get a per-file weighted mean CCF
    mean_ccf = np.nansum(ccf_cube_norm, axis=0)

    return ccf_cube_norm, mean_ccf


def plot_weighted_mean_ccfs(ccf_files, ccf_RV, mean_ccf, batch_name, saveplots=False, showplots=False) :

    fig,ax = plt.subplots(nrows = 1, ncols = 1)
    for i in range(len(ccf_files)):
        color = [i/len(ccf_files),1-i/len(ccf_files),1-i/len(ccf_files)]
        ax.plot(ccf_RV,mean_ccf[:,i],color = color,alpha = 0.2)
    
    ax.set(xlabel = 'Velocity [km/s]',ylabel = 'CCF depth', title = 'Mean CCFs')
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_CCFs.pdf'.format(batch_name))
    if showplots :
        plt.show()


def plot_median_ccfs_and_residuals(ccf_cube, med_ccf, ccf_RV, batch_name, saveplots=False, showplots=False) :
    
    plt.imshow(med_ccf, aspect = 'auto',vmin = 0.8,vmax= 1.05,extent = [np.min(ccf_RV),np.max(ccf_RV),49,0])
    plt.xlabel('Velocity bin [km/s] ',fontsize=16)
    plt.ylabel('Nth order',fontsize=16)
    plt.title('Median CCF',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_medianccf.pdf'.format(batch_name))
    if showplots :
        plt.show()

    plt.imshow(ccf_cube[:,:,0]-med_ccf,aspect = 'auto',vmin = -0.1,vmax= 0.1,extent = [np.min(ccf_RV),np.max(ccf_RV),49,0])
    plt.xlabel('Velocity bin [km/s]',fontsize=16)
    plt.ylabel('Nth order',fontsize=16)
    plt.title('Sample residual CCF map',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_residualccf.pdf'.format(batch_name))
    if showplots :
        plt.show()


def plot_snr(tbl, batch_name, saveplots=False, showplots=False) :

    plt.plot(tbl['MJDATE'], tbl['EXTSN035'], 'g.')
    plt.xlabel('MJDATE',fontsize=16)
    plt.ylabel('SNR for order 35\n(around 1.6 $\mu$m)',fontsize=16)
    plt.title('Signal-to-noise ratio',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_snr.pdf'.format(batch_name))
    if showplots :
        plt.show()


def run_bisector_method(tbl, ccf_files, ccf_RV, mean_ccf, bis_min=0.3, bis_max=0.8, verbose=False) :
    
    tbl['RV_BIS'] = np.zeros_like(ccf_files,dtype = float)  # bisector mid point
    tbl['BIS_SLOPE'] = np.zeros_like(ccf_files,dtype = float)  # bisector slope
    tbl['BIS_WIDTH'] = np.zeros_like(ccf_files,dtype = float)  # bisector width
    tbl['Vt'] = np.zeros_like(ccf_files,dtype = float)  # bisector velocity 'top' in perryman
    tbl['Vt_ERR'] = np.zeros_like(ccf_files,dtype = float)  # bisector velocity 'top' in perryman
    tbl['Vb'] = np.zeros_like(ccf_files,dtype = float)  # bisector velocity 'bottom' in perryman
    tbl['Vb_ERR'] = np.zeros_like(ccf_files,dtype = float)  # bisector velocity 'bottom' in perryman
    tbl['BIS'] = np.zeros_like(ccf_files,dtype = float)  # bisector velocity width
    tbl['BIS_ERR'] = np.zeros_like(ccf_files,dtype = float)  # bisector velocity width

    for i in range(len(ccf_files)):
        
        try:
            depth, bis, width = bisector(ccf_RV, mean_ccf[:,i], low_high_cut=0.2)
        
            fit = np.polyfit(depth[(depth > bis_min) & (depth < bis_max)] - (bis_min + bis_max) / 2, bis[(depth > bis_min) & (depth < bis_max)], 1)

            tbl['RV'][i] =  fit[1]
            # just in case you want to have both bisector and
            # template, we keep a RV that is specific to this method
            tbl['RV_BIS'][i] =  fit[1]
            tbl['BIS_SLOPE'][i] =  fit[0]
            tbl['BIS_WIDTH'][i] = np.mean(width[(depth > bis_min) & (depth < bis_max)])

            # mean 'top' CCF between 55 and 80% of depth
            tbl['Vt'][i] = np.nanmean(bis[(depth>0.55)*(depth<0.80)])
            tbl['Vt_ERR'][i] = np.nanstd(bis[(depth>0.55)*(depth<0.80)])
            # mean 'bottom' CCF between 20-40%
            tbl['Vb'][i] = np.nanmean(bis[(depth>0.20)*(depth<0.40)])
            tbl['Vb_ERR'][i] = np.nanstd(bis[(depth>0.20)*(depth<0.40)])

            tbl['BIS'][i] = tbl['Vt'][i] - tbl['Vb'][i]
            tbl['BIS_ERR'][i] = np.sqrt(tbl['Vt_ERR'][i]**2 + tbl['Vb_ERR'][i]**2)

        except :
            if verbose :
                print('There has been an error with file {0} computing the bisector'.format(ccf_files[i]))
                print('Values will be reported as NaN')
            tbl['RV'][i] =  np.nan
            tbl['RV_BIS'][i] =  np.nan
            tbl['BIS_SLOPE'][i] =  np.nan
            tbl['BIS_WIDTH'][i] = np.nan
            tbl['Vt'][i] = np.nan
            tbl['Vt_ERR'][i] = np.nan
            tbl['Vb'][i] = np.nan
            tbl['Vb_ERR'][i] = np.nan
            tbl['BIS'][i] = np.nan
            tbl['BIS_ERR'][i] = np.nan
    return tbl


def plot_bisector_method(tbl, batch_name, saveplots=False, showplots=False) :
    
    fig,ax = plt.subplots(nrows = 2, ncols=1,sharex = True)
    ax[0].plot(tbl['MJDATE'], tbl['RV_BIS'], 'g.')
    ax[0].set(title='Velocity',xlabel = 'MJDATE',ylabel = 'RV [km/s]')
    ax[1].plot(tbl['MJDATE'], tbl['BIS_SLOPE'], 'g.')
    ax[1].set(title='Bisector slope',xlabel = 'MJDATE',ylabel = 'slope [km/s/fract. depth]')
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_bisector_RV.pdf'.format(batch_name))
    if showplots :
        plt.show()


def run_gaussian_method(tbl, ccf_files, ccf_RV, mean_ccf) :
    
    tbl['RV_GAUSS'] = np.zeros_like(ccf_files,dtype = float) # mean gauss velocity
    tbl['GAUSS_WIDTH'] = np.zeros_like(ccf_files,dtype = float)  # gauss width
    tbl['GAUSS_AMP'] = np.zeros_like(ccf_files,dtype = float)  # gauss depth
    tbl['GAUSS_ZP'] = np.zeros_like(ccf_files,dtype = float)  # gauss zp

    imin = np.argmin(np.nanmedian(mean_ccf, axis=1))
    
    for i in range(len(ccf_files)):
        p0 = [ccf_RV[imin],1,1,-0.1]
        fit, pcov = curve_fit(gauss,ccf_RV,mean_ccf[:,i],p0 = p0)
        fit_err = np.sqrt(np.diag(pcov))
        
        tbl['RV'][i] = fit[0]

        # just in case you want to have gauss/bisector and
        # template, we keep a RV that is specific to this method
        tbl['RV_GAUSS'][i] = fit[0]
        tbl['GAUSS_WIDTH'][i] = fit[1]
        tbl['GAUSS_AMP'][i] = fit[3]
        tbl['GAUSS_ZP'][i] = fit[2]

    return tbl


def run_gp_gaussian_method(tbl, ccf_files, ccf_RV, mean_ccf, med_corr_ccf) :
    
    tbl['RV_GAUSS'] = np.zeros_like(ccf_files,dtype = float) # mean gauss velocity
    tbl['RV_GAUSS_ERR'] = np.zeros_like(ccf_files,dtype = float) # mean gauss velocity
    tbl['GAUSS_WIDTH'] = np.zeros_like(ccf_files,dtype = float)  # gauss width
    tbl['GAUSS_WIDTH_ERR'] = np.zeros_like(ccf_files,dtype = float)  # gauss width
    tbl['GAUSS_AMP'] = np.zeros_like(ccf_files,dtype = float)  # gauss depth
    tbl['GAUSS_AMP_ERR'] = np.zeros_like(ccf_files,dtype = float)  # gauss depth
    tbl['GAUSS_ZP'] = np.zeros_like(ccf_files,dtype = float)  # gauss zp

    residuals = np.full_like(mean_ccf, np.nan)
    for i in range(len(ccf_files)):
        residuals[:,i] = mean_ccf[:,i] - med_corr_ccf
    ccf_errors = np.nanstd(residuals,axis=1)

    imin = np.argmin(np.nanmedian(mean_ccf, axis=1))
    
    for i in range(len(ccf_files)):
        for iter in range(10) :
            try :
                fit, err = gp_lib.fit_gaussian_profile(ccf_RV, mean_ccf[:,i], ccf_errors, burnin=100, nsamples=500, plot=False, verbose=False)
                break
            except :
                print("file={} iter={} : GP Gaussian fit failed, trying again ...".format(i, iter))

        #tbl['RV'][i] = fit[2]

        # just in case you want to have gauss/bisector and
        # template, we keep a RV that is specific to this method
        tbl['GAUSS_AMP'][i] = fit[2]
        tbl['GAUSS_AMP_ERR'][i] = err[2]

        tbl['RV_GAUSS'][i] = fit[3]
        tbl['RV_GAUSS_ERR'][i] = err[3]

        tbl['GAUSS_WIDTH'][i] = np.sqrt(np.exp(fit[4]))
        maxwidth = np.sqrt(np.exp(fit[4] + err[4]))
        minwidth = np.sqrt(np.exp(fit[4] - err[4]))
        width_err = (maxwidth - minwidth)/2
        tbl['GAUSS_WIDTH_ERR'][i] = width_err

    return tbl


def plot_gaussian_method(tbl, batch_name, saveplots=False, showplots=False) :

    fig,ax = plt.subplots(nrows = 2, ncols=1,sharex = True)
    ax[0].plot(tbl['MJDATE'], tbl['RV_GAUSS'], 'g.')
    ax[0].set(title='Velocity',xlabel = 'MJDATE',ylabel = 'RV [km/s]')
    ax[1].plot(tbl['MJDATE'], tbl['GAUSS_WIDTH']*2.354, 'g.')
    ax[1].set(title='Gaussian width',xlabel = 'MJDATE',ylabel = 'Gaussian FWHM [km/s]')
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_gaussian_RV.pdf'.format(batch_name))
    if showplots :
        plt.show()


def run_template_method(tbl, ccf_files, ccf_RV, mean_ccf, id_min, velocity_window, nite_max = 20, doplot=False, showplots=False, verbose=False) :

    g = np.abs(ccf_RV - ccf_RV[id_min]) < velocity_window

    rv_prev = np.array(tbl['RV'])

    ite = 0
    rms_rv_ite = np.inf
    
    # we iterate until we have an rms from iteration to iteration of <10 cm/s or we reached a max of 20 iterations

    corr_ccf = np.array(mean_ccf)
    
    if doplot:
        fig,ax = plt.subplots(nrows =1, ncols = 2)
        # funky scaling of imshow
        vmin = np.nanpercentile(corr_ccf,3)
        vmax = np.nanpercentile(corr_ccf,97)
        ax[0].imshow(corr_ccf,aspect = 'auto',vmin = vmin,vmax = vmax,extent = [0,len(ccf_files),np.min(ccf_RV),np.max(ccf_RV)])
        ax[0].set(xlabel='Nth observation',ylabel='Velocity [km/s]',title='Before CCF register')

    per_ccf_rms = np.ones(len(ccf_files))
    while (rms_rv_ite>1e-5) and (ite<nite_max):
        if ite ==0:
            tbl['RV'] = 0

        w = 1/per_ccf_rms**2
        w/=np.nansum(w)
        med_corr_ccf = np.zeros(len(ccf_RV))
        for i in range(len(w)):
            med_corr_ccf+=(corr_ccf[:,i]*w[i])

        # normalize continuum to 1
        continuum = np.abs(ccf_RV-ccf_RV[id_min])>velocity_window
        med_corr_ccf/=np.nanmedian(med_corr_ccf[continuum])

        fit = np.polyfit(ccf_RV[continuum], med_corr_ccf[continuum], 2)
        corr = np.polyval(fit, ccf_RV)
        corr -= np.nanmean(corr)
        med_corr_ccf -= corr

        for i in range(len(ccf_files)):
            spline = ius(ccf_RV,mean_ccf[:,i],ext=3,k=5)
            corr_ccf[:,i] = spline(ccf_RV+tbl['RV'][i])

            # correcting median of CCF
            med =  np.nanmedian(corr_ccf[:,i] - med_corr_ccf)
            mean_ccf[:, i] -= med

            # correcting depth of CCF
            amp = np.nansum( (corr_ccf[:,i] - np.nanmean(corr_ccf[:,i]))*(med_corr_ccf - np.nanmean(med_corr_ccf)) )/np.nansum((med_corr_ccf - np.nanmean(med_corr_ccf))**2)
            mean_ccf[:, i] = (mean_ccf[:,i] - np.nanmean(mean_ccf[:,i]))/np.sqrt(amp)+np.nanmean(mean_ccf[:,i])

            # correcting 2rd order polynomial structures in continuum
            fit = np.polyfit(ccf_RV,med_corr_ccf-corr_ccf[:,i],3)

            corr = np.polyval(fit, ccf_RV)
            mean_ccf[:, i] += corr/2

        deriv = np.gradient(med_corr_ccf) / np.gradient(ccf_RV)
        deriv = deriv[g]
        deriv = deriv / np.nansum(deriv ** 2)

        for i in range(len(ccf_files)):
            residu = corr_ccf[:,i] - med_corr_ccf
            per_ccf_rms[i] = np.nanstd(residu)
            tbl['RV'][i]-=np.nansum(residu[g]*deriv)

        tbl['RV'] -= np.nanmean(tbl['RV'])
        #plt.plot( tbl['RV'],'.')
        rms_rv_ite = np.nanstd(rv_prev - tbl['RV'])
        if verbose :
            print('Template CCF iteration number {0:3}, rms RV change {1:3.4f} km/s for this step'.format(ite+1,rms_rv_ite))
        rv_prev = np.array(tbl['RV'])
        ite+=1

    tbl['RV_TEMPLATE'] = np.array(tbl['RV'])

    if doplot:
        vmin = np.nanpercentile(corr_ccf,3)
        vmax = np.nanpercentile(corr_ccf,97)
        ax[1].imshow(corr_ccf,aspect = 'auto',vmin = vmin,vmax = vmax,extent = [0,len(ccf_files),np.min(ccf_RV),np.max(ccf_RV)])
        ax[1].set(xlabel='Nth observation',ylabel='Velocity [km/s]',title='After CCF register')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if showplots :
            plt.show()

    return tbl, med_corr_ccf, corr_ccf


def add_gaussian_systemic_velocity(tbl, ccf_RV, med_corr_ccf) :
    
    imin = np.argmin(med_corr_ccf)
    p0 = [ccf_RV[imin],1,1,-0.1]
    fit, pcov = curve_fit(gauss, ccf_RV, med_corr_ccf, p0 = p0)
    fit_err = np.sqrt(np.diag(pcov))

    tbl['RV']+= fit[0]
    
    return tbl


def add_bisector_systemic_velocity(tbl, ccf_RV, med_corr_ccf, low_high_cut=0.3, object="", saveplots=False, showplots=False, doplot=False) :
    
    bisector_ccf_plot_file = ''
    
    if saveplots :
        bisector_ccf_plot_file = 'bisector_ccf_{0}.pdf'.format(object)
    
    # we get the systemic velocity from the BISECTOR between 0.3 and 0.7 depth
    depth, bis, width = bisector(ccf_RV, med_corr_ccf, low_high_cut=low_high_cut, figure_title = 'mean CCF\ndebug plot', doplot=doplot, ccf_plot_file=bisector_ccf_plot_file, showplot=showplots)
    tbl['RV']+= np.nanmedian(bis)

    return tbl


def plot_corr_ccf(ccf_files, ccf_RV, corr_ccf, batch_name, id_min, saveplots=False, showplots=False):
    
    fig,ax = plt.subplots(nrows = 2, ncols = 1)
    for i in range(len(ccf_files)):
        color = [i/len(ccf_files),1-i/len(ccf_files),1-i/len(ccf_files)]
        ax[0].plot(ccf_RV,corr_ccf[:,i],color=color,alpha=0.2)
        ax[1].plot(ccf_RV,corr_ccf[:,i],color=color,alpha=0.2)

    ax[0].set(xlabel = 'Velocity [km/s]',ylabel = 'CCF depth', title = 'Mean CCFs')
    ax[1].set(xlabel = 'Velocity [km/s]',ylabel = 'CCF depth', title = 'Mean CCFs',xlim = [ccf_RV[id_min]-10, ccf_RV[id_min]+10])
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_template.pdf'.format(batch_name))
    if showplots :
        plt.show()


def add_stddev_to_ccf(ccf_files, tbl, ccf_RV, mean_ccf, id_min, doplot=False) :
    
    corr_ccf = np.array(mean_ccf)
    for i in range(len(ccf_files)):
        spline = ius(ccf_RV,mean_ccf[:,i],ext=3)
        corr_ccf[:,i] = spline(ccf_RV+tbl['RV'][i]-np.mean(tbl['RV']))
    
    med_corr_ccf = np.nanmedian(corr_ccf, axis=1)

    if doplot:
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(ccf_RV, med_corr_ccf, color='black', alpha=0.4,label = 'median CCF', linewidth=2)
        plt.show()

    return med_corr_ccf, corr_ccf


def calculate_resid_ccf_projections(ccf_files, tbl, ccf_RV, med_corr_ccf, corr_ccf, id_min, velocity_window, pixel_size_in_kps=2.3) :
    
    g = np.abs(ccf_RV - ccf_RV[id_min]) < velocity_window

    d2 = np.gradient(np.gradient(med_corr_ccf) / np.gradient(ccf_RV))
    d3 = np.gradient(np.gradient(np.gradient(med_corr_ccf) / np.gradient(ccf_RV)))
    # second derivatives
    tbl['D2_RESIDUAL_CCF'] = np.zeros_like(ccf_files, dtype = float)
    # third derivatives
    tbl['D3_RESIDUAL_CCF'] = np.zeros_like(ccf_files, dtype = float)
    # RMS of residual CCF w.r.t. the median ccf
    tbl['CCF_RESIDUAL_RMS']= np.zeros_like(ccf_files, dtype = float)

    # pix scale expressed in CCF pixels
    pix_scale = pixel_size_in_kps / np.nanmedian(np.gradient(ccf_RV))
    
    residuals = np.full_like(corr_ccf, np.nan)
    
    for i in range(len(ccf_files)):
        residual = corr_ccf[:,i] - med_corr_ccf
        
        residuals[:,i] = residual
        
        tbl['D2_RESIDUAL_CCF'][i] = np.nansum(residual*d2) / np.nansum(d2)
        tbl['D3_RESIDUAL_CCF'][i] = np.nansum(residual*d3) / np.nansum(d3)
        tbl['CCF_RESIDUAL_RMS'][i] = np.std(residual[g])
        
        # 1/dvrms -avoids division by zero
        inv_dvrms = (np.gradient(med_corr_ccf) / np.gradient(ccf_RV))/(np.nanstd(residual) * np.sqrt(pix_scale))
        tbl['ERROR_RV'][i] = 1 / np.sqrt(np.nansum(inv_dvrms ** 2))
        
    return tbl


def plot_residual_ccf(ccf_files, ccf_RV, med_corr_ccf, corr_ccf, batch_name, saveplots=False, showplots=False) :

    residuals = []
    
    fig,ax = plt.subplots(nrows = 2, ncols = 1, sharex=True)
    
    for i in range(len(ccf_files)):
        residual = corr_ccf[:,i] - med_corr_ccf

        color = [i/len(ccf_files),1-i/len(ccf_files),1-i/len(ccf_files)]

        ax[0].plot(ccf_RV, med_corr_ccf, color = "green", lw=2, label="median CCF")
        
        ax[0].plot(ccf_RV,corr_ccf[:,i],color = color, alpha = 0.2)
        
        ax[1].plot(ccf_RV,residual+1,color = color,alpha = 0.2)
    
        residuals.append(residual)

    residuals = np.array(residuals, dtype=float)
    ax[0].set(xlabel = 'Velocity [km/s]',ylabel = 'CCF depth', title = 'Mean CCFs')
    ax[1].set(xlabel = 'Velocity [km/s]',ylabel = 'CCF residual depth', title = 'Residual CCFs')
    plt.tight_layout()
    #plt.legend()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_residual_CCF.pdf'.format(batch_name))
    if showplots :
        plt.show()

    #vmin = np.nanpercentile(residuals,1)
    #vmax = np.nanpercentile(residuals,99)
    vmin = np.nanmin(residuals)
    vmax = np.nanmax(residuals)
    plt.imshow(residuals,aspect = 'auto',vmin = vmin, vmax = vmax,extent = [np.min(ccf_RV),np.max(ccf_RV),len(ccf_files),0])
    plt.ylabel('Nth observation',fontsize=16)
    plt.xlabel('Velocity [km/s]',fontsize=16)
    plt.title('Residual CCF',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if showplots :
        plt.show()


def save_ccf_to_fits(ccf_files, ccf_RV, med_corr_ccf, corr_ccf, output="", plot=False) :
    '''
        Function to save CCF data to FITS file
        '''
    
    residuals = []
    
    for i in range(len(ccf_files)):
        residual = corr_ccf[:,i] - med_corr_ccf
        if plot :
            color = [i/len(ccf_files),1-i/len(ccf_files),1-i/len(ccf_files)]
            plt.plot(ccf_RV,corr_ccf[:,i],color = color, alpha = 0.2)
        residuals.append(residual)
    
    residuals = np.array(residuals, dtype=float)

    ccf_err = np.nanmedian(np.abs(residuals), axis=0) / 0.67449

    if plot :
        plt.errorbar(ccf_RV, med_corr_ccf, yerr=ccf_err, fmt='o')
        plt.xlabel("Velocity [km/s]",fontsize=16)
        plt.ylabel("CCF depth",fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    # create header unit
    header = fits.Header()
    header.set('ORIGIN', "ccf2rv")
    
    # create primary hdu
    primary_hdu = fits.PrimaryHDU(header=header)
    
    # set extension hdu to save data
    ext_data = np.asarray(ccf_RV, dtype='float32')
    
    cols = []
    cols.append(fits.Column(name='RV', format='E', array=ccf_RV))
    cols.append(fits.Column(name='CCF', format='E', array=med_corr_ccf))
    cols.append(fits.Column(name='CCFERR', format='E', array=ccf_err))

    # set up hdu for all data extensions
    hdu = fits.BinTableHDU.from_columns(cols,name='DATA')
    
    # set up hdulist with primary + data HDUs
    fits_format = fits.HDUList([primary_hdu, hdu])

    # write output fits file
    if output != "" :
        fits_format.writeto(output, overwrite=True)

    return fits_format


def plot_residual_timeseries(tbl, batch_name, saveplots=False, showplots=False) :

    t3 = Time(tbl['MJDATE'], format='mjd')
    plt.plot_date(t3.plot_date, tbl['D2_RESIDUAL_CCF'], 'go')
    plt.title('Second derivative \n activity indicator')
    plt.xlabel('Date')
    plt.ylabel('CCF residual projection on\nCCF 2nd derivative')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if saveplots :
        plt.savefig('{0}_d2_activity.pdf'.format(batch_name))
    if showplots :
        plt.show()


def create_collections(ccf_files, fpccf=False, verbose=False) :
    """
        Description: function to detect and organize collection of files
                     based on: object, ccfmask, sanitize, and DRS version
        ccf_files: list, list of file paths
        return: dict, where:
            dict["modes"] : list of keys to identify collections
            dict["key"] : list of files for a given collection
        """
    file_modes = []
    valid_files = []
    
    for i in range(len(ccf_files)) :
        try :
            hdr = fits.getheader(ccf_files[i], 1)
        
            filename = hdr['FILENAME']
            if fpccf :
                object = hdr['OBJECT'].upper().replace(" ","") + "_Fiber" + hdr['FIBER']
                ccfmask = hdr['WFPMASK']
                drs_version = "none"
            else :
                object = hdr['OBJECT'].upper().replace(" ","")
                ccfmask = hdr['CCFMASK']
                drs_version = hdr['VERSION']

            search_key = "SANI"
            res = [key for key, val in hdr.items() if search_key in key]
            if len(res) :
                sanit = "sani"
            else :
                sanit = "tcorr"

            reduction_mode = "{0}__{1}__{2}__{3}".format(object, ccfmask, sanit, drs_version)
            
            file_modes.append(reduction_mode)
            abs_path = os.path.abspath(ccf_files[i])
            valid_files.append(abs_path)
        except :
            if verbose :
                print("WARNING: could not read file",ccf_files[i], ", skippping ... ")
            continue

    file_modes = np.array(file_modes)
    valid_files = np.array(valid_files)
    
    modes = Counter(file_modes)
    most_common_modes = modes.most_common()   # Returns all unique items and their counts
    
    collection = {}
    collection["modes"] = []
    for m in most_common_modes :
        # create mask to select files within the same mode
        mode_mask = file_modes == m[0]
        collection["modes"].append(m[0])
        collection[m[0]] = valid_files[mode_mask]

    return collection


def save_bisector_time_series_in_rdb_format(tbl, output, time_in_rjd=True, rv_in_mps=False) :
    
    rv, rverr = tbl['RV_BIS'], tbl['ERROR_RV']
    #bjd = tbl['BJD'] + (tbl['MJDEND'] - tbl['MJDATE']) / 2.
    bjd = tbl['BJD']
    
    bis_slope, bis_width = tbl['BIS_SLOPE'], tbl['BIS_WIDTH']
    
    # mean 'top' CCF between 55 and 80% of depth
    vt = tbl['Vt']
    # mean 'bottom' CCF between 20-40%
    vb = tbl['Vb']
    # Vt - Vb
    bis = tbl['BIS']

    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\tbis_slope\tbis_width\tv_top\tv_bottom\tbis\n")
    outfile.write("---\t----\t-----\t---------\t---------\t-----\t--------\t---\n")
    
    for i in range(len(bjd)) :
        
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:+.2f}\t{2:+.2f}\t{3:+.5f}\t{4:+.2f}\t{5:+.2f}\t{6:+.2f}\t{7:+.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i], bis_slope[i], 1000. * bis_width[i], 1000. * vt[i], 1000. * vb[i], 1000. * bis[i]))
        else :
            outfile.write("{0:.10f}\t{1:+.5f}\t{2:+.5f}\t{3:+.5f}\t{4:+.5f}\t{5:+.5f}\t{6:+.5f}\t{7:+.5f}\n".format(rjd, rv[i], rverr[i], bis_slope[i], bis_width[i], vt[i], vb[i], bis[i]))

    outfile.close()


def save_rv_time_series_in_rdb_format(tbl, output, time_in_rjd=True, rv_in_mps=False, rv_key='RV', rverr_key='ERROR_RV', correct_rv_drift=True, fpccf=False) :
    
    if correct_rv_drift and 'RV_DRIFT' in tbl.keys():
        rv = tbl[rv_key] - tbl['RV_DRIFT']
        if 'ERVDRIFT' in tbl.keys() :
            rverr = np.sqrt(tbl[rverr_key]*tbl[rverr_key] + tbl['ERVDRIFT']*tbl['ERVDRIFT'])
        else :
            rverr = tbl[rverr_key]

    else :
        rv = tbl[rv_key]
        rverr = tbl[rverr_key]
        
    if fpccf :
        bjd = tbl['MJDATE']
    else :
        #bjd = tbl['BJD'] + (tbl['MJDEND'] - tbl['MJDATE']) / 2.
        bjd = tbl['BJD']

    save_rv_time_series(output, bjd, rv, rverr, time_in_rjd=time_in_rjd, rv_in_mps=rv_in_mps)


def save_drift_time_series_in_rdb_format(tbl, output, time_in_rjd=True, rv_in_mps=False, rv_key='RV', rverr_key='ERROR_RV') :

    rv = tbl[rv_key]
    rverr = tbl[rverr_key]

    # MJD must be used because calibrations don't have BJD
    jd = tbl['MJDATE'] + 2400000.5

    # density filter
    densfilt = tbl['SBCDEN_P']

    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\tSBCDEN_P\tDATE-OBS\tWAVETIME\tFIBER\tEXPTYPE\tFILENAME\t\n")
    outfile.write("---\t----\t-----\t--------\t--------\t-------\t-------\t-------\n")
    
    for i in range(len(jd)) :
        if time_in_rjd :
            rjd = jd[i] - 2400000.
        else :
            rjd = jd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4}\t{5:.10f}\t{6}\t{7}\t{8}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i], densfilt[i], tbl['DATE-OBS'][i], tbl['WAVETIME'][i], tbl['FIBER'][i], tbl['EXPTYPE'][i], tbl['FILENAME'][i]))
        else :
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\t{3:.2f}\t{4}\t{5:.10f}\t{6}\t{7}\t{8}\n".format(rjd, rv[i], rverr[i], densfilt[i], tbl['DATE-OBS'][i], tbl['WAVETIME'][i], tbl['FIBER'][i], tbl['EXPTYPE'][i], tbl['FILENAME'][i]))

    outfile.close()


def save_rv_time_series(output, bjd, rv, rverr, time_in_rjd=True, rv_in_mps=False) :
    
    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\n")
    outfile.write("---\t----\t-----\n")
    
    for i in range(len(bjd)) :
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:.2f}\t{2:.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i]))
        else :
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\n".format(rjd, rv[i], rverr[i]))

    outfile.close()


def read_rv_time_series(rvfile) :
    """
        Description: function to read RV data from *.rdb file
        """
    rvdata = ascii.read(rvfile,data_start=2)
    bjd = np.array(rvdata['rjd']) + 2400000.
    rv, rverr = np.array(rvdata["vrad"]), np.array(rvdata["svrad"])
    
    return bjd, rv, rverr


def fitorbit(bjd, rvs, rverrs, guess=[], fixed_period=True, period=10., plot=True, verbose=True) :
    """
        Description: function to fit orbit to RV data
        """
    if len(guess) == 0 :
        if fixed_period :
            def orbit_model (t, a, b, t0):
                return a + b * np.sin(2*np.pi*(t-t0)/period)
            a0, b0, t0 = np.mean(rvs), np.abs(np.max(rvs) - np.min(rvs)), bjd[0]
            guess = [a0, b0, t0]
        else :
            def orbit_model (t, a, b, t0, per):
                return a + b * np.sin(2*np.pi*(t-t0)/per)
            a0, b0, t0, p0 = np.mean(rvs), np.abs(np.max(rvs) - np.min(rvs)), bjd[0], period
            guess = [a0, b0, t0, p0]

    pfit, pcov = curve_fit(orbit_model, bjd, rvs, p0=guess)
    efit = np.sqrt(np.diag(pcov))

    log_string = ''
    log_string += "Systemic velocity: {0:.3f}+-{1:.3f} km/s\n".format(pfit[0],efit[0])
    log_string += "K: {0:.3f}+-{1:.3f} km/s\n".format(pfit[1],efit[1])
    log_string += "T0: {0:.6f}+-{1:.6f} BJD\n".format(pfit[2],efit[2])
    if not fixed_period :
        log_string += "Period: {0:.3f}+-{1:.3f} days\n".format(pfit[3],efit[3])
        period = pfit[3]

    residuals = rvs - orbit_model(bjd, *pfit)
    rms_res = np.nanstd(residuals)
    mad_res = np.nanmedian(np.abs(residuals)) / 0.67449

    log_string += "RMS of residuals: {0:.1f} m/s\n".format(rms_res*1000)
    log_string += "MAD of residuals: {0:.1f} m/s".format(mad_res*1000)

    if verbose :
        print(log_string)

    model_bjds = np.linspace(bjd[0] - period/4, bjd[-1] + period/4, 1000)

    fit_model = orbit_model(model_bjds, *pfit)

    if plot :
        fig, ax = plt.subplots(nrows = 2, ncols = 1,sharex = True)
        #plt.text(np.mean(bjd)-(bjd[-1]-bjd[0])/2, np.mean(rvs), log_string)
        ax[0].errorbar(bjd, rvs, yerr=rverrs, linestyle="None", fmt='o',color = 'k')
        ax[0].plot(model_bjds, fit_model,'r:')
        ax[0].set(ylabel = 'Velocity [km/s]',title = object)
        ax[1].errorbar(bjd, residuals, yerr=rverrs, linestyle="None", fmt='o', capsize = 2, color = 'k')
        ax[1].plot(model_bjds, np.zeros(len(model_bjds)),'r:')
        ax[1].set(xlabel = 'BJD-2400000', ylabel = 'Residuals [km/s]')
        plt.tight_layout()
        #plt.savefig((options.input).replace('.rdb','.png'))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    loc = {}
    loc["systemic_rv"] = pfit[0]
    loc["systemic_rverr"] = efit[0]
    loc["K"] = pfit[1]
    loc["Kerr"] = efit[1]
    loc["t0"] = pfit[2]
    loc["t0err"] = efit[2]
    loc["period"] = period
    if not fixed_period :
        loc["perioderr"] = efit[2]
    loc["fit_model"] = orbit_model(bjd, *pfit)
    loc["residuals"] = residuals
    loc["rms_residuals"] = rms_res*1000
    loc["mad_residuals"] = mad_res*1000
    
    loc["bjd_long_model"] = model_bjds
    loc["rv_long_model"] = orbit_model(model_bjds, *pfit)
    
    return loc


def combine_rvs_per_epoch(bjd, rv, rverr, one_epoch_size=1.0, median=True, nsig=0., verbose=False) :
    
    """
        Description: this function combines all RV data points within the same epoch,
                     where the size of an epoch can be given as input parameter
    """
    t0 = np.min(bjd)
    tf = np.max(bjd)
    epochs = []
    time_tmp = t0 - 0.5 * one_epoch_size
    
    while time_tmp <= tf + 0.5 * one_epoch_size :
        epochs.append(time_tmp)
        time_tmp += one_epoch_size
    epochs = np.array(epochs)

    digitized = np.digitize(bjd, epochs)

    out_bjd, out_rv, out_rverr = [], [], []

    for i in range(len(epochs)):
        if len(bjd[digitized == i]) :
            out_bjd.append(bjd[digitized == i].mean())
            if median :
                if verbose:
                    print("Calculating the median of {0} RV measurements in epoch {1:.2f}+-{2:.2f}".format(len(rv[digitized == i]),epochs[i],one_epoch_size/2))

                median_rv = np.median(rv[digitized == i])
                rverror = np.median(np.abs(rv[digitized == i] - median_rv))  / 0.67499
                out_rv.append(median_rv)
                out_rverr.append(rverror)
            else :
                if verbose:
                    print("Calculating the weighted mean of {0} RV measurements in epoch {1:.2f}+-{2:.2f}".format(len(rv[digitized == i]),epochs[i],one_epoch_size/2))

                weights = 1.0 / (rverr[digitized == i] * rverr[digitized == i])
                weighted_mean = np.average(rv[digitized == i], weights=weights)
                rverror = np.sqrt(np.average((rv[digitized == i] - weighted_mean)**2, weights=weights))

                if nsig :
                    sigclip = np.where(np.logical_and(rv[digitized == i] > weighted_mean - nsig*rverror, rv[digitized == i] < weighted_mean + nsig*rverror))
                    
                    if len(rverr[digitized == i][sigclip]) :
                        weighted_mean = np.average(rv[digitized == i][sigclip], weights=weights[sigclip])
                        rverror = np.sqrt(np.average((rv[digitized == i][sigclip] - weighted_mean)**2, weights=weights[sigclip]))
                
                out_rv.append(weighted_mean)
                out_rverr.append(rverror)

    out_bjd = np.array(out_bjd)
    out_rv = np.array(out_rv)
    out_rverr = np.array(out_rverr)

    return out_bjd, out_rv, out_rverr
