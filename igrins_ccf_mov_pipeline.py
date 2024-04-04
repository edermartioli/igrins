# -*- coding: iso-8859-1 -*-
"""
    Created on Feb 24 2022
    
    Description: Reduce a time series of IGRINS spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_ccf_pipeline.py --input=*e.fits -pv
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_ccf_pipeline.py --input=SDCHK_20211004_053?e.fits --ccf_mask=/Volumes/Samsung_T5/Science/IGRINS/ccf_masks/montreal_masks/AUMic_neg_depth.mas --object_spectrum=/Volumes/Samsung_T5/SLS-DATA/AUMIC/Template_s1d_AUMIC_sc1d_w_file_AB.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob

import numpy as np
import matplotlib.pyplot as plt

import reduc_lib

igrins_dir = os.path.dirname(__file__)
telluric_mask_repository = os.path.join(igrins_dir,'ccf_masks/telluric/')
h2o_mask = os.path.join(telluric_mask_repository,'trans_h2o_abso_ccf.mas')
tel_mask = os.path.join(telluric_mask_repository,'trans_others_abso_ccf.mas')


def save_rv_time_series(output, bjd, rv, rverr, obj_rvs, obj_rverrs, tel_rvs, tel_rverrs, h2o_rvs, h2o_rverrs, airmass, snr, time_in_rjd=True, rv_in_mps=False) :
    
    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\tobjrv\tobjrverr\ttellrv\ttellrverr\th2orv\th2orverr\tairmass\tsnr\n")
    outfile.write("---\t----\t-----\n")
    
    for i in range(len(bjd)) :
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:.2f}\t{2:.2f}\t{2:.2f}\t{2:.2f}\t{2:.2f}\t{2:.2f}\t{2:.2f}\t{2:.2f}\t{2:.3f}\t{2:.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i], 1000. * obj_rvs[i], 1000. * obj_rverrs[i], 1000. * tel_rvs[i], 1000. * tel_rverrs[i], 1000. * h2o_rvs[i], 1000. * h2o_rverrs[i], airmass[i], snr[i]))
        else :
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\t{2:.5f}\t{2:.5f}\t{2:.5f}\t{2:.5f}\t{2:.5f}\t{2:.5f}\t{2:.3f}\t{2:.2f}\n".format(rjd, rv[i], rverr[i], obj_rvs[i], obj_rverrs[i], tel_rvs[i], tel_rverrs[i], h2o_rvs[i], h2o_rverrs[i], airmass[i], snr[i]))

    outfile.close()



parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *e.fits data pattern",type='string',default="*e.fits")
parser.add_option("-b", "--object", dest="object", help="Object ID",type='string',default="AU Mic")
parser.add_option("-o", "--output", dest="output", help="Output RV time series file name",type='string',default="")
parser.add_option("-m", "--ccf_mask", dest="ccf_mask", help="Input CCF mask",type='string',default="")
parser.add_option("-s", "--object_spectrum", dest="object_spectrum", help="Object spectrum",type='string',default="")
parser.add_option("-l", "--output_template", dest="output_template", help="Output template spectrum",type='string',default="")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Input source RV (km/s)",type='float',default=0.)
parser.add_option("-f", "--ccf_width", dest="ccf_width", help="CCF half width (km/s)",type='string',default="150")
parser.add_option("-w", "--window", dest="window", help="Window size in number of frames",type='int',default=10)
parser.add_option("-a", "--vel_sampling", dest="vel_sampling", help="Velocity sampling for the template spectrum (km/s)",type='float',default=1.8)
parser.add_option("-t", action="store_true", dest="telluric_ccf", help="Run telluric CCF", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h igrins_ccf_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral e.fits data pattern: ', options.input)
    print('Object ID: ', options.object)
    print('Output RV time series file name: ', options.output)
    if options.ccf_mask != "":
        print('Input CCF mask: ', options.ccf_mask)
    print('Object spectrum: ', options.object_spectrum)
    if options.output_template != "":
        print('Output template spectrum: ', options.output_template)
    if options.source_rv != 0 :
        print('Input source RV (km/s): ', options.source_rv)
    print('Initial CCF width (km/s): ', options.ccf_width)
    print('Window size in number of frames: ', options.window)
    print('Velocity sampling (km/s): ', options.vel_sampling)


# make list of tfits data files
if options.verbose:
    print("Creating list of e.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

window = options.window

bjds, airmass, snr = np.array([]),np.array([]),np.array([])
obj_rvs, tel_rvs, h2o_rvs = np.array([]), np.array([]), np.array([])
obj_rverrs, tel_rverrs, h2o_rverrs = np.array([]), np.array([]), np.array([])

for i in range(len(inputdata)) :

    min_i, max_i = i-window, i+window+1
    if min_i < 0 :
        min_i, max_i = 0, 2*window+1
    if max_i >= len(inputdata) :
        min_i, max_i = len(inputdata) - 2*window - 1, len(inputdata)
    if min_i < 0 : min_i=0
    
    subset = inputdata[min_i:max_i]

    if options.verbose :
        print ("Frame number {} -- running CCF pipeline for subset: {} - {}".format(i, inputdata[min_i], inputdata[max_i-1]))
    # Initialize drift containers with zeros
    drifts = reduc_lib.get_zero_drift_containers(subset)

    # Run reduction routines
    reduced = reduc_lib.reduce_timeseries_of_spectra(subset, options.ccf_mask, object_name=options.object, stellar_spectrum_file=options.object_spectrum, tel_mask=tel_mask, h2o_mask=h2o_mask, telluric_rv=options.telluric_ccf, verbose=options.verbose)

    # Run CCF routines
    ccfs = reduc_lib.run_igrins_ccf(reduced, options.ccf_mask, drifts, telluric_rv=options.telluric_ccf, normalize_ccfs=True, save_output=True, source_rv=options.source_rv, ccf_width=options.ccf_width, vel_sampling=options.vel_sampling, run_analysis=True, output_template=options.output_template, tel_mask=tel_mask, h2o_mask=h2o_mask, verbose=options.verbose, plot=options.plot)

    obj_ccf = ccfs['OBJ_CCF']['TABLE_CCF']
    tel_ccf = ccfs['TELL_CCF']['TABLE_CCF']
    h2o_ccf = ccfs['H2O_CCF']['TABLE_CCF']
        
    bjds = np.append(bjds,obj_ccf['BJD'][i-min_i])
    airmass = np.append(airmass,obj_ccf['AIRMASS'][i-min_i])
    snr = np.append(snr,obj_ccf['SNR'][i-min_i])

    obj_rvs = np.append(obj_rvs,obj_ccf['RV'][i-min_i])
    obj_rverrs = np.append(obj_rverrs,obj_ccf['ERROR_RV'][i-min_i])
    tel_rvs = np.append(tel_rvs,tel_ccf['RV'][i-min_i])
    tel_rverrs = np.append(tel_rverrs,tel_ccf['ERROR_RV'][i-min_i])
    h2o_rvs = np.append(h2o_rvs,h2o_ccf['RV'][i-min_i])
    h2o_rverrs = np.append(h2o_rverrs,h2o_ccf['ERROR_RV'][i-min_i])


mobj_rv = np.nanmedian(obj_rvs)
plt.plot(bjds,obj_rvs-mobj_rv,label="OBJ RV - {} km/s".format(mobj_rv))

mtel_rv = np.nanmedian(tel_rvs)
plt.plot(bjds,tel_rvs-mtel_rv,label="TELL RV - {} km/s".format(mtel_rv))

mh2o_rv = np.nanmedian(h2o_rvs)
plt.plot(bjds,h2o_rvs-mh2o_rv,label="H2O RV - {} km/s".format(mh2o_rv))

plt.xlabel(r"BJD")
plt.ylabel(r"Velocity [km/s]")
plt.legend()
plt.show()

plt.plot(bjds,(obj_rvs-(tel_rvs-mtel_rv))*1000,label="TELLURIC CORRECTED")
plt.plot(bjds,(obj_rvs-(h2o_rvs-mh2o_rv))*1000,label="H2O CORRECTED")

plt.xlabel(r"BJD")
plt.ylabel(r"Velocity [m/s]")
plt.legend()
plt.show()

output = options.object.replace(" ","") + "_ccf_mov_rv.rdb"
if options.output != "" :
    output = options.output
save_rv_time_series(output, bjds, (obj_rvs-(tel_rvs-mtel_rv)), obj_rverrs, obj_rvs, obj_rverrs, tel_rvs, tel_rverrs, h2o_rvs, h2o_rverrs, airmass, snr, time_in_rjd=True, rv_in_mps=False)
