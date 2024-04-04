# -*- coding: iso-8859-1 -*-
"""
    Created on Feb 24 2022
    
    Description: Reduce a time series of IGRINS spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_ccf_pipeline.py --input=*e.fits -pv
    
    # Transit of AUMic b
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_ccf_pipeline.py --input=SDCHK_20210803_*e.fits --ccf_mask=/Volumes/Samsung_T5/Science/IGRINS/ccf_masks/montreal_masks/AUMic_neg_depth.mas --object_spectrum=/Volumes/Samsung_T5/IGRINS-DATA/Template_s1d_AUMIC_sc1d_w_file_AB.fits --output_template=AUMic_b_template.fits -pvt

    $ Transit of AUMic c
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_ccf_pipeline.py --input=SDCHK_20211004_0_*e.fits --ccf_mask=/Volumes/Samsung_T5/Science/IGRINS/ccf_masks/montreal_masks/AUMic_neg_depth.mas --object_spectrum=/Volumes/Samsung_T5/IGRINS-DATA/Template_s1d_AUMIC_sc1d_w_file_AB.fits --output_template=AUMic_c_template.fits -pvt
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob

import reduc_lib

igrins_dir = os.path.dirname(__file__)
telluric_mask_repository = os.path.join(igrins_dir,'ccf_masks/telluric/')
h2o_mask = os.path.join(telluric_mask_repository,'trans_h2o_abso_ccf.mas')
tel_mask = os.path.join(telluric_mask_repository,'trans_others_abso_ccf.mas')

NORDERS = 54

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *e.fits data pattern",type='string',default="*e.fits")
parser.add_option("-b", "--object", dest="object", help="Object ID",type='string',default="AU Mic")
parser.add_option("-m", "--ccf_mask", dest="ccf_mask", help="Input CCF mask",type='string',default="")
parser.add_option("-s", "--object_spectrum", dest="object_spectrum", help="Object spectrum",type='string',default="")
parser.add_option("-o", "--output_template", dest="output_template", help="Output template spectrum",type='string',default="")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Input source RV (km/s)",type='float',default=0.)
parser.add_option("-w", "--ccf_width", dest="ccf_width", help="CCF half width (km/s)",type='string',default="150")
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
    if options.ccf_mask != "":
        print('Input CCF mask: ', options.ccf_mask)
    print('Object spectrum: ', options.object_spectrum)
    if options.output_template != "":
        print('Output template spectrum: ', options.output_template)
    if options.source_rv != 0 :
        print('Input source RV (km/s): ', options.source_rv)
    print('Initial CCF width (km/s): ', options.ccf_width)
    print('Velocity sampling (km/s): ', options.vel_sampling)


# make list of tfits data files
if options.verbose:
    print("Creating list of e.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

# Initialize drift containers with zeros
drifts = reduc_lib.get_zero_drift_containers(inputdata)

# Run reduction routines
reduced = reduc_lib.reduce_timeseries_of_spectra(inputdata, options.ccf_mask, object_name=options.object, stellar_spectrum_file=options.object_spectrum, tel_mask=tel_mask, h2o_mask=h2o_mask, max_gap_size=3.0, convolve_spectra=True, to_resolution=30000, telluric_rv=options.telluric_ccf, verbose=options.verbose)

# Run CCF routines
reduc_lib.run_igrins_ccf(reduced, options.ccf_mask, drifts, telluric_rv=options.telluric_ccf, normalize_ccfs=True, save_output=True, source_rv=options.source_rv, ccf_width=options.ccf_width, vel_sampling=options.vel_sampling, run_analysis=True, output_template=options.output_template, tel_mask=tel_mask, h2o_mask=h2o_mask, verbose=options.verbose, plot=options.plot)


# To implement:
#       0. Set star template and fit continuum (OK)
#       1. get original spectra in earth reference frame
#       2. remove object flux and continuum
#       3. stack spectra and calculate telluric template (update telluric template)
#       4. get original spectra in star reference frame
#       5. remove telluric and continuum
#       7. stack spectra and calculate star template (update star template)
#       8. get original spectra in earth reference frame
#       9. remove telluric and object flux
#      10. stack spectra and fit continuum (update continuum)
#      11. remove continuum, calculate variance and compare with previous iteration
#      12. repeat step 1 through 11 until variance does not change significantly

#### Next:
####    A) calculate CCFs for tellurics and for object spectra
####    B) save telluric subtracted spectra *t.fits style
####    C) do science!

### science goals:
###         - measure RM effect -- detect the planets
###         - detect line variability
###         - measure spot crossing, rotation, flares
###         - detect exo-atmosphere --  H2O, CO2


