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

import reduc_lib

igrins_dir = os.path.dirname(__file__)


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

reduced = reduc_lib.reduce_timeseries_of_spectra(inputdata, options.ccf_mask, object_name=options.object, stellar_spectrum_file=options.object_spectrum, verbose=options.verbose)

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

### science topics:
###         - RM effect -- detect the planets
###         - line variability
###         - spot crossing, rotation, flares
###         - exoatmos -- try to detect: H2O, CO2
