"""
    Created on Nov 16 2021
    
    Description: This routine reduces IGRINS spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil
    Institut d'Astrophysique de Paris, France
    
    Simple usage example:
    
    python /Users/eder/Science/IGRINS/igrins_template.py --input=SDCH*.spec.fits

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob
import numpy as np

#import reduc_lib
import igrinslib

import matplotlib.pyplot as plt

igrins_dir = os.path.dirname(__file__)

#-- end of spirou_ccf routine
parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral FITS data pattern",type='string',default="*t.fits")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h igrins_template.py")
    sys.exit(1)

if options.verbose:
    print('Spectral t.fits data pattern: ', options.input)
    
# make list of tfits data files
if options.verbose:
    print("Creating list of t.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

spectra = igrinslib.load_igrins_spectra(inputdata)


