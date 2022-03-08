# Description: Script to make a log of IGRINS observations
# Author: Eder Martioli
# Laboratorio Nacional de Astrofisica, Brazil
# Nov 2021
#

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os, sys
import glob
import astropy.io.fits as fits
import numpy as np

parser = OptionParser()
parser.add_option("-d", "--datadir", dest="datadir", help="data directory",type='string',default="./")
parser.add_option("-p", "--pattern", dest="pattern", help="data pattern",type='string',default="*.fits")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with log_igrins.py -h ")
    sys.exit(1)

if options.verbose:
    print('Data directory: ', options.datadir)
    print('Data pattern: ', options.pattern)

currdir = os.getcwd()
os.chdir(options.datadir)

inputdata = sorted(glob.glob(options.pattern))

print("INSTRUME","OBJECT","OBSCLASS","BAND","OBJTYPE","FRMTYPE","EXPTIME","DATE-OBS","AMSTART","PASTART","HUMIDITY","SLIT_ANG")

for filename in inputdata :
    hdu = fits.open(filename)
    hdr = hdu[0].header
    print(filename,hdr["INSTRUME"],hdr["OBJECT"].replace(" ",""),hdr["OBSCLASS"],hdr["BAND"],hdr["OBJTYPE"],hdr["FRMTYPE"],hdr["EXPTIME"],hdr["DATE-OBS"],hdr["AMSTART"],hdr["PASTART"],hdr["HUMIDITY"],hdr["SLIT_ANG"])

os.chdir(currdir)
