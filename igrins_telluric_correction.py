"""
    Created on Nov 23 2021
    
    Description: This routine performs telluric correction of IGRINS spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil
    Institut d'Astrophysique de Paris, France
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/IGRINS/igrins_telluric_correction.py --input=SDCHK_20211004_053?e.fits --object_spectrum=/Volumes/Samsung_T5/SLS-DATA/AUMIC/Template_s1d_AUMIC_sc1d_w_file_AB.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob
import numpy as np

import reduc_lib
import igrinslib
import telluric_lib

import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import constants


from scipy.interpolate import interp1d


igrins_dir = os.path.dirname(__file__)
telluric_grid_path = os.path.join(igrins_dir, 'TelluricGrids/')


#-- end of spirou_ccf routine
parser = OptionParser()
parser.add_option("-b", "--object", dest="object", help="Object ID",type='string',default="AU Mic")
parser.add_option("-i", "--input", dest="input", help="Input e.fits spectral data pattern",type='string',default="*e.fits")
parser.add_option("-s", "--object_spectrum", dest="object_spectrum", help="Object spectrum",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h igrins_telluric_correction.py")
    sys.exit(1)

if options.verbose:
    print('Object ID: ', options.object)
    print('Spectral e.fits data pattern: ', options.input)
    print('Object spectrum: ', options.object_spectrum)


wl1,wl2 = 1440,2500

# make list of tfits data files
if options.verbose:
    print("Creating list of e.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

# First load spectra into a container
spectra = reduc_lib.load_array_of_igrins_spectra(inputdata, object_name=options.object, verbose=options.verbose)["spectra"]

object_spectrum = igrinslib.load_spirou_s1d_template(options.object_spectrum, wl1=wl1, wl2=wl2, to_resolution=0, normalize=True, plot=False)

tell_spectrum = telluric_lib.get_telluric_model_from_grid(telluric_grid_path, airmass=1.5, pwv=100, wl1=wl1, wl2=wl2, to_resolution=300000)

#plt.plot(object_spectrum["wl"],object_spectrum["flux"], '-', color='darkgreen', lw=2)
#plt.fill_between(object_spectrum['wl'], object_spectrum['flux'] - object_spectrum['fluxerr'], object_spectrum['flux']+object_spectrum['fluxerr'], color="#2ca02c", alpha=0.8, edgecolor="none", zorder=2, label="Template spectrum")
i=0

speed_of_light_in_kps = constants.c / 1000.
tell_wl = tell_spectrum["wl"] * (1.0 + spectra[i]['BERV'] / speed_of_light_in_kps) 
#plt.plot(tell_wl, tell_spectrum["trans"], '-', color="#d62728", lw=2.0, alpha=0.8, label="Telluric spectrum")
    
nchunks = 4
    
for order in range(54) :
#for i in range(len(spectra)) :
    
    spectrum = spectra[i]
    keep = np.isfinite(spectrum['flux'][order])
    flux = spectrum['flux'][order][keep]
    fluxerr = spectrum['fluxerr'][order][keep]
    wl_sf = spectrum['wl_sf'][order][keep]
    wl = spectrum['wl'][order][keep]
    if len(wl_sf) > 300 :
        continuum = reduc_lib.fit_continuum(wl_sf, flux, function='polynomial', order=4, nit=10, rej_low=1.0, rej_high=4.0, grow=1, med_filt=1, percentile_low=0., percentile_high=100.,min_points=100, xlabel="wavelength", ylabel="flux", plot_fit=False, silent=True)
    else :
        continue
        
        
    if len(wl_sf) == 0 :
        continue

    dwl = (wl_sf[-1] - wl_sf[0]) / nchunks
    
    for j in range(nchunks) :
        print("order=",order, "chunk=",j)
        
        wl0 = wl_sf[0] + dwl*j
        wlf = wl_sf[0] + dwl*(j+1)
        if wlf > wl_sf[-1] :
            wlf = wl_sf[-1]

        chunk = (wl_sf > wl0) & (wl_sf < wlf)
        
        if len(wl_sf[chunk]) > 300 :

            #plt.errorbar(wl_sf[chunk], flux[chunk]/continuum[chunk], yerr=fluxerr[chunk]/continuum[chunk], fmt='.', color="grey", alpha=0.5)

            okeep = (object_spectrum['wl'] > wl0-1) & (object_spectrum['wl'] < wlf+1)
            
            if len(object_spectrum['wl'][okeep]) == 0 :
                continue
            
            if object_spectrum['wl'][okeep][0] > wl0 or object_spectrum['wl'][okeep][-1] < wlf :
                continue
                
            reduc_spectrum = reduc_lib.fit_template(wl_sf[chunk], flux[chunk]/continuum[chunk], fluxerr[chunk]/continuum[chunk], object_spectrum['wl'][okeep], object_spectrum['flux'][okeep], plot=True)
    
            #gp = reduc_lib.fit_template_gp(wl_sf, flux, fluxerr, object_spectrum['wl'][okeep], object_spectrum['flux'][okeep], plot=True, verbose=True)

#plt.xlabel("wavelength [nm]",fontsize=18)
#plt.ylabel("flux",fontsize=18)
#plt.legend(fontsize=18)
#plt.show()

