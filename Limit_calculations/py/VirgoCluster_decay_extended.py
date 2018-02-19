# take care of matplotlib to use correct
import matplotlib as mpl
mpl.use("agg")

# stop ROOT hijacking the command line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

import numpy as np
import pdb
import os
import shutil
import matplotlib.pyplot as plt

from astropy.io import fits

import argparse

import sys
sys.path.append(".")
from Sources import *
from LimitCalculator import *

calc_TS = True#False
lifetime_lo = -1e10
lifetime_hi = 1e40

class DecaySources(Sources):

    def __init__(self, model):
        self.ebl_model_name    = None
        self.DM_model          = model
        self.M87_fits_template_org = "../../D_factor_calculations/templates/M87_{}_Dfactor_template.fits".format(DM_model)
        self.M49_fits_template_org = "../../D_factor_calculations/templates/M49_{}_Dfactor_template.fits".format(DM_model)
        self.M87_fits_template = self.M87_fits_template_org
        self.M49_fits_template = self.M49_fits_template_org
        
    def set_M87(self):
        # set source
        spec_M87               = DMDecayFlux()
        # extended template
        shape_M87    = SpatialTemplate_2D()
        #M87_fits_template = "../../D_factor_calculations/templates/M87_{}_Dfactor_template.fits".format(DM_model)
        shape_M87.load_file(self.M87_fits_template)
        f_M87 = open(self.M87_fits_template.replace(".fits", ".txt"), "r")
        line  = f_M87.readline()
        Dmax_M87 = float(line.split("=")[1])
        self.ra_M87  = fits.open(self.M87_fits_template)[0].header['CRVAL1']
        self.dec_M87 = fits.open(self.M87_fits_template)[0].header['CRVAL2']
        # spectrum
        self.source_M87 = ExtendedSource("M87",spatial_shape=shape_M87,spectral_shape=spec_M87)
        spec_M87.mass          = self.mass
        spec_M87.D             = np.power(10.,Dmax_M87)
        spec_M87.tau.bounds = (lifetime_lo, lifetime_hi)#(1e-24,1e-20)
        spec_M87.tau        = 1e26
        spec_M87.channel       = self.channel
        spec_M87.D.fix         = True
        if self.ebl_model_name is not None:
            spec_M87.set_EBL_model(self.ebl_model_name, 0.004)
        
    def set_M49(self):
        # set source
        spec_M49 = DMDecayFlux()
        # extended template
        shape_M49    = SpatialTemplate_2D()
        #M49_fits_template = "../../D_factor_calculations/templates/M49_{}_Jfactor_template.fits".format(DM_model)
        shape_M49.load_file(self.M49_fits_template)
        f_M49 = open(self.M49_fits_template.replace(".fits", ".txt"), "r")
        line  = f_M49.readline()
        Dmax_M49 = float(line.split("=")[1])
        self.ra_M49  = fits.open(self.M49_fits_template)[0].header['CRVAL1']
        self.dec_M49 = fits.open(self.M49_fits_template)[0].header['CRVAL2']
        # spectrum
        self.source_M49 = ExtendedSource("M49",spatial_shape=shape_M49,spectral_shape=spec_M49)
        spec_M49.mass          = self.mass
        spec_M49.D             = np.power(10.,Dmax_M49)
        spec_M49.tau.bounds = (lifetime_lo, lifetime_hi)#(1e-24,1e-20)
        spec_M49.tau        = 1e26
        spec_M49.channel       = self.channel
        spec_M49.D.fix         = True
        if self.ebl_model_name is not None:
            spec_M49.set_EBL_model(self.ebl_model_name, 0.004)
            
class DecayLimitCalculator(LimitCalculator):

    def find_max_TS(self):
        val = np.logspace(np.log10(self.min), np.log10(self.max), 50)
        TS  = []
        last_TS  = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.DMDecayFlux.tau.value = curr_val
            current_TS = (self.llh.calc_TS())
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
                print(curr_val, current_TS)
                break
            last_TS = current_TS
        TS = np.array(TS)
        max_index = np.argmax(TS)
        if self.verbose:
            print("max at: {} max TS: {}".format(val[max_index], TS[max_index]))
        print("second iteration")

        print(val[max_index-1], val[max_index+1])
        val = np.linspace(val[max_index-1], val[max_index+1], 50)
        TS  = []
        last_TS  = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.DMDecayFlux.tau.value = curr_val
            current_TS = (self.llh.calc_TS())
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
                print(curr_val, current_TS)
                break
            last_TS = current_TS
        TS = np.array(TS)
        max_index = np.argmax(TS)
        if self.verbose:
            print("max at: {} max TS: {}".format(val[max_index], TS[max_index]))

        print("third iteration")
        val = np.linspace(val[max_index-1], val[max_index+1], 100)
        TS  = []
        last_TS = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.DMDecayFlux.tau.value = curr_val
            current_TS = (self.llh.calc_TS())
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
                print(curr_val, current_TS)
                break
            last_TS = current_TS
        TS = np.array(TS)
        max_index = np.argmax(TS)
        if self.verbose:
            print("max at: {} max TS: {}".format(val[max_index], TS[max_index]))
        return val[max_index], TS[max_index]
    
    def calculate_limit(self, rel_err=1.e-2):
        best_fit, TS_max = self.find_max_TS()

        if best_fit < 0:
            print("the best fit is negative. taking care of it now")
            best_fit = 0
            self.model.M49.spectrum.main.DMAnnihilationFlux.tau.value = best_fit
            TS_max = self.llh.calc_TS()

        hi = best_fit
        hi_TS = TS_max
        del_hi_TS = 2.71 - (TS_max-hi_TS)
        lo = hi/50.
        if lo == 0:
            lo = 1e22
        self.model.M49.spectrum.main.DMDecayFlux.tau.value = hi
        hi_TS = self.llh.calc_TS()
        del_hi_TS = 2.71 - (TS_max-hi_TS)

        while True:
            mid = (lo+hi)/2.
            self.model.M49.spectrum.main.DMDecayFlux.tau.value = mid
            mid_TS = self.llh.calc_TS()
            del_mid_TS = 2.71 - (TS_max-mid_TS)
            if np.fabs(del_mid_TS) < rel_err:
                norm_95cl = mid
                TS_95cl   = mid_TS
                print("difference: {}".format(TS_95cl))
                print("limit:      {}".format(norm_95cl))
                break
            else:
                if del_mid_TS*del_hi_TS > 0:
                    hi = mid
                else:
                    lo = mid
                print("current value: {}".format(mid))
                print("current TS:    {}".format(mid_TS))
                print("current diff:  {}".format(del_mid_TS))

parser = argparse.ArgumentParser(description="This script is to run extended source DM search for the Virgo Cluster")
parser.add_argument("-m",    dest="mass",    help="Mass of the DM particle (in GeV)",   required=True, type=float)
parser.add_argument("-c",    dest="channel", help="Decay channel",               required=True, choices=[1, 2, 3, 4, 5], type=int)
parser.add_argument("-t",    dest="model",   help="DM Template",                        required=True, choices=["min", "med"])
parser.add_argument("-a",    dest="add",     help="A flag to add the M87 point source", default=0, type=int)
parser.add_argument("-e",    dest="exp",     help="Experiment name of the add flag is True", choices=["VERITAS", "MAGIC", "HAWC"])
parser.add_argument("-v",    dest="verbose", help="Verbosity of the script",            default=True)
parser.add_argument("-sra",  dest="shiftra", help="shift in ra for the expected limit calculations",  default=0., type=float)
parser.add_argument("-sdec", dest="shiftdec", help="shift in dec for the expected limit calculations", default=0., type=float)
parser.add_argument("--ebl", dest="ebl",  help="EBL model to be used in the flux calculations", choices=['gilmore', 'dominguez', 'finke'], default=None)
parser.add_argument("--maptree",  dest="maptree", default="../../data/maptree.root")
parser.add_argument("--response", dest="response", default="../../data/response.root")
parsed_args = parser.parse_args()

mass             = parsed_args.mass
channel          = parsed_args.channel
DM_model         = parsed_args.model
add_point_source = parsed_args.add
experiment       = parsed_args.exp
verbose          = parsed_args.verbose
ra_shift         = parsed_args.shiftra
dec_shift        = parsed_args.shiftdec
ebl_model        = parsed_args.ebl
maptree          = parsed_args.maptree

print("running for mass {} GeV".format(mass))
print("            channel {}".format(channel))

sources = DecaySources(DM_model)
sources.set_mass(float(mass))
sources.set_channel(int(float(channel)))
sources.set_EBL_model(ebl_model_name=ebl_model)
#sources.shift_templates(ra_shift, dec_shift)
sources.setup(data=experiment)

if add_point_source == 0:
    model = Model(sources.source_M87, sources.source_M49)
else:
    model = Model(sources.source_M87, sources.source_M49,
                  sources.point_M87)

model.link(model.M87.spectrum.main.DMDecayFlux.tau,
           model.M49.spectrum.main.DMDecayFlux.tau,
           Identity())

if verbose:
    model.display()

lc = DecayLimitCalculator(maptree, "../../data/response.root", model, verbose=verbose)
lc.set_ROI(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius)
lc.set_range(1e24, 1e35)
lc.calculate_limit()
