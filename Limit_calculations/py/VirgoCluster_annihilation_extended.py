# set matplotlib
import matplotlib as mpl
mpl.use("agg")

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
from DMModels import *

calc_TS = True#False
crosssection_lo = 1e-30
crosssection_hi = 1e-20

class Sources():

    def __init__(self, model):
        self.DM_model          = model
        self.M87_fits_template_org = "../../J_factor_calculations/templates/M87_{}_Jfactor_template.fits".format(DM_model)
        self.M49_fits_template_org = "../../J_factor_calculations/templates/M49_{}_Jfactor_template.fits".format(DM_model)
        self.M87_fits_template = self.M87_fits_template_org
        self.M49_fits_template = self.M49_fits_template_org
        
    def shift_templates(self, ra, dec):
        hdus = fits.open(self.M87_fits_template)
        hdus[0].header['CRVAL1'] += ra
        hdus[0].header['CRVAL2'] += dec
        if os.path.isfile(self.M87_fits_template_org.replace(".fits", "_shifted.fits")):
            print("removing existing shifted template")
            os.remove(self.M87_fits_template_org.replace(".fits", "_shifted.fits"))
        hdus.writeto(self.M87_fits_template_org.replace(".fits", "_shifted.fits"))
        shutil.copy(self.M87_fits_template_org.replace(".fits", ".txt"), self.M87_fits_template.replace(".fits", "_shifted.txt")) 
        self.M87_fits_template = self.M87_fits_template_org.replace(".fits", "_shifted.fits")
        hdus = fits.open(self.M49_fits_template)
        hdus[0].header['CRVAL1'] += ra
        hdus[0].header['CRVAL2'] += dec
        if os.path.isfile(self.M49_fits_template_org.replace(".fits", "_shifted.fits")):
            print("removing existing shifted template")
            os.remove(self.M49_fits_template_org.replace(".fits", "_shifted.fits"))
        shutil.copy(self.M49_fits_template_org.replace(".fits", ".txt"), self.M49_fits_template_org.replace(".fits", "_shifted.txt")) 
        hdus.writeto(self.M49_fits_template_org.replace(".fits", "_shifted.fits"))
        self.M49_fits_template = self.M49_fits_template_org.replace(".fits", "_shifted.fits")
        
    def set_mass(self, mass):
        self.mass = mass

    def set_channel(self, channel):
        self.channel = channel

    def set_model(self, model_name):
        self.model_name = model_name

    def set_M87(self):
        # set source
        spec_M87               = DMAnnihilationFlux()
        # extended template
        shape_M87    = SpatialTemplate_2D()
        #M87_fits_template = "../../J_factor_calculations/templates/M87_{}_Jfactor_template.fits".format(DM_model)
        shape_M87.load_file(self.M87_fits_template)
        f_M87 = open(self.M87_fits_template.replace(".fits", ".txt"), "r")
        line  = f_M87.readline()
        Jmax_M87 = float(line.split("=")[1])
        self.ra_M87  = fits.open(self.M87_fits_template)[0].header['CRVAL1']
        self.dec_M87 = fits.open(self.M87_fits_template)[0].header['CRVAL2']
        # spectrum
        self.source_M87 = ExtendedSource("M87",spatial_shape=shape_M87,spectral_shape=spec_M87)
        spec_M87.mass          = self.mass
        spec_M87.J             = np.power(10.,Jmax_M87)
        spec_M87.sigmav.bounds = (crosssection_lo, crosssection_hi)#(1e-24,1e-20)
        spec_M87.sigmav        = 1e-23
        spec_M87.channel       = self.channel
        spec_M87.J.fix         = True

    def set_M49(self):
        # set source
        spec_M49 = DMAnnihilationFlux()
        # extended template
        shape_M49    = SpatialTemplate_2D()
        #M49_fits_template = "../../J_factor_calculations/templates/M49_{}_Jfactor_template.fits".format(DM_model)
        shape_M49.load_file(self.M49_fits_template)
        f_M49 = open(self.M49_fits_template.replace(".fits", ".txt"), "r")
        line  = f_M49.readline()
        Jmax_M49 = float(line.split("=")[1])
        self.ra_M49  = fits.open(self.M49_fits_template)[0].header['CRVAL1']
        self.dec_M49 = fits.open(self.M49_fits_template)[0].header['CRVAL2']
        # spectrum
        self.source_M49 = ExtendedSource("M49",spatial_shape=shape_M49,spectral_shape=spec_M49)
        spec_M49.mass          = self.mass
        spec_M49.J             = np.power(10.,Jmax_M49)
        spec_M49.sigmav.bounds = (crosssection_lo, crosssection_hi)#(1e-24,1e-20)
        spec_M49.sigmav        = 1e-23
        spec_M49.channel       = self.channel
        spec_M49.J.fix         = True

    def set_M87_pointsource(self, data):
        if data == "HAWC":
            spec_M87_point          = Cutoff_powerlaw()
            self.point_M87          = PointSource("M87Point", ra=187.70, dec=12.39, spectral_shape=spec_M87_point)
            spec_M87_point.K            = 1e-21
            spec_M87_point.K.bounds     = (1e-26, 1e-19)
            spec_M87_point.piv          = 1.0 * u.TeV
            spec_M87_point.piv.fix      = True
            spec_M87_point.index        = -3.0
            spec_M87_point.index.bounds = (-5.0, -1.0)
            spec_M87_point.xc           = 7.0 * u.TeV
            spec_M87_point.xc.fix       = True
        else:
            spec_M87_point = Powerlaw()
            self.point_M87  = PointSource("M87Point", ra=187.70, dec=12.39, spectral_shape=spec_M87_point)
            if data=='HESS-lower':
                index = -2.62
                norm  = 2.43e-13
            elif data=='HESS-upper':
                index = -2.22
                norm  = 11.7e-13
            elif data=='VERITAS':
                index = -2.5
                norm  = 5.6e-13
            elif data=='VERITAS-upper':
                index = -2.4
                norm  = 15.9e-13
            else:
                print("problem with the spectral data")
            spec_M87_point.index = index
            spec_M87_point.K = norm*1e9
            spec_M87_point.index.fix = True
            spec_M87_point.K.fix = True

    def set_ROI(self):
        self.ROI_radius = np.max([self.ra_M87-self.ra_M49, self.dec_M87-self.dec_M49])+10.
        self.ROI_RA     = np.mean([self.ra_M87, self.ra_M49])
        self.ROI_DEC    = np.mean([self.dec_M87, self.dec_M49])

    def setup(self, data='VERITAS'):
        self.set_M87()
        self.set_M49()
        self.set_M87_pointsource(data)
        self.set_ROI()

class Identity(Function1D):
    r"""
    description :
        Identity function

    latex : $ x $

    parameters :

       scale :
            desc : scale
            initial value : 1
            fix : yes
    tests :
        - { x : 10., function value: 10., tolerance: 1e-20}
        - { x : 100., function value: 100., tolerance: 1e-20}

    """
    __metaclass__ = FunctionMeta

    def _set_units(self, x_unit, y_unit):
        self.scale.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, scale):
        return scale * x


parser = argparse.ArgumentParser(description="This script is to run extended source DM search for the Virgo Cluster")
parser.add_argument("-m",    dest="mass",    help="Mass of the DM particle (in GeV)",   required=True, type=float)
parser.add_argument("-c",    dest="channel", help="Annihilation channel",               required=True, choices=[1, 2, 3, 4, 5], type=int)
parser.add_argument("-t",    dest="model",   help="DM Template",                        required=True, choices=["GAO", "B01"])
parser.add_argument("-a",    dest="add",     help="A flag to add the M87 point source", default=0, type=int)
parser.add_argument("-e",    dest="exp",     help="Experiment name of the add flag is True", choices=["VERITAS", "MAGIC", "HAWC"])
parser.add_argument("-v",    dest="verbose", help="Verbosity of the script",            default=True)
parser.add_argument("-sra",  dest="shiftra", help="shift in ra for the expected limit calculations",  default=0., type=float)
parser.add_argument("-sdec", dest="shiftdec", help="shift in dec for the expected limit calculations", default=0., type=float)
parsed_args = parser.parse_args()

mass             = parsed_args.mass
channel          = parsed_args.channel
DM_model         = parsed_args.model
add_point_source = parsed_args.add
experiment       = parsed_args.exp
verbose          = parsed_args.verbose
ra_shift         = parsed_args.shiftra
dec_shift        = parsed_args.shiftdec

print("running for mass {} GeV".format(mass))
print("            channel {}".format(channel))

sources = Sources(DM_model)
sources.set_mass(float(mass))
sources.set_channel(int(float(channel)))
sources.shift_templates(ra_shift, dec_shift)
sources.setup(data=experiment)

if add_point_source == 0:
    model = Model(sources.source_M87, sources.source_M49)
else:
    model = Model(sources.source_M87, sources.source_M49,
                  sources.point_M87)

model.link(model.M87.spectrum.main.DMAnnihilationFlux.sigmav,
           model.M49.spectrum.main.DMAnnihilationFlux.sigmav,
           Identity())

if verbose:
    model.display()

llh = HAWCLike("VirgoCluster", "../../data/maptree.root", "../../data/response.root")
llh.set_active_measurements(1, 9)

#llh.set_ROI(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius, True)
llh.set_template_ROI("../../data/maskedROI_nSide1024.fits.gz", 0.5, True)
print("ROI is set to ({:.2f}, {:.2f}) with r={:.2f}".format(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius))
llh.set_model(model)
datalist = DataList(llh)

jl = JointLikelihood(model, datalist, verbose=True)
jl.set_minimizer("ROOT")

if calc_TS:
    jl.fit(quiet=True)
    best_fit_TS = llh.calc_TS()
    jl.fit(quiet=True)
    TS_max = llh.calc_TS()
    print("TS_max: {}".format(TS_max))
    print("Best fit sigmav: {}".format(model.M87.spectrum.main.DMAnnihilationFlux.sigmav))
    llh.write_model_map("../results/model_maps/{}GeV_{}.root".format(mass, channel))
    
# calculate the profile for 95% CL computation
search_size = 60
best_fit    = model.M87.spectrum.main.DMAnnihilationFlux.sigmav.value
norms = np.linspace(np.log10(best_fit)-.1, np.log10(best_fit)+1., search_size)
LLs  = np.zeros(search_size)
TSs  = np.zeros(search_size)
print("will start the LL calculations")

# this part is being re-written
lo = np.log10(best_fit)
lo_TS = TS_max
del_lo_TS = 2.71 - (TS_max-lo_TS)
hi = lo + 3.0
model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = 10**hi
hi_TS = llh.calc_TS()
del_hi_TS = 2.71 - (TS_max-hi_TS)
tol = 1e-3
while True:
    mid = (lo+hi)/2.
    model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = 10**mid
    mid_TS = llh.calc_TS()
    del_mid_TS = 2.71 - (TS_max-mid_TS)
    if np.fabs(del_mid_TS) < tol:
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
    
#for i in range(search_size):
#    #model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = 10**norms[i]
#    #TSs[i] = llh.calc_TS()
#    #print(norms[i], TSs[i])
#    LLs[i] = jl.minus_log_like_profile(norms[i])
#    
#    print(llh.calc_TS())

#plt.semilogx(10**norms, TSs-best_fit_TS)
#plt.show()

#delLLs = LLs - LLs.min()
#imin   = np.argmin(delLLs)
#plt.semilogx(10**norms, delLLs)
#plt.xlim(np.log10(best_fit)-.1, np.log10(best_fit)+.6)
#plt.ylim(0, 3)
#plt.savefig("../results/LL_profiles/annihilation_{}GeV_{}_{}.eps".format(mass, channel, DM_model))
#interpolator = interp1d(delLLs[imin:],10**norms[imin:], kind='cubic', fill_value='extrapolate')
#norm_95cl = interpolator(2.71/2.)

print("95% CL norm: {}".format(10**norm_95cl))

