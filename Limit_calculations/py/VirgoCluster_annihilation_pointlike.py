from threeML import *
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

from astropy.io import fits

import sys
sys.path.append(".")
from DMModels import *

RA  = 187.70355641
DEC = 12.39063989

Jfactor_M87_med = 3.40e+18
Jfactor_M49_med = 1.06e+18

calc_TS = True#False

class Sources():

    def set_mass(self, mass):
        self.mass = mass

    def set_channel(self, channel):
        self.channel = channel

    def set_model(self, model_name):
        self.model_name = model_name
        
    def set_M87(self):
        print(DM_model)
        # extended template
        M87_fits_template = "../../J_factor_calculations/templates/M87_{}_Jfactor_template.fits".format(DM_model)
        self.ra_M87  = fits.open(M87_fits_template)[0].header['CRVAL1']
        self.dec_M87 = fits.open(M87_fits_template)[0].header['CRVAL2']
        # spectrum
        spec_M87               = DMAnnihilationFlux()
        spec_M87.mass          = self.mass
        spec_M87.J             = Jfactor_M87_med
        spec_M87.sigmav.bounds = (1e-26,1e-20)
        spec_M87.sigmav        = 1e-22
        spec_M87.channel       = self.channel
        spec_M87.J.fix         = True
        # set source
        self.source_M87 = PointSource("M87", dec=self.dec_M87, ra=self.ra_M87, spectral_shape=spec_M87)
    
    def set_M49(self):
        # extended template
        M49_fits_template = "../../J_factor_calculations/templates/M49_{}_Jfactor_template.fits".format(DM_model)
        self.ra_M49  = fits.open(M49_fits_template)[0].header['CRVAL1']
        self.dec_M49 = fits.open(M49_fits_template)[0].header['CRVAL2']
        # spectrum
        spec_M49 = DMAnnihilationFlux()
        spec_M49.mass          = self.mass
        spec_M49.J             = Jfactor_M49_med
        spec_M49.sigmav.bounds = (1e-26,1e-20)
        spec_M49.sigmav        = 1e-22
        spec_M49.channel       = self.channel
        spec_M49.J.fix         = True
        # set source
        self.source_M49 = PointSource("M49", dec=self.dec_M49, ra=self.ra_M49, spectral_shape=spec_M49)
    
    def set_M87_pointsource(self, data):
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
        # spectrum
        spec_M87_point = Powerlaw()
        spec_M87_point.index = index
        spec_M87_point.K = norm*1e9
        spec_M87_point.index.fix = True
        spec_M87_point.K.fix = True
        # set source
        self.point_M87  = PointSource("M87Point", ra=187.70, dec=12.39, spectral_shape=spec_M87_point)
    
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

mass    = float(sys.argv[1])
channel = int(float(sys.argv[2]))
DM_model = sys.argv[3]
add_point_source = sys.argv[4]
if add_point_source:
    try:
        experiment = sys.argv[5]
    except:
        experiment = 'VERITAS'
        
verbose = True
print("running for mass {} GeV".format(mass))
print("            channel {}".format(channel))

sources = Sources()
sources.set_mass(float(mass))
sources.set_channel(int(float(channel)))
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
llh.set_ROI(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius, True)
print("ROI is set to ({:.2f}, {:.2f}) with r={:.2f}".format(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius))
llh.set_model(model)
datalist = DataList(llh)

jl = JointLikelihood(model, datalist, verbose=True)
jl.set_minimizer("ROOT")

if calc_TS:
    jl.fit(quiet=False)
    print("TS_max: {}".format(llh.calc_TS()))
    print("Best fit sigmav: {}".format(model.M87.spectrum.main.DMAnnihilationFlux.sigmav))
    llh.write_model_map("../results/model_maps/point_source/{}GeV_{}.root".format(mass, channel))

# calculate the profile for 95% CL computation
search_size = 40
best_fit    = model.M87.spectrum.main.DMAnnihilationFlux.sigmav.value
norms = np.linspace(np.log10(best_fit), np.log10(best_fit)+2., search_size)
LLs  = np.zeros(search_size)
print("will start the LL calculations")

for i in range(search_size):
    LLs[i] = jl.minus_log_like_profile(norms[i])

delLLs = LLs - LLs.min()
imin   = np.argmin(delLLs)
plt.semilogx(10**norms, delLLs)
plt.xlim(np.log10(best_fit)-.1, np.log10(best_fit)+.6)
plt.ylim(0, 3)

plt.savefig("../results/LL_profiles/annihilation_{}GeV_{}_{}.eps".format(mass, channel, DM_model))
interpolator = interp1d(delLLs[imin:],10**norms[imin:], kind='cubic', fill_value='extrapolate')
norm_95cl = interpolator(2.71/2.)

print("95% CL norm: {}".format(norm_95cl))

