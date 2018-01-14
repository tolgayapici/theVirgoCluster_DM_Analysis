
# coding: utf-8

# In[1]:


from threeML import *
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

from astropy.io import fits

import sys
sys.path.append("../py")
from DMModels import *

calc_TS = False


# In[ ]:


# GAO + Bullock
Jfactor_M87_lo  = 1.43e+18
Jfactor_M87_med = 3.40e+18
Jfactor_M87_hi  = 8.13e+18
Jfactor_M49_lo  = 4.58e+17
Jfactor_M49_med = 1.06e+18
Jfactor_M49_hi  = 2.85e+18

# GAO + Sanchez
Jfactor_M87_lo  = 1.32e+18
Jfactor_M87_med = 2.69e+18
Jfactor_M87_hi  = 5.27e+18
Jfactor_M49_lo  = 4.10e+17
Jfactor_M49_med = 7.66e+17
Jfactor_M49_hi  = 1.66e+18


# In[2]:


class Sources():

    def set_mass(self, mass):
        self.mass = mass

    def set_channel(self, channel):
        self.channel = channel

    def set_process(self, process):
        self.process = process
        
    def set_model(self, model_name):
        self.model_name = model_name
        
    def set_M87_factor(self, value):
        self.M87_Jfactor = value
        
    def set_M49_factor(self, value):
        self.M49_Jfactor = value
        
    def set_M87(self):
        # spectrum
        spec_M87 = DMAnnihilationFlux()
        # set source
        self.source_M87 = PointSource("M87", dec=12.39, ra=187.70, spectral_shape=spec_M87)
        # set the spectral information
        spec_M87.mass    = self.mass
        spec_M87.J       = np.power(10., self.M87_Jfactor)
        if self.process == 1:
            spec_M87.tau.bounds  = (1e20,1e30)
            spec_M87.tau         = 1e25
            spec_M87.sigmav.fix  = True
        else:
            spec_M87.sigmav.bounds  = (1e-24,1e-20)
            spec_M87.sigmav         = 1e-23
            spec_M87.tau.fix        = True
        spec_M87.process     = self.process
        spec_M87.channel     = self.channel
        spec_M87.J.fix       = True
        spec_M87.process.fix = True
        """
        # show the spectrum that will be used
        xx = np.logspace(8, np.log10(self.mass*1e6))
        flux = spec_M87.evaluate(x=xx, mass=self.mass, channel=self.channel, J=10**19.66, process=self.process, sigmav=1e-23, tau=1e26)
        plt.clf()
        plt.loglog(xx/1e9, flux, label="tolga", color='black')
        plt.show()
        """
        
    def set_M49(self):
        # spectrum
        spec_M49 = DMAnnihilationFlux()
        # set source
        self.source_M49 = PointSource("M49", ra=187.44, dec=8.00, spectral_shape=spec_M49)
        # set the spectral information
        spec_M49.mass    = self.mass
        spec_M49.J       = np.power(10., self.M49_Jfactor)
        if self.process == 1:
            spec_M49.tau.bounds  = (1e20,1e30)
            spec_M49.tau         = 1e25
            spec_M49.sigmav.fix  = True
        else:
            spec_M49.sigmav.bounds  = (1e-24,1e-20)
            spec_M49.sigmav         = 1e-23
            spec_M49.tau.fix        = True
        spec_M49.process     = self.process
        spec_M49.channel     = self.channel
        spec_M49.J.fix       = True
        spec_M49.process.fix = True
        """
        # show the spectrum that will be used
        xx = np.logspace(8, np.log10(self.mass*1e6))
        flux = spec_M49.evaluate(x=xx, mass=self.mass, channel=self.channel, J=10**19.66, process=self.process, sigmav=1e-23, tau=1e26)
        plt.clf()
        plt.loglog(xx/1e9, flux, label="tolga", color='black')
        plt.show()
        """
    
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
        self.ROI_radius = 14    #  np.max([self.ra_M87-self.ra_M49, self.dec_M87-self.dec_M49])+10.
        self.ROI_RA     = 187.0 #  np.mean([self.ra_M87, self.ra_M49])
        self.ROI_DEC    = 10.0  #  np.mean([self.dec_M87, self.dec_M49])

    def setup(self, data='VERITAS'):
        self.set_M87()
        self.set_M49()
        self.set_M87_pointsource(data)
        self.set_ROI()        


# In[3]:


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


# In[4]:


mass    = 10000.      # float(sys.argv[1])
channel = 1          # int(float(sys.argv[2]))
process  = 0         # int(float(sys.argv[3]))
add_point_source = 1 #sys.argv[5]
if add_point_source:
    experiment = 'VERITAS'
        
verbose = True
print("running for mass {} GeV".format(mass))
print("            channel {}".format(channel))

sources = Sources()
sources.set_mass(float(mass))
sources.set_channel(int(float(channel)))
sources.set_process(process)
sources.set_M87_factor(1e19)
sources.set_M49_factor(1e19)
sources.setup(data=experiment)

model = Model(sources.source_M87)
"""
if add_point_source == 0:
    model = Model(sources.source_M87, sources.source_M49)
else:
    model = Model(sources.source_M87, sources.source_M49,
                  sources.point_M87)
"""
"""
if sources.process == 1:
    model.link(model.M87.spectrum.main.DMAnnihilationFlux.tau, 
               model.M49.spectrum.main.DMAnnihilationFlux.tau,
               Identity())
else:
    model.link(model.M87.spectrum.main.DMAnnihilationFlux.sigmav, 
               model.M49.spectrum.main.DMAnnihilationFlux.sigmav,
               Identity())
"""

if verbose:
    model.display()

llh = HAWCLike("VirgoCluster", "../../data/maptree.root", "../../data/response.root")
llh.set_active_measurements(1, 9)
#llh.set_ROI(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius, True)
print("ROI is set to ({:.2f}, {:.2f}) with r={:.2f}".format(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius))
llh.set_model(model)
datalist = DataList(llh)

jl = JointLikelihood(model, datalist, verbose=True)
jl.set_minimizer("ROOT")

if calc_TS:
    jl.fit(quiet=False)
    print("TS_max: {}".format(llh.calc_TS()))
    print("Best fit sigmav: {}".format(model.M87.spectrum.main.DMAnnihilationFlux.tau))
    llh.write_model_map("../results/model_maps/{}GeV_{}.root".format(mass, channel))

# calculate the profile for 95% CL computation
search_size = 50
if process == 0:
    norms = np.linspace(-25, -20, search_size)
    #norms = np.linspace(10**-25, 10**-20, search_size)
else:
    norms = np.linspace(24, 30, search_size)
    #norms = np.linspace(10**24, 10**30, search_size)    
LLs  = np.zeros(search_size)
for i in range(search_size):
    LLs[i] = jl.minus_log_like_profile(norms[i])
delLLs = LLs - LLs.min()
imin   = np.argmin(delLLs)
plt.semilogx(10**norms, delLLs)
#plt.semilogx(norms, delLLs)
if process == 0:
    plt.xlim(1e-26, 1e-22)
else:
    plt.xlim(1e25, 1e30)
plt.ylim(0, 3)
if process == 0:
    plt.savefig("../results/LL_profiles/annihilation_{}GeV_{}_{}.eps".format(mass, channel, "example"))
    #interpolator = interp1d(delLLs[imin:],10**norms[imin:], kind='cubic', fill_value='extrapolate')
    #interpolator = interp1d(delLLs[imin:], norms[imin:], kind='cubic', fill_value='extrapolate')
else:
    plt.savefig("../results/LL_profiles/decay_{}GeV_{}_{}.eps".format(mass, channel, "example"))
    #interpolator = interp1d(delLLs[:imin],10**norms[:imin], kind='cubic', fill_value='extrapolate')
    #interpolator = interp1d(delLLs[:imin], norms[:imin], kind='cubic', fill_value='extrapolate')
norm_95cl = interpolator(2.71/2.)
print("95% CL norm: {}".format(norm_95cl))


# In[ ]:




