import matplotlib as mpl
#mpl.use("agg")

import matplotlib.pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

from hawc_hal import HAL, HealpixConeROI
from scipy.interpolate import interp1d
import progressbar

class LimitCalculator:

    def __init__(self, maptree, response, models, verbose=False):
        self.maptree  = maptree
        self.response = response
        self.verbose  = verbose
        self.models   = models
        self.fluxUnit = (1. / (u.TeV * u.cm**2 * u.s))
        # - construct HAL
        self.roi      = HealpixConeROI(data_radius=13., model_radius=25., ra=187.5745, dec=10.1974)
        self.hawc     = HAL("VirgoCluster", maptree, response, self.roi)
        self.hawc.set_active_measurements(1, 9)
        self.hawc.display()
        self.datalist = DataList(self.hawc)
        # - construct joint likelihood
        self.jl_M87       = JointLikelihood(self.models[0], self.datalist, verbose=True)
        self.jl_M49       = JointLikelihood(self.models[1], self.datalist, verbose=True)
        self.jl_linked    = JointLikelihood(self.models[2], self.datalist, verbose=False)
        self.jl_notlinked = JointLikelihood(self.models[3], self.datalist, verbose=True)
        # - set the minimizer
        self.jl_M87.set_minimizer("minuit")
        self.jl_M49.set_minimizer("minuit")
        self.jl_linked.set_minimizer("minuit")
        self.jl_notlinked.set_minimizer("minuit")
        
    def set_range(self, minimum, maximum):
        self.min = minimum
        self.max = maximum

    def get_M87_likelihood(self):
        return self.jl_M87

    def get_M49_likelihood(self):
        return self.jl_M49

    def get_linked_likelihood(self):
        return self.jl_linked
        
    def calculate_limit(self, ll):
        num_steps = 100
        bar = progressbar.ProgressBar(maxval=num_steps,
                                      widgets=[ ' [', progressbar.Timer(), '] ',
                                                progressbar.Bar(),
                                                ' (', progressbar.ETA(), ') ',])
        TS  = []
        result = ll.fit()
        bar.start()
        bar.update(0)
        best_fit = ll.likelihood_model.M87.spectrum.main.RangedPowerlaw.K.value
        vals = np.logspace(np.log10(ll.likelihood_model.M87.spectrum.main.RangedPowerlaw.K.value)-.25,
                           np.log10(ll.likelihood_model.M87.spectrum.main.RangedPowerlaw.K.value)+3.0,
                           num_steps)
        for ival, val in enumerate(vals):
            ll.likelihood_model.M87.spectrum.main.RangedPowerlaw.K.value = val
            curr_LL = ll.data_list.values()[0].inner_fit()
            #print(val, curr_LL)
            TS.append(curr_LL)
            bar.update(ival)

        TS = np.array(TS)
        TS = -TS
        TS -= np.min(TS)
        plt.semilogx(vals, TS)
        plt.savefig("example_step1.pdf")
        plt.show()

        interpolator = interp1d(TS, vals)
        return (ll.likelihood_model.M87.spectrum.main.RangedPowerlaw.K, interpolator(1.35))
