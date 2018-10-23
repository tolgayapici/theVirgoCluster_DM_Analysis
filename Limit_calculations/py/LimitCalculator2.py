import matplotlib as mpl
mpl.use("agg")

import matplotlib.pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

from hawc_hal import HAL, HealpixConeROI
from scipy.interpolate import interp1d
import progressbar

class LimitCalculator:

    def __init__(self, maptree, response, model, verbose=False, process="annihilation", energy_bins=True):
        self.maptree    = maptree
        self.response   = response
        self.model      = model
        self.verbose    = verbose
        self.DM_process = process
        # - construct the log likelihood
        self.roi      = HealpixConeROI(data_radius=13., model_radius=25., ra=187.5745, dec=10.1974)
        self.hawc     = HAL("VirgoCluster", maptree, response, self.roi)
        # these are the bins from the Energy Estimator, Spectrum Fitting Memo
        if energy_bins:
            self.hawc.set_active_measurements(bin_list=["1c", "1d", "1e", "2c", "2d", "2e", "2f", 
                                                        "3c", "3d", "3e", "3f", "4c", "4d", "4e", "4f", "4g", 
                                                        "5d", "5e", "5f", "5g", "5h", "6e", "6f", "6g", "6h", 
                                                        "7f", "7g", "7h", "7i", "8f", "8g", "8h", "8i", "8j", 
                                                        "9g", "9h", "9i", "9j", "9k", "9l"])
        else:
            self.hawc.set_active_measurements(1, 9)
        self.hawc.display()
        self.datalist = DataList(self.hawc)
        self.jl = JointLikelihood(self.model, self.datalist, verbose=True)

    """
    def set_ROI(self, ra, dec, radius):
        self.llh.set_ROI(ra, dec, radius, True)
        if self.verbose:
            print("ROI is set to ({:.2f}, {:.2f}) with r={:.2f}".format(ra, dec, radius))

    def set_masked_ROI(self, maskFilename):
        self.llh.set_template_ROI(maskFilename, 0.5, True)
        if self.verbose:
            print("ROI is adjusted according to {filename}".format(filename=maskFilename))

    def set_range(self, minimum, maximum):
        self.min = minimum
        self.max = maximum

    def calculate_limit(self, rel_err=1.e-3, do_flip=True):
        best_fit, TS_max = self.find_max_TS_3ml_style()
        #best_fit, TS_max = self.find_max_TS()

        self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.bounds = (-1e-24, 1e-15)

        if best_fit < 0 and do_flip:
            print("the best fit is negative. taking care of it now")
            best_fit = 0
            self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = best_fit
            TS_max = self.llh.calc_TS()

        lo = best_fit
        lo_TS = TS_max
        del_lo_TS = 2.71 - (TS_max-lo_TS)
        hi = lo*20.
        if hi == 0:
            hi = 1e-15
        self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = hi
        hi_TS = self.llh.calc_TS()
        del_hi_TS = 2.71 - (TS_max-hi_TS)

        while True:
            mid = (lo+hi)/2.
            self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = mid
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
    """
    
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
        if self.DM_process == "annihilation":
            best_fit = ll.likelihood_model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value
        else:
            best_fit = ll.likelihood_model.M49.spectrum.main.DMDecayFlux.tau.value
        vals = np.logspace(np.log10(best_fit)-1.0,
                           np.log10(best_fit)+1.0,
                           num_steps)
        for ival, val in enumerate(vals):
            if self.DM_process == "annihilation":
                ll.likelihood_model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = val
            else:
                ll.likelihood_model.M49.spectrum.main.DMDecayFlux.tau.value = val
            curr_LL = ll.data_list.values()[0].inner_fit()
            TS.append(curr_LL)
            bar.update(ival)

        TS = np.array(TS)
        TS = -TS
        TS -= np.min(TS)
        plt.semilogx(vals, TS)
        plt.savefig("example_step1.pdf")
        plt.show()

        if self.DM_process == "annihilation":
            selected_indices = vals > best_fit
        else:
            selected_indices = vals < best_fit
        
        interpolator = interp1d(TS[selected_indices], vals[selected_indices])
        return (best_fit, interpolator(1.35))
