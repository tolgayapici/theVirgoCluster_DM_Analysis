import matplotlib as mpl
mpl.use("agg")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

from hawc_hal import HAL, HealpixConeROI

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
        self.jl_linked    = JointLikelihood(self.models[2], self.datalist, verbose=True)
        self.jl_notlinked = JointLikelihood(self.models[3], self.datalist, verbose=True)
        # - set the minimizer
        self.jl_M87.set_minimizer("minuit")
        self.jl_M49.set_minimizer("minuit")
        self.jl_linked.set_minimizer("minuit")
        self.jl_notlinked.set_minimizer("minuit")
        
    def set_range(self, minimum, maximum):
        self.min = minimum
        self.max = maximum

    def find_max_TS_3ml_style(self, make_model_map=True):
        TS  = []
        last_TS  = -9e9
        #self.model.M87.spectrum.main.RangedPowerlaw.K.value = 1e-23
        #current_TS = self.jl.minus_log_like_profile(1e-23)
        result = self.jl_linked.fit()
        current_TS = self.jl_linked.minus_log_like_profile(self.jl_linked.likelihood_model.M87.spectrum.main.RangedPowerlaw.K.value,
                                                           self.jl_linked.likelihood_model.M87.spectrum.main.RangedPowerlaw.K.value)
        print(current_TS)
        """
        current_TS = self.jl_M49.minus_log_like_profile(self.models[2].M87.spectrum.main.RangedPowerlaw.K.value)
        print(current_TS)

        current_TS = self.jl_linked.minus_log_like_profile(self.models[2].M87.spectrum.main.RangedPowerlaw.K.value,
                                                           self.models[2].M87.spectrum.main.RangedPowerlaw.K.value)
        print(current_TS)
        """

        #print(result)
        #print(self.jl.compute_TS("M49", result[1]))
        results = self.jl.get_contours("M87.spectrum.main.RangedPowerlaw.K",
                                       1e-10,
                                       1e-25,
                                       10)
        print(results)
        ###profile_vals = self.jl.minus_log_like_profile(val)
        ###print(profile_vals)
                
        for curr_val in val:
            self.model.M49.spectrum.main.RangedPowerlaw.K.value = curr_val
            current_TS = self.jl.minus_log_like_profile(curr_val)
            print(curr_val, current_TS)
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
                break
            last_TS = current_TS
        TS = np.array(TS)
        max_index = np.argmax(TS)
        if self.verbose:
            print("max at: {} max TS: {}".format(val[max_index], TS[max_index]))
        
        self.min = val[max_index-1]/2.
        self.max = val[max_index+1]*2.
        print("this is where I print to screen")
        print(self.min, self.max)
        if self.min > self.max:
            max_ = self.max
            self.max = self.min
            self.min = max_
        print(self.model.M49.spectrum.main.RangedPowerlaw.K.value)
        self.model.M49.spectrum.main.RangedPowerlaw.K.bounds = (self.min, self.max)
        #self.jl.set_minimizer("ROOT")
        self.jl.set_minimizer("minuit")
        self.jl.fit(quiet=False)
        val = (self.model.M49.spectrum.main.RangedPowerlaw.K.value)
        TS  = (self.llh.calc_TS())
        
        if make_model_map:
            self.llh.write_model_map("../results/model_maps/results.root")

        print(val, TS)
        return val, TS

    def find_max_TS(self):
        #self.redefine_range()
        val = np.linspace(self.min, self.max, 100)
        TS  = []
        last_TS  = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.RangedPowerlaw.K.value = curr_val
            current_TS = (self.llh.calc_TS())
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
                break
            last_TS = current_TS
        TS = np.array(TS)
        max_index = np.argmax(TS)
        if self.verbose:
            print("max at: {} max TS: {}".format(val[max_index], TS[max_index]))
        print("second iteration")

        val = np.linspace(val[max_index-1], val[max_index+1])
        TS  = []
        last_TS  = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.RangedPowerlaw.K.value = curr_val
            current_TS = (self.llh.calc_TS())
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
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
            self.model.M49.spectrum.main.RangedPowerlaw.K.value = curr_val
            current_TS = (self.llh.calc_TS())
            TS.append(current_TS)
            if self.verbose:
                print(curr_val, current_TS)
            if last_TS > current_TS:
                break
            last_TS = current_TS
        TS = np.array(TS)
        max_index = np.argmax(TS)
        if self.verbose:
            print("max at: {} max TS: {}".format(val[max_index], TS[max_index]))
        return val[max_index], TS[max_index]

    def calculate_limit(self, rel_err=1.e-3, do_flip=True):
        best_fit, TS_max = self.find_max_TS_3ml_style()
        #best_fit, TS_max = self.find_max_TS()

        self.model.M49.spectrum.main.RangedPowerlaw.K.bounds = (-1e-24, 1e-15)

        if best_fit < 0 and do_flip:
            print("the best fit is negative. taking care of it now")
            best_fit = 0
            self.model.M49.spectrum.main.RangedPowerlaw.K.value = best_fit
            TS_max = self.llh.calc_TS()

        lo = best_fit
        lo_TS = TS_max
        del_lo_TS = 2.71 - (TS_max-lo_TS)
        hi = lo*20.
        if hi == 0:
            hi = 1e-15
        self.model.M49.spectrum.main.RangedPowerlaw.K.value = hi
        hi_TS = self.llh.calc_TS()
        del_hi_TS = 2.71 - (TS_max-hi_TS)

        while True:
            mid = (lo+hi)/2.
            self.model.M49.spectrum.main.RangedPowerlaw.K.value = mid
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
