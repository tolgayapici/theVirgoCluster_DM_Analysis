import matplotlib as mpl
mpl.use("agg")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

class LimitCalculator:

    def __init__(self, maptree, response, model, verbose=False):
        self.maptree   = maptree
        self.response = response
        self.model    = model
        self.verbose  = verbose
        # - construct the log likelihood
        self.llh      = HAWCLike("VirgoCluster", maptree, response)
        self.llh.set_active_measurements(1, 9)
        self.llh.set_model(model)
        self.datalist = DataList(self.llh)
        self.jl = JointLikelihood(self.model, self.datalist, verbose=True)

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

    def redefine_range(self):
        self.model.M49.spectrum.main.RangedPowerlaw.K.value = 1e-10
        val1 = (self.llh.calc_TS())
        self.model.M49.spectrum.main.RangedPowerlaw.K.value = 1e-25
        val2 = (self.llh.calc_TS())
        if val1 < 0:
            self.max = 0
        if val2 < 0:
            self.min = 0

    def find_max_TS_3ml_style(self, make_model_map=True):
        val = np.linspace(self.min, self.max, 50)
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
        
        # self.redefine_range()
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
