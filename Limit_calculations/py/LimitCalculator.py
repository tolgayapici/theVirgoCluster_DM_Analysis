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

    def find_max_TS(self):
        val = np.linspace(self.min, self.max)
        TS  = []
        last_TS  = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = curr_val
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
            self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = curr_val
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
        val = np.linspace(val[max_index-1], val[max_index+1])
        TS  = []
        last_TS = -9e9
        for curr_val in val:
            self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = curr_val
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

    def calculate_limit(self, rel_err=1.e-2):
        best_fit, TS_max = self.find_max_TS()

        if best_fit < 0:
            print("the best fit is negative. taking care of it now")
            best_fit = 0
            self.model.M49.spectrum.main.DMAnnihilationFlux.sigmav.value = best_fit
            TS_max = self.llh.calc_TS()

        lo = best_fit
        lo_TS = TS_max
        del_lo_TS = 2.71 - (TS_max-lo_TS)
        hi = lo*20.
        if hi == 0:
            hi = 1e-22
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
