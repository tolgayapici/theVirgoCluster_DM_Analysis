import numpy as np
from scipy.interpolate import interp1d
import astropy.units as astropy_units

class DMSpectra():
    r"""


    """

    def __init__(self, mass, channel):
        self.mass = mass
        self.channel = channel
        self.channel_mapping = {1:'b', 2:'t', 3:'W', 4:'tau', 5:'mu'}

    def find_nearest(self, array, value):
        return (np.abs(array-value)).argmin()
        
    def get_spectra(self):
        channel = self.channel_mapping[self.channel.value]
        x = np.load("../../data/all_spectra_processed.npz")
        dnde    = []
        masses  = np.sort(list(x[channel][()].keys()))
        idx     = self.find_nearest(masses, self.mass.value)
        spectrum = x[channel][()][masses[idx]]
        spectrum_m1 = x[channel][()][masses[idx-1]]
        spectrum_p1 = x[channel][()][masses[idx+1]]
        spectrum_m2 = x[channel][()][masses[idx-2]]
        spectrum_p2 = x[channel][()][masses[idx+2]]
        spectrum_m3 = x[channel][()][masses[idx-3]]
        spectrum_p3 = x[channel][()][masses[idx+3]]
        interpolator_m3 = interp1d(spectrum_m3[0][spectrum_m3[1]>0]/masses[idx-3], np.log10(spectrum_m3[1][spectrum_m3[1]>0]),
                                   bounds_error=False, fill_value=None)
        interpolator_m2 = interp1d(spectrum_m2[0][spectrum_m2[1]>0]/masses[idx-2], np.log10(spectrum_m2[1][spectrum_m2[1]>0]),
                                   bounds_error=False, fill_value=None)
        interpolator_m1 = interp1d(spectrum_m1[0][spectrum_m1[1]>0]/masses[idx-1], np.log10(spectrum_m1[1][spectrum_m1[1]>0]),
                                   bounds_error=False, fill_value=None)
        interpolator   = interp1d(spectrum[0][spectrum[1]>0]/masses[idx], np.log10(spectrum[1][spectrum[1]>0]),
                                  bounds_error=False, fill_value=None)
        interpolator_p1 = interp1d(spectrum_p1[0][spectrum_p1[1]>0]/masses[idx+1], np.log10(spectrum_p1[1][spectrum_p1[1]>0]),
                                   bounds_error=False, fill_value=None)
        interpolator_p2 = interp1d(spectrum_p2[0][spectrum_p2[1]>0]/masses[idx+2], np.log10(spectrum_p2[1][spectrum_p2[1]>0]),
                                   bounds_error=False, fill_value=None)
        interpolator_p3 = interp1d(spectrum_p3[0][spectrum_p3[1]>0]/masses[idx+3], np.log10(spectrum_p3[1][spectrum_p3[1]>0]),
                                   bounds_error=False, fill_value=None)
        for energy in spectrum[0]/masses[idx]:
            interpolator = interp1d([masses[idx-3], masses[idx-2], masses[idx-1],
                                     masses[idx],
                                     masses[idx+1], masses[idx+2], masses[idx+3]],
                                    [interpolator_m3(energy), interpolator_m2(energy), interpolator_m1(energy),
                                     interpolator(energy),
                                     interpolator_p1(energy), interpolator_p2(energy), interpolator_p3(energy)],
                                     fill_value='extrapolate', kind='zero')
            dnde.append(interpolator(self.mass.value))
        dnde = np.array(dnde)
        self._interp = interp1d(spectrum[0]*astropy_units.GeV, 10**dnde/astropy_units.TeV, 
                                bounds_error=False, fill_value=None, kind='zero')
        return self._interp
