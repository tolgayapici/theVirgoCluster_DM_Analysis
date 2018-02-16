import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import astropy.units as astropy_units

from astromodels.functions.function import Function1D, FunctionMeta

from threeML import EBLattenuation

from DMSpectra import *

class DMAnnihilationFlux(Function1D):
    r"""
        description :

            Class that evaluates the spectrum for a DM particle of a given
            mass, channel, cross section, and J-factor.

            The parameterization is given by

            F(x) = 1 / (8 * pi) * (1/mass^2) * sigmav * J * dN/dE(E,mass,i)

            Note that this class assumes that mass and J-factor are provided
            in units of GeV and GeV^2 cm^-5

        latex : $$

        parameters :

            mass :
                desc : DM mass (GeV)
                initial value : 10000
                fix : yes

            J :
                desc : Target total J-factor (GeV^2 cm^-5)
                initial value : 1.e20
                fix : yes

            channel :
                desc : DM annihilation channel
                initial value : 4
                fix : yes

            sigmav :
                desc : DM annihilation cross section (cm^3/s)
                initial value : 1.e-23
                min value : -1e-20
                max value : 1e-20
                fix: no

        """

    __metaclass__ = FunctionMeta

    def _setup(self):
        self.avail_EBL_models = ['gilmore', 'dominguez', 'finke']
        self.channel_mapping = {1:'b', 2:'t', 3:'W', 4:'tau', 5:'mu'}
        self.ebl             = None
        self.print_spec      = False

    def print_EBL_models(self):
        print self.avail_EBL_models

    def print_channel_in_text(self):
        return self.channel_mapping[self.channel.value]

    def set_EBL_model(self, model_name, red_shift):
        print("EBL MODEL IN DMMODELS {}".format(model_name))
        if model_name in self.avail_EBL_models:
            print("Setting EBL model to {}".format(model_name))
            self.ebl          = EBLattenuation()
            self.red_shift    = red_shift
            self.ebl.redshift = red_shift
            self.ebl.set_ebl_model(modelname=model_name)
        else:
            print("Unknown EBL model... It will not be set...")

    def set_print_spec(self, choice):
        self.print_spec = choice

    def unset_EBL_model(self):
        self.ebl = None

    def update_setup(self):
        spectrum = DMSpectra(self.mass, self.channel)
        self._interp = spectrum.get_spectra()

    def _set_units(self, x_unit, y_unit):
        self.mass.unit = astropy_units.GeV
        self.channel.unit = astropy_units.dimensionless_unscaled
        self.sigmav.unit = astropy_units.cm ** 3 / astropy_units.s
        self.J.unit = astropy_units.GeV ** 2 / astropy_units.cm ** 5

    # noinspection PyPep8Naming
    def evaluate(self, x, mass, J, channel, sigmav):
        # careful about the unit of x. It is in keV.
        self.update_setup()
        if isinstance(x, astropy_units.Quantity):
            # We need to convert to GeV
            xx = x.to(astropy_units.GeV)
        else:
            # If no unit for x is defined, assume it is keV
            x_with_unit = x*astropy_units.keV
            xx = x*1e-6*astropy_units.GeV #x_with_unit.to(astropy_units.GeV)
        phip = 1. / (8. * np.pi) * np.power(mass, -2) * (sigmav * J)  # 1/GeV^2 * cm^3/s * GeV^2/cm^5 = 1/cm^2.s
        dnde = self._interp(xx)
        flux = (phip*dnde/x)
        flux[np.isnan(flux)] = 0.

        if self.ebl is not None:
            EBL_attenuation = self.ebl.evaluate(x=x, redshift=self.red_shift)
            flux *= EBL_attenuation

        """ THIS PART IS ADDED FOR DEBUGGING """
        if self.print_spec:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.loglog(x, flux)
            plt.savefig("example_spectrum_{}GeV_{}.pdf".format(self.mass.value, self.channel.value))
            self.print_spec = False
        """ END OF DEBUGGING """

        return flux

class DMDecayFlux(Function1D):
    r"""
        description :

            Class that evaluates the spectrum for a DM particle of a given
            mass, channel, cross section, and J-factor.

            The parameterization is given by

            F(x) = 1 / (4 * pi * tau * mass) * J * dN/dE(E,mass,i)

            Note that this class assumes that mass and J-factor are provided
            in units of GeV and GeV cm^-2

        latex : $$

        parameters :

            mass :
                desc : DM mass (GeV)
                initial value : 10000
                fix : yes

            D :
                desc : Target total D-factor (GeV cm^-2)
                initial value : 1.e20
                fix : yes

            channel :
                desc : DM annihilation channel
                initial value : 4
                fix : yes

            tau :
                desc : DM decay lifetime (s)
                initial value : 1.e25
                min value : -1e20
                max value : 1e30
                fix: no

        """

    __metaclass__ = FunctionMeta

    def _setup(self):
        self.avail_EBL_models = ['gilmore', 'dominguez', 'finke']
        self.channel_mapping = {1:'b', 2:'t', 3:'W', 4:'tau', 5:'mu'}
        self.ebl             = None
        self.print_spec      = False

    def print_EBL_models(self):
        print self.avail_EBL_models

    def print_channel_in_text(self):
        return self.channel_mapping[self.channel.value]

    def set_EBL_model(self, model_name, red_shift):
        if model_name in self.avail_EBL_models:
            self.ebl          = EBLattenuation()
            self.red_shift    = red_shift
            self.ebl.redshift = red_shift
            self.ebl.set_ebl_model(modelname=model_name)
        else:
            print("Unknown EBL model... It will not be set...")

    def set_print_spec(self, choice):
        self.print_spec = choice

    def unset_EBL_model(self):
        self.ebl = None

    def update_setup(self):
        spectrum = DMSpectra(self.mass, self.channel)
        self._interp = spectrum.get_spectra()

    def _set_units(self, x_unit, y_unit):
        self.mass.unit = astropy_units.GeV
        self.channel.unit = astropy_units.dimensionless_unscaled
        self.tau.unit = astropy_units.s
        self.D.unit = astropy_units.GeV / astropy_units.cm ** 2

    # noinspection PyPep8Naming
    def evaluate(self, x, mass, D, channel, tau):
        # careful about the unit of x. It is in keV.
        self.update_setup()
        if isinstance(x, astropy_units.Quantity):
            # We need to convert to GeV
            xx = x.to(astropy_units.GeV)
        else:
            # If no unit for x is defined, assume it is keV
            x_with_unit = x*astropy_units.keV
            xx = x*1e-6*astropy_units.GeV #x_with_unit.to(astropy_units.GeV)
        phip = 1. / (4. * np.pi * mass * tau) * D  # 1/GeV * 1/s * GeV/cm^2 = 1/cm^2.s
        dnde = self._interp(xx)
        flux = (phip*dnde/x)
        flux[np.isnan(flux)] = 0.

        if self.ebl is not None:
            EBL_attenuation = self.ebl.evaluate(x=x, redshift=self.red_shift)
            flux *= EBL_attenuation

        """ THIS PART IS ADDED FOR DEBUGGING """
        if self.print_spec:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.loglog(x, flux)
            plt.savefig("example_spectrum_{}GeV_{}.pdf".format(self.mass.value, self.channel.value))
            self.print_spec = False
        """ END OF DEBUGGING """

        return flux
