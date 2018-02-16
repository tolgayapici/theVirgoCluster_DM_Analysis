import matplotlib as mpl
mpl.use("agg")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

from DMModels import *
crosssection_lo = -1e-18
crosssection_hi = 1e-15

class Sources():

    def __init__(self, model):
        self.ebl_model_name    = None
        self.DM_model          = model
        self.M87_fits_template_org = "../../J_factor_calculations/templates/M87_{}_Jfactor_template.fits".format(self.DM_model)
        self.M49_fits_template_org = "../../J_factor_calculations/templates/M49_{}_Jfactor_template.fits".format(self.DM_model)
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

    def set_EBL_model(self, ebl_model_name=None):
        print("EBL model: {}".format(ebl_model_name))
        self.ebl_model_name = ebl_model_name

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
        if self.ebl_model_name is not None:
            spec_M87.set_EBL_model(self.ebl_model_name, 0.004)

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
        if self.ebl_model_name is not None:
            spec_M49.set_EBL_model(self.ebl_model_name, 0.004)

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
