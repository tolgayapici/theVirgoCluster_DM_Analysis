import os

# take care of matplotlib to use correct
if os.environ.get('HOME') is not "/home/tyapici":
    import matplotlib as mpl
    mpl.use("agg")

# stop ROOT hijacking the command line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *

import numpy as np
import pdb
import shutil
import matplotlib.pyplot as plt

from astropy.io import fits

import argparse

import sys
sys.path.append(".")
from Sources import *
from LimitCalculator import *

parser = argparse.ArgumentParser(description="This script is to run extended source DM search for the Virgo Cluster")
parser.add_argument("-m",    dest="mass",    help="Mass of the DM particle (in GeV)",   required=True, type=float)
parser.add_argument("-c",    dest="channel", help="Annihilation channel",               required=True, choices=[1, 2, 3, 4, 5], type=int)
parser.add_argument("-t",    dest="model",   help="DM Template",                        required=True, choices=["GAO", "B01", "min", "med", "max"])
parser.add_argument("-a",    dest="add",     help="A flag to add the M87 point source", default=0, type=int)
parser.add_argument("-e",    dest="exp",     help="Experiment name of the add flag is True", choices=["VERITAS", "MAGIC", "HAWC"])
parser.add_argument("-v",    dest="verbose", help="Verbosity of the script",            default=True)
parser.add_argument("-sra",  dest="shiftra", help="shift in ra for the expected limit calculations",  default=0., type=float)
parser.add_argument("-sdec", dest="shiftdec", help="shift in dec for the expected limit calculations", default=0., type=float)
parser.add_argument("--ebl", dest="ebl",  help="EBL model to be used in the flux calculations", choices=['gilmore', 'dominguez', 'finke'], default=None)
parser.add_argument("--maptree",  dest="maptree", default="../../data/maptree.root")
parser.add_argument("--response", dest="response", default="../../data/response.root")
parsed_args = parser.parse_args()

mass             = parsed_args.mass
channel          = parsed_args.channel
DM_model         = parsed_args.model
add_point_source = parsed_args.add
experiment       = parsed_args.exp
verbose          = parsed_args.verbose
ebl_model        = parsed_args.ebl
ra_shift         = parsed_args.shiftra
dec_shift        = parsed_args.shiftdec
maptree          = parsed_args.maptree

print("running for mass {} GeV".format(mass))
print("            channel {}".format(channel))
print("            {} EBL Model".format(ebl_model))

sources = Sources(DM_model)
sources.set_mass(float(mass))
sources.set_channel(int(float(channel)))
sources.set_EBL_model(ebl_model_name=ebl_model)
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

lc = LimitCalculator(maptree, "../../data/response.root", model, verbose=verbose)
lc.set_ROI(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius)
lc.set_range(-1e-21, 1e-20)
lc.calculate_limit()
