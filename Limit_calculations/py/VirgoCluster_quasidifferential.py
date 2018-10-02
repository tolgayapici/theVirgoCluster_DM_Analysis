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
from LimitCalculator_quasidiff import *

parser = argparse.ArgumentParser(description="This script is to run extended source DM search for the Virgo Cluster")
parser.add_argument("-e",    dest="exp",     help="Experiment name of the add flag is True", choices=["VERITAS", "MAGIC", "HAWC"])
parser.add_argument("-v",    dest="verbose", help="Verbosity of the script",            default=True)
parser.add_argument("--maptree",  dest="maptree", default="../../data/maptree.root")
parser.add_argument("--response", dest="response", default="../../data/response.root")
parsed_args = parser.parse_args()

experiment       = parsed_args.exp
verbose          = parsed_args.verbose
maptree          = parsed_args.maptree

sources = Sources()
sources.setup(data=experiment, isQuasiDiff=True)

model = Model(sources.source_M87, sources.source_M49)

model.link(model.M87.spectrum.main.RangedPowerlaw.K,
           model.M49.spectrum.main.RangedPowerlaw.K,
           Identity())

if verbose:
    model.display()

lc = LimitCalculator(maptree, "../../data/response.root", model, verbose=verbose)
lc.set_ROI(sources.ROI_RA, sources.ROI_DEC, sources.ROI_radius)
lc.set_range(1e-25, 1e-10)
lc.calculate_limit()
