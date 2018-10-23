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
parser.add_argument("-b",    dest="ebin",    help="Energy bin for quasi-differential limit", default=0, type=int)
parser.add_argument("-i",    dest="index",   help="Index for power law", default=-2.0, type=float)
parser.add_argument("--maptree",  dest="maptree", default="../../data/maptree.root")
parser.add_argument("--response", dest="response", default="../../data/response.root")
parsed_args = parser.parse_args()

experiment       = parsed_args.exp
verbose          = parsed_args.verbose
maptree          = parsed_args.maptree
ebin             = parsed_args.ebin
index            = parsed_args.index

lo_range = [0.562, 1.778, 5.623, 17.783, 56.234]
hi_range = [1.778, 5.623, 17.783, 56.234, 177.828]
#lo_range = [1.]
#hi_range = [20.]

sources = Sources()
sources.setup(data=experiment, isQuasiDiff=True, index=index, enLo=lo_range[ebin], enHi=hi_range[ebin])

model_M87 = Model(sources.source_M87)
model_M49 = Model(sources.source_M49)
model_linked    = Model(sources.source_M87, sources.source_M49)
model_notlinked = Model(sources.source_M87, sources.source_M49)

model_linked.link(model_linked.M49.spectrum.main.RangedPowerlaw.K,
                  model_linked.M87.spectrum.main.RangedPowerlaw.K,
                  Identity())

models = [model_M87, model_M49, model_linked, model_notlinked]

if verbose:
    model_linked.display()

lc = LimitCalculator(maptree, "../../data/response.root", models, verbose=verbose)
lc.set_range(1e-25, 1e-17)
ll = lc.get_linked_likelihood()
limit = lc.calculate_limit(ll)
print(limit)
