# take care of matplotlib to use correct
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
import os
import matplotlib.pyplot as plt

from astropy.io import fits

import argparse

import sys
sys.path.append(".")
from DMModels import *

RA  = sys.argv[1]
DEC = sys.argv[2]

calc_TS = True#False

def run_analysis():
    # set source
    spec       = Cutoff_powerlaw()
    point_M87  = PointSource("M87Point", ra=RA, dec=DEC, spectral_shape=spec)
    model      = Model(point_M87)
    # spectrum
    spec.K        = 1e-21
    spec.K.bounds = (1e-26, 1e-19)
    spec.piv      = 2.0 * u.TeV
    spec.piv.fix  = True
    spec.index        = -3.0
    spec.index.bounds = (-5.0, -1.0)
    spec.xc     = 6. * u.TeV
    spec.xc.fix = True
    model.display()
    llh = HAWCLike("VirgoCluster", "../../data/maptree.root", "../../data/response.root")
    llh.set_active_measurements(1, 9)
    llh.set_ROI(RA, DEC, 3, True)
    llh.set_model(model)
    datalist = DataList(llh)
    jl = JointLikelihood(model, datalist, verbose=False)
    jl.set_minimizer("ROOT")
    xcs = np.arange(2., 20., 1.)
    spec.piv = 6. * u.TeV
    jl.fit(quiet=False)
    print("XC: {} TeV, TS_max: {}".format(6.0, llh.calc_TS()))

TS = run_analysis()

