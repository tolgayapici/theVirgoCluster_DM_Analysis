#!/usr/bin/env python

import numpy as np
import healpy as hp
import astropy.units as u

RA  = 187.7035
DEC = 12.3906

for nSide in [1024]:

    print('Making mask with nSide %d' %nSide)

    # Empty map
    deg = np.pi/180.
    m = np.zeros(nSide*nSide*12)

    # Add Crab
    ra = RA
    dec = DEC
    radius = 12.
    a = hp.query_disc(nSide,hp.ang2vec((90-dec)*deg,ra*deg),radius*deg)
    for p in a:
        m[p] = 1.

    # Remove B0540
    ra  = RA
    dec = DEC
    radius = 0.5
    c = hp.query_disc(nSide,hp.ang2vec((90-dec)*deg,ra*deg),radius*deg)
    for p in c:
        m[p] = 0.

    outFile = '../../data/mask_noB_nSide%d.fits.gz' %nSide
    hp.write_map(outFile,m)
    print('Wrote file %s' %outFile)

