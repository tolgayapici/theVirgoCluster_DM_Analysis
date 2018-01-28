#!/usr/bin/env python

import numpy as np
import healpy as hp
import astropy.units as u

center_RA  = 187.7035
center_DEC = 12.3906

mask_center_RA  = 180.0
mask_center_DEC = 12.0

for nSide in [1024]:

    print('Making mask with nSide %d' %nSide)

    # Empty map
    deg = np.pi/180.
    m = np.zeros(nSide*nSide*12)

    # Add Crab
    ra = center_RA
    dec = center_DEC
    radius = 12.
    a = hp.query_disc(nSide,hp.ang2vec((90-dec)*deg,ra*deg),radius*deg)
    for p in a:
        m[p] = 1.

    # Remove B0540
    ra  = mask_center_RA
    dec = mask_center_DEC
    radius = 1.0
    c = hp.query_disc(nSide,hp.ang2vec((90-dec)*deg,ra*deg),radius*deg)
    for p in c:
        m[p] = 0.

    outFile = '../../data/maskedROI_nSide%d.fits.gz' %nSide
    hp.write_map(outFile,m)
    print('Wrote file %s' %outFile)

