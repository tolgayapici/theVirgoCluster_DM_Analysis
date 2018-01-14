
# This script demonstrates to transform the output of CLUMPY to a template we can use for the HAWC analysis

# import the necessary packages
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt # may not be necessary
import matplotlib.colors as colors
import healpy as hp
from scipy.interpolate import interp2d
import pickle
import os

from matplotlib import rcParams
rcParams['figure.figsize'] = [8.0, 6.0]
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 11

# M87
lon_M87 = 283.77
lat_M87 = 74.49

# M49
lon_M49 = 286.92
lat_M49 = 70.20

# Virgo Cluster
lon_Virgo = 283.77
lat_Virgo = 74.49

def select_model(profile, fov, alpha):
    if alpha == 0.06:
        nside = 2048
    else:
        nside = 1024
    M87_filename    = "../output/annihil_M87_{}2D_FOVdiameter{:.1f}deg_rse1_alphaint{:.2f}deg_nside{}.fits".format(profile, fov, alpha, nside)
    M49_filename    = "../output/annihil_M49_{}2D_FOVdiameter{:.1f}deg_rse1_alphaint{:.2f}deg_nside{}.fits".format(profile, fov, alpha, nside)
    VirgoC_filename = "../output/annihil_VirgoC_{}2D_FOVdiameter{:.1f}deg_rse1_alphaint{:.2f}deg_nside{}.fits".format(profile, fov, alpha, nside)

    M87_hdus = fits.open(M87_filename)
    M49_hdus = fits.open(M49_filename)
    Virgo_hdus = fits.open(VirgoC_filename)
    
    for i in range(1, len(M87_hdus)):
        print("header {}: {}".format(i, M87_hdus[i].header['EXTNAME']))

    return M87_hdus, M49_hdus, Virgo_hdus

def generate_cartesian_file(hdus, lon, lat, label, index=1):
    content = hdus[index].header
    theta_0 = content['THETA_0']
    psi_0   = content['PSI_0']
    size_x  = content['SIZE_X']
    size_y  = content['SIZE_Y']
    dangle  = content['ALPHAINT']

    content = hdus[index].data
    pixels  = content['PIXEL']
    Jtotal  = content['Jtot']
    Jsmooth = content['Jsmooth']
    Jsub    = content['Jsub']
    NSIDE    = hdus[2].header['NSIDE']
    ordering = hdus[2].header['ORDERING']
    if ordering == "NESTED":
        nested = True
    else:
        nested = False

    nPix = hp.nside2npix(NSIDE)
    num_used_pixels = len(pixels)
    theta_rad, phi_rad = hp.pixelfunc.pix2ang(NSIDE, pixels, nest=nested)
    # change the angles according to our conventions
    theta, phi = -np.degrees(theta_rad-np.pi/2.-lon), np.degrees(np.pi*2.-phi_rad+lat)
    phi[np.where(phi>360)] -= 360.
    # Process the data for 3ML use
    nxPix = 140
    nyPix = 140
    refX  = 70.5
    refY  = 70.5
    delX  = -dangle
    delY  = dangle
    x = np.zeros(nPix)
    x[pixels] = Jtotal/np.max(Jtotal)
    dmROI = np.zeros([nxPix, nyPix])
    for i in range(nxPix):
        for j in range(nyPix):
            ra_roi = (i-np.int(refX))*delX                                                                          
            dec_roi = (j-np.int(refY))*delY
            hpix = hp.pixelfunc.ang2pix(NSIDE,np.radians(-dec_roi+90.),np.radians(360.-ra_roi), nest=nested)
            #print(hpix)
            dmROI[i,j] = x[hpix]
    dmROI = np.multiply(dmROI, (np.degrees(1)**2/delX**2))
    # convert from galactic coordinates to fk5
    coords = SkyCoord(lon*u.degree, lat*u.degree, frame='galactic', unit='degree')
    RA  = coords.fk5.ra.degree
    DEC = coords.fk5.dec.degree
    # write the output to the new file
    new_hdu = fits.PrimaryHDU(dmROI)
    new_hdu.header['CTYPE1'] = 'RA'
    new_hdu.header['CTYPE2'] = 'DEC'
    new_hdu.header['CUNIT1'] = 'deg'
    new_hdu.header['CUNIT2'] = 'deg'
    new_hdu.header['CRPIX1'] = refX
    new_hdu.header['CRPIX2'] = refY
    new_hdu.header['CRVAL1'] = RA
    new_hdu.header['CRVAL2'] = DEC
    new_hdu.header['CDELT1'] = delX
    new_hdu.header['CDELT2'] = delY
    #new_hdu.header['DMAX']   = np.log10(Jtotal.max())
    hdulist = fits.HDUList([new_hdu])
    hdulist.writeto('../templates/{}_Jfactor_template.fits'.format(label))
    f = open('../templates/{}_Jfactor_template.txt'.format(label), "w")
    f.write("For {}: Jmax={}".format(label, np.log10(Jtotal.max())))

"""
M87_hdus, M49_hdus, Virgo_hdus = select_model("GAO", 7.0, 0.11)
    
generate_cartesian_file(M87_hdus, lon_M87, lat_M87, "M87_GAO")
generate_cartesian_file(M49_hdus, lon_M49, lat_M49, "M49_GAO")
generate_cartesian_file(Virgo_hdus, lon_Virgo, lat_Virgo, "VirgoC_GAO")
"""
M87_hdus, M49_hdus, Virgo_hdus = select_model("B01", 7.0, 0.11)

generate_cartesian_file(M87_hdus, lon_M87, lat_M87, "M87_B01")
generate_cartesian_file(M49_hdus, lon_M49, lat_M49, "M49_B01")
generate_cartesian_file(Virgo_hdus, lon_Virgo, lat_Virgo, "VirgoC_B01")
