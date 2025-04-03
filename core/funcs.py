"""Functions file used for calibration of MIRI data.

:History:

Created on Thu Mar 01 10:58:50 2018

@author: Ioannis Argyriou (KULeuven, Belgium, ioannis.argyriou@kuleuven.be)
"""

# import python modules
import os
import pickle
import itertools
import numpy as np
from numpy import arcsin
import scipy.special as sp
from scipy.optimize import curve_fit, least_squares
from astropy.io import fits
from astropy.modeling.functional_models import Moffat1D
import scipy.interpolate as scp_interpolate
import matplotlib.pyplot as plt


def point_source_centroiding(band,sci_img,d2cMaps,spec_grid=None,fit='2D',center=None,offset_slice=0,verbose=True):
    # distortion maps
    sliceMap  = d2cMaps['sliceMap']
    lambdaMap = d2cMaps['lambdaMap']
    alphaMap  = d2cMaps['alphaMap']
    betaMap   = d2cMaps['betaMap']
    nslices   = d2cMaps['nslices']
    MRS_alphapix = {'1':0.196,'2':0.196,'3':0.245,'4':0.273} # arcseconds
    MRS_FWHM = {'1':2.16*MRS_alphapix['1'],'2':3.30*MRS_alphapix['2'],
                '3':4.04*MRS_alphapix['3'],'4':5.56*MRS_alphapix['4']} # MRS PSF
    mrs_fwhm  = MRS_FWHM[band[0]]
    lambcens,lambfwhms = spec_grid[0],spec_grid[1]
    unique_betas = np.sort(np.unique(betaMap[(sliceMap>100*int(band[0])) & (sliceMap<100*(int(band[0])+1))]))
    fov_lims  = [alphaMap[np.nonzero(lambdaMap)].min(),alphaMap[np.nonzero(lambdaMap)].max()]

    if verbose:
        print('STEP 1: Rough centroiding')
    if center is None:
        # premise> center of point source is located in slice with largest signal
        # across-slice center:
        sum_signals = np.zeros(nslices)
        for islice in range(1,1+nslices):
            sum_signals[islice-1] = sci_img[(sliceMap == 100*int(band[0])+islice) & (~np.isnan(sci_img))].sum()
        source_center_slice = np.argmax(sum_signals)+1
        source_center_slice+=offset_slice

        # along-slice center:
        det_dims = (1024,1032)
        img = np.full(det_dims,0.)
        sel = (sliceMap == 100*int(band[0])+source_center_slice)
        img[sel]  = sci_img[sel]

        source_center_alphas = []
        for row in range(det_dims[0]):
            source_center_alphas.append(alphaMap[row,img[row,:].argmax()])
        source_center_alphas = np.array(source_center_alphas)
        source_center_alpha  = np.average(source_center_alphas[~np.isnan(source_center_alphas)])
    else:
        source_center_slice,source_center_alpha = center[0],center[1]
    if verbose:
        # summary:
        print( 'Slice {} has the largest summed flux'.format(source_center_slice))
        print( 'Source position: beta = {}arcsec, alpha = {}arcsec \n'.format(round(unique_betas[source_center_slice-1],2),round(source_center_alpha,2)))

    if fit == '0D':
        return source_center_slice,unique_betas[source_center_slice-1],source_center_alpha

    if verbose:
        print( 'STEP 2: 1D Gaussian fit')

    # Fit Gaussian distribution to along-slice signal profile
    sign_amp,alpha_centers,alpha_fwhms,bkg_signal = [np.full((len(lambcens)),np.nan) for j in range(4)]
    sign_amp_sliceoffsetminus1,alpha_centers_sliceoffsetminus1,alpha_fwhms_sliceoffsetminus1,bkg_signal_sliceoffsetminus1 = [np.full((len(lambcens)),np.nan) for j in range(4)]
    sign_amp_sliceoffsetplus1,alpha_centers_sliceoffsetplus1,alpha_fwhms_sliceoffsetplus1,bkg_signal_sliceoffsetplus1 = [np.full((len(lambcens)),np.nan) for j in range(4)]
    failed_fits = []
    for ibin in range(len(lambcens)):
        coords = np.where((sliceMap == 100*int(band[0])+source_center_slice) & (np.abs(lambdaMap-lambcens[ibin])<=lambfwhms[ibin]/2.) & (~np.isnan(sci_img)))
        coords_sliceoffsetminus1 = np.where((sliceMap == 100*int(band[0])+source_center_slice-1) & (np.abs(lambdaMap-lambcens[ibin])<=lambfwhms[ibin]/2.) & (~np.isnan(sci_img)))
        coords_sliceoffsetplus1 = np.where((sliceMap == 100*int(band[0])+source_center_slice+1) & (np.abs(lambdaMap-lambcens[ibin])<=lambfwhms[ibin]/2.) & (~np.isnan(sci_img)))
        if len(coords[0]) == 0:
            failed_fits.append(ibin); continue
        try:
            popt,pcov = curve_fit(gauss1d_wBaseline, alphaMap[coords], sci_img[coords], p0=[sci_img[coords].max(),source_center_alpha,mrs_fwhm/2.355,0],method='lm')
            popt_sliceoffsetminus1,pcov = curve_fit(gauss1d_wBaseline, alphaMap[coords_sliceoffsetminus1], sci_img[coords_sliceoffsetminus1], p0=[sci_img[coords_sliceoffsetminus1].max(),source_center_alpha,mrs_fwhm/2.355,0],method='lm')
            popt_sliceoffsetplus1,pcov = curve_fit(gauss1d_wBaseline, alphaMap[coords_sliceoffsetplus1], sci_img[coords_sliceoffsetplus1], p0=[sci_img[coords_sliceoffsetplus1].max(),source_center_alpha,mrs_fwhm/2.355,0],method='lm')
        except:
            failed_fits.append(ibin); continue
        sign_amp[ibin]      = popt[0]+popt[3]
        alpha_centers[ibin] = popt[1]
        alpha_fwhms[ibin]   = 2.355*np.abs(popt[2])
        bkg_signal[ibin]    = popt[3]

        sign_amp_sliceoffsetminus1[ibin]      = popt_sliceoffsetminus1[0]+popt_sliceoffsetminus1[3]
        alpha_centers_sliceoffsetminus1[ibin] = popt_sliceoffsetminus1[1]
        alpha_fwhms_sliceoffsetminus1[ibin]   = 2.355*np.abs(popt_sliceoffsetminus1[2])
        bkg_signal_sliceoffsetminus1[ibin]    = popt_sliceoffsetminus1[3]

        sign_amp_sliceoffsetplus1[ibin]      = popt_sliceoffsetplus1[0]+popt_sliceoffsetplus1[3]
        alpha_centers_sliceoffsetplus1[ibin] = popt_sliceoffsetplus1[1]
        alpha_fwhms_sliceoffsetplus1[ibin]   = 2.355*np.abs(popt_sliceoffsetplus1[2])
        bkg_signal_sliceoffsetplus1[ibin]    = popt_sliceoffsetplus1[3]

    # omit outliers
    for i in range(len(np.diff(sign_amp))):
        if np.abs(np.diff(alpha_centers)[i]) > 0.05:
            sign_amp[i],sign_amp[i+1],alpha_centers[i],alpha_centers[i+1],alpha_fwhms[i],alpha_fwhms[i+1] = [np.nan for j in range(6)]

    if verbose:
        print( '[Along-slice fit] The following bins failed to converge:')
        print( failed_fits)

    # Fit Gaussian distribution to across-slice signal profile (signal brute-summed in each slice)
    summed_signal,beta_centers,beta_fwhms = [np.full((len(lambcens)),np.nan) for j in range(3)]
    failed_fits = []
    for ibin in range(len(lambcens)):
        if np.isnan(alpha_centers[ibin]):
            failed_fits.append(ibin)
            continue
        sel = (np.abs(lambdaMap-lambcens[ibin])<=lambfwhms[ibin]/2.) & (~np.isnan(sci_img))
        try:
            signals = np.array([sci_img[(sliceMap == 100*int(band[0])+islice) & sel][np.abs(alphaMap[(sliceMap == 100*int(band[0])+islice) & sel]-alpha_centers[ibin]).argmin()] for islice in range(1,1+nslices)])
            signals[source_center_slice-2:source_center_slice+1] = np.array([sign_amp_sliceoffsetminus1[ibin],sign_amp[ibin],sign_amp_sliceoffsetplus1[ibin]])
        except ValueError:
            failed_fits.append(ibin)
            continue
        try:
            popt,pcov = curve_fit(gauss1d_wBaseline, unique_betas, signals, p0=[signals.max(),unique_betas[source_center_slice-1],mrs_fwhm/2.355,0],method='lm')
        except:
            failed_fits.append(ibin)
            continue
        summed_signal[ibin] = popt[0]+popt[3]
        beta_centers[ibin]  = popt[1]
        beta_fwhms[ibin]    = 2.355*np.abs(popt[2])

    # # omit outliers
    # for i in range(len(np.diff(summed_signal))):
    #     if np.abs(np.diff(beta_centers)[i]) > 0.05:
    #         summed_signal[i],summed_signal[i+1],beta_centers[i],beta_centers[i+1],beta_fwhms[i],beta_fwhms[i+1] = [np.nan for j in range(6)]
    if verbose:
        print( '[Across-slice fit] The following bins failed to converge:')
        print( failed_fits)
        print( '')

    if fit == '1D':
        sigma_alpha, sigma_beta = alpha_fwhms/2.355, beta_fwhms/2.355
        return sign_amp,alpha_centers,beta_centers,sigma_alpha,sigma_beta,bkg_signal

    elif fit == '2D':
        if verbose:
            print( 'STEP 3: 2D Gaussian fit')
        sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D = [np.full((len(lambcens)),np.nan) for j in range(6)]
        failed_fits = []

        for ibin in range(len(lambcens)):
            # initial guess for fitting, informed by previous centroiding steps
            amp,alpha0,beta0  = sign_amp[ibin],alpha_centers[ibin],beta_centers[ibin]
            sigma_alpha, sigma_beta = alpha_fwhms[ibin]/2.355, beta_fwhms[ibin]/2.355
            base = 0
            guess = [amp, alpha0, beta0, sigma_alpha, sigma_beta, base]
            bounds = ([0,-np.inf,-np.inf,0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

            # data to fit
            coords = (np.abs(lambdaMap-lambcens[ibin])<lambfwhms[ibin]/2.) & (~np.isnan(sci_img))
            alphas, betas, zobs   = alphaMap[coords],betaMap[coords],sci_img[coords]
            alphabetas = np.array([alphas,betas])

            # perform fitting
            try:
                popt,pcov = curve_fit(gauss2d, alphabetas, zobs, p0=guess,bounds=bounds)
            except:
                failed_fits.append(ibin); continue

            sign_amp2D[ibin]      = popt[0]
            alpha_centers2D[ibin] = popt[1]
            beta_centers2D[ibin]  = popt[2]
            sigma_alpha2D[ibin]   = popt[3]
            sigma_beta2D[ibin]    = popt[4]
            bkg_amp2D[ibin]       = popt[5]
        if verbose:
            print( 'The following bins failed to converge:')
            print( failed_fits)

        return sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D

#--fit
# 1d
def gauss1d_wBaseline(x, A, mu, sigma, baseline):
    return  A*np.exp(-(x-mu)**2/(2*sigma)**2) + baseline

def gauss1d_woBaseline(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma)**2)

# 2d
def gauss2d(xy, amp, x0, y0, sigma_x, sigma_y, base):
    # assert that values are floats
    amp, x0, y0, sigma_x, sigma_y, base = float(amp),float(x0),float(y0),float(sigma_x),float(sigma_y),float(base)
    x, y = xy
    a = 1/(2*sigma_x**2)
    b = 1/(2*sigma_y**2)
    inner = a * (x - x0)**2
    inner += b * (y - y0)**2
    return amp * np.exp(-inner) + base

#--find
def find_nearest(array,value):
    return np.abs(array-value).argmin()

def detpixel_trace(band,d2cMaps,sliceID=None,alpha_pos=None):
    # detector dimensions
    det_dims=(1024,1032)
    # initialize placeholders
    ypos,xpos = np.arange(det_dims[0]),np.zeros(det_dims[0])
    slice_img,alpha_img = [np.full(det_dims,0.) for j in range(2)]
    # create pixel masks
    sel_pix = (d2cMaps['sliceMap'] == 100*int(band[0])+sliceID) # select pixels with correct slice number
    slice_img[sel_pix] = d2cMaps['sliceMap'][sel_pix]           # image containing single slice
    alpha_img[sel_pix] = d2cMaps['alphaMap'][sel_pix]           # image containing alpha positions in single slice

    # find pixel trace
    for row in ypos:
        if band[0] in ['1','4']:
            xpos[row] = np.argmin(alpha_img[row,:])+find_nearest(alpha_img[row,:][(slice_img[row,:]!=0)],alpha_pos)
        elif band[0] in ['2','3']:
            xpos[row] = np.argmax(alpha_img[row,:])+find_nearest(alpha_img[row,:][(slice_img[row,:]!=0)],alpha_pos)
    xpos = xpos.astype(int)

    return ypos,xpos

def detpixel_trace_compactsource(sci_img,band,d2cMaps,offset_slice=0,verbose=False):
    # detector dimensions
    det_dims  = (1024,1032)
    nslices   = d2cMaps['nslices']
    sliceMap  = d2cMaps['sliceMap']
    lambdaMap = d2cMaps['lambdaMap']
    ypos,xpos = np.arange(det_dims[0]),np.zeros(det_dims[0])

    sum_signals = np.zeros(nslices)
    for islice in range(1+nslices):
        sum_signals[islice-1] = sci_img[(sliceMap == 100*int(band[0])+islice) & (~np.isnan(sci_img))].sum()
    source_center_slice = np.argmax(sum_signals)+1
    if verbose==True:
        print( 'Source center slice ID: {}'.format(source_center_slice))

    signal_img = np.full(det_dims,0.)
    sel_pix = (sliceMap == 100*int(band[0])+source_center_slice+offset_slice)
    signal_img[sel_pix] = sci_img[sel_pix]
    for row in ypos:
        signal_row = signal_img[row,:].copy()
        signal_row[np.isnan(signal_row)] = 0.
        xpos[row] = np.argmax(signal_row)
        if (row>1) & (row<=512) & (xpos[row]<xpos[row-1]):
            xpos[row] = xpos[row-1]
        elif (row>1) & (row>512) & (xpos[row]>xpos[row-1]):
            xpos[row] = xpos[row-1]
    xpos = xpos.astype(int)
    # correct edge effects
    xpos[:2] = xpos[2]
    xpos[-2:] = xpos[-3]
    # there can be no jumps/discontinuities of more than 3 pixels, run loops twice
    if len(np.where(abs(np.diff(xpos))>1)[0]) > 0:
        xpos[np.where(abs(np.diff(xpos))>1)[0][0]+1:] = xpos[1023-ypos[np.where(abs(np.diff(xpos))>1)[0][0]+1:]]
    if len(np.where(abs(np.diff(xpos))>1)[0]) > 0:
        xpos[np.where(abs(np.diff(xpos))>1)[0][0]+1:] = xpos[1023-ypos[np.where(abs(np.diff(xpos))>1)[0][0]+1:]]

    return ypos,xpos
