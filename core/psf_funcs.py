import numpy as np
from astropy.io import fits
import scipy.interpolate as interp
import scipy.optimize as opt
import pickle as pkl
import pandas as pd
import core.funcs as funcs
import core.bspline as bs
from core.distortion import d2cMapping
from scipy.ndimage import gaussian_filter1d
from os import listdir,path,makedirs
from numpy.random import poisson
import miricoord.mrs.mrs_tools as coord

def _create_incremental_alphaMap(band, d2cMaps):
    
    y_len, x_len = d2cMaps["alphaMap"].shape

    if band[0] in ['1','4']:
        customAM = np.copy(d2cMaps["alphaMap"][:, :512])
        
        grad = customAM[:, 1:] - customAM[:, :-1]
        grad = grad.clip(min=0)

    else:
        customAM = np.copy(d2cMaps["alphaMap"][:, 512:])
    
        grad = customAM[:, 1:] - customAM[:, :-1]
        grad = grad.clip(max=0)
        
#     print(len(grad[:,0]))
    x_range = np.arange(len(customAM[0,:])-1)

    for ii in range(y_len):
        grad_ii = grad[ii]
        non0s = np.where(grad_ii!=0)[0][:]

        filled_grad = np.interp(x_range, x_range[non0s], grad_ii[non0s])
        grad[ii] = filled_grad

        # Avoid gradient too small (too close to 0) where there is no alphaMap
        grad[np.where((grad<0.0001) & (grad>0))] = 0.0001
        grad[np.where((grad>-0.0001) & (grad<0))] = -0.0001
        
    for kk in range(1,len(customAM[0,:])-1):
        customAM[:, kk] = customAM[:,kk-1]+ grad[:,kk]
    
    customAM[:,-1] = customAM[:,-2] + 2*grad[:,kk]

    return customAM

def centroiding(test_rate,d2cMaps,band):
    
    lambmin = np.nanmin(d2cMaps['lambdaMap'][np.where(d2cMaps['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps['lambdaMap']) # micron
    lambcens = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    lambfwhms = np.ones(len(lambcens))*(2*(lambmax-lambmin)/1024.)

    sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D = funcs.point_source_centroiding(band,test_rate,d2cMaps,spec_grid=[lambcens,lambfwhms],fit='2D')

    alpha_ind=np.where(np.abs(alpha_centers2D)<=np.nanmax(np.abs(d2cMaps['alphaMap'])))[0][:]
    beta_ind=np.where(np.abs(beta_centers2D)<=np.nanmax(np.abs(d2cMaps['betaMap'])))[0][:]

    popt_alpha = np.polyfit(lambcens[alpha_ind],alpha_centers2D[alpha_ind],4)
    alpha = np.poly1d(popt_alpha)

    popt_beta  = np.polyfit(lambcens[beta_ind],beta_centers2D[beta_ind],4)
    beta = np.poly1d(popt_beta)
    
    return alpha, beta

def make_wave_img(wave,lambdaMap,band):
    
    if band[0] in ['1','4']:
        npix = 512
        pix0 = 0
        k = np.arange(npix)
    else:
        pix0 = 512
        npix = 1032-pix0
        k = np.arange(pix0,pix0+npix)
        
    test_img = np.zeros((len(lambdaMap[:,0]),npix))
    
    for x in range(0,npix):
        test_img[np.argmin(np.abs(lambdaMap[:,x]-wave)),x] = 1
        
    return test_img

def wave_norm(data_array,err,band,lambdaMap):
    
    if band[0] in ['1','4']:
        npix = 512
        pix0 = 0
        k = np.arange(npix)
    else:
        pix0 = 512
        npix = 1032-pix0
        k = np.arange(pix0,pix0+npix)
    
    norm_array = np.full(np.shape(data_array),np.nan)
    variance = np.zeros(np.shape(data_array))
    
    row_variance = np.zeros(len(data_array[:,0]))

    nwavs = len(lambcens)
    
    for n in range(nwavs):
        
        #--determine pixels contributing to spectral bin
        test_img = make_wave_img(lambcens[n],lambdaMap,band)

        # ascertain that only one pixel is selected per detector column
        outliers = np.where(np.sum(test_img,axis=0)==2)[0]

        for col in outliers:
            Y_ = np.argsort(np.abs(lambdaMap[:,col]-lambcens[n]))[1]
            test_img[Y_,col] = 0

        sel_valid = (test_img==1)

        idx = np.where(sel_valid)[1]
        idy = np.where(sel_valid)[0]

        idy = idy[np.argsort(idx)]
        idx = idx[np.argsort(idx)]

        norm_array[idy,idx-pix0] = data_array[idy,idx-pix0]/np.nansum(data_array[idy,idx-pix0])
        
        for i in range(len(idy)):
            
            row_variance[n] = row_variance[n] + err[idy[i],idx[i]-pix0]**2 + 2*np.sqrt(row_variance[n])*err[idy[i],idx[i]-pix0]

        variance[idy,idx-pix0] = norm_array[idy,idx-pix0]**2 * ((err[idy,idx-pix0]/data_array[idy,idx-pix0])**2 + (np.sqrt(row_variance[n])/np.nansum(data_array[idy,idx-pix0]))**2 - 2*(err[idy,idx-pix0]*np.sqrt(row_variance[n]))/(data_array[idy,idx-pix0]*np.nansum(data_array[idy,idx-pix0])))
        
    return norm_array, variance

def bspline_wrap(xvec,yvec,nbkpts=50,wrapsig_low=3,wrapsig_high=3,wrapiter=10,verbose=False):
    xvec_use=xvec.copy()
    yvec_use=yvec.copy()
    
    for ii in range(0,wrapiter):
        sset,_=bs.iterfit(xvec_use,yvec_use,nbkpts=nbkpts)
        ytemp,_=sset.value(xvec_use)
        diff=yvec_use-ytemp
        rms=np.nanstd(diff)
        rej=np.where((diff < -wrapsig_low*rms)|(diff > wrapsig_high*rms))
        keep=np.where((diff >= -wrapsig_low*rms)&(diff <= wrapsig_high*rms))
        if verbose:
            print('Rejected',len(rej[0]),'Kept',len(keep[0]))
        xvec_use=xvec_use[keep]
        yvec_use=yvec_use[keep]
        
    return sset

def make_PSF(norm_data,K,band,errMap=False,normPSF=False,zeros=False,minim=0.5,minlen=False):

    data_array = np.zeros((K,1024))
    plotval = np.zeros((K,1024))
    err = np.full((K,1024),np.nan)
    
    print('Fitting splines...')
    
    for k in range(K):

        if k%100 == 0:
            print('Working on polynomial {}'.format(k))
        
        data_array[k,:] = np.copy(norm_data[:,k])

        if not zeros:
            indx = np.where(data_array[k,:] < 1e-5)[0][:]
            data_array[k,indx] = np.nan
            
        indx=np.where((np.isfinite(data_array[k,:]) == True))[0][:]
        indx = np.array(indx,int)

        if not minlen:
            if normPSF:
                minlen = 700
            else:
                minlen = 800
        
        if len(indx) > minlen: #500:

            xfit = np.array(np.arange(0,1024),float)

            if errMap:
                
                iternum = 5
                sig = 3
                nbkpts = 20

            elif normPSF:
                if band[0] == '2':
                    iternum = 2 #3
                    sig = 3
                    nbkpts = 50
                else:
                    iternum = 10
                    sig = 3
                    nbkpts = 50
                        
            else:

                if band[0] in ['2','3','4']:
                    
                    if len(indx) > 800:
                        if len(np.where(data_array[k,indx] < minim)[0][:]) > 700:
                            iternum = 2
                            sig = 3
                            nbkpts = 20
                        else:
                            iternum = 2
                            sig = 3
                            nbkpts = 100
                    else:
                        iternum = 2
                        sig = 3
                        nbkpts = 20
                else:

                    if band[1] in ['B','C']:
                        iternum = 2
                        sig = 3
                        if len(indx) > 900:
                            nbkpts = 100
                        else:
                            nbkpts = 20
                    else:
                        iternum = 5
                        sig = 3
                        if len(indx) > 900:
                            nbkpts = 80
                        else:
                            nbkpts = 20
            
            spline_fit = bspline_wrap(xfit[indx],data_array[k,indx],nbkpts=nbkpts,wrapsig_low=sig,wrapsig_high=sig,wrapiter=iternum,verbose=False)

            xplot = np.array(np.arange(0,len(norm_data[:,0])),float)
            plotval[k,:],_ = spline_fit.value(xplot)

            indx = ~np.isnan(data_array[k,:])

            err[k,indx] = (data_array[k,indx] - plotval[k,indx])**2
          
    return plotval,err
    
def resample_PSF(PSF_model,cent_PSF,new_cent,X_map,band,row=False,ypos=0):
    
    from scipy.interpolate import interp1d
    
    if row:
        
        if band[0] in ['1','4']:
            PSF_model_out = np.empty((1,512))
        else:
            PSF_model_out = np.empty((1,520))

        X_map_new = PSF_model['{}-Xmap'.format(ypos)] + (new_cent - cent_PSF[ypos])

        PSF_model_new = interp1d(X_map_new,PSF_model[ypos](PSF_model['{}-Xmap'.format(ypos)]),fill_value='extrapolate')
        PSF_model_out = PSF_model_new(X_map)
        
    else:
        if band[0] in ['1','4']:
            PSF_model_out = np.empty((1024,512))
        else:
            PSF_model_out = np.empty((1024,520))

        for ypos in range(1024):
            X_map_new = PSF_model['{}-Xmap'.format(ypos)] + (new_cent - cent_PSF[ypos])

            PSF_model_new = interp1d(X_map_new,PSF_model[ypos](PSF_model['{}-Xmap'.format(ypos)]),fill_value='extrapolate')
            PSF_model_out[ypos,:] = PSF_model_new(X_map[ypos,:])
        
    return PSF_model_out

def rescale_PSF(PSF_model,sci_data,band,row=False):
    
    if row:
        normsum = np.nansum((sci_data[10:500]))/np.nansum((PSF_model[10:500]))

        PSF_norm = np.copy(PSF_model * normsum)

    else:
        normsum = np.nansum((sci_data[:,10:500]),axis=1)/np.nansum((PSF_model[:,10:500]),axis=1)

        PSF_norm = np.copy(PSF_model)

        for i in range(len(normsum)):
            PSF_norm[i,:] *= normsum[i]
    
    return PSF_norm

def weighted_mean(PSF_model_set,betas,sci_beta,row = False):
    
    if row:
        
        idmin = np.argmin(betas[:,512])
        idmax = np.argmax(betas[:,512])

        ## select models to mean
        if sci_beta < betas[1,512]:

            dbeta = np.abs(betas[1,512] - betas[idmin,512])

            w1 = np.abs(sci_beta - betas[idmin,512])/dbeta
            w2 = np.abs(sci_beta - betas[1,512])/dbeta

            #Flip weights because closest should be weighted highest
            PSF_out = PSF_model_set[idmin,:] * w2 + PSF_model_set[1,:] * w1

        elif sci_beta > betas[1,512]:
            dbeta = np.abs(betas[idmax,512] - betas[1,512])

            w1 = np.abs(sci_beta - betas[1,512])/dbeta
            w2 = np.abs(sci_beta - betas[idmax,512])/dbeta

            #Flip weights because closest should be weighted highest
            PSF_out = PSF_model_set[1,:] * w2 + PSF_model_set[idmax,:] * w1
                  
        elif sci_beta == betas[1,512]:
            
            PSF_out = PSF_model_set[1,:]
        else:
            print('Science falls outside grid, cannot interpolate!')
            PSF_out = PSF_model_set[1,:]
    
    else:
        PSF_out = np.copy(PSF_model_set[1,:,:])

        idmin = np.argmin(betas[:,512])
        idmax = np.argmax(betas[:,512])

        ## select models to mean
        if sci_beta[512] < betas[1,512]:

            dbeta = np.abs(betas[1,:] - betas[idmin,:])

            w1 = np.abs(sci_beta - betas[idmin,:])/dbeta
            w2 = np.abs(sci_beta - betas[1,:])/dbeta

            #Flip weights because closest should be weighted highest

            for i in range(len(sci_beta)):
                PSF_out[i,:] = PSF_model_set[idmin,i,:] * w2[i] + PSF_model_set[1,i,:] * w1[i]

        elif sci_beta[512] > betas[1,512]:
            dbeta = np.abs(betas[idmax,:] - betas[1,:])

            w1 = np.abs(sci_beta - betas[1,:])/dbeta
            w2 = np.abs(sci_beta - betas[idmax,:])/dbeta

            #Flip weights because closest should be weighted highest

            for i in range(len(sci_beta)):
                PSF_out[i,:] = PSF_model_set[1,i,:] * w2[i] + PSF_model_set[idmax,i,:] * w1[i]
        else:
            print('Science falls outside grid, cannot interpolate!')
            
    return PSF_out

def do_PSF_sub_noscale(PSF_model_set,sci_rate,ref_alpha,ref_beta,sci_alpha,sci_beta,fitrange,band,X_map):
    
    PSF_model_resamp = np.empty((3,np.shape(sci_rate)[0],np.shape(sci_rate)[1]))
    
    PSF_model_resamp[0,:,:] = resample_PSF(PSF_model_set[0],ref_alpha[0],sci_alpha,X_map)
    PSF_model_resamp[1,:,:] = resample_PSF(PSF_model_set[1],ref_alpha[1],sci_alpha,X_map)
    PSF_model_resamp[2,:,:] = resample_PSF(PSF_model_set[2],ref_alpha[2],sci_alpha,X_map)

    PSF_mean = weighted_mean(PSF_model_resamp,ref_beta,sci_beta)
    
    PSF_rescale = rescale_PSF(PSF_mean,sci_rate,band)
    
    PSF_sub = np.copy(sci_rate)
    
    if band[0] in ['1','4']:
        PSF_sub = sci_rate - PSF_rescale
        
        sum_out = np.nansum(np.abs(PSF_sub[50:-50])/np.nanmax(sci_rate[50:-50]))
    else:
        PSF_sub = sci_rate - PSF_rescale

        sum_out = np.nansum(np.abs(PSF_sub[50:-50])/np.nanmax(sci_rate[50:-50]))
    
    return PSF_sub, PSF_rescale, sum_out

def do_PSF_sub_scale_long(PSF_model,sci_rate,sci_alpha,sci_beta,band,S,ypos,d2cMaps):
    
    PSF_resamp = point_cloud_proj(PSF_model,sci_alpha,sci_beta,band,d2cMaps)

    PSF_rescale = rescale_PSF(PSF_resamp,sci_rate,band)
    
    PSF_sub = np.copy(sci_rate)
    
    if band[0] in ['1','4']:
        PSF_sub[:,:512] = sci_rate[:,:512] - PSF_rescale * S
        
        sum_out = np.nansum(np.abs(PSF_sub[ypos,50:512-50]))
    else:
        PSF_sub[:,512:] = sci_rate[:,512:] - PSF_rescale * S

        sum_out = np.nansum(np.abs(PSF_sub[ypos,512+50:-50]))
    
    return PSF_sub, PSF_rescale * S, sum_out


def do_PSF_sub_rowscale(PSF_model,sci_rate,S,band):
    
    PSF_sub = np.copy(sci_rate)
    
    if band[0] in ['1','4']:
        PSF_sub[:512] = sci_rate[:512] - PSF_model * S

        sum_out = np.nansum(np.abs(PSF_sub[50:512-50]))
    else:
        PSF_sub[512:] = sci_rate[512:] - PSF_model * S

        sum_out = np.nansum(np.abs(PSF_sub[512+50:-50]))
    
    return PSF_sub, PSF_model * S, sum_out

def do_PSF_sub_fast(PSF_model,sci_rate,dsci_alpha,S,band,X_map):
    
    X_map_new = X_map - dsci_alpha
    PSF_model_new = interp.interp1d(X_map_new,PSF_model(X_map),fill_value='extrapolate')

    PSF_model_resamp = PSF_model_new(X_map)
    
    if band[0] in ['1','4']:
        PSF_sub = sci_rate[:512] - PSF_model_resamp * S
        
        sum_out = np.nansum(np.abs(PSF_sub[30:-30])/np.nanmax(sci_rate[30:-30]))
    else:
        PSF_sub = sci_rate[512:] - PSF_model_resamp * S

        sum_out = np.nansum(np.abs(PSF_sub[30:-30])/np.nanmax(sci_rate[30:-30]))
    
    return PSF_sub, PSF_model_resamp * S, sum_out

class fitClass:

    def __init__(self):
        pass
        
    def gaussian(self,alpha,alpha0,a):
    
        sigma = self.fwhm / np.sqrt(8 * np.log(2))
        
        return a*np.exp(-(alpha-alpha0)**2/(2*sigma**2))

    def gaussian_nocent(self,alpha,a):
    
        sigma = self.fwhm / np.sqrt(8 * np.log(2))
        
        return a*np.exp(-(alpha-self.alpha0)**2/(2*sigma**2))

def get_PSF_set(bandsub,dither,dithnum,sci_alpha,sci_beta,alphaMapCont,pos_abs=True,version='01.05.05'):
    if pos_abs:
        ## Define normalised PSF model

        PSF_model2 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_2_{}.pickle'.format(bandsub,dither,dithnum,version))
        PSF_plot2 = resample_PSF(PSF_model2,PSF_model2['cent_alpha'],sci_alpha,alphaMapCont,bandsub)
    
        #___
    
        PSF_model1 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_1_{}.pickle'.format(bandsub,dither,dithnum,version))
        PSF_plot1 = resample_PSF(PSF_model1,PSF_model1['cent_alpha'],sci_alpha,alphaMapCont,bandsub)
    
        #___
    
        PSF_model3 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_3_{}.pickle'.format(bandsub,dither,dithnum,version))
        PSF_plot3 = resample_PSF(PSF_model3,PSF_model3['cent_alpha'],sci_alpha,alphaMapCont,bandsub)

    else:
        ## Define normalised PSF model
    
        PSF_model1 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_1_{}.pickle'.format(bandsub,dither,dithnum,version))
        ref_alpha = np.ones(len(sci_alpha)) * PSF_model1['cent_alpha']
        PSF_plot1 = resample_PSF(PSF_model1,ref_alpha,sci_alpha,alphaMapCont,bandsub)
    
        #___
    
        PSF_model2 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_2_{}.pickle'.format(bandsub,dither,dithnum,version))
        ref_alpha = np.ones(len(sci_alpha)) * PSF_model2['cent_alpha']
        PSF_plot2 = resample_PSF(PSF_model2,ref_alpha,sci_alpha,alphaMapCont,bandsub)
    
        #___
    
        PSF_model3 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_3_{}.pickle'.format(bandsub,dither,dithnum,version))
        ref_alpha = np.ones(len(sci_alpha)) * PSF_model3['cent_alpha']
        PSF_plot3 = resample_PSF(PSF_model3,ref_alpha,sci_alpha,alphaMapCont,bandsub)

    PSF_model_set = np.empty((3,np.shape(PSF_plot1)[0],np.shape(PSF_plot1)[1]))

    PSF_model_set[0,:,:] = np.copy(PSF_plot1)
    PSF_model_set[1,:,:] = np.copy(PSF_plot2)
    PSF_model_set[2,:,:] = np.copy(PSF_plot3)

    alphas = np.empty((3,len(sci_alpha)))
    alphas[0,:] = np.copy(PSF_model1['cent_alpha'])
    alphas[1,:] = np.copy(PSF_model2['cent_alpha'])
    alphas[2,:] = np.copy(PSF_model3['cent_alpha'])
    
    betas = np.empty((3,len(sci_beta)))
    betas[0,:] = np.copy(PSF_model1['cent_beta'])
    betas[1,:] = np.copy(PSF_model2['cent_beta'])
    betas[2,:] = np.copy(PSF_model3['cent_beta'])
    
    return PSF_model_set,alphas,betas

def extrapolate_beta(PSF_model_set,b,betas,beta,d2cMaps,alphaMapCont,row=False,yrange=0):

    if b in ['1','4']:
        npix = 512
        pix0 = 0
        k = np.arange(npix)
    else:
        pix0 = 512
        npix = 1032-pix0
        k = np.arange(pix0,pix0+npix)
    
    new_PSF = {}

    if row:
        yrange = range(yrange,yrange+1)
    else:
        yrange = range(0,1024)


    for model in [0,1,2]:

        new_PSF[model] = {}
        
        for ypos in yrange:

            nonzero = np.where(d2cMaps['sliceMap'][ypos,k] != 0)
            nslicecent = d2cMaps['sliceMap'][ypos,k][nonzero][np.nanargmin(np.abs(d2cMaps['betaMap'][ypos,k][nonzero] - beta))]
            slicecent = d2cMaps['sliceMap'][ypos,k][nonzero][np.nanargmin(np.abs(d2cMaps['betaMap'][ypos,k][nonzero] - (betas[1,512])))]
            npix_slice = len(np.where(d2cMaps['sliceMap'][ypos,k][nonzero] == slicecent)[0][:])
            dslice = nslicecent - slicecent

            new_PSF[model][ypos] = np.zeros(np.shape(PSF_model_set[model][ypos](PSF_model_set[model]['{}-Xmap'.format(ypos)])))
            new_PSF[model]['{}-Xmap'.format(ypos)] = np.copy(PSF_model_set[model]['{}-Xmap'.format(ypos)])
            
            for i in (set(list(d2cMaps['sliceMap'][ypos,k]))):
                if (i > 0):
                    if (i + dslice < list(set(list(d2cMaps['sliceMap'][ypos,k])))[-1]) and (i + dslice > 100):
                        idslice = np.where(d2cMaps['sliceMap'][ypos,k] == i)[0][:]
                        idslicen = np.where(d2cMaps['sliceMap'][ypos,k] == i + dslice)[0][:]
                        
                        idx = np.where((PSF_model_set[model]['{}-Xmap'.format(ypos)] > np.nanmin(alphaMapCont[ypos,idslice])) & (PSF_model_set[model]['{}-Xmap'.format(ypos)] < np.nanmax(alphaMapCont[ypos,idslice])))[0][:]
                        idxn = np.where((PSF_model_set[model]['{}-Xmap'.format(ypos)] > np.nanmin(alphaMapCont[ypos,idslicen])) & (PSF_model_set[model]['{}-Xmap'.format(ypos)] < np.nanmax(alphaMapCont[ypos,idslicen])))[0][:]
                        new_PSF[model][ypos][idxn] = np.copy(PSF_model_set[model][ypos](PSF_model_set[model]['{}-Xmap'.format(ypos)][idx[0]:idx[0]+len(idxn)]))
        
                    else:
                        idslice = np.where(d2cMaps['sliceMap'][ypos,k] == i)[0][:]
                        nextslice = d2cMaps['sliceMap'][ypos,k[idslice[0]-int(npix_slice/2)]]
                        
                        if ((nextslice + dslice) < np.nanmax(d2cMaps['sliceMap'][ypos,k][nonzero])) & ((nextslice + dslice) > np.nanmin(d2cMaps['sliceMap'][ypos,k][nonzero])):
                            idslicen = np.where(d2cMaps['sliceMap'][ypos,k] == nextslice + dslice)[0][:]
                            nextslice = d2cMaps['sliceMap'][ypos,k[idslicen[-1]+int(npix_slice/2)]]

                            if nextslice > 0:
                                idslicen = np.where(d2cMaps['sliceMap'][ypos,k] == nextslice)[0][:]
                
                                idx = np.where((PSF_model_set[model]['{}-Xmap'.format(ypos)] > np.nanmin(alphaMapCont[ypos,idslice])) & (PSF_model_set[model]['{}-Xmap'.format(ypos)] < np.nanmax(alphaMapCont[ypos,idslice])))[0][:]
                                idxn = np.where((PSF_model_set[model]['{}-Xmap'.format(ypos)] > np.nanmin(alphaMapCont[ypos,idslicen])) & (PSF_model_set[model]['{}-Xmap'.format(ypos)] < np.nanmax(alphaMapCont[ypos,idslicen])))[0][:]

                                new_PSF[model][ypos][idxn] = np.copy(PSF_model_set[model][ypos](PSF_model_set[model]['{}-Xmap'.format(ypos)][idx[0]:idx[0]+len(idxn)]))
                
            zeros = np.where(new_PSF[model][ypos][:] != 0)[0][:]
            
            if len(zeros) > 100:
                new_PSF[model][ypos] = interp.interp1d(new_PSF[model]['{}-Xmap'.format(ypos)][zeros],new_PSF[model][ypos][zeros],fill_value=(0,0),bounds_error=False) #(new_PSF[model]['{}-Xmap'.format(ypos)])

    return new_PSF

def opt_beta(PSF_model,sci_rate_full,ref_alpha,ref_beta,dsci_alpha,sci_beta,S,band,X_map,photMap,d2cMaps,fitpeak=1):
    
    PSF_model_resamp = np.empty((3,len(X_map[512,:])))
    PSF_model_med = np.empty((3,len(X_map[512,:])))
    
    PSF_model_set = extrapolate_beta(PSF_model,band[0],ref_beta,sci_beta,d2cMaps,X_map,row=True,yrange=512)
    PSF_model_resamp[1,:] =  resample_PSF(PSF_model_set[1],ref_alpha[1,:],dsci_alpha,X_map[512,:],band,row=True,ypos=512)
    PSF_model_med[1,:] = np.copy(PSF_model_resamp[1,:])
    
    sci_rate = np.nanmedian((sci_rate_full),axis=0)

    PSF_model_med[1,:] = PSF_model_med[1,:] * np.nanmedian((photMap),axis=0)

    nonneg = ~(sci_rate < 0) & ~np.isnan(PSF_model_med[1,:])
    
    ## Compare to PSF-model values
    idcomp = find_peaks(sci_rate[nonneg],fitpeak,2,False)

    PSF_rescale = (PSF_model_med[1,:] * np.nansum(sci_rate[nonneg][idcomp])) / np.nansum((PSF_model_med[1,nonneg])[idcomp])

    PSF_sub = sci_rate - (PSF_rescale * S)
    PSF_sub[PSF_sub<0] *= 5

    sum_out = np.nansum(np.abs(PSF_sub[nonneg][idcomp]))
    
    return sum_out

def clip(data,sigma_smooth,sigma_clip,itern=1,pol_clip=False):
    
    smooth = np.copy(data)
    output = np.copy(data)
    
    for i in range(itern):
        if pol_clip:
            smooth = np.polyval(np.polyfit(np.arange(len(output))[~np.isnan(output)],output[~np.isnan(output)],deg=2),np.arange(len(output)))
        else:
            smooth[~np.isnan(output)] = gaussian_filter1d(output[~np.isnan(output)],sigma_smooth,mode='nearest')
        std = np.nanstd(output-smooth)
        idclip = np.where(np.abs(output-smooth) > sigma_clip * std)
        output[idclip] = np.nan

    return output

def det_aperture(cent,apsize,fwhm,alphas,betacent=None,betas=None):

    if betacent is None:
        cent = np.abs(cent)
        alphas = np.abs(alphas)
    
        alphamin = cent - (apsize*fwhm)/2
        alphamax = cent + (apsize*fwhm)/2
    
        # alphas = np.sort(alphas_in)
    
        dalphas = alphas[1:] - alphas[:-1]
    
        alphaleft = np.copy(alphas)
        alpharight = np.copy(alphas)
        
        alphaleft[1:] = alphas[1:] - dalphas/2
        alphaleft[0] = alphas[0] - dalphas[0]/2
        alpharight[:-1] = alphaleft[1:] #alphas[:-1] + dalphas/2
        alpharight[-1] = alphaleft[-1] + dalphas[-1]

    if betacent is not None:

        cents_map = np.sqrt((alphas-cent)**2 + (betas-betacent)**2)
        mask = (cents_map<(apsize*fwhm)/2)
        
        pixweights = np.zeros((len(betas)))
        pixweights[mask] = 1
    else:
        try:
            includepix = np.where((alpharight > alphamin) & (alphaleft < alphamax))[0][:]
        
            weight_l = np.abs((alpharight[includepix[0]] - alphamin)/(alpharight[includepix[0]] - alphaleft[includepix[0]]))
            weight_r = np.abs((alphaleft[includepix[-1]] - alphamax)/(alpharight[includepix[-1]] - alphaleft[includepix[-1]]))
            
            pixweights = np.zeros((len(alphas)))
            pixweights[includepix[1:-1]] = 1
            pixweights[includepix[0]] = weight_l
            pixweights[includepix[-1]] = weight_r
            
        except:
            pixweights = np.nan

    return pixweights

def get_PSF_set_int(bandsub,dither,dithnum,version = '01.05.05',beta = np.nan,d2cMaps=0,alphaMapCont=0):

    ## Define normalised PSF model
    
    PSF_model1 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_1_{}.pickle'.format(bandsub,dither,dithnum,version))

    #___

    PSF_model2 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_2_{}.pickle'.format(bandsub,dither,dithnum,version))

    #___

    PSF_model3 = pd.read_pickle('./references/PSF/MIRIFUSHORT_{}_detectorPSF_{}{}_3_{}.pickle'.format(bandsub,dither,dithnum,version))

    PSF_model_set = {}
    PSF_model_set[0] = PSF_model1
    PSF_model_set[1] = PSF_model2
    PSF_model_set[2] = PSF_model3

    alphas = np.empty((3,1024))
    alphas[0,:] = np.copy(PSF_model1['cent_alpha'])
    alphas[1,:] = np.copy(PSF_model2['cent_alpha'])
    alphas[2,:] = np.copy(PSF_model3['cent_alpha'])
    
    betas = np.empty((3,1024))
    betas[0,:] = np.copy(PSF_model1['cent_beta'])
    betas[1,:] = np.copy(PSF_model2['cent_beta'])
    betas[2,:] = np.copy(PSF_model3['cent_beta'])

    if beta is not np.nan:
        PSF_model_set = extrapolate_beta(PSF_model_set,bandsub[0],betas,beta,d2cMaps,alphaMapCont)
    
    return PSF_model_set,alphas,betas

def do_PSF_sub_fitall_fast(PSF_model_set,sci_rate,ref_alpha,ref_beta,dsci_alpha,sci_beta,S,ped,slope,band,X_map,photMap,n,idn,fitpeak=1,norm=True):
    
    PSF_model_resamp = np.empty((3,np.shape(X_map)[0]))

    PSF_model_resamp[0,:] =  resample_PSF(PSF_model_set[0],ref_alpha[0,:],dsci_alpha,X_map,band,row=True,ypos=n)
    
    PSF_model_resamp[1,:] =  resample_PSF(PSF_model_set[1],ref_alpha[1,:],dsci_alpha,X_map,band,row=True,ypos=n)
    
    PSF_model_resamp[2,:] =  resample_PSF(PSF_model_set[2],ref_alpha[2,:],dsci_alpha,X_map,band,row=True,ypos=n)
    
    PSF_mean = weighted_mean(PSF_model_resamp,ref_beta,sci_beta,row=True)
    
    PSF_mean = PSF_mean * photMap
    nonneg = ~(sci_rate < 0) & ~np.isnan(PSF_mean[idn])
    
    ## Compare to PSF-model values
    idcomp = find_peaks(PSF_mean[idn][nonneg],fitpeak,1,False)

    PSF_rescale = (PSF_mean * np.nansum(sci_rate[nonneg][idcomp]-(ped+slope*np.arange(len(PSF_mean))[idn][nonneg][idcomp])) / np.nansum((PSF_mean[idn][nonneg])[idcomp]))

    PSF_sub = np.copy(sci_rate)
    
    PSF_sub = sci_rate - (PSF_rescale[idn] * S + (ped+slope*np.arange(len(PSF_mean))[idn]))

    if norm:
        sum_out = np.nansum(np.abs(PSF_sub[nonneg][idcomp]/(sci_rate[nonneg][idcomp])))
    else:
        sum_out = np.nansum(np.abs(PSF_sub[nonneg][idcomp]/np.nanmax(sci_rate[nonneg][idcomp])))
    
    return PSF_sub, PSF_rescale * S + (ped+slope*np.arange(len(PSF_mean))), sum_out
    
def get_PSF_long(brun):
    
    if brun[0] in ['1','2']:
        det = 'SHORT'
    else:
        det = 'LONG'

    if brun[1] == 'A':
        subband = 'SHORT'
    elif brun[1] == 'B':
        subband = 'MEDIUM'
    else:
        subband = 'LONG'
        
    PSF_model = pd.read_pickle('./output/PSF/MIRIFULONG_{}_detectorPSF_01.00.00.pickle'.format(bandsub,dither,dithnum))
    
    return PSF_model

def pca_bkg_sub_short(PSF_mean,sci_data,sci_alpha,sci_beta,band,d2cMaps,photomMap,itnum=1):
    ## Apply PCA in loop

    PSF_mean = PSF_mean/photomMap
    sci_data = sci_data/photomMap

    new_fit_rate = np.copy(sci_data-PSF_mean)
    
    band2 = str(int(band[0]) + 1) + band[1]
    
    lambcens = {}
    
    lambmin = np.nanmin(d2cMaps[band]['lambdaMap'][np.where(d2cMaps[band]['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps[band]['lambdaMap']) # micron
    lambcens[band] = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    
    lambmin = np.nanmin(d2cMaps[band2]['lambdaMap'][np.where(d2cMaps[band2]['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps[band2]['lambdaMap']) # micron
    lambcens[band2] = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    
    if band[0] in ['1','2']:
        ifu = 'SHORT'
    else:
        ifu = 'LONG'
        
    if band[1] == 'A':
        subband = 'SHORT'
    elif band[1] == 'B':
        subband = 'MEDIUM'
    else:
        subband = 'LONG'
    
    bkgV = np.loadtxt('./output/BACKGROUND/MIRIFU{}_{}_backgroundPCA.dat'.format(ifu,subband))

    ## bkg + PSF sub iteration
    for i in range(itnum):
        #Mask PSF in data

        border_mask = np.ones(new_fit_rate.shape)
        border_mask[:20,:]=0
        border_mask[-20:,:]=0
        border_mask[:,-20:]=0
        border_mask[:,:20]=0

        ignore = np.isnan(new_fit_rate)

        radius_map = np.sqrt((d2cMaps[band]['alphaMap']-sci_alpha[band])**2 + (d2cMaps[band]['betaMap']-sci_beta[band])**2)
        mask_L = (radius_map>4*0.31 ) * d2cMaps['1A']["sliceMap"]

        radius_map = np.sqrt((d2cMaps[band2]['alphaMap']-sci_alpha[band2])**2 + (d2cMaps[band2]['betaMap']-sci_beta[band2])**2)
        mask_R = (radius_map>2*0.5 ) * d2cMaps[band2]["sliceMap"]

        mask  = ~ignore * border_mask * mask_L * (mask_L * new_fit_rate < np.nanpercentile(mask_L * new_fit_rate, 80))
        maskR = ~ignore * border_mask * mask_R * (mask_R * new_fit_rate < np.nanpercentile(mask_R * new_fit_rate, 80))
        mask[:,512:] = maskR[:,512:]
        mask = (mask !=0)

        masked_frame = (mask * new_fit_rate).flatten()
        maskedV = mask.flatten() * bkgV

        noNaN = ~np.isnan(masked_frame)

        # Compute proj coeff using masked PC and masked frame.
        coeff = np.dot(masked_frame[noNaN], maskedV[:,noNaN].T)

        # Apply coeff to unmasked PC to obtain bkg model
        bkgPCA = np.dot(coeff.T, bkgV)

        # Subtract
        pca_residu = (new_fit_rate.flatten() - bkgPCA)

        # Reshape
        bkg = bkgPCA.reshape(new_fit_rate.shape)
        pca_residu = pca_residu.reshape(new_fit_rate.shape)

        new_rate = sci_data - bkg

        PSF_rescale = np.copy(PSF_mean)

        PSF_rescale[:,:512] = rescale_PSF(PSF_mean,new_rate,band)[:,:512]
        PSF_rescale[:,512:] = rescale_PSF(PSF_mean,new_rate,band2)[:,512:]

        PSF_sub = np.copy(pca_residu)
        PSF_sub = sci_data - PSF_rescale - bkg

        new_fit_rate = np.copy(PSF_sub + bkg)

    
    return (sci_data - bkg) * photomMap, bkg * photomMap


## Point cloud methods
def point_cloud_proj(PSF_model,sci_alpha,sci_beta,brun,d2cMaps):

    lambmin = np.nanmin(d2cMaps['lambdaMap'][np.where(d2cMaps['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps['lambdaMap']) # micron
    lambcens = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)

    projected = np.zeros((1024,1032))

    for l in range(1024):
        
        wvl = lambcens[l]
        idl = np.where((d2cMaps['lambdaMap'] < lambcens[l+1]) & (d2cMaps['lambdaMap'] > lambcens[l]))
        
        newgrid = np.copy(PSF_model[l]['grid'])
        newgrid[0,:] += sci_alpha
        newgrid[1,:] += sci_beta
        
        PSF_newgrid = interp.LinearNDInterpolator(newgrid,PSF_model[l]['interp'](PSF_model[l]['grid'][0,:],PSF_model[l]['grid'][1,:]), fill_value=np.nan, rescale=False)
        
        alphagrid = d2cMaps['alphaMap'][idl]
        betagrid = d2cMaps['alphaMap'][idl]
        
        projected[idl] = PSF_newgrid[l]['interp'](alphagrid,betagrid)
        
    return projected

def pca_bkg_sub_long(PSF_mean,sci_data,sci_alpha,sci_beta,band,d2cMaps):
    ## Apply PCA in loop

    new_fit_rate = np.copy(sci_data)
    
    band2 = str(int(band[0]) + 1) + band[1]
    
    lambcens = {}
    
    lambmin = np.nanmin(d2cMaps[band]['lambdaMap'][np.where(d2cMaps[band]['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps[band]['lambdaMap']) # micron
    lambcens[band] = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    
    lambmin = np.nanmin(d2cMaps[band2]['lambdaMap'][np.where(d2cMaps[band2]['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps[band2]['lambdaMap']) # micron
    lambcens[band2] = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    
    if band[0] in ['1','2']:
        ifu = 'SHORT'
    else:
        ifu = 'LONG'
        
    if band[1] == 'A':
        subband = 'SHORT'
    elif band[1] == 'B':
        subband = 'MEDIUM'
    else:
        subband = 'LONG'
    
    bkgV = np.loadtxt('./output/BACKGROUND/MIRIFU{}_{}_backgroundPCA.dat'.format(ifu,subband))

    ## bkg + PSF sub iteration
    for i in range(3):
        #Mask PSF in data

        border_mask = np.ones(new_fit_rate.shape)
        border_mask[:20,:]=0
        border_mask[-20:,:]=0
        border_mask[:,-20:]=0
        border_mask[:,:20]=0

        ignore = np.isnan(new_fit_rate)

        radius_map = np.sqrt((d2cMaps[band]['alphaMap']-sci_alpha[band](lambcens[band][512]))**2 + (d2cMaps[band]['betaMap']-sci_beta[band](lambcens[band][512]))**2)
        mask_L = (radius_map>2*0.6 ) * d2cMaps['1A']["sliceMap"]

        radius_map = np.sqrt((d2cMaps[band2]['alphaMap']-sci_alpha[band2](lambcens[band2][512]))**2 + (d2cMaps[band2]['betaMap']-sci_beta[band2](lambcens[band2][512]))**2)
        mask_R = (radius_map>2*0.8 ) * d2cMaps[band2]["sliceMap"]

        mask  = ~ignore * border_mask * mask_L * (mask_L * new_fit_rate < np.nanpercentile(mask_L * new_fit_rate, 80))
        maskR = ~ignore * border_mask * mask_R * (mask_R * new_fit_rate < np.nanpercentile(mask_R * new_fit_rate, 80))
        mask[:,512:] = maskR[:,512:]
        mask = (mask !=0)

        masked_frame = (mask * new_fit_rate).flatten()
        maskedV = mask.flatten() * bkgV

        noNaN = ~np.isnan(masked_frame)

        # Compute proj coeff using masked PC and masked frame.
        coeff = np.dot(masked_frame[noNaN], maskedV[:,noNaN].T)

        # Apply coeff to unmasked PC to obtain bkg model
        bkgPCA = np.dot(coeff.T, bkgV)

        # Subtract
        pca_residu = (new_fit_rate.flatten() - bkgPCA)

        # Reshape
        bkg = bkgPCA.reshape(new_fit_rate.shape)
        pca_residu = pca_residu.reshape(new_fit_rate.shape)

        new_rate = sci_data - bkg
    
        PSF_rescale = np.copy(PSF_mean)

#         PSF_rescale[:,:512] = rescale_PSF(PSF_mean,new_rate,band2)
        PSF_rescale[:,512:] = rescale_PSF(PSF_mean,new_rate,band)

        PSF_sub = np.copy(pca_residu)
        PSF_sub = sci_data - PSF_rescale - bkg

        new_fit_rate = np.copy(PSF_sub + bkg)
    
    return new_fit_rate

def cont_lambda(band,dist_ver,X_map,lambdaMap):

    if band[0] in ['1','4']:
        npix = 512
        pix0 = 0
        k = np.arange(npix)
    else:
        pix0 = 512
        npix = 1032-pix0
        k = np.arange(pix0,pix0+npix)
    
    lambmin = np.nanmin(lambdaMap[np.where(lambdaMap!=0)]) # micron
    lambmax = np.nanmax(lambdaMap) # micron
    lambcens = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)

    for ypos in range(len(lambdaMap[:,0])):
        lambdaMap[ypos,lambdaMap[ypos,:] == 0] = np.nan
        nanids = ~np.isnan(lambdaMap[ypos,:])
        interp_obj = interp.interp1d(np.arange(1032)[nanids],lambdaMap[ypos,:][nanids],fill_value='extrapolate')
        lambdaMap[ypos,k] = interp_obj(k)

    lambdaMap[np.isnan(lambdaMap)] = 0

    return lambdaMap,lambcens

def find_peaks(PSF,npeaks=3,radius=2,exclude_peaks=False):

    optidlist = []
    
    for i in range(npeaks):
        idcomp = np.nanargmax(PSF)
        PSF[idcomp-10:idcomp+10] = np.nan
    
        if len(PSF) < (idcomp + radius):
            if exclude_peaks:
                optidlist.extend(np.arange(idcomp-radius,idcomp+1))
                optidlist.extend(np.arange(idcomp+1,len(PSF)))
            else:
                optidlist.extend(np.arange(idcomp-radius,len(PSF)))
        elif (idcomp - radius) < 0:
            if exclude_peaks:
                optidlist.extend(np.arange(0,idcomp+1))
                optidlist.extend(np.arange(idcomp+1,idcomp+radius))
            else:
                optidlist.extend(np.arange(0,idcomp+radius))
        else:
            if exclude_peaks:
                optidlist.extend(np.arange(idcomp-radius,idcomp+1))
                optidlist.extend(np.arange(idcomp+1,idcomp+radius))
            else:
                optidlist.extend(np.arange(idcomp-radius,idcomp+radius))
    
    return optidlist

def find_file(band,crdsDir,file='readnoise'):
    
    startdate = datetime.datetime(1900,1,1)

    if band[1] == 'A':
        subband = 'SHORT'
    elif band[1] == 'B':
        subband = 'MEDIUM'
    else:
        subband = 'LONG'
    
    if band[0] in ['1','2']:
        det = 'MIRIFUSHORT'
    elif band[0] in ['3','4']:
        det = 'MIRIFULONG'

    try:
        for filename in listdir(crdsDir+'references/jwst/miri/'):
            if filename.startswith('jwst_miri_{}'.format(file)):
                if file == 'readnoise':
                    hdu = fits.open(crdsDir+'references/jwst/miri/'+filename)
                    datestr = (hdu[0].header['DATE']).split('T')
                    date = datetime.datetime.strptime(datestr[0]+' '+datestr[1],'%Y-%m-%d %H:%M:%S.%f')
                    refdet = hdu[0].header['DETECTOR']
                    hdu.close()
                    
                    if refdet == det:
                        if date > startdate:
                            reffile = filename
                            startdate = date
    
                else:
                    hdu = fits.open(crdsDir+'references/jwst/miri/'+filename)
                    datestr = (hdu[0].header['DATE']).split('T')
                    date = datetime.datetime.strptime(datestr[0]+' '+datestr[1],'%Y-%m-%d %H:%M:%S.%f')
                    refdet = hdu[0].header['DETECTOR']
                    refband = hdu[0].header['BAND']
                    hdu.close()
                    
                    if refdet == det:
                        if refband == 'N/A':
                            if date > startdate:
                                reffile = filename
                                startdate = date
                        elif refband == subband:
                            if date > startdate:
                                reffile = filename
                                startdate = date
        
        ref_file = crdsDir+'references/jwst/miri/'+reffile
        
    except:
        for filename in listdir(crdsDir+'references/jwst/miri/'):
            if filename.startswith('jwst_miri_{}'.format(file)):
                if file == 'gain':
                    hdu = fits.open(crdsDir+'references/jwst/miri/'+filename)
                    datestr = (hdu[0].header['DATE']).split('T')
                    date = datetime.datetime.strptime(datestr[0]+' '+datestr[1],'%Y-%m-%d %H:%M:%S.%f')
                    refdet = hdu[0].header['DETECTOR']
                    refband = hdu[0].header['BAND']
                    hdu.close()
                    
                    if refdet == det:
                        if refband == 'N/A':
                            if date > startdate:
                                reffile = filename
                                startdate = date
                        elif subband in refband:
                            if date > startdate:
                                reffile = filename
                                startdate = date
        
        ref_file = crdsDir+'references/jwst/miri/'+reffile
    
    return ref_file

def opt_extract(bandlist,rateDir,calDir,outDir,dist_ver,psfVer,bright=False,indith=True,PSFsub=False,calSubDir=None,offa=None,offb=None,peakfitlist=None,do_apcorr=False,apsize=1.2):

    ## Gather files to run
    test_list = {}
    cal_list = {}

    if PSFsub:
        sub_list = {}

    if not path.exists(outDir):
        makedirs(outDir)
    
    for b in bandlist:
        print(b)
        
        test_list[b] = []
        cal_list[b] = []

        if PSFsub:
            sub_list[b] = []
        
        if b[0] in ['1','2']:
            ifu = 'short'
        else:
            ifu = 'long'
        
        if b[1] == 'A':
            subband = 'SHORT'
        elif b[1] == 'B':
            subband = 'MEDIUM'
        else:
            subband = 'LONG'
        
        for file in listdir(rateDir):
            if file.startswith('jw') and file.endswith('{}_rate.fits'.format(ifu)):
                hdu = fits.open(rateDir + file)
                sband = hdu[0].header['BAND']
                hdu.close()
                if sband == subband:
                    test_list[b].append(rateDir + file)
                    print('Using',file)
    
        for file in listdir(calDir):
            if file.startswith('jw') and file.endswith('{}_cal.fits'.format(ifu)):
                hdu = fits.open(calDir + file)
                sband = hdu[0].header['BAND']
                hdu.close()
                if sband == subband:
                    cal_list[b].append(calDir + file)
                    print('Using',file)

        if PSFsub:
            for file in listdir(calSubDir):
                if not file.startswith('.') and file.endswith('cal.fits'):
                    hdu = fits.open(calSubDir + file)
                    sband = hdu[0].header['BAND']
                    det = hdu[0].header['DETECTOR']
                    hdu.close()
                    if (sband == subband) and (det[6:] == ifu.upper()):
                        sub_list[b].append(calSubDir + file)
                        print('Using',file)

    ## Get maps
    d2cMaps = {}

    for b in bandlist:
        d2cMaps[b] = d2cMapping(b,'./references/DISTORTION/',slice_transmission='10pc',fileversion = "flt9")

    ## Find where observation is located
    alpha = {}
    beta = {}
    
    alphadic = {}
    betadic = {}
    lambcens = {}
    
    for b in bandlist:
        lambmin = np.nanmin(d2cMaps[b]['lambdaMap'][np.where(d2cMaps[b]['lambdaMap']!=0)]) # micron
        lambmax = np.nanmax(d2cMaps[b]['lambdaMap']) # micron
        lambcens[b] = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
        lambfwhms = np.ones(len(lambcens[b]))*(2*(lambmax-lambmin)/1024.)
        
        alpha[b] = {}
        beta[b] = {}
        
        for i in range(len(cal_list[b])):
            
            test_rate = fits.getdata(test_list[b][i])
            dith = int((test_list[b][i].split('/')[-1]).split('_')[2][-1])
    
            popt_alpha, popt_beta = centroiding(test_rate,d2cMaps[b],b)
    
            alpha[b][dith] = np.nanmedian(popt_alpha(lambcens[b]))
            beta[b][dith] = np.nanmedian(popt_beta(lambcens[b]))

    ## Start optimal extraction
    spec_optimal = {}
    sub_save = {}
    
    for b in bandlist:
        print('Working on band {}'.format(b))

        if PSFsub:
            peakfit = peakfitlist[int(b[0])-1]
        
        if b[0] in ['1','4']:
            npix = 512
            pix0 = 0
            k = np.arange(npix)
        else:
            pix0 = 512
            npix = 1032-pix0
            k = np.arange(pix0,pix0+npix)
            
        if b[0] in ['1','4']:
            idxrange = [0,512]
        else:
            idxrange = [512,1032]
            
        if b[0] in ['1','2']:
            ifu = 'SHORT'
        else:
            ifu = 'LONG'
                
        if b[1] == 'A':
            bandsub = 'SHORT'
        elif b[1] == 'B':
            bandsub = 'MEDIUM'
        else:
            bandsub = 'LONG'
        
        lambdaMap = np.copy(d2cMaps[b]['lambdaMap'])
    
        alphaMapCont = _create_incremental_alphaMap(b, d2cMaps[b])
        lambdaMap,_ = cont_lambda(b,dist_ver,alphaMapCont,lambdaMap)
    
        nwavs = len(lambcens[b])
        lambdas = lambdaMap[np.nonzero(lambdaMap)].flatten()
        
        # photMapCont = fits.getdata('./output/PHOTOM/photom_continuous_{}_{}.fits'.format(ifu,bandsub))
        photMapCont = fits.getdata('./references/PHOTOM/photom_continuous_{}_{}_v2.fits'.format(ifu,bandsub))
        pixelMapCont = fits.getdata('./references/PIXEL/fflat_continuous_{}_{}.fits'.format(ifu,bandsub))
        pixSize = (fits.open('./references/PHOTOM/photom_continuous_{}_{}_v2.fits'.format(ifu,bandsub))[5]).data
    
        photomMapFull = photMapCont * pixSize / pixelMapCont
        photomMapCont = photMapCont[:,idxrange[0]:idxrange[-1]] * pixSize[:,idxrange[0]:idxrange[-1]] / pixelMapCont[:,idxrange[0]:idxrange[-1]]
        
        spec_optimal[b] = {}
        sub_save[b] = {}
        
        for i in range(len(cal_list[b])):
    
            acoord = []
            bcoord = []
    
            if PSFsub:
                print('Working on file {}/{}'.format(i+1,len(sub_list[b])))
                print('Dither number:',(int((cal_list[b][i].split('/')[-1]).split('_')[2][-1])))
            
                test = fits.open(sub_list[b][i])
                test_cal = test['SCI'].data * pixSize
                dither = test[0].header['DITHDIRC']
                test.close()
            else:
                print('Working on file {}/{}'.format(i+1,len(cal_list[b])))
                print('Dither number:',(int((cal_list[b][i].split('/')[-1]).split('_')[2][-1])))
            
                test = fits.open(cal_list[b][i])
                test_cal = test['SCI'].data * pixSize
                dither = test[0].header['DITHDIRC']
                test.close()
                
            print(dither)
            
            err_hdu = fits.open(test_list[b][i])
            if bright:
                test_err = np.sqrt(err_hdu['VAR_POISSON'].data) #err_hdu['VAR_RNOISE'].data + 
            else:
                test_err = err_hdu['ERR'].data
            var_read = err_hdu['VAR_RNOISE'].data
            var_tot = np.sqrt(err_hdu['VAR_RNOISE'].data + err_hdu['VAR_POISSON'].data)[:,k]
            test_err_uncal = err_hdu['ERR'].data
            err_hdu.close
    
            print('Smoothing error map...')
    
            test_err_band = np.copy(test_err[:,idxrange[0]:idxrange[-1]])
            test_err_band,_ = make_PSF(test_err_band,npix,b,errMap=True)
            test_err_band = np.transpose(test_err_band)
            test_err_band = test_err_band
            test_err_band[test_err_band < 0] = 10
    
            test_cal[:,:20] = 0
            test_cal_band = np.copy(test_cal[:,idxrange[0]:idxrange[-1]])
            
            dithnum = int((cal_list[b][i].split('/')[-1]).split('_')[2][-1])
            
            sci_pos = np.empty((2,len(lambcens[b])))
            
            if indith:
                sci_pos[0,:] = alpha[b][dithnum]
                sci_pos[1,:] = beta[b][dithnum]
            else:
                sci_pos[0,:] = alpha[b][dithnum] + offa
                sci_pos[1,:] = beta[b][dithnum]
            
            if b[0] in ['1','2']:
    
                if indith:
                    PSF_model_set,alphas,betas = get_PSF_set_int(b[0]+bandsub,dither,dithnum,version = psfVer)
    
                else:
    
                    med_band = np.nanmedian((test_cal_band),axis=0)
    
                    if dithnum in [0]: #[3]: #,1,2]:
                        
                        beta_slice = d2cMaps[b]['betaMap'][512,10+np.nanargmax(med_band[10:-10])]
                        offb = beta_slice - sci_pos[1,512]
        
                        alpha_slice = d2cMaps[b]['alphaMap'][512,10+np.nanargmax(med_band[10:-10])]
                        offa = alpha_slice - sci_pos[0,512]
        
                        print(offa,offb)
        
                        sci_pos[0,:] = alpha[b][dithnum] + offa
                        
    
                    PSF_model_set,alphas,betas = get_PSF_set_int(b[0]+bandsub,dither,dithnum,version = psfVer,beta = sci_pos[1,512] + offb,d2cMaps=d2cMaps[b],alphaMapCont=alphaMapCont)
                    
                    idfit = find_peaks(PSF_model_set[1][512](alphaMapCont[512,:]),npeaks=peakfit,radius=2,exclude_peaks=False) #med_band
                    idmax = find_peaks(PSF_model_set[1][512](alphaMapCont[512,:]),npeaks=1,radius=2,exclude_peaks=False)
                    slices = list(set(list(d2cMaps[b]['sliceMap'][512,k][idfit])))
                    slicemax = (d2cMaps[b]['sliceMap'][512,k][idfit])[0]
                    slices.remove(slicemax)
                    slices.insert(0,slicemax)
                    
                    alphacent = d2cMaps[b]['alphaMap'][512,10+np.nanargmax(med_band[10:-10])]
                    print(slices)
    
                    idarray = np.zeros((np.shape(test_cal_band)))
    
                    maxn = np.zeros((peakfit),dtype=int)
                    
                    for p in range(peakfit):
    
                        idtrace = funcs.detpixel_trace(b,d2cMaps[b],sliceID=slices[p] - int(b[0])*100,alpha_pos=alphacent)
                        idtrace[1][:] = idtrace[1][:] - pix0
    
                        for y in range(len(idtrace[0])):
                            idarray[y,idtrace[1][y]-6:idtrace[1][y]+7] = 1
    
                        maxn[p] = idtrace[1][512]
    
                    test_fit = np.copy(test_cal_band)
                    test_fit[idarray == 0] = np.nan
    
                    fwhm = np.loadtxt('./references/PSF/MIRIFU{}_{}{}_FWHM.dat'.format(ifu,b[0],bandsub))
                    fwhm = interp.interp1d(fwhm[:,0],fwhm[:,1],fill_value='extrapolate')
                    fwhm = interp.interp1d(lambcens[b][:],np.polyval(np.polyfit(lambcens[b][:],fwhm(lambcens[b][:]),deg=1),lambcens[b][:]))
    
                    cents = {}
                    peaks = {}
                    smoothcents = {}
                    smoothpeaks = {}
    
                    fit_PSF = np.zeros((np.shape(test_cal_band)))
                    weightpix = np.copy(fit_PSF)
    
                    for p in range(peakfit):
                        
                        if p == 0:
                            itrange = range(0,2)
                        else:
                            itrange = range(1,2)
                            
                        for it in itrange:
                            
                            if it == 0:
                                cents[slices[p]] = np.full((nwavs),np.nan)
                            peaks[slices[p]] = np.full((nwavs),np.nan)
        
                            if it == 0:
                                smoothcents[slices[p]] = np.full((nwavs),np.nan)
                            elif p != 0:
                                smoothcents[slices[p]] = np.copy(smoothcents[slices[0]]) - alphaMapCont[512,maxn[0]] + alphaMapCont[512,maxn[p]]
                                cents[slices[p]] = np.copy(smoothcents[slices[p]])
                            
                            sliceMap = np.zeros(np.shape(test_cal_band))
                            sliceMap[d2cMaps[b]['sliceMap'][:,k] == slices[p]] = 1
                            
                            for n in range(0,nwavs,4):
    
                                meddata = np.full(np.shape(test_cal_band),np.nan)
                                medalpha = np.full(np.shape(test_cal_band),np.nan)
    
                                c = ['tab:blue','tab:orange','tab:green','tab:red']
    
                                test_img = make_wave_img(lambcens[b][n],lambdaMap,b)
    
                                sortlist = np.argsort(alphaMapCont[test_img == 1])
                                testplot = test_cal_band[test_img == 1][sortlist]
                                alphaplot = alphaMapCont[test_img == 1][sortlist]
                                sliceplot = sliceMap[test_img == 1][sortlist]
    
                                for j in range(4):
                                    #--determine pixels contributing to spectral bin
                                    test_img = make_wave_img(lambcens[b][n+j],lambdaMap,b)
    
                                    meddata[(test_img == 1) & (sliceMap == 1)] = test_cal_band[(test_img == 1) & (sliceMap == 1)]
                                    medalpha[(test_img == 1) & (sliceMap == 1)] = alphaMapCont[(test_img == 1) & (sliceMap == 1)]
    
                                    fitplot = test_cal_band[(test_img == 1) & (sliceMap == 1)]
                                    alphaplot = alphaMapCont[(test_img == 1) & (sliceMap == 1)]
    
                                fitdata = np.nanmedian(meddata,axis=0)
                                alphaFit = np.nanmedian(medalpha,axis=0)
                                
                                alphaFit = alphaFit[~np.isnan(fitdata) & (fitdata > 0)]
                                fitdata = fitdata[~np.isnan(fitdata) & (fitdata > 0)]
            
                                if (len(fitdata)>3) & (it == 0):
                                    inst = fitClass()
            
                                    inst.fwhm = fwhm(lambcens[b][n+j])
        
                                    alphaFit = np.array(alphaFit)
                                    fitdata = np.array(fitdata)
                                    
                                    try:
                                        sortlist = np.argsort(alphaFit)
                                        alphaFit = alphaFit[sortlist]
                                        fitdata = fitdata[sortlist]
                                        
                                        coeffs, coeffs_cov = curve_fit(inst.gaussian,alphaFit,fitdata,p0=[alphaFit[np.nanargmax(fitdata)],np.abs(np.nanmax(fitdata))]) #,bounds=((alphaFit[np.nanargmax(fitdata)]-0.5,0),(alphaFit[np.nanargmax(fitdata)]+0.5,np.abs(np.nanmax(fitdata))*1.1)))
                
                                        cents[slices[p]][n] = coeffs[0]
                                        peaks[slices[p]][n] = coeffs[1]
    
                                    except:
                                        cents[slices[p]][n] = np.nan
                                        peaks[slices[p]][n] = np.nan
    
                                        
                                elif (len(fitdata)>3) & (it > 0):
                                    inst = fitClass()
            
                                    inst.fwhm = fwhm(lambcens[b][n+j])
                                    inst.alpha0 = smoothcents[slices[p]][n+j]
        
                                    alphaFit = np.array(alphaFit)
                                    fitdata = np.array(fitdata)
                                    
                                    # try:
                                    sortlist = np.argsort(alphaFit)
                                    alphaFit = alphaFit[sortlist]
                                    fitdata = fitdata[sortlist]
                                    
                                    coeffs, coeffs_cov = curve_fit(inst.gaussian_nocent,alphaFit,fitdata,p0=np.abs(np.nanmax(fitdata)))
            
                                    peaks[slices[p]][n] = coeffs[0]
    
                            if it == 0:
                                smoothcents[slices[p]] = clip(cents[slices[p]],2,1.5,10,pol_clip=True)
    
                            smoothpeaks[slices[p]] = clip(peaks[slices[p]],2,2,15)
    
                            if it == 0:
                                coeffs = np.polyfit(lambcens[b][~np.isnan(smoothcents[slices[p]])],smoothcents[slices[p]][~np.isnan(smoothcents[slices[p]])],2) #2
                                smoothcents[slices[p]] = np.polyval(coeffs,lambcens[b])
    
                            coeffs = np.polyfit(lambcens[b][~np.isnan(smoothpeaks[slices[p]])],smoothpeaks[slices[p]][~np.isnan(smoothpeaks[slices[p]])],6)
                            smoothpeaks[slices[p]] = np.polyval(coeffs,lambcens[b])
    
                            for n in range(nwavs):
        
                                #--determine pixels contributing to spectral bin
                                test_img = make_wave_img(lambcens[b][n],lambdaMap,b)
                                alphaPlot = alphaMapCont[(test_img == 1) & (sliceMap == 1)]
        
                                inst = fitClass()
                                inst.fwhm = fwhm(lambcens[b][n])
        
                                fit_PSF[(test_img == 1) & (sliceMap == 1)] = inst.gaussian(alphaPlot,smoothcents[slices[p]][n],smoothpeaks[slices[p]][n]) #smoothpeaks[slices[p]][n])
    
                                if do_apcorr:
                                    weightpix[(test_img == 1) & (sliceMap == 1)] = det_aperture(smoothcents[slices[p]][n],apsize,fwhm(lambcens[b][n]),alphaMapCont[(test_img == 1) & (sliceMap == 1)])
    
    
                    fit_PSF[fit_PSF <= 0] = 0
    
    
            PSF_recent = np.zeros((len(test_cal_band[:,0]),len(test_cal_band[0,:]))) #np.copy(PSF_model_set[0,:,:])
        
            # Perform detector-based spectral extraction
            spec_optimal[b][i] = np.full((nwavs),np.nan)
    
            acoord = np.full((nwavs),np.nan)
            bcoord = np.full((nwavs),np.nan)
            
            weight_map = np.zeros(np.shape(test_err_band))
            sub_save[b][dithnum] = np.full(np.shape(test_cal_band),np.nan)
            Var = np.copy(test_err_band)
            
            # Omit NaNs from analysis
            nan_idx = ~np.isnan(test_cal_band) & ~np.isnan(photomMapCont)
            
            rescale_row_PSF = np.zeros(np.shape(PSF_recent))
            norm_PSF = np.zeros(np.shape(PSF_recent))
            
            sel_val_map = {}
    
            if indith:
                for n in range(nwavs):
        
                    if n%100 == 0:
                        print('{}/{}'.format(n,nwavs))
                        
                    #--determine pixels contributing to spectral bin
                    test_img = make_wave_img(lambcens[b][n],lambdaMap,b)
        
                    # ascertain that only one pixel is selected per detector column
                    outliers = np.where(np.sum(test_img,axis=0)==2)[0]
        
                    for col in outliers:
                        Y_ = np.argsort(np.abs(lambdaMap[:,col]-lambcens[b][n]))[1]
                        test_img[Y_,col] = 0
                        
                    sel_valid = (test_img==1)
                    
                    idxw = np.where(sel_valid)[1]
                    idyw = np.where(sel_valid)[0]
        
                    idyw = idyw[np.argsort(idxw)]
                    idxw = idxw[np.argsort(idxw)]
        
                    # Remove NaNs
                    test_img[(~nan_idx) | (test_cal_band < 0)] = 0
                    
                    if len(np.where((test_img.flatten()) == 1)[0][:]) > 100:
                            
                        sel_valid = (test_img==1) & nan_idx & ~np.isnan(Var)
    
                        idx = np.where(sel_valid)[1]
                        idy = np.where(sel_valid)[0]
    
                        idy = idy[np.argsort(idx)]
                        idx = idx[np.argsort(idx)]
    
                        ## Get sub indices
                        idn = []
                        for m in range(len(idx)):
                            id1 = np.where(idx[m] == idxw)[0][:]
                            id2 = np.where(idy[m] == idyw)[0][:]
                            for add in range(len(id1)):
                                if id1[add] in id2:
                                    idn.append(id1[add])
    
                        if bright:
                            min_func_noscale = lambda param: do_PSF_sub_fitall_fast(PSF_model_set,test_cal_band[idy,idx],alphas,betas,sci_pos[0,512]+param[0],sci_pos[1,512]+param[1],param[2],param[3],param[4],b,alphaMapCont[idyw,idxw],photomMapCont[idyw,idxw],n,idn,3,False)[2]
                        else:
                            min_func_noscale = lambda param: do_PSF_sub_fitall_fast(PSF_model_set,test_cal_band[idy,idx],alphas,betas,sci_pos[0,512]+param[0],sci_pos[1,512]+param[1],param[2],param[3],param[4],b,alphaMapCont[idyw,idxw],photomMapCont[idyw,idxw],n,idn,2,False)[2] #USE 2 FOR FAINT SOURCES, 3 for bright
                        
                        res = opt.minimize(min_func_noscale,[0,0,1,0,0],bounds=[(-50e-3,50e-3),(-30e-3,30e-3),(0.85,1.15),(0,0),(0,0)],tol=1e-20,method='L-BFGS-B') #0.85, 1.15
                        _,PSF_recent[idyw,idxw],sumout = do_PSF_sub_fitall_fast(PSF_model_set,test_cal_band[idy,idx],alphas,betas,sci_pos[0,512]+res['x'][0],sci_pos[1,512]+res['x'][1],res['x'][2],res['x'][3],res['x'][4],b,alphaMapCont[idyw,idxw],photomMapCont[idyw,idxw],n,idn) #+res['x'][0] +res['x'][1] res['x'][2]
    
                        acoord[n] = res['x'][0]
                        bcoord[n] = res['x'][1]
    
    
                acoord_med = np.copy(acoord)
                bcoord_med = np.copy(bcoord)
    
                acoord_med[(np.abs(acoord) == 0.050)] = np.nan
                bcoord_med[(np.abs(bcoord) == 0.03)] = np.nan
                acoord_med[:50] = np.nan
                bcoord_med[:50] = np.nan
                acoord_med[-50:] = np.nan
                bcoord_med[-50:] = np.nan
                acoord_med = clip(acoord_med,40,1)
                bcoord_med = clip(bcoord_med,40,1)
                acoord_med[~np.isnan(acoord_med)] = gaussian_filter1d(acoord_med[~np.isnan(acoord_med)],20,mode='nearest') #np.nanmean(acoord)
                bcoord_med[~np.isnan(bcoord_med)] = gaussian_filter1d(bcoord_med[~np.isnan(bcoord_med)],20,mode='nearest') #np.nanmean(bcoord)
                acoord_med[~np.isnan(acoord)] = (interp.interp1d(np.arange(len(acoord_med))[~np.isnan(acoord_med)],acoord_med[~np.isnan(acoord_med)],fill_value='extrapolate'))(np.arange(len(acoord_med))[~np.isnan(acoord)])
                bcoord_med[~np.isnan(bcoord)] = (interp.interp1d(np.arange(len(bcoord_med))[~np.isnan(bcoord_med)],bcoord_med[~np.isnan(bcoord_med)],fill_value='extrapolate'))(np.arange(len(bcoord_med))[~np.isnan(bcoord)])
    
            if indith:
                PSF_avg = np.full(np.shape(PSF_recent),np.nan)
            else:
                PSF_avg = np.copy(fit_PSF)
                PSF_avg[PSF_avg < 0] = 0
    
            test_img_full = np.zeros((np.shape(test_cal_band)))
            peakcorr = np.full((nwavs),np.nan)
    
            PSF_test_img = np.copy(test_img_full)
    
            if do_apcorr:
                apcorr = np.full((nwavs),np.nan)
    
            if indith:
                if do_apcorr:
                    fwhm = np.loadtxt('./output/PSF/MIRIFU{}_{}{}_FWHM.dat'.format(ifu,b[0],bandsub))
                    fwhm = interp.interp1d(fwhm[:,0],fwhm[:,1],fill_value='extrapolate')
                    fwhm = interp.interp1d(lambcens[b][:],np.polyval(np.polyfit(lambcens[b][:],fwhm(lambcens[b][:]),deg=1),lambcens[b][:]))
                    
                    weightpix = np.zeros(np.shape(PSF_avg))
    
            
            for n in range(nwavs):
                #--determine pixels contributing to spectral bin
                test_img = make_wave_img(lambcens[b][n],lambdaMap,b)
                test_img_full[test_img==1] = 1
    
                # ascertain that only one pixel is selected per detector column
                outliers = np.where(np.sum(test_img,axis=0)==2)[0]
    
                for col in outliers:
                    Y_ = np.argsort(np.abs(lambdaMap[:,col]-lambcens[b][n]))[1]
                    test_img[Y_,col] = 0
                    
                sel_valid = (test_img==1)
                
                idxw = np.where(sel_valid)[1]
                idyw = np.where(sel_valid)[0]
    
                idyw = idyw[np.argsort(idxw)]
                idxw = idxw[np.argsort(idxw)]
    
                if indith:
                    # Remove NaNs
                    test_img[(~nan_idx) | (test_cal_band < 0)] = 0
    
                    if len(np.where((test_img.flatten()) == 1)[0][:]) > 100:
    
                        sel_valid = (test_img==1)
    
                        idx = np.where(sel_valid)[1]
                        idy = np.where(sel_valid)[0]
    
                        idy = idy[np.argsort(idx)]
                        idx = idx[np.argsort(idx)]
    
                        ## Get sub indices
                        idn = []
                        for m in range(len(idx)):
                            id1 = np.where(idx[m] == idxw)[0][:]
                            id2 = np.where(idy[m] == idyw)[0][:]
                            for add in range(len(id1)):
                                if id1[add] in id2:
                                    idn.append(id1[add])
    
                        _,PSF_avg[idyw,idxw],_ = do_PSF_sub_fitall_fast(PSF_model_set,test_cal_band[idy,idx],alphas,betas,sci_pos[0,512]+acoord_med[n],sci_pos[1,512]+bcoord_med[n],1,0,0,b,alphaMapCont[idyw,idxw],photomMapCont[idyw,idxw],n,idn)
            
                        PSF_avg[PSF_avg < 0] = 0
                        norm_PSF[idyw,idxw] = PSF_avg[idyw,idxw]/np.nansum((PSF_avg[idyw,idxw]))
    
                        if do_apcorr:
                            weightpix[idyw,idxw] = det_aperture(sci_pos[0,512]+acoord_med[n],apsize,fwhm(lambcens[b][n]),d2cMaps[b]['alphaMap'][:,k][idyw,idxw],betacent = sci_pos[1,512]+bcoord_med[n],betas = d2cMaps[b]['betaMap'][:,k][idyw,idxw])
                            apcorr[n] = (np.nansum(norm_PSF[idyw,idxw] * weightpix[idyw,idxw]))
                        
                else:
                
                    norm_PSF[idyw,idxw] = PSF_avg[idyw,idxw]/np.nansum((PSF_avg[idyw,idxw]))
    
                    peakcorr[n] = (np.nansum(PSF_model_set[1][n](alphaMapCont[(test_img == 1)])) / np.nansum(PSF_model_set[1][n](alphaMapCont[(idarray == 1) & (test_img == 1)])))
    
                    if do_apcorr:
                        apcorr[n] = (np.nansum(norm_PSF[test_img == 1] * weightpix[test_img == 1]))
    
            if do_apcorr:
                plt.figure()
                smoothap = clip(apcorr,1,3,15)
                plt.plot(lambcens[b],smoothap,lw=0,marker='o',c='tab:green',alpha=0.4)
                if indith:
                    apcorr = np.polyfit(np.arange(nwavs)[~np.isnan(smoothap) & (smoothap > 0)],smoothap[~np.isnan(smoothap) & (smoothap > 0)],deg=3)
                else:
                    apcorr = np.polyfit(np.arange(nwavs)[~np.isnan(smoothap) & (smoothap > 0)],smoothap[~np.isnan(smoothap) & (smoothap > 0)],deg=1)
                plt.plot(lambcens[b],np.polyval(apcorr,np.arange(nwavs)))
                apcorr = np.polyval(apcorr,np.arange(nwavs))
    
            if not indith:
                if b[0] == '1':
                    peakcorr[~np.isnan(peakcorr)] = gaussian_filter1d(peakcorr[~np.isnan(peakcorr)],5,mode='nearest')
                else:
                    peakcorr = np.polyval(np.polyfit(lambcens[b][~np.isnan(peakcorr)],peakcorr[~np.isnan(peakcorr)],2),lambcens[b])
                    
            norm_PSF[test_img_full == 0] = np.nan
    
            if indith:
                norm_PSF,_ = make_PSF(norm_PSF,npix,b,normPSF=True,zeros=False) #make_PSF(norm_PSF,npix,b,normPSF=False,zeros=True)
                norm_PSF = np.transpose(norm_PSF)
                norm_PSF[norm_PSF < 0] = 0
    
            fit_PSF = np.zeros((np.shape(test_cal_band)))
            
            for n in range(nwavs):
                
                #--determine pixels contributing to spectral bin
                test_img = make_wave_img(lambcens[b][n],lambdaMap,b)
    
                # ascertain that only one pixel is selected per detector column
                outliers = np.where(np.sum(test_img,axis=0)==2)[0]
    
                for col in outliers:
                    Y_ = np.argsort(np.abs(lambdaMap[:,col]-lambcens[b][n]))[1]
                    test_img[Y_,col] = 0
                    
                sel_valid = (test_img==1)
                
                idxw = np.where(sel_valid)[1]
                idyw = np.where(sel_valid)[0]
    
                idyw = idyw[np.argsort(idxw)]
                idxw = idxw[np.argsort(idxw)]
    
                # Remove NaNs
                test_img[(~nan_idx) | (test_cal_band < 0)] = 0
                
                if (np.nansum(norm_PSF[idyw,idxw]) > 0.5):
                    # try:
    
                    sel_valid = (test_img==1)
    
                    idx = np.where(sel_valid)[1]
                    idy = np.where(sel_valid)[0]
    
                    idy = idy[np.argsort(idx)]
                    idx = idx[np.argsort(idx)]
    
                    ## Get sub indices
                    idn = []
                    for m in range(len(idx)):
                        id1 = np.where(idx[m] == idxw)[0][:]
                        id2 = np.where(idy[m] == idyw)[0][:]
                        for add in range(len(id1)):
                            if id1[add] in id2:
                                idn.append(id1[add])
    
                    var_loop = 0
    
                    while var_loop < 4: #4
                        sel_valid = (test_img==1) & nan_idx & ~np.isnan(Var) & (norm_PSF > 0)
    
                        try:
    
                            ## Update estimate
                            if var_loop > 0:
                                f = spec_optimal[b][i][n]
                                Var[idyw,idxw] = test_err_band[idyw,idxw]**2 + (np.abs(f*norm_PSF[idyw,idxw]/photomMapCont[idyw,idxw]))
                                weight_map[idyw,idxw] = norm_PSF[idyw,idxw]**2 / (Var[idyw,idxw] * photomMapCont[idyw,idxw])
                            else:
                                ## First estimate
                                Var[idyw,idxw] = test_err_band[idyw,idxw]**2 * photomMapCont[idyw,idxw]
                                weight_map[idyw,idxw] = norm_PSF[idyw,idxw]**2 / Var[idyw,idxw]
        
                                f = np.nansum(weight_map[sel_valid] * (test_cal_band[sel_valid]) / norm_PSF[sel_valid]) / np.nansum(weight_map[sel_valid])
            
                            outlier = 10000
                            count = 0
            
                            outlier_test = np.ones(np.shape(test_img))
            
                            sig = np.nanstd((test_cal_band[sel_valid] - f * norm_PSF[sel_valid])**2/(Var[sel_valid]))
            
                            while (outlier >= sig*5):
                                sel_valid = (test_img==1) & nan_idx & ~np.isnan(Var) & (norm_PSF > 0)
    
                                outlier_test[sel_valid] = ((test_cal_band[sel_valid] - f * norm_PSF[sel_valid])**2/(Var[sel_valid]))
            
                                outlier = np.nanmax(np.abs(outlier_test[sel_valid]))
                                idout = np.nanargmax(np.abs(outlier_test[sel_valid]))
            
                                if (outlier >= sig*5):
                                    idx = np.where(sel_valid)[0][idout]
                                    idy = np.where(sel_valid)[1][idout]
                                    test_img[idx,idy] = 0
                                    outlier = 10000

                            # --spectral extraction
                            sel_valid = (test_img==1) & nan_idx & ~np.isnan(Var) & (test_cal_band > 0) & (norm_PSF > 0) # (outlier_test == 1)
                            
                            sel_val_map[n] = np.copy(sel_valid)
    
                            test_img2 = make_wave_img(lambcens[b][n],lambdaMap,b)
                            sel_valid2 = (test_img2==1) & (norm_PSF > 0)
    
                            if indith:
                                if do_apcorr:
                                    spec_optimal[b][i][n] = np.nansum(weight_map[sel_valid] * test_cal_band[sel_valid] * weightpix[sel_valid] / norm_PSF[sel_valid]) / np.nansum(weight_map[sel_valid] * weightpix[sel_valid]) 
                                    spec_optimal[b][i][n] = spec_optimal_nocent[b][i][n] + (1-(apcorr[n])) * spec_optimal_nocent[b][i][n]
                                    
                                else:
                                    spec_optimal[b][i][n] = np.nansum(weight_map[sel_valid] * test_cal_band[sel_valid] / norm_PSF[sel_valid]) / np.nansum(weight_map[sel_valid])
                            else:
                                if do_apcorr:
                                    spec_optimal[b][i][n] = np.nansum(weight_map[sel_valid] * test_cal_band[sel_valid] * weightpix[sel_valid] / norm_PSF[sel_valid]) / np.nansum(weight_map[sel_valid] * weightpix[sel_valid]) * peakcorr[n]
                                    spec_optimal[b][i][n] = spec_optimal_nocent[b][i][n] + (1-(apcorr[n])) * spec_optimal_nocent[b][i][n]
                                else:
                                    spec_optimal[b][i][n] = np.nansum(weight_map[sel_valid] * test_cal_band[sel_valid] / norm_PSF[sel_valid]) / np.nansum(weight_map[sel_valid]) * peakcorr[n]
    
                            var_loop += 1
    
                        except:

                            var_loop += 1
    
                            continue
    
        print('Done!')

        print('Saving spectra...')

        med_spec = np.full((np.shape(lambcens[b])),np.nan)
        mean_spec = np.full((np.shape(lambcens[b])),np.nan)
    
        dithlen = 4
    
        for i in range(len(lambcens[b])):
            
            val = np.full((dithlen),np.nan)
            weighted_list = []
        
            for n in range(0,dithlen):
                val[n] = spec_optimal[b][n][i]
    
            if (~np.isnan(val).all()) and not all(v == 0 for v in val):
                std = np.nanstd(val)
                med = np.nanmean(val)
                
                for n in range(0,dithlen):
        
                    if (val[n] < med+std) and (val[n] > med-std):

                        weighted_list.append(val[n])

                        count += 1

    
                if len(weighted_list) > 0:
                
                    med_spec[i] = np.nanmedian(weighted_list)
                    mean_spec[i] = np.nanmean(weighted_list)
    
            else:
                
                med_spec[i] = np.nan
                mean_spec[i] = np.nan

        save_dic = {}
        save_dic['WAVELENGTH'] = lambcens[b][50:-50]
        save_dic[1] = (spec_optimal[b][0][50:-50] * 10**6)
        save_dic[2] = (spec_optimal[b][1][50:-50] * 10**6)
        save_dic[3] = (spec_optimal[b][2][50:-50] * 10**6)
        save_dic[4] = (spec_optimal[b][3][50:-50] * 10**6)
        save_dic['MEAN'] = mean_spec[50:-50] * 10**6
        save_dic['MEDIAN'] = med_spec[50:-50] * 10**6
        
        with open(outDir + 'opt_spec_{}.pickle'.format(b), 'wb') as handle:
            pkl.dump(save_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)


def forward_sim(posA,posB,specA,specB,band,nints,ngroups,cache_path,dist_ver='flt8',ndither=range(1,5),direction='NEGATIVE',psfVer='01.05.05'):

    import spectres
    
    if band[0] in ['1','4']:
        npix = 512
        pix0 = 0
        k = np.arange(npix)
    else:
        pix0 = 512
        npix = 1032-pix0
        k = np.arange(pix0,pix0+npix)
        
    if band[0] in ['1','4']:
        idxrange = [0,512]
    else:
        idxrange = [512,1032]
        
    if band[0] in ['1','2']:
        ifu = 'SHORT'
    else:
        ifu = 'LONG'
            
    if band[1] == 'A':
        bandsub = 'SHORT'
    elif band[1] == 'B':
        bandsub = 'MEDIUM'
    else:
        bandsub = 'LONG'

    ## Generate continuous distortion maps
    d2cMaps = d2cMapping(band,'./data/',slice_transmission='10pc',fileversion = "flt8")
    alphaMapCont = _create_incremental_alphaMap(band, d2cMaps)
    lambdaMap,lambcens = cont_lambda(band,dist_ver,alphaMapCont,d2cMaps['lambdaMap'])

    lambdaMap = lambdaMap[:,k]

    nwavs = len(lambcens)
    lambdas = lambdaMap[np.nonzero(lambdaMap)].flatten()

    ## Resample input spectra to same wavelengths
    specA = spectres.spectres(lambcens,specA[:,0],specA[:,1],verbose=False)
    specB = spectres.spectres(lambcens,specB[:,0],specB[:,1],verbose=False)

    ## Get PHOTOM and pixel maps
    photomMapCont = fits.getdata('./output/PHOTOM/photom_continuous_{}_{}_v2.fits'.format(ifu,bandsub))[:,idxrange[0]:idxrange[-1]]
    pixelMapCont = fits.getdata('./output/PIXEL/fflat_continuous_{}_{}.fits'.format(ifu,bandsub))[:,idxrange[0]:idxrange[-1]]
    pixSize = (fits.open('./output/PHOTOM/photom_continuous_{}_{}_v2.fits'.format(ifu,bandsub))[5]).data[:,idxrange[0]:idxrange[-1]]
    
    photomMapCont = photomMapCont * pixSize / pixelMapCont

    ## Get noise and gain properties
    read = fits.getdata(find_file(band,cache_path,file='readnoise'))[:,k]
    gain = fits.getdata(find_file(band,cache_path,file='gain'))[:,k]

    ## Get PCA background model
    #bkgV = np.loadtxt('./output/BACKGROUND/MIRIFU{}_{}_backgroundPCA.dat'.format(ifu,subband))

    PSF_out = {}
    PSF_out['Total'] = {}
    PSF_out['Source A'] = {}
    PSF_out['Source B'] = {}
    PSF_out['Rate'] = {}
    PSF_out['pos A'] = {}
    PSF_out['pos B'] = {}
    PSF_out['err'] = {}

    for n in ndither:

        ## Generate model for primary
        PSF_model_set,alphas,betas = get_PSF_set(band[0]+bandsub,direction,n,posA[0]*np.ones(1024),posA[1]*np.ones(1024),alphaMapCont,pos_abs=False,version=psfVer)
        posAabs = [posA[0]+alphas[1,512],posA[1]+betas[1,512]]
        PSFA = weighted_mean(PSF_model_set,betas,posA[1],pos_abs=False)
        PSFA = PSFA*photomMapCont
        PSFA = wave_norm(PSFA,band,lambdaMap)
        
        PSFA_scaled = np.full((PSFA.shape),np.nan)
        for wvl in range(nwavs):
            sel = (make_wave_img(lambcens[wvl],lambdaMap,band) == 1)
            
            PSFA_scaled[sel] = PSFA[sel]*specA[wvl]
        
        ## Generate model for secondary
        PSF_model_set,alphas,betas = get_PSF_set(band[0]+bandsub,direction,n,(posA[0]+posB[0])*np.ones(1024),(posA[1]+posB[1])**np.ones(1024),alphaMapCont,pos_abs=False,version=psfVer)
        posBabs = [posB[0]+alphas[1,512],posB[1]+betas[1,512]]
        PSFB = weighted_mean(PSF_model_set,betas,(posA[1]+posB[1])*np.ones(1024),pos_abs=False)
        PSFB = PSFB*photomMapCont
        PSFB = wave_norm(PSFB,band,lambdaMap)
        PSFB_scaled = np.full((PSFB.shape),np.nan)
        for wvl in range(nwavs):
            sel = (make_wave_img(lambcens[wvl],lambdaMap,band) == 1)
            
            PSFB_scaled[sel] = PSFB[sel]*specB[wvl]
        
        ## Add PSF images
        PSF_tot = PSFA_scaled+PSFB_scaled

        for x in range(len(PSF_tot[0,:])):
            nanid = ~np.isnan(PSF_tot[:,x])
            
            PSF_tot_int = interp.interp1d(np.arange(len(PSF_tot[:,x]))[nanid],PSF_tot[nanid,x],fill_value='extrapolate')
            PSFA_int = interp.interp1d(np.arange(len(PSFA_scaled[:,x]))[nanid],PSFA_scaled[nanid,x],fill_value='extrapolate')
            PSFB_int = interp.interp1d(np.arange(len(PSFB_scaled[:,x]))[nanid],PSFB_scaled[nanid,x],fill_value='extrapolate')

            PSF_tot[:,x] = PSF_tot_int(np.arange(len(PSF_tot[:,x])))
            PSFA_scaled[:,x] = PSFA_int(np.arange(len(PSF_tot[:,x])))
            PSFB_scaled[:,x] = PSFB_int(np.arange(len(PSF_tot[:,x])))

        ## Add background
        # Do later

        ## Make ramps & do ramp fit
        PSF_ramp = np.zeros((nints,ngroups,len(PSF_tot[:,0]),len(PSF_tot[0,:])))
        
        ## Arrays for ramp fit
        PSF_rate = np.full((nints,len(PSF_tot[:,0]),len(PSF_tot[0,:])),np.nan)
        errArray = np.full((nints,len(PSF_tot[:,0]),len(PSF_tot[0,:])),np.nan)

        for ints in range(nints):

            for group in range(1,ngroups):
    
                PSF_ramp[ints,group,:,:] = (PSF_tot[:,:]/photomMapCont)*(2.775*group) #(PSF_ramp[group-1,:,:]) + (PSF_tot[:,:]/photomMapCont)*(2.775)
                
                for x in range(len(PSF_tot[0,:])):
                    for y in range(len(PSF_tot[:,0])):
                        if (~np.isnan(PSF_ramp[ints,group,y,x])) & (PSF_ramp[ints,group,y,x] > 0):
    
                            ## Add read and poisson noise
                            PSF_ramp[ints,group,y,x] = poisson(PSF_ramp[ints,group,y,x] * gain[y,x])/gain[y,x]
                            PSF_ramp[ints,group,y,x] = np.round(PSF_ramp[ints,group,y,x]+np.random.normal(0,read[y,x]))

            for x in range(len(PSF_tot[0,:])):
                    for y in range(len(PSF_tot[:,0])):
                            PSF_rate[ints,y,x] = np.polyfit(np.arange(0,ngroups*2.775,2.775),PSF_ramp[ints,:,y,x],deg=1)[0]
                            errArray[ints,y,x] = np.nanstd(PSF_ramp[ints,:,y,x]-np.polyval([PSF_rate[ints,y,x],0],np.arange(0,ngroups*2.775,2.775)))

        PSF_rate = np.nanmean((PSF_rate),axis=0)
        errArray = np.nanmean((errArray),axis=0)
                    
        PSF_tot = PSF_rate * photomMapCont
        
        PSF_out['Total'][n] = PSF_tot
        PSF_out['Source A'][n] = PSFA_scaled
        PSF_out['Source B'][n] = PSFB_scaled
        PSF_out['Rate'][n] = PSF_rate
        PSF_out['pos A'][n] = posAabs
        PSF_out['pos B'][n]= posBabs
        PSF_out['err'][n] = errArray
        
    return PSF_out


