"""Function file get the interpolate the linearity coefficients and generate a custom file

:History:

Created on Fri 16 Feb 2024

@author: Danny Gasman (KULeuven, Belgium, danny.gasman@kuleuven.be)
"""

# import python modules
import numpy as np
from pandas import read_pickle
from .distortion import d2cMapping
from .funcs import point_source_centroiding,detpixel_trace,detpixel_trace_compactsource
import datetime
from os import listdir
from astropy.io import fits

# main function to call
def gen_custom_linearity(rate_file,dist_dir,crds_dir,num=1,file_ver='v2',dist_ver='flt7'):
    detector = rate_file[0].header['DETECTOR']
    subband = rate_file[0].header['BAND']
    
    if detector == 'MIRIFULONG':
        print('Cannot generate solution for long wavelength detector')
        
    else:
        band = ['1{}'.format(subband),'2{}'.format(subband)]
        
        # Get linearity reference file
        lin_ref = find_ref_linearity(band[0],crds_dir)
        
        coeffs_save = np.copy(lin_ref['COEFFS'].data)
        
        rate = rate_file['SCI'].data
        
        d2cMaps = {}
        alphadic = {}
        betadic = {}
        
        for b in band:
            
            print('Working on band {}'.format(b))
            
            if b[1:] == 'SHORT':
                subbandl = 'A'
            elif b[1:] == 'MEDIUM':
                subbandl = 'B'
            else:
                subbandl = 'C'
            
            # Get coordinate maps
            d2cMaps[b] = d2cMapping(b[0]+subbandl,dist_dir,slice_transmission='10pc',fileversion = dist_ver)
            
            # Centroid input data
            alphadic[b[0]],betadic[b[0]] = return_centroids(rate,b,d2cMaps[b])
            
            # Get coefficient file
            coeff_img1,coeff_img2 = open_coeff_ref(b,file_ver)
            
            # Loop over detector and find solutions
            ypos,xpos = detpixel_trace_compactsource(rate,b+subbandl,d2cMaps[b],offset_slice=0)
            sliceID = d2cMaps[b]['sliceMap'][ypos[512],xpos[512]] - 100*int(b[0])
            
            ypos,xpos = detpixel_trace(b,d2cMaps[b],sliceID=sliceID,alpha_pos=alphadic[b[0]]
            
            alpha_off_list,y_list,x_list = [],[],[]
            
            for slice_off in [-1,0,1]:
                coord=np.where(d2cMaps[b]['sliceMap'][ypos[512],xpos[512]] == sliceID+slice_off)
                ypos,xpos = detpixel_trace(b,d2cMaps[b],sliceID=sliceID+slice_off,alpha_pos=alphadic[b[0]])#alpha)
                
                for xpixoff in [-2,-1,0,1,2]:
                    for i in range(len(ypos)):
                        alpha_off = d2cMaps[b]['alphaMap'][ypos[i],xpos[i]+xpixoff]-alphadic[b[0]]
                        
                        if ypos[i] < 10:
                            y = 10
                        elif ypos[i]>1010:
                            y = 1010
                        else:
                            y = ypos[i]
                        
                        alpha_off_list.append(alpha_off)
                        y_list.append(ypos[i])
                        x_list.append(xpos[i]+xpixoff)

            coeff1,replace = spline2d_get_curve(coeff_img1,alpha_off_list,y_list)
            coeff2,replace = spline2d_get_curve(coeff_img2,alpha_off_list,y_list)
            
            for i in range(len(x_list)):
                if np.abs(alpha_off_list[i])<=0.4:

                    if replace[i]:
                        coeffs_save[2,y_list[i],x_list[i]] = coeff1[i]
                        coeffs_save[3,y_list[i],x_list[i]] = coeff2[i]
        
        lin_ref['COEFFS'].data = coeffs_save
        
        lin_ref.writeto('custom_linearity_ref_{}_{}.fits'.format(num,file_ver),overwrite=True)
            
def find_nearest_grid(rate_file,dist_dir,lin_dir,num=1,lin_version='01.05.00',dist_ver='flt7'):
    detector = rate_file[0].header['DETECTOR']
    subband = rate_file[0].header['BAND']
    
    if detector == 'MIRIFULONG':
        print('Cannot generate solution for long wavelength detector')
        
    else:
        band = ['1{}'.format(subband)] #,'2{}'.format(subband)]
        
        rate = rate_file['SCI'].data
        dithdirc = rate_file[0].header['DITHDIRC']
        
        d2cMaps = {}
        alphadic = {}
        betadic = {}
        
        for b in band:
            
            if b[1:] == 'SHORT':
                subbandl = 'A'
            elif b[1:] == 'MEDIUM':
                subbandl = 'B'
            else:
                subbandl = 'C'
            
            # Get coordinate maps
            d2cMaps[b] = d2cMapping(b[0]+subbandl,dist_dir,slice_transmission='10pc',fileversion = "flt7")
            
            # Centroid input data
            alphadic[b[0]],betadic[b[0]] = return_centroids(rate,b,d2cMaps[b])
            print(alphadic[b[0]],betadic[b[0]])

            startdate = datetime.datetime(1900,1,1)
            if b[0] in ['1','2']:
                det = 'MIRIFUSHORT'
            elif b[0] in ['3','4']:
                det = 'MIRIFULONG'
                
            diff = 100
            diffbeta = []
            diffalpha = []
            filelist = []
                
            for filename in listdir(lin_dir):
                
                if filename.startswith(det+'_'+subband+'_PIXEL_LINEARITY_{}{}'.format(dithdirc,num)) and filename.endswith('{}.fits'.format(lin_version)):

                    hdu = fits.open(lin_dir+filename)
                    datestr = (hdu[0].header['DATE']).split('T')
                    date = datetime.datetime.strptime(datestr[0],'%Y-%m-%d')
                    refalpha = hdu[1].header['ALPHA{}'.format(b[0])]
                    refbeta = hdu[1].header['BETA{}'.format(b[0])]
                    hdu.close()

                    diffalpha.append(refalpha-alphadic[b[0]])
                    diffbeta.append(refbeta-betadic[b[0]])
                    filelist.append(filename)

            lowest = sorted(np.abs(diffbeta))[:3]

            for i in range(len(diffbeta)):
                if np.abs(diffbeta[i]) in lowest:
                    if np.abs(diffalpha[i]) < diff:
                        reffile = filelist[i]
                        diff = np.abs(diffalpha[i])

    return reffile

def return_centroids(rate,b,d2cMaps):
    
    if b[1:] == 'SHORT':
        subband = 'A'
    elif b[1:] == 'MEDIUM':
        subband = 'B'
    else:
        subband = 'C'
    
    lambmin = np.nanmin(d2cMaps['lambdaMap'][np.where(d2cMaps['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps['lambdaMap']) # micron
    lambcens = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    lambfwhms = np.ones(len(lambcens))*(2*(lambmax-lambmin)/1024.)
    
    sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D = point_source_centroiding(b,rate,d2cMaps,spec_grid=[lambcens,lambfwhms],fit='2D')

    alpha_ind=np.where(np.abs(alpha_centers2D)<=np.nanmax(np.abs(d2cMaps['alphaMap'])))[0][:]
    beta_ind=np.where(np.abs(beta_centers2D)<=np.nanmax(np.abs(d2cMaps['betaMap'])))[0][:]

    popt_alpha = np.polyfit(lambcens[alpha_ind],alpha_centers2D[alpha_ind],4)
    alpha = np.poly1d(popt_alpha)

    popt_beta  = np.polyfit(lambcens[beta_ind],beta_centers2D[beta_ind],4)
    beta = np.poly1d(popt_beta)

    return alpha(lambcens[512]),beta(lambcens[512])

def open_coeff_ref(band,version):
    if band[1:] == 'SHORT':
        subband = 'A'
    elif band[1:] == 'MEDIUM':
        subband = 'B'
    else:
        subband = 'C'
    
    coeff_dic1 = read_pickle('./core/coeff_fits/coeff_img_L2_{}_{}.pickle'.format(band[0]+subband,version))
    coeff_dic2 = read_pickle('./core/coeff_fits/coeff_img_L3_{}_{}.pickle'.format(band[0]+subband,version))
    
    return coeff_dic1,coeff_dic2

def find_ref_linearity(band,crds_dir):
    
    startdate = datetime.datetime(1900,1,1)
    
    crdsDir = crds_dir+'/references/jwst/miri/'
    
    subband = band[1:]
    
    if band[0] in ['1','2']:
        det = 'MIRIFUSHORT'
    elif band[0] in ['3','4']:
        det = 'MIRIFULONG'
    for filename in listdir(crdsDir):
        if filename.startswith('jwst_miri_linearity'):
            hdu = fits.open(crdsDir+filename)
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
    
    ref_file = fits.open(crdsDir+reffile)
    
    return ref_file
        
def find_nearest_grid_fringe(file,cent_alpha,cent_beta,band,fringedir,vers):
    
    hdu = fits.open(file)

    detector = hdu[0].header['DETECTOR']
    subbandl = hdu[0].header['BAND']
    dithdir = hdu[0].header['DITHDIRC']
    n = (scifile[i].split('_')[2])[-1]

    if subbandl == 'SHORT':
        subband = 'A'
    elif subbandl == 'MEDIUM':
        subband = 'B'
    else:
        subband = 'C'

    if detector == 'MIRIFUSHORT':
        band = ['1{}'.format(subband),'2{}'.format(subband)]
    else:
        band = ['3{}'.format(subband),'4{}'.format(subband)]
        
    data = hdu['SCI'].data

    hdu.close()

    alpha = {}
    beta = {}
    
    for b in band:        
        if b[0] in ['1','2','3']:
        
            fringe_file[b,n] = []
            reffilelist = []

            for filename in listdir(fringedir):
                if filename.startswith('{}_{}_PS_FRINGE_{}{}'.format(detector,subbandl,dithdir,n)) and filename.endswith('{}.fits'.format(vers)):
                    reffilelist.append(filename)

    diff = 1000
    
    diffalpha = []
    diffbeta = []
    filelist = []
    
    for filename in reffilelist:
        hdu = fits.open(fringedir+filename)
        refalpha = hdu[1].header['A{}'.format(band[0])]
        refbeta = hdu[1].header['B{}'.format(band[0])]
        hdu.close()

        diffalpha.append(refalpha-cent_alpha)
        diffbeta.append(refbeta-cent_beta)
        filelist.append(filename)

    lowest = sorted(np.abs(diffbeta))[:3]

    for i in range(len(diffbeta)):
        if np.abs(diffbeta[i]) in lowest:
            if np.abs(diffalpha[i]) < diff:
                reffile = filelist[i]
                diff = np.abs(diffalpha[i])
    
    return reffile
                                       
# Get coefficient per alpha offset
def spline2d_get_curve(coeff_img,x,y):
    
    vals_out = np.empty((len(x))) #np.empty((len(x),len(y)))
    dic_keys = np.array(list(coeff_img.keys()))
    
    replace = []
    
    #Compute chunks
    for i in range(len(x)):
        try:
            dic_id = np.where((y[i]==dic_keys[:,0]) & (x[i]>=dic_keys[:,1]) & (x[i]<dic_keys[:,2]))[0][0]

            vals_out[i]=coeff_img[dic_keys[dic_id,0],dic_keys[dic_id,1],dic_keys[dic_id,2]](x[i])
            replace.append(True)
        except:
            replace.append(False)
            continue
            
    return np.array(vals_out),replace
        
# Used to generate the spline fits, not needed by general user
def spline2d_in_chunks(x,y,z,x_min,x_max,x_space,y_min,y_max,y_space,pad=0.01,fitdeg=3,test=False):
    ny,nx = y_space,x_space # chunks
    pix_arr = np.linspace(y_min,y_max,ny) # pixels
    pos_arr = np.linspace(x_min,x_max,nx) # alpha offsets

    coeff_img = {}
    
    if test:
        plt.figure()
    
    #Compute chunks
    for i in range(len(pix_arr)-1):
        idy = np.where((y>=pix_arr[i]) & (y<pix_arr[i+1]))[0][:]
        zfity = z[idy]
        for j in range(len(pos_arr)-1):
            idx = np.where((x[idy]>=pos_arr[j]-pad) & (x[idy]<pos_arr[j+1]+pad))[0][:]
            zfitx = zfity[idx]
            if test:
                plt.scatter(x[idy[idx]],zfitx,alpha=0.3)
            
            #Fit chunk
            try:
                id_sorted = np.argsort(x[idy[idx]])
                
                coeffs = np.polyfit(x[idy[idx[id_sorted]]],zfitx[id_sorted],deg=fitdeg)
                model = np.poly1d(coeffs)
                
                if test:
                    plt.plot(x[idy[idx[id_sorted]]],model(x[idy[idx[id_sorted]]]),c='red')
                coeff_img[pix_arr[i],pos_arr[j],pos_arr[j+1]] = model
            except:
                continue
            
    return coeff_img


