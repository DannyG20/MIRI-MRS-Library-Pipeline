"""Function file get the interpolated pixel flat and generate a custom file

:History:

Created on Wed 25 Sep 2024

@author: Danny Gasman (KULeuven, Belgium, danny.gasman@kuleuven.be)
"""

# import python modules
import numpy as np
from pandas import read_pickle
import itertools
from .distortion import d2cMapping
from .funcs import point_source_centroiding
import datetime
from os import listdir
from astropy.io import fits

#import sys
#sys.path.append('./miricoord/mrs')
import miricoord.mrs.mrs_tools as coord

# main function to call
def gen_custom_pixel_flat(rate_file,dist_dir,cache_dir,alphadic,betadic,num=1,dist_ver='flt7',version='01.00.00'):
    hdu_rate = fits.open(rate_file)

    detector = hdu_rate[0].header['DETECTOR']
    subband = hdu_rate[0].header['BAND']
    
    if detector == 'MIRIFUSHORT':
        band = ['1{}'.format(subband),'2{}'.format(subband)]
    else:
        band = ['3{}'.format(subband),'4{}'.format(subband)]

    rate = hdu_rate['SCI'].data
    hdu_rate.close()
    ref_name = find_ref_flat(band[0],cache_dir)
    
    ref_flat = fits.open(cache_dir+'/references/jwst/miri/'+ref_name)

    d2cMaps = {}
    coeff_img = {}
    beta_frac = {}
    maps = {}
    map_out = np.copy(ref_flat['SCI'].data)

    if band[0][1:] == 'SHORT':
        subbandl = 'A'
    elif band[0][1:] == 'MEDIUM':
        subbandl = 'B'
    else:
        subbandl = 'C'

    for b in band:

	maps[b] = np.empty(np.shape(rate))

	if b in ['4MEDIUM','4LONG']:
		maps[b][:,:] = 1
	else:

        	# Get coordinate maps
        	d2cMaps[b] = d2cMapping(b[0]+subbandl,dist_dir,slice_transmission='10pc',fileversion = dist_ver)
        
        	# Find slice fraction
        	beta_frac[b] = find_slice_cent(betadic[b[0]+subbandl],d2cMaps[b]['betaMap'],d2cMaps[b]['sliceMap'],b[0]+subbandl)

        	# Get coefficient file
        	coeff_img[b] = open_coeff_ref(b,num,version)

        	# Create pixel map
        	Xmap = beta_frac[b]
        
        	# Fill map
        	for y in range(len(d2cMaps[b]['lambdaMap'][:,0])):
            	for x in range(len(d2cMaps[b]['lambdaMap'][0,:])):
                	maps[b][y,x] = polyval2d(Xmap, d2cMaps[b]['lambdaMap'][y,x], coeff_img[b])
                       
        	id1 = np.where((d2cMaps[b]['lambdaMap'] == 0) | (d2cMaps[b]['lambdaMap'] == np.nan))
        	maps[b][id1] = 1
        
            

    # Combine maps
    if detector == 'MIRIFUSHORT':
        map_out[:,:512] = maps[band[0]][:,:512]
        map_out[:,512:] = maps[band[1]][:,512:]
        ifu = 'short'
    elif detector == 'MIRIFULONG':
        map_out[:,:512] = maps[band[1]][:,:512]
        map_out[:,512:] = maps[band[0]][:,512:]
        ifu = 'long'
                      
    ref_flat['SCI'].data = map_out

    ref_flat.writeto('./references/PIXEL/custom_pixel_ref_{}_{}.fits'.format(ifu,num),overwrite=True)

    ref_flat.close()
    
# main function to call
def gen_photom(rate_file,dist_dir,alphadic,betadic,num=1,version='01.00.00',dist_ver='flt8',ref_available=False):
    hdu_rate = fits.open(rate_file)

    detector = hdu_rate[0].header['DETECTOR']
    subband = hdu_rate[0].header['BAND']
    dithdir = hdu_rate[0].header['DITHDIRC']
    
    if detector == 'MIRIFUSHORT':
        b = '1{}'.format(subband)

        rate = hdu_rate['SCI'].data
        
        ref_list = find_ref_photom(b,version)

        if b[1:] == 'SHORT':
            subbandl = 'A'
        elif b[1:] == 'MEDIUM':
            subbandl = 'B'
        else:
            subbandl = 'C'
        
        d2cMaps = {}

        # Get coordinate maps
        d2cMaps[b] = d2cMapping(b[0]+subbandl,dist_dir,slice_transmission='10pc',fileversion = dist_ver)
        
        diff = 1000
        
        diffalpha = []
        diffbeta = []
        filelist = []

        for file in ref_list:
            filedirc = (file.split('_'))[4]

            if dithdir == filedirc:
                hdu = fits.open('./references/PHOTOM/'+file)
                refalpha = hdu[1].header['ALPHA{}_{}'.format(b[0],num)]
                refbeta = hdu[1].header['BETA{}_{}'.format(b[0],num)]
                hdu.close()

                diffalpha.append(refalpha-alphadic)
                diffbeta.append(refbeta-betadic)
                filelist.append(file)

        lowest = sorted(np.abs(diffbeta))[:3]

        for i in range(len(diffbeta)):
            if np.abs(diffbeta[i]) in lowest:
                if np.abs(diffalpha[i]) < diff:
                    photom_name = filelist[i]
                    diff = np.abs(diffalpha[i])
    
    else:
        print('Invalid detector, no need to find PHOTOM by pointing.')

    hdu_rate.close()

    return photom_name
                       
def find_ref_flat(band,crds_dir):
    
    startdate = datetime.datetime(1900,1,1)
    
    crdsDir = crds_dir+'/references/jwst/miri/'
    
    subband = band[1:]
    
    if band[0] in ['1','2']:
        det = 'MIRIFUSHORT'
    elif band[0] in ['3','4']:
        det = 'MIRIFULONG'
    for filename in listdir(crdsDir):
        if filename.startswith('jwst_miri_flat'):
            hdu = fits.open(crdsDir+filename)
            datestr = (hdu[0].header['DATE']).split('T')
            date = datetime.datetime.strptime(datestr[0]+' '+datestr[1],'%Y-%m-%d %H:%M:%S.%f')
            refdet = hdu[0].header['DETECTOR']
            refband = hdu[0].header['BAND']
            
            if refdet == det:
                if refband == 'N/A':
                    if date > startdate:
                        reffile = filename
                        startdate = date
                elif refband == subband:
                    if date > startdate:
                        reffile = filename
                        startdate = date
    
    return reffile

def find_ref_photom(band,version):

    crdsDir = './references/PHOTOM/'

    subband = band[1:]

    reffile_list = []

    if band[0] in ['1','2']:
        det = 'MIRIFUSHORT'
    else:
        det = 'MIRIFULONG'

    for filename in listdir(crdsDir):
        if filename.startswith(det +'_'+ subband) and filename.endswith(version+'.fits'):
            reffile_list.append(filename)

    return reffile_list

def open_coeff_ref(band,dithnum,version):
    if band[0] in ['1','2']:
        ifu = 'SHORT'
    else:
        ifu = 'LONG'
    
    coeffs = np.loadtxt('./references/PIXEL/MIRIFU{}_{}{}_PS_1DPIXELFLAT_{}_{}.txt'.format(ifu,band[0],band[1:],dithnum,version))
    
    return coeffs

## Find nearest slice and offset
def find_slice_cent(beta_cent,betaMap,sliceMap,band):
    
    slice_widths = [0.176,0.277,0.387,0.645]
    
    #ignore outside slice
    id0 = np.where((sliceMap != 0) & ((sliceMap - int(band[0])*100 > 0) & (sliceMap - int(band[0])*100 < 100)))
    idmin = np.argmin(np.abs(betaMap[id0]-beta_cent))
    beta_nearest = betaMap[id0[0][idmin],id0[1][idmin]]
    #get fraction offset
    frac_off = (beta_cent - beta_nearest)/slice_widths[int(band[0])-1]
    
    return frac_off

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z = z + a * (x)**j * y**i
    return z
