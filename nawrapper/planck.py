"""Utility functions for processing Planck maps into spectra."""

from astropy.io import fits
import healpy as hp
import numpy as np
import nawrapper as nw

param_2018 = {
    'pol_efficiency' : {'100' : 0.9995, '143' : 0.999, '217' : 0.999},
    'map_dir' : 'maps/PR3/frequencyMaps',  # relative to base directory
    'mask_dir' : 'maps/PR3/maskMaps/',  # relative to base directory
    'beam_dir' : 'BeamWf_HFI_R3.01/',  # relative to base directory
    'nside' : 2048,
    'lmax' : 2508
}


def load_planck_data(base_data_dir, par=param_2018, 
                     freq1='143', freq2='143', split1='1', split2='2'):
    
    map_dir, mask_dir, beam_dir = par['map_dir'], par['mask_dir'], par['beam_dir']
    nside = par['nside']
    
    # read effective cross beams
    # beams are only available as f1 x f2 with f1 \leq f2
    # splits are only 1 x 1, 1 x 2, 2 x 2 (i.e. no 2 x 1)
    if int(freq1) < int(freq2):
        beam_Wl_hdu = fits.open(
            f'{base_data_dir}/{beam_dir}/Wl_R3.01_plikmask' + 
            f'_{freq1}hm{split1}x{freq2}hm{split2}.fits')
    elif int(freq1) > int(freq2):
        beam_Wl_hdu = fits.open(
            f'{base_data_dir}/{beam_dir}/Wl_R3.01_plikmask' + 
            f'_{freq2}hm{split2}x{freq1}hm{split1}.fits')
    else: # they are equal
        if int(split1) > int(split2):
            beam_Wl_hdu = fits.open(
                f'{base_data_dir}/{beam_dir}/Wl_R3.01_plikmask' + 
                f'_{freq2}hm{split2}x{freq1}hm{split1}.fits')
        else:
            beam_Wl_hdu = fits.open(
                f'{base_data_dir}/{beam_dir}/Wl_R3.01_plikmask' + 
                f'_{freq1}hm{split1}x{freq2}hm{split2}.fits')
    
    beam_temp = np.sqrt(beam_Wl_hdu[1].data['TT_2_TT'][0])
    beam_pol = np.sqrt(beam_Wl_hdu[2].data['EE_2_EE'][0])
    
    mfile_1 = f'{base_data_dir}/{map_dir}/HFI_SkyMap_{freq1}' + \
        f'_2048_R3.01_halfmission-{split1}.fits'
    mfile_2 = f'{base_data_dir}/{map_dir}/HFI_SkyMap_{freq2}' + \
        f'_2048_R3.01_halfmission-{split2}.fits'

    maskfile1 = f'{base_data_dir}/{mask_dir}/' + \
        f'COM_Mask_Likelihood-temperature-{freq1}-hm{split1}_2048_R3.00.fits'
    maskfile2 = f'{base_data_dir}/{mask_dir}/' + \
        f'COM_Mask_Likelihood-temperature-{freq2}-hm{split2}_2048_R3.00.fits'
    maskfile1_pol = f'{base_data_dir}/{mask_dir}/' + \
        f'COM_Mask_Likelihood-polarization-{freq1}-hm{split1}_2048_R3.00.fits'
    maskfile2_pol = f'{base_data_dir}/{mask_dir}/' + \
        f'COM_Mask_Likelihood-polarization-{freq2}-hm{split2}_2048_R3.00.fits'
    
    # read maps
    pol_fac_1 = par['pol_efficiency'][freq1]
    m1_map_I = hp.read_map(mfile_1, field=0, verbose=False)
    m1_map_Q = hp.read_map(mfile_1, field=1, verbose=False) * pol_fac_1
    m1_map_U = hp.read_map(mfile_1, field=2, verbose=False) * pol_fac_1

    pol_fac_2 = par['pol_efficiency'][freq2]
    m2_map_I = hp.read_map(mfile_2, field=0, verbose=False)
    m2_map_Q = hp.read_map(mfile_2, field=1, verbose=False) * pol_fac_2
    m2_map_U = hp.read_map(mfile_2, field=2, verbose=False) * pol_fac_2

    # read masks
    mask1 = hp.read_map(maskfile1, verbose=False)
    mask2 = hp.read_map(maskfile2, verbose=False)
    mask1_pol = hp.read_map(maskfile1_pol, verbose=False)
    mask2_pol = hp.read_map(maskfile2_pol, verbose=False)
    
    # SHTs on maps
    m1 = nw.namap(
        map_I=m1_map_I, mask=mask1, beam=beam_temp, 
        map_Q=m1_map_Q, map_U=m1_map_U, mask_pol=mask1_pol, beam_pol=beam_pol,
        unpixwin=True,
        nside=nside, sub_monopole=True, sub_dipole=True)
    m2 = nw.namap(
        map_I=m2_map_I, mask=mask2, beam=beam_temp, 
        map_Q=m2_map_Q, map_U=m2_map_U, mask_pol=mask2_pol, beam_pol=beam_pol,
        unpixwin=True,
        nside=nside, sub_monopole=True, sub_dipole=True)
    
    return m1, m2


class PlanckCov:
    def __init__(self, ellspath, covpath='covmat.dat', 
                 clpath='data_extracted.dat'):
        self.cov = np.linalg.inv(np.genfromtxt(covpath))
        self.ells = np.genfromtxt(ellspath, usecols=0, unpack=True)
        self.cls = np.genfromtxt(clpath, usecols=1, unpack=True)
                 
    def get_subcov(self, spec, debug=True):
        """
        spec: {TT, TE, or EE},  {'100x100', '143x143', '143x217', '217x217'}
        cross: string, i.e.
        
        returns tuple: 
            ells, cl, subcovariance matrix 
        """
        subarray_indices = np.arange(
            self.cov.shape[0]-1)[(np.diff(self.ells) < 0)] + 1
        
        subarray_indices = np.hstack( ( [0], subarray_indices, len(self.cls) ) )
        
        keys = ['TT_100x100', 'TT_143x143', 'TT_143x217', 'TT_217x217',
                'EE_100x100', 'EE_100x143', 'EE_100x217', 
                    'EE_143x143', 'EE_143x217', 'EE_217x217',
                'TE_100x100', 'TE_100x143', 'TE_100x217', 
                    'TE_143x143', 'TE_143x217', 'TE_217x217']
        key_i = list(range(len(keys)))
        key_index_dict = dict(zip(keys,key_i))
        i = key_index_dict[spec]
        print(spec, subarray_indices[i], subarray_indices[i+1])
        subcov = self.cov[subarray_indices[i]:subarray_indices[i+1], 
                              subarray_indices[i]:subarray_indices[i+1]]
        if not debug:
            return subcov
        
        ells = self.ells[subarray_indices[i]:subarray_indices[i+1]]
        cl = self.cls[subarray_indices[i]:subarray_indices[i+1]]
        err = np.sqrt(np.diag(subcov))
        
        if debug:
            return ells, cl, err, subcov