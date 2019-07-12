import nawrapper as nw
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot


def kfilter_map(m, apo, kx_cut, ky_cut, unpixwin=True, legacy_steve=False):
    """Apply k-space filter on a map."""
    alm = enmap.fft(m * apo, normalize=True)
    
    if unpixwin: # remove pixel window in Fourier space
        wy = np.sinc(np.fft.fftfreq(alm.shape[-2]))
        wx = np.sinc(np.fft.fftfreq(alm.shape[-1]))
        alm /= (wy[:,np.newaxis])
        alm /= (wx[np.newaxis,:])

    ly, lx = enmap.lmap(alm.shape, alm.wcs)
    kfilter_x = (np.abs(lx) > kx_cut)
    kfilter_y = (np.abs(ly) > ky_cut)

    if legacy_steve: # Steve's kspace filter appears to do this
        cut_x_k = np.unique(lx[(np.abs(lx) <= kx_cut )])
        cut_y_k = np.unique(ly[(np.abs(ly) <= ky_cut )])
        # keep most negative kx and most positive ky
        kfilter_x[ np.isclose(lx, cut_x_k[0]) ] = True 
        kfilter_y[ np.isclose(ly, cut_y_k[-1]) ] = True 

    result = enmap.ifft(alm * kfilter_x * kfilter_y,
                        normalize=True).real
    result[apo > 0.0] = result[apo > 0.0] / apo[apo > 0.0]
    return result

    
def get_steve_apo(shape, wcs, width, N_cut=0):
    """Generates a taper at the edges of the box."""
    apo = enmap.ones(shape, wcs=wcs)
    apo_i= np.arange(width)
    apo_profile = 1-(-np.sin(2.0*np.pi*(width-apo_i)/
                             (width-N_cut))/(2.0*np.pi)
                     + (width-apo_i)/(width-N_cut))
    
    # set it up for x and y edges
    apo[:width,:] *= apo_profile[:, np.newaxis]
    apo[:,:width] *= apo_profile[np.newaxis, :]
    apo[-width:,:] *= apo_profile[::-1, np.newaxis]
    apo[:,-width:] *= apo_profile[np.newaxis, ::-1]
    return apo

def get_bins_from_file(file, lmax=7925, is_Dell=True):
    binleft, binright, bincenter = np.loadtxt( 
        file, unpack=True, 
        dtype={'names': ('binleft', 'binright', 'bincenter'),
               'formats': ('i', 'i', 'f')})
    ells = np.arange(lmax)
    bpws=-1+np.zeros_like(ells) #Array of bandpower indices
    for i, (bl, br) in enumerate(zip(binleft[1:], binright[1:])):
        bpws[bl:br+1] = i

    weights = np.array([1.0 / np.sum(bpws == bpws[l]) for l in range(lmax)])
    b = nmt.NmtBin(2048, bpws=bpws, ells=ells, weights=weights, 
                   lmax=lmax, is_Dell=is_Dell)
    return b

def get_cross_spectra(namap_list, bins, mc=None):
    ps_dict = {}
    cross_spectra = []

    
    # we can reuse the workspace w0 from earlier
    for i in range(len(namap_list)):
        for j in range(len(namap_list)):
            if i >= j:
                if mc is None:
                    mc_temp = nw.mode_coupling(
                        namap_list[i], namap_list[j], bins)
                    Cb = mc_temp.Cb
                else:
                    Cb = mc.get_Cb(namap_list[i], namap_list[j])
                ps_dict[f"{i},{j}"] = Cb
                if i > j:
                    cross_spectra += [Cb[0]]

    return ps_dict, cross_spectra

class namap:
    
    def __init__(self, shape, wcs, 
                 map_data=None, mask_data=None, beam_data=None,
                 map_file=None, mask_file=None, beam_file=None,
                 kx=0, ky=0, kspace_apo=40, unpixwin=True,
                legacy_steve=False):
        """Create a namap.
        
        Specify either filenames (map_file, mask_file, beam_file)
        or objects.
        
        Parameters
        ----------
        shape : tuple
            In order to compute power spectra, every map and mask
            must be extracted into a common shape and WCS. This 
            argument specifies this common shape.
        wcs : astropy wcs object
            Common WCS used for power spectra calculations.
            
        map_data : pixell.enmap
            Contains the map you want to operate on.
        mask_data : pixell.enmap
            The mask for the map.
        beam_data: 1D numpy array
            Beam transfer function :math:`B_{\ell}`.
        map_file : string
            filename of map FITS file
        mask_file : string
            filename of mask FITS file
        beam_file : string
            filename of ASCII file containing beam transfer function
            
        kx : float
            wavenumber to cut in abs(kx)
        ky : float
            wavenumber to cut in abs(ky)
        legacy_steve : boolean
            If true, adds (-1,-1) to input map `wcs.crpix` 
            to mimic the behavior of Steve's code.
        """
        
        self.shape = shape
        self.wcs = wcs
        self.legacy_steve = legacy_steve
        
        # make sure inputs are good
        assert not ( (map_data is None) and (map_file is None) )
        assert not ( (mask_data is None) and (mask_file is None) )
        assert not ( (beam_data is None) and (beam_file is None) )
        
        if map_data is None:
            map_data = enmap.read_fits(map_file)
        if mask_data is None:    
            mask_data = enmap.read_fits(mask_file)
        if beam_data is None:
            beam_t = np.loadtxt(beam_file)
            lmax_beam = abs(int(180.0/np.min(wcs.wcs.cdelt))) + 1
            beam_data = np.zeros(lmax_beam)
            beam_data[:beam_t.shape[0]] = beam_t[:,1].astype(float)
            
        # needed to reproduce steve's spectra
        if legacy_steve:
            map_data.wcs.wcs.crpix += np.array([-1,-1])
            
        self.beam_data = beam_data
        # extract to common shape and wcs
        self.map_data = enmap.extract(map_data, shape, wcs)
        self.mask_data = enmap.extract(mask_data, shape, wcs)
        
        if (kx > 0) or (ky > 0):
            apo = get_steve_apo(self.shape, self.wcs, 
                                             kspace_apo)
            self.map_data = nw.kfilter_map(
                self.map_data, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)
        
        self.field = nmt.NmtField(self.mask_data, 
                                  [self.map_data], beam=self.beam_data, wcs=wcs, n_iter=0)

        
class mode_coupling:
    
    def __init__(self, namap1, namap2, bins):
        cl_coupled = nmt.compute_coupled_cell(namap1.field, namap2.field)
        self.lb = bins.get_effective_ells()
        self.w0 = nmt.NmtWorkspace()
        self.w0.compute_coupling_matrix(namap1.field, namap2.field, 
                                        bins, n_iter=0)
        self.Cb = self.w0.decouple_cell(cl_coupled)
        
    def get_Cb(self, namap1, namap2):
        cl_coupled = nmt.compute_coupled_cell(namap1.field, namap2.field)
        return self.w0.decouple_cell(cl_coupled)