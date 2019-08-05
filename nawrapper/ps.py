import nawrapper as nw
import pymaster as nmt
import scipy
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

# adapted from PSpipe
def get_distance(input_mask):
    pixSize_arcmin= np.sqrt(input_mask.pixsize()*(60*180/np.pi)**2)
    dist = scipy.ndimage.distance_transform_edt( np.asarray(input_mask) )
    dist *= pixSize_arcmin/60
    return dist

def apod_C2(input_mask,radius):
    """
    @brief C2 apodisation as defined in https://arxiv.org/pdf/0903.2350.pdf
    
    input_mask: enmap.ndmap with values \geq 0
    radius: apodization radius in degrees
    """
    
    if radius==0:
        return input_mask
    else:
        dist=get_distance(input_mask)
        id=np.where(dist > radius)
        win=dist/radius-np.sin(2*np.pi*dist/radius)/(2*np.pi)
        win[id]=1
        
    return( enmap.ndmap(win, input_mask.wcs) )


def read_bins(file, lmax=7925, is_Dell=False):
    """Read bins from an ASCII file and create a NmtBin object."""
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


def read_beam(beam_file, wcs=None):
    if wcs is None:
        # assume it's a regular ACT map
        cdelts = 0.00833333
    else:
        cdelts = wcs.wcs.cdelt
    beam_t = np.loadtxt(beam_file)
    lmax_beam = int(180.0/abs(np.min(cdelts))) + 1
    beam_data = np.zeros(lmax_beam)
    beam_data[:beam_t.shape[0]] = beam_t[:,1].astype(float)
    return beam_data


def get_cross_spectra(namap_list, bins, mc=None):
    """Loop over all pairs and compute spectra.
    
    If mc is None, a new mode coupling matrix will be generated 
    for each pair. If mc is a nawrapper.mode_coupling object, then
    that mode coupling matrix is used for all the spectra.
    """
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


def compute_spectra(namap1, namap2, bins=None, mc=None):
    if bins is None and mc is None:
        raise ValueError(
            "You must specify either a binning or a mode coupling object.")
        
    if mc is None:
        mc = nw.mode_coupling(namap1, namap2, bins)
    
    Cb={}
    Cb['TT'] = mc.compute_master(
        namap1.field_spin0, namap2.field_spin0, mc.w00)[0]
        
    if namap1.pol and namap2.pol:
        spin1 = mc.compute_master(
            namap1.field_spin0, namap2.field_spin2, mc.w02)
        Cb['TE'] = spin1[0]
        Cb['TB'] = spin1[1]
        spin2 = mc.compute_master(
            namap1.field_spin2, namap2.field_spin2, mc.w22)
        Cb['EE'] = spin2[0]
        Cb['EB'] = spin2[1]
        Cb['BE'] = spin2[2]
        Cb['BB'] = spin2[3]
        spin1 = mc.compute_master(
            namap1.field_spin2, namap2.field_spin0, mc.w20)
        Cb['ET'] = spin1[0]
        Cb['BT'] = spin1[1]
      
    Cb['ell'] = mc.lb
    return Cb

class namap:
    
    def __init__(self, 
                 map_I, mask, beam=None,
                 map_Q=None, map_U=None,
                 mask_pol=None,
                 shape=None, wcs=None, 
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
            
        map : pixell.enmap
            Contains the map you want to operate on.
        mask : pixell.enmap
            The mask for the map.
        beam: 1D numpy array
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
        
        if wcs is None:
            self.shape = mask.shape
            self.wcs = mask.wcs
        else:
            self.shape = shape
            self.wcs = wcs
        
        self.legacy_steve = legacy_steve
            
        # needed to reproduce steve's spectra
        if legacy_steve:
            map_I.wcs.wcs.crpix += np.array([-1,-1])
            
        if beam is None:
            lmax_beam = int(180.0/abs(np.min(self.wcs.wcs.cdelt))) + 1
            self.beam = np.ones(lmax_beam)
        else:
            self.beam = beam
            
        # extract to common shape and wcs
        self.map_I = enmap.extract(map_I, self.shape, self.wcs)
        
        if map_Q is not None:
            self.pol = True
            self.map_Q = enmap.extract(map_Q, self.shape, self.wcs)
            self.map_U = enmap.extract(map_U, self.shape, self.wcs)
            self.mask_pol = enmap.extract(mask_pol, self.shape, self.wcs)
        else:
            self.pol = False
        
        self.mask = enmap.extract(mask, self.shape, self.wcs)
        
        if (kx > 0) or (ky > 0):
            apo = get_steve_apo(self.shape, self.wcs, 
                                             kspace_apo)
            self.map_I = nw.kfilter_map(
                self.map_I, apo, kx, ky, unpixwin=unpixwin,
                    legacy_steve=legacy_steve)
            
            if self.pol:
                self.map_Q = nw.kfilter_map(
                    self.map_Q, apo, kx, ky, unpixwin=unpixwin,
                    legacy_steve=legacy_steve)
                self.map_U = nw.kfilter_map(
                    self.map_U, apo, kx, ky, unpixwin=unpixwin,
                    legacy_steve=legacy_steve)
        
        self.field_spin0 = nmt.NmtField(self.mask, 
                                  [self.map_I], 
                                  beam=self.beam, 
                                  wcs=self.wcs, n_iter=0)
        if self.pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol,[self.map_Q, self.map_U],
                beam=self.beam, wcs=self.wcs, n_iter=0)
            
        
class mode_coupling:
    """Wrapper around the NaMaster workspace object."""
    
    def __init__(self, namap1, namap2, bins):
        self.lb = bins.get_effective_ells()
        self.w00 = nmt.NmtWorkspace()
        self.w00.compute_coupling_matrix(namap1.field_spin0, namap2.field_spin0, 
                                        bins, n_iter=0)
        
        if namap1.pol and namap2.pol:
            self.pol = True
            self.w02 = nmt.NmtWorkspace()
            self.w02.compute_coupling_matrix(
                namap1.field_spin0, namap2.field_spin2,
                bins, n_iter=0)
            self.w20 = nmt.NmtWorkspace()
            self.w20.compute_coupling_matrix(
                namap1.field_spin2, namap2.field_spin0,
                bins, n_iter=0)
            self.w22=nmt.NmtWorkspace()
            self.w22.compute_coupling_matrix(
                namap1.field_spin2, namap2.field_spin2,
                bins, n_iter=0)
        else:
            self.pol = False
            

        
    def compute_master(self, f_a, f_b, wsp) :
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled
        
#     def get_Cb(self, namap1, namap2):
#         cl_coupled = nmt.compute_coupled_cell(namap1.field, namap2.field)
#         return self.w0.decouple_cell(cl_coupled)