import nawrapper as nw
import pymaster as nmt
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot


def kfilter_map(m, apo, kx_cut, ky_cut, unpixwin=True, legacy_steve=False):
    """Apply a k-space filter on a map.

    Parameters
    ----------
    m : enmap
        Input map which this function will filter.
    apo : enmap
        This map is a smooth tapering of the edges of the map, to be multiplied
        into the map prior to filtering. The filtered map is divided by the
        nonzero pixels of this map at the end. This is required because
        maps of actual data are unlikely to be periodic, which will induce
        ringing when one applies a k-space filter. To solve this, we taper the
        edges of the map to zero prior to filtering.

        See :py:func:`nawrapper.ps.get_steve_apo`
    kx_cut : float
        We cut modes with wavenumber :math:`|k_x| < k_x^{\mathrm{cut}}`.
    ky_cut : float
        We cut modes with wavenumber :math:`|k_y| < k_y^{\mathrm{cut}}`.
    unpixwin : bool
        Correct for the CAR pixel window if True.
    legacy_steve : bool
        Use a slightly different filter if True, to reproduce Steve's pipeline.
        Steve's k-space filter as of June 2019 had a bug where two k-space
        modes (the most positive cut mode in x and most negative cut mode in y)
        were not cut. This has a very small effect on the spectrum, but for
        reproducibility purposes we offer this behavior. By default we do not
        use this.

        To reproduce Steve's code behavior you should set
        set `legacy_steve=True` here and in the constructor for each
        :py:class:`nawrapper.ps.namap`.

    Returns
    -------
    result : enmap
        The map with the specified k-space filter applied.

    """
    alm = enmap.fft(m * apo, normalize=True)

    if unpixwin: # remove pixel window in Fourier space
        wy, wx = enmap.calc_window(m.shape)
        alm /= (wy[:,np.newaxis])
        alm /= (wx[np.newaxis,:])

    ly, lx = enmap.lmap(alm.shape, alm.wcs)
    kfilter_x = (np.abs(lx) >= kx_cut)
    kfilter_y = (np.abs(ly) >= ky_cut)

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
    """Generates a tapered mask at the edges of the box.

    Maps of actual data are unlikely to be periodic, which will induce
    ringing when one applies a k-space filter. To solve this, we taper the
    edges of the map to zero prior to filtering. This taper was written to match
    the output of Steve's power spectrum pipeline, for reproducibility.

    See :py:func:`nawrapper.ps.kfilter_map`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the map to be tapered.
    wcs : astropy wcs object
        WCS information for the map
    width : float
        width of the taper
    N_cut : int
        number of pixels to set to zero at the edges of the map

    Returns
    -------
    apo : enmap
        A smooth mask that is one in the center and tapers to zero at the edges.
    """
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


def get_distance(input_mask):
    """
    Construct a map of the distance to the nearest zero pixel in the input.

    Parameters
    ----------
    input_mask : enmap
        The input mask

    Returns
    -------
    dist : enmap
        This map is the same size as the `input_mask`. Each pixel of this map
        contains the distance to the nearest zero pixel at the corresponding
        location in the input_mask.
    """
    pixSize_arcmin= np.sqrt(input_mask.pixsize()*(60*180/np.pi)**2)
    dist = scipy.ndimage.distance_transform_edt( np.asarray(input_mask) )
    dist *= pixSize_arcmin/60
    return dist

def apod_C2(input_mask,radius):
    """
    Apodizes an input mask over a radius in degrees.

    A sharp mask will cause complicated mode coupling and ringing. One solution
    is to smooth out the sharp edges. This function applies the C2 apodisation
    as defined in 0903.2350_.

    .. _0903.2350: https://arxiv.org/abs/0903.2350

    Parameters
    ----------
    input_mask: enmap
        The input mask (must have all pixel values be non-negative).
    radius: float
        Apodization radius in degrees.

    Returns
    -------
    result : enmap
        The apodized mask.
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
    """Read bins from an ASCII file and create a NmtBin object.

    This is a utility function to read ACT binning files and create a
    NaMaster NmtBin object. The file format consists of three columns: the
    left bin edge (int), the right bin edge (int), and the bin center (float).

    Parameters
    ----------
    file : str
        Filename of the binning to read.
    lmax : int
        Maximum ell to create bins to.
    is_Dell : bool
        Will generate :math:`D_{\ell} = \ell (\ell + 1) C_{\ell} / (2 \pi)`
        if True, instead of :math:`C_{\ell}`. This may cause order 1% effects
        due to weighting inside bins.

    Returns
    -------
    b : pymaster.NmtBin
        Returns a NaMaster binning object.
    """
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
    """Read a beam file from disk.

    Parameters
    ----------
    beam_file : str
        The filename of the beam
    wcs : astropy.wcs object (optional)
        If you are using a non-standard pixelization, the maximum multipole
        that NaMaster requires may be different. If this is the case, pass
        your strange pixelization's WCS object here.

    Returns
    -------
    numpy array (float)
        Contains the beam :math:`B_{\ell}` where index `i` corresponds to
        multipole `i` (i.e. this array starts at ell = 0).
    """
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


def compute_spectra(namap1, namap2, bins=None, mc=None):
    """Compute all of the spectra between two maps.

    This computes all cross spectra between two :py:class:`nawrapper.ps.namap`.
    If both input namap objects have polarization information, then polarization
    cross-spectra will be computed. In all cases, TT is computed.

    Parameters
    ----------
    namap1 : :py:class:`nawrapper.ps.namap` object.
        The first map to compute correlations with.
    namap2 : :py:class:`nawrapper.ps.namap` object.
        To be correlated with `namap1`.
    bins (optional) : NaMaster NmtBin object
        At least one of `bins` or `mc` must be specified. If you specify
        `bins` (possibly from the output of :py:func:`nawrapper.ps.read_bins`)
        then a new mode coupling matrix will be computed within this function
        call. If you have already computed a relevant mode-coupling matrix,
        then pass `mc` instead.
    mc : :py:class:`nawrapper.ps.mode_coupling` object
        This object contains precomputed mode-coupling matrices.

    Returns
    -------
    Cb : dictionary
        Binned spectra, with the relevant cross spectra (i.e. 'TT', 'TE', 'EE')
        as dictionary keys. This also contains the bin centers as key 'ell'.
    """

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
    """
    This object organizes the various ingredients that are required for a
    map to be used in power spectra analysis. Each map has an associated

    1. I (optional QU) map
    2. mask, referring to the product of hits and point source mask
    3. beam transfer function

    This object also does k-space filtering upon creation, to avoid
    having to compute the spherical harmonics of the map multiple times.

    By default, we do not reproduce the output of Steve's code. For
    reproducibility, we do offer this functionality. Set the optional flag
    `legacy_steve=True` to offset the mask and the map by one pixel in each
    dimension.
    """

    def __init__(self,
                 map_I, mask, beam=None,
                 map_Q=None, map_U=None,
                 mask_pol=None,
                 shape=None, wcs=None,
                 kx=0, ky=0, kspace_apo=40, unpixwin=True,
                legacy_steve=False):
        """Create a new namap.
        
        This multiplies the apodized k-space taper into your mask. In 
        general, your mask should already have the edges tapered, so this
        will not change your results significantly.

        Parameters
        ----------
        map_I : pixell.enmap
            The intensity map you want to operate on.
        map_Q : pixell.enmap
            The Stokes Q map you want to operate on.
        map_I : pixell.enmap
            The Stokes U map you want to operate on.

        mask : pixell.enmap
            The mask for the map.
        beam: 1D numpy array
            Beam transfer function :math:`B_{\ell}`.

        shape : tuple
            In order to compute power spectra, every map and mask
            must be extracted into a common shape and WCS. This
            argument specifies this common shape.
        wcs : astropy wcs object
            Common WCS used for power spectra calculations.

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

        # k-space filter step (also correct for pixel window here!)
        apo = get_steve_apo(self.shape, self.wcs,
                                         kspace_apo)
        mask *= apo # multiply the apodized taper into your mask
        
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

        # construct the a_lm of the maps
        self.field_spin0 = nmt.NmtField(self.mask,
                                  [self.map_I],
                                  beam=self.beam,
                                  wcs=self.wcs, n_iter=0)
        if self.pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol,[self.map_Q, self.map_U],
                beam=self.beam, wcs=self.wcs, n_iter=0)


class mode_coupling:
    """Wrapper around the NaMaster workspace object.

    This object contains the computationally intensive parts of mode coupling.
    It stores `w00`, the (spin-0, spin-0) mode coupling matrix, and optionally
    `w02`, `w20`, and `w22`, the (spin-0, spin-2), (spin-2, spin-0) and
    (spin-2, spin-2) mode coupling matrices if both maps have polarization
    data.
    """

    def __init__(self, namap1, namap2, bins):
        """
        Create a `mode_coupling` object.

        namap1: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        namap2: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        bins : pymaster NmtBin object
            We generate binned mode coupling matrices with this NaMaster binning
            object.
        """


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
        """Utility function for computing mode-coupling-corrected spectra.

        Parameters
        ----------
        f_a : NmtField
            First field to correlate.
        f_b : NmtField
            Second field to correlate.
        wsp : NmtWorkspace
            Workspace containing the mode coupling matrix and binning.

        Returns
        -------
        numpy array
            Contains the TT spectrum if correlating two spin-0 maps, or
            the TE/TB or EE/EB spectra if working with (spin-0, spin-2)
            and (spin-2, spin-2) maps respectively."""
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled
