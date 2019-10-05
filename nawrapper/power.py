"""Power spectrum objects and utilities."""

import pymaster as nmt
import healpy as hp
import numpy as np
from pixell import enmap
import nawrapper.maputils as maputils


def compute_spectra(namap1, namap2, bins=None, mc=None):
    r"""Compute all of the spectra between two maps.

    This computes all cross spectra between two :py:class:`nawrapper.ps.namap`.
    If both input namap objects have polarization information, then polarization
    cross-spectra will be computed. In all cases, TT is computed.

    Parameters
    ----------
    namap1 : :py:class:`nawrapper.ps.namap` object.
        The first map to compute correlations with.
    namap2 : :py:class:`nawrapper.ps.namap` object.
        To be correlated with `namap1`.
    bins : NaMaster NmtBin object (optional)
        At least one of `bins` or `mc` must be specified. If you specify
        `bins` (possibly from the output of :py:func:`nawrapper.ps.read_bins`)
        then a new mode coupling matrix will be computed within this function
        call. If you have already computed a relevant mode-coupling matrix,
        then pass `mc` instead.
    mc : :py:class:`nawrapper.ps.mode_coupling` object (optional)
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
        mc = mode_coupling(namap1, namap2, bins)

    Cb = {}
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
    """Object for organizing map products."""

    def __init__(self,
                 # basic parameters
                 map_I, mask, beam=None,
                 map_Q=None, map_U=None,
                 mask_pol=None, beam_pol=None,
                 unpixwin=True,

                 # CAR specific parameters
                 shape=None, wcs=None,
                 kx=0, ky=0, kspace_apo=40,
                 legacy_steve=False,

                 # healpix specific parameters
                 nside=None,
                 sub_monopole=False, sub_dipole=False
                 ):
        r"""Create a new namap.

        This object organizes the various ingredients that are required for a
        map to be used in power spectra analysis. Each map has an associated

        1. I (optional QU) map
        2. mask, referring to the product of hits, point source mask, etc.
        3. beam transfer function

        This object also does k-space filtering upon creation, to avoid
        having to compute the spherical harmonics of the map multiple times.

        By default, we do not reproduce the output of Steve's code. We do offer
        this functionality: set the optional flag `legacy_steve=True` to offset
        the mask and the map by one pixel in each dimension. This constructor
        multiplies the apodized k-space taper into your mask.
        In general, your mask should already have the edges tapered, so this
        will not change your results significantly.

        This objects defaults to CAR maps. You can instead use healpix maps
        by specifying an `nside`. Pass in IQU maps and masks as
        numpy arrays, and don't use the CAR specific parameters (
        `shape`, `wcs`, and the k-space filtering options). **You cannot compute
        spectra between a CAR and a healpix map.** If this is your goal, convert
        both maps to one pixelization first.

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

        nside : int
            nside describes the size of a healpix map. Pass this if you want
            to use healpix pixelization instead of CAR.
        sub_monopole : bool
            Turn on to fit and remove the monopole from the I map. Only for
            healpix.
        sub_dipole : bool
            Turn on to fit and remove the dipole from the I map. Only for
            healpix. Cannot be used without sub_monopole.

        """
        self.map_I, self.map_Q, self.map_U = map_I, map_Q, map_U
        self.mask, self.mask_pol = mask, mask_pol

        # check to see if there is polarization information
        if map_Q is not None:
            self.pol = True
            if mask_pol is None:
                mask_pol = mask
        else:
            self.pol = False
        
        if beam_pol is None:
            beam_pol = beam

        # branch here based on CAR or healpix
        if nside is None:
            # assuming CAR if nside is not specified
            self.mode = 'CAR'
            self.__initialize_CAR_map(map_I, mask, beam, map_Q, map_U,
                                      mask_pol, beam_pol, unpixwin, shape, wcs,
                                      kx, ky, kspace_apo, legacy_steve)
        else:
            # construct healpix maps if nside is specified
            self.mode = 'healpix'
            self.__initialize_hp_map(map_I, mask, beam, map_Q, map_U, 
                                     mask_pol, beam_pol, unpixwin,
                                     nside, sub_monopole, sub_dipole)

    def __initialize_CAR_map(self, map_I, mask, beam, 
                             map_Q, map_U, mask_pol, beam_pol, 
                             unpixwin, shape, wcs,
                             kx, ky, kspace_apo, legacy_steve):
        if wcs is None:  # inherit the mask's shape and WCS if not specified
            shape = mask.shape
            wcs = mask.wcs
        self.shape = shape
        self.wcs = wcs
        self.lmax_beam = int(180.0/abs(np.min(self.wcs.wcs.cdelt))) + 1
        self.legacy_steve = legacy_steve
        # needed to reproduce steve's spectra
        if legacy_steve:
            self.map_I.wcs.wcs.crpix += np.array([-1, -1])
        self.set_beam(beam, beam_pol)
        self.extract_and_filter_CAR(kx, ky, kspace_apo,
                                    legacy_steve, unpixwin)

        # construct the a_lm of the maps
        self.field_spin0 = nmt.NmtField(
            self.mask, [self.map_I],
            beam=self.beam, wcs=self.wcs, n_iter=0)
        if self.pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol, [self.map_Q, self.map_U],
                beam=self.beam_pol, wcs=self.wcs, n_iter=0)

    def __initialize_hp_map(self,
                            map_I, mask, beam, 
                            map_Q, map_U, mask_pol, beam_pol, 
                            unpixwin,
                            nside, sub_monopole, sub_dipole):
        self.nside = nside
        self.lmax_beam = 3 * nside
        self.set_beam(beam, beam_pol)
        pixwin_T, pixwin_P = hp.sphtfunc.pixwin(self.nside, pol=True)
        if unpixwin:  # apply healpix pixel window
            self.beam *= pixwin_T[:len(self.beam)]
            self.beam_pol *= pixwin_P[:len(self.beam_pol)]
        if sub_monopole:  # subtract TT monopole and dipole
            self.map_I = maputils.sub_mono_di(self.map_I, self.mask,
                                              nside, sub_dipole)
        # construct the a_lm of the maps
        self.field_spin0 = nmt.NmtField(
            self.mask, [self.map_I],
            beam=self.beam, n_iter=0)
        if self.pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol, [self.map_Q, self.map_U],
                beam=self.beam_pol, n_iter=0)

    def set_beam(self, beam, beam_pol, apply_healpix_window=False):
        """Set and extend the object's beam."""
        if beam is None:
            self.beam = np.ones(self.lmax_beam)
            self.beam_pol = np.ones(self.lmax_beam)
        else:
            self.beam = np.zeros(self.lmax_beam)
            self.beam[:len(beam)] = beam
            self.beam_pol = np.zeros(self.lmax_beam)
            self.beam_pol[:len(beam_pol)] = beam_pol

    def extract_and_filter_CAR(self, kx, ky, kspace_apo, legacy_steve, unpixwin):
        """Extract and filter this initialized CAR namap.

        See constructor for parameters.
        """
        # extract to common shape and wcs
        self.map_I = enmap.extract(self.map_I, self.shape, self.wcs)

        if self.pol:
            self.map_Q = enmap.extract(self.map_Q, self.shape, self.wcs)
            self.map_U = enmap.extract(self.map_U, self.shape, self.wcs)
            self.mask_pol = enmap.extract(self.mask_pol, self.shape, self.wcs)

        self.mask = enmap.extract(self.mask, self.shape, self.wcs)

        # k-space filter step (also correct for pixel window here!)
        apo = maputils.get_steve_apo(self.shape, self.wcs, kspace_apo)
        self.mask *= apo  # multiply the apodized taper into your mask

        self.map_I = maputils.kfilter_map(
            self.map_I, apo, kx, ky, unpixwin=unpixwin,
            legacy_steve=legacy_steve)

        if self.pol:
            self.map_Q = maputils.kfilter_map(
                self.map_Q, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)
            self.map_U = maputils.kfilter_map(
                self.map_U, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)


class mode_coupling:
    r"""Wrapper around the NaMaster workspace object.

    This object contains the computationally intensive parts of mode coupling.
    It stores `w00`, the (spin-0, spin-0) mode coupling matrix, and optionally
    `w02`, `w20`, and `w22`, the (spin-0, spin-2), (spin-2, spin-0) and
    (spin-2, spin-2) mode coupling matrices if both maps have polarization
    data.
    """

    def __init__(self, namap1, namap2, bins):
        r"""
        Create a `mode_coupling` object.

        namap1: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        namap2: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        bins : pymaster NmtBin object
            We generate binned mode coupling matrices with this NaMaster binning
            object.

        """
        self.bins = bins
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
            self.w22 = nmt.NmtWorkspace()
            self.w22.compute_coupling_matrix(
                namap1.field_spin2, namap2.field_spin2,
                bins, n_iter=0)
        else:
            self.pol = False

    def compute_master(self, f_a, f_b, wsp):
        """Compute mode-coupling-corrected spectra.

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
            and (spin-2, spin-2) maps respectively.

        """
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled


def read_bins(file, lmax=7925, is_Dell=False):
    r"""Read bins from an ASCII file and create a NmtBin object.

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
    bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
    for i, (bl, br) in enumerate(zip(binleft[1:], binright[1:])):
        bpws[bl:br+1] = i

    weights = np.array([1.0 / np.sum(bpws == bpws[l]) for l in range(lmax)])
    b = nmt.NmtBin(2048, bpws=bpws, ells=ells, weights=weights,
                   lmax=lmax, is_Dell=is_Dell)
    return b


def get_unbinned_bins(lmax, nside=None):
    """Generate an unbinned NaMaster binning for l>1.

    Parameters
    ----------
    lmax : int
        maximum multipole to include bins to
    nside : int
        The NmtBin actually chooses the maximum multipole as the
        minimum of `lmax`, `3*nside-1`.

    Returns
    -------
    b : NmtBin
        contains a bin for every ell up to lmax

    """
    bpws_ell = np.arange(lmax+1)
    bpws = bpws_ell - 2  # array of bandpower indices
    bpws[bpws < 0] = -1  # set ell=0,1 to -1 : i.e. not included
    weights = np.ones_like(bpws)
    if nside is None:
        nside = lmax
    b = nmt.NmtBin(nside, bpws=bpws, ells=bpws_ell, weights=weights, lmax=lmax)
    return b


def read_beam(beam_file):
    r"""Read a beam file from disk.

    This function will interpolate a beam file with columns
    :math:`\ell, B_{\ell}` to a 1D array where index corresponds to
    :math:`\ell`.

    Parameters
    ----------
    beam_file : str
        The filename of the beam

    multiply_healpix_window : bool

    Returns
    -------
    numpy array (float)
        Contains the beam :math:`B_{\ell}` where index `i` corresponds to
        multipole `i` (i.e. this array starts at ell = 0).

    """
    beam_t = np.loadtxt(beam_file)
    max_beam_l = np.max(beam_t[:, 0].astype(int))
    beam_data = np.zeros(max_beam_l)
    beam_data = np.interp(np.arange(max_beam_l),
                          fp=beam_t[:, 1].astype(float), xp=beam_t[:, 0])
    return beam_data


def bin_spec_dict(Cb, binleft, binright, lmax):
    """Bin an unbinned spectra dictionary with a specified l^2 binning."""
    ell_sub_list = [np.arange(l, r) for (l, r) in zip(binleft, binright+1)]
    lb = np.array([np.sum(ell_sub) / len(ell_sub) for ell_sub in ell_sub_list])

    result = {}
    for spec_key in Cb:
        ell_sub_list = [np.arange(l, r) for (l, r) in zip(binleft, binright+1)]
        lb = np.array([np.sum(ell_sub) / len(ell_sub) for ell_sub in ell_sub_list])
        cl_from_zero = np.zeros(lmax + 1)
        cl_from_zero[Cb['ell'].astype(int)] = Cb[spec_key] * 1e12
        weights = np.arange(lmax + 1) * (np.arange(lmax + 1) + 1)
        result[spec_key] = np.array(
            [np.sum((weights * cl_from_zero)[ell_sub]) /
             np.sum(weights[ell_sub]) for ell_sub in ell_sub_list])

    result['ell'] = lb
    return result
