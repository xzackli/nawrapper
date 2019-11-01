"""Power spectrum objects and utilities."""
from __future__ import print_function
import pymaster as nmt
import healpy as hp
import numpy as np
from pixell import enmap
import nawrapper.maptools as maptools
import json
import os
import abc, six


def compute_spectra(namap1, namap2, 
                    bins=None, mc=None, lmax=None, verbose=True):
    r"""Compute all of the spectra between two maps.

    This computes all cross spectra between two :py:class:`nawrapper.ps.namap`
    for which there is information. For example, TE spectra will be computed
    only if the first map has a temperature map, and the second has a
    polarization map.

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
    
    
    if bins is None and mc is None and lmax is None:
        raise ValueError(
            "You must specify either a binning, lmax, "
            "or a mode coupling object.")
        
    if lmax is not None and bins is None and mc is None:
        if verbose: 
            print("Assuming unbinned and computing the mode coupling matrix.")
        bins = get_unbinned_bins(lmax) # will choose lmax, nside ignored

    if mc is None:
        mc = mode_coupling(namap1, namap2, bins)

    # compute the TEB spectra as appropriate
    Cb = {}
    if namap1.has_temp and namap2.has_temp:
        Cb['TT'] = mc.compute_master(
            namap1.field_spin0, namap2.field_spin0, mc.w00)[0]
    if namap1.has_temp and namap2.has_pol:
        spin1 = mc.compute_master(
            namap1.field_spin0, namap2.field_spin2, mc.w02)
        Cb['TE'] = spin1[0]
        Cb['TB'] = spin1[1]
    if namap1.has_pol and namap2.has_temp:
        spin1 = mc.compute_master(
            namap1.field_spin2, namap2.field_spin0, mc.w20)
        Cb['ET'] = spin1[0]
        Cb['BT'] = spin1[1]
    if namap1.has_pol and namap2.has_pol:
        spin2 = mc.compute_master(
            namap1.field_spin2, namap2.field_spin2, mc.w22)
        Cb['EE'] = spin2[0]
        Cb['EB'] = spin2[1]
        Cb['BE'] = spin2[2]
        Cb['BB'] = spin2[3]

    Cb['ell'] = mc.lb
    return Cb

# need this decorator to make this class abstract
@six.add_metaclass(abc.ABCMeta)
class abstract_namap():
    """Object for organizing map products."""

    @abc.abstractmethod
    def __init__(self, maps, masks=None, beams=None, 
                 unpixwin=True, verbose=True):
        r"""Generic abstract 

        Objects which inherit from this will organize the various ingredients 
        that are required for a map to be used in power spectra analysis. 
        Each map has an associated

        1. IQU maps
        2. mask, referring to the product of hits, point source mask, etc.
        3. beam transfer function

        Parameters
        ----------
        maps : ndarray or tuple
            The maps you want to operate on. This needs to be a tuple of 
            length 3, or an array of length `(3,) + map.shape`. 
        masks : ndarray or tuple
            The masks you want to operate on.
        beams: list or tuple
            The beams you want to use.
        unpixwin: bool
            If true, we account for the pixel window function when computing 
            power spectra. For healpix this is accomplished by modifying the 
            beam in-place.
        verbose : bool
            Print various information about what is being assumed. You should
            probably enable this the first time you try to run a particular 
            scenario, but you can set this to false if you find it's annoying 
            to have it printing so much stuff, like if you are computing many 
            spectra in a loop.

        """

        # check the input to make sure nothing funny is happening. we want
        # input tuples! A tuple of three maps.
        if hasattr(maps, '__len__'):
            if len(maps) == 3:
                self.map_I, self.map_Q, self.map_U = maps
            elif type(maps) is enmap.ndmap and maps.ndim==2:
                self.map_I, self.map_Q, self.map_U = maps, None, None
            elif type(maps) is np.ndarray and maps.ndim==1:
                self.map_I, self.map_Q, self.map_U = maps, None, None
            else:
                raise ValueError(
                    "Pass a tuple or list of maps, which needs to be of\n"
                    "length 3 (for IQU).\n" 
                    "For T only, try setting maps=(i_map, None, None).\n If "
                    "you wanted just pol maps, try maps=(None, q_map, u_map).")


        if isinstance(self.map_I, enmap.ndmap):
            self.mode = 'car'
        else:
            self.mode = 'healpix'

        # a tuple of two masks
        if hasattr(masks, '__len__'):
            if len(masks) == 2:
                self.mask_temp, self.mask_pol = masks
            elif isinstance(masks, np.ndarray):
                if verbose: print("Assuming the same mask for both I and QU.")
                self.mask_temp, self.mask_pol = masks, masks
        else:
            self.mask_temp, self.mask_pol = None, None

        # a tuple of two beams
        if hasattr(beams, '__len__'):
            if len(beams) == 2:
                self.beam_temp, self.beam_pol = beams[0].copy(), beams[1].copy()
            else:
                if verbose: print("Assuming the same beams for both I and QU.")
                self.beam_temp, self.beam_pol = beams.copy(), beams.copy()
        else:
            if verbose: print("Setting beam to 1.")
            self.beam_temp, self.beam_pol = None, None
        
        # remember which spectra to compute
        self.has_temp = (self.map_I is not None)
        self.has_pol = (self.map_Q is not None) and (self.map_U is not None)
        
        if verbose: 
            print('Creating a %s namap. temperature: %s, polarization: %s' % 
                (self.mode, self.has_temp, self.has_pol))

        if ((self.map_Q is None and self.map_U is not None) or
                (self.map_Q is not None and self.map_U is None)):
            raise ValueError("Q and U must both be specified for pol maps.")
        if ((self.map_I is None) and (self.map_U is None) and (self.map_Q is None)):
            raise ValueError("Must specify at least I or QU maps.")

        # set masks = 1 if not specified.
        if self.has_temp and self.mask_temp is None:
            self.mask_temp = self.map_I * 0.0 + 1.0
            if verbose: 
                print('mask_temp not specified, setting '
                      'temperature mask to one.')
        if self.has_pol and self.mask_pol is None:
            self.mask_pol = self.map_Q * 0.0 + 1.0
            if verbose: 
                print('mask_pol not specified, setting '
                      'polarization mask to one.')

    def set_beam(self, 
                 apply_healpix_window=False, verbose=False):
        """Set and extend the object's beam up to lmax."""
        if self.has_temp:
            beam_temp = np.ones(self.lmax_beam)
            if self.beam_temp is None:
                if verbose: print("beam_temp not specified, setting " +
                                  "temperature beam transfer function to 1.")
            else:
                beam_temp[:len(self.beam_temp)] = self.beam_temp
                beam_temp[len(self.beam_temp):] = self.beam_temp[-1]
            self.beam_temp = beam_temp
        if self.has_pol:
            beam_pol = np.ones(self.lmax_beam)
            if self.beam_pol is None:
                 if verbose: print("beam_pol not specified, setting " +
                                  "polarization beam transfer function to 1.")
            else:
                beam_pol[:len(self.beam_pol)] = self.beam_pol
                beam_pol[len(self.beam_pol):] = self.beam_pol[-1]
            self.beam_pol = beam_pol

class namap_car(abstract_namap):
    r"""Map container for CAR pixellization

    By default, we do not reproduce the output of Steve's code. We do offer
    this functionality: set the optional flag `legacy_steve=True` to offset
    the mask and the map by one pixel in each dimension. This constructor
    multiplies the apodized k-space taper into your mask.
    In general, your mask should already have the edges tapered, so this
    will not change your results significantly.
    """

    def __init__(self, maps, masks=None, beams=None, 
                 unpixwin=True, kx=0, ky=0, kspace_apo=40, legacy_steve=False, 
                 verbose=True, sub_shape=None, sub_wcs=None):

        super(namap_car, self).__init__(
            maps=maps, masks=masks, beams=beams, 
            unpixwin=unpixwin, verbose=verbose)

        self.shape = sub_shape
        self.wcs = sub_wcs
        if self.shape is None: self.shape = masks_temp.shape
        if self.wcs is None: self.wcs = masks_temp.wcs
        
        self.lmax_beam = int(180.0/abs(np.min(self.wcs.wcs.cdelt))) + 1
        self.set_beam(verbose=verbose)
        self.legacy_steve = legacy_steve
        # needed to reproduce steve's spectra
        if legacy_steve:
            self.map_I.wcs.wcs.crpix += np.array([-1, -1])
            if verbose: print('Applying legacy_steve correction.')
        self.extract_and_filter_CAR(kx, ky, kspace_apo,
                                    legacy_steve, unpixwin, verbose=verbose)

        if verbose: print("Computing spherical harmonics.\n")
        # construct the a_lm of the maps
        if self.has_temp:
            self.field_spin0 = nmt.NmtField(
                self.mask_temp, [self.map_I],
                beam=self.beam_temp, wcs=self.wcs, n_iter=0)
        if self.has_pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol, [self.map_Q, self.map_U],
                beam=self.beam_pol, wcs=self.wcs, n_iter=0)


    def extract_and_filter_CAR(self, kx, ky, kspace_apo, 
                               legacy_steve, unpixwin, verbose=False):
        """Extract and filter this initialized CAR namap.

        See constructor for parameters.
        """
        if verbose: print(
            ('Applying a k-space filter (kx=%s, ky=%s' % (kx, ky)) +
            ', apo=%s), unpixwin: %s' % (kspace_apo, unpixwin))

        # extract to common shape and wcs
        if self.has_temp:
            self.map_I = enmap.extract(self.map_I, self.shape, self.wcs)
            self.mask_temp = enmap.extract(self.mask_temp, self.shape, self.wcs)

        if self.has_pol:
            self.map_Q = enmap.extract(self.map_Q, self.shape, self.wcs)
            self.map_U = enmap.extract(self.map_U, self.shape, self.wcs)
            self.mask_pol = enmap.extract(self.mask_pol, self.shape, self.wcs)

        apo = maputils.get_steve_apo(self.shape, self.wcs, kspace_apo)

        # k-space filter step (also correct for pixel window here!)
        if self.has_temp:
            self.mask_temp *= apo  # multiply the apodized taper into your mask
            self.map_I = maputils.kfilter_map(
                self.map_I, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)

        if self.has_pol:
            self.mask_pol *= apo  # multiply the apodized taper into your mask
            self.map_Q = maputils.kfilter_map(
                self.map_Q, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)
            self.map_U = maputils.kfilter_map(
                self.map_U, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)


class namap_hp(abstract_namap):

    def __init__(self, maps, masks=None, beams=None, unpixwin=True, 
                 sub_monopole=True, sub_dipole=False, verbose=True, n_iter=3):

        super(namap_hp, self).__init__(
            maps=maps, masks=masks, beams=beams,
            unpixwin=unpixwin, verbose=verbose)

        self.nside = hp.npix2nside(len(self.map_I))
        self.lmax_beam = 3 * self.nside
        self.set_beam(verbose=verbose)
        self.pixwin_T, self.pixwin_P = hp.sphtfunc.pixwin(self.nside, pol=True)
        if verbose: print("Including the healpix pixel window function.")
        if self.has_temp: self.pixwin_T = self.pixwin_T[:len(self.beam_temp)]
        if self.has_pol: self.pixwin_P = self.pixwin_P[:len(self.beam_pol)]
        
        # this is written so that the beam is kept separate from the pixel window
        if self.has_temp: 
            self.beam_temp *= self.pixwin_T
        if self.has_pol: 
            self.beam_pol *= self.pixwin_P
        
        # construct the a_lm of the maps, depending on what data is available
        if verbose: print("Computing spherical harmonics.\n")
        if self.has_temp:
            self.field_spin0 = nmt.NmtField(
                self.mask_temp, [self.map_I],
                beam=self.beam_temp, n_iter=n_iter)
        if self.has_pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol, [self.map_Q, self.map_U],
                beam=self.beam_pol, n_iter=n_iter)
            

class mode_coupling:
    r"""Wrapper around the NaMaster workspace object.

    This object contains the computationally intensive parts of mode coupling.
    It stores `w00`, the (spin-0, spin-0) mode coupling matrix, and optionally
    `w02`, `w20`, and `w22`, the (spin-0, spin-2), (spin-2, spin-0) and
    (spin-2, spin-2) mode coupling matrices if both maps have polarization
    data.
    """

    def __init__(self, namap1=None, namap2=None, bins=None, mcm_dir=None):
        r"""
        Create a `mode_coupling` object.

        namap1: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        namap2: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        bins: pymaster NmtBin object
            We generate binned mode coupling matrices with this NaMaster binning
            object.

        mcm_dir: string
            Specify a directory which contains the workspace files for this 
            mode-coupling object.
        """

        if mcm_dir is not None:
            self.load_from_dir(mcm_dir)
        else:
            self.bins = bins
            self.lb = bins.get_effective_ells()
            
            if namap1.mode != namap2.mode: 
                raise ValueError(
                    'pixel types m1:%s, m2:%s incompatible' % 
                    (namap1.mode, namap2.mode))

            self.has_temp = namap1.has_temp and namap2.has_temp
            self.has_pol = namap1.has_pol and namap2.has_pol

            # compute whichever mode coupling matrices we have data for
            if self.has_temp:
                self.w00 = nmt.NmtWorkspace()
                self.w00.compute_coupling_matrix(
                    namap1.field_spin0, namap2.field_spin0, bins, n_iter=0)

            if self.has_temp and self.has_pol:
                self.w02 = nmt.NmtWorkspace()
                self.w02.compute_coupling_matrix(
                    namap1.field_spin0, namap2.field_spin2, bins, n_iter=0)
                self.w20 = nmt.NmtWorkspace()
                self.w20.compute_coupling_matrix(
                    namap1.field_spin2, namap2.field_spin0, bins, n_iter=0)

            if self.has_pol:
                self.w22 = nmt.NmtWorkspace()
                self.w22.compute_coupling_matrix(
                    namap1.field_spin2, namap2.field_spin2, bins, n_iter=0)

    def load_from_dir(self, mcm_dir):
        """Read information from a nawrapper mode coupling directory."""
        with open(os.path.join(mcm_dir,'mcm.json'), 'r') as read_file:
            data = (json.load(read_file))
            
            # convert lists into numpy arrays
            for bk in ['ells', 'bpws', 'weights']:
                data['bin_kwargs'][bk] = np.array( data['bin_kwargs'][bk])
            self.bins =  nmt.NmtBin(**data['bin_kwargs'])
            self.lb = self.bins.get_effective_ells()

            self.has_temp = data['has_temp']
            self.has_pol = data['has_pol']

            if self.has_temp:
                self.w00 = nmt.NmtWorkspace()
                self.w00.read_from(os.path.join(mcm_dir, data['w00']))

            if self.has_temp and self.has_pol:
                self.w02 = nmt.NmtWorkspace()
                self.w02.read_from(os.path.join(mcm_dir, data['w02']))
                self.w20 = nmt.NmtWorkspace()
                self.w20.read_from(os.path.join(mcm_dir, data['w20']))

            if self.has_pol:
                self.w22 = nmt.NmtWorkspace()
                self.w22.read_from(os.path.join(mcm_dir, data['w22']))
             

    def write_to_dir(self, mcm_dir):
        # create directory
        mkdir_p(mcm_dir)

        # extract bin kwargs
        lmax = self.bins.lmax
        bpws = -np.ones(lmax+1).astype(int)
        weights = np.ones(lmax+1)
        l_eff = self.bins.get_effective_ells()
        for i in range(len(l_eff)):
            bpws[self.bins.get_ell_list(i)] = i
            weights[self.bins.get_ell_list(i)] = self.bins.get_weight_list(i)

        # basic json
        data = {
            'has_temp': self.has_temp,
            'has_pol': self.has_pol
        }

        # write binaries
        if self.has_temp:
            self.w00.write_to(mcm_dir+'/w00.bin')
            data.update({'w00': 'w00.bin'})

        if self.has_temp and self.has_pol:
            self.w02.write_to(os.path.join(mcm_dir, data['w02']))
            self.w20.write_to(os.path.join(mcm_dir, data['w20']))
            data.update({'w02': 'w02.bin', 'w20': 'w20.bin'})

        if self.has_pol:
            self.w22.write_to(os.path.join(mcm_dir, data['w22']))
            data.update({'w22': 'w22.bin'})

        # write bin kwargs
        data['bin_kwargs'] = {
            'nside' : 2048,
            'lmax' : lmax,
            'ells' : np.arange(lmax+1).tolist(),
            'bpws' : bpws.tolist(),
            'weights' : weights.tolist()
        }

        with open(os.path.join(mcm_dir,'mcm.json'), 'w') as write_file:
            json.dump(data, write_file)


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
        maximum multipole to include bins
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
        cl_from_zero[Cb['ell'].astype(int)] = Cb[spec_key]
        weights = np.arange(lmax + 1) * (np.arange(lmax + 1) + 1)
        result[spec_key] = np.array(
            [np.sum((weights * cl_from_zero)[ell_sub]) /
             np.sum(weights[ell_sub]) for ell_sub in ell_sub_list])

    result['ell'] = lb
    return result


def mkdir_p(path):
    """Make a directory and its parents."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
