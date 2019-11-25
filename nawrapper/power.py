"""Power spectrum objects and utilities."""
from __future__ import print_function
import pymaster as nmt
from pymaster import nmtlib as lib
import healpy as hp
import numpy as np
from pixell import enmap
import nawrapper.maptools as maptools
import json
import os
import abc, six, errno


def compute_spectra(namap1, namap2, 
                    bins=None, mc=None, lmax=None, verbose=True):
    r"""Compute all of the spectra between two maps.

    This computes all cross spectra between two :py:class:`nawrapper.power.abstract_namap`
    for which there is information. For example, TE spectra will be computed
    only if the first map has a temperature map, and the second has a
    polarization map.

    Parameters
    ----------
    namap1 : :py:class:`nawrapper.power.namap_hp` or :py:class:`nawrapper.ps.namap_car`.
        The first map to compute correlations with.
    namap2 : :py:class:`nawrapper.power.namap_hp` or :py:class:`nawrapper.ps.namap_car`.
        To be correlated with `namap1`.
    bins : NaMaster NmtBin object (optional)
        At least one of `bins` or `mc` must be specified. If you specify
        `bins` (possibly from the output of :py:func:`nawrapper.power.read_bins`)
        then a new mode coupling matrix will be computed within this function
        call. If you have already computed a relevant mode-coupling matrix,
        then pass `mc` instead.
    mc : :py:class:`nawrapper.power.mode_coupling` object (optional)
        This object contains precomputed mode-coupling matrices.

    Returns
    -------
    Cb : dictionary
        Binned spectra, with the relevant cross spectra (i.e. 'TT', 'TE', 'EE')
        as dictionary keys. This also contains the bin centers as key 'ell'.

    """
    
    if (bins is None) and (mc is None) and (lmax is None):
        raise ValueError(
            "You must specify either a binning, lmax, "
            "or a mode coupling object.")
        
    if (lmax is not None) and (bins is None) and (mc is None):
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
#         r"""Generic abstract 

#         Objects which inherit from this will organize the various ingredients 
#         that are required for a map to be used in power spectra analysis. 
#         Each map has an associated

#         1. IQU maps
#         2. mask, referring to the product of hits, point source mask, etc.
#         3. beam transfer function

#         Parameters
#         ----------
#         maps : ndarray or tuple
#             The maps you want to operate on. This needs to be a tuple of 
#             length 3, or an array of length `(3,) + map.shape`. 
#         masks : ndarray or tuple
#             The masks you want to operate on.
#         beams: list or tuple
#             The beams you want to use.
#         unpixwin: bool
#             If true, we account for the pixel window function when computing 
#             power spectra. For healpix this is accomplished by modifying the 
#             beam in-place.
#         verbose : bool
#             Print various information about what is being assumed. You should
#             probably enable this the first time you try to run a particular 
#             scenario, but you can set this to false if you find it's annoying 
#             to have it printing so much stuff, like if you are computing many 
#             spectra in a loop.

#         """

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


        if isinstance(self.map_I, enmap.ndmap) or isinstance(self.map_Q, enmap.ndmap):
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
            self.beam_temp, self.beam_pol = None, None # we'll set to 1 later
        
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
                print('temperature mask not specified, setting '
                      'temperature mask to one.')
        if self.has_pol and self.mask_pol is None:
            self.mask_pol = self.map_Q * 0.0 + 1.0
            if verbose: 
                print('polarization mask not specified, setting '
                      'polarization mask to one.')

    def set_beam(self, 
                 apply_healpix_window=False, verbose=False):
        """Set and extend the object's beam up to lmax."""
        if self.has_temp:
            if self.beam_temp is None:
                if verbose: print("temperature beam not specified, setting " +
                                  "temperature beam to 1.")
                beam_temp = np.ones(self.lmax_beam)
            else:
                beam_temp = np.ones(max(self.lmax_beam, len(self.beam_temp)))
                beam_temp[:len(self.beam_temp)] = self.beam_temp
                beam_temp[len(self.beam_temp):] = self.beam_temp[-1]
            self.beam_temp = beam_temp
        if self.has_pol:
            if self.beam_pol is None:
                if verbose: print("polarization beam not specified, setting " +
                                  "P beam to 1.")
                beam_pol = np.ones(self.lmax_beam)
            else:
                beam_pol = np.ones(max(self.lmax_beam, len(self.beam_pol)))
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
                 verbose=True, sub_shape=None, sub_wcs=None,
                 purify_e=False, purify_b=False):
        r"""Generate a CAR map container

        This bundles CAR pixelization map products, in particular

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
        kx : float
            k-space horizontal filter mode, ky-modes with absolute value less than
            kx are filtered.
        kx : float
            k-space vertical filter mode, kx-modes with absolute value less than
            ky are filtered.
        kspace_apo : float
            optional parameter specifying width of apodization
        legacy_steve : bool
            Shifts the mask by 1,1 relative to the maps to preproduce Steve's code.
        verbose : bool
            Print various information about what is being assumed. You should
            probably enable this the first time you try to run a particular 
            scenario, but you can set this to false if you find it's annoying 
            to have it printing so much stuff, like if you are computing many 
            spectra in a loop.
        """
            
        super(namap_car, self).__init__(
            maps=maps, masks=masks, beams=beams, 
            unpixwin=unpixwin, verbose=verbose)

        self.shape = sub_shape
        self.wcs = sub_wcs
        if self.shape is None: self.shape = self.mask_temp.shape
        if self.wcs is None: self.wcs = self.mask_temp.wcs
        
        self.lmax_beam = int(180.0/abs(np.min(self.wcs.wcs.cdelt))) + 1
        self.set_beam(verbose=verbose)
        self.legacy_steve = legacy_steve
        # needed to reproduce steve's spectra
        if legacy_steve:
            if self.has_temp:
                self.map_I.wcs.wcs.crpix += np.array([-1, -1])
            if self.has_pol:
                self.map_Q.wcs.wcs.crpix += np.array([-1, -1])
                self.map_U.wcs.wcs.crpix += np.array([-1, -1])
            if verbose: print('Applying legacy_steve correction.')
        
        self.extract_and_filter_CAR(kx, ky, kspace_apo,
                                    legacy_steve, unpixwin, verbose=verbose)

        if verbose: print("Computing spherical harmonics.\n")
        # construct the a_lm of the maps
        if self.has_temp:
            self.field_spin0 = nmt.NmtField(
                self.mask_temp, [self.map_I],
                beam=self.beam_temp, wcs=self.wcs, n_iter=0, 
                purify_e=purify_e, purify_b=purify_b)
        if self.has_pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol, [self.map_Q, self.map_U],
                beam=self.beam_pol, wcs=self.wcs, n_iter=0,
                purify_e=purify_e, purify_b=purify_b)


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

        apo = maptools.get_steve_apo(self.shape, self.wcs, kspace_apo)

        # k-space filter step (also correct for pixel window here!)
        if self.has_temp:
            self.mask_temp *= apo  # multiply the apodized taper into your mask
            self.map_I = maptools.kfilter_map(
                self.map_I, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)

        if self.has_pol:
            self.mask_pol *= apo  # multiply the apodized taper into your mask
            self.map_Q = maptools.kfilter_map(
                self.map_Q, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)
            self.map_U = maptools.kfilter_map(
                self.map_U, apo, kx, ky, unpixwin=unpixwin,
                legacy_steve=legacy_steve)


class namap_hp(abstract_namap):

    def __init__(self, maps, masks=None, beams=None, unpixwin=True, 
                 verbose=True, n_iter=3,
                 purify_e=False, purify_b=False):
        r"""Generate a healpix map container

        This bundles healpix pixelization map products, in particular

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
        n_iter : int
            Number of spherical harmonic iterations, because healpix is not
            a very good pixellization.
        verbose : bool
            Print various information about what is being assumed. You should
            probably enable this the first time you try to run a particular 
            scenario, but you can set this to false if you find it's annoying 
            to have it printing so much stuff, like if you are computing many 
            spectra in a loop.
        """
        
        super(namap_hp, self).__init__(
            maps=maps, masks=masks, beams=beams,
            unpixwin=unpixwin, verbose=verbose)

        if self.has_temp:
            self.nside = hp.npix2nside(len(self.map_I))
        else:
            self.nside = hp.npix2nside(len(self.map_Q))

        self.lmax_beam = 3 * self.nside
        self.set_beam(verbose=verbose)
        self.pixwin_temp, self.pixwin_pol = hp.sphtfunc.pixwin(self.nside, pol=True)
        if verbose: print("Including the healpix pixel window function.")
        if self.has_temp: self.pixwin_temp = self.pixwin_temp[:len(self.beam_temp)]
        if self.has_pol: self.pixwin_pol = self.pixwin_pol[:len(self.beam_pol)]
        
        # this is written so that the beam is kept separate from the pixel window
        if self.has_temp:
            beam_temp = self.beam_temp[:len(self.pixwin_temp)] * self.pixwin_temp
        if self.has_pol:
            beam_pol = self.beam_pol[:len(self.pixwin_pol)] * self.pixwin_pol
        
        # construct the a_lm of the maps, depending on what data is available
        if verbose: print("Computing spherical harmonics.\n")
        if self.has_temp:
            self.field_spin0 = nmt.NmtField(
                self.mask_temp, [self.map_I],
                beam=beam_temp, n_iter=n_iter,
                purify_e=purify_e, purify_b=purify_b)
        if self.has_pol:
            self.field_spin2 = nmt.NmtField(
                self.mask_pol, [self.map_Q, self.map_U],
                beam=beam_pol, n_iter=n_iter,
                purify_e=purify_e, purify_b=purify_b)
            

class mode_coupling:
    r"""Wrapper around the NaMaster workspace object.

    This object contains the computationally intensive parts of mode coupling.
    It stores `w00`, the (spin-0, spin-0) mode coupling matrix, and optionally
    `w02`, `w20`, and `w22`, the (spin-0, spin-2), (spin-2, spin-0) and
    (spin-2, spin-2) mode coupling matrices if both maps have polarization
    data.
    """

    def __init__(self, namap1=None, namap2=None, bins=None, mcm_dir=None, 
        overwrite=False, verbose=True):
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

        if bins is None and mcm_dir is None:
            raise ValueError("Must specify binning, unless loading from disk.")

        self.mcm_dir = mcm_dir

        if (mcm_dir is not None) and (not overwrite) and os.path.isfile(os.path.join(self.mcm_dir, 'mcm.json')):
            if verbose: print("Loading mode-coupling matrices from disk.")
            self.load_from_dir(mcm_dir)
        else:
            if verbose: print("Computing new mode-coupling matrices.")
            self.bins = bins
            self.lb = bins.get_effective_ells()
            
            if namap1.mode != namap2.mode: 
                raise ValueError(
                    'pixel types m1:%s, m2:%s incompatible' % 
                    (namap1.mode, namap2.mode))

            self.workspace_dict = {}
            self.w00, self.w02, self.w20, self.w22 = None, None, None, None

            # compute whichever mode coupling matrices we have data for
            if namap1.has_temp and namap2.has_temp:
                self.w00 = nmt.NmtWorkspace()
                self.w00.compute_coupling_matrix(
                    namap1.field_spin0, namap2.field_spin0, bins, n_iter=0)
                self.workspace_dict[(0,0)] = self.w00

            if namap1.has_temp and namap2.has_pol:
                self.w02 = nmt.NmtWorkspace()
                self.w02.compute_coupling_matrix(
                    namap1.field_spin0, namap2.field_spin2, bins, n_iter=0)
                self.workspace_dict[(0,2)] = self.w02
            
            if namap1.has_pol and namap2.has_temp:
                self.w20 = nmt.NmtWorkspace()
                self.w20.compute_coupling_matrix(
                    namap1.field_spin2, namap2.field_spin0, bins, n_iter=0)
                self.workspace_dict[(2,0)] = self.w20

            if namap1.has_pol and namap2.has_pol:
                self.w22 = nmt.NmtWorkspace()
                self.w22.compute_coupling_matrix(
                    namap1.field_spin2, namap2.field_spin2, bins, n_iter=0)
                self.workspace_dict[(2,2)] = self.w22

            if self.mcm_dir is not None:
                self.write_to_dir(self.mcm_dir)
                if verbose: print("Saving mode-coupling matrices to " + self.mcm_dir)


    def load_from_dir(self, mcm_dir):
        """Read information from a nawrapper mode coupling directory."""
        with open(os.path.join(mcm_dir,'mcm.json'), 'r') as read_file:
            data = (json.load(read_file))
            
            # convert lists into numpy arrays
            for bk in ['ells', 'bpws', 'weights']:
                data['bin_kwargs'][bk] = np.array( data['bin_kwargs'][bk])
            self.bins =  nabin(**data['bin_kwargs'])
            self.lb = self.bins.get_effective_ells()

            self.workspace_dict = {}

            if 'w00' in data:
                self.w00 = nmt.NmtWorkspace()
                self.w00.read_from(os.path.join(mcm_dir, data['w00']))
                self.workspace_dict[(0,0)] = self.w00

            if 'w02' in data:
                self.w02 = nmt.NmtWorkspace()
                self.w02.read_from(os.path.join(mcm_dir, data['w02']))
                self.workspace_dict[(0,2)] = self.w02

            if 'w20' in data:
                self.w20 = nmt.NmtWorkspace()
                self.w20.read_from(os.path.join(mcm_dir, data['w20']))
                self.workspace_dict[(2,0)] = self.w20

            if 'w22' in data:
                self.w22 = nmt.NmtWorkspace()
                self.w22.read_from(os.path.join(mcm_dir, data['w22']))
                self.workspace_dict[(2,2)] = self.w22
             

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
        data = {}

        # write binaries
        if self.w00 is not None:
            data.update({'w00': 'w00.bin'})
            self.w00.write_to(os.path.join(mcm_dir, data['w00']))

        if self.w02 is not None:
            data.update({'w02': 'w02.bin'})
            self.w02.write_to(os.path.join(mcm_dir, data['w02']))

        if self.w20 is not None:
            data.update({'w20': 'w20.bin'})
            self.w20.write_to(os.path.join(mcm_dir, data['w20']))

        if self.w22 is not None:
            data.update({'w22': 'w22.bin'})
            self.w22.write_to(os.path.join(mcm_dir, data['w22']))

        # write bin kwargs
        data['bin_kwargs'] = {
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


class nabin(nmt.NmtBin):

    def __init__(self, lmax, bpws=None, ells=None, weights=None,
                 nlb=None, is_Dell=False, f_ell=None):

        # remember a dictionary so we can turn this into a JSON later
        self.init_dict = {
            'lmax' : lmax,
            'bpws' : bpws,
            'ells' : ells,
            'weights' : weights,
            'nlb' : nlb,
            'is_Dell' : is_Dell,
            'f_ell' : f_ell
        }
        # can't json a numpy array
        for array_key in ('bpws', 'ells', 'weights', 'f_ell'):
            if isinstance(self.init_dict[array_key], np.ndarray):
                self.init_dict[array_key] = self.init_dict[array_key].tolist()


        if (bpws is None) and (ells is None) and (weights is None) \
           and (nlb is None):
            raise KeyError("Must supply bandpower arrays or constant "
                           "bandpower width")

        if nlb is None:
            if (bpws is None) or (ells is None) or (weights is None):
                raise KeyError("Must provide bpws, ells and weights")
            if f_ell is None:
                if is_Dell:
                    f_ell = ells * (ells + 1.) / (2 * np.pi)
                else:
                    f_ell = np.ones(len(ells))
            self.bin = lib.bins_create_py(bpws.astype(np.int32),
                                          ells.astype(np.int32),
                                          weights, f_ell, int(lmax))
        else:
            self.bin = lib.bins_constant(nlb, lmax, int(is_Dell))
        self.lmax = lmax
        self.lb = self.get_effective_ells()


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
    b = nabin(lmax=lmax, bpws=bpws, ells=ells, weights=weights,
              is_Dell=is_Dell)
    return b


def create_binning(lmax, lmin=2, widths=1, weight_function=None):
    """Create a nabin object conveniently."""

    # if widths is an integer, create a constant 
    ells = np.arange(lmax+1).astype(int)
    bpws = -np.ones_like(ells).astype(int)  # Array of bandpower indices

    if not hasattr(widths, '__len__'):
        # we have an array of lengths!
        widths_list = [widths for i in 
            range(np.ceil((lmax-lmin)/widths).astype(int)+1)]
        widths = widths_list

    bin_left = lmin
    bin_num = 0
    for bin_num, w in enumerate(widths):
        bpws[bin_left:bin_left+w] = bin_num
        bin_left += w

    if weight_function is None:
        weights = weights = np.ones_like(ells)
    else:
        weights = [weight_function(l) for l in ells]
    
    b = nabin(
        lmax=lmax, bpws=bpws, ells=ells, 
        weights=weights, is_Dell=False)
    return b
    

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



def util_bin_FFTspec_CAR(data,modlmap,bin_edges):
    digitized = np.digitize(np.ndarray.flatten(modlmap), bin_edges,right=True)
    return np.bincount(digitized,(data).reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]

def util_bin_FFT_CAR(map1, map2, mask, beam1, beam2, lmax=8000):
    """Compute the FFTs, multiply, bin
    
    Beams are multiplied at bin centers. This is the worst
    job you could do for calculating power spectra.
    """
    beam_ells = np.arange(lmax+1)

    kmap1 = enmap.fft(map1*mask, normalize="phys")
    kmap2 = enmap.fft(map2*mask, normalize="phys")
    power = (kmap1*np.conj(kmap2)).real
    
    bin_edges = np.arange(0,lmax,40)
    centers = (bin_edges[1:] + bin_edges[:-1])/2.
    w2 = np.mean(mask**2.)
    modlmap = enmap.modlmap(map1.shape,map1.wcs)
    binned_power = util_bin_FFTspec_CAR(power/w2,modlmap,bin_edges)
    binned_power *= beam1[centers.astype(int)]
    binned_power *= beam2[centers.astype(int)]
    return centers, binned_power



