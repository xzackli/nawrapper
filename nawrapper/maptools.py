"""Utility functions for preprocessing maps."""

import scipy
import numpy as np
from pixell import enmap
import healpy as hp


def kfilter_map(m, apo, kx_cut, ky_cut, unpixwin=True, legacy_steve=False):
    r"""Apply a k-space filter on a map.

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

        See :py:func:`nawrapper.ps.rectangular_apodization`
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

    if unpixwin:  # remove pixel window in Fourier space
        wy, wx = enmap.calc_window(m.shape)
        alm /= wy[:, np.newaxis]
        alm /= wx[np.newaxis, :]

    ly, lx = enmap.lmap(alm.shape, alm.wcs)
    kfilter_x = np.abs(lx) >= kx_cut
    kfilter_y = np.abs(ly) >= ky_cut

    if legacy_steve:  # Steve's kspace filter appears to do this
        cut_x_k = np.unique(lx[(np.abs(lx) <= kx_cut)])
        cut_y_k = np.unique(ly[(np.abs(ly) <= ky_cut)])
        # keep most negative kx and most positive ky
        kfilter_x[np.isclose(lx, cut_x_k[0])] = True
        kfilter_y[np.isclose(ly, cut_y_k[-1])] = True

    result = enmap.ifft(alm * kfilter_x * kfilter_y, normalize=True).real
    result[apo > 0.0] = result[apo > 0.0] / apo[apo > 0.0]
    return result


def rectangular_apodization(shape, wcs, width, N_cut=0):
    r"""Generate a tapered mask at the edges of the box.

    Maps of actual data are unlikely to be periodic, which will induce
    ringing when one applies a k-space filter. To solve this, we taper the
    edges of the map to zero prior to filtering. This taper was written to
    match the output of Steve's power spectrum pipeline, for reproducibility.

    See :py:func:`nawrapper.maptools.kfilter_map`.

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
        A smooth mask that is one in the center and tapers to zero at the
        edges.

    """
    apo = enmap.ones(shape, wcs=wcs)
    apo_i = np.arange(width)
    apo_profile = 1 - (
        -np.sin(2.0 * np.pi * (width - apo_i) / (width - N_cut)) / (2.0 * np.pi)
        + (width - apo_i) / (width - N_cut)
    )

    # set it up for x and y edges
    apo[:width, :] *= apo_profile[:, np.newaxis]
    apo[:, :width] *= apo_profile[np.newaxis, :]
    apo[-width:, :] *= apo_profile[::-1, np.newaxis]
    apo[:, -width:] *= apo_profile[np.newaxis, ::-1]
    return apo


def get_distance(input_mask):
    r"""
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
    pixSize_arcmin = np.sqrt(input_mask.pixsize() * (60 * 180 / np.pi) ** 2)
    dist = scipy.ndimage.distance_transform_edt(np.asarray(input_mask))
    dist *= pixSize_arcmin / 60
    return dist


def apod_C2(input_mask, radius):
    r"""
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
    if radius == 0:
        return input_mask
    else:
        dist = get_distance(input_mask)
        id = np.where(dist > radius)
        win = dist / radius - np.sin(2 * np.pi * dist / radius) / (2 * np.pi)
        win[id] = 1

    return enmap.ndmap(win, input_mask.wcs)


def legacy_steve_shift(target_enmap):
    """Applies a one-pixel shift to the WCS for reproducing Steve spectra.
    
    Arguments:
        target_enmap {enmap} -- the enmap whose WCS we will modify in place.
    """
    target_enmap.wcs.wcs.crpix += np.array([-1, -1])


def sub_mono_di(map_in, mask_in, nside, sub_dipole=True, verbose=False):
    """Subtract monopole and dipole from a healpix map."""
    map_masked = hp.ma(map_in)
    map_masked.mask = mask_in < 1
    mono, dipole = hp.pixelfunc.fit_dipole(map_masked)
    if verbose:
        print("mono:", mono, ", dipole:", dipole)
    m = map_in.copy()
    npix = hp.nside2npix(nside)
    bunchsize = npix // 24
    for ibunch in range(npix // bunchsize):  # adapted from healpy
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, False)
        if sub_dipole:
            m.flat[ipix] -= dipole[0] * x
            m.flat[ipix] -= dipole[1] * y
            m.flat[ipix] -= dipole[2] * z
        m.flat[ipix] -= mono
    return m


def get_cmb_sim_hp(signal, nside_out):
    """Generate a healpix realization of the spectra.

    Parameters
    ----------
    signal : dictionary
        dictionary containing spectra starting from ell=0 with keys
        TT, EE, BB, TE.
    nside_out : int
        output map resolution
    """
    cmb_sim = hp.synfast(
        cls=(signal["TT"], signal["EE"], signal["BB"], signal["TE"]),
        nside=nside_out,
        pixwin=True,
        verbose=False,
        new=True,
    )
    return cmb_sim
