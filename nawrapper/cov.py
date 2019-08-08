"""Power spectrum objects and utilities."""

import pymaster as nmt
import healpy as hp
import numpy as np
from pixell import enmap
import nawrapper.maputils as maputils


def compute_cov(namap1, namap2, bins=None, mc=None):
    r"""Compute all of the covariance matrices between two maps.

    This computes all cross covariances between two :py:class:`nawrapper.ps.namap`.
    If both input namap objects have polarization information, then polarization
    covariances will be computed. In all cases, TT is computed.

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
    cw = nmt.NmtCovarianceWorkspace()
    # This is the time-consuming operation
    # Note that you only need to do this once,
    # regardless of spin
    cw.compute_coupling_coefficients(f1t, f2t, f1t, f2t, lmax=lmax)
    


class nacov:
    r"""Wrapper around the NaMaster covariance workspace object.

    This object contains the computationally intensive parts of covariance 
    computation -- the coupling coefficients.
    """

    def __init__(self, namap1, namap2, mc, theory):
        r"""
        Create a `nacov` object.

        namap1: namap
            We use the mask in this namap to compute the mode-coupling matrices.
        namap2: namap
            We use the mask in this namap to compute the mode-coupling matrices.

        """
        self.lmax = mc.bins.lmax
        self.cw = nmt.NmtCovarianceWorkspace()
        # This is the time-consuming operation. You need to redo this for
        # every combination of masks. For ACT this is basically every time.
        
        # TT
        self.cw.compute_coupling_coefficients(
            namap1.field_spin0, namap2.field_spin0, 
            namap1.field_spin0, namap2.field_spin0, 
            lmax=self.lmax)

        covar_00_00 = nmt.gaussian_covariance(
            cw,
            0, 0, 0, 0,  # Spins of the 4 fields
            [clth + get_Nl(l_max=2509)], 
            [clth],  # TT
            [clth],  # TT
            [clth + get_Nl(l_max=2509)],  # TT
            mc.w00, wb=mc.w00)
        
        
def get_Nl(theta_fwhm=(10., 7., 5.),
             sigma_T=(68.1, 42.6, 65.4),
             f_sky=0.6, l_min=2, l_max=2500,
             verbose=False):
    """
    Get Knox TT noise curve.
    Uses the Planck bluebook parameters by default.
    
    Parameters
    ----------
        theta_fwhm (list of float): beam resolution in arcmin
        sigma_T (list of float): temperature resolution in muK
        sigma_P (list of float): polarization resolution in muK
        f_sky (float): sky fraction covered
        l_min (int): minimum ell for CMB power spectrum
        l_max (int): maximum ell for CMB power spectrum
        verbose (boolean): flag for printing out debugging output
    """

    # convert from arcmin to radians
    theta_fwhm = theta_fwhm * np.array([np.pi/60./180.])
    sigma_T = sigma_T * np.array([np.pi/60./180.])
    num_channels = len(theta_fwhm)
    f_sky = f_sky
    ells = np.arange(l_max)

    # compute noise in muK**2, adapted from Monte Python
    noise_T = np.zeros(l_max, 'float64')
    for l in range(l_min, l_max):
        noise_T[l] = 0
        for channel in range(num_channels):
            noise_T[l] += sigma_T[channel]**-2 *\
                np.exp(
                    -l*(l+1)*theta_fwhm[channel]**2/8./np.log(2.))
        noise_T[l] = 1/noise_T[l]
    return noise_T
        