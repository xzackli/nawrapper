"""Power spectrum objects and utilities."""

import pymaster as nmt
import healpy as hp
import numpy as np
from pixell import enmap
import nawrapper.maputils as maputils


class nacov:
    r"""Wrapper around the NaMaster covariance workspace object.

    This object contains the computationally intensive parts of covariance 
    computation -- the coupling coefficients.
    """

    def __init__(self, namap1, namap2, mc, theory):
        r"""
        Create a `nacov` object.

        Parameters
        ----------
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
            self.cw,
            0, 0, 0, 0,  # Spins of the 4 fields
            [theory['T1T1']],
            [theory['T1T2']],
            [theory['T2T1']],
            [theory['T2T2']],
            mc.w00, wb=mc.w00)
        ct = covar_00_00.reshape(
            [self.lmax-1, 1, self.lmax-1, 1])
        self.covT1T2T1T2 = ct[:, 0, :, 0]
        
        
def get_Nl(theta_fwhm=(10., 7., 5.),
             sigma_T=(68.1, 42.6, 65.4),
             f_sky=0.6, l_min=2, l_max=2509,
             verbose=False):
    """
    Get Knox TT noise curve.
    Uses the Planck bluebook parameters by default.
    
    Parameters
    ----------
        theta_fwhm : list of float: 
            beam resolution in arcmin
        sigma_T : list of float
            temperature resolution in muK
        sigma_P : list of float
            polarization resolution in muK
        f_sky : float
            sky fraction covered
        l_min : int
            minimum ell for CMB power spectrum
        l_max : int
            maximum ell for CMB power spectrum
        verbose : bool
            flag for printing out debugging output

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

