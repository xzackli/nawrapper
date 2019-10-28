"""Power spectrum objects and utilities."""

import pymaster as nmt
import healpy as hp
import numpy as np
from pixell import enmap
import nawrapper.maputils as maputils
import nawrapper.power as power
from scipy.signal import savgol_filter

class nacov:
    r"""Wrapper around the NaMaster covariance workspace object.

    This object contains the computationally intensive parts of covariance 
    computation -- the coupling coefficients.
    """

    def __init__(self, namap1, namap2, mc,
                 signal=None, noise=None,
                 smoothing_window=11, smoothing_polyorder=3):
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
        self.mc = mc
        self.namap1 = namap1
        self.namap2 = namap2
        
        self.cw00 = nmt.NmtCovarianceWorkspace()
        self.cw02 = nmt.NmtCovarianceWorkspace()
        self.cw20 = nmt.NmtCovarianceWorkspace()
        self.cw22 = nmt.NmtCovarianceWorkspace()
        
        self.num_ell = len(self.mc.bins.get_effective_ells())
        # This is the time-consuming operation. You need to redo this for
        # every combination of masks. For ACT this is basically every time.
        
        self.Cl12 = power.compute_spectra(namap1, namap2, mc=mc)
        self.Cl11 = power.compute_spectra(namap1, namap1, mc=mc)
        self.Cl22 = power.compute_spectra(namap2, namap2, mc=mc)
        
        if signal is None:
            self.signal = {}
        else:
            self.signal = signal
        
        if noise is None:
            self.noise = {}
        else:
            self.noise = noise
        
        
        spec_list = []
        if namap1.has_temp and namap2.has_temp: spec_list += ['TT']
        if namap1.has_pol and namap2.has_pol: spec_list += ['EE', 'BB']
            
        for XY in spec_list:
            if XY not in self.signal.keys():
                self.signal[XY] = self.smooth_and_interpolate(
                    mc.lb, self.Cl12[XY], 
                    smoothing_window, smoothing_polyorder)
            if XY not in self.noise.keys():
                self.noise[XY] = self.smooth_and_interpolate(
                    mc.lb, (self.Cl11[XY] + self.Cl22[XY])/2.0, 
                    smoothing_window, smoothing_polyorder) - self.signal[XY]
                self.noise[XY] = np.maximum(self.noise[XY], 0.0)
        
        
        
    def compute(self):
        
        namap1, namap2, mc = self.namap1, self.namap2, self.mc
        
        if namap1.has_temp and namap2.has_temp:
            # TT
            self.cw00.compute_coupling_coefficients(
                namap1.field_spin0, namap2.field_spin0, 
                namap1.field_spin0, namap2.field_spin0, 
                lmax=self.lmax)

            beam_tt = (namap1.beam_temp * namap2.beam_temp *
                       namap1.pixwin_T * namap2.pixwin_T)[:self.lmax+1]
            covar_00_00 = nmt.gaussian_covariance(
                self.cw00,
                0, 0, 0, 0,  # Spins of the 4 fields
                [(self.signal['TT'] + self.noise['TT']) * beam_tt],  # TT
                [self.signal['TT'] * beam_tt],  # TT
                [self.signal['TT'] * beam_tt],  # TT
                [(self.signal['TT'] + self.noise['TT']) * beam_tt],  # TT
                mc.w00, wb=mc.w00).reshape([self.num_ell, 1,
                                            self.num_ell, 1])
            self.covar_TT_TT = covar_00_00[:, 0, :, 0]
        
        ## EE
        if namap1.has_pol and namap2.has_pol:
            self.cw22.compute_coupling_coefficients(
                namap1.field_spin2, namap2.field_spin2, 
                namap1.field_spin2, namap2.field_spin2, 
                lmax=self.lmax)

            beam_ee = (namap1.beam_pol * namap2.beam_pol)[:self.lmax+1]
            covar_22_22 = nmt.gaussian_covariance(
                self.cw22, 2, 2, 2, 2,  # Spins of the 4 fields
                [(self.signal['EE']+self.noise['EE']) * beam_ee, (self.signal['EB']) * beam_ee,
                 (self.signal['EB']) * beam_ee, (self.signal['BB']+self.noise['BB']) * beam_ee],  # EE, EB, BE, BB
                [(self.signal['EE']) * beam_ee, (self.signal['EB']) * beam_ee,
                 (self.signal['EB']) * beam_ee, (self.signal['EB']) * beam_ee],  # EE, EB, BE, BB
                [(self.signal['EE']) * beam_ee, (self.signal['EB']) * beam_ee,
                 (self.signal['EB']) * beam_ee, (self.signal['EB']) * beam_ee],  # EE, EB, BE, BB
                [(self.signal['EE']+self.noise['EE']) * beam_ee, (self.signal['EB']) * beam_ee,
                 (self.signal['EB']) * beam_ee, (self.signal['BB']+self.noise['BB']) * beam_ee],  # EE, EB, BE, BB
                mc.w22, wb=mc.w22).reshape([self.num_ell, 4,
                                      self.num_ell, 4])

            self.covar_EE_EE = covar_22_22[:, 0, :, 0]
            self.covar_EE_EB = covar_22_22[:, 0, :, 1]
            self.covar_EE_BE = covar_22_22[:, 0, :, 2]
            self.covar_EE_BB = covar_22_22[:, 0, :, 3]
            self.covar_EB_EE = covar_22_22[:, 1, :, 0]
            self.covar_EB_EB = covar_22_22[:, 1, :, 1]
            self.covar_EB_BE = covar_22_22[:, 1, :, 2]
            self.covar_EB_BB = covar_22_22[:, 1, :, 3]
            self.covar_BE_EE = covar_22_22[:, 2, :, 0]
            self.covar_BE_EB = covar_22_22[:, 2, :, 1]
            self.covar_BE_BE = covar_22_22[:, 2, :, 2]
            self.covar_BE_BB = covar_22_22[:, 2, :, 3]
            self.covar_BB_EE = covar_22_22[:, 3, :, 0]
            self.covar_BB_EB = covar_22_22[:, 3, :, 1]
            self.covar_BB_BE = covar_22_22[:, 3, :, 2]
            self.covar_BB_BB = covar_22_22[:, 3, :, 3]
        
        
    """Smooth and interpolate a spectrum up to lmax.
    The goal of this is to produce a smooth theory curve for use in covariance.
    """
    def smooth_and_interpolate(self, lb, cb, smoothing_window, smoothing_polyorder):
        return np.interp(
            x=np.arange(self.lmax+1), 
            xp=lb, 
            fp=savgol_filter(cb,smoothing_window,smoothing_polyorder), 
            right=0)
        
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

