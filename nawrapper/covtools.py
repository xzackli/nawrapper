"""Power spectrum objects and utilities."""
import numpy as np
from scipy.signal import savgol_filter
from collections import defaultdict
import pymaster as nmt
import nawrapper.power as power

# import nawrapper.maptools as maptools
# import healpy as hp
# from pixell import enmap


def delta(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


class nacov:
    r"""Wrapper around the NaMaster covariance workspace object.

    This object contains the computationally intensive parts of covariance
    computation -- the coupling coefficients.
    """

    def __init__(
        self,
        namap1,
        namap2,
        mc_11,
        mc_12,
        mc_22,
        signal=None,
        noise=None,
        noise_smoothing_mode="savgol",
        smoothing_window=11,
        smoothing_polyorder=3,
        cosmic_variance=True,
        different_signals=False
    ):
        r"""
        Create a `nacov` object.

        Parameters
        ----------
        namap1: namap
            We use the mask in this namap to compute the mode-coupling
            matrices.
        namap2: namap
            We use the mask in this namap to compute the mode-coupling
            matrices.

        """
        self.lmax = mc_12.bins.lmax
        self.mc_11 = mc_11
        self.mc_12 = mc_12
        self.mc_22 = mc_22
        self.lb = mc_12.lb
        self.namap1 = namap1
        self.namap2 = namap2
        self.bins = mc_12.bins
        self.num_ell = len(mc_12.bins.get_effective_ells())
        self.cosmic_variance = cosmic_variance
        self.different_signals = different_signals

        self.Cl11 = power.compute_spectra(namap1, namap1, mc=mc_11)
        self.Cl12 = power.compute_spectra(namap1, namap2, mc=mc_12)
        self.Cl22 = power.compute_spectra(namap2, namap2, mc=mc_22)

        if signal is None:
            self.signal = {}
        else:
            self.signal = signal
            # process keys in signal, cutting at lmax
            for k in self.signal:
                self.signal[k] = self.signal[k][: self.lmax + 1]

        if noise is None:
            self.noise = {}
        else:
            self.noise = noise

        spec_list = []
        if namap1.has_temp and namap2.has_temp:
            spec_list += ["TT"]
        if namap1.has_pol and namap2.has_pol:
            spec_list += ["EE", "BB"]
        if namap1.has_temp and namap2.has_pol:
            spec_list += ["TE"]
        if namap1.has_pol and namap2.has_temp:
            spec_list += ["ET"]

        l_theory = np.arange(self.lmax + 1)
        Cl_dict = {"11": self.Cl11, "12": self.Cl12, "22": self.Cl22}


        for XY in spec_list:
            X, Y = XY

            if self.different_signals:  # store everything in signal
                for AB in ("11", "12", "22"):
                    signal_id = X + AB[0] + Y + AB[1]
                    if signal_id not in self.signal:
                        if noise_smoothing_mode == "savgol":
                            self.signal[signal_id] = np.abs(
                                self.smooth_and_interpolate(
                                    l_theory,
                                    np.interp(l_theory, self.lb, (Cl_dict[AB])[XY]),
                                    smoothing_window,
                                    smoothing_polyorder,
                                )
                            )
                        elif noise_smoothing_mode == "poly":
                            raise NotImplementedError(
                                "polynomial smoothing with differing signals")
                
            else:  # default behavior: estimate signal and noise
                if XY not in self.signal:
                    # linear interpolation
                    self.signal[XY] = self.smooth_and_interpolate(
                        l_theory,
                        np.interp(l_theory, self.lb, self.Cl12[XY]),
                        smoothing_window,
                        smoothing_polyorder,
                    )

                if (X + "1" + Y + "1") not in self.noise:

                    if noise_smoothing_mode == "savgol":
                        self.noise[X + "1" + Y + "1"] = np.abs(
                            self.smooth_and_interpolate(
                                l_theory,
                                np.interp(l_theory, self.lb, self.Cl11[XY]),
                                smoothing_window,
                                smoothing_polyorder,
                            )
                            - self.signal[XY]
                        )
                    elif noise_smoothing_mode == "poly":
                        self.noise[X + "1" + Y + "1"] = self.get_smooth_noise(
                            cb=self.Cl11[XY],
                            signal=self.signal[XY],
                            smoothing_polyorder=smoothing_polyorder,
                        )

                if (X + "2" + Y + "2") not in self.noise:
                    if noise_smoothing_mode == "savgol":
                        self.noise[X + "2" + Y + "2"] = np.abs(
                            self.smooth_and_interpolate(
                                l_theory,
                                np.interp(l_theory, self.lb, self.Cl22[XY]),
                                smoothing_window,
                                smoothing_polyorder,
                            )
                            - self.signal[XY]
                        )
                    elif noise_smoothing_mode == "poly":
                        self.noise[X + "2" + Y + "2"] = self.get_smooth_noise(
                            cb=self.Cl22[XY],
                            signal=self.signal[XY],
                            smoothing_polyorder=smoothing_polyorder,
                        )

        # any signal or noise not specified is set to zero
        self.noise = defaultdict(lambda: np.zeros(self.lmax + 1), self.noise)
        self.signal = defaultdict(lambda: np.zeros(self.lmax + 1), self.signal)

        self.beam = {}

        # set temperature beams
        if namap1.has_temp:
            self.beam["T1"] = (
                namap1.beam_temp[: self.lmax + 1] *
                namap1.pixwin_temp[: self.lmax + 1]
            )
        if namap2.has_temp:
            self.beam["T2"] = (
                namap2.beam_temp[: self.lmax + 1] *
                namap2.pixwin_temp[: self.lmax + 1]
            )

        # set polarization beams
        if namap1.has_pol:
            self.beam["E1"] = (
                namap1.beam_pol[: self.lmax + 1] *
                namap1.pixwin_pol[: self.lmax + 1]
            )
            self.beam["B1"] = (
                namap1.beam_pol[: self.lmax + 1] *
                namap1.pixwin_pol[: self.lmax + 1]
            )
        if namap2.has_pol:
            self.beam["E2"] = (
                namap2.beam_pol[: self.lmax + 1] *
                namap2.pixwin_pol[: self.lmax + 1]
            )
            self.beam["B2"] = (
                namap2.beam_pol[: self.lmax + 1] *
                namap2.pixwin_pol[: self.lmax + 1]
            )

        # for iterating over the output.
        # doing this explicity because it's actually shorter than some loops
        self.ordering = {
            (0, 0): ("TT",),
            (0, 2): ("TE", "TB"),
            (2, 0): ("ET", "BT"),
            (2, 2): ("EE", "EB", "BE", "BB"),
        }

        # covmat storage dict
        self.covmat = {}

    def get_field(self, namap_in, field_spin):
        if str(field_spin) == "0":
            return namap_in.field_spin0
        if str(field_spin) == "2":
            return namap_in.field_spin2

    def total_spec(self, XY, m1, m2):
        X, Y = XY

        # if signals are different, then we don't use the noise dict
        if self.different_signals:
            return (
                (self.signal[X + str(m1) + Y + str(m2)])
                * self.beam[X + str(m1)]
                * self.beam[Y + str(m2)])

        if self.cosmic_variance:
            return (
                (self.signal[XY] + self.noise[X + str(m1) + Y + str(m2)])
                * self.beam[X + str(m1)]
                * self.beam[Y + str(m2)]
            )
        else:
            return (
                (self.noise[X + str(m1) + Y + str(m2)])
                * self.beam[X + str(m1)]
                * self.beam[Y + str(m2)]
            )

    def cl_inputs(self, s1, s2, m1, m2):
        return [self.total_spec(XY, m1, m2) for XY in self.ordering[(s1, s2)]]

    def get_cov_input_spectra(self, spins):

        assert len(spins) == 4

        # i.e.
        # a_1 a_2 b_1 b_2
        # T_1 T_2 T_1 T_2

        # a1b1 is spins[0] and spins[2]
        a1b1 = self.cl_inputs(spins[0], spins[2], 1, 1)
        a1b2 = self.cl_inputs(spins[0], spins[3], 1, 2)
        a2b1 = self.cl_inputs(spins[1], spins[2], 2, 1)
        a2b2 = self.cl_inputs(spins[1], spins[3], 2, 2)

        return a1b1, a1b2, a2b1, a2b2

    def compute_subcovmat(self, spins):

        cw = nmt.NmtCovarianceWorkspace()
        cw.compute_coupling_coefficients(
            self.get_field(self.namap1, spins[0]),
            self.get_field(self.namap2, spins[1]),
            self.get_field(self.namap1, spins[2]),
            self.get_field(self.namap2, spins[3]),
            lmax=self.lmax,
        )

        a1b1, a1b2, a2b1, a2b2 = self.get_cov_input_spectra(spins)
        ordering_a = self.ordering[(spins[0], spins[1])]
        ordering_b = self.ordering[(spins[2], spins[3])]

        covar = nmt.gaussian_covariance(
            cw,
            spins[0],
            spins[1],
            spins[2],
            spins[3],
            a1b1,
            a1b2,
            a2b1,
            a2b2,
            self.mc_12.workspace_dict[(spins[0], spins[1])],
            wb=self.mc_12.workspace_dict[(spins[2], spins[3])],
        ).reshape([self.num_ell, len(ordering_a), self.num_ell, len(ordering_b)])

        for i, AB in enumerate(ordering_a):
            for j, CD in enumerate(ordering_b):
                self.covmat[AB + CD] = covar[:, i, :, j]

    def compute(self, verbose=False):
        # This is the time-consuming operation. You need to redo this for
        # every combination of masks. For ACT this is basically every time.

        # cl arrays passed to gaussian_covariance are:
        # a1b1, a1b2, a2b1, a2b2

        map_1_spins = []
        if self.namap1.has_temp:
            map_1_spins.append(0)
        if self.namap1.has_pol:
            map_1_spins.append(2)
        map_2_spins = []
        if self.namap2.has_temp:
            map_2_spins.append(0)
        if self.namap2.has_pol:
            map_2_spins.append(2)

        for a in map_1_spins:
            for b in map_2_spins:
                for c in map_1_spins:
                    for d in map_2_spins:
                        if verbose:
                            print(a, b, c, d)
                        self.compute_subcovmat(spins=(a, b, c, d))

    def get_smooth_noise(self, cb, signal, smoothing_polyorder):
        diff = np.abs(cb - self.bins.bin_cell(signal))
        poly = np.poly1d(
            np.polyfit(
                self.lb,
                np.log(diff),
                w=1 / np.log(np.sqrt(2 / (2 * self.lb + 1)) * diff),
                deg=smoothing_polyorder,
            )
        )
        return np.exp(poly(np.arange(self.lmax + 1)))

    """Smooth and interpolate a spectrum up to lmax.
    The goal of this is to produce a smooth theory curve for use in covariance.
    """

    def smooth_and_interpolate(self, lb, cb, smoothing_window, smoothing_polyorder):
        return np.interp(
            x=np.arange(self.lmax + 1),
            xp=lb,
            fp=savgol_filter(cb, smoothing_window, smoothing_polyorder),
            right=0,
        )


def compute_covmat(
    namap1,
    namap2,
    bins,
    mc_11=None,
    mc_12=None,
    mc_22=None,
    signal=None,
    noise=None,
    smoothing_window=11,
    smoothing_polyorder=3,
    cosmic_variance=True,
    different_signals=False,
    verbose=True,
):
    r"""Compute all of the relevant covariances between two namap.

    This computes all covarainces between two
    :py:class:`nawrapper.power.abstract_namap`
    for which there is information.

    Parameters
    ----------
    namap1 : :py:class:`nawrapper.power.namap_hp` or
        :py:class:`nawrapper.ps.namap_car`.
        The first map to compute correlations with.
    namap2 : :py:class:`nawrapper.power.namap_hp` or
        :py:class:`nawrapper.ps.namap_car`.
        To be correlated with `namap1`.

    bins : NaMaster NmtBin object
        The binning scheme used for this. This must match the 
        mode-coupling objects, if you pass those.

    mc_11 : :py:class:`nawrapper.power.mode_coupling` object (optional)
        This object contains precomputed mode-coupling matrices for the
        auto-spectrum of `namap1`.
    mc_12 : :py:class:`nawrapper.power.mode_coupling` object (optional)
        This object contains precomputed mode-coupling matrices for the
        cross-spectrum between `namap1` and `namap2`.
    mc_22 : :py:class:`nawrapper.power.mode_coupling` object (optional)
        This object contains precomputed mode-coupling matrices for the
        auto-spectrum of `namap2`.

    signal: dictionary
        The unbinned input spectra, must have keys TT, TE, EE, etc.
        Any unspecified signal spectra will be estimated from the cross-
        spectrum beteween namap1 and namap2.
    noise: dictionary
        The unbinned noise power spectra. Must have keys T1T1, T2T2,
        and others of the form XiYj, where X,Y are in T, E, B, and i,j are
        1 or 2. These refer to the noise spectra in various maps.
        Unspecified noise where `i == j` will be computed from the auto minus
        cross-spectra, and `i != j` will be set to zero.


    smoothing_window : int
        Savgol filter smoothing window, for smoothing the estimated
        signal and noise spectra.
    smoothing_polyorder : int
        Savgol filter polynomial order, for smoothing the estimated
        signal and noise spectra.

    Returns
    -------
    dictionary
        Binned covariances, with the relevant covariances (i.e. `TETT`)
        as dictionary keys. This also contains the bin centers as key 'ell'.
    """

    if mc_11 is None:
        mc_11 = power.mode_coupling(namap1, namap1, bins=bins, verbose=False)
    if mc_12 is None:
        mc_12 = power.mode_coupling(namap1, namap2, bins=bins, verbose=False)
    if mc_22 is None:
        mc_22 = power.mode_coupling(namap2, namap2, bins=bins, verbose=False)

    my_cov = nacov(
        namap1,
        namap2,
        mc_11=mc_11,
        mc_12=mc_12,
        mc_22=mc_22,
        signal=signal,
        noise=noise,
        smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_polyorder,
        cosmic_variance=cosmic_variance,
        different_signals=different_signals
    )
    my_cov.compute()

    return my_cov.covmat


def get_Nl(
    theta_fwhm=(10.0, 7.0, 5.0),
    sigma_T=(68.1, 42.6, 65.4),
    f_sky=0.6,
    l_min=2,
    l_max=2509,
    verbose=False,
):
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
    theta_fwhm = theta_fwhm * np.array([np.pi / 60.0 / 180.0])
    sigma_T = sigma_T * np.array([np.pi / 60.0 / 180.0])
    num_channels = len(theta_fwhm)
    f_sky = f_sky
    # ells = np.arange(l_max)

    # compute noise in muK**2, adapted from Monte Python
    noise_T = np.zeros(l_max, "float64")
    for l in range(l_min, l_max):
        noise_T[l] = 0
        for channel in range(num_channels):
            noise_T[l] += sigma_T[channel] ** -2 * np.exp(
                -l * (l + 1) * theta_fwhm[channel] ** 2 / 8.0 / np.log(2.0)
            )
        noise_T[l] = 1 / noise_T[l]
    return noise_T
