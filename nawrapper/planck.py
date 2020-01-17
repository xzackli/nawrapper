"""Utility functions for processing Planck maps into spectra."""
from __future__ import print_function
from astropy.io import fits, ascii
import healpy as hp
import numpy as np
import nawrapper as nw
import os

param_2018 = {
    "pol_efficiency": {"100": 0.9995, "143": 0.999, "217": 0.999},
    "nside": 2048,
    "lmax": 2508,
}


def load_planck_half_missions(
    map_dir,
    mask_dir,
    beam_dir,
    par=None,
    freq1="143",
    freq2="143",
    split1="1",
    split2="2",
):

    # make sure we're not updating with None
    if par is None:
        par = {}

    # update default parameters
    base_par_copy = param_2018.copy()
    par.update(base_par_copy)
    par = base_par_copy
    print(par)

    # read effective cross beams
    # beam files are only available as f1 x f2 with f1 \leq f2
    # splits are only 1 x 1, 1 x 2, 2 x 2 (i.e. no 2 x 1)

    if int(freq1) < int(freq2):
        beam_Wl_hdu = fits.open(
            os.path.join(
                beam_dir,
                "Wl_R3.01_plikmask_"
                + freq1
                + "hm"
                + split1
                + "x"
                + freq2
                + "hm"
                + split2
                + ".fits",
            )
        )
    elif int(freq1) > int(freq2):
        beam_Wl_hdu = fits.open(
            os.path.join(
                beam_dir,
                (
                    "Wl_R3.01_plikmask_"
                    + freq2
                    + "hm"
                    + split2
                    + "x"
                    + freq1
                    + "hm"
                    + split1
                    + ".fits"
                ),
            )
        )
    else:  # they are equal
        if int(split1) > int(split2):
            beam_Wl_hdu = fits.open(
                os.path.join(
                    beam_dir,
                    (
                        "Wl_R3.01_plikmask_"
                        + freq2
                        + "hm"
                        + split2
                        + "x"
                        + freq1
                        + "hm"
                        + split1
                        + ".fits"
                    ),
                )
            )
        else:
            beam_Wl_hdu = fits.open(
                os.path.join(
                    beam_dir,
                    (
                        "Wl_R3.01_plikmask_"
                        + freq1
                        + "hm"
                        + split1
                        + "x"
                        + freq2
                        + "hm"
                        + split2
                        + ".fits"
                    ),
                )
            )

    beam_TT = np.sqrt(np.maximum(1e-13, beam_Wl_hdu[1].data["TT_2_TT"][0]))
    beam_EE = np.sqrt(np.maximum(1e-13, beam_Wl_hdu[2].data["EE_2_EE"][0]))

    mfile_1 = os.path.join(
        map_dir, ("HFI_SkyMap_" + freq1 + "_2048_R3.01_halfmission-" + split1 + ".fits")
    )
    mfile_2 = os.path.join(
        map_dir, ("HFI_SkyMap_" + freq2 + "_2048_R3.01_halfmission-" + split2 + ".fits")
    )

    maskfile1 = os.path.join(
        mask_dir,
        (
            "COM_Mask_Likelihood-temperature-"
            + freq1
            + "-hm"
            + split1
            + "_2048_R3.00.fits"
        ),
    )
    maskfile2 = os.path.join(
        mask_dir,
        (
            "COM_Mask_Likelihood-temperature-"
            + freq2
            + "-hm"
            + split2
            + "_2048_R3.00.fits"
        ),
    )
    maskfile1_pol = os.path.join(
        mask_dir,
        (
            "COM_Mask_Likelihood-polarization-"
            + freq1
            + "-hm"
            + split1
            + "_2048_R3.00.fits"
        ),
    )
    maskfile2_pol = os.path.join(
        mask_dir,
        (
            "COM_Mask_Likelihood-polarization-"
            + freq2
            + "-hm"
            + split2
            + "_2048_R3.00.fits"
        ),
    )

    # read maps
    maps_1 = hp.read_map(mfile_1, field=(0, 1, 2), verbose=False)
    maps_2 = hp.read_map(mfile_2, field=(0, 1, 2), verbose=False)

    # read masks
    masks_1 = (
        hp.read_map(maskfile1, verbose=False),
        hp.read_map(maskfile1_pol, verbose=False),
    )
    masks_2 = (
        hp.read_map(maskfile2, verbose=False),
        hp.read_map(maskfile2_pol, verbose=False),
    )
    beams = (beam_TT, beam_EE)

    return maps_1, masks_1, maps_2, masks_2, beams


def preprocess_maps(maps, masks, missing_pixel, pol_eff, nside):

    maps = np.array(maps, copy=True)
    masks = np.array(masks, copy=True)

    masks[0][missing_pixel] = 0.0
    masks[1][missing_pixel] = 0.0

    # convert to muK
    for i in range(3):
        if i > 0:
            maps[i] *= 1e6 * pol_eff
        else:
            maps[i] *= 1e6

    # subtract monopole from temperature
    maps[0] = nw.maptools.sub_mono_di(maps[0], masks[0], nside)

    return maps, masks


def get_half_mission_namap(freq1, freq2, map_dir, mask_dir, beam_dir):

    # maps and masks with _raw are those which have not been proprocessed
    maps_1_raw, masks_1_raw, maps_2_raw, masks_2_raw, beams = load_planck_half_missions(
        map_dir=map_dir, mask_dir=mask_dir, beam_dir=beam_dir, freq1=freq1, freq2=freq2
    )

    pol_eff_1 = param_2018["pol_efficiency"][freq1]
    pol_eff_2 = param_2018["pol_efficiency"][freq2]

    maps_1, masks_1 = preprocess_maps(
        maps_1_raw, masks_1_raw, (maps_1_raw[0] < -1e30), pol_eff_1, 2048
    )
    maps_2, masks_2 = preprocess_maps(
        maps_2_raw, masks_2_raw, (maps_2_raw[0] < -1e30), pol_eff_2, 2048
    )

    m1 = nw.namap_hp(maps=maps_1, masks=masks_1, beams=beams, unpixwin=True)
    m2 = nw.namap_hp(maps=maps_2, masks=masks_2, beams=beams, unpixwin=True)
    return m1, m2


class PlanckCov:
    def __init__(self, ellspath, covpath="covmat.dat", clpath="data_extracted.dat"):
        self.cov = np.linalg.inv(np.genfromtxt(covpath))
        self.ells = np.genfromtxt(ellspath, usecols=0, unpack=True)
        self.cls = np.genfromtxt(clpath, usecols=1, unpack=True)

        self.keys = [
            "TT_100x100",
            "TT_143x143",
            "TT_143x217",
            "TT_217x217",
            "EE_100x100",
            "EE_100x143",
            "EE_100x217",
            "EE_143x143",
            "EE_143x217",
            "EE_217x217",
            "TE_100x100",
            "TE_100x143",
            "TE_100x217",
            "TE_143x143",
            "TE_143x217",
            "TE_217x217",
        ]

        subarray_indices = (
            np.arange(self.cov.shape[0] - 1)[(np.diff(self.ells) < 0)] + 1
        )
        self.sub_indices = np.hstack(([0], subarray_indices, len(self.cls)))
        key_ind = list(range(len(self.keys)))
        self.key_index_dict = dict(zip(self.keys, key_ind))

    def get_spec(self, spec):
        i = self.key_index_dict[spec]

        subcov = self.get_subcov(spec, verbose=False)
        ells = self.ells[self.sub_indices[i] : self.sub_indices[i + 1]]
        cl = self.cls[self.sub_indices[i] : self.sub_indices[i + 1]]
        err = np.sqrt(np.diag(subcov))

        return ells, cl, err, subcov

    def get_subcov(self, spec1, spec2=None, verbose=True):
        """
        spec: {TT, TE, or EE},  {'100x100', '143x143', '143x217', '217x217'}
        cross: string, i.e.
        
        returns tuple: 
            ells, cl, subcovariance matrix 
        """

        i = self.key_index_dict[spec1]
        if spec2 is None:
            j = i
        else:
            j = self.key_index_dict[spec2]

        if verbose:
            print(spec1, self.sub_indices[i], self.sub_indices[i + 1])
            print(spec2, self.sub_indices[j], self.sub_indices[j + 1])

        subcov = self.cov[
            self.sub_indices[i] : self.sub_indices[i + 1],
            self.sub_indices[j] : self.sub_indices[j + 1],
        ]
        return subcov


def get_planck_total_spectrum(theory_file, foreground_file, lmax):
    # read planck theory and foregrounds
    th = ascii.read(theory_file)
    fg = ascii.read(foreground_file)
    fg["BB143X143"] = fg["EE143X143"] * 0.0
    ell_fac = fg["l"][: lmax - 1] * (fg["l"][: lmax - 1] + 1) / (2 * np.pi)

    def ap(x):
        return np.hstack(((0.0, 0.0), x))

    signal_dict = {}
    for XY in ["TT", "TE", "EE", "BB"]:
        signal_dict[XY] = ap(
            (th[XY][: lmax - 1] + fg[XY + "143X143"][: lmax - 1]) / ell_fac
        )
    signal_dict["EB"] = signal_dict["EE"] * 0.0
    return signal_dict
