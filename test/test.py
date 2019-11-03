import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
import nawrapper as nw


def test_cov_TT():
    nside = 256
    lmax = nside * 3 - 1
    B_ell = hp.sphtfunc.gauss_beam(np.deg2rad(0.5), lmax=lmax)

    ells = np.arange(0,lmax)
    cl = np.zeros(len(ells))
    cl[2:] = 1/ells[2:]**2.5  # don't want monopole/dipole
    nl = np.exp( (ells / 100)**1.5 ) / 1e8
    nl[0:2] = 0.0
    window_func = hp.sphtfunc.pixwin(nside=nside)

    def get_maps():
        # apply the pixel window transfer function (pixwin=True)
        m = hp.synfast(B_ell[:lmax]**2 * cl, nside, verbose=False, pixwin=True)
        n1 = hp.synfast(B_ell[:lmax]**2 * nl, nside, verbose=False, pixwin=True)
        n2 = hp.synfast(B_ell[:lmax]**2 * nl,nside, verbose=False, pixwin=True)
        m1 = nw.namap_hp(maps=(m + n1, None, None), 
                      beams=B_ell, verbose=False, unpixwin=True)
        m2 = nw.namap_hp(maps=(m + n2, None, None), 
                      beams=B_ell, verbose=False, unpixwin=True)
        return m1, m2

    def get_spec():
        m1, m2 = get_maps()
        return nw.compute_spectra(m1, m2, lmax=lmax, verbose=False)['TT']

    samples = [get_spec() for i in range(50)]

    m1, m2 = get_maps()

    mc = nw.mode_coupling(m1, m2, bins=nw.get_unbinned_bins(lmax-1, nside=nside))
    cov = nw.nacov(m1, m2, mc)
    cov.compute()

    assert (
        np.std(np.sqrt(np.diag(cov.covar_TT_TT))[200:500] / 
               np.std(samples, axis=0)[200:500]
        )
    ) < 0.2


def test_unbinned_bins():
    assert 1 == 1

    b = nw.get_unbinned_bins(10)
    assert np.all(b.get_effective_ells() == np.arange(2,11).astype(float))

    
    