import numpy as np
import healpy as hp
import pymaster as nmt
import nawrapper as nw


def test_cov_TT():
    nside = 64
    lmax = nside * 3 - 1
    B_ell = hp.sphtfunc.gauss_beam(np.deg2rad(0.5), lmax=lmax)

    ells = np.arange(0,lmax)
    cl = np.zeros(len(ells))
    cl[2:] = 1/ells[2:]**2.5  # don't want monopole/dipole
    nl = np.exp( (ells / 100)**1.5 ) / 1e6
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

    print('Computing samples.')
    samples = [get_spec() for i in range(100)]

    m1, m2 = get_maps()

    print('Computing mode coupling matrix.')
    mc = nw.mode_coupling(m1, m2, bins=nw.get_unbinned_bins(lmax-1, nside=nside))
    cov = nw.nacov(m1, m2, mc_11=mc, mc_12=mc, mc_22=mc)
    cov.compute()
    
    # test the covariance
    assert (np.std(np.sqrt(np.diag(cov.covmat['TTTT']))[40:120] / 
                   np.std(samples, axis=0)[40:120]
            )) < 0.3
    
    assert abs(np.mean(np.sqrt(np.diag(cov.covmat['TTTT']))[40:120] / 
               np.std(samples, axis=0)[40:120]
        ) - 1 ) < 0.05


    # test the mean spectrum
    assert (np.std(cl[40:120] / 
                   np.mean(samples, axis=0)[40:120]
            )) < 0.1

    assert abs(np.mean(cl[2:][40:120] / 
               np.mean(samples, axis=0)[40:120]
        ) - 1) < 0.01

def test_unbinned_bins():

    b = nw.get_unbinned_bins(10)
    assert np.all(b.get_effective_ells() == np.arange(2,11).astype(float))

    
    