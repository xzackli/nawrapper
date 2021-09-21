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
    mc = nw.mode_coupling(m1, m2, bins=nw.create_binning(lmax))
    cov = nw.nacov(m1, m2, mc_11=mc, mc_12=mc, mc_22=mc)
    cov.compute()
    
    # test the covariance
    assert (np.std(np.sqrt(np.diag(cov.covmat['TTTT']))[40:120] / 
                   np.std(samples, axis=0)[40:120]
            )) < 0.5
    
    assert abs(np.mean(np.sqrt(np.diag(cov.covmat['TTTT']))[40:120] / 
               np.std(samples, axis=0)[40:120]
        ) - 1 ) < 0.5


    # test the mean spectrum
    assert (np.std(cl[40:120] / 
                   np.mean(samples, axis=0)[40:120]
            )) < 0.5

    assert abs(np.mean(cl[2:][40:120] / 
               np.mean(samples, axis=0)[40:120]
        ) - 1) < 0.5

def test_unbinned_bins():

    b = nw.create_binning(10)
    assert np.all(b.get_effective_ells() == np.arange(2,11).astype(float))


def test_kwargs():

    # HEALPix map resolution
    nside = 256

    # Let us first create a square mask:
    msk = np.zeros(hp.nside2npix(nside))
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    ph[np.where(ph > np.pi)[0]] -= 2 * np.pi
    msk[np.where((th < 2.63) & (th > 1.86) &
                (ph > -np.pi / 4) & (ph < np.pi / 4))[0]] = 1.

    # Now we apodize the mask. The pure-B formalism requires the mask to be
    # differentiable along the edges. The 'C1' and 'C2' apodization types
    # supported by mask_apodization achieve this.
    msk_apo = nmt.mask_apodization(msk, 10.0, apotype='C1')

    # Select a binning scheme
    b = nmt.NmtBin.from_nside_linear(nside, 16)
    leff = b.get_effective_ells()

    # Read power spectrum and provide function to generate simulated skies
    l, cltt, clee, clbb, clte = np.loadtxt('test/cls.txt', unpack=True)

    mp_t, mp_q, mp_u = hp.synfast([cltt, clee, clbb, clte],
        nside=nside, new=True, verbose=False)


    # This creates a spin-2 field without purifying either E or B
    f2_np = nmt.NmtField(msk_apo, [mp_q, mp_u])
    # This creates a spin-2 field with both pure E and B.
    f2_yp = nmt.NmtField(msk_apo, [mp_q, mp_u], purify_e=True, purify_b=True)

    w_np = nmt.NmtWorkspace()
    w_np.compute_coupling_matrix(f2_np, f2_np, b)
    w_yp = nmt.NmtWorkspace()
    w_yp.compute_coupling_matrix(f2_yp, f2_yp, b)

    # This wraps up the two steps needed to compute the power spectrum
    # once the workspace has been initialized
    def compute_master(f_a, f_b, wsp):
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled

    cl_np = compute_master(f2_np, f2_np, w_np)
    cl_yp = compute_master(f2_yp, f2_yp, w_yp)

    namap_np = nw.namap_hp(maps=(mp_t, mp_q, mp_u), masks=msk_apo, unpixwin=False, verbose=True)
    namap_yp = nw.namap_hp(maps=(mp_t, mp_q, mp_u), masks=msk_apo, unpixwin=False, purify_e=True, purify_b=True)
    nw_cl_np = nw.compute_spectra(namap_np, namap_np, bins=b)
    nw_cl_yp = nw.compute_spectra(namap_yp, namap_yp, bins=b)

    assert np.all(cl_np[0] ==  nw_cl_np['EE'])
    assert np.all(cl_yp[0] ==  nw_cl_yp['EE'])
