"""Example script, copy of the quickstart in the documentation."""

import nawrapper.power as nw
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap

# map information
shape, wcs = enmap.geometry(shape=(1024, 1024),
                            res=np.deg2rad(0.5/60.), pos=(0, 0))

# create power spectrum information
ells = np.arange(0, 6000, 1)
ps = np.zeros(len(ells))
ps[2:] = 1/ells[2:]**2.5  # don't want monopole/dipole

# generate a realization
imap = enmap.rand_map(shape, wcs, ps[np.newaxis, np.newaxis])
# plt.imshow(imap)

mask = enmap.ones(imap.shape, imap.wcs)

N_point_sources = 50
for i in range(N_point_sources):
    mask[
         np.random.randint(low=0, high=mask.shape[0]),
         np.random.randint(low=0, high=mask.shape[1])] = 0
# apodize the pixels to make fake sources
point_source_map = 1-nw.apod_C2(mask, 0.1)

imap += point_source_map  # add our sources to the map
mask = nw.apod_C2(mask, 0.5)  # apodize the mask

# # plot our cool results
# fig, axes = plt.subplots(1, 2, figsize=(8,16))
# axes[0].imshow(imap)
# axes[1].imshow(mask)

ells = np.arange(0, len(ps), 1)
nl = np.ones(len(ells)) * 1e-8

noise_map_1 = enmap.rand_map(shape, wcs, nl[np.newaxis, np.newaxis])
noise_map_2 = enmap.rand_map(shape, wcs, nl[np.newaxis, np.newaxis])

# plt.plot(ps, label="ps")
# plt.plot(nl, label="noise")
# plt.yscale('log')
# plt.legend()

namap_1 = nw.namap(map_I=imap + noise_map_1, mask=mask)
namap_2 = nw.namap(map_I=imap + noise_map_2, mask=mask)

binfile = '../notebooks/data/BIN_ACTPOL_50_4_SC_low_ell'
bins = nw.read_bins(binfile)
mc = nw.mode_coupling(namap_1, namap_2, bins)

Cb = nw.compute_spectra(namap_1, namap_2, mc=mc)

plt.plot(ps, 'k-', label='input')
plt.plot(Cb['ell'], Cb['TT'], 'r.', label='computed')
plt.legend()
plt.yscale('log')
