import numpy as np
from classy import Class

params = {'output': 'tCl pCl lCl', 'lensing':'yes'}
cosmo = Class()
cosmo.set(params); cosmo.compute()
cls = cosmo.lensed_cl(768)

for k in ['tt', 'te', 'ee', 'bb']:
    cls[k] *= 1e12
    
np.savetxt('data/example_cls.txt', np.array([cls['ell'], cls['tt'], cls['te'], cls['ee'], cls['bb']]).T, fmt=('%i %e %e %e %e'),
           header='ell TT TE EE BB',
           delimiter=',')