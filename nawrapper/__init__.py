from .ps import *

def debug_plot(m1, m2, xval=3000, yval=700, xr=50, yr=50, xoff=0, yoff=0, apo=None):
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(10,7), sharex='col', sharey='row')
    axes[0,0].plot( ((m1)[yval,xval-xr:xval+xr]), label='1' )
    axes[0,0].plot( ((m2)[yval+yoff,xval-xr+xoff:xval+xr+xoff]), label='2' )
    axes[0,0].legend()
    axes[0,1].plot( (m1)[yval-yr:yval+yr,xval] )
    axes[0,1].plot( (m2)[yval-yr+yoff:yval+yr+yoff,xval+xoff] )

    axes[1,0].plot( (m2[yval+yoff,xval-xr+xoff:xval+xr+xoff] - m1[yval,xval-xr:xval+xr]) )
    axes[1,1].plot( (m2[yval+yoff,xval-xr+xoff:xval+xr+xoff] - m1[yval,xval-xr:xval+xr]) )
    
    if apo is None:
        axes[2,0].plot( ((m1)[yval,:xr]), label='1' )
        axes[2,0].plot( ((m2)[yval+yoff,xoff:xr+xoff]), label='2' )
        axes[2,0].legend()
        axes[2,1].plot( (m1)[:yr,xval] )
        axes[2,1].plot( (m2)[yoff:yr+yoff,xval+xoff] )
    else:
        axes[2,0].plot( ((m1*apo)[yval,:xr]), label='1' )
        axes[2,0].plot( ((m2*apo)[yval+yoff,xoff:xr+xoff]), label='2' )
        axes[2,0].legend()
        axes[2,1].plot( (m1*apo)[:yr,xval] )
        axes[2,1].plot( (m2*apo)[yoff:yr+yoff,xval+xoff] )
    
    axes[0,0].set_title('x')
    axes[0,1].set_title('y')
    
    axes[0,0].set_ylabel('map level')
    axes[1,0].set_ylabel('$m_1 - m_2$')
    
    plt.tight_layout();
    return axes
    
    
def get_steve_apo_from_pixbox(shape, wcs, width, N_cut=0):
    from pixell import enmap
    
    apo = enmap.ones(shape, wcs=wcs)
    apo_i= np.arange(width)
    apo_profile = 1-(-np.sin(2.0*np.pi*(width-apo_i)/(width-N_cut))/(2.0*np.pi)
                     + (width-apo_i)/(width-N_cut))
    
    # set it up for x and y edges
    apo[:width,:] *= apo_profile[:, np.newaxis]
    apo[:,:width] *= apo_profile[np.newaxis, :]
    apo[-width:,:] *= apo_profile[::-1, np.newaxis]
    apo[:,-width:] *= apo_profile[np.newaxis, ::-1]
    return apo

def kfilter_map(m, apo, kx_cut, ky_cut, unpixwin=True, legacy_steve=False):
    from pixell import enmap
    import numpy as np
    
    alm = enmap.fft(m * apo, normalize=True)
    
    if unpixwin: # remove pixel window in Fourier space
        wy = np.sinc(np.fft.fftfreq(alm.shape[-2]))
        wx = np.sinc(np.fft.fftfreq(alm.shape[-1]))
        alm /= (wy[:,np.newaxis])
        alm /= (wx[np.newaxis,:])

    ly, lx = enmap.lmap(alm.shape, alm.wcs)
    kfilter_x = (np.abs(lx) > kx_cut)
    kfilter_y = (np.abs(ly) > ky_cut)

    if legacy_steve: # Steve's kspace filter appears to do this
        cut_x_k = np.unique(lx[(np.abs(lx) <= kx_cut )])
        cut_y_k = np.unique(ly[(np.abs(ly) <= ky_cut )])
        kfilter_x[ np.isclose(lx, cut_x_k[0]) ] = True # keep most positive x
        kfilter_y[ np.isclose(ly, cut_y_k[-1]) ] = True # keep most negative y

    result = enmap.ifft(alm * kfilter_x * kfilter_y, normalize=True).real
    result[apo > 0.0] = result[apo > 0.0] / apo[apo > 0.0]
    return result
