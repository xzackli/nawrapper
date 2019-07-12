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
    

