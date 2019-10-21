
import numpy as np
import nawrapper as nw

def test_basic():
    assert 1 == 1

    b = nw.get_unbinned_bins(10)
    assert np.all(b.get_effective_ells() == np.arange(2,11).astype(float))
