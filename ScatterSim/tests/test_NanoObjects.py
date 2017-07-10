from ..NanoObjects import SphereNanoObject
from numpy.testing import assert_almost_equal
import numpy as np
from nose.tools import assert_raises

# tests should just be functions that return errors
# if behaviour is not as expected
def test_sphereNanoObject():
    # don't specify radius, defaults to 1?
    sphere = SphereNanoObject()
    assert_almost_equal(sphere.pargs['radius'], 1.0)

    sphere = SphereNanoObject(pargs={'radius': 10})
    q = np.array([1,1,1])
    V1 = sphere.V(q)
    q = np.array([[1,1,1]]).T
    V2 = sphere.V(q)

    # transpose of this should raise ValueError
    assert_raises(ValueError, sphere.V, q.T)


