from ..NanoObjects import SphereNanoObject
import numpy as np

from nose.tools import assert_raises
from numpy.testing import assert_almost_equal, assert_array_almost_equal

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


def test_add():
    ''' Test the adding of NanoObjects'''

    pargs1 = {'radius' : 10}
    sphere1 = SphereNanoObject(pargs1)
    pargs2 = {'radius' : 10, 'x0': 10}
    sphere2 = SphereNanoObject(pargs2)

    sphere3 = sphere1 + sphere2

    x = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x,x)
    R = np.array([X, Y, X*0])

    V3 = sphere3.V(R)

    assert_array_almost_equal(V3[40:60, 50], np.array([0., 0., 0., 0., 0., 15.,
                                                       15., 15., 30., 30., 30.,
                                                       30., 15., 15., 15., 0.,
                                                       0., 0., 0., 0.]))
