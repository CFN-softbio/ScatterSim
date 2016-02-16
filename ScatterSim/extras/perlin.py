#!/usr/bin/env python
# $Id: perlin.py,v 1.11 2001/05/31 04:56:23 kevint Exp $
#
# Algorithm by Ken Perlin, http://mrl.nyu.edu/~perlin/
#
# This Python implementation by Kevin Turner,
#  http://www.purl.org/wiki/python/KevinTurner
#  <acapnotic@users.sourceforge.net>
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be enjoyable, but
# without any warranty; without even the implied warranty of merchantability
# or fitness for a particular purpose.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA

"""A Perlin noise generator, for all dimensions.

An implementation of Ken Perlin's OSCAR-winning algorithm for making
noise, as described by him on http://www.noisemachine.com/ .  Its main
use is in procedural textures.

This module suppies the class PerlinNoise, see its documentation for
usage details.

Perlin states his algorithm is not patented, and while this module is
heavily based on Perlin's algorithm and his suggestions on
implementing it, it does not borrow from any copyrighted code.
"""

# A grid of n dimensions

# For each grid point, choose a pseudo-random gradient vector G

# pick texture point P, and for each surrounding grid point Q,
# do G dot (P - Q)

# Interpolate results using cross-fade curve

import random
from Numeric import *
import math

_FORCE_GENERIC_CASE = 0
_DISABLE_DYNAMIC_CODE = 0

class PerlinNoise:
    """A generator of Perlin Noise.

    constructor: PerlinNoise(shape)

    The constructor takes one argument, shape, as used in the
    construction of Numeric multiarrays.
    """

    __methods__ = ["value_at"]
    __members__ = [] # No public data attributes.
    # TODO: Add subclass for fractal noise.

    def __init__(self, shape, interp_func=None):
	# TODO: Add frequency and interpolation function

	self.shape = shape
	self.dimensions = len(shape)
	self.gradient_lut = _make_gradient_lut(self.dimensions)
	self.interp_func = interp_func

	# Use an optimized value_at if available for this dimension.
	if hasattr(self, "_value_at_%dD" % self.dimensions) and \
	   not _FORCE_GENERIC_CASE:
	    self.value_at = getattr(self, "_value_at_%dD" % self.dimensions)
	else:
	    self.value_at = self._value_at

	if not _DISABLE_DYNAMIC_CODE:
	    make_gradient_at_point_func(self, self.dimensions)
	else:
	    self._gradient_at_point = self._gradient_at_point_static


    def _gradient_at_point(self, point_vector):
	"""Returns the random gradient at that point in the grid."""

	raise NotImplementedError, \
	      "This should be overloaded on instance creation."


    def _gradient_at_point_static(self, point_vector):
	"""Returns the random gradient at that point in the grid.

	point_vector should be array of Int."""

	# TODO: Make tileable

	# dummy thing
	# if alltrue(point_vector & 1):
	#    return (math.pi/2,) * self.dimensions
	#else:
	#    return (-math.pi/2,) * self.dimensions

	key = reduce(_vector_to_key, point_vector) & 0xFF
	return self.gradient_lut[key]

    def value_at(self, point):
	"""Returns the texture's value at a given point.

	The range of the result lies mostly within [-1,1], but not strictly.

	TODO: Make this accessible with slice notation."""

	raise NotImplementedError, \
	      "This should be overloaded on instance creation."

    def _value_at(self, point):
	"""generic value_at method for any number of dimensions."""

	point = asarray(point)

	fillerfunc = _CubeFiller(self, point)

	hypercube = zeros((2,) * self.dimensions, Float)

	# XXX - fromfunction does NOT work as anticipated
	# hypercube = fromfunction(fillerfunc, (2,) * self.dimensions)

	arraypoint = zeros((self.dimensions,),Int)
	for i in range(len(hypercube.flat)):
	    x = i
	    bit = 1
	    while x > 0:
		"""Thanks to dash (AllenShort) for the int to binary-sequence
		conversion."""
		coordinate = x & 1
		arraypoint[-bit] = coordinate
		x = x >> 1
		bit += 1
	    hypercube[arraypoint] = fillerfunc(arraypoint)

	return interpolate(hypercube, point, self.interp_func)

    def _value_at_2D(self, point):
	"""value_at, optimized for the 2D case."""

	if self.interp_func:
	    f = self.interp_func
	else:
	    f = linear_interpolation

	point_floor = floor(point).astype(Int)

	xyfracpart = (subtract(point,point_floor))[::-1]
	(yd, xd) = xyfracpart

	point_floor_rev = point_floor[::-1]

	topleft = self._gradient_at_point(point_floor_rev)
	topleft = topleft[0] * yd + topleft[1] * xd

	topright = self._gradient_at_point(add(point_floor_rev,(0,1)))
	topright = topright[0] * yd + topright[1] * (xd-1)

	top = topleft + f(xd) * (topright - topleft)

	botleft = self._gradient_at_point(add(point_floor_rev,(1,0)))
	botleft = botleft[0] * (yd-1) + botleft[1] * xd

	botright = self._gradient_at_point(add(point_floor_rev,(1,1)))
	botright = botright[0] * (yd-1) + botright[1] * (xd-1)
	#botright = innerproduct(botright, (yd-1, xd-1))

	bot = botleft + f(xd) * (botright - botleft)

	value = top + f(yd) * (bot - top)

	return value

    def __str__(self):
	return "<perlin.PerlinNoise instance, %dD>" % self.dimensions


class _CubeFiller:
    """Provides a callable object to obtain vertices' contributing factors.

    Synopsis:

    _CubeFiller(noise_obj, point)

    Where noise_obj is an instance of PerlinNoise, and point is the
    vector describing the location in space for which you are building
    a cube around.  The resulting object is callable, and may be fed
    an index into the multiarray which represents such a cube.

    To interpolate the value of a given point, you need to know the
    surrounding factors you're interpolating from, thus our identifing
    a cube surrounding the point.

    We'd like a function that, given only a vertex location on the
    cube, returns that vertex's infulence on the current point.  But
    such a function requires knowledge of the "current point", as well
    as the gradients on the grid.  So this class provides an object
    which knows these things, and is callable, so you get your
    function.
    """

    def __init__(self, noise_obj, point):
	"""_CubeFiller(noise_obj, point)

	noise_obj is an instance of PerlinNoise,
	point is a vector."""

	self.point = point

	self.point_rev_floor = floor(point[::-1]).astype(Int)

	self.noise_obj = noise_obj

    def __call__(self, array_index):
	"""Given an array index identifying a vertex...

	returns that value you use for interpolating the stuff.
	"""

	grid_point = self.point_rev_floor + array_index

	gradient = self.noise_obj._gradient_at_point(grid_point)

	value = innerproduct(gradient, (self.point[::-1] - grid_point))

	return value


def make_gradient_at_point_func(instance, dimensions):
    s = "def _gradient_at_point(point_vector):"

    expression = "point_vector[0] & 0xFF"

    for i in range(1,dimensions):
	expression = "(point_vector[%d] + _permutation_lut[%s]) & 0xFF" % \
		     (i, expression)

    expression = "gradient_lut[%s]" % expression

    s = "%s\n\treturn %s\n" % (s, expression)

    compile(s,__name__,'exec')

    namespace_dict = instance.__dict__

    namespace_dict['gradient_lut'] = instance.gradient_lut
    namespace_dict['_permutation_lut'] = _permutation_lut

    exec s in namespace_dict


# A vector utility function.
def magnitude(a):
    """Returns the magnitude (length) of a vector.

    Why isn't this already defined?"""

    return sqrt(innerproduct(a,a))


# The peroid of the Python random() function is nearly 7e12.  That's
# swell, but you know, Perlin says we can get away with a texture that
# repeats every few hundred (or 256) units, because the texture has no
# large-scale features, if you're zoomed out far enough to see that
# much of the texture, you won't be able to see the small details
# anyway.

# So here's Perlin's random fun.

def _make_permutation_lut():
    """Returns a random permutation of range(256)."""

    permutation_lut = arange(256)
    random.shuffle(permutation_lut)

    return permutation_lut


def _make_gradient_lut(dimensions):
    """Returns a populated gradient LUT with 256 entries.

    A gradient is an n-dimensional vector of unit length, pointing in a
    random direction.  There's one of these at each point in the grid,
    but the assumption is that 256 directions is enough, as it's okay
    if different points share the same gradient."""

    i = 0
    lut = zeros((256,dimensions),Float)

    while i <= 255:
	"""Perlin's Monte Carlo method of gathering random gradients.

	Generate a random vector somewhere in the unit n-cube.  If
	it's inside the unit n-circle, it's worth keeping, so
	normalize it and store it.

	We don't normalize and keep the ones outside the n-circle because
	that wouldn't give us an even spherical distribution.

	TODO: limit the steepness of the gradients to prevent
	over-enthusiastic bumps.
	"""

	# XXX - fromfunction does not work as anticipated
	# random_vector = fromfunction(lambda *i: random.random(), (dimensions,))

	random_vector = zeros((dimensions,),Float)

	for d in range(dimensions):
	    random_vector[d]= random.random() * 2.0 - 1.0

	if magnitude(random_vector) <= 1.0:
	    lut[i] = random_vector / magnitude(random_vector)
	    i += 1

    return lut


def _vector_to_key(running_total, next_val):
    """Perlin's fast psuedo-random coordinate to gradient mapper.

    By alternating addition and modulo math with the permutation
    table, we get a "random" number, yet given the same input
    coordinate, we always get the same result.

    Intended only for use within gradient_at_point()."""

    # "& 0xFF" is an idiom for "% 256"
    return running_total + _permutation_lut[next_val & 0xFF]


# (end random fun)


def linear_interpolation(x):
    "f(x) = x"
    return x

def ease_interpolation(x):
    "f(x) = 3(x**2) - 2(x**3)"
    return 3*x**2 - 2*x**3

def interpolate(endpoints, point, f=linear_interpolation):
    """interpolate (box, point, [interpolation_function])

    endpoints is an n-dimensional array with just two elements in each
    dimension, containing the values to be interpolated from.  Perhaps
    it should be named "corners", as they are endpoints in 2D, but in
    general they are corners of the n-cube surrounding the point.

    point is a vector describing a point in n dimensions, for which a
    value is to be interpolated.

    interpolation_function is a cross-fade curve used to weight the
    contributions from the endpoints.  Several are provided by perlin
    module, including:
      linear_interpolation
      ease_interpolation

    Returns the interpolated value at point.
    """

    dimensions = len(shape(endpoints))

    assert(len(point) >= dimensions)

    if dimensions > 1:
	"""We can only interpolate between two points.  Given a
	higher-dimensional structure, we break it into halves and look
	at the two faces, a face having one less dimension than the
	original shape.  Wash, rinse, repeat as necessary.
	"""

	left = interpolate(endpoints[0], point, f)
	right = interpolate(endpoints[1], point, f)
	point = point[1:]
	endpoints = [left,right]


    if f:
	weight = f(math.modf(point[0])[0])
    else:
	weight = math.modf(point[0])[0]

    value = endpoints[0] + weight * (endpoints[1] - endpoints[0])

    return value


_permutation_lut = _make_permutation_lut()
