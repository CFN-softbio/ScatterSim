.. toctree::
  :maxdepth: 2
  :caption: Contents:


NanoObject
==========
The Nano Object is the base object for all ScatterSim code. It is
comprised of known scattering.

It is defined in the following way::

 from ScatterSim.NanoObjects import SphereNanoObject
 sphere = SphereNanoObject(pargs=(dict(radius=2, rho_object=1.)))

where here ``SphereNanoObject`` is a ``NanoObject`` and ``pargs`` are the
potential arguments of the nano object. In the case of a sphere, the
only potential argument that describes it is its ``radius`` and
``rho_object``, its density (this is normally electron density for xray
scattering but can also be the
`scattering length density
<http://gisaxs.com/index.php/Scattering_Length_Density>`_, as in neutron
scattering).

Orientation and Placement
=========================
One of the powers of ScatterSim is the ability to place these
NanoObjects in various positions and orientations, and compound them
together. Before looking at how to compound objects together, we'll look
at how we translate and rotate them.

Rotations
---------
A ``NanoObject`` is first rotated.
The rotations are defined through three `extrinsic rotations
<https://en.wikipedia.org/wiki/Euler_angles>`_, which is just a fancy
way of saying that that the object is successively rotated about three
different axes with respect to a **fixed** reference frame.

The rotations are as follows:

  1. :math:`\eta` : rotate the sample about the +z axis, counter clockwise
  in the x-y plane.

  2. :math:`\phi` : rotate the sample about the +y axis, clockwise in the z
  plane.

  3. :math:`\theta` : rotate the sample about the +z, counter-clockwisze in
  the x-y plane

Each NanoObject will initially have some orientation with respect the
the coordinate system, described in their docstrings. For example, the
docstring of the ``CylinderNanoObject`` is defined as follows::

 """A cylinder nano-object. The canonical (unrotated) version
  has the circular-base in the x-y plane, with the length along z.
  """

The rotations are defined by adding them to the pargs. For example, to
rotate the cylinder such that it is aligned along the y-axis, and not
the z-axis, one would do the following::

  pargs = {
    'radius' : 3, # in nm
    'height' : 10, # in nm
    'eta' : 0, # 0 degrees about zaxis
    'phi' : 90, # 90 degrees about y axis
    'theta' : 90, # 90 degrees about z axis
  }
  cylinder = CylinderNanoObject(pargs=pargs)

This would re-orient the cylinder along the yaxis.

Note : The rotation is performed about the origin of the object's
coordinate system. It is recommended that any new `NanoObjects` are
defined so that their center of mass is aligned with the origin.

Translations
------------

The translations are straight forward. They are always performed after
the rotations of the objects and are defined by adding the ``x0``,
``y0``, and ``z0`` parameters for x, y and z translations, respectively.
For example, to now translate this cylinder along z by 10nm, one would
add::

  pargs = {
    'radius' : 3, # in nm
    'height' : 10, # in nm
    'eta' : 0, # 0 degrees about zaxis
    'phi' : 90, # 90 degrees about y axis
    'theta' : 90, # 90 degrees about z axis
    'z0' : 10, # translate +z by 10 nm
  }
  cylinder = CylinderNanoObject(pargs=pargs)

