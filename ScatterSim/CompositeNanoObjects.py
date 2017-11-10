from .NanoObjects import NanoObject, PyramidNanoObject, CylinderNanoObject,\
    SphereNanoObject
import numpy as np
from copy import deepcopy
# This file is where more complex nano objects can be stored
# These interface the same way as NanoObjects but since they're a little more
# complex it makes sense to separate them from NanoObjects
# TODO : Better simpler API for making composite object
# like make_composite( objectslist, pargslist)

UnitTrans = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                     ], dtype=float)


class CompositeNanoObject(NanoObject):
    ''' This is a nano object made up of a collection of nano objects.
        You specify them with a list of objects and pargs (dictionaries).
        The pargs can contain positions and rotations as well.
        Need to redefine all the form factors etc since now they're computed
        for each element.

        objlist : either list of object identifier classes or objects
        themselves
        parglist : parameters for the object classes. If set to None, then
        objlist is assume a list of objects, and not classes

        There is a new parameter in the pargs, it is 'sign'. If it's one, the
        sample adds. If it's negative, it subtracts. Useful for things like
        core shell. Playing with 'rho' is too risky so I reserved a parameter
        for it.

        This is new compared to Kevin's old code. This is redundant with his
        Lattice class but allows to make more complex objects without
        worrying about the lattice.


        Notes
        -----

        Be very careful about subtracting objects. Make sure to think about the
        scenario where there exists an ambient solution (rho_ambient).
    '''

    def __init__(self, objlist, parglist=None, pargs={}):
        '''
            Composite object

            objlist : either list of classes or instantiated objects
                    (pargslist must be None for the latter)
        '''
        super(CompositeNanoObject, self).__init__(pargs=pargs)

        # now define the objects in a list
        self.nano_objects = list()

        if parglist is None:
            for obj in objlist:
                # just in case list has repeating object
                # make a deepcopy
                self.nano_objects.append(deepcopy(obj))
        else:
            for nobj, pargobj in zip(objlist, parglist):

                self.nano_objects.append(nobj(pargs=pargobj))

        # set defaults
        for obj in self.nano_objects:
            if 'sign' not in obj.pargs:
                # defaults to additive
                obj.pargs['sign'] = 1.

    def map_tcoord(self, parent_tranmat=UnitTrans, origin=True):
        ''' This returns a list of objects and transformations.

            This is a recursive relation. This will keep calling map_tcoord
                of each child object until it reaches a leaf (NanoObject).
        '''
        newlist = []
        tranmat = self.get_transformation_matrix(origin=origin)
        tranmat = np.tensordot(tranmat, parent_tranmat, axes=(1,0))
        for nobj in self.nano_objects:
            newlist.extend(nobj.map_tcoord(parent_tranmat=tranmat))
        return newlist

    # TODO : remove when form_factor works
    def form_factor_old(self, qvector):
        """Returns the complex-amplitude of the form factor at the given
            q-coordinates.
            qvector is an array as such: [qx, qy, qz] where qx, qy, qz need
            matching dimensions
        """
        # Phase must be retrieved *before* mapping in q
        phase = self.get_phase(qvector)

        # first rotate just as for V
        qvector = self.map_qcoord(qvector)

        F = np.zeros_like(qvector[0], dtype=complex)
        for nobj in self.nano_objects:
            F += nobj.pargs['sign'] * nobj.form_factor(qvector) * phase

        return F

    def form_factor(self, qvector):
        """Returns the complex-amplitude of the form factor at the given
            q-coordinates.
            qvector is an array as such: [qx, qy, qz] where qx, qy, qz need
            matching dimensions
        """
        # recursive call to get coordinates for each objects (get all the
        # leaves)
        translist = self.map_tcoord()

        res = np.zeros(qvector.shape[1:], dtype=complex)

        # now run the computations
        for nobj, transmat in translist:
            newqvec = np.tensordot(transmat[:3,:3], qvector, axes=(1,0))
            trans_coord = transmat[3,:3]
            newqvec += transmat[3,:3][:,np.newaxis, np.newaxis]
            phase = np.exp(qvector[0]*trans_coord[0])
            phase *= np.exp(qvector[1]*trans_coord[1])
            phase *= np.exp(qvector[2]*trans_coord[2])
            res += nobj.form_factor(newqvec)*phase

        return res

    def to_vtk_actors(self, parent_trans=UnitTrans):
        ''' Returns the transformed object.'''
        actors = list()
        # always transform first
        # set origin to True for positions
        translist = self.map_tcoord(origin=True)

        for nobj, trans in translist:
            print(trans)
            actors.append(nobj.to_vtk_actor(trans_mat=trans))

        return actors


    def volume(self):
        ''' Return the sum of the volume of all objects.
        '''
        volume = 0.
        for nobj in self.nano_objects:
            volume += nobj.pargs['sign'] * nobj.volume()
        return volume

    def V_old(self, rvec):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        rvec : [x,y,z]
        """
        # always transform first
        rvec = self.map_rcoord(rvec)

        Vtot = np.zeros_like(rvec[0])
        for nobj in self.nano_objects:
            Vtot += nobj.pargs['sign'] * nobj.V(rvec)

        return Vtot

    def V(self, rvec):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        rvec : [x,y,z]
        """
        newvec = np.ones((1, *rvec.shape[1:]))
        rvec = np.concatenate((rvec, newvec), axis=0)
        # always transform first
        # set origin to True for positions
        translist = self.map_tcoord(origin=True)
        # print(translist)

        Vtot = np.zeros_like(rvec[0])

        for nobj, trans in translist:
            newcoord = np.tensordot(trans, rvec, axes=(1,0))[:3]

            Vtot += nobj.pargs['sign']*nobj.V(newcoord)

        return Vtot



class OctahedronNanoObject(CompositeNanoObject):
    ''' An octahedron made of two pyramids.
        This is a composite nanoobject.
    '''

    def __init__(self, pargs={}):
        objslist = list()
        pargslist = list()

        # need to correctly set up positions and angles
        # parameters : obj, x0, y0, z0, eta, theta, phi, sign (set to 1)
        parameters = [
            [PyramidNanoObject, 0, 0, 0, 0, 0, 0, 1],
            [PyramidNanoObject, 0, 0, 0, 0, 180, 0, 1],
        ]
        for i in range(len(parameters)):
            objslist.append(parameters[i][0])

            pargslist.append(pargs.copy())
            pargslist[i]['x0'] = parameters[i][1]
            pargslist[i]['y0'] = parameters[i][2]
            pargslist[i]['z0'] = parameters[i][3]
            pargslist[i]['eta'] = parameters[i][4]
            pargslist[i]['phi'] = parameters[i][5]
            pargslist[i]['theta'] = parameters[i][6]
            pargslist[i]['sign'] = parameters[i][7]

        super(OctahedronNanoObject, self).__init__(objslist, pargslist, pargs)


class HollowOctahedronNanoObject(CompositeNanoObject):
    ''' An octahedron made of two pyramids.
        This is a composite nanoobject. Composite nanoobjects are just lists
            of nano objects. They're convenient in that they don't need any
            extra routines added. Everything should just work with the
            initializer and inheriting the rest.

        outer_radius : outer radius of octahedron
        radius_ratio : ratio of inner to outer radius (inner/outer). I chose it
            this way so that this could be supplied to a 1 parameter
            polydisperse object. Note this should be less than 1.

        inner_radius : inner radius of octahedron (not a used parg)

    '''

    def __init__(self, pargs={}):
        objslist = list()
        pargslist = list()

        if 'radius' not in pargs:
            raise ValueError("Need to specify the outer radius")
        if 'radius_ratio' not in pargs:
            raise ValueError("Need to specify ratio of inner to outer radius")

        outer_radius = pargs['radius']
        inner_radius = outer_radius * pargs['radius_ratio']
        # parameters : obj, x0, y0, z0, eta, theta, phi, sign (1 adds, -1
        # subtracts)
        parameters = [
            [OctahedronNanoObject, 0, 0, 0, 0, 0, 0, inner_radius, -1],
            [OctahedronNanoObject, 0, 0, 0, 0, 0, 0, outer_radius, 1],
        ]
        for i in range(len(parameters)):
            objslist.append(parameters[i][0])

            pargslist.append(pargs.copy())
            pargslist[i]['x0'] = parameters[i][1]
            pargslist[i]['y0'] = parameters[i][2]
            pargslist[i]['z0'] = parameters[i][3]
            pargslist[i]['eta'] = parameters[i][4]
            pargslist[i]['theta'] = parameters[i][5]
            pargslist[i]['phi'] = parameters[i][6]
            pargslist[i]['radius'] = parameters[i][7]
            pargslist[i]['sign'] = parameters[i][8]

        super(
            HollowOctahedronNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs)

# OctahedronCylindersNanoObject


class OctahedronCylindersNanoObject(CompositeNanoObject):
    """An octahedron object made of cylinders or like object. The canonical
        (unrotated) version has the square cross-section in the x-y plane, with
        corners pointing along +z and -z.  The corners are on the x-axis and
        y-axis. The edges are 45 degrees to the x and y axes.
    The canonical (unrotated) version of the cylinders should be aligned along
    the z axis. Replace cylinders with spheres, cylindrical shells, prolate
    ellipsoids etc as you wish.

            It is best to just brute force define all the terms in one shot,
                which I chose to do here.
            notes about rotations:
                - i need to dot the rotation matrix of the octahedron with each
                    individual cylinder's rotationo element
            edgelength : length of edge
            edgespread : how much to expand the rods by (not a good name)
                positive is expansion, negative is compression
            edgesep : separation of element from edge
            rest of pargs are used for the cylinder object
                ex : radius, height
            linkerlength : if specified, adds linkers of specified length,
                centered in between the octahedra
            linkerradius : if linkerlength specified, will add linkers of this
            radius
            rho_linker : if linkerlength specified, adds linkers of this
            density (defaults to same density as cylinders in octahedra)
            linkerobject : the object to use for the linkers (defaults to
            baseObject)
    """

    def __init__(
            self,
            baseObject=None,
            linkerObject=None,
            pargs={},
            seed=None):
        if baseObject is None:
            baseObject = CylinderNanoObject
        if linkerObject is None:
            linkerObject = baseObject

        # Set defaults
        if 'edgeshift' not in pargs:
            pargs['edgeshift'] = 0.0
        if 'edgespread' not in pargs:
            pargs['edgespread'] = 0.0
        if 'linkerlength' in pargs:
            addlinkers = True
        else:
            addlinkers = False

        # raise errors for undefined parameters
        if 'edgelength' not in pargs:
            raise ValueError("Need to specify an edgelength for this object")

        # these are slight shifts per cyl along the axis
        # positive is away from COM and negative towards
        shiftlabels = [
            # these correspond to the poslist
            'CYZ1', 'CXZ1', 'CYZ2', 'CXZ2',
            'CXY1', 'CXY4', 'CXY3', 'CXY2',
            'CYZ3', 'CXZ3', 'CYZ4', 'CXZ4',
            'linker1', 'linker2', 'linker3', 'linker4',
            'linker5', 'linker6',
        ]

        # you flip x or y from original shifts to move along edge axis
        # not a good explanation but some sort of personal bookkeeping for
        # now...
        shiftfacs = [
            # top
            [0, -1, 1],
            [-1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            # middle
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            # bottom
            [0, 1, -1],
            [1, 0, -1],
            [0, -1, -1],
            [-1, 0, -1]
        ]

        for lbl1 in shiftlabels:
            if lbl1 not in pargs:
                pargs[lbl1] = 0.

        # calculate shift of COM from edgelength and edgespread
        fac1 = np.sqrt(2) / 2. * \
            ((.5 * pargs['edgelength']) + pargs['edgespread'])
        eL = pargs['edgelength']
        if addlinkers:
            sL = pargs['linkerlength']

        poslist = [
            # eta, theta, phi, x0, y0, z0
            # top part
            [0, 45, -90, 0, fac1, fac1],
            [0, 45, 0, fac1, 0, fac1],
            [0, 45, 90, 0, -fac1, fac1],
            [0, -45, 0, -fac1, 0, fac1],
            # now the flat part
            [0, 90, 45, fac1, fac1, 0],
            [0, 90, -45, fac1, -fac1, 0],
            [0, 90, 45, -fac1, -fac1, 0],
            [0, 90, -45, -fac1, fac1, 0],
            # finally bottom part
            [0, 45, -90, 0, -fac1, -fac1],
            [0, 45, 0, -fac1, 0, -fac1],
            [0, 45, 90, 0, fac1, -fac1],
            [0, -45, 0, fac1, 0, -fac1],
        ]

        if addlinkers:
            poslist_linker = [
                # linkers
                [0, 0, 0, 0, 0, eL / np.sqrt(2) + sL / 2.],
                [0, 0, 0, 0, 0, -eL / np.sqrt(2) - sL / 2.],
                [0, 90, 0, eL / np.sqrt(2) + sL / 2., 0, 0],
                [0, 90, 0, -eL / np.sqrt(2) - sL / 2., 0, 0],
                [0, 90, 90, 0, eL / np.sqrt(2) + sL / 2., 0],
                [0, 90, 90, 0, -eL / np.sqrt(2) - sL / 2., 0],
            ]
            for row in poslist_linker:
                poslist.append(row)
            shiftfacs_linker = [
                [0, 0, 1],
                [0, 0, -1],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
            ]
            for row in shiftfacs_linker:
                shiftfacs.append(row)

        poslist = np.array(poslist)
        shiftfacs = np.array(shiftfacs)

        # now add the shift factors
        for i in range(len(poslist)):
            poslist[i, 3:] += np.sqrt(2) / 2. * \
                shiftfacs[i] * pargs[shiftlabels[i]]

        # need to create objslist and pargslist
        objlist = list()
        pargslist = list()
        for i, pos in enumerate(poslist):
            objlist.append(baseObject)

            eta, phi, theta, x0, y0, z0 = pos
            pargstmp = dict()
            pargstmp.update(pargs)
            pargstmp['eta'] = eta
            pargstmp['theta'] = theta
            pargstmp['phi'] = phi
            pargstmp['x0'] = x0
            pargstmp['y0'] = y0
            pargstmp['z0'] = z0

            labeltmp = shiftlabels[i]
            if 'linker' in labeltmp:
                if 'rho_linker' in pargs:
                    pargstmp['rho_object'] = pargs['rho_linker']
                if 'linkerlength' in pargs:
                    pargstmp['height'] = pargs['linkerlength']
                if 'linkerradius' in pargs:
                    pargstmp['radius'] = pargs['linkerradius']

            pargslist.append(pargstmp)

        super(
            OctahedronCylindersNanoObject,
            self).__init__(
            objlist,
            pargslist,
            pargs=pargs)

# CubicCylindersNanoObject


class CubicCylindersNanoObject(CompositeNanoObject):
    """ A cylinder nano object aligned along axes, meant for cubic structure.
    """

    def __init__(
            self,
            baseObject=None,
            linkerObject=None,
            pargs={},
            seed=None):
        if baseObject is None:
            baseObject = CylinderNanoObject
        if linkerObject is None:
            linkerObject = baseObject

        # raise errors for undefined parameters
        if 'height' not in pargs:
            raise ValueError("Need to specify a height for this object")
        if 'radius' not in pargs:
            raise ValueError("Need to specify a radius for this object")

        eL = pargs['height']

        poslist = [
            # eta, theta, phi, x0, y0, z0
            # top part
            [0, 0, 0, 0, 0, eL / 2.],
            [0, 90, 0, eL / 2., 0, 0],
            [0, 90, 90, 0, eL / 2., 0],
        ]

        # need to create objslist and pargslist
        objlist = list()
        pargslist = list()
        for i, pos in enumerate(poslist):
            objlist.append(baseObject)

            eta, phi, theta, x0, y0, z0 = pos
            pargstmp = dict()
            pargstmp.update(pargs)
            pargstmp['eta'] = eta
            pargstmp['theta'] = theta
            pargstmp['phi'] = phi
            pargstmp['x0'] = x0
            pargstmp['y0'] = y0
            pargstmp['z0'] = z0

            pargslist.append(pargstmp)

        super(
            CubicCylindersNanoObject,
            self).__init__(
            objlist,
            pargslist,
            pargs=pargs)


class CoreShellNanoObject(CompositeNanoObject):
    ''' A core shell nano object.
        This is a composite nanoobject.

        pargs:
            radius_inner : inner radius
            rho_object_inner : inner density
            radius_outer : outer radius
            rho_object_outer : outer density
            rho_ambient : the ambient density (solvent)

            Note : all rho's must be *positive*

        Notes
        -----
        Use of non-zero ambient has not been tested.
        This has been thought through but should be tested.

    '''

    def __init__(self, pargs={}):
        objslist = list()
        pargslist = list()

        objslist = [SphereNanoObject, SphereNanoObject]

        # check for arguments
        if 'rho_object_inner' not in pargs or\
            'radius_inner' not in pargs or\
            'rho_object_outer' not in pargs or\
            'radius_outer' not in pargs or\
                'rho_ambient' not in pargs:
            raise ValueError("Missing args, please check correct syntax")

        deltarho_inner = pargs['rho_object_inner'] - pargs['rho_object_outer']
        deltarho_outer = pargs['rho_object_outer'] - pargs['rho_ambient']

        # get the signs and take absolute value
        sign_inner = 1
        if deltarho_inner < 0:
            sign_inner = -1
            deltarho_inner = np.abs(deltarho_inner)

        sign_outer = 1
        if deltarho_outer < 0:
            sign_outer = -1
            deltarho_outer = np.abs(deltarho_outer)

        pargs_inner = {
            'radius': pargs['radius_inner'],
            'rho_object': deltarho_inner,
            'rho_ambient': 0,
            'sign': sign_inner
        }

        pargs_outer = {
            'radius': pargs['radius_outer'],
            'rho_object': deltarho_outer,
            'rho_ambient': 0,
            'sign': sign_outer
        }

        pargslist = [pargs_inner, pargs_outer]

        super(
            CoreShellNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs=pargs)


class SixCylinderNanoObject(CompositeNanoObject):
    ''' A nano object comprised of six cylinders.'''

    def __init__(self, pargs={}):
        ''' Initalized the object.
        '''
        objslist = list()
        pargslist = list()

        # use default packing if not specified
        if 'packradius' not in pargs:
            pargs['packradius'] = 2 * pargs['radius'] / (2 * np.pi / 6.)

        packradius = pargs['packradius']

        dphi = 2 * np.pi / 6.
        for i in range(6):
            x0, y0, z0 = packradius * \
                np.cos(i * dphi), packradius * np.sin(i * dphi), 0
            pargs_tmp = {
                'radius': pargs['radius'],
                'height': pargs['height'],
                'x0': x0,
                'y0': y0,
                'z0': z0,
            }
            pargslist.append(pargs_tmp)
            objslist.append(CylinderNanoObject)

        super(
            SixCylinderNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs=pargs)


class CylinderDumbbellNanoObject(CompositeNanoObject):
    ''' A nano object comprised of a cylinder with two spheres on edges (like a
    dummbell).'''

    def __init__(self, pargs={}):
        ''' Initalized the object.
        '''
        objslist = list()
        pargslist = list()

        # use default packing if not specified

        cyl_rad = pargs.pop('cylinder_radius')
        cyl_den = pargs.pop('cylinder_density')
        cyl_height = pargs.pop('cylinder_height')
        sph_rad = pargs.pop('sphere_radius')
        # sph_den = pargs.pop('sphere_density')

        pargs_cyl = {
            'radius': cyl_rad,
            'height': cyl_height,
            'x0': 0,
            'y0': 0,
            'z0': 0,
            'rho_ambient': cyl_den
        }

        el = (cyl_height) * .5

        pargs_sphere1 = {
            'radius': sph_rad,
            'x0': 0,
            'y0': 0,
            'z0': -el,
        }

        pargs_sphere2 = {
            'radius': sph_rad,
            'x0': 0,
            'y0': 0,
            'z0': el,
        }

        pargslist = [pargs_cyl, pargs_sphere1, pargs_sphere2]
        objslist = [CylinderNanoObject, SphereNanoObject, SphereNanoObject]

        super(
            CylinderDumbbellNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs=pargs)


class CylinderBulgeNanoObject(CompositeNanoObject):
    ''' A nano object comprised of a cylinder with a sphere bulge in center.'''

    def __init__(self, pargs={}):
        ''' Initalized the object.
        '''
        objslist = list()
        pargslist = list()

        # use default packing if not specified

        cyl_rad = pargs.pop('cylinder_radius')
        cyl_den = pargs.pop('cylinder_density')
        cyl_height = pargs.pop('cylinder_height')
        sph_rad = pargs.pop('sphere_radius')
        sph_den = pargs.pop('sphere_density')

        pargs_cyl = {
            'radius': cyl_rad,
            'height': cyl_height,
            'x0': 0,
            'y0': 0,
            'z0': 0,
            'rho_object': cyl_den
        }

        pargs_sphere = {
            'radius': sph_rad,
            'x0': 0,
            'y0': 0,
            'z0': 0,
            'rho_object': sph_den,
        }

        pargslist = [pargs_cyl, pargs_sphere]
        objslist = [CylinderNanoObject, SphereNanoObject]

        super(
            CylinderBulgeNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs=pargs)


class TetrahedraCylindersNanoObject(CompositeNanoObject):
    ''' A nano object made up of cylinders in a tetrahedral shape.
        NOTE : Needs to be verified...
        The canonical unrotated version is aligned along the z-axis.
        The top inverted tetrahedra has one edge aligned with the x axis and
        one vertex pointing to the +y direction.
        The bottom tetrahedra is a mirror image of this.
    '''
    # TODO : Add density as parameter

    def __init__(self, pargs={}):
        ''' Initalized the object.
        '''
        objslist = list()
        pargslist = list()

        # use default packing if not specified
        r = pargs["radius"]
        L = pargs["edgelength"]
        H = pargs["height"]
        # the triangle angles (degrees)
        beta = 30
        # the triangle angles (radians)
        beta_r = np.radians(beta)
        # the face angle (radians)
        alpha_r = np.arccos(.5/np.cos(beta_r))
        # the face angle (degrees)
        alpha = np.degrees(alpha_r)
        # midway point along height
        D = L/2./np.cos(beta_r)

        # midway to intersection points
        A = D/2. # also it is: L/2.*np.tan(beta_r)
        # note this is NOT the height parameter

        # in order of x, y, z, eta, phi, theta
        posanglelist = [
                   # the base
                   [0, -A, 0, 0, 90, 0],
                   [L/2*(1-np.sin(beta_r)), L/2*(np.cos(beta_r) - np.tan(beta_r)), 0, 0, 90, 60],
                   [-L/2*(1-np.sin(beta_r)), L/2*(np.cos(beta_r) - np.tan(beta_r)), 0, 0, 90, -60],
                   # the height
                   [0,D - L/2*np.cos(alpha_r), L/2*np.sin(alpha_r), 0, 90-alpha, -90],
                   [L*np.cos(alpha_r)*np.cos(beta_r)-L/2*np.cos(alpha_r)*np.cos(beta_r),
                      -L*np.cos(alpha_r)*np.sin(beta_r)+L/2*np.cos(alpha_r)*np.sin(beta_r),
                      L/2*np.sin(alpha_r), 0, 90-alpha, 30],
                   [-L*np.cos(alpha_r)*np.cos(beta_r)+L/2*np.cos(alpha_r)*np.cos(beta_r),
                      -L*np.cos(alpha_r)*np.sin(beta_r)+L/2*np.cos(alpha_r)*np.sin(beta_r),
                      L/2*np.sin(alpha_r), 0, -(90-alpha), -30],
        ]


        eta, theta, phi = 0, np.pi/4., 0


        # the vector pointing to center of rotation for this construction
        Rc = L*np.sqrt(3/8.)
        tetra_height = L*np.sin(alpha_r)
        z_center = tetra_height - Rc
        COM = np.array([0, 0, -z_center])


        pargs_cyl = {
            'radius': r,
            'height': H,
            'x0': 0,
            'y0': 0,
            'z0': 0,
        }

        pargslist = list()
        objslist = list()
        for x0, y0, z0, eta, phi, theta in posanglelist:
            pargstmp = pargs_cyl.copy()
            pargstmp['x0'] = x0 + COM[0]
            pargstmp['y0'] = y0 + COM[1]
            pargstmp['z0'] = z0 + COM[2]
            pargstmp['eta'] = eta
            pargstmp['phi'] = phi
            pargstmp['theta'] = theta
            pargslist.append(pargstmp)
            objslist.append(CylinderNanoObject)

        super(
            TetrahedraCylindersNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs=pargs)


class HourglassCylindersNanoObject(CompositeNanoObject):
    ''' A nano object made up of cylinders in a tetrahedral shape.'''
    # TODO : Add density as parameter

    def __init__(self, pargs={}):
        ''' Initalized the object.
        '''

        # use default packing if not specified
        L = pargs["edgelength"]
        Rc = L*np.sqrt(3/8.)

        pargs_bot = pargs.copy()
        pargs_bot['z0'] = -Rc
        pargs_top = pargs.copy()
        pargs_top['phi'] = 180
        pargs_top['theta'] = 180
        pargs_top['z0'] = Rc

        objslist = [TetrahedraCylindersNanoObject,
                    TetrahedraCylindersNanoObject]
        pargslist = [pargs_bot, pargs_top]

        super(
            HourglassCylindersNanoObject,
            self).__init__(
            objslist,
            pargslist,
            pargs=pargs)
