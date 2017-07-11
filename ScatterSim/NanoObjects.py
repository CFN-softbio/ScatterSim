import numpy as np
# cylndrical and spherical Bessel functions
from scipy.special import j0, j1, spherical_jn


# NanoObject
####################################################################
class NanoObject:
    """Defines a nano-object, which can then be placed within a lattice
    for computing scattering data. A nano-object can be anisotropic.

        This is the base class all nano-objects should inherit
    """
    conversion_factor = 1E-4        # Converts units from 1E-6 A^-2 into nm^-2
    def __init__(self, pargs={}):
        '''
            pargs are the potential arguments, a dictionary of arguments
        '''
        # look for and set defaults
        eta = self.check_arg('eta', pargs, default=0)
        theta = self.check_arg('theta', pargs, default=0)
        phi = self.check_arg('phi', pargs, default=0)
        x0 = self.check_arg('x0', pargs, default=0)
        y0 = self.check_arg('y0', pargs, default=0)
        z0 = self.check_arg('z0', pargs, default=0)

        # now update object's pargs with pargs
        self.check_arg('rho_ambient', pargs, default=0.0)
        self.check_arg('rho_object', pargs, default=15.0)

        self.pargs = dict()
        self.pargs.update(pargs)
        # delta rho is difference between sample and solvent density
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho_object'] )

        self.set_origin(x0, y0, z0)
        # this will set the rotation matrix
        self.set_angles(eta=eta, phi=phi, theta=theta)

    def __add__(self, other):
        ''' The add operator should now
            just create a new composite NanoObject.
            There are four cases:
                NanoObject + NanoObject
                NanoObject + CompositeNanoObject
                CompositeNanoObject + CompositeNanoObject
                CompositeNanoObject + NanoObject

            # Note : the pargs of the new object is overridden by original object
        '''
        if isinstance(self, CompositeNanoObject):
            nano_objects = self.nano_objects
            pargs = self.pargs
        else:
            nano_objects = [self]
            # no pargs for parent composite object set yet
            pargs = {}

        nano_objects.append(other)

        # TODO : should move to composite nano object code
        obj = CompositeNanoObject(nano_objects, pargs=pargs)
        obj.pargs['sign'] = 1.
        return obj

    def check_arg(self, name, pargs, default=0):
        ''' Check dictionary for a parameter. If not there, set the parg
            to the default.

            Returns the result in both cases
        '''
        if name in pargs:
            return pargs[name]
        else:
            pargs[name] = default
            return default


    def rebuild(self, pargs={}):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)
        # also recompute the delta_rho
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho_object'] )


    def set_angles(self, eta=None, phi=None, theta=None):
        """Update one or multiple orientation angles (degrees).
            These are the typical Euler coordinates.

            See Also
            -------
                rotation_elements
        """
        if eta is not None:
            self.pargs['eta'] = np.copy(eta)
        if phi is not None:
            self.pargs['phi'] = np.copy(phi)
        if theta is not None:
            self.pargs['theta'] = np.copy(theta)

        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )


    def set_origin(self,x0=None, y0=None, z0=None):
        ''' Set the origin of the sample.
        '''
        if x0 is not None:
            self.pargs['x0'] = np.copy(x0)
        else:
            x0 = self.pargs['x0']
        if y0 is not None:
            self.pargs['y0'] = np.copy(y0)
            y0 = self.pargs['y0']
        else:
            y0 = self.pargs['y0']
        if z0 is not None:
            self.pargs['z0'] = np.copy(z0)
            z0 = self.pargs['z0']
        else:
            z0 = self.pargs['z0']

        self.origin = np.array([x0, y0, z0])

    def thresh_array(self, r, val):
        ''' threshold array to have minimum value.'''
        w = np.where(np.abs(r) < val)
        if len(w[0]) > 0:
            r[w] = val


    def rotation_elements(self, eta, phi, theta):
        """Converts angles into an appropriate rotation matrix.

        Three-axis rotation:
            1. Rotate about +z by eta (counter-clockwise in x-y plane)
            2. Tilt by phi with respect to +z (rotation about y-axis,
            clockwise in x-z plane) then
            3. rotate by theta in-place (rotation about z-axis,
            counter-clockwise in x-y plane)
        """

        eta = np.radians( eta )
        phi = np.radians( phi )
        theta = np.radians( theta )

        c1 = np.cos(eta);c2 = np.cos(phi); c3 = np.cos(theta);
        s1 = np.sin(eta); s2 = np.sin(phi); s3 = np.sin(theta);

        rotation_elements = np.array([
            [c1*c2*c3 - s1*s3, -c3*s1-c1*c2*s3, c1*s2],
            [c1*s3 + c2*c3*s1, c1*c3 - c2*s1*s3, s1*s2],
            [-c3*s2, s2*s3, c2],
        ]);

        return rotation_elements

    def get_phase(self, qvec):
        ''' Get the phase factor from the shift, for the q position.'''
        phase = np.exp(1j*qvec[0]*self.pargs['x0'])
        phase *= np.exp(1j*qvec[1]*self.pargs['y0'])
        phase *= np.exp(1j*qvec[2]*self.pargs['z0'])

        return phase



    def map_qcoord(self,qcoord):
        ''' Map the reciprocal space coordinates from the parent object to
        child object (this one).  The origin shift is not needed here, and so
        this function basically just rotates the coordinates given, using the
        internal rotation matrix.
        Translation is a phase which is computed separately.

            Parameters
            ----------
            qcoord : float array, the q coordinates to rotate.
                Leftermost index are the q components: [qx, qy, qz] where qx,
                qy, qz are any dimension

            Returns
            -------
            qcoord : the rotated coordinates

            See Also
            --------
            map_coord (to specify arbitrary rotation matrix)
            set_origin
            set_angles
        '''
        return self.map_coord(qcoord,self.rotation_matrix)

    def map_rcoord(self,rcoord):
        ''' Map the real space coordinates from the parent object to child
        object's coordinates (this one), using the internal rotation matrix and
        origin (set by set_angles, set_origin). This sets the 0 and orientation
        of sample to align with internal orientation. This function is
        generally recursively called for every traversal into child
        coordinates.

            Parameters
            ----------
            rcoord : float array, the r coordinates to map
                Leftermost index are the r components: [x, y, z] where
                x, y, z are any dimension

            Returns
            -------
            qcoord : the rotated coordinates

            See Also
            --------
            map_coord (to specify your own rotations/translations)
            set_origin
            set_angles
            '''
        # tmp fix for objects that dont  set origin
        if not hasattr(self,'origin'):
            raise ValueError("Origin has not been set (please set origin to something"
                    " like (0,0,0) in the object initialization")
        # update the origin from the x0, y0, z0 values
        self.set_origin()

        return self.map_coord(rcoord,self.rotation_matrix,origin=self.origin)

    def map_coord(self, coord, rotation_matrix, origin=None):
        ''' Map the real space coordinates from the parent object to child
        object's coordinates (this one), using the specified rotation matrix
        and origin. This sets the 0 and orientation of sample to align with
        internal orientation. This function is generally recursively called for
        every traversal into child coordinates.

            Parameters
            ----------
            coord : float array, the r coordinates to map
            rotation_matrix : 3x3 ndarray, the rotation matrix
            origin : length 3 array (optional), the origin (default [0,0,0])

            Returns
            -------
            coord : the rotated coordinates

            Notes
            -----
            Most objects will use map_rcoord and map_qcoord
            instead, which know what to do with the internal rotation
            matrices of the objects.

            The slowest varying index of r is the coordinate

            See Also
            --------
            map_coord (to specify your own rotations/translations)
            set_origin
            set_angles
            '''
        if coord.shape[0] != 3:
            raise ValueError("Error slowest varying dimension is not coord"
                             "(3)")

        if origin is None:
            x0, y0, z0 = 0, 0, 0
        else:
            x0, y0, z0 = origin

        # first subtract origin
        # save time, don't do it no origin specified
        if origin is not None:
            coord = np.copy(coord)
            coord[0] = coord[0] - x0
            coord[1] = coord[1] - y0
            coord[2] = coord[2] - z0

        #next dot product
        coord = np.tensordot(rotation_matrix, coord, axes=(1,0))

        return coord


    def form_factor_numerical(self, qvector, num_points=100, size_scale=None):
        ''' This is a brute-force calculation of the form-factor, using the
        realspace potential. This is computationally intensive and should be
        avoided in preference to analytical functions which are put into the
        "form_factor(qx,qy,qz)" function.

            Parameters
            ----------
            qvector : float arrays the reciprocal space coordinates
            num_points : int, optional, the number of points to sample
            rotation_elements : rotation matrix

            Returns
            -------
            coord : the complex form factor

            Note : NOT TESTED YET
            '''
        qvector = self.map_qcoord(qvector)

        if size_scale==None:
            if 'radius' in self.pargs:
                size_scale = 2.0*self.pargs['radius']
            else:
                size_scale = 2.0

        x_vals, dx = np.linspace( -size_scale, size_scale, num_points, endpoint=True, retstep=True)
        y_vals, dy = np.linspace( -size_scale, size_scale, num_points, endpoint=True, retstep=True)
        z_vals, dz = np.linspace( -size_scale, size_scale, num_points, endpoint=True, retstep=True)

        dVolume = dx*dy*dz

        f = 0.0+0.0j

        # Triple-integral over 3D space
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    r = (x, y, z)
                    V = self.V(x, y, z, rotation_elements=rotation_elements)

                    # not sure which axes need to be dotted
                    f += V*cexp( 1j*np.tensordot(qvector,r,axes=(0,0)) )*dVolume

        return self.pargs['delta_rho']*f


    def form_factor_squared_numerical(self, qvector, num_points=100, size_scale=None):
        """Returns the square of the form factor."""
        f = self.form_factor_numerical(qvector, num_points=num_points, size_scale=size_scale)
        g = f*f.conjugate()
        return g.real


    def form_factor_squared(self, qvector):
        """Returns the square of the form factor.
            Value returned is real.
            Note : Need to implement form_factor.
        """

        f = self.form_factor(qvector)
        g = f*np.conj(f)
        return g.real

    def func_orientation_spread(self, q, func, dtype, num_phi=50, num_theta=50, orientation_spread=None):
        """ Compute an orientation average of function func from a sphere of points whose radius is q

            Parameters
            ----------
            q : q points, can be an array
            func :  should take a qvector as sole argument ([qx,qy,qz])
                This is a general function used by form_factors calcs (isotropic or orient spread)

            dtype :  the data type (complex or float)

            orientation_spread: If set, overrides the one in pargs. If not set,
                searches in pargs.  The default distribution is uniform. Endpoints
                are given by pargs['orientation_spread'] which are phi_start,
                phi_end, theta_start, theta_end

            Returns
            -------
                F : result of function func, averaged over specified orientations

            Notes
            -----
            This function is intended to be used to create a effective form factor
            when particle orientations have some distribution.
            Math:
                solid angle is dS = r^2 sin(th) dth dphi
                ignore the r part (or q)

        """
        if orientation_spread is None:
            if 'orientation_spread' not in self.pargs:
                raise ValueError("form_factor_orientation_spread : Sorry, orientation_spread not "
                        "set in defining potential arguments"
                        " (pargs). Please define this parameter and run again.")
            else:
                orientation_spread = self.pargs['orientation_spread']

        phi_start, phi_end, theta_start, theta_end = orientation_spread

        # Phi is orientation around z-axis (in x-y plane)
        phi_vals, dphi = np.linspace( phi_start, phi_end, num_phi, endpoint=False, retstep=True )
        # Theta is tilt with respect to +z axis
        theta_vals, dtheta = np.linspace( theta_start, theta_end, num_theta, endpoint=False, retstep=True )


        F = np.zeros_like(q, dtype=dtype)
        dStot = 0.

        for theta in theta_vals:
            qz =  q*np.cos(theta)
            dS = np.sin(theta)*dtheta*dphi
            dStot += dS*num_phi

            for phi in phi_vals:
                qx = -q*np.sin(theta)*np.cos(phi)
                qy =  q*np.sin(theta)*np.sin(phi)
                qvector = np.array([qx,qy,qz])

                F += func(qvector) * dS

        # when the orientation spread is full solid angle, still I think it's better
        # to use dStot than 4*np.pi
        F /= dStot

        return F


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation.
                Math:
                    solid angle is dS = r^2 sin(th) dth dphi
                    ignore the r part (or q)
        """
        # phi_start, phi_end, theta_start, theta_end
        orientation_spread = (0, 2*np.pi, 0, np.pi)
        return self.func_orientation_spread(q, self.form_factor, complex,
                num_phi=num_phi, num_theta=num_theta,
                orientation_spread=orientation_spread)


    def form_factor_squared_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the square of the form factor, under the assumption of
        random orientation. In other words, we average over every possible
        orientation.
        This value is denoted by P(q)

        Note this is different from form_factor_isotropic, where the difference is:
            |<F>|^2 versis <|F|^2>. See "Periodic lattices of arbitrary
            nano-objects: modeling and applications for self-assembled systems"
            (DOI: 10.1107/S160057671302832X)
        """
        # phi_start, phi_end, theta_start, theta_end
        orientation_spread = (0, 2*np.pi, 0, np.pi)
        return self.func_orientation_spread(q, self.form_factor_squared,
                float, num_phi=num_phi, num_theta=num_theta,
                orientation_spread=orientation_spread)

    def form_factor_orientation_spread(self, q, num_phi=50, num_theta=50, orientation_spread=None):
        """Returns the particle form factor, averaged over some orientations.
        This function is intended to be used to create a effective form factor
        when particle orientations have some distribution.

        orientation_spread: If set, overrides the one in pargs. If not set, searches in pargs.

        The default distribution is uniform. Endpoints are given by pargs['orientation_spread']
            which are phi_start, phi_end, theta_start, theta_end
        """
        if orientation_spread is None:
            if 'orientation_spread' not in self.pargs:
                raise ValueError("form_factor_orientation_spread : Sorry, orientation_spread not "
                        "set in defining potential arguments"
                        " (pargs). Please define this parameter and run again.")
            else:
                orientation_spread = self.pargs['orientation_spread']

        return self.func_orientation_spread(q, self.form_factor, complex,
                num_phi=nump_phi, num_theta=num_theta,
                orientation_spread=orientation_spread)


    def form_factor_squared_orientation_spread(self, q, num_phi=50, num_theta=50, orientation_spread=None):
        """Returns the particle form factor, averaged over some orientations.
        This function is intended to be used to create a effective form factor
        when particle orientations have some distribution.

        orientation_spread: If set, overrides the one in pargs. If not set, searches in pargs.

        The default distribution is uniform. Endpoints are given by pargs['orientation_spread']
            which are phi_start, phi_end, theta_start, theta_end
        """
        if orientation_spread is None:
            if 'orientation_spread' not in self.pargs:
                raise ValueError("form_factor_orientation_spread : Sorry, orientation_spread not "
                        "set in defining potential arguments"
                        " (pargs). Please define this parameter and run again.")
            else:
                orientation_spread = self.pargs['orientation_spread']

        return self.func_orientation_spread(q, self.form_factor_squared, float,
                num_phi=nump_phi, num_theta=num_theta,
                orientation_spread=orientation_spread)

    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""
        numerator = np.abs(self.form_factor_isotropic(q, num_phi=num_phi, num_theta=num_theta))**2
        denominator = self.form_factor_squared_isotropic(q, num_phi=num_phi,num_theta=num_theta)
        return numerator/denominator

    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_squared) and beta_ratio.
        This function can be highly optimized in derived classes."""

        P = self.form_factor_squared_isotropic(q, num_phi=num_phi, num_theta=num_theta)
        beta = self.beta_ratio(q, num_phi=num_phi, num_theta=num_theta, approx=approx)

        return P, beta


    def to_string(self):
        """Returns a string describing the object."""
        s = "Base NanoObject (zero potential everywhere)."
        return s

    def to_short_string(self):
        """Returns a short string describing the object's variables.
        (Useful for distinguishing objects of the same class.)"""
        s = "(0)"
        return s

    def form_factor(self, qvector):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            qvector is an array as such: [qx, qy, qz] where qx, qy, qz need
            matching dimensions
        """

        # example:
        #qvector = self.map_qcoord(qvector)
        raise NotImplementedError("Needs to be implemented by inheriting object")


    def V(self, rvec, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        rvec : [x,y,z]

        This method should be overwritten.
        """
        raise NotImplementedError("This needs to be implemented by the inherting object")

    def projections(self, length, npoints=100):
        ''' Compute the xy, yz, and xz projections (in that order).

            This is a convenience routine to allow one to see approximately
            what the nano object looks like. Useful when creating new composite
            nano objects.

            Parameters
            ----------

            length : length of the box to compute the projections.
                Will compute a 3D box of [-length, +length] in x, y and z

            npoints : the number of points to calculate per dimension
                default 100
                WARNING : This creates a npoints x npoints x npoints array

            Returns
            -------

            V_xy : the xy projection
            V_xz : the xz projection
            V_yz : the xz projection

            Notes
            -----
            To compute the projection, this function must first compute a 3D
            density field of the object, and add various projections.  This
            code can become very slow and memory intensive if npoints is too
            large.

            It also is strongly reliant on how well Obj.V() was coded.  When
            adding new NanoObjects, be careful to program in the Obj.V()
            function properly.
            This is a great tool for CompositeNanoObjects
        '''
        x = np.linspace(-length, length, npoints)
        # ij indexing means that we index in V[x,y,z]
        # Note that rightermost index is fastest varying index
        x, y, z = np.meshgrid(x,x,x,indexing='ij')
        V = self.V(np.array([x,y,z]))
        V_xy = np.sum(V,axis=2).T
        V_xz = np.sum(V,axis=1).T
        V_yz = np.sum(V,axis=0).T

        return V_xy, V_xz, V_yz

# Variations of NanoObjects
# PolydisperseNanoObject
class PolydisperseNanoObject(NanoObject):
    """Defines a polydisperse nano-object, which has a distribution in
        argname of width argstdname
        if not specified, argname defaults to the parameter 'radius'
            and argstdname defaults to 'sigma_R'
        the width is in absolute units (not a percentage or ratio of the
            argument it's varying)

        Note : this is slow, but more general. If slow, it may be worth hard
        coding the form factor (if  not too complex).
    """

    def __init__(self, baseNanoObjectClass, pargs={}, argname=None, argstdname=None):

        # this should set the orientation and origin defaults
        NanoObject.__init__(self, pargs=pargs)

        if argname is None:
            argname = 'radius'
        if argstdname is None:
            argstdname = 'sigma_R'

        self.argname = argname
        self.argstdname = argstdname

        # More defaults specific to the polydisperse object
        if argstdname not in self.pargs:
            raise ValueError("Error : did not specify a {} in pargs.".format(argstdname) +
                    " Please specify, or else do not use the polydisperse modifier."
                    )
        if argname not in self.pargs:
            self.pargs[argname] = 1.

        if 'distribution_type' not in self.pargs:
            self.pargs['distribution_type'] = 'gaussian'
        if 'distribution_num_points' not in self.pargs:
            self.pargs['distribution_num_points'] = 5

        self.baseNanoObjectClass = baseNanoObjectClass

        self.distribution_list = []


    def rebuild(self, pargs={}):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs.

        For a polydisperse object, need to update pargs for elements in
        distribution, omitting variable that's being modified.
        """
        self.pargs.update(pargs)
        self.distribution_list = self.distribution(force=True)


    def distribution(self, spread=2.5, force=False):
        ''' Compute the probability distribution.
            Returns the previously calculated distribution
                or recomputes if doesn't exist.

            Run rebuild to reset.
        '''
        if self.distribution_list==[] or force:
            argname = self.argname
            argstdname = self.argstdname
            # Build the distribution
            mean_val = self.pargs[argname]
            rms_val = self.pargs[argstdname]
            n = self.pargs['distribution_num_points']
            if self.pargs['distribution_type'] == 'gaussian':
                self.distribution_list = \
                    self.distribution_gaussian(mean=mean_val, rms=rms_val,
                                               num_points=n, spread=spread)
            elif self.pargs['distribution_type'] == 'lognormal':
                self.distribution_list = \
                    self.distribution_lognormal(mean=mean_val, rms=rms_val,
                                               num_points=n, spread=spread)
            else:
                print( "Unknown distribution type in distribution()." )

        # Return the existing distribution
        return self.distribution_list


    def distribution_gaussian(self, mean=1.0, rms=0.01, num_points=11, spread=2.5):
        '''
            Gaussian distribution of parameters.

            mean : the mean value

            rms : the rms value

        '''

        distribution_list = []

        step = 2*spread*rms/(num_points-1)
        # the sampled value
        sample = mean - step*(num_points-1)/2.0

        prefactor = 1/( rms*np.sqrt(2*np.pi) )

        for i in range(num_points):
            delta = mean - sample
            wt = prefactor*np.exp( - (delta**2)/(2*( rms**2 ) ) )

            curNanoObject = self.baseNanoObjectClass(pargs=self.pargs)
            curNanoObject.rebuild( pargs={self.argname : sample} )

            distribution_list.append( [sample, step, wt, curNanoObject] )

            sample += step

        return distribution_list

    def distribution_lognormal(self, mean=1.0, rms=0.01, num_points=91, spread=10):
        '''
            Lognormal distribution of parameters.

            mean : this here will mean the scale of the lognorm (the higher the further the peak)

            rms : this will mean the shape of the lognorm distribution

        '''
        from scipy.stats import lognorm

        distribution_list = []
        actual_mean = lognorm.mean(rms, loc=0, scale=mean)
        actual_std = lognorm.std(rms, loc=0, scale=mean)

        # use the actual mean and std from lognormal distribution, dont
        # use the parameters, which are for the underlying Gaussian distribution
        step = 2*spread*actual_std/(num_points-1)
        # the sampled value
        sample = actual_mean - step*(num_points-1)/2.0

        for i in range(num_points):
            wt = lognorm.pdf(sample, rms, loc=0, scale=mean)

            curNanoObject = self.baseNanoObjectClass(pargs=self.pargs)
            curNanoObject.rebuild( pargs={self.argname : sample} )

            distribution_list.append( [sample, step, wt, curNanoObject] )

            sample += step

        return distribution_list


    def V(self,rvec):
        """Returns the average potential"""
        return self.dist_sum('V', rvec[0].shape, float, rvec)

    def volume(self):
        ''' ret avg volume'''
        return self.dist_sum('volume', 1, float)[0]



    def dist_sum(self, funcname, shape, dtype, *args, **kwargs):
        ''' Sum the function with name 'funcname' over variable 'vec' over the current
        distribution.
        Forwards other keyword arguments to function

        Parameters
        ---------
        funcname : the function name
        shape : the shape of the result
        dtype : the data type
        args : arguments to the function
        components : specifies if vec is of form [qx,qy,qz] (True)
            or just q (False)
        kwargs : keyword arguments to function

        '''
        res = np.zeros(shape,dtype=dtype)
        cts = 0.

        for R, dR, wt, curNanoObject in self.distribution():
            res_R = getattr(curNanoObject, funcname)(*args,**kwargs)
            res += wt*res_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError("Nothing was added to distribution? \n"
                    "Distribution list is: {}".format(self.distribution())
                    )

        return res/cts

    def form_factor(self, qvec):
        """Returns the complex-amplitude of the form factor at the given
            <F>_d
        q-coordinates."""
        return self.dist_sum('form_factor', qvec[0].shape, complex, qvec)

    def form_factor_distavg_squared(self, qvec):
        '''
            |<F>_d|^2
        '''
        return np.abs(self.dist_sum('form_factor', qvec[0].shape, complex, qvec))**2

    def form_factor_squared(self, qvec):
        """Returns the square of the form factor.

            <|F|^2>_d
        """
        return self.dist_sum('form_factor_squared', qvec[0].shape, float, qvec)

    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        ''' Returns the isotropic form factor
            < <F>_iso >_d

        '''
        return self.dist_sum('form_factor_isotropic', q.shape, complex, q, num_phi=num_phi,num_theta=num_theta)

    def form_factor_squared_isotropic(self, q, num_phi=50, num_theta=50):
        ''' Returns the isotropic form factor
            < <|F|^2>_iso >_d
        '''
        return self.dist_sum('form_factor_squared_isotropic', q.shape, float, q, num_phi=num_phi,num_theta=num_theta)

    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<<F(q)>_d>_iso|^2"""
        return np.abs(self.form_factor_isotropic(q,num_phi=nump_phi, num_theta=num_theta))**2

    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso
        """

        return self.func_orientation_spread(q, self.form_factor_distavg_squared,num_phi=num_phi, num_theta=num_theta)


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<<F(q)>_iso>_d|^2 / <<|F(q)|^2>_iso>_d This
        ratio depends on polydispersity: for a monodisperse system, beta = 1
        for all q. """

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            # numerator and denominator
            beta_num = self.beta_numerator(q,num_phi=num_phi, num_theta=num_theta)
            beta_den = self.form_factor_squared_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            return beta_num/beta_den


class RandomizedNanoObject(NanoObject):
    """Defines a nano-object of which certain parameters are randomized

        This is a little more useful than a polydisperse nano object
            in that it allows you to randomize a few parameters at once.
        The more parameters you randomize, the higher you may want to set
        nsamples for accuracy.

        Parameters
        ----------

        pargs : the base potential arguments (parameters) of the system
            In pargs, you need to specify the parameters necessary to build the
            base NanoObject Class supplied. On top of this, you also need to
            specify one more parameter:
                'distribution_num_points' : this is the number of points to
                randomly sample. Basically, this class will spawn this number
                of nano-objects, with each nano-object having parameters that
                are randomly varied by argdict (described next).
                This defaults to 21. You should set this larger for better
                accuracy.

        argdict:
            This will be randomized parameters. These will override the pargs.
            It is a dictionary, with the key of the dictionary entries being
            the parameter names. The entries themselves are dictionaries
            containing more information. For example, for a polydisperse sphere
            whose radius follows the lognormal distribution, you would put:
                {
                 'radius' : {
                             'distribution_type' : 'lognormal',
                             'mean' : 1, # average val
                             'sigma' : .1,  #std deviation
                            },
                 # add more arguments here etc...
                }

        Notes
        -----
        Call 'build_objects' to build a list of objects with parameters
        randomly varied. The form factor etc will then be computed by averaging
        over these quanties. Note that form_factor_squared != form_factor^2 in
        this case! See Beta(q) in [1] for more information.


        References
        ---------
        [1] Yager, Kevin G., et al. "Periodic lattices of arbitrary nano-objects:
        modeling and applications for self-assembled systems." Journal of
        Applied Crystallography 47.1 (2014): 118-129.

    """

    def __init__(self, baseNanoObjectClass, pargs={}, argdict=None):

        # this should set the orientation and origin defaults
        NanoObject.__init__(self, pargs=pargs)

        if argdict is None:
            argdict = dict(radius={'distribution' : 'gaussian',
                'sigma' : .1, 'mean' : 1})
        if 'distribution_num_points' not in pargs:
            pargs['distribution_num_points'] = 21


        self.argdict = argdict
        self.pargs = pargs

        # check that parameters in args dict are there
        for key, val in self.argdict.items():
            if key not in self.pargs:
                raise ValueError("Error {} not present in pargs".format(key))

            if 'distribution_type' not in val:
                val['distribution_type'] = 'gaussian'

        self.baseNanoObjectClass = baseNanoObjectClass
        self.object_list = []
        self.rebuild()


    def rebuild(self, pargs={}, argdict={}):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs.

        For a polydisperse object, need to update pargs for elements in
        distribution, omitting variable that's being modified.
        """
        self.pargs.update(pargs)
        self.argdict.update(argdict)
        self.build_objects()


    def build_objects(self):
        ''' Build a list of objects whose parameters are randomly
                sampled according to a probability distribution.

            Compute the probability distribution.
            Returns the previously calculated distribution
                or recomputes if doesn't exist.

            Since we can have multipe parameters with different
                distributions, this is a distribution list.

            Run rebuild to reset.
            # TODO : Make distribution from numbers
                when summing run same object, rebuild and re-calculate
                since it's random, you don't want to cache either
                'distribution_num_points' : 21,
                {'radius' : {'distribution_type' : 'gaussian',
                    # parameters specific to distribution
                    'avg' : 1, # average val
                    'std' : .1,  #std deviation
                    'spread' : 2.5}}
        '''
        self.object_list = list()
        for i in range(self.pargs['distribution_num_points']):
            pargs_tmp = self.pargs.copy()
            for key, entry in self.argdict.items():
                pargs_tmp[key] = self.sample_point(entry)
            self.object_list.append(self.baseNanoObjectClass(pargs=pargs_tmp))

    def sample_point(self, dist_dict):
        ''' Sample a point from a distribution specified by
            dist_dict
            Currently supported:
                'distribution_type' (case insensitive)
                    'Gaussian' or 'normal'
                        parameters :
                            'mean' : average of Gaussian (normal) distribution
                            'sigma' : standard deviation of Gaussian (normal) distribution
                    'uniform' :
                        parameters :
                            'low' : lower bound of distribution
                            'high' : upper bound of distribution
                    'lognormal'
                        parameters:
                            'mean' : mean value of underlying normal distribution
                            'sigma' : standard deviation of underlying normal distribution


        '''
        _supported_distributions = ['gaussian', 'normal', 'uniform', 'lognormal']
        distribution_type = dist_dict['distribution_type'].lower()
        if distribution_type == 'gaussian' or distribution_type == 'normal':
            mean = dist_dict['mean']
            sigma = dist_dict['sigma']
            return np.random.normal(loc=mean, scale=sigma)
        elif distribution_type == 'uniform':
            low = dist_dict['low']
            high = dist_dict['high']
            return np.random.uniform(low=low, high=high)
        elif distribution_type == 'lognormal':
            mean = dist_dict['mean']
            sigma = dist_dict['sigma']
            return np.random.lognormal(mean=mean, sigma=sigma)
        else:
            errorstr = "Error, distribution {} not supported".format(distribution_type)
            errorstr = errorstr + "\nSupported are: {}".format(_supported_distributions)
            raise ValueError(errorstr)

    def V(self,rvec):
        """Returns the average potential"""
        return self.dist_sum('V', rvec[0].shape, float, rvec)

    def volume(self):
        ''' ret avg volume'''
        return self.dist_sum('volume', 1, float)[0]

    def dist_sum(self, funcname, shape, dtype, *args, **kwargs):
        ''' Sum the function with name 'funcname' over variable 'vec' over the current
        distribution.
        Forwards other keyword arguments to function

        Parameters
        ---------
        funcname : the function name
        shape : the shape of the result
        dtype : the data type
        args : arguments to the function
        components : specifies if vec is of form [qx,qy,qz] (True)
            or just q (False)
        kwargs : keyword arguments to function

        '''
        res = np.zeros(shape,dtype=dtype)
        cts = 0.

        for curNanoObject in self.object_list:
            res_R = getattr(curNanoObject, funcname)(*args,**kwargs)
            res += res_R
            cts += 1

        if cts ==0.:
            raise ValueError("Nothing was added to distribution? \n"
                    "Distribution list is: {}".format(self.distribution())
                    )

        return res/cts

    def form_factor(self, qvec):
        """Returns the complex-amplitude of the form factor at the given
            <F>_d
        q-coordinates."""
        return self.dist_sum('form_factor', qvec[0].shape, complex, qvec)

    def form_factor_distavg_squared(self, qvec):
        '''
            |<F>_d|^2
        '''
        return np.abs(self.dist_sum('form_factor', qvec[0].shape, complex, qvec))**2

    def form_factor_squared(self, qvec):
        """Returns the square of the form factor.

            <|F|^2>_d
        """
        return self.dist_sum('form_factor_squared', qvec[0].shape, float, qvec)

    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        ''' Returns the isotropic form factor
            < <F>_iso >_d

        '''
        return self.dist_sum('form_factor_isotropic', q.shape, complex, q, num_phi=num_phi,num_theta=num_theta)

    def form_factor_squared_isotropic(self, q, num_phi=50, num_theta=50):
        ''' Returns the isotropic form factor
            < <|F|^2>_iso >_d
        '''
        return self.dist_sum('form_factor_squared_isotropic', q.shape, float, q, num_phi=num_phi,num_theta=num_theta)

    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<<F(q)>_d>_iso|^2"""
        return np.abs(self.form_factor_isotropic(q,num_phi=nump_phi, num_theta=num_theta))**2

    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso
        """

        return self.func_orientation_spread(q, self.form_factor_distavg_squared,num_phi=num_phi, num_theta=num_theta)


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<<F(q)>_iso>_d|^2 / <<|F(q)|^2>_iso>_d This
        ratio depends on polydispersity: for a monodisperse system, beta = 1
        for all q. """

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            # numerator and denominator
            beta_num = self.beta_numerator(q,num_phi=num_phi, num_theta=num_theta)
            beta_den = self.form_factor_squared_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            return beta_num/beta_den




# Next are NanoObjects

class SphereNanoObject(NanoObject):
    ''' This is as the name of object describes, a sphere.'''
    def __init__(self, pargs={}):
        super(SphereNanoObject, self).__init__(pargs=pargs)

        if 'radius' not in self.pargs:
            # Set a default size
            self.pargs['radius'] = 1.0


    def V(self, rvec):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""

        rvec = self.map_rcoord(np.array(rvec))
        R = self.pargs['radius']
        r = np.sqrt( rvec[0]**2 + rvec[1]**2 + rvec[2]**2 )
        return (r < R).astype(float)*self.pargs['delta_rho']

    def volume(self):
        return 4/3.*np.pi*self.pargs['radius']**3

    def form_factor(self, qvec):
        ''' Compute the form factor of a sphere. '''
        phase = self.get_phase(qvec)

        qx, qy, qz = qvec
        R = self.pargs['radius']

        volume = self.volume()

        q = np.sqrt( qx**2 + qy**2 + qz**2 )
        qR = q*R

        # threshold to avoid zero values
        qR = np.maximum(1e-8, qR)

        # use: 3* j1(qR)/qR (let scipy handle it)
        # numerically more stable than its equivalent:
        # 3*( np.sin(qR) - qR*np.cos(qR) )/( qR**3 )

        F = self.pargs['delta_rho']*volume*3*spherical_jn(1,qR)/qR*phase

        return F

    # override some complex functions to save time
    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation.
        """
        return self.form_factor( np.array([q, 0, 0]))

    def form_factor_squared_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor squared, averaged over every possible orientation.
        """
        # numpy should broadcast 0 to same length as q
        return self.form_factor_squared( np.array([q, 0, 0]))

    def to_string(self):
        """Returns a string describing the object."""
        s = "SphereNanoObject: A sphere of radius = %.3f nm" % self.pargs['radius']
        return s

    def to_short_string(self):
        """Returns a short string describing the object's variables.
        (Useful for distinguishing objects of the same class.)"""
        s = "R = %.3f nm" % self.pargs['radius']
        return s

# CylinderNanoObject
class CylinderNanoObject(NanoObject):
    """A cylinder nano-object. The canonical (unrotated) version
    has the circular-base in the x-y plane, with the length along z.

    self.pargs contains parameters:
        rho_ambient : the cylinder density
        rho_object : the solvent density
        radius : (default 1.0) the cylinder radius
        length : (default 1.0) the cylinder length

        eta,phi,eta: Euler angles
        x0, y0, z0 : the position of cylinder COM relative to origin
        The object is rotated first about origin, then translated to
            where x0, y0, and z0 define it to be.

    these are calculated after the fact:
        delta_rho : rho_ambient - rho1
    """

    def __init__(self, pargs={}):
        super(CylinderNanoObject, self).__init__(pargs=pargs)

        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0

        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0

    def V(self, rvec):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        Returns 1 if in the space, 0 otherwise.
        Can be arrays.
        Rotate then translate.

            rotation_matrix is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)
        """
        R = self.pargs['radius']
        L = self.pargs['height']

        rvec = self.map_rcoord(rvec)
        # could be in one step, but making explicit
        x, y, z = rvec

        r = np.hypot(x,y)
        result = np.zeros(x.shape)
        w = np.where((z <= L/2.)*(z >= -L/2.)*(np.abs(r) <= R))
        if len(w[0]) > 0:
            result[w] = self.pargs['delta_rho']

        return result

    def volume(self):
        return np.pi*self.pargs['radius']**2*self.pargs['height']


    def form_factor(self, qvec):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.

            The first set of q are assumed to be that of the root qx,qy,qz
            The next sets are the parent coordinates. These are only supplied if the parent is not the root
            The root q is only necessary for the form factor, where the phase factor needs to be calculated
                with respect to the origin of that coordinate system.
        """
        # Phase must be retrieved *before* mapping in q
        phase = self.get_phase(qvec)

        # first rotate just as for V
        qvec = self.map_qcoord(qvec)
        self.thresh_array(qvec,1e-4)
        qx,qy,qz = qvec

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = self.volume()

        qr = np.hypot(qx, qy)

        # NOTE : Numpy's sinc function adds
        # a factor of pi in we need to remove.
        # Why numpy... why??? ><
        F = 2*np.sinc(qz*H/2./np.pi)*j1(qr*R)/qr/R + 1j*0
        F *= phase
        F *= self.pargs['delta_rho']*volume

        return F

# PyramidNanoObject
class PyramidNanoObject(NanoObject):
    """ A square-based truncated pyramid nano-object. The canonical (unrotated) version
    has the square-base in the x-y plane, with the peak pointing along +z.
    The base-edges are parallel to the x-axis and y-axis (i.e. the corners
    point at 45 degrees to axes.
    Edge length of base is 2*R

    A regular pyramid (for octahedra) will have equilateral triangles as faces
        and not be truncated.
        Thus H = sqrt(2)*R and face_angle = arctan(sqrt(2)) = 54.7356 degrees
        Only set "R" to get this regular pyramid.

    From [Kevin's website] (http://gisaxs.com/index.php/Form_Factor:Pyramid):
        For pyramid of base edge-length 2R, and height H. The angle of the
        pyramid walls is \alpha. If H < R/ \tan\alpha then the pyramid is
        truncated (flat top).

    Originally from:
    Lazzari, Rémi. "IsGISAXS: a program for grazing-incidence small-angle X-ray
    scattering analysis of supported islands." Journal of Applied
    Crystallography 35.4 (2002): 406-421.
    doi:10.1107/S0021889802006088
    """
    def __init__(self, pargs={}):
        super(PyramidNanoObject, self).__init__(pargs=pargs)

        # defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.
        if 'height' not in self.pargs:
            # Assume user wants a 'regular' pyramid
            self.pargs['height'] = np.sqrt(2.0)*self.pargs['radius']
        if 'pyramid_face_angle' not in self.pargs:
            self.pargs['pyramid_face_angle'] = np.degrees(np.arctan(np.sqrt(2)))


    def V(self, rvec):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""

        # first rotate and shift
        rvec = self.map_rcoord(rvec)
        x,y,z = rvec

        R = self.pargs['radius']
        H = min( self.pargs['height'] , R*np.tan( np.radians(self.pargs['pyramid_face_angle']) ) )
        R_z = R - z/np.tan( np.radians(self.pargs['pyramid_face_angle']) )
        V = ((z < H)*(z > 0)*(np.abs(x) < np.abs(R_z))*(np.abs(y) < np.abs(R_z))).astype(float)

        return V

    def volume(self):
        ''' Volume of a pyramid.
        '''
        #height = self.pargs['height']
        #base_len = self.pargs['radius']/np.sqrt(2)*2
        #base_area = base_len**2
        #return base_area*height/3.

        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = np.tan(np.radians(self.pargs['pyramid_face_angle']))
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )

        return volume

    def form_factor(self, qvec):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
        From Kevin's website

        Notes : I have checked that the q=0 scaling scales as volume and it does.
            So composite objects (sums and differences) should scale properly.
            For example, for a regular pyramid (equilateral triangles), the volume
            is 4*sqrt(2)/3*R^3 where 2*R is the edge length at the base. I checked
            and the q=0 scattering is exactly volume^2 as it should be
            (assuming other prefactors are 1).
        """

        # Phase must be retrieved *before* mapping in q
        phase = self.get_phase(qvec)

        qvec = self.map_qcoord(qvec)
        # fix divide by zero errors, threshold values (abs val)
        self.thresh_array(qvec, 1e-6)

        qx,qy,qz = qvec

        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = np.tan(np.radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = self.volume()



        q1 = 0.5*( (qx-qy)*amod + qz )
        q2 = 0.5*( (qx-qy)*amod - qz )
        q3 = 0.5*( (qx+qy)*amod + qz )
        q4 = 0.5*( (qx+qy)*amod - qz )


        K1 = np.sinc(q1*H/np.pi)*np.exp( +1.0j * q1*H ) + np.sinc(q2*H/np.pi)*np.exp( -1.0j * q2*H )
        K2 = -1.0j*np.sinc(q1*H/np.pi)*np.exp( +1.0j * q1*H ) + 1.0j*np.sinc(q2*H/np.pi)*np.exp( -1.0j * q2*H )
        K3 = np.sinc(q3*H/np.pi)*np.exp( +1.0j * q3*H ) + np.sinc(q4*H/np.pi)*np.exp( -1.0j * q4*H )
        K4 = -1.0j*np.sinc(q3*H/np.pi)*np.exp( +1.0j * q3*H ) + 1.0j*np.sinc(q4*H/np.pi)*np.exp( -1.0j * q4*H )

        F = (H/(qx*qy))*( K1*np.cos((qx-qy)*R) + K2*np.sin((qx-qy)*R) - K3*np.cos((qx+qy)*R) - K4*np.sin((qx+qy)*R) )
        F *= self.pargs['delta_rho']

        return F

from .CompositeNanoObjects import CompositeNanoObject
