# -*- coding: utf-8 -*-
###################################################################
# MultiComponentModel.py
# version 0.9.0
# March 8, 2013
###################################################################
# Author:
#    Dr. Kevin G. Yager
#     Brookhaven National Lab, Center for Functional Nanomaterials
# With contributions from:
#    Dr. Yugang Zhang
#     Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# Description:
#  Computes transmission scattering curves for 'multi-component'
#  systems. Specifically, for ordered lattices of nanoparticles,
#  the unit cell may have different kinds of particles at the lattice
#  sites.
###################################################################
###################################################################


from ScatterSim.BaseClasses import Potential, Model, arrays_equal, value_at
from ScatterSim.BaseClasses import radians, cos, sin, degrees
from ScatterSim import gamma
import os, sys

from cmath import exp as cexp

from numpy import sinc
import random
import numpy as np
import matplotlib.pyplot as plt
# Bessel functions of the first kind, orders 0 and 1
from scipy.special import j0, j1

#from tqdm import tqdm



# NanoObject
####################################################################
class NanoObject(Potential):
    """Defines a nano-object, which can then be placed within a lattice
    for computing scattering data. A nano-object can be anisotropic."""
    # TODO: Don't ignore the rotation_matrix/rotation_elements stuff
    conversion_factor = 1E-4        # Converts units from 1E-6 A^-2 into nm^-2
    def __init__(self, pargs={}, seed=None):
        self.rotation_matrix = np.identity(3)
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                    }
        self.pargs['cache_results'] = True
        self.form_factor_isotropic_already_computed = {}
        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )


    def rebuild(self, pargs={}, seed=None):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)
        if seed:
            self.seed=seed



    def set_angles(self, eta=None, phi=None, theta=None):
        """Update one or multiple orientation angles (degrees).
        """
        if eta is not None:
            self.pargs['eta'] = np.copy(eta)
        if phi is not None:
            self.pargs['phi'] = np.copy(phi)
        if theta is not None:
            self.pargs['theta'] = np.copy(theta)

        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )


    def set_origin(self,x0=None, y0=None, z0=None):
        ''' Set the origin of the sample.'''
        if x0 is not None:
            self.pargs['x0'] = np.copy(x0)
        else:
            x0 = self.pargs['x0']
        if y0 is not None:
            self.pargs['y0'] = np.copy(y0)
            y0 = self.pargs['y0']
        if z0 is not None:
            self.pargs['z0'] = np.copy(z0)
            z0 = self.pargs['z0']
        self.origin = np.array([x0, y0, z0])


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

        c1 = cos(eta);c2 = cos(phi); c3 = cos(theta);
        s1 = sin(eta); s2 = sin(phi); s3 = sin(theta);

        rotation_elements = np.array([
            [c1*c2*c3 - s1*s3, -c3*s1-c1*c2*s3, c1*s2],
            [c1*s3 + c2*c3*s1, c1*c3 - c2*s1*s3, s1*s2],
            [-c3*s2, s2*s3, c2],
        ]);

        return rotation_elements

    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase

    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.

        This method should be overwritten.
        """
        return self.conversion_factor*0.0


    def rotate_coord(self, qx, qy, qz, rotation_matrix=None):
        """Rotates the q-vector in the way given by the internal
        rotation_matrix, which should have been set using "set_angles"
        or the appropriate pargs (eta, phi, theta).
        rotation_matrix is an extra after the fact rotation matrix to apply
            after the builtin rotation elements.
            Deprecated: Use map coord instead
        """
        DeprecationWarning("rotate_coord is deprecated, use map_coord instead")
        return self.map_coord(np.array([qx,qy,qz]), self.rotation_matrix, origin=None)

    def map_qcoord(self,qcoord):
        ''' Map the reciprocal space coordinates from the parent object to
        child object (this one).  The origin shift is not needed here, and so
        this function basically just rotates the coordinates given, using the
        internal rotation matrix.
        Translation is a phase which is computed separately.

            Parameters
            ----------
            qcoord : float array, the q coordinates to rotate.

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
            self.set_origin(x0=self.pargs['x0'],y0=self.pargs['y0'],z0=self.pargs['z0'])

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


    def form_factor_numerical(self, qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None):
        ''' This is a brute-force calculation of the form-factor, using the
        realspace potential. This is computationally intensive and should be
        avoided in preference to analytical functions which are put into the
        "form_factor(qx,qy,qz)" function.

            Parameters
            ----------
            qx, qy, qz: float arrays the reciprocal space coordinates
            num_points : int, optional, the number of points to sample
            size_scale : float, optional, number of points to sample in x, y, z
            rotation_elements : rotation matrix

            Returns
            -------
            coord : the complex form factor
            '''
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))

        q = (qx, qy, qz)

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

                    #a = np.dot(q,r)
                    #b = cexp( 1j*np.dot(q,r) )
                    #val = V*cexp( 1j*np.dot(q,r) )*dV
                    #print x, y, z, V, a, b, dV, val

                    f += V*cexp( 1j*np.dot(q,r) )*dVolume

        return self.pargs['delta_rho']*f


    def form_factor_squared_numerical(self, qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None):
        """Returns the square of the form factor."""
        f = self.form_factor_numerical(qx,qy,qz, num_points=num_points, size_scale=size_scale, rotation_elements=rotation_elements)
        g = f*f.conjugate()
        return g.real


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))

        return self.pargs['delta_rho']*0.0 + self.pargs['delta_rho']*0.0j

    def form_factor_squared(self, qx, qy, qz):
        """Returns the square of the form factor."""

        f = self.form_factor(qx,qy,qz)
        g = f*f.conjugate()
        return g.real


    def form_factor_intensity(self, qx, qy, qz):
        """Returns the intensity of the form factor."""
        f = self.form_factor(qx,qy,qz)

        return abs(f)


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation."""

        # Check cache
        #if self.pargs['cache_results'] and q in self.form_factor_isotropic_already_computed:
            #return self.form_factor_isotropic_already_computed[q]

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        F = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F += self.form_factor(qx, qy, qz) * dS


        # Update cache
        #if self.pargs['cache_results']:
            #self.form_factor_isotropic_already_computed[q] = F

        return F


    def form_factor_orientation_spread(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over some orientations.
        This function is intended to be used to create a effective form factor
        when particle orientations have some distribution.

        The default distribution is uniform. Endpoints are given by pargs['form_factor_orientation_spread']
        """

        # Usage:
        # pargs['form_factor_orientation_spread'] = [0, 2*pi, 0, pi] # Isotropic

        # WARNING: This function essentially ignores particle orientation,
        # since it remaps a given q_hkl to q, which is then converted back into
        # a spread of qx, qy, qz. Thus, all q_hkl are collapsed unphysically.
        # As such, it does not correctly account for the orientation
        # distribution of a particle. It should only be considered a crude
        # approximation.
        # TODO: Don't ignore particle orientation.

        # Check cache
        if self.pargs['cache_results'] and q in self.form_factor_isotropic_already_computed:
            return self.form_factor_isotropic_already_computed[q]

        phi_start, phi_end, theta_start, theta_end = self.pargs['form_factor_orientation_spread']

        # Phi is orientation around z-axis (in x-y plane)
        phi_vals, dphi = np.linspace( phi_start, phi_end, num_phi, endpoint=False, retstep=True )
        # Theta is tilt with respect to +z axis
        theta_vals, dtheta = np.linspace( theta_start, theta_end, num_theta, endpoint=False, retstep=True )


        F = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F += self.form_factor(qx, qy, qz) * dS


        # Update cache
        if self.pargs['cache_results']:
            self.form_factor_isotropic_already_computed[q] = F

        return F


    def form_factor_isotropic_array__replaced__(self, q_list, num_phi=50, num_theta=50):
        """Returns a 1D array of the isotropic form factor."""

        F = np.zeros( (len(q_list)), dtype=np.complex )
        for i, q in enumerate(q_list):

            F[i] = self.form_factor_isotropic( q, num_phi=num_phi, num_theta=num_theta )

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                P += self.form_factor_squared(qx, qy, qz) * dS


        return P




    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<F(q)>|^2"""

        # For a monodisperse system, this is simply P(q)
        return self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""
        return 1.0


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        P = self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)
        beta = self.beta_ratio(q, num_phi=num_phi, num_theta=num_theta, approx=approx)

        return P, beta



    def plot_form_factor_amplitude(self, qtuple, filename='form_factor_amplitude.png', ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values."""
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = np.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = []
        for q in q_list:
            int_list.append( self.form_factor(0,0,q).real )

        #q_zeros = np.zeros(len(q_list))
        #int_list = self.form_factor_array(q_zeros,q_zeros,q_list)

        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)


        plt.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )

        if ylog:
            plt.semilogy()
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        plt.ylabel( r'$F(q)$' )

        #xi, xf, yi, yf = plt.axis()
        #yf = 5e5
        #yi = 0
        #plt.axis( [xi, xf, yi, yf] )

        plt.savefig( filename )

        return int_list



    def plot_form_factor_intensity(self, qtuple, filename='form_factor_intensity.png', ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = np.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = []
        for q in q_list:
            int_list.append( self.form_factor_intensity(0,0,q) )


        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)


        plt.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )

        if ylog:
            plt.semilogy()
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        plt.ylabel( r'$F(q)$' )

        #xi, xf, yi, yf = plt.axis()
        #yf = 5e5
        #plt.axis( [xi, xf, yi, yf] )

        plt.savefig( filename )

        return int_list


    def to_string(self):
        """Returns a string describing the object."""
        s = "Base NanoObject (zero potential everywhere)."
        return s

    def to_short_string(self):
        """Returns a short string describing the object's variables.
        (Useful for distinguishing objects of the same class.)"""
        s = "(0)"
        return s

    def to_povray_string(self, full_size=80.0, minibox_size=4.0):
        """Returns a string with POV-Ray code for visualizing the object."""

        # Without any additional information, the best we can do
        # is to build an approximation of the object using the real-space
        # potential as a guide.

        L = minibox_size/2.0

        s = "union {\n"


        for x in np.arange(-full_size, full_size, minibox_size):
            for y in np.arange(-full_size, full_size, minibox_size):
                for z in np.arange(-full_size, full_size, minibox_size):
                    if self.V(x,y,z)>0:
                        s += "    cube { -%f,%f translate <%f,%f,%f> }\n" % (L,L,x,y,z)


        texture_num = int(self.pargs['rho1']*10)
        s = "    texture { obj_texture%05d }\n" % (texture_num)

        s += "}"

        return s



# PolydisperseNanoObject
###################################################################
class PolydisperseNanoObject(NanoObject):
    """Defines a polydisperse nano-object, which has a distribution in
    size ('radius') of width 'sigma_R'

    For objects with more parameters, you may specify these extra
    paramters:
        height, sigma_H : object height, and sdev in height
        sigma_theta_x : object tilt from x axis
        sigma_theta_y : object tilt from y axis
        sigma_theta_z : object tilt from z axis

    Note : this is slow, but more general. If slow, it may be worth hard
    coding the form factor (if  not too complex).

    distribution type will be the same for all parameters unfortunately.
    Perhaps define different instances of the object for different
    distributions.

    ."""

    def __init__(self, baseNanoObjectClass, pargs={}, seed=None):

        NanoObject.__init__(self, pargs=pargs, seed=seed)

        # Set defaults
        if 'sigma_R' not in self.pargs:
            self.pargs['sigma_R'] = 0.
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.
        if 'sigma_H' not in self.pargs:
            self.pargs['sigma_H'] = 0.
        if 'sigma_theta_x' not in self.pargs:
            self.pargs['sigma_theta_x'] = 0.
        if 'sigma_theta_y' not in self.pargs:
            self.pargs['sigma_theta_y'] = 0.
        if 'sigma_theta_z' not in self.pargs:
            self.pargs['sigma_theta_z'] = 0.
        if 'distribution_type' not in self.pargs:
            self.pargs['distribution_type'] = 'gaussian'
        if 'distribution_num_points' not in self.pargs:
            self.pargs['distribution_num_points'] = 5
        if 'iso_external' not in self.pargs:
            self.pargs['iso_external'] = False


        self.baseNanoObjectClass = baseNanoObjectClass

        self.distribution_list = []


        self.pargs['cache_results'] = False

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []


    def rebuild(self, pargs={}, seed=None):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)

        if seed:
            self.seed=seed

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []

        self.distribution_list = []


    def distribution(self, spread=2.5, force=False):
        if self.distribution_list==[] or force:

            # Build the distribution
            r = self.pargs['radius']
            n = self.pargs['distribution_num_points']
            if self.pargs['distribution_type'] == 'gaussian':
                self.distribution_list = self.distribution_gaussian(r, self.pargs['sigma_R']*r, n, spread=spread)
            else:
                print( "Unknown distribution type in distribution()." )


        # Return the existing distribution
        return self.distribution_list


    def distribution_gaussian(self, radius=1.0, sigma=0.01, num_points=11, spread=2.5):

        distribution_list = []

        step = 2*spread*sigma/(num_points-1)
        R = radius - step*(num_points-1)/2.0

        prefactor = 1/( sigma*np.sqrt(2*np.pi) )

        for i in range(num_points):
            delta = radius-R
            wt = prefactor*np.exp( - (delta**2)/(2*( sigma**2 ) ) )

            curNanoObject = self.baseNanoObjectClass(pargs=self.pargs)
            curNanoObject.rebuild( pargs={'radius':R} )

            distribution_list.append( [R, step, wt, curNanoObject] )

            R += step

        return distribution_list


    def get_cache_form_factor(self, qx, qy, qz):
        if qx in self.form_factor_already_computed and self.pargs['cache_results']:
            if qy in self.form_factor_already_computed[qx]:
                if qz in self.form_factor_already_computed[qx][qy]:
                    return True, self.form_factor_already_computed[qx][qy][qz]

        return False, 0.0


    def set_cache_form_factor(self, qx, qy, qz, value):
        if self.pargs['cache_results']:
            if qx not in self.form_factor_already_computed:
                self.form_factor_already_computed[qx] = {}
            if qy not in self.form_factor_already_computed[qx]:
                self.form_factor_already_computed[qx][qy] = {}

            self.form_factor_already_computed[qx][qy][qz] = value

    def V(self,x,y,z):
        """Returns the average volume"""

        #found, v = self.get_cache_form_factor(qx, qy, qz)
        #if found:
            #return v

        v = np.zeros_like(x)
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.V(x, y, z)
            v += wt*v_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError
        return v/cts


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        #found, v = self.get_cache_form_factor(qx, qy, qz)
        #if found:
            #return v

        v = 0.0
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor(qx, qy, qz)
            v += wt*v_R*dR
            cts += wt*dR

        self.set_cache_form_factor(qx, qy, qz, v)
        if cts ==0.:
            raise ValueError

        return v/cts


    def form_factor_array(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        v = np.zeros(len(qx))
        cnts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_array(qx, qy, qz)
            v += wt*v_R*dR
            cnts += wt*dR

        if cts ==0.:
            raise ValueError

        return v/cts


    def form_factor_squared(self, qx, qy, qz):
        """Returns the square of the form factor."""
        v = 0.0
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_squared(qx, qy, qz)
            v += wt*v_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError

        return v/cts


    def form_factor_intensity(self, qx, qy, qz):
        """Returns the intensity of the form factor."""
        v = 0.0
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity(qx, qy, qz)
            v += wt*v_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError
        return v/cts


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        v = 0.0
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError

        return v/cts


    def form_factor_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        v = 0.0
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError

        return v/cts


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        # Check cache
        if self.pargs['cache_results'] and q in self.form_factor_intensity_isotropic_already_computed:
            return self.form_factor_intensity_isotropic_already_computed[q]

        v = 0.0
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError

        # Update cache
        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_already_computed[q] = v

        return v/cts


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        # Check cache
        if self.pargs['cache_results'] and arrays_equal( q_list, self.form_factor_intensity_isotropic_array_already_computed_qlist ):
            return self.form_factor_intensity_isotropic_array_already_computed

        v = np.zeros(len(q_list))
        cts = 0.
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R.astype(float)*dR
            cts += wt*dR

        if cts ==0.:
            raise ValueError

        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_array_already_computed_qlist = q_list
            self.form_factor_intensity_isotropic_array_already_computed = v

        return v/cts



    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<F(q)>|^2"""

        if self.pargs['cache_results'] and q in self.beta_numerator_already_computed:
            return self.beta_numerator_already_computed[q]

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external( q, num_phi=num_phi, num_theta=num_theta )

        else:

            # TODO: Check this code. (Bug suspected, since beta ends up being >1.)

            F = self.form_factor_isotropic( q, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_already_computed[q] = G

        return G


    def beta_numerator_array(self, q_list, num_phi=50, num_theta=50):

        if self.pargs['cache_results'] and arrays_equal( q_list, self.beta_numerator_array_already_computed_qlist ):
            return self.beta_numerator_array_already_computed

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external_array( q_list, num_phi=num_phi, num_theta=num_theta )

        else:
            F = self.form_factor_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_array_already_computed_qlist = q_list
            self.beta_numerator_array_already_computed = G

        return G


    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor(qx, qy, qz)
                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_numerator_iso_external_array__replaced__(self, q_list, num_phi=50, num_theta=50):

        G = np.zeros(len(q_list))

        for i, q in enumerate(q_list):
            G[i] = self.beta_numerator_iso_external(q, num_phi=num_phi, num_theta=num_theta)

        return G


    def beta_numerator_iso_external_array(self, q_list, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = np.zeros(len(q_list))

        for theta in theta_vals:
            qz =  q_list*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q_list*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor_array(qx, qy, qz)

                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            P, beta = self.P_beta( q, num_phi=num_phi, num_theta=num_theta )
            return beta


    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns a 1D array of the beta ratio."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
            return beta
        else:
            P, beta = self.P_beta_array( q_list, num_phi=num_phi, num_theta=num_theta )

            return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and q in self.P_beta_already_computed:
            return self.P_beta_already_computed[q]

        P = self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            G = self.beta_numerator(q, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_already_computed[q] = P, beta

        return P, beta


    def P_beta_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and arrays_equal( q_list, self.P_beta_array_already_computed_qlist ):
            return self.P_beta_array_already_computed

        P = self.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
        else:
            G = self.beta_numerator_array(q_list, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_array_already_computed_qlist = q_list
            self.P_beta_array_already_computed = P, beta

        return P, beta

class PolydisperseOrientSpreadNanoObject(NanoObject):
    """Defines a polydisperse orientational spread nano-object, which has a distribution in
    size ('radius') of width 'sigma_R'

    For objects with more parameters, you may specify these extra
    paramters:
        height, sigma_H : object height, and sdev in height
        sigma_theta_x : object tilt from x axis
        sigma_theta_y : object tilt from y axis
        sigma_theta_z : object tilt from z axis

    Note : this is slow, but more general. If slow, it may be worth hard
    coding the form factor (if  not too complex).

    distribution type will be the same for all parameters unfortunately.
    Perhaps define different instances of the object for different
    distributions.

    Note this code can be greatly sped up if we can assume that P(q) = P(qx,qy)*P(qz)
        See the rod code for example.
    TODO : make the rod code include rotations like in here

    ."""

    def __init__(self, baseNanoObjectClass, pargs={}, seed=None):

        NanoObject.__init__(self, pargs=pargs, seed=seed)

        # Set defaults
        if 'sigma_R' not in self.pargs:
            self.pargs['sigma_R'] = 0.
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.
        if 'sigma_H' not in self.pargs:
            self.pargs['sigma_H'] = 0.
        if 'sigma_theta_x' not in self.pargs:
            self.pargs['sigma_theta_x'] = 0.
        if 'sigma_theta_y' not in self.pargs:
            self.pargs['sigma_theta_y'] = 0.
        if 'sigma_theta_z' not in self.pargs:
            self.pargs['sigma_theta_z'] = 0.
        if 'distribution_type' not in self.pargs:
            self.pargs['distribution_type'] = 'gaussian'
        if 'distribution_num_points' not in self.pargs:
            self.pargs['distribution_num_points'] = 5
        if 'iso_external' not in self.pargs:
            self.pargs['iso_external'] = False


        self.baseObject = baseNanoObjectClass(pargs=pargs)

        self.distribution_list = []


        self.pargs['cache_results'] = False

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []


    def rebuild(self, pargs={}, seed=None):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)

        if seed:
            self.seed=seed

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []

        self.distribution_list = []

    def distribution_gaussian(self, x0, sigma_x, num_points=100, spread=2.5,positive=True):
        ''' Gaussian distribution. Here, assumes that x can only be positive.
            Chooses a range of X and provides its weights. Normalizes by
            the total sum rather than the probability distribution.
            This is really a Gaussian prob dist with cutoff.
            sigma_x : standard deviation
            positive : assumes x can only be positive (useful for radii etc, not for angles)
        '''
        # Note, I don't think we need dX since I normalize wts by its
        # sum which also would have a delta X factor in it
        if positive:
            Xs = np.linspace(np.maximum(0,x0-spread*sigma_x),
                             x0+spread*sigma_x, num_points)
        else:
            Xs = np.linspace(x0-spread*sigma_x, x0+spread*sigma_x, num_points)

        # Gaussian weights
        wts = np.exp( - (Xs-x0)**2/(2*( sigma_x**2 ) ) )
        # should be normalized to 1
        wts /= np.sum(wts)

        return Xs, wts

    def distribution_uniform(self, x0, sigma_x, num_points=100, positive=True):
        ''' Gaussian distribution. Here, assumes that x can only be positive.
            Chooses a range of X and provides its weights. Normalizes by
            the total sum rather than the probability distribution.
            This is really a Gaussian prob dist with cutoff.
            sigma_x : percentage standard deviation
        '''
        # Note, I don't think we need dX since I normalize wts by its
        # sum which also would have a delta X factor in it
        if positive:
            Xs = np.linspace(np.maximum(0,x0-sigma_x), x0 + sigma_x,
                             num_points)
        else:
            Xs = np.linspace(x0-sigma_x, x0+sigma_x, num_points)

        # uniform weights
        wts = np.ones_like(Xs)
        # should be normalized to 1
        wts /= np.sum(wts)

        return Xs, wts

    def distribution_Rs(self, num_points=100, spread=2.5):
        ''' Get the distribution of R vectors with the normalized
            probabilities.
            Here I just assume a Gaussian distribution.
        '''
        radius = self.pargs['radius']
        sigma_R = self.pargs['sigma_R']*radius
        # if zero, ignore
        if np.abs(sigma_R) < 1e-12:
            return np.array([radius]),np.array([1])
        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            return self.distribution_gaussian(radius, sigma_R,
                                              num_points=num_points,
                                              spread=spread)
        elif dist_type == 'uniform':
            return self.distribution_uniform(radius, sigma_R,
                                              num_points=num_points,
                                              spread=spread)

    def distribution_Hs(self, num_points=100, spread=2.5):
        ''' Get the distribution of R vectors with the normalized
            probabilities.
            Here I just assume a Gaussian distribution.
            sigma_H : percentage std dev
        '''
        height = self.pargs['height']
        sigma_H = self.pargs['sigma_H']*height
        # if zero, ignore
        if np.abs(sigma_H) < 1e-12:
            return np.array([height]),np.array([1])
        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            return self.distribution_gaussian(height, sigma_H,
                                              num_points=num_points,
                                              spread=spread)
        elif dist_type == 'uniform':
            return self.distribution_uniform(height, sigma_H,
                                             num_points=num_points,
                                             spread=spread)


    def distribution_thetas(self, arg, num_points=10, spread=2.5):
        ''' orientation distribution
            returns theta_xs, P(theta_x)
            useful for the orientation angles
            arg is 'sigma_theta_x' or etc...

        '''
        # always starts at zero
        alpha = 0.
        # this should be in degrees
        sigma_Alpha = self.pargs[arg]
        # if it's practically zero, don't compute:
        if np.abs(sigma_Alpha) < 1e-6:
            return np.array([0]),np.array([1])

        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            alphas, P_Alpha = self.distribution_gaussian(alpha, sigma_Alpha,
                                                         num_points=num_points,
                                                         spread=spread,
                                                         positive=False)
        elif dist_type == 'uniform':
            alphas, P_Alpha = self.distribution_uniform(alpha, sigma_Alpha,
                                             num_points=num_points,
                                             spread=spread, positive=False)

        # times the sin weight (see log for derivation)
        # need to convert to radians
        P_Alpha *= np.abs(np.sin(np.radians(alphas)))
        # normalize to 1
        P_Alpha /= np.sum(P_Alpha)

        return alphas, P_Alpha


    def rot_sigmaz(self, theta,phi):
        ''' sigmaz rotation matrix (for tilts about z).
            Y1 Z2 Y3
            To precess need to rotate about phi axis
        '''
        theta = np.radians(theta)
        phi= np.radians(phi)
        c1 = cos(theta)
        s1 = sin(theta)
        c2 = cos(phi)
        s2 = sin(phi)
        c3 = 1.
        s3 = 0.
        rotmat = np.array([
            [c1*c2*c3 - s1*s3, -c1*s2, c3*s1 +c1*c2*s3],
            [c3*s2, c2, s2*s3],
            [-c1*s3 -c2*c3*s1, s1*s2, c1*c3 - c2*s1*s3],
        ])
        return rotmat


    def rot_sigmax(self, theta,phi):
        ''' sigmaz rotation matrix (for tilts about z).
            Y1 X2 Y3
        '''
        theta = np.radians(theta)
        phi= np.radians(phi)
        c1 = cos(theta)
        s1 = sin(theta)
        c2 = cos(phi)
        s2 = sin(phi)
        c3 = 1.
        s3 = 0.
        rotmat = np.array([
            [c1*c3 - c1*s2*s3, s1*s2, c1*s3 +c2*c3*s1],
            [s2*s3, c2, -c3*s2],
            [-c3*s1 - c1*c2*s3, c1*s2, c1*c2*c3 -s1*s3]
        ])
        return rotmat


    def rot_sigmay(self, theta,phi):
        ''' sigmaz rotation matrix (for tilts about z).
            X1 Y2 X3
        '''
        theta = np.radians(theta)
        phi= np.radians(phi)
        c1 = cos(theta)
        s1 = sin(theta)
        c2 = cos(phi)
        s2 = sin(phi)
        c3 = 1.
        s3 = 0.
        rotmat = np.array([
            [c2 , s2*s3, c3*s2],
            [s1*s2, c1*c3 - c2*s1*s3, -c1*s3 - c2*c3*s1],
            [-c1*s2, c3*s1 + c1*c2*s3, c1*c2*c3 - s1*s3]
        ])
        return rotmat


    def get_cache_form_factor(self, qx, qy, qz):
        if qx in self.form_factor_already_computed and self.pargs['cache_results']:
            if qy in self.form_factor_already_computed[qx]:
                if qz in self.form_factor_already_computed[qx][qy]:
                    return True, self.form_factor_already_computed[qx][qy][qz]

        return False, 0.0


    def set_cache_form_factor(self, qx, qy, qz, value):
        if self.pargs['cache_results']:
            if qx not in self.form_factor_already_computed:
                self.form_factor_already_computed[qx] = {}
            if qy not in self.form_factor_already_computed[qx]:
                self.form_factor_already_computed[qx][qy] = {}

            self.form_factor_already_computed[qx][qy][qz] = value



    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        # fix for non arrays need to improve later
        qx= np.array(qx)
        qy= np.array(qy)
        qz= np.array(qz)
        if qx.ndim==0:
            qx = np.array([qx])
            qy = np.array([qy])
            qz = np.array([qz])
        #threshold near zero, need to improve later
        thresh = 1e-8
        w = np.where(np.abs(qx) < thresh)
        if len(w[0]) > 0:
            qx[w] = 1e-8
        w = np.where(np.abs(qy) < thresh)
        if len(w[0]) > 0:
            qy[w] = 1e-8
        w = np.where(np.abs(qz) < thresh)
        if len(w[0]) > 0:
            qz[w] = 1e-8
        # hard coded for now
        num_points = 50
        # save the old rotation matrix
        rotation_matrix_orig = self.rotation_matrix
        radius_orig = self.pargs['radius']
        height_orig = self.pargs['height']
        # distribution of radii to go over (could just be one elem)
        rs, rwts = self.distribution_Rs()
        # distribution of heights to go over (could just be one elem)
        hs, hwts = self.distribution_Hs()
        # this is distribution of angles to go over
        # can specify num_points as argument
        alphaxs, walphaxs = self.distribution_thetas('sigma_theta_x',num_points=10)
        alphays, walphays = self.distribution_thetas('sigma_theta_y',num_points=10)
        alphazs, walphazs = self.distribution_thetas('sigma_theta_z',num_points=10)

        # chnage 10 to 100!
        phis, wphis = self.distribution_uniform(180, 180, positive=False,num_points=100)

        # I assume dr, dh, dphi, dalpha all linearly spaced for now
        # (so they cancel out in average)
        Pq = np.zeros(qx.shape,dtype=complex)

        # number of each (could be one)
        Nr = len(rs)
        Nh = len(hs)
        Nalphax = len(alphaxs)
        Nalphay = len(alphays)
        Nalphaz = len(alphazs)

        print("preparing to calculate {} combinations...".format(3*10*100*len(hs)*len(rs)))
        # volume in hyperdimensional probability space
        dV = 0.
        #loop in R
        for r, wr in zip(rs, rwts):
            self.pargs['radius'] = r
            #loop in H
            for h, wh in zip(hs, hwts):
                self.pargs['height'] = h
                #loop in thetax rotation
                # hard code the rotation matrix
                # loop in thetax, only if there is a spread in alpha
                # (else I'm assuming alpha is zero at zero since code
                # here makes this assumption)
                if len(alphaxs) > 1:
                    for alpha, walpha in zip(alphaxs, walphaxs):
                        for phi, wphi in zip(phis, wphis):
                            print("r: {}, h: {}, anglex: {}, phi: {}".format(r,h,alpha,phi))
                            # hard code the rotation matrix
                            self.baseObject.rotation_matrix = self.rot_sigmax(alpha, phi)
                            self.baseObject.rebuild(pargs=self.pargs)
                            weight = wr*wh*walpha*wphi*np.abs(np.sin(alpha))
                            dV += weight
                            Pq += self.baseObject.form_factor(qx, qy, qz)*weight
                else:
                    # NOTE : this weights the code more than it should alpha=phi=0 should
                    # *only* be computed once
                    self.baseObject.rotation_matrix = self.rot_sigmax(0, 0)
                    self.baseObject.rebuild(pargs=self.pargs)
                    weight = wr*wh*1.#*walpha*wphi*np.abs(np.sin(alpha))
                    dV += weight
                    Pq += self.baseObject.form_factor(qx, qy, qz)*weight
                #loop in thetay
                if len(alphays) > 1:
                    for alpha, walpha in zip(alphays, walphays):
                        for phi, wphi in zip(phis, wphis):
                            print("r: {}, h: {}, angley: {}, phi: {}".format(r,h,alpha,phi))
                            # hard code the rotation matrix
                            self.baseObject.rotation_matrix = self.rot_sigmay(alpha, phi)
                            self.baseObject.rebuild(pargs=self.pargs)
                            weight = wr*wh*walpha*wphi*np.abs(np.sin(alpha))
                            dV += weight
                            Pq += self.baseObject.form_factor(qx, qy, qz)*weight
                else:
                    # NOTE : this weights the code more than it should alpha=phi=0 should
                    # *only* be computed once
                    self.baseObject.rotation_matrix = self.rot_sigmax(0, 0)
                    self.baseObject.rebuild(pargs=self.pargs)
                    weight = wr*wh*1.#*walpha*wphi*np.abs(np.sin(alpha))
                    dV += weight
                    Pq += self.baseObject.form_factor(qx, qy, qz)*weight
                #loop in thetaz
                if len(alphazs) > 1:
                    for alpha, walpha in zip(alphazs, walphazs):
                        for phi, wphi in zip(phis, wphis):
                            print("r: {}, h: {}, anglez: {}, phi: {}".format(r,h,alpha,phi))
                            # hard code the rotation matrix
                            self.baseObject.rotation_matrix = self.rot_sigmaz(alpha, phi)
                            self.baseObject.rebuild(pargs=self.pargs)
                            weight = wr*wh*walpha*wphi*np.abs(np.sin(alpha))
                            dV += weight
                            Pq += self.baseObject.form_factor(qx, qy, qz)*weight
                else:
                    # NOTE : this weights the code more than it should alpha=phi=0 should
                    # *only* be computed once
                    self.baseObject.rotation_matrix = self.rot_sigmax(0, 0)
                    self.baseObject.rebuild(pargs=self.pargs)
                    weight = wr*wh*1.#*walpha*wphi*np.abs(np.sin(alpha))
                    dV += weight
                    Pq += self.baseObject.form_factor(qx, qy, qz)*weight
        # divide by all volume
        Pq /= dV

        # reset parameters back to original
        self.baseObject.rotation_matrix = rotation_matrix_orig
        self.baseObject.pargs['height'] = height_orig
        self.baseObject.pargs['radius'] = radius_orig
        self.baseObject.rebuild(pargs=self.pargs)

        return Pq


    def form_factor_squared(self, qx, qy, qz):
        """Returns the square of the form factor."""
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_squared(qx, qy, qz)
            v += wt*v_R*dR

        return v


    def form_factor_intensity(self, qx, qy, qz):
        """Returns the intensity of the form factor."""
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity(qx, qy, qz)
            v += wt*v_R*dR

        return v


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        return v


    def form_factor_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        return v


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        # Check cache
        if self.pargs['cache_results'] and q in self.form_factor_intensity_isotropic_already_computed:
            return self.form_factor_intensity_isotropic_already_computed[q]

        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        # Update cache
        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_already_computed[q] = v

        return v


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        # Check cache
        if self.pargs['cache_results'] and arrays_equal( q_list, self.form_factor_intensity_isotropic_array_already_computed_qlist ):
            return self.form_factor_intensity_isotropic_array_already_computed

        v = np.zeros(len(q_list))
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R.astype(float)*dR

        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_array_already_computed_qlist = q_list
            self.form_factor_intensity_isotropic_array_already_computed = v

        return v



    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<F(q)>|^2"""

        if self.pargs['cache_results'] and q in self.beta_numerator_already_computed:
            return self.beta_numerator_already_computed[q]

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external( q, num_phi=num_phi, num_theta=num_theta )

        else:

            # TODO: Check this code. (Bug suspected, since beta ends up being >1.)

            F = self.form_factor_isotropic( q, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_already_computed[q] = G

        return G


    def beta_numerator_array(self, q_list, num_phi=50, num_theta=50):

        if self.pargs['cache_results'] and arrays_equal( q_list, self.beta_numerator_array_already_computed_qlist ):
            return self.beta_numerator_array_already_computed

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external_array( q_list, num_phi=num_phi, num_theta=num_theta )

        else:
            F = self.form_factor_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_array_already_computed_qlist = q_list
            self.beta_numerator_array_already_computed = G

        return G


    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor(qx, qy, qz)
                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_numerator_iso_external_array__replaced__(self, q_list, num_phi=50, num_theta=50):

        G = np.zeros(len(q_list))

        for i, q in enumerate(q_list):
            G[i] = self.beta_numerator_iso_external(q, num_phi=num_phi, num_theta=num_theta)

        return G


    def beta_numerator_iso_external_array(self, q_list, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = np.zeros(len(q_list))

        for theta in theta_vals:
            qz =  q_list*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q_list*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor_array(qx, qy, qz)

                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            P, beta = self.P_beta( q, num_phi=num_phi, num_theta=num_theta )
            return beta


    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns a 1D array of the beta ratio."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
            return beta
        else:
            P, beta = self.P_beta_array( q_list, num_phi=num_phi, num_theta=num_theta )

            return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and q in self.P_beta_already_computed:
            return self.P_beta_already_computed[q]

        P = self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            G = self.beta_numerator(q, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_already_computed[q] = P, beta

        return P, beta


    def P_beta_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and arrays_equal( q_list, self.P_beta_array_already_computed_qlist ):
            return self.P_beta_array_already_computed

        P = self.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
        else:
            G = self.beta_numerator_array(q_list, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_array_already_computed_qlist = q_list
            self.P_beta_array_already_computed = P, beta

        return P, beta

# PolydisperseRodNanoObject
###################################################################
class PolydisperseRodNanoObject(NanoObject):
    """Defines a polydisperse anisotropic object that has a rod-like shape,
    which has a distribution in:
        size ('radius') of width 'sigma_R' (percentage sigma).
        height ('height') of width 'sigma_H' (percentage sigma).
        and tilts:
        orientation spread 'sigma_tilt_x' (percentage sigma).
        orientation spread 'sigma_tilt_y' (percentage sigma).
        orientation spread 'sigma_tilt_z' (percentage sigma).

    Note : this doesn't have to be a cylindrical object. This could even
    be a sphere (by setting sigma_H to 0 as well as sigma th). Still
    experimental.

    To average over orientations, a new rotation matrix will need to be
    computed.

        """


    def __init__(self, pargs={}, seed=None):

        NanoObject.__init__(self, baseObjectClass=None, pargs=pargs,
                            seed=seed)

        # Set defaultsm no polydispersity if not set
        if 'sigma_R' not in self.pargs:
            self.pargs['sigma_R'] = 0.
        if 'sigma_H' not in self.pargs:
            self.pargs['sigma_H'] = 0.
        if 'sigma_tilt_x' not in self.pargs:
            self.pargs['sigma_tilt_x'] = 0.
        if 'sigma_tilt_y' not in self.pargs:
            self.pargs['sigma_tilt_y'] = 0.
        if 'sigma_tilt_z' not in self.pargs:
            self.pargs['sigma_tilt_z'] = 0.
        if 'height' not in self.pargs:
            self.pargs['height'] = 0.
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 0.
        if 'distribution_type' not in self.pargs:
            self.pargs['distribution_type'] = 'gaussian'
        if 'distribution_num_points' not in self.pargs:
            self.pargs['distribution_num_points'] = 5
        if 'iso_external' not in self.pargs:
            self.pargs['iso_external'] = False
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0


        if baseObjectClass is None:
            baseObjectClass = CylindricalNanoObject
        self.baseObject = baseObjectClass(pargs=pargs)

        self.distribution_list = []


        # caching doesn't work right now, need to resolve this later
        self.pargs['cache_results'] = False

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []


    def rebuild(self, pargs={}, seed=None):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)

        if seed:
            self.seed=seed

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []

        self.distribution_list = []


    def distribution_gaussian(self, x0, sigma_x, num_points=100, spread=2.5):
        ''' Gaussian distribution. Here, assumes that x can only be positive.
            Chooses a range of X and provides its weights. Normalizes by
            the total sum rather than the probability distribution.
            This is really a Gaussian prob dist with cutoff.
            sigma_x : percentage standard deviation
        '''
        # Note, I don't think we need dX since I normalize wts by its
        # sum which also would have a delta X factor in it
        sigma_x = x0*sigma_x
        Xs = np.linspace(np.maximum(0,x0-spread*sigma_x), x0+spread*sigma_x,
                         num_points)
        # normalizes Gaussian dist to 1
        wts = np.exp( - (Xs-x0)**2/(2*( sigma_x**2 ) ) )
        # should be normalized to 1
        wts /= np.sum(wts)

        return Xs, wts

    def distribution_uniform(self, x0, sigma_x, num_points=100, spread=2.5):
        ''' Gaussian distribution. Here, assumes that x can only be positive.
            Chooses a range of X and provides its weights. Normalizes by
            the total sum rather than the probability distribution.
            This is really a Gaussian prob dist with cutoff.
            sigma_x : percentage standard deviation
        '''
        # Note, I don't think we need dX since I normalize wts by its
        # sum which also would have a delta X factor in it
        sigma_x = x0*sigma_x
        Xs = np.linspace(np.maximum(0,x0-spread*sigma_x), x0+spread*sigma_x,
                         num_points)
        # normalizes Gaussian dist to 1
        wts = np.ones_like(Xs)
        # should be normalized to 1
        wts /= np.sum(wts)

        return Xs, wts


    ''' Does not work right now, ignore
        def get_cache_form_factor(self, qx, qy, qz):
            if qx in self.form_factor_already_computed and self.pargs['cache_results']:
                if qy in self.form_factor_already_computed[qx]:
                    if qz in self.form_factor_already_computed[qx][qy]:
                        return True, self.form_factor_already_computed[qx][qy][qz]

            return False, 0.0


        def set_cache_form_factor(self, qx, qy, qz, value):
            if self.pargs['cache_results']:
                if qx not in self.form_factor_already_computed:
                    self.form_factor_already_computed[qx] = {}
                if qy not in self.form_factor_already_computed[qx]:
                    self.form_factor_already_computed[qx][qy] = {}

                self.form_factor_already_computed[qx][qy][qz] = value

    '''

    def form_factor_qr(self, qr, R):
        ''' The qr part of the cylinder form factor, depends on R.'''
        # threshold zero values to something low
        # if it's not low enough, it will be obvious in S(q)'s, but should be
        # okay. Not very fast but better than nothing, simple to type
        return j1(qr*R)/qr/R

    def form_factor_qz(self, qz, h):
        ''' The qz part of the cylinder form factor, depends on h.'''
        return j0(qz*h)

    def distribution_Rs(self, num_points=100, spread=2.5):
        ''' Get the distribution of R vectors with the normalized
            probabilities.
            Here I just assume a Gaussian distribution.
        '''
        radius = self.pargs['radius']
        sigma_R = self.pargs['sigma_R']
        # if zero, ignore
        if np.abs(sigma_R) < 1e-12:
            return np.array([radius]),np.array([1])
        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            return self.distribution_gaussian(radius, sigma_R,
                                              num_points=num_points,
                                              spread=spread)
        elif dist_type == 'uniform':
            return self.distribution_uniform(radius, sigma_R,
                                              num_points=num_points,
                                              spread=spread)

    def distribution_Hs(self, num_points=100, spread=2.5):
        ''' Get the distribution of R vectors with the normalized
            probabilities.
            Here I just assume a Gaussian distribution.
        '''
        height = self.pargs['height']
        sigma_H = self.pargs['sigma_H']
        # if zero, ignore
        if np.abs(sigma_H) < 1e-12:
            return np.array([height]),np.array([1])
        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            return self.distribution_gaussian(height, sigma_H,
                                              num_points=num_points,
                                              spread=spread)
        elif dist_type == 'uniform':
            return self.distribution_uniform(height, sigma_H,
                                             num_points=num_points,
                                             spread=spread)

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            This is the polydisperse version. assumes radius and height
            polydispersity uncorrelated.
        """
        ''' Steps for the form factor average:
                1. First, calc qr, qz
                2. Next, get the distribution of R's and weights
                F(qr), F(qz)
                3. Figure out the range of q's needed
                4. average over this range each component
                5. finally, linearly interpolate
        '''
        qx = np.array(qx)
        # quick fix for one point calculations
        if qx.ndim==0:
            qx = np.array([qx])
            qy = np.array([qy])
            qz = np.array([qz])
        # hard coded, number of points for interpolation, usually 1000 is not so crazy
        Q_interp_points = 1000

        # always get phase before rotation
        phase = self.get_phase(qx,qy,qz)
        # rotate just as for V
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))

        qr = np.hypot(qx, qy)

        # slows down the code but prevents zeros, i.e. eventual np.nans
        qr = np.maximum(np.abs(qr),1e-12)
        qz = np.maximum(np.abs(qz),1e-12)

        Rs, P_R = self.distribution_Rs()
        Hs, P_H = self.distribution_Hs()

        # find the values of qr and qz we need
        qrs_tmp = np.linspace(np.min(qr), np.max(qr), Q_interp_points)
        qzs_tmp = np.linspace(np.min(qz), np.max(qz), Q_interp_points)
        # first, Gaussian, but later will try anything (so move out of function
        # after)

        # now perform the average, slowest varying index is leftermost
        # so [qrs_tmp, Rs] is indexing
        qrs_tmp, Rs = np.meshgrid(qrs_tmp, Rs, indexing='ij')
        qzs_tmp, Hs = np.meshgrid(qzs_tmp, Hs, indexing='ij')
        # integral over radii, broadcast P_R
        F_qr_tmp = np.sum(P_R[np.newaxis,:]*self.form_factor_qr(qrs_tmp, Rs),axis=1)
        # integral over heights, broadcast P_H
        F_qz_tmp = np.sum(P_H[np.newaxis,:]*self.form_factor_qz(qzs_tmp, Hs),axis=1)
        #  average volume
        R2_avg = np.sum(P_R*Rs[0,:]**2)
        H_avg= np.sum(P_H*Hs[0,:])
        # pi R^2 H
        volume = np.pi*R2_avg*H_avg

        # finally, interpolate
        F_qr = np.interp(qr, qrs_tmp[:,0], F_qr_tmp)
        F_qz = np.interp(qz, qzs_tmp[:,0], F_qz_tmp)



        F = 2*F_qr*F_qz*phase
        F *= self.pargs['delta_rho']*volume


        return F

    def form_factor_squared(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            This is the polydisperse version. assumes radius and height
            polydispersity uncorrelated.
        """
        ''' Steps for the form factor average:
                1. First, calc qr, qz
                2. Next, get the distribution of R's and weights
                F(qr), F(qz)
                3. Figure out the range of q's needed
                4. average over this range each component
                5. finally, linearly interpolate
        '''
        # hard coded, number of points for interpolation, usually 1000 is not so crazy
        Q_interp_points = 1000

        # always get phase before rotation
        # not necessary for FF squared
        # phase = self.get_phase(qx,qy,qz)
        # rotate just as for V
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))

        qr = np.hypot(qx, qy)

        # slows down the code but prevents zeros, i.e. eventual np.nans
        qr = np.maximum(np.abs(qr),1e-12)
        qz = np.maximum(np.abs(qz),1e-12)

        Rs, P_R = self.distribution_Rs()
        Hs, P_H = self.distribution_Hs()

        # find the values of qr and qz we need
        qrs_tmp = np.linspace(np.min(qr), np.max(qr), Q_interp_points)
        qzs_tmp = np.linspace(np.min(qz), np.max(qz), Q_interp_points)
        # first, Gaussian, but later will try anything (so move out of function
        # after)

        # now perform the average, slowest varying index is leftermost
        # so [qrs_tmp, Rs] is indexing
        qrs_tmp, Rs = np.meshgrid(qrs_tmp, Rs, indexing='ij')
        qzs_tmp, Hs = np.meshgrid(qzs_tmp, Hs, indexing='ij')
        # integral over radii, broadcast P_R
        F_qr_tmp = np.sum(P_R[np.newaxis,:]*self.form_factor_qr(qrs_tmp, Rs)**2,axis=1)
        # integral over heights, broadcast P_H
        F_qz_tmp = np.sum(P_H[np.newaxis,:]*self.form_factor_qz(qzs_tmp, Hs)**2,axis=1)

        #  average volume
        R2_avg = np.sum(P_R*Rs[0,:]**2)
        H_avg= np.sum(P_H*Hs[0,:])
        # pi <R'^2>_R <H'>_H
        volume = np.pi*R2_avg*H_avg

        # finally, interpolate
        F_qr = np.interp(qr, qrs_tmp[:,0], F_qr_tmp)
        F_qz = np.interp(qz, qzs_tmp[:,0], F_qz_tmp)



        F = 4*F_qr*F_qz
        F *= (self.pargs['delta_rho']*volume)**2


        return F

    '''
    def form_factor_squared(self, qx, qy, qz):
        """Returns the square of the form factor."""
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_squared(qx, qy, qz)
            v += wt*v_R*dR

        return v
    '''


    def form_factor_intensity(self, qx, qy, qz):
        """Returns the intensity of the form factor."""
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity(qx, qy, qz)
            v += wt*v_R*dR

        return v


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        return v


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        # Check cache
        if self.pargs['cache_results'] and q in self.form_factor_intensity_isotropic_already_computed:
            return self.form_factor_intensity_isotropic_already_computed[q]

        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        # Update cache
        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_already_computed[q] = v

        return v


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        # Check cache
        if self.pargs['cache_results'] and arrays_equal( q_list, self.form_factor_intensity_isotropic_array_already_computed_qlist ):
            return self.form_factor_intensity_isotropic_array_already_computed

        v = np.zeros(len(q_list))
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R.astype(float)*dR

        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_array_already_computed_qlist = q_list
            self.form_factor_intensity_isotropic_array_already_computed = v

        return v



    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<F(q)>|^2"""

        if self.pargs['cache_results'] and q in self.beta_numerator_already_computed:
            return self.beta_numerator_already_computed[q]

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external( q, num_phi=num_phi, num_theta=num_theta )

        else:

            # TODO: Check this code. (Bug suspected, since beta ends up being >1.)

            F = self.form_factor_isotropic( q, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_already_computed[q] = G

        return G


    def beta_numerator_array(self, q_list, num_phi=50, num_theta=50):

        if self.pargs['cache_results'] and arrays_equal( q_list, self.beta_numerator_array_already_computed_qlist ):
            return self.beta_numerator_array_already_computed

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external_array( q_list, num_phi=num_phi, num_theta=num_theta )

        else:
            F = self.form_factor_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_array_already_computed_qlist = q_list
            self.beta_numerator_array_already_computed = G

        return G


    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor(qx, qy, qz)
                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_numerator_iso_external_array__replaced__(self, q_list, num_phi=50, num_theta=50):

        G = np.zeros(len(q_list))

        for i, q in enumerate(q_list):
            G[i] = self.beta_numerator_iso_external(q, num_phi=num_phi, num_theta=num_theta)

        return G


    def beta_numerator_iso_external_array(self, q_list, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = np.zeros(len(q_list))

        for theta in theta_vals:
            qz =  q_list*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q_list*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor_array(qx, qy, qz)

                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            P, beta = self.P_beta( q, num_phi=num_phi, num_theta=num_theta )
            return beta


    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns a 1D array of the beta ratio."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
            return beta
        else:
            P, beta = self.P_beta_array( q_list, num_phi=num_phi, num_theta=num_theta )

            return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and q in self.P_beta_already_computed:
            return self.P_beta_already_computed[q]

        P = self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            G = self.beta_numerator(q, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_already_computed[q] = P, beta

        return P, beta


    def P_beta_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and arrays_equal( q_list, self.P_beta_array_already_computed_qlist ):
            return self.P_beta_array_already_computed

        P = self.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
        else:
            G = self.beta_numerator_array(q_list, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_array_already_computed_qlist = q_list
            self.P_beta_array_already_computed = P, beta

        return P, beta

# PolydisperseOrientationSpreadCylindricalObject
###################################################################
class PolydisperseOrientationSpreadCylindricalObject(NanoObject):
    """Defines a polydisperse cylindrical-shaped object, which has a
    distribution in:
        size ('radius') of width 'sigma_R' (percentage sigma).
        height ('height') of width 'sigma_H' (percentage sigma).
        (ignore this one for now:)
            orientation spread 'sigma_th' (percentage sigma).

        The orientational spread assumes a rod that can tilt in theta up to delta_theta,
            but isotropic in azimuth phi.
            This is like a tilt in spherical coordinates in theta but 2*pi in phi

        sigma_Alpha : convention is degrees

        """


    def __init__(self, pargs={}, seed=None):

        NanoObject.__init__(self, pargs=pargs, seed=seed)

        # Set defaults
        if 'sigma_R' not in self.pargs:
            self.pargs['sigma_R'] = 0.01
        if 'sigma_H' not in self.pargs:
            self.pargs['sigma_H'] = 0.01
        # convention is always degrees
        if 'sigma_Alpha' not in self.pargs:
            self.pargs['sigma_Alpha'] = 1.
        if 'distribution_type' not in self.pargs:
            self.pargs['distribution_type'] = 'gaussian'
        if 'distribution_num_points' not in self.pargs:
            self.pargs['distribution_num_points'] = 5
        if 'iso_external' not in self.pargs:
            self.pargs['iso_external'] = False
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0


        self.baseNanoObjectClass =  []

        self.distribution_list = []


        # caching doesn't work right now
        self.pargs['cache_results'] = False

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []


    def rebuild(self, pargs={}, seed=None):
        """Allows the object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)

        if seed:
            self.seed=seed

        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}

        self.form_factor_intensity_isotropic_array_already_computed_qlist = []
        self.form_factor_intensity_isotropic_array_already_computed = []

        self.beta_numerator_already_computed = {}
        self.beta_numerator_array_already_computed_qlist = []
        self.beta_numerator_array_already_computed = []

        self.P_beta_already_computed = {}
        self.P_beta_array_already_computed_qlist = []
        self.P_beta_array_already_computed = []

        self.distribution_list = []


    def distribution_gaussian(self, x0, sigma_x, num_points=100, spread=2.5,positive=True):
        ''' Gaussian distribution. Here, assumes that x can only be positive.
            Chooses a range of X and provides its weights. Normalizes by
            the total sum rather than the probability distribution.
            This is really a Gaussian prob dist with cutoff.
            sigma_x : standard deviation
            positive : assumes x can only be positive (useful for radii etc, not for angles)
        '''
        # Note, I don't think we need dX since I normalize wts by its
        # sum which also would have a delta X factor in it
        if positive:
            Xs = np.linspace(np.maximum(0,x0-spread*sigma_x),
                             x0+spread*sigma_x, num_points)
        else:
            Xs = np.linspace(x0-spread*sigma_x, x0+spread*sigma_x, num_points)

        # Gaussian weights
        wts = np.exp( - (Xs-x0)**2/(2*( sigma_x**2 ) ) )
        # should be normalized to 1
        wts /= np.sum(wts)

        return Xs, wts

    def distribution_uniform(self, x0, sigma_x, num_points=100, positive=True):
        ''' Gaussian distribution. Here, assumes that x can only be positive.
            Chooses a range of X and provides its weights. Normalizes by
            the total sum rather than the probability distribution.
            This is really a Gaussian prob dist with cutoff.
            sigma_x : percentage standard deviation
        '''
        # Note, I don't think we need dX since I normalize wts by its
        # sum which also would have a delta X factor in it
        if positive:
            Xs = np.linspace(np.maximum(0,x0-sigma_x), x0 + sigma_x,
                             num_points)
        else:
            Xs = np.linspace(x0-sigma_x, x0+sigma_x, num_points)

        # uniform weights
        wts = np.ones_like(Xs)
        # should be normalized to 1
        wts /= np.sum(wts)

        return Xs, wts


    ''' Does not work right now, ignore
        def get_cache_form_factor(self, qx, qy, qz):
            if qx in self.form_factor_already_computed and self.pargs['cache_results']:
                if qy in self.form_factor_already_computed[qx]:
                    if qz in self.form_factor_already_computed[qx][qy]:
                        return True, self.form_factor_already_computed[qx][qy][qz]

            return False, 0.0


        def set_cache_form_factor(self, qx, qy, qz, value):
            if self.pargs['cache_results']:
                if qx not in self.form_factor_already_computed:
                    self.form_factor_already_computed[qx] = {}
                if qy not in self.form_factor_already_computed[qx]:
                    self.form_factor_already_computed[qx][qy] = {}

                self.form_factor_already_computed[qx][qy][qz] = value

    '''

    def form_factor_qr(self, qr, R):
        ''' The qr part of the cylinder form factor, depends on R.'''
        # threshold zero values to something low
        # if it's not low enough, it will be obvious in S(q)'s, but should be
        # okay. Not very fast but better than nothing, simple to type
        return j1(qr*R)/qr/R

    def form_factor_qz(self, qz, h):
        ''' The qz part of the cylinder form factor, depends on h.'''
        return j0(qz*h)

    def distribution_Rs(self, num_points=100, spread=2.5):
        ''' Get the distribution of R vectors with the normalized
            probabilities.
            Here I just assume a Gaussian distribution.
        '''
        radius = self.pargs['radius']
        sigma_R = self.pargs['sigma_R']*radius
        # if zero, ignore
        if np.abs(sigma_R) < 1e-12:
            return np.array([radius]),np.array([1])
        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            return self.distribution_gaussian(radius, sigma_R,
                                              num_points=num_points,
                                              spread=spread)
        elif dist_type == 'uniform':
            return self.distribution_uniform(radius, sigma_R,
                                              num_points=num_points,
                                              spread=spread)

    def distribution_Hs(self, num_points=100, spread=2.5):
        ''' Get the distribution of R vectors with the normalized
            probabilities.
            Here I just assume a Gaussian distribution.
            sigma_H : percentage std dev
        '''
        height = self.pargs['height']
        sigma_H = self.pargs['sigma_H']*height
        # if zero, ignore
        if np.abs(sigma_H) < 1e-12:
            return np.array([height]),np.array([1])
        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            return self.distribution_gaussian(height, sigma_H,
                                              num_points=num_points,
                                              spread=spread)
        elif dist_type == 'uniform':
            return self.distribution_uniform(height, sigma_H,
                                             num_points=num_points,
                                             spread=spread)

    def distribution_Alphas(self, num_points=10, spread=2.5):
        ''' orientation distribution
            returns alphas, phis, P_phi, P(alpha)

        '''
        # always starts at zero
        alpha = 0.
        # this should be in degrees
        sigma_Alpha = self.pargs['sigma_Alpha']
        # if it's practically zero, don't compute:
        if np.abs(sigma_Alpha) < 1e-6:
            return np.array([0]),np.array([0]), np.array([1.]), np.array([1.])

        dist_type = self.pargs['distribution_type']
        if dist_type == 'gaussian':
            alphas, P_Alpha = self.distribution_gaussian(alpha, sigma_Alpha,
                                                         num_points=num_points,
                                                         spread=spread,
                                                         positive=False)
        elif dist_type == 'uniform':
            alphas, P_Alpha = self.distribution_uniform(alpha, sigma_Alpha,
                                             num_points=num_points,
                                             spread=spread, positive=False)

        # times the sin weight (see log for derivation)
        # need to convert to radians
        P_Alpha *= np.abs(np.sin(np.radians(alphas)))
        # normalize to 1
        P_Alpha /= np.sum(P_Alpha)


        # uniform in phi, needs to also be degrees
        phis, P_Phi = self.distribution_uniform(180, 180, num_points, positive=False)

        return alphas, phis, P_Alpha, P_Phi

    def form_factor(self, qx, qy, qz, num_points=10):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            This is the polydisperse version. assumes radius and height
            polydispersity uncorrelated.
        """
        ''' Steps for the form factor average:
                1. First, calc qr, qz
                2. Next, get the distribution of R's and weights
                F(qr), F(qz)
                3. Figure out the range of q's needed
                4. average over this range each component
                5. finally, linearly interpolate
        '''
        # hard coded, number of points for interpolation, usually 1000 is not so crazy
        Q_interp_points = 1000

        # always get phase before rotation
        phase = self.get_phase(qx,qy,qz)

        # rotate just as for V
        qcoord = np.array([qx,qy,qz])
        qx, qy, qz = self.map_qcoord(qcoord)
        qcoord = np.array([qx,qy,qz])

        qr = np.hypot(qx, qy)

        # slows down the code but prevents zeros, i.e. eventual np.nans
        # also converts to absolute value, this ff is symmetric so it's okay
        qr = np.maximum(np.abs(qr),1e-12)
        qz = np.maximum(np.abs(qz),1e-12)

        Rs, P_R = self.distribution_Rs()
        Hs, P_H = self.distribution_Hs()

        # find the values of qr and qz we need
        # this time, since we average alpha, worst case we have a 90 deg rotation
        # and r maps into z, thus both should have similar ranges
        # unfortunately, qmin needs to be zero (here I take close to zero)
        qmax = np.maximum(np.max(qr), np.max(qz))
        qrs_interp = np.linspace(1e-12, qmax, Q_interp_points)
        qzs_interp = np.linspace(1e-12, qmax, Q_interp_points)

        # now perform the average, slowest varying index is leftermost
        # so [qrs_tmp, Rs] is indexing
        # buld the interpolation table
        qrs_interp, Rs = np.meshgrid(qrs_interp, Rs, indexing='ij')
        qzs_interp, Hs = np.meshgrid(qzs_interp, Hs, indexing='ij')
        # integral over radii, broadcast P_R
        F_qr_interp = np.sum(P_R[np.newaxis,:]*self.form_factor_qr(qrs_interp, Rs),axis=1)
        # integral over heights, broadcast P_H
        F_qz_interp = np.sum(P_H[np.newaxis,:]*self.form_factor_qz(qzs_interp, Hs),axis=1)

        alphas, phis, P_Alpha, P_Phi = self.distribution_Alphas(num_points=num_points)
        F = np.zeros_like(qr,dtype=complex)
        # now average over orientation < < < F >_r >_h >_o
        #for i in tqdm(range(len(alphas))):
        for i in range(len(alphas)):
            for j in range(len(phis)):
                # rotate the coordinates
                rotmat = self.rotation_elements(0, alphas[i], phis[j])
                # calc new coordinates
                qxnew, qynew, qznew = self.map_coord(qcoord, rotmat)
                qrnew = np.hypot(qxnew, qynew)
                # interpolate from qr, qz
                # only depends on absolute value
                F_qr_tmp = np.interp(np.abs(qrnew), qrs_interp[:,0], F_qr_interp)
                F_qz_tmp = np.interp(np.abs(qznew), qzs_interp[:,0], F_qz_interp)
                # finally calcuate form factor with probabilities
                ftmp = P_Alpha[i]*P_Phi[j]*2*F_qr_tmp*F_qz_tmp
                # add to final result
                F += ftmp

        # no need to normalize since factors were weighted

        #  average volume
        R2_avg = np.sum(P_R*Rs[0,:]**2)
        H_avg= np.sum(P_H*Hs[0,:])
        # pi R^2 H
        volume = np.pi*R2_avg*H_avg


        F *= self.pargs['delta_rho']*volume*phase


        return F

    def form_factor_squared(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            This is the polydisperse version. assumes radius and height
            polydispersity uncorrelated.
        """
        ''' Steps for the form factor average:
                1. First, calc qr, qz
                2. Next, get the distribution of R's and weights
                F(qr), F(qz)
                3. Figure out the range of q's needed
                4. average over this range each component
                    we now have an interpolation table for <f>r,h
                5.
                5. finally, linearly interpolate
        '''
        # hard coded, number of points for interpolation, usually 1000 is not so crazy
        Q_interp_points = 1000

        # always get phase before rotation
        # not needed for form fact squared
        # phase = self.get_phase(qx,qy,qz)

        # rotate just as for V
        qcoord = np.array([qx,qy,qz])
        qx, qy, qz = self.map_qcoord(qcoord)
        qcoord = np.array([qx,qy,qz])

        qr = np.hypot(qx, qy)

        # slows down the code but prevents zeros, i.e. eventual np.nans
        qr = np.maximum(np.abs(qr),1e-12)
        qz = np.maximum(np.abs(qz),1e-12)

        Rs, P_R = self.distribution_Rs()
        Hs, P_H = self.distribution_Hs()

        # find the values of qr and qz we need
        # this time, since we average alpha, worst case we have a 90 deg rotation
        # and r maps into z, thus both should have similar ranges
        # unfortunately, qmin needs to be zero (here I take close to zero)
        qmax = np.maximum(np.max(qr), np.max(qz))
        qrs_interp = np.linspace(1e-12, qmax, Q_interp_points)
        qzs_interp = np.linspace(1e-12, qmax, Q_interp_points)

        # now perform the average, slowest varying index is leftermost
        # so [qrs_tmp, Rs] is indexing
        # buld the interpolation table
        qrs_interp, Rs = np.meshgrid(qrs_interp, Rs, indexing='ij')
        qzs_interp, Hs = np.meshgrid(qzs_interp, Hs, indexing='ij')
        # integral over radii, broadcast P_R
        F_qr_interp = np.sum(P_R[np.newaxis,:]*self.form_factor_qr(qrs_interp, Rs)**2,axis=1)
        # integral over heights, broadcast P_H
        F_qz_interp = np.sum(P_H[np.newaxis,:]*self.form_factor_qz(qzs_interp, Hs)**2,axis=1)

        alphas, phis, P_Alpha, P_Phi = self.distribution_Alphas()
        F = np.zeros_like(qr)
        # now average over orientation < < < F >_r >_h >_o
        #for i in tqdm(range(len(alphas))):
        for i in range(len(alphas)):
            for j in range(len(phis)):
                # rotate the coordinates
                rotmat = self.rotation_elements(0, alphas[i], phis[j])
                # calc new coordinates
                qxnew, qynew, qznew = self.map_coord(qcoord, rotmat)
                qrnew = np.hypot(qxnew, qynew)
                # interpolate from qr, qz
                # only depends on absolute value
                F_qr_tmp = np.interp(qrnew, qrs_interp[:,0], F_qr_interp)
                F_qz_tmp = np.interp(np.abs(qznew), qzs_interp[:,0], F_qz_interp)
                #raise ValueError
                # finally calcuate form factor with probabilities
                ftmp = P_Alpha[i]*P_Phi[j]*4*F_qr_tmp*F_qz_tmp
                # add to final result
                F += ftmp

        # no need to normalize since factors were weighted

        #  average volume
        R2_avg = np.sum(P_R*Rs[0,:]**2)
        H_avg= np.sum(P_H*Hs[0,:])
        # pi R^2 H
        volume = np.pi*R2_avg*H_avg


        F *= (self.pargs['delta_rho']*volume)**2

        return F

    '''
    def form_factor_squared(self, qx, qy, qz):
        """Returns the square of the form factor."""
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_squared(qx, qy, qz)
            v += wt*v_R*dR

        return v
    '''


    def form_factor_intensity(self, qx, qy, qz):
        """Returns the intensity of the form factor."""
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity(qx, qy, qz)
            v += wt*v_R*dR

        return v


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        return v


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        # Check cache
        if self.pargs['cache_results'] and q in self.form_factor_intensity_isotropic_already_computed:
            return self.form_factor_intensity_isotropic_already_computed[q]

        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

        # Update cache
        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_already_computed[q] = v

        return v


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        # Check cache
        if self.pargs['cache_results'] and arrays_equal( q_list, self.form_factor_intensity_isotropic_array_already_computed_qlist ):
            return self.form_factor_intensity_isotropic_array_already_computed

        v = np.zeros(len(q_list))
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R.astype(float)*dR

        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_array_already_computed_qlist = q_list
            self.form_factor_intensity_isotropic_array_already_computed = v

        return v



    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<F(q)>|^2"""

        if self.pargs['cache_results'] and q in self.beta_numerator_already_computed:
            return self.beta_numerator_already_computed[q]

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external( q, num_phi=num_phi, num_theta=num_theta )

        else:

            # TODO: Check this code. (Bug suspected, since beta ends up being >1.)

            F = self.form_factor_isotropic( q, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_already_computed[q] = G

        return G


    def beta_numerator_array(self, q_list, num_phi=50, num_theta=50):

        if self.pargs['cache_results'] and arrays_equal( q_list, self.beta_numerator_array_already_computed_qlist ):
            return self.beta_numerator_array_already_computed

        if self.pargs['iso_external']:
            G = self.beta_numerator_iso_external_array( q_list, num_phi=num_phi, num_theta=num_theta )

        else:
            F = self.form_factor_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta)
            G = F*F.conjugate()

            # Note that we have to divide by 4*pi
            # This is because the spherical integration inside the ||^2, so the
            # surface area term gets squared, to (4*pi)^2
            G = G.real/(4*np.pi)

        if self.pargs['cache_results']:
            self.beta_numerator_array_already_computed_qlist = q_list
            self.beta_numerator_array_already_computed = G

        return G


    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor(qx, qy, qz)
                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_numerator_iso_external_array__replaced__(self, q_list, num_phi=50, num_theta=50):

        G = np.zeros(len(q_list))

        for i, q in enumerate(q_list):
            G[i] = self.beta_numerator_iso_external(q, num_phi=num_phi, num_theta=num_theta)

        return G


    def beta_numerator_iso_external_array(self, q_list, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        G = np.zeros(len(q_list))

        for theta in theta_vals:
            qz =  q_list*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q_list*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F = self.form_factor_array(qx, qy, qz)

                F2 = F*F.conjugate()
                G += F2.real * dS

        return G


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            P, beta = self.P_beta( q, num_phi=num_phi, num_theta=num_theta )
            return beta


    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns a 1D array of the beta ratio."""

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
            return beta
        else:
            P, beta = self.P_beta_array( q_list, num_phi=num_phi, num_theta=num_theta )

            return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and q in self.P_beta_already_computed:
            return self.P_beta_already_computed[q]

        P = self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            G = self.beta_numerator(q, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_already_computed[q] = P, beta

        return P, beta


    def P_beta_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""

        if self.pargs['cache_results'] and arrays_equal( q_list, self.P_beta_array_already_computed_qlist ):
            return self.P_beta_array_already_computed

        P = self.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)

        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = np.exp( -( (radius*sigma_R*q_list)**2 ) )
        else:
            G = self.beta_numerator_array(q_list, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        if self.pargs['cache_results']:
            self.P_beta_array_already_computed_qlist = q_list
            self.P_beta_array_already_computed = P, beta

        return P, beta






# CubeNanoObject
###################################################################
class CubeNanoObject(NanoObject):

    def __init__(self, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                    }


        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0

        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""
        R = self.pargs['radius']
        if abs(in_x)<R and abs(in_y)<R and abs(in_z)<R:
            return 1.0
        else:
            return 0.0


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        import numpy
        from numpy import sinc
        qx, qy, qz = self.rotate_coord(qx, qy, qz)

        R = self.pargs['radius']
        volume = (2*R)**3

        F = self.pargs['delta_rho']*volume*sinc(qx*R)*sinc(qy*R)*sinc(qz*R)+0.0j

        return F


    def form_factor_array(self, qx, qy, qz):

        R = self.pargs['radius']
        volume = (2*R)**3

        F = np.zeros( (len(qx)), dtype=np.complex )

        qx, qy, qz = self.rotate_coord(qx, qy, qz)
        F = self.pargs['delta_rho']*volume*np.sinc(qx*R/np.pi)*np.sinc(qy*R/np.pi)*np.sinc(qz*R/np.pi)

        return F


    def form_factor_isotropic_unoptimized(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation."""

        # TODO: This function is no longer necessary, and can be removed

        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi/2, num_theta, endpoint=False, retstep=True )


        F = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F += self.form_factor(qx, qy, qz) * dS


        return 8*F


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation."""

        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi/2, num_theta, endpoint=False, retstep=True )

        # The code below is optimized (especially removing loop invariants)

        R = self.pargs['radius']
        #volume = (2*R)**3
        #prefactor = self.pargs['delta_rho']*volume/(R**3)
        prefactor = self.pargs['delta_rho']*8 # Factor of eight accounts for normalized volume

        F = 0.0+0.0j
        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q*cos(theta)
                dS = sin(theta)*dtheta*dphi

                # Start computing partial values
                qy_partial =  q*sin(theta)
                for phi in phi_vals:
                    # When phi==0, the integrand is zero
                    if phi!=0:
                        qx = -qy_partial*cos(phi)
                        qy = qy_partial*sin(phi)

                        F += dS * sin(qx*R)*sin(qy*R)*sin(qz*R)/(qx*qy*qz)


        return 8*prefactor*F # Factor of eight accounts for having only done one octant


    def form_factor_intensity_isotropic(self, q, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi/2, num_theta, endpoint=False, retstep=True )

        R = self.pargs['radius']
        volume = (2*R)**3

        if q==0:
            return ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi

        prefactor = 128*( (self.pargs['delta_rho']*volume)**2 )/( (q*R)**6 ) # Includes the factor (8) to account for only measuring one octant
        #prefactor = 16*64*( (self.pargs['delta_rho'])**2 )/( q**6 )

        P = 0.0

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q*cos(theta)
                theta_part = (sin(qz*R))/(sin(2*theta))
                theta_part = dtheta*dphi*(theta_part**2)/(sin(theta))

                # Start computing partial values
                qy_partial =  q*sin(theta)
                for phi in phi_vals:

                    # When phi==0, the integrand is zero
                    if phi!=0:
                        qx = -qy_partial*cos(phi)
                        qy = qy_partial*sin(phi)

                        phi_part = sin(qx*R)*sin(qy*R)/( sin(2*phi) )

                        P += theta_part*( phi_part**2 )


        return prefactor*P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi/2, num_theta, endpoint=False, retstep=True )

        R = self.pargs['radius']
        volume = (2*R)**3

        prefactor = 128*( (self.pargs['delta_rho']*volume)**2 )/( (q_list*R)**6 )
        #prefactor = 16*64*( (self.pargs['delta_rho'])**2 )/( q**6 )

        P = np.zeros( len(q_list) )

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q_list*cos(theta)
                theta_part = (np.sin(qz*R))/(sin(2*theta))
                theta_part = dtheta*dphi*(theta_part**2)/(sin(theta))

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    # When phi==0, the integrand is zero
                    if phi!=0:
                        qx = -qy_partial*cos(phi)
                        qy = qy_partial*sin(phi)

                        phi_part = np.sin(qx*R)*np.sin(qy*R)/( sin(2*phi) )

                        P += theta_part*( phi_part**2 )

        P *= prefactor
        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi

        return P


    def to_string(self):
        """Returns a string describing the object."""
        L = 2.0*self.pargs['radius']
        s = "CubeNanoObject: A cube of edge length = %.3f nm" % L
        return s


    def to_short_string(self):
        """Returns a short string describing the object's variables.
        (Useful for distinguishing objects of the same class.)"""
        L = 2.0*self.pargs['radius']
        s = "L = %.3f nm" % L
        return s


    def to_povray_string(self):
        """Returns a string with POV-Ray code for visualizing the object."""

        texture_num = int(self.pargs['rho1']*10)
        s = "cube { -%f, %f texture { obj_texture%05d } }" % (self.pargs['radius'],self.pargs['radius'],texture_num)

        return s


# CubePolydisperseNanoObject
###################################################################
class CubePolydisperseNanoObject(PolydisperseNanoObject):

    def __init__(self, pargs={}, seed=None):

        PolydisperseNanoObject.__init__(self, CubeNanoObject, pargs=pargs, seed=seed)





# SuperballNanoObject
###################################################################
class SuperballNanoObject(NanoObject):

    def __init__(self, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                    }


        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0


        self.form_factor_intensity_isotropic_already_computed = {}

        if 'superball_p' not in self.pargs:
            # 0.5 gives an octahedron
            # 1.0 gives a sphere
            # infinity gives a cube
            self.pargs['superball_p'] = 2.5

        if 'num_points_realspace' not in self.pargs:
            # How to partition realspace (used for numerical calculation of form factor, etc.)
            self.pargs['num_points_realspace'] = 30

        if 'cache_results' not in self.pargs:
            self.pargs['cache_results' ] = True



        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )


        # Generate realspace representation
        size = self.pargs['num_points_realspace']
        extent = 1.1*self.pargs['radius'] # Size of the box that contains the shape
        p = self.pargs['superball_p']
        threshold = abs(self.pargs['radius'])**(2.0*p)
        X, Y, Z = np.mgrid[ -extent:+extent:size*1j , -extent:+extent:size*1j , -extent:+extent:size*1j ]
        self.realspace_box = np.where( ( np.power(np.abs(X),2.0*p) + np.power(np.abs(Y),2.0*p) + np.power(np.abs(Z),2.0*p) )<threshold, 1, 0 )

        # self.realspace_box is a 3D box, with each grid-point having either a 1 or 0, representing the interior and exterior of the superball shape


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""

        p = self.pargs['superball_p']
        threshold = abs(self.pargs['radius'])**(2.0*p)

        if abs(in_x)**(2.0*p)<threshold and abs(in_y)**(2.0*p)<threshold and abs(in_z)**(2.0*p)<threshold:
            return 1.0
        else:
            return 0.0


    def form_factor_numerical(self, qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None):
        """Note that num_points and rotation_elements are ignored"""

        #print( 'form_factor_numerical start: qx = %.4f, qy = %.4f, qz = %.4f' % (qx, qy, qz) )
        size = self.pargs['num_points_realspace']
        extent = 1.1*self.pargs['radius'] # Size of the box that contains the shape
        x_vector, dx = np.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        exp_iqxx = np.exp( 1j*qx*x_vector )
        y_vector, dy = np.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        exp_iqyy = np.exp( 1j*qy*y_vector )
        z_vector, dz = np.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        exp_iqzz = np.exp( 1j*qz*z_vector )

        F_matrix = self.realspace_box*( exp_iqzz.reshape(size,1,1) )*( exp_iqyy.reshape(1,size,1) )*( exp_iqxx.reshape(1,1,size) )

        F = F_matrix.sum()

        #print( 'form_factor_numerical done.' )
        return self.pargs['delta_rho']*F*dx*dy*dz


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        qx, qy, qz = self.rotate_coord(qx, qy, qz)

        return self.form_factor_numerical(qx, qy, qz, num_points=100 )


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        if q in self.form_factor_intensity_isotropic_already_computed and self.pargs['cache_results']:
            return self.form_factor_intensity_isotropic_already_computed[q]

        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = np.linspace( 0, np.pi/2.0, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi/2.0, num_theta, endpoint=False, retstep=True )

        size = self.pargs['num_points_realspace']
        extent = 1.1*self.pargs['radius'] # Size of the box that contains the shape
        x_vector, dx = np.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        y_vector, dy = np.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        z_vector, dz = np.linspace(-extent, +extent, size, endpoint=True, retstep=True)

        prefactor = 8*( (self.pargs['delta_rho']*dx*dy*dz)**2 ) # Factor of eight accounts for only doing one octant


        P = 0.0

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q*cos(theta)
                exp_iqzz = np.exp( 1j*qz*z_vector ).reshape(size,1,1)
                F_matrix_partial = self.realspace_box*( exp_iqzz )
                dS = sin(theta)*dtheta*dphi

                for phi in phi_vals:
                    # When phi==0, the integrand is zero
                    if phi!=0:

                        qx = -q*sin(theta)*cos(phi)
                        exp_iqxx = np.exp( 1j*qx*x_vector ).reshape(1,1,size)
                        qy =  q*sin(theta)*sin(phi)
                        exp_iqyy = np.exp( 1j*qy*y_vector ).reshape(1,size,1)

                        F_matrix = F_matrix_partial*( exp_iqyy )*( exp_iqxx )
                        F = F_matrix.sum()

                        P += F*F.conjugate() * dS

        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_already_computed[q] = prefactor*P

        return prefactor*P


    def form_factor_intensity_isotropic_standard(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)

        This is a generic unoptimized method
        """

        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi/2, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                P += self.form_factor_squared(qx, qy, qz) * dS


        return 8*P


    def to_short_string(self):
        """Returns a short string describing the object's variables.
        (Useful for distinguishing objects of the same class.)"""
        R = 1.0*self.pargs['radius']
        s = "R = %.3f nm" % R
        return s


# SuperballPolydisperseNanoObject
###################################################################
class SuperballPolydisperseNanoObject(PolydisperseNanoObject):

    def __init__(self, pargs={}, seed=None):

        PolydisperseNanoObject.__init__(self, SuperballNanoObject, pargs=pargs, seed=seed)




# SphereNanoObject
##################################################################
class SphereNanoObject(NanoObject):



    def __init__(self, pargs={}, seed=None):
        self.rotation_matrix = np.identity(3)
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                    }



        # caching doesn't work right now
        self.pargs['cache_results'] = False
        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        if 'radius' not in self.pargs:
            # Set a default size
            self.pargs['radius'] = 1.0

        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0

        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0

        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0


        # Store values we compute to avoid laborious re-computation
        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""

        in_x, in_y, in_z = self.map_rcoord(np.array([in_x, in_y, in_z]))
        R = self.pargs['radius']
        r = np.sqrt( in_x**2 + in_y**2 + in_z**2 )
        return (r < R).astype(float)



    def form_factor(self, qx, qy, qz):
        qx = np.array(qx)
        if qx.ndim == 0:
            qx = np.array([qx])
            qy = np.array([qy])
            qz = np.array([qz])

        R = self.pargs['radius']
        volume = (4.0/3.0)*np.pi*(R**3)
        qR = R*np.sqrt( qx**2 + qy**2 + qz**2 )
        # threshold to avoid zero values
        qR = np.maximum(1e-8, qR)

        F = np.zeros( (len(qx)), dtype=np.complex )
        #for i, qxi in enumerate(qx):
            #F[i] = self.form_factor(qx[i], qy[i], qz[i])

        F = 3.0*self.pargs['delta_rho']*volume*( np.sin(qR) - qR*np.cos(qR) )/( qR**3 )

        return F


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation."""

        return 4*np.pi * self.form_factor( q, 0, 0)


    def form_factor_intensity_isotropic(self, q_list, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        q_list = np.array(q_list)

        if q_list.ndim == 0:
            q_list = np.array([q_list])

        # threshold
        q = np.maximum(np.abs(q_list),1e-8)

        R = self.pargs['radius']

        volume = (4.0/3.0)*np.pi*(R**3)

        prefactor = 36*np.pi*( (self.pargs['delta_rho']*volume)**2 )


        P = np.empty( (len(q_list)) )
        P = prefactor*( (np.sin(q_list*R)-q_list*R*np.cos(q_list*R))**2 )/((q_list*R)**6)

        return P

    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        return self.form_factor_intensity_isotropic(q_list,num_phi=num_phi, num_theta=num_theta)

    def to_string(self):
        """Returns a string describing the object."""
        s = "SphereNanoObject: A sphere of radius = %.3f nm" % self.pargs['radius']
        return s

    def to_short_string(self):
        """Returns a short string describing the object's variables.
        (Useful for distinguishing objects of the same class.)"""
        s = "R = %.3f nm" % self.pargs['radius']
        return s

    def to_povray_string(self):
        """Returns a string with POV-Ray code for visualizing the object."""

        s = "sphere { <0,0,0>, %f texture { obj_texture%05d } }" % (self.pargs['radius'], texture_num)

        return s


# SpherePolydisperseNanoObject
###################################################################
class SpherePolydisperseNanoObject(PolydisperseNanoObject):

    def __init__(self, pargs={}, seed=None):

        PolydisperseNanoObject.__init__(self, SphereNanoObject, pargs=pargs, seed=seed)

        if 'iso_external' not in pargs:
            self.pargs['iso_external'] = False


# PyramidNanoObject
###################################################################
class PyramidNanoObject(NanoObject):
    """A square-based pyramid nano-object. The canonical (unrotated) version
    has the square-base in the x-y plane, with the peak pointing along +z.
    The base-edges are parallel to the x-axis and y-axis (i.e. the corners
    point at 45 degrees to axes.
    """

    def __init__(self, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'height': None, \
                            'pyramid_face_angle': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        if self.pargs['height']==None:
            # Assume user wants a 'regular' pyramid
            self.pargs['height'] = np.sqrt(2.0)*self.pargs['radius']
        if self.pargs['pyramid_face_angle']==None:
            self.pargs['pyramid_face_angle'] = 54.7356

        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0

        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}



    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""

        in_x = np.array(in_x)
        in_y = np.array(in_y)
        in_z = np.array(in_z)

        # first rotate
        in_x, in_y, in_z = self.map_rcoord(np.array([in_x, in_y, in_z]))

        R = self.pargs['radius']
        H = min( self.pargs['height'] , R/np.tan( np.radians(self.pargs['pyramid_face_angle']) ) )
        R_z = R - in_z/np.tan( np.radians(self.pargs['pyramid_face_angle']) )
        V = ((in_z < H)*(in_z > 0)*(np.abs(in_x) < np.abs(R_z))*(np.abs(in_y) < np.abs(R_z))).astype(float)
        return V

    def thresh_near_zero(self, q):
        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = np.tan(np.radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )

        w = np.where(q < 1e-6)
        if len(w[0]) > 0:
            q[w] = 1e-6#self.pargs['delta_rho']*volume


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        qx = np.array(qx,ndmin=1)
        qy = np.array(qy,ndmin=1)
        qz = np.array(qz,ndmin=1)

        qx, qy, qz = self.rotate_coord(qx, qy, qz)


        #F = self.form_factor_numerical(qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None)

        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = np.tan(np.radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )


        # NOTE: (partial kludge) The computation below will hit a divide-by-zero
        # if qx or qy are zero. Because F is smooth near the origin, we will obtain
        # the correct limiting value by using a small, but non-zero, value for qx/qy

        self.thresh_near_zero(qx)
        self.thresh_near_zero(qy)
        self.thresh_near_zero(qz)

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



    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = np.tan(np.radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )

        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += 4 * F*F.conjugate() * dS

        return P




# HollowOctahedronNanoObject
###################################################################
class HollowOctahedronNanoObject(NanoObject):
    """ Hollow octahdedron

        params:
            inner_radius:
            outer_radius:


    """

    def __init__(self, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'height': None, \
                            'pyramid_face_angle': None, \
        }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0

        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )

        pargs_inner = pargs.copy()
        pargs_inner['radius'] = pargs['inner_radius']
        pargs_inner['height'] = None
        pargs_inner['pyramid_face_angle'] = None
        self.inner_octa = OctahedronNanoObject(pargs_inner)

        pargs_outer = pargs.copy()
        pargs_outer['radius'] = pargs['outer_radius']
        pargs_outer['height'] = None
        pargs_outer['pyramid_face_angle'] = None
        self.outer_octa= OctahedronNanoObject(pargs_outer)

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}



    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""

        in_x = np.array(in_x)
        in_y = np.array(in_y)
        in_z = np.array(in_z)

        # first rotate
        in_x, in_y, in_z = self.map_rcoord(np.array([in_x, in_y, in_z]))

        V_inner = self.inner_octa.V(in_x, in_y, in_z)
        V_outer = self.outer_octa.V(in_x, in_y, in_z)

        return V_outer - V_inner


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        qx, qy, qz = self.rotate_coord(qx, qy, qz)
        F_inner = self.inner_octa.form_factor(qx, qy, qz)
        F_outer = self.outer_octa.form_factor(qx, qy, qz)

        return F_outer - F_inner


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        Rin = self.pargs['inner_radius']
        Rout = self.pargs['outer_radius']
        Hout = self.outer_octa.pargs['height']
        Hin = self.inner_octa.pargs['height']
        tan_alpha = np.tan(np.radians(self.inner_octa.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volumeout = (4.0/3.0)*tan_alpha*( Rout**3 - (Rout - Hout/tan_alpha)**3 )
        volumein = (4.0/3.0)*tan_alpha*( Rin**3 - (Rin - Hin/tan_alpha)**3 )
        volume = volumeout - volumein

        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += 4 * F*F.conjugate() * dS

        return P



# OctahedronNanoObject
###################################################################
class OctahedronNanoObject(PyramidNanoObject):
    """An octahedral nano-object. The canonical (unrotated) version
    has the square cross-section in the x-y plane, with corners pointing along +z and -z.
    The square's edges are parallel to the x-axis and y-axis (i.e. the corners
    point at 45 degrees to axes.
    """


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""
        Vu = super(OctahedronNanoObject, self).V(*self.map_rcoord(np.array([in_x, in_y, in_z])))
        Vd = super(OctahedronNanoObject, self).V(*self.map_rcoord(np.array([in_x, in_y, -in_z])))

        return Vu+Vd




    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        Fu = super(OctahedronNanoObject, self).form_factor( qx, qy, qz )
        Fd = super(OctahedronNanoObject, self).form_factor( qx, qy, -qz )

        return Fu + Fd


    def form_factor_array(self, qx, qy, qz):

        Fu = super(OctahedronNanoObject, self).form_factor_array( qx, qy, qz )
        Fd = super(OctahedronNanoObject, self).form_factor_array( qx, qy, -qz )

        return Fu + Fd


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""


        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                Fu = super(OctahedronNanoObject, self).form_factor( qx, qy, qz )
                Fd = super(OctahedronNanoObject, self).form_factor( qx, qy, -qz )
                F = Fu + Fd

                P += 4 * F*F.conjugate() * dS

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = np.tan(np.radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = 2*(4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )


        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )  # Integral from +z-axis to -z-axis


        P = np.zeros( len(q_list) )

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q_list*cos(theta)
                theta_part = dtheta*dphi*sin(theta)

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    qx = -qy_partial*cos(phi)
                    qy = qy_partial*sin(phi)

                    Fu = super(OctahedronNanoObject, self).form_factor_array( qx, qy, qz )
                    Fd = super(OctahedronNanoObject, self).form_factor_array( qx, qy, -qz )
                    F = Fu + Fd
                    if np.any(np.isnan(F)):
                        raise ValueError

                    P += 4 * (F*F.conjugate()).real * theta_part


        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi

        return P



# OctahedronPolydisperseNanoObject
###################################################################
class OctahedronPolydisperseNanoObject(PolydisperseNanoObject):

    def __init__(self, pargs={}, seed=None):

        PolydisperseNanoObject.__init__(self, OctahedronNanoObject, pargs=pargs, seed=seed)



# CylinderNanoObject
###################################################################
class CylinderNanoObject(NanoObject):
    """A cylinder nano-object. The canonical (unrotated) version
    has the circular-base in the x-y plane, with the length along z.

    self.pargs contains parameters:
        rho_ambient : the cylinder density
        rho1 : the solvent density (I think, JL sept 2016)
        radius : (default 1.0) the cylinder radius
        length : (default 1.0) the cylinder length

        eta,phi,eta: Euler angles
        x0, y0, z0 : the position of cylinder COM relative to origin
        The object is rotated first about origin, then translated to
            where x0, y0, and z0 define it to be.

    these are calculated after the fact:
        delta_rho : rho_ambient - rho1
    """

    def __init__(self, pargs={}, seed=None):
        #self.rotation_matrix = np.identity(3)
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        if self.pargs['radius']==None or self.pargs['height']==None:
            # user did not specify radius or height should be an error
            print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            x0 = 0.
            self.pargs['x0'] = x0
        if 'y0' not in self.pargs:
            y0 = 0.
            self.pargs['y0'] = y0
        if 'z0' not in self.pargs:
            z0 = 0.
            self.pargs['z0'] = z0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        Returns 1 if in the space, 0 otherwise.
        Can be arrays.
        Rotate then translate.

            rotation_matrix is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)
        """

        in_x = np.array(in_x)
        in_y = np.array(in_y)
        in_z = np.array(in_z)


        R = self.pargs['radius']
        L = self.pargs['height']

        in_x, in_y, in_z = self.map_rcoord(np.array([in_x, in_y, in_z]))

        r = np.hypot(in_x, in_y)

        # it's an array
        result = np.zeros(in_x.shape)
        w = np.where((in_z <= L/2.)*(in_z >= -L/2.)*(np.abs(r) <= R))
        if len(w[0]) > 0:
            result[w] = 1

        return result


    def thresh_near_zero(self, values, threshold=1e-7):
        # Catch values that are exactly zero
        # test for values being a value or array
        if values.ndim > 0:
            idx = np.where( values==0.0 )
            if len(idx[0]) > 0:
                values[idx] = +threshold

            idx = np.where( abs(values)<threshold )
            if len(idx[0]) > 0:
                values[idx] = np.sign(values[idx])*threshold
        else:
            # if not array, return rather than modify variable
            if values ==0.0:
                values = values + threshold
            elif np.abs(values) < threshold:
                values = np.sign(values)*threshold
            return values



    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            The first set of q are assumed to be that of the root qx,qy,qz
            The next sets are the parent coordinates. These are only supplied if the parent is not the root
            The root q is only necessary for the form factor, where the phase factor needs to be calculated
                with respect to the origin of that coordinate system.
        """
        # IMPORTANT : phase must be retrieved *before* mapping in q
        # next a translation is a phase shift
        qx = np.array(qx)
        qy = np.array(qy)
        qz = np.array(qz)
        if qx.ndim==0:
            qx = np.array([qx])
            qy = np.array([qy])
            qz = np.array([qz])

        phase = self.get_phase(qx,qy,qz)

        # first rotate just as for V
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))


        #F = self.form_factor_numerical(qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None)

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H


        # NOTE: (partial kludge) The computation below will hit a divide-by-zero
        # if qx or qy are zero. Because F is smooth near the origin, we will obtain
        # the correct limiting value by using a small, but non-zero, value for qx/qy
        #if qx==0 and qy==0 and qz==0:
            # F(0,0,0) = rho*V
            #return self.pargs['delta_rho']*volume

        # works on arrays, no need to return result
        self.thresh_near_zero(qx)
        self.thresh_near_zero(qy)
        self.thresh_near_zero(qz)

        qr = np.hypot(qx, qy)

        F = 2*j0(qz*H)*j1(qr*R)/qr/R + 1j*0
        F *= phase
        F *= self.pargs['delta_rho']*volume

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption of
        random orientation of the cylinder. In other words, we average over
        every possible orientation. This value is denoted by P(q)
        P(q=0) = 4*pi*volume^2*(delta rho)^2
            Note there is a 4pi factor extra compared to the non isotropic version
        """
        q = np.array(q)
        if q.ndim == 0:
            q = np.array([q])

        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += np.abs(F)**2 * dS

        return P


# EllipsoidNanoObject
###################################################################
class EllipsoidNanoObject(NanoObject):
    """A an oblate spheroid nano-object.
    Follows:
        x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
        for the canonical (unrotated version)

    If (a = b) < c then it's an oblate spheroid,
    If (a = b) > c then it's a prolate spheroid
    If a = b = c, then it's a sphere.

    self.pargs contains parameters:
        rho_ambient : the cylinder density
        rho1 : the solvent density (I think, JL sept 2016)
        a : (default 1.0) the spheroid radius along the minor axes
        b : (default 1.0) the cylinder length
        c : (default 1.0) the cylinder length

        eta,phi,eta: Euler angles
        x0, y0, z0 : the position of cylinder COM relative to origin
        The object is rotated first about origin, then translated to
            where x0, y0, and z0 define it to be.

    these are calculated after the fact:
        delta_rho : rho_ambient - rho1
    """

    def __init__(self, pargs={}, seed=None):
        #self.rotation_matrix = np.identity(3)
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None:
            # user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            x0 = 0.
            self.pargs['x0'] = x0
        if 'y0' not in self.pargs:
            y0 = 0.
            self.pargs['y0'] = y0
        if 'z0' not in self.pargs:
            z0 = 0.
            self.pargs['z0'] = z0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        Returns 1 if in the space, 0 otherwise.
        Can be arrays.
        Rotate then translate.

            rotation_matrix is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)
        """

        in_x = np.array(in_x)
        in_y = np.array(in_y)
        in_z = np.array(in_z)


        a = self.pargs['a']
        b = self.pargs['b']
        c = self.pargs['c']

        in_x, in_y, in_z = self.map_rcoord(np.array([in_x, in_y, in_z]))

        in_x /= a
        in_y /= b
        in_z /= c

        # it's an array
        result = np.zeros(in_x.shape)
        w = np.where(in_x**2 + in_y**2 + in_z**2 <= 1)
        if len(w[0]) > 0:
            result[w] = 1

        return result


    def thresh_near_zero(self, values, threshold=1e-7):
        # Catch values that are exactly zero
        # test for values being a value or array
        if values.ndim > 0:
            idx = np.where( values==0.0 )
            if len(idx[0]) > 0:
                values[idx] = +threshold

            idx = np.where( abs(values)<threshold )
            if len(idx[0]) > 0:
                values[idx] = np.sign(values[idx])*threshold
        else:
            # if not array, return rather than modify variable
            if values ==0.0:
                values = values + threshold
            elif np.abs(values) < threshold:
                values = np.sign(values)*threshold
            return values



    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            The first set of q are assumed to be that of the root qx,qy,qz
            The next sets are the parent coordinates. These are only supplied if the parent is not the root
            The root q is only necessary for the form factor, where the phase factor needs to be calculated
                with respect to the origin of that coordinate system.
        """
        # IMPORTANT : phase must be retrieved *before* mapping in q
        # next a translation is a phase shift
        phase = self.get_phase(qx,qy,qz)

        # first rotate just as for V
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))


        #F = self.form_factor_numerical(qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None)

        a = self.pargs['a']
        b = self.pargs['b']
        c = self.pargs['c']

        qy *= b/a
        qz *= c/a
        R = a


        # NOTE: (partial kludge) The computation below will hit a divide-by-zero
        # if qx or qy are zero. Because F is smooth near the origin, we will obtain
        # the correct limiting value by using a small, but non-zero, value for qx/qy
        #if qx==0 and qy==0 and qz==0:
            # F(0,0,0) = rho*V
            #return self.pargs['delta_rho']*volume

        # works on arrays, no need to return result
        self.thresh_near_zero(qx)
        self.thresh_near_zero(qy)
        self.thresh_near_zero(qz)

        volume = 4/3.*np.pi*a*b*c

        qR = R*np.sqrt( qx**2 + qy**2 + qz**2 )

        #for i, qxi in enumerate(qx):
            #F[i] = self.form_factor(qx[i], qy[i], qz[i])

        F = 3.0*self.pargs['delta_rho']*volume*( np.sin(qR) - qR*np.cos(qR) )/( qR**3 )

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption of
        random orientation of the cylinder. In other words, we average over
        every possible orientation. This value is denoted by P(q)"""

        a = self.pargs['a']
        b = self.pargs['b']
        c = self.pargs['c']

        volume = 4/3.*np.pi*a*b*c

        '''
        to sample orientations properly, we need to use random variables u,v
            uniform in the range (0,1) and set theta and phi to be:
            theta = 2*np.pi*u
            phi = np.arccos(2*nu-1)

            I will change: theta = 2*np.pi*(u-.5)
            for a restricted set, set theta to restricted range
            and nu should be chosen such that cos(phimax) = 2*nu-1
            nu = .5*(cos(phimax) + 1)

            We don't need this here since this code takes dS into account
                but could be useful in the future
        '''
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += F*F.conjugate() * dS

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True ) # Integral from +z-axis to -z-axis


        P = np.zeros( len(q_list) )

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q_list*cos(theta)
                theta_part = dtheta*dphi*sin(theta)

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    qx = -qy_partial*cos(phi)
                    qy = qy_partial*sin(phi)

                    F = self.form_factor_array( qx, qy, qz )

                    # Factor of 8 accounts for the fact that we only integrate
                    # one of the eight octants
                    P += 4 * F*F.conjugate() * theta_part


        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi


        return P


# CylindicalShellsNanoObject
###################################################################
class CylindricalShellsNanoObject(NanoObject):
    """cylindrical shells. The canonical (unrotated) version
    has the circular-base in the x-y plane, with the length along z.

    self.pargs contains parameters:
        rho_ambient : the cylinder density
        rho1 : the solvent density (I think, JL sept 2016)
        inner_radius : (default 1.0) the cylinder inner radius
        outer_radius : (default 1.0) the cylinder outer radius
        length : (default 1.0) the cylinder length

        eta,phi,eta: Euler angles
        x0, y0, z0 : the position of cylinder COM relative to origin
        The object is rotated first about origin, then translated to
            where x0, y0, and z0 define it to be.

    these are calculated after the fact:
        delta_rho : rho_ambient - rho1
    """

    def __init__(self, pargs={}, seed=None):
        #self.rotation_matrix = np.identity(3)
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None:
            # user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'outer_radius' not in self.pargs:
            self.pargs['outer_radius'] = 1.0
        if 'inner_radius' not in self.pargs:
            self.pargs['inner_radius'] = 0.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            x0 = 0.
            self.pargs['x0'] = x0
        if 'y0' not in self.pargs:
            y0 = 0.
            self.pargs['y0'] = y0
        if 'z0' not in self.pargs:
            z0 = 0.
            self.pargs['z0'] = z0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        Returns 1 if in the space, 0 otherwise.
        Can be arrays.
        Rotate then translate.

            rotation_matrix is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)
        """

        in_x = np.array(in_x)
        in_y = np.array(in_y)
        in_z = np.array(in_z)


        Rin = self.pargs['inner_radius']
        Rout = self.pargs['outer_radius']
        L = self.pargs['height']

        in_x, in_y, in_z = self.map_rcoord(np.array([in_x, in_y, in_z]))

        r = np.hypot(in_x, in_y)

        # it's an array
        result = np.zeros(in_x.shape)
        w = np.where((in_z <= L/2.)*(in_z >= -L/2.)*(np.abs(r) <= Rout)*(np.abs(r) >= Rin))
        if len(w[0]) > 0:
            result[w] = 1

        return result


    def thresh_near_zero(self, values, threshold=1e-7):
        # Catch values that are exactly zero
        # test for values being a value or array
        if values.ndim > 0:
            idx = np.where( values==0.0 )
            if len(idx[0]) > 0:
                values[idx] = +threshold

            idx = np.where( abs(values)<threshold )
            if len(idx[0]) > 0:
                values[idx] = np.sign(values[idx])*threshold
        else:
            # if not array, return rather than modify variable
            if values ==0.0:
                values = values + threshold
            elif np.abs(values) < threshold:
                values = np.sign(values)*threshold
            return values



    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates.
            The first set of q are assumed to be that of the root qx,qy,qz
            The next sets are the parent coordinates. These are only supplied if the parent is not the root
            The root q is only necessary for the form factor, where the phase factor needs to be calculated
                with respect to the origin of that coordinate system.
        """
        # IMPORTANT : phase must be retrieved *before* mapping in q
        # next a translation is a phase shift
        phase = self.get_phase(qx,qy,qz)

        # first rotate just as for V
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))


        #F = self.form_factor_numerical(qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None)

        Rin = self.pargs['inner_radius']
        Rout = self.pargs['outer_radius']
        H = self.pargs['height']
        volume = np.pi*(Rout**2-Rin**2)*H


        # NOTE: (partial kludge) The computation below will hit a divide-by-zero
        # if qx or qy are zero. Because F is smooth near the origin, we will obtain
        # the correct limiting value by using a small, but non-zero, value for qx/qy
        #if qx==0 and qy==0 and qz==0:
            # F(0,0,0) = rho*V
            #return self.pargs['delta_rho']*volume

        # works on arrays, no need to return result
        self.thresh_near_zero_array(qx)
        self.thresh_near_zero_array(qy)
        self.thresh_near_zero_array(qz)

        qr = np.hypot(qx, qy)

        Fz = 2*j0(qz*H)
        if Rin > Rout:
            raise ValueError("Rin is greater than Rout")
        #Fin = j1(qr*Rin)/qr/Rin + 1j*0
        #Fout = j1(qr*Rout)/qr/Rout + 1j*0
        if Rin == 0:
            F = j1(qr*Rout)/qr/Rout + 1j*0
        else:
            # need to multiply volume as well (area here)
            F = (j1(qr*Rout)*Rin*np.pi*Rout**2 - j1(qr*Rin)*Rout*np.pi*Rin**2)/qr/Rin/Rout + 1j*0
        F *= Fz
        F *= phase
        F *= self.pargs['delta_rho']*H

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption of
        random orientation of the cylinder. In other words, we average over
        every possible orientation. This value is denoted by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        '''
        to sample orientations properly, we need to use random variables u,v
            uniform in the range (0,1) and set theta and phi to be:
            theta = 2*np.pi*u
            phi = np.arccos(2*nu-1)

            I will change: theta = 2*np.pi*(u-.5)
            for a restricted set, set theta to restricted range
            and nu should be chosen such that cos(phimax) = 2*nu-1
            nu = .5*(cos(phimax) + 1)

            We don't need this here since this code takes dS into account
                but could be useful in the future
        '''
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += F*F.conjugate() * dS

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True ) # Integral from +z-axis to -z-axis


        P = np.zeros( len(q_list) )

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q_list*cos(theta)
                theta_part = dtheta*dphi*sin(theta)

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    qx = -qy_partial*cos(phi)
                    qy = qy_partial*sin(phi)

                    F = self.form_factor_array( qx, qy, qz )

                    # Factor of 8 accounts for the fact that we only integrate
                    # one of the eight octants
                    P += 4 * F*F.conjugate() * theta_part


        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi


        return P


# SixCylinderNanoObject
###################################################################
class SixCylinderNanoObject(NanoObject):
    """A six cylinder nano-object. The canonical (unrotated) version
    has the circular-base in the x-y plane, with the length along z.

    self.pargs contains parameters:
        rho_ambient : the cylinder density
        rho1 : the solvent density (I think, JL sept 2016)
        radius : (default 1.0) the cylinder radius, there are 6 of these
            packed.
        length : (default 1.0) the cylinder length

        eta,phi,eta: Euler angles
        x0, y0, z0 : the position of cylinder COM relative to origin
        The object is rotated first about origin, then translated to
            where x0, y0, and z0 define it to be.

    these are calculated after the fact:
        delta_rho : rho_ambient - rho1
    """

    def __init__(self, baseObject=None, pargs={}, seed=None):
        ''' baseObject : the base object

        '''
        if baseObject is None:
            baseObject = CylinderNanoObject

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                            'packradius': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None or \
           #self.pargs['packradius'] == None:
            ## user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")

        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'packradius' not in self.pargs:
            self.pargs['packradius'] = 2*1.0/(2*np.pi/6.)
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            x0 = 0.
            self.pargs['x0'] = x0
        if 'y0' not in self.pargs:
            y0 = 0.
            self.pargs['y0'] = y0
        if 'z0' not in self.pargs:
            z0 = 0.
            self.pargs['z0'] = z0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        #radius = pargs['radius']
        packradius = pargs['packradius']

        poslist = list()
        dphi = 2*np.pi/6.
        for i in range(6):
            poslist.append([0, 0, 0, packradius*np.cos(i*dphi), packradius*np.sin(i*dphi), 0])
        poslist = np.array(poslist)

        self.cylinderobjects = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0 = pos

            cyl = baseObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        Returns 1 if in the space, 0 otherwise.
        Can be arrays.
        Rotate then translate.

            rotation_matrix is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)
        """
        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def thresh_near_zero(self, values, threshold=1e-7):
        # Catch values that are exactly zero
        # test for values being a value or array
        values = np.array(values)
        if values.ndim > 0:
            idx = np.where( values==0.0 )
            if len(idx[0]) > 0:
                values[idx] = +threshold

            idx = np.where( abs(values)<threshold )
            if len(idx[0]) > 0:
                values[idx] = np.sign(values[idx])*threshold
        else:
            # if not array, return rather than modify variable
            if values ==0.0:
                values = values + threshold
            elif np.abs(values) < threshold:
                values = np.sign(values)*threshold
            return values

    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        # mapping is with respect to the parent coordinates
        phase = self.get_phase(qx,qy,qz)
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx, qy, qz)

        F *= phase

        return F



    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption of
        random orientation of the cylinder. In other words, we average over
        every possible orientation. This value is denoted by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        '''
        to sample orientations properly, we need to use random variables u,v
            uniform in the range (0,1) and set theta and phi to be:
            theta = 2*np.pi*u
            phi = np.arccos(2*nu-1)

            I will change: theta = 2*np.pi*(u-.5)
            for a restricted set, set theta to restricted range
            and nu should be chosen such that cos(phimax) = 2*nu-1
            nu = .5*(cos(phimax) + 1)

            We don't need this here since this code takes dS into account
                but could be useful in the future
        '''
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += F*F.conjugate() * dS

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True ) # Integral from +z-axis to -z-axis


        P = np.zeros( len(q_list) )

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q_list*cos(theta)
                theta_part = dtheta*dphi*sin(theta)

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    qx = -qy_partial*cos(phi)
                    qy = qy_partial*sin(phi)

                    F = self.form_factor_array( qx, qy, qz )

                    # Factor of 8 accounts for the fact that we only integrate
                    # one of the eight octants
                    P += 4 * F*F.conjugate() * theta_part


        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi


        return P


# SixCylindricalShellsNanoObject
###################################################################
class SixCylindricalShellsNanoObject(NanoObject):
    """A six cylinder nano-object. The canonical (unrotated) version
    has the circular-base in the x-y plane, with the length along z.

    self.pargs contains parameters:
        rho_ambient : the cylinder density
        rho1 : the solvent density (I think, JL sept 2016)
        radius : (default 1.0) the cylinder radius, there are 6 of these
            packed.
        length : (default 1.0) the cylinder length

        eta,phi,eta: Euler angles
        x0, y0, z0 : the position of cylinder COM relative to origin
        The object is rotated first about origin, then translated to
            where x0, y0, and z0 define it to be.

    these are calculated after the fact:
        delta_rho : rho_ambient - rho1
    """

    def __init__(self, pargs={}, seed=None):
        ''' baseObject : the base object

        '''

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                            'packradius': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None or \
           #self.pargs['packradius'] == None:
            ## user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")

        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'packradius' not in self.pargs:
            self.pargs['packradius'] = 2*1.0/(2*np.pi/6.)
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            x0 = 0.
            self.pargs['x0'] = x0
        if 'y0' not in self.pargs:
            y0 = 0.
            self.pargs['y0'] = y0
        if 'z0' not in self.pargs:
            z0 = 0.
            self.pargs['z0'] = z0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        #radius = pargs['radius']
        packradius = pargs['packradius']

        poslist = list()
        dphi = 2*np.pi/6.
        for i in range(6):
            poslist.append([0, 0, 0, packradius*np.cos(i*dphi), packradius*np.sin(i*dphi), 0])
        poslist = np.array(poslist)

        self.cylinderobjects = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0 = pos

            cyl = CylindricalShellsNanoObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)

        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
        Returns 1 if in the space, 0 otherwise.
        Can be arrays.
        Rotate then translate.

            rotation_matrix is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)
        """
        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def thresh_near_zero(self, values, threshold=1e-7):
        # Catch values that are exactly zero
        # test for values being a value or array
        values = np.array(values)
        if values.ndim > 0:
            idx = np.where( values==0.0 )
            if len(idx[0]) > 0:
                values[idx] = +threshold

            idx = np.where( abs(values)<threshold )
            if len(idx[0]) > 0:
                values[idx] = np.sign(values[idx])*threshold
        else:
            # if not array, return rather than modify variable
            if values ==0.0:
                values = values + threshold
            elif np.abs(values) < threshold:
                values = np.sign(values)*threshold
            return values

    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        # mapping is with respect to the parent coordinates
        phase = self.get_phase(qx,qy,qz)
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx, qy, qz)

        F *= phase

        return F



    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption of
        random orientation of the cylinder. In other words, we average over
        every possible orientation. This value is denoted by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        '''
        to sample orientations properly, we need to use random variables u,v
            uniform in the range (0,1) and set theta and phi to be:
            theta = 2*np.pi*u
            phi = np.arccos(2*nu-1)

            I will change: theta = 2*np.pi*(u-.5)
            for a restricted set, set theta to restricted range
            and nu should be chosen such that cos(phimax) = 2*nu-1
            nu = .5*(cos(phimax) + 1)

            We don't need this here since this code takes dS into account
                but could be useful in the future
        '''
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )


        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor( qx, qy, qz )

                P += F*F.conjugate() * dS

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        volume = np.pi*R**2*H

        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True ) # Integral from +z-axis to -z-axis


        P = np.zeros( len(q_list) )

        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:

                qz =  q_list*cos(theta)
                theta_part = dtheta*dphi*sin(theta)

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    qx = -qy_partial*cos(phi)
                    qy = qy_partial*sin(phi)

                    F = self.form_factor_array( qx, qy, qz )

                    # Factor of 8 accounts for the fact that we only integrate
                    # one of the eight octants
                    P += 4 * F*F.conjugate() * theta_part


        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*np.pi


        return P

# OctahedronCylindersNanoObject
###################################################################
class OctahedronCylindersNanoObject(NanoObject):
    """An octahedral cylinders nano-object. The canonical (unrotated) version
    has the square cross-section in the x-y plane, with corners pointing along +z and -z.
    The corners are on the x-axis and y-axis. The edges are 45 degrees to the
        x and y axes.

        Some notes: I had a few options here. One is I could have successively
            built objects upon objects:
                - first take two cylinders together, make one piece
                - rotate these pairs around z to make top pyramid
                - flip and sum both to make pyramid
            however, each level of the object requires a recursive set of rotations
            It is best to just brute force define all the terms in one shot,
                which I chose to do here.
            notes about rotations:
                - i need to dot the rotation matrix of the ocahedron with each individula
                        cylinder's rotationo element
            edgelength : length of edge
            edgespread : how much to expand the rods by (not a good name)
                positive is expansion, negative is compression
            edgesep : separation of element from edge
    """
    def __init__(self, baseObject=None, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                            'edgelength': None, \
                            'edgesep': None, \
                    }

        if baseObject is None:
            baseObject = CylinderNanoObject

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None:
            # user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'edgeshift' not in self.pargs:
            self.pargs['edgeshift'] = 1.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'edgelength' not in self.pargs:
            self.pargs['edgelength'] = 1.0
        if 'edgespread' not in self.pargs:
            self.pargs['edgespread'] = 0.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'delta_theta' not in self.pargs:
            self.pargs['delta_theta'] = 0.0
        if 'num_theta' not in self.pargs:
            self.pargs['num_theta'] = 50
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0

        # these are slight shifts per cyl along the axis
        # positive is away from COM and negative towards
        shiftlabels = [
            # these correspond to the poslist
            'CYZ1', 'CXZ1', 'CYZ2', 'CXZ2',
            'CXY1', 'CXY4', 'CXY3', 'CXY2',
            'CYZ3', 'CXZ3', 'CYZ4', 'CXZ4',
        ]

        # you flip x or y from original shifts to move along edge axis
        # not a good explanation but some sort of personal bookkeeping for now...
        shiftfacs = np.array([
            # top
            [0,-1,1],
            [-1,0,1],
            [0,1,1],
            [1, 0,1],
            # middle
            [-1,1,0],
            [-1,-1,0],
            [1,-1,0],
            [1,1,0],
            # bottom
            [0,1,-1],
            [1, 0, -1],
            [0,-1, -1],
            [-1,0,-1]
        ])

        for lbl1 in shiftlabels:
            if lbl1 not in self.pargs:
                self.pargs[lbl1] = 0.


        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        fac1 = np.sqrt(2)/2.*((.5*self.pargs['edgelength']) + self.pargs['edgespread'])


        poslist = np.array([
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
        [0, 45, -90, 0, -fac1,-fac1],
        [0, 45, 0, -fac1, 0, -fac1],
        [0, 45, 90, 0, fac1, -fac1],
        [0, -45, 0, fac1, 0, -fac1],
        ])

        # now add the shift factors
        for i in range(len(poslist)):
            poslist[i, 3:] += np.sqrt(2)/2.*shiftfacs[i]*self.pargs[shiftlabels[i]]


        self.cylinderobjects = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0 = pos

            cyl = baseObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)


        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
            rotation_elements is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)

        """
        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        # phase must be calculated with respect to parent coordinate
        phase = self.get_phase(qx,qy,qz)

        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx, qy, qz)

        F *= phase

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""


        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        P = 0.0
        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor(qx, qy, qz)

                P += np.abs(F)**2 * dS


        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        return self.form_factor_intensity_isotropic(q_list, num_phi=num_phi, num_theta=num_theta)

class SpinyOctahedronCylindersNanoObject(NanoObject):
    """An octahedral cylinders nano-object with protruding cylinders. The
    canonical (unrotated) version has the square cross-section in the x-y
    plane, with corners pointing along +z and -z.  The corners are on the
    x-axis and y-axis. The edges are 45 degrees to the x and y axes.

        Some notes: I had a few options here. One is I could have successively
            built objects upon objects:
                - first take two cylinders together, make one piece
                - rotate these pairs around z to make top pyramid
                - flip and sum both to make pyramid
            however, each level of the object requires a recursive set of rotations
            It is best to just brute force define all the terms in one shot,
                which I chose to do here.
            notes about rotations:
                - i need to dot the rotation matrix of the ocahedron with each individula
                        cylinder's rotationo element
            edgelength : length of edge
            edgespread : how much to expand the rods by (not a good name)
                positive is expansion, negative is compression
            edgesep : separation of element from edge
    """
    def __init__(self, baseObject=None, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                            'edgelength': None, \
                            'spinelength': None, \
                            'edgesep': None, \
                    }

        if baseObject is None:
            baseObject = CylinderNanoObject

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None:
            # user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'edgeshift' not in self.pargs:
            self.pargs['edgeshift'] = 1.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'edgelength' not in self.pargs:
            self.pargs['edgelength'] = 1.0
        if 'spinelength' not in self.pargs:
            self.pargs['edgelength'] = 1.0
        if 'edgespread' not in self.pargs:
            self.pargs['edgespread'] = 0.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'delta_theta' not in self.pargs:
            self.pargs['delta_theta'] = 0.0
        if 'num_theta' not in self.pargs:
            self.pargs['num_theta'] = 50
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0

        # these are slight shifts per cyl along the axis
        # positive is away from COM and negative towards
        shiftlabels = [
            # these correspond to the poslist
            'CYZ1', 'CXZ1', 'CYZ2', 'CXZ2',
            'CXY1', 'CXY4', 'CXY3', 'CXY2',
            'CYZ3', 'CXZ3', 'CYZ4', 'CXZ4',
            'spinepz', 'spinemz', 'spinepx', 'spinemx',
            'spinepy', 'spinemy',
        ]

        # you flip x or y from original shifts to move along edge axis
        # not a good explanation but some sort of personal bookkeeping for now...
        shiftfacs = np.array([
            # top
            [0,-1,1],
            [-1,0,1],
            [0,1,1],
            [1, 0,1],
            # middle
            [-1,1,0],
            [-1,-1,0],
            [1,-1,0],
            [1,1,0],
            # bottom
            [0,1,-1],
            [1, 0, -1],
            [0,-1, -1],
            [-1,0,-1],
            # spines
            [0,0,1],
            [0,0,-1],
            [1,0,0],
            [-1,0,0],
            [0,1,0],
            [0,-1,0],
        ])

        for lbl1 in shiftlabels:
            if lbl1 not in self.pargs:
                self.pargs[lbl1] = 0.


        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        fac1 = np.sqrt(2)/2.*((.5*self.pargs['edgelength']) + self.pargs['edgespread'])
        sL = self.pargs['spinelength']
        eL = self.pargs['edgelength']

        poslist = np.array([
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
        [0, 45, -90, 0, -fac1,-fac1],
        [0, 45, 0, -fac1, 0, -fac1],
        [0, 45, 90, 0, fac1, -fac1],
        [0, -45, 0, fac1, 0, -fac1],
        # now the spines
        [0, 0, 0, 0, 0, eL/np.sqrt(2) + sL/2.],
        [0, 0, 0, 0, 0, -eL/np.sqrt(2) - sL/2.],
        [0, 90, 0, eL/np.sqrt(2) + sL/2.,0,0],
        [0, 90, 0, -eL/np.sqrt(2) - sL/2.,0,0],
        [0, 90, 90, 0, eL/np.sqrt(2) + sL/2., 0],
        [0, 90, 90, 0, -eL/np.sqrt(2) - sL/2., 0],
        ])

        # now add the shift factors
        for i in range(len(poslist)):
            poslist[i, 3:] += np.sqrt(2)/2.*shiftfacs[i]*self.pargs[shiftlabels[i]]


        self.cylinderobjects = list()
        for i, pos  in enumerate(poslist):
            eta, phi, theta, x0, y0, z0 = pos
            pargsnew = pargs.copy()

            if 'spine' in shiftlabels[i]:
                pargsnew['height'] = sL
                if np.abs(sL) < 1e-8:
                    continue
            cyl = baseObject(pargs=pargsnew)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)


        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
            rotation_elements is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)

        """
        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        # phase must be calculated with respect to parent coordinate
        phase = self.get_phase(qx,qy,qz)

        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx, qy, qz)

        F *= phase

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""


        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        P = 0.0
        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor(qx, qy, qz)

                P += np.abs(F)**2 * dS


        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        return self.form_factor_intensity_isotropic(q_list, num_phi=num_phi, num_theta=num_theta)
# CylinderCrossNanoObject
###################################################################
class CylinderCrossNanoObject(NanoObject):
    """An cylinder 3D cross, 6 cylinders make this up. The canonical (unrotated) version
        is aligned along each of the axes.
        Note: it's possible to have cylinders not touch by simply varying their heigh with
            respect to the lattice constant. They will be assumed to be centered in a symmetric fashion.

        Some notes: I had a few options here. One is I could have successively
            built objects upon objects:
                - first take two cylinders together, make one piece
                - rotate these pairs around z to make top pyramid
                - flip and sum both to make pyramid
            however, each level of the object requires a recursive set of rotations
            It is best to just brute force define all the terms in one shot,
                which I chose to do here.
            notes about rotations:
                - i need to dot the rotation matrix of the ocahedron with each individula
                        cylinder's rotationo element
            edgelength : length of edge
            edgesep : separation of element from edge
    """
    def __init__(self, baseObject=None, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                            'edgelength': None, \
                            'edgesep': None, \
                    }

        if baseObject is None:
            baseObject = CylinderNanoObject

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        #if self.pargs['radius']==None or self.pargs['height']==None:
            # user did not specify radius or height should be an error
            #print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'edgelength' not in self.pargs:
            self.pargs['edgelength'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'delta_theta' not in self.pargs:
            self.pargs['delta_theta'] = 0.0
        if 'num_theta' not in self.pargs:
            self.pargs['num_theta'] = 50
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        L = self.pargs['edgelength']

        poslist = np.array([
        # z axis top part
        [0, 0, 0, 0, 0, L/2.],
        [0, 0, 0, 0, 0, -L/2.],
        # y axis top part
        [0, 90, 0, L/2., 0, 0],
        [0, 90, 0, -L/2., 0., 0],
        # x axis top part
        [0, 90, 90, 0, L/2., 0],
        [0, 90, 90, 0, -L/2., 0],
        ])


        self.cylinderobjects = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0 = pos

            cyl = baseObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)


        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
            rotation_elements is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)

        """
        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        # phase must be calculated with respect to parent coordinate
        phase = self.get_phase(qx,qy,qz)

        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx, qy, qz)

        F *= phase

        return F


    def _form_factor(self, qxroot, qyroot, qzroot, qxparent=None, qyparent=None,
                    qzparent=None):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)
            The orientational spread is treated as an independent tilt in each
            axis. In other words, we tilt in x, bring back to zero, tilt in y,
            and so on
            If delta_theta is None or less than 1e-6, no orientational spread is attempted.
        """
        if qxparent is None:
            qxparent, qyparent, qzparent = qxroot, qyroot, qzroot
        elif qyparent is None or qzparent is None:
            raise ValueError("Error supplied x but not y or z")

        delta_theta = self.pargs['delta_theta']
        num_theta = self.pargs['num_theta']


        if delta_theta is None:
            delta_theta = 0.

        if np.abs(delta_theta) > 1e-6:
            # do an orientational average
            theta_vals, dtheta = np.linspace(-delta_theta/2., delta_theta/2.,
                                             num_theta, endpoint=False,
                                             retstep=True)

            F = 0.0 + 1j
            dStot = 0.
            numth = self.pargs['num_theta']
            for i, theta in enumerate(theta_vals):
                print("orientational avg, {} of {}".format(i, numth))
                # tilt in z is just eta = dtheta
                rotz = self.rotation_elements(theta, 0., 0.)
                qx, qy, qz = self.map_coord(np.array([qxparent,qyparent,qzparent]),rotz)
                F += self._form_factor(qxroot, qyroot, qzroot, qx, qy, qz)
                dStot += dtheta

                # tilt in y is just theta = dtheta
                roty = self.rotation_elements(0, theta, 0.)
                qx, qy, qz = self.map_coord(np.array([qxparent,qyparent,qzparent]),roty)
                F += self._form_factor(qxroot, qyroot, qzroot, qx, qy, qz)
                dStot += dtheta

                # tilt in z is just eta = 90, theta = dtheta
                rotx = self.rotation_elements(np.pi/2., theta, 0.)
                qx, qy, qz = self.map_coord(np.array([qxparent,qyparent,qzparent]),rotx)
                F += self._form_factor(qxroot, qyroot, qzroot, qx, qy, qz)
                dStot += dtheta


            F /= dStot

        else:
            F = self._form_factor(qxroot, qyroot, qzroot, qxparent, qyparent,
                                  qzparent)

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""


        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        P = 0.0
        dthtmp = self.pargs['delta_theta']
        self.pargs['delta_theta'] = 0
        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor(qx, qy, qz)

                P += np.abs(F*F.conjugate() * dS)

        self.pargs['delta_theta'] = dthtmp

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        return self.form_factor_intensity_isotropic(q_list, num_phi=num_phi, num_theta=num_theta)


# OctahedronSixCylindersNanoObject
###################################################################
class OctahedronSixCylindersNanoObject(NanoObject):
    """An octahedral cylinders nano-object. The canonical (unrotated) version
    has the square cross-section in the x-y plane, with corners pointing along +z and -z.
    The corners are on the x-axis and y-axis. The edges are 45 degrees to the
        x and y axes.

        Some notes: I had a few options here. One is I could have successively
            built objects upon objects:
                - first take two cylinders together, make one piece
                - rotate these pairs around z to make top pyramid
                - flip and sum both to make pyramid
            however, each level of the object requires a recursive set of rotations
            It is best to just brute force define all the terms in one shot,
                which I chose to do here.
            notes about rotations:
                - i need to dot the rotation matrix of the ocahedron with each individula
                        cylinder's rotationo element
        edgelength : the length of an edge on the octahedra
        height : the height of the cylinders in the octahedra
        radius : the radius of the cylinders
        packradius : the packing radius of the cylinders. This is where the center of
            the disc cross-section of this sample resides.
    """
    def __init__(self, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'packradius': None, \
                            'radius': None, \
                            'height': None, \
                            'edgelength': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )

        if self.pargs['radius']==None or self.pargs['height']==None or \
           self.pargs['packradius'] == None or self.pargs['edgelength'] == None:
            # user did not specify radius or height should be an error
            print("Error, did not specify radius or height, setting defaults")


        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'packradius' not in self.pargs:
            self.pargs['packradius'] = 2*1.0/(2*np.pi/6.)
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'edgelength' not in self.pargs:
            self.pargs['edgelength'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        #fac1 = np.sqrt(2)/2.*.5*self.pargs['edgelength']
        fac1 = np.sqrt(2)/2.*(.5*self.pargs['edgelength'] + self.pargs['packradius'] + 2*self.pargs['radius'])

        poslist = np.array([
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
        [0, 45, -90, 0, -fac1,-fac1],
        [0, 45, 0, -fac1, 0, -fac1],
        [0, 45, 90, 0, fac1, -fac1],
        [0, -45, 0, fac1, 0, -fac1],
        ])


        self.cylinderobjects = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0 = pos

            cyl = SixCylinderNanoObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)


        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
            rotation_elements is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)

        """



        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift
        phase = self.get_phase(qx,qy,qz)

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx,qy,qz)

        F *= phase

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""


        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor(qx, qy, qz)

                P += F*F.conjugate() * dS

        return P

    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        return self.form_factor_intensity_isotropic(q_list, num_phi=num_phi, num_theta=num_theta)

# OctahedronSpheresNanoObject
###################################################################
class OctahedronSpheresNanoObject(NanoObject):
    """An octahedral cylinders nano-object. The canonical (unrotated) version
    has the square cross-section in the x-y plane, with corners pointing along +z and -z.
    The corners are on the x-axis and y-axis. The edges are 45 degrees to the
        x and y axes.

        Some notes: I had a few options here. One is I could have successively
            built objects upon objects:
                - first take two cylinders together, make one piece
                - rotate these pairs around z to make top pyramid
                - flip and sum both to make pyramid
            however, each level of the object requires a recursive set of rotations
            It is best to just brute force define all the terms in one shot,
                which I chose to do here.
            notes about rotations:
                - i need to dot the rotation matrix of the ocahedron with each individula
                        cylinder's rotationo element
    """
    def __init__(self, pargs={}, seed=None):

        #self.rotation_matrix = np.identity(3)

        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'radius': None, \
                            'height': None, \
                    }

        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )
        if self.pargs['radius']==None or self.pargs['height']==None or \
           self.pargs['edgelength'] == None:
            # user did not specify radius or height should be an error
            print("Error, did not specify radius or height, setting defaults")

        # Set defaults
        if 'radius' not in self.pargs:
            self.pargs['radius'] = 1.0
        if 'height' not in self.pargs:
            self.pargs['height'] = 1.0
        if 'edgelength' not in self.pargs:
            self.pargs['edgelength'] = 1.0
        if 'eta' not in self.pargs:
            self.pargs['eta'] = 0.0
        if 'phi' not in self.pargs:
            self.pargs['phi'] = 0.0
        if 'theta' not in self.pargs:
            self.pargs['theta'] = 0.0
        if 'x0' not in self.pargs:
            self.pargs['x0'] = 0.0
        if 'y0' not in self.pargs:
            self.pargs['y0'] = 0.0
        if 'z0' not in self.pargs:
            self.pargs['z0'] = 0.0

        self.set_angles(eta=self.pargs['eta'], phi=self.pargs['phi'], theta=self.pargs['theta'])
        self.set_origin(x0=self.pargs['x0'], y0=self.pargs['y0'], z0=self.pargs['z0'])

        fac1 = np.sqrt(2)/2.*.5*self.pargs['edgelength']

        poslist = np.array([
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
        [0, 45, -90, 0, -fac1,-fac1],
        [0, 45, 0, -fac1, 0, -fac1],
        [0, 45, 90, 0, fac1, -fac1],
        [0, -45, 0, fac1, 0, -fac1],
        ])


        self.cylinderobjects = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0 = pos

            cyl = SphereNanoObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.cylinderobjects.append(cyl)


        #if 'cache_results' not in self.pargs:
            #self.pargs['cache_results' ] = True
        #self.form_factor_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates.
            rotation_elements is an extra rotation to add on top of the built
            in rotation (from eta, phi, theta elements in object)

        """



        # TODO: This makes assumptions about there being no rotation
        # need to add extra rotation of octahedron
        V = 0.
        for cyl in self.cylinderobjects:
            V = V + cyl.V(*self.map_rcoord(np.array([in_x, in_y, in_z])))

        return V


    def get_phase(self, qx, qy, qz):
        ''' Get the phase factor from the shift'''
        phase = np.exp(1j*qx*self.pargs['x0'])
        phase *= np.exp(1j*qy*self.pargs['y0'])
        phase *= np.exp(1j*qz*self.pargs['z0'])

        return phase

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        phase = self.get_phase(qx,qy,qz)
        qx, qy, qz = self.map_qcoord(np.array([qx, qy, qz]))
        # next a translation is a phase shift

        F = 0 + 1j*0
        for cyl in self.cylinderobjects:
            F = F + cyl.form_factor(qx,qy,qz)

        F *= phase

        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""


        phi_vals, dphi = np.linspace( 0, np.pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        P = 0.0

        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi

            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)

                F = self.form_factor(qx, qy, qz)

                P += F*F.conjugate() * dS

        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        return self.form_factor_intensity_isotropic(q_list, num_phi=num_phi, num_theta=num_theta)



# PeakShape
###################################################################
class PeakShape(object):
    """Defines an x-ray scattering peak. Once initialized, the object will return
    the height of the peak for any given qs position (distance from peak center)."""

    infinity = 1e300



    def __init__(self, nu=0, delta=0.1, gauss_cutoff=200, lorentz_cutoff=0.005, product_terms=None, gamma_method=False):

        self.sigma = None
        self.fwhm = None

        self.gauss_cutoff = gauss_cutoff
        self.lorentz_cutoff = lorentz_cutoff
        self.gamma_method = gamma_method

        self.infinity = max(self.infinity, 1.01*self.gauss_cutoff)

        self.requested_product_terms = product_terms
        self.reshape( nu, delta )


    def reshape(self, nu=0, delta=0.1):
        self.nu = 1.0*nu
        self.delta = 1.0*delta

        if self.requested_product_terms == None:
            # Figure out how many product terms we need, based on the value of nu
            if self.nu<1:
                self.product_terms = 20
            elif self.nu>1000:
                self.product_terms = 2000
            else:
                self.product_terms = int( ( (self.nu-1)/(1000-1) )*(2000-20) + 20 )
        else:
            self.product_terms = self.requested_product_terms

        if self.nu>=self.lorentz_cutoff and self.nu<=self.gauss_cutoff:
            self.gamma_nu = np.sqrt(np.pi)*gamma.Gamma( (self.nu+1)/2 )/gamma.Gamma( self.nu/2 )

        self.already_computed = {}

    def gaussian(self, sigma=None, delta=None, fwhm=None):
        """Sets the peak to be a pure Gaussian (overrides any other setting)."""

        self.nu = self.infinity
        self.already_computed = {}

        if sigma==None and delta==None and fwhm==None:
            print( "WARNING: No width specified for Gaussian peak. A width has been assumed." )
            self.sigma = 0.1
            self.delta = np.sqrt(8/np.pi)*self.sigma
            self.fwhm = 2*np.sqrt(2*np.log(2))*self.sigma
        elif sigma!=None:
            # Sigma takes priority
            self.sigma = sigma
            self.delta = np.sqrt(8/np.pi)*self.sigma
            self.fwhm = 2*np.sqrt(2*np.log(2))*self.sigma
        elif fwhm!=None:
            self.fwhm = fwhm
            self.sigma = self.fwhm/( 2*np.sqrt(2*np.log(2)) )
            self.delta = np.sqrt(8/np.pi)*self.sigma
        else:
            # Use delta to define peak width
            self.delta = delta
            self.sigma = np.sqrt(np.pi/8)*self.delta
            self.fwhm = 2*np.sqrt(2*np.log(2))*self.sigma

    def lorentzian(self, delta=None, fwhm=None):
        """Sets the peak to be a pure Lorentzian (overrides any other setting)."""

        self.nu = 0
        self.already_computed = {}

        if delta==None and fwhm==None:
            print( "WARNING: No width specified for Lorentzian peak. A width has been assumed." )
            self.delta = 0.1
            self.fwhm = self.delta
        elif delta!=None:
            self.delta = delta
            self.fwhm = self.delta
        else:
            self.fwhm = fwhm
            self.delta = self.fwhm

        self.sigma = None


    def val(self, qs):
        """Returns the height of the peak at the given qs position.
        The peak is centered about qs = 0. The shape and width of the peak is based
        on the parameters it was instantiated with."""

        qs = abs(qs)    # Peak is symmetric

        # If we've already computed this position, just use the lookup table
        if len(self.already_computed) > 0 and qs in self.already_computed:
            return self.already_computed[qs]

        if self.nu>self.gauss_cutoff:
            # Gaussian
            val = (2/(np.pi*self.delta))*np.exp( -(4*(qs**2))/(np.pi*(self.delta**2)) )
        elif self.nu<self.lorentz_cutoff:
            # Lorentzian
            val = (self.delta/(2*np.pi))/(qs**2 + ((self.delta/2)**2) )
        else:
            # Brute-force the term
            val = (2/(np.pi*self.delta))

            if self.gamma_method:

                print( "WARNING: The gamma method does not currently work." )

                # Use gamma functions
                y = (4*(qs**2))/( (np.pi**2) * (self.delta**2) )

                # Note that this equivalence comes from the paper:
                #   Scattering Curves of Ordered Mesoscopic Materials
                #   S. Förster, A. Timmann, M. Konrad, C. Schellbach, A. Meyer, S.S. Funari, P. Mulvaney, R. Knott,
                #   J. Phys. Chem. B, 2005, 109 (4), pp 1347–1360 DOI: 10.1021/jp0467494
                #   (See equation 27 and last section of Appendix.)
                # However there seems to be a typo in the paper, since it does not match the brute-force product.

                numerator = gamma.GammaComplex( (self.nu/2) + 1.0j*self.gamma_nu*y )
                #numerator = gamma.GammaComplex( (self.nu/2) + 1.0j*self.gamma_nu*(sqrt(y)) )
                denominator = gamma.GammaComplex( self.nu/2 )
                term = numerator/denominator

                val *= 0.9*term*term.conjugate()

            else:
                # Use a brute-force product calculation
                for n in range(0, self.product_terms):
                    #print n, self.nu, self.gamma_nu
                    term1 = (self.gamma_nu**2)/( (n+self.nu/2)**2 )
                    #print "  " + str(term1)
                    term2 = (4*(qs**2))/( (np.pi**2) * (self.delta**2) )
                    val *= 1/(1+term1*term2)


        #self.already_computed[qs] = val

        return val

    def val_array(self, q_list, q_center):
        """Returns the height of the peak for the given array of positions, under the
        assumption of a peak centered about q_center."""

        val = np.empty( (len(q_list)) )
        for i, q in enumerate(q_list):
            val[i] = self.val(abs(q-q_center))

        return val


    def plot(self, plot_width=1.0, num_points=200, filename='peak.png', ylog=False):

        q_list = np.linspace( -plot_width, plot_width, num_points )
        int_list = self.val_array( q_list, 0.0 )

        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)

        plt.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )

        if ylog:
            plt.semilogy()

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        plt.ylabel( 'Intensity (a.u.)', size=30 )

        plt.savefig( filename )

        return int_list


# background
###################################################################
class background(object):

    def __init__(self, constant=0.0, prefactor=0.0, alpha=-2.0, prefactor2=0.0, alpha2=-1.5):
        self.constant=constant
        self.prefactor=prefactor
        self.alpha=alpha

        self.prefactor2=prefactor2
        self.alpha2=alpha2

    def update(self, constant=0.0, prefactor=0.0, alpha=-2.0, prefactor2=0.0, alpha2=-1.5):
        self.constant=constant
        self.prefactor=prefactor
        self.alpha=alpha

        self.prefactor2=prefactor2
        self.alpha2=alpha2

    def val(self, q):
        return self.prefactor*( q**(self.alpha) ) + self.prefactor2*( q**(self.alpha2) ) + self.constant

    def val_array(self, q_list):
        return self.val(q_list)


# Lattice
###################################################################
class Lattice(object):
    """Defines a lattice type and provides methods for adding objects to the
    lattice. This is the starting point of all the crystalline samples.  It is
    recommended here to set the basis vectors (lattice spacing, and alpha,
    beta, gamma) and then have objects inherit this, such as FCC BCC here.
    Finally, you need to add a list of NanoObjects with form factors for this
    to work.

    sigma_D : the DW factor. Can be one number or a two tuple:
            (unit normal, DW factor) pair
    """

    # Initialization
    ########################################
    def __init__(self, objects, lattice_spacing_a=1.0, lattice_spacing_b=None, lattice_spacing_c=None, alpha=90, beta=None, gamma=None, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 8
        self.object_count(objects)

        self.objects = objects

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b
        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a
        else:
            self.lattice_spacing_c = lattice_spacing_c

        self.alpha = radians(alpha)
        if beta==None:
            self.beta = radians(alpha)
        else:
            self.beta = radians(beta)
        if gamma==None:
            self.gamma = radians(alpha)
        else:
            self.gamma = radians(gamma)


        self.sigma_D = np.array(sigma_D)          # Lattice disorder



        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'triclinic'
        self.symmetry['crystal system'] = 'triclinic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'pedial'
        self.symmetry['point group'] = '1'
        self.symmetry['space group'] = 'P1'

        self.positions = ['---']
        self.lattice_positions = ['front face, top left', \
                                    'front face, top right', \
                                    'front face, bottom left', \
                                    'front face, bottom right', \
                                    'bottom face, top left', \
                                    'bottom face, top right', \
                                    'bottom face, bottom left', \
                                    'bottom face, bottom right', \
                                ]
        self.lattice_coordinates = [ (1.0, 0.0, 1.0), \
                                        (1.0, 1.0, 1.0), \
                                        (1.0, 0.0, 0.0), \
                                        (1.0, 1.0, 0.0), \
                                        (0.0, 0.0, 1.0), \
                                        (0.0, 1.0, 1.0), \
                                        (0.0, 0.0, 0.0), \
                                        (0.0, 1.0, 0.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                ]

    def init_containers(self):

        self.hkl_list = []
        self.hkl_list_1d ={}

        # Caching (for performance)
        self.q_list_cache = []
        self.lattice_cache = []
        self.peak_cache = []
        self.PS_cache = []



    def update_sigma_D(self, sigma_D):
        '''sigma_D : the DW factor. Can be one number or a two tuple:
        (unit normal, DW factor) pair '''
        self.sigma_D = np.array(sigma_D)


    def object_count(self, objects):
        if len(objects)<self.min_objects:
            print( "WARNING: %s received only %d objects in enumeration, but requires at least %d." % (self.__class__.__name__, len(objects), self.min_objects) )
            exit()

    def unit_cell_volume(self):
        V = np.sqrt( 1 - (cos(self.alpha))**2 - (cos(self.beta))**2 - (cos(self.gamma))**2 + 2*cos(self.alpha)*cos(self.beta)*cos(self.gamma) )
        V *= self.lattice_spacing_a*self.lattice_spacing_b*self.lattice_spacing_c

        return V

    def V(self, x, y, z):
        ''' Look at a slice'''
        V = np.zeros_like(x)
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            # trick to get positions right, set object at displaced position
            # calculate the volume then bring back
            x0 = obj.pargs['x0']
            y0 = obj.pargs['y0']
            z0 = obj.pargs['z0']
            obj.pargs['x0'] = x0 + xi*self.lattice_spacing_a
            obj.pargs['y0'] = y0 + yi*self.lattice_spacing_b
            obj.pargs['z0'] = z0 + zi*self.lattice_spacing_c
            V += obj.V(x,y,z)
            obj.pargs['x0'] = x0
            obj.pargs['y0'] = y0
            obj.pargs['z0'] = z0
        return V



    # Components of intensity calculation
    ########################################

    def iterate_over_objects(self):
        """Returns a sequence of the distinct particle/object
        types in the unit cell. It thus defines the unit cell."""

        # r will contain the return value, an array with rows that contain:
        # number, position in unit cell, relative coordinates in unit cell, object
        #r = []

        for i, pos in enumerate(self.lattice_positions):
            xi, yi, zi = self.lattice_coordinates[i]
            obj = self.lattice_objects[i]
            # Julien : this should be changed to yield
            #r.append( [i, pos, xi, yi, zi, obj] )
            yield [i, pos, xi, yi, zi, obj]

        #return r


    def multiplicity_lookup(self, h, k, l):
        """Returns the peak multiplicity for the given reflection."""
        return 1

    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""

        # NOTE: This is assuming cubic/rectangular only!
        qhkl_vector = ( 2*np.pi*h/(self.lattice_spacing_a), \
                        2*np.pi*k/(self.lattice_spacing_b), \
                        2*np.pi*l/(self.lattice_spacing_c) )
        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        qhkl, qhkl_vector = self.q_hkl(h,k,l)

        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return qhkl

    def iterate_over_hkl_compute(self, max_hkl=6):
        """Returns a sequence of hkl lattice peaks (reflections)."""

        # r will contain the return value, an array with rows that contain:
        # h, k, l, peak multiplicity, symmetry factor, qhkl, qhkl_vector
        r = []

        for h in range(-max_hkl,max_hkl+1):
            for k in range(-max_hkl,max_hkl+1):
                for l in range(-max_hkl,max_hkl+1):

                    # Don't put a reflection at origin
                    if not (h==0 and k==0 and l==0):
                        m = self.multiplicity_lookup(h, k, l)
                        f = self.symmetry_factor(h, k, l)

                        if m!=0 and f!=0:
                            qhkl, qhkl_vector = self.q_hkl(h,k,l)
                            r.append( [ h, k, l, m, f, qhkl, qhkl_vector ] )

        return r


    def iterate_over_hkl(self, max_hkl=6):

        if self.hkl_list==[]:
            self.hkl_list = self.iterate_over_hkl_compute(max_hkl=max_hkl)

        return self.hkl_list


    def iterate_over_hkl_1d(self, max_hkl=6, tolerance=1e-10):

        # Build up a local dictionary of peaks
        hkl_list_1d = {}

        for h, k, l, m, f, qhkl, qhkl_vector in self.iterate_over_hkl_compute(max_hkl=max_hkl):

            # Search dictionary, see if this qhkl already exists
            found = False
            for q in sorted(hkl_list_1d.keys()):
                if abs(qhkl-q)<tolerance:
                    found = True
                    found_q = q
            if found:
                # Possible bug: We are assuming that any peak with the same qhkl
                # necessarily has the same f (symmetry factor)

                # Add multiplicity
                h_old, k_old, l_old, m_old, f_old = hkl_list_1d[found_q]
                hkl_list_1d[found_q] = [ h_old, k_old, l_old, m_old+m, f_old ]
            else:
                # Add new peak
                hkl_list_1d[qhkl] = [ abs(h), abs(k), abs(l), m, f ]


        self.hkl_list_1d = hkl_list_1d

        # Each element in dictionary is like:
        # q : [ h, k, l, m, f ]
        return self.hkl_list_1d


    def sum_over_objects(self, qhkl_vector, h, k, l):
        """Returns the sum over particles in the unit cell."""

        summation = 0
        # TODO: doublecheck rotation
        # (rotate qhkl based on the orientation of the particle in the unit cell; right now
        # this is being handled by setting orientation angles to the particular NanoObjects)
        qx, qy, qz = qhkl_vector

        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            e_term = np.exp( 2*np.pi * 1j * ( xi*h + yi*k + zi*l ) )

            summation += e_term*obj.form_factor( qx, qy, qz )
            if np.any(np.isnan(summation)):
                raise ValueError
            #summation += e_term*obj.form_factor_isotropic( np.sqrt(qx**2+qy**2+qz**2) ) # Make the object isotropic
            #summation += e_term*obj.form_factor_orientation_spread( np.sqrt(qx**2+qy**2+qz**2) ) # Spread intensity


        return summation


    def sum_over_hkl(self, q, peak, max_hkl=6):


        summation = 0

        for h, k, l, m, f, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):

            fs = self.sum_over_objects(qhkl_vector, h, k, l)
            term1 = fs*fs.conjugate()
            # if
            #term2 = np.exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
            term2 = self.G_q(qhkl_vector)
            term3 = peak.val( q-qhkl )

            summation += (m*(f**2)) * term1.real * term2 * term3

        return summation

    def sum_over_hkl_array(self, q_list, peak, max_hkl=6):

        summation = np.zeros( (len(q_list)) )
        # save objects in a list first, then grab them, then compute the values
        # faster generally... this is crude but eventually we should be
        # able to improve it
        # TODO : have not finished this yet, keeping old version uncommented
        hkl_info = self.iterate_over_hkl(max_hkl=max_hkl)
        # expand
        #hkl_info = np.array([h,k,l,m,f,qhkl, qvec[0], qvec[1], qvec[2] for
        #                         h,k,l,m,f,qhkl,qvec in hkl_info])
        #hkls = hkl_info[:,0:3]
        #ms = hkl_info[:,3]
        #fs = hkl_info[:,4]
        #qhkls = hkl_info[:,5]
        #qhkl_vecs = hkl_info[:,6:9]


        for h, k, l, m, f, qhkl, qhkl_vector in hkl_info:

            fs = self.sum_over_objects(qhkl_vector, h, k, l)
            term1 = fs*fs.conjugate()
            # Debye Waller factor
            #term2 = np.exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
            term2 = self.G_q(qhkl_vector)

            summation += (m*(f**2)) * term1.real * term2 * peak.val_array( q_list, qhkl )

        return summation



    def sum_over_hkl_array_old(self, q_list, peak, max_hkl=6):
        ''' Deprecated version. Leaving for now just for comparisons to new code.'''


        summation = np.zeros( (len(q_list)) )

        for h, k, l, m, f, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):

            fs = self.sum_over_objects(qhkl_vector, h, k, l)
            term1 = fs*fs.conjugate()
            term2 = np.exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )

            summation += (m*(f**2)) * term1.real * term2 * peak.val_array( q_list, qhkl )

        return summation


    def sum_over_hkl_cache(self, q_list, peak, max_hkl=6):


        hkl_list = self.iterate_over_hkl(max_hkl=max_hkl)

        if self.lattice_cache==[]:
            # Recalculate the lattice part
            self.lattice_cache = np.zeros( (len(hkl_list),len(q_list)) )

            i_hkl = 0
            for h, k, l, m, f, qhkl, qhkl_vector in hkl_list:
                fs = self.sum_over_objects(qhkl_vector, h, k, l)
                term1 = fs*fs.conjugate()
                self.lattice_cache[i_hkl] = term1.real
                i_hkl += 1

        if self.peak_cache==[]:
            # Recalculate the peak part (DW and peak shape)
            self.peak_cache = np.zeros( (len(hkl_list),len(q_list)) )

            i_hkl = 0
            for h, k, l, m, f, qhkl, qhkl_vector in hkl_list:
                term2 = np.exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
                self.peak_cache[i_hkl] = term2 * peak.val_array( q_list, qhkl )
                i_hkl += 1

        summation = np.zeros( (len(q_list)) )
        i_hkl = 0
        for h, k, l, m, f, qhkl, qhkl_vector in hkl_list:
            summation += (m*(f**2)) * self.lattice_cache[i_hkl] * self.peak_cache[i_hkl]
            i_hkl += 1

        return summation


    # Form factor computations
    ########################################

    def G_q(self,q):
        ''' the G(q) from the Debye-Waller factor.
            If sigma_D is a two tuple, then compute the DW factor per unit
            normal.
            q must be a 3-vector

            Note : normalized to lattice_spacing for now for backwards compatibility
            so sigma = 1/(2sd ld)
        '''
        if self.sigma_D.ndim == 0:
            res = np.exp( -(self.sigma_D**2) * (np.linalg.norm(q)**2 * self.lattice_spacing_a**2) )
        else:
            res = 1.
            for sd, unrmx, unrmy, unrmz in self.sigma_D:
                # need to fix later, should be array form
                u = np.array([unrmx,unrmy, unrmz])
                res *= np.exp( -(sd**2) * (np.dot(q,u)**2) * (self.lattice_spacing_a**2) )
        return res




    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the isotropic form factor for the lattice. This is the
        scattering intensity one would measure if the component particles were
        freely dispersed, and thus randomly oriented in solution."""

        # Compute P(q) by summing each object's P(q)
        P = 0
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            P += obj.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)

        return P

    def form_factor_intensity_array(self, qx, qy, qz):

        # Compute P(q) by summing each object's P(q) (form_factor_intensity_isotropic)
        P = np.zeros_like(qx,dtype=complex)
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            xi = xi*self.lattice_spacing_a
            yi = yi*self.lattice_spacing_b
            zi = zi*self.lattice_spacing_c
            P += obj.form_factor(qx, qy, qz)*np.exp(1j*(qx*xi + qy*yi + qz*zi))

        return np.abs(P)**2

    def form_factor_intensity_isotropic_oriented(self, q_list, num_phi=50, num_theta=50):
        """Returns a 1D array of the isotropic form factor.
            This is like form_factor_intensity_isotropic, except the distances
            between the particles is assumed to remain the same.

            Note for polydispersity :  a correction of <|F|^2> - <|F|>^2 needs to be added
                (Fisotmp and Fiso2tmp)
                For non polydisperse samples, this is just zero
        """
        # Using array methods is at least 2X faster
        phi_vals, dphi = np.linspace( 0, 2*np.pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = np.linspace( 0, np.pi, num_theta, endpoint=False, retstep=True )

        F = np.zeros( (len(q_list)), dtype=np.float )
        Ftmp = np.zeros( (len(q_list)), dtype=np.complex )
        Fisotmp = np.zeros( (len(q_list)), dtype=np.float)
        Fiso2tmp = np.zeros( (len(q_list)), dtype=np.float)

        dStot = 0.

        # for each theta phi, compute isotropic intensity
        for theta in theta_vals:
            qz =  q_list*cos(theta)
            dS = sin(theta)*dtheta*dphi

            qy_partial = q_list*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)
                Ftmp *= 0
                Fisotmp *= 0

                for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
                    xi = xi*self.lattice_spacing_a
                    yi = yi*self.lattice_spacing_b
                    zi = zi*self.lattice_spacing_c
                    Ftmp += obj.form_factor(qx, qy, qz)*np.exp(1j*(qx*xi + qy*yi + qz*zi))
                    Fisotmp += np.abs(obj.form_factor(qx,qy,qz))**2
                    Fiso2tmp += obj.form_factor_squared(qx,qy,qz)

                F += (np.abs(Ftmp)**2 + Fiso2tmp - Fisotmp)*dS
                # now subtract form factor iso and add form factor squared
                dStot += dS

        return F/dStot

    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):

        # Compute P(q) by summing each object's P(q) (form_factor_intensity_isotropic)
        P = np.zeros( (len(q_list)) )
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            P += obj.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)

        return P

    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<U(q)>|^2 = Sum_j[ |<F_j(q)>|^2 ] """

        G = 0.0
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            G += obj.beta_numerator(q, num_phi=num_phi, num_theta=num_theta)

        return G


    def beta_numerator_array(self, q_list, num_phi=50, num_theta=50):

        G = np.zeros( (len(q_list)) )
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            G += obj.beta_numerator_array(q_list, num_phi=num_phi, num_theta=num_theta)

        return G


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<U(q)>|^2 / <|U(q)|^2>
        for the lattice."""
        if approx:
            beta = 0.0
            n = 0
            for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
                beta += obj.beta_ratio(q, num_phi=num_phi, num_theta=num_theta, approx=True)
                n += 1
            beta = beta/n
        else:
            P, beta = self.P_beta(q, num_phi=num_phi, num_theta=num_theta)

        return beta


    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<U(q)>|^2 / <|U(q)|^2>
        for the lattice."""
        if approx:
            beta = np.zeros( (len(q_list)) )
            n = 0
            for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
                beta += obj.beta_ratio_array(q_list, num_phi=num_phi, num_theta=num_theta, approx=True)
                n += 1
            beta = beta/n
        else:
            P, beta = self.P_beta_array(q_list, num_phi=num_phi, num_theta=num_theta)

        return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):

        P = self.form_factor_intensity_isotropic_array( q, num_phi=num_phi, num_theta=num_theta)
        if approx:
            beta = self.beta_ratio(q, num_phi=num_phi, num_theta=num_theta, approx=True)
        else:
            G = self.beta_numerator( q, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        return P, beta


    def P_beta_array(self, q_list, num_phi=50, num_theta=50, approx=False):

        P = self.form_factor_intensity_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta)
        if approx:
            beta = self.beta_ratio_array(q_list, num_phi=num_phi, num_theta=num_theta, approx=True)
        else:
            G = self.beta_numerator_array( q_list, num_phi=num_phi, num_theta=num_theta)
            beta = G/P

        return P, beta



    # Final S(q) and Z_0(q)
    ########################################

    def structure_factor_isotropic(self, q, peak, c=1.0, background=None, max_hkl=6):
        """Returns the structure factor S(q) for the specified q-position."""

        P = self.form_factor_intensity_isotropic(q)

        S = (c/(q**2 * P))*self.sum_over_hkl(q, peak, max_hkl=max_hkl)

        if background==None:
            return S
        else:
            return S + background.val(q)/P


    def structure_factor_isotropic_array(self, q_list, peak, c=1.0, background=None, max_hkl=6):
        """Returns the structure factor S(q) for the specified q-position."""

        P = self.form_factor_intensity_isotropic_array(q_list)

        S = (c/(q_list**2 * P))*self.sum_over_hkl_array(q_list, peak, max_hkl=max_hkl)
        #S = (c/(q_list**2 ))*self.sum_over_hkl_array(q_list, peak, max_hkl=max_hkl)

        if background==None:
            return S
        else:
            return S + background.val_array(q_list)/P


    def intensity(self, q, peak, c=1.0, background=None, max_hkl=6):
        """Returns the predicted scattering intensity.
            This is Z0(q) in Kevin's paper
        """

        PS = (c/(q**2))*self.sum_over_hkl(q, peak, max_hkl=max_hkl)

        if background==None:
            return PS
        else:
            return background.val(q) + PS

    def intensity_array(self, q_list, peak, c=1.0, background=None, max_hkl=6):
        """Returns the predicted scattering intensity."""

        PS = (1.0/(q_list**2))*self.sum_over_hkl_array(q_list, peak, max_hkl=max_hkl)

        if background==None:
            return c*PS
        else:
            back = background.val_array(q_list)
            return back + c*PS


    def intensity_cache(self, q_list, peak, c=1.0, background=None, max_hkl=6):
        """Returns the predicted scattering intensity.
        This version using caching to increase performance. You must be mindful
        to call cache_clear whenever parameters change."""

        if self.PS_cache!=[]:
            if background==None:
                return c*self.PS_cache
            else:
                back = background.val_array(q_list)
                return back + c*self.PS_cache
        else:
            # Something needs to be recalculated

            PS = (1.0/(q_list**2))*self.sum_over_hkl_cache(q_list, peak, max_hkl=max_hkl)
            self.PS_cache = PS

            if background==None:
                return c*PS
            else:
                back = background.val_array(q_list)
                return back + c*PS


    def cache_clear(self, lattice=False, DW=False, peak=False, q_list=False):
        if q_list:
            lattice=True
            DW=True
            peak=True
            self.q_list_cache = []

        if lattice:
            self.lattice_cache = []
            self.PS_cache = []

        if DW or peak:
            self.peak_cache = []
            self.PS_cache = []


    # Plotting
    ########################################

    def plot_structure_factor_isotropic(self, qtuple, peak, filename='S_of_q.png', c=1.0, background=None, max_hkl=6, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = np.linspace( q_initial, q_final, num_q, endpoint=True )
        S_list = self.structure_factor_isotropic_array( q_list, peak, c=c, background=background, max_hkl=max_hkl )


        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)


        plt.plot( q_list, S_list, color=(0,0,0), linewidth=3.0 )

        if ylog:
            plt.semilogy()
            fig.subplots_adjust(left=0.15)
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        plt.ylabel( r'$S(q)$' )

        #xi, xf, yi, yf = plt.axis()
        #plt.axis( [xi, xf, yi, yf] )

        plt.savefig( filename )

        return S_list



    def plot_intensity(self, qtuple, peak, filename='intensity.png', c=1.0, background=None, max_hkl=6, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = np.linspace( q_initial, q_final, num_q, endpoint=True )

        int_list = self.intensity_array( q_list, peak, c=c, background=background, max_hkl=max_hkl )


        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)

        plt.plot( q_list, int_list, color=(0,0,0), linewidth=2.0 )

        if ylog:
            plt.semilogy()

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        plt.ylabel( 'Intensity (a.u.)', size=30 )

        #xi, xf, yi, yf = plt.axis()
        #plt.axis( [xi, xf, yi, yf] )

        plt.savefig( filename )

        return int_list


    def plot_form_factor_intensity_isotropic(self, qtuple, filename='form_factor_isotropic.png', ylog=False):
        """Outputs a plot of the P(q) vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = np.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = self.form_factor_intensity_isotropic_array( q_list, num_phi=50, num_theta=50)


        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.17, bottom=0.15, right=0.94, top=0.94)

        plt.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )

        if ylog:
            plt.semilogy()

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        plt.ylabel( r'$P(q)$', size=30 )

        #xi, xf, yi, yf = plt.axis()
        #plt.axis( [xi, xf, yi, yf] )

        plt.savefig( filename )

        return int_list


    def plot_beta_ratio(self, qtuple, filename='beta_ratio.png', ylog=False):
        """Outputs a plot of the beta ratio vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = np.linspace( q_initial, q_final, num_q, endpoint=True )

        int_list = self.beta_ratio_array( q_list )


        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 'xx-large'
        plt.rcParams['ytick.labelsize'] = 'xx-large'
        fig = plt.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)

        plt.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )

        if ylog:
            plt.semilogy()

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        plt.ylabel( r'$\beta(q)$', size=30 )

        if not ylog:
            xi, xf, yi, yf = plt.axis()
            yi = 0.0
            yf = 1.05
            plt.axis( [xi, xf, yi, yf] )

        plt.savefig( filename )

        return int_list



    # Outputs
    ########################################

    def to_string(self):
        """Returns a string describing the lattice."""
        s = "Lattice of type: " + self.__class__.__name__ + "\n"

        s += "    Family: %s   System: %s   Bravais: %s   Class: %s   Space Group: %s\n" % (self.symmetry['crystal family'], \
                                                                                    self.symmetry['crystal system'], \
                                                                                    self.symmetry['Bravais lattice'], \
                                                                                    self.symmetry['crystal class'], \
                                                                                    self.symmetry['space group'], \
                                                                                    )

        s += "    (a, b, c) = (%.3f,%.3f,%.3f) in nm\n" % (self.lattice_spacing_a,self.lattice_spacing_b,self.lattice_spacing_c)
        s += "    (alpha, beta, gamma) = (%.3f,%.3f,%.3f) in radians\n" % (self.alpha,self.beta,self.gamma)
       # s += "                         = (%.2f,%.2f,%.2f) in degrees\n" % (degrees(self.alpha),degrees(self.beta),degrees(self.gamma))
        s += "    volume = %.4f nm^3\n\n" % self.unit_cell_volume()
        s += "    Objects:\n"
        for i, obj in enumerate(self.objects):
            if i<len(self.positions):
                pos = self.positions[i]
            else:
                pos = '---'
            s += "        %d (%s)\t %s (%s)\n" % (i, pos, obj.__class__.__name__, obj.to_short_string() )
        s += "    Unit cell:\n"
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            objstr = obj.__class__.__name__
            s += "        %d (%s)\n" % (i, pos)
            s += "           \t%s (%s)\n" % (objstr, obj.to_short_string())
            s += "           \t  at pos = (%.3f,%.3f,%.3f)\n" % (xi, yi, zi)
        return s


    def to_povray_string(self):
        """Returns a string with POV-Ray code for visualizing the object."""

        s = "#declare lattice_spacing_a = %f;\n" % (self.lattice_spacing_a)
        s += "#declare lattice_spacing_b = %f;\n" % (self.lattice_spacing_b)
        s += "#declare lattice_spacing_c = %f;\n" % (self.lattice_spacing_c)
        s += "#declare %s = union {\n" % (self.__class__.__name__)

        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            s += "    object { %s translate <%f*lattice_spacing_a,%f*lattice_spacing_b,%f*lattice_spacing_c> }\n" % (obj.to_povray_string(), xi, yi, zi)

        s += "}\nobject { %s }\n" % (self.__class__.__name__)

        return s

# HexagonalLattice
###################################################################
class HexagonalLattice(Lattice):


    # Initialization
    ########################################
    def __init__(self, objects, lattice_spacing_a=1.0, lattice_spacing_b=None, lattice_spacing_c=None, alpha=90, beta=90, gamma=60, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        self.objects = objects

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b
        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a
        else:
            self.lattice_spacing_c = lattice_spacing_c

        self.alpha = radians(alpha)
        if beta==None:
            self.beta = radians(alpha)
        else:
            self.beta = radians(beta)
        if gamma==None:
            self.gamma = radians(alpha)
        else:
            self.gamma = radians(gamma)


        self.sigma_D = np.array(sigma_D)          # Lattice disorder



        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'triclinic'
        self.symmetry['crystal system'] = 'triclinic'
        self.symmetry['Bravais lattice'] = '?'
        self.symmetry['crystal class'] = '?'
        self.symmetry['point group'] = '?'
        self.symmetry['space group'] = '?'

        self.positions = ['---']
        self.lattice_positions = ['main', \
                                ]
        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                ]

    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""

        # NOTE: Valid for ideal hexagonal only
        qhkl_vector = ( 2*np.pi*h/(self.lattice_spacing_a), \
                        2*np.pi*(h+2*k)/(np.sqrt(3)*self.lattice_spacing_b), \
                        2*np.pi*l/(self.lattice_spacing_c) )
        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return (qhkl, qhkl_vector)


#OctahedralLattice
###################################################################
class OctahedralCylindersLattice(Lattice):

    ''' octahedral lattice of cylinders.'''

    # Initialization
    ########################################
    def __init__(self, pargs=None, lattice_spacing=1.0,
                 edgelength=1.0, sigma_D=0.01):
        ''' objects not necessary.'''

        self.init_containers()
        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = '?'
        self.symmetry['crystal system'] = '?'
        self.symmetry['Bravais lattice'] = '?'
        self.symmetry['crystal class'] = '?'
        self.symmetry['point group'] = '?'
        self.symmetry['space group'] = '?'


        self.lattice_spacing_a = lattice_spacing
        self.lattice_spacing_b = lattice_spacing
        self.lattice_spacing_c = lattice_spacing


        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        fac1 = np.sqrt(2)/2.*(.5*edgelength + 2*pargs['radius'])


        poslist = np.array([
        # top part
        [0, 45, -90, 0, fac1, fac1, 'top'],
        [0, 45, 0, fac1, 0, fac1, 'top'],
        #[0, 45, 90, 0, -fac1, fac1, 'top'],
        #[0, -45, 0, -fac1, 0, fac1, 'top'],
        # now the flat part
        #[0, 90, 45, fac1, fac1, 0, 'middle'],
        #[0, 90, -45, fac1, -fac1, 0, 'middle'],
        #[0, 90, 45, -fac1, -fac1, 0, 'middle'],
        #[0, 90, -45, -fac1, fac1, 0, 'middle'],
        # finally bottom part
        #[0, 45, -90, 0, -fac1,-fac1, 'bottom'],
        #[0, 45, 0, -fac1, 0, -fac1, 'bottom'],
        #[0, 45, 90, 0, fac1, -fac1, 'bottom'],
        #[0, -45, 0, fac1, 0, -fac1, 'bottom'],
        ])
        objects = list()
        self.lattice_coordinates = list()
        self.lattice_positions = list()
        for pos  in poslist:
            eta, phi, theta, x0, y0, z0,name = pos
            eta = float(eta)
            theta = float(theta)
            phi = float(phi)
            x0, y0, z0 = float(x0), float(y0), float(z0)

            cyl = CylinderNanoObject(pargs=pargs)
            cyl.set_angles(eta=eta, phi=phi, theta=theta)
            #cyl.set_origin(x0=x0, y0=y0, z0=z0)
            self.lattice_coordinates.append((x0, y0, z0))
            objects.append(cyl)
            self.lattice_positions.append(name)


        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        self.objects = objects
        self.lattice_objects = objects

        #self.positions = ['---']


# BCCLattice
###################################################################
class BCCLattice(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        # We only need one object for pure BCC. Ignore everything else.
        self.objects = [ objects[0] ]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'I'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Im3m'


        self.positions = ['all']
        self.lattice_positions = ['corner', 'center']

        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.5), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                ]




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        if (h+k+l)%2==0:
            return 2
        else:
            return 0

    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""

        prefactor = (2*np.pi/self.lattice_spacing_a)
        qhkl_vector = ( prefactor*h, \
                        prefactor*k, \
                        prefactor*l )
        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        prefactor = (2*np.pi/self.lattice_spacing_a)
        qhkl = prefactor*np.sqrt( h**2 + k**2 + l**2 )

        return qhkl


    def iterate_over_hkl_1d_alternate(self, max_hkl=6, tolerance=1e-10):
        """Alternative method of computing the hkl peaks.
        (This method is slightly slower, so there is no advantage in using it.)"""

        self.hkl_list_1d = {}
        self.hkl_list_1d[0.0] = [0, 0, 0, 1, 2]

        for h in range(1,max_hkl+1):
            # m_h00 = 6
            f = self.symmetry_factor(h, 0, 0)
            if f>0:
                self.hkl_list_1d[self.q_hkl_length(h,0,0)] = [h, 0, 0, 6, f]

            # m_hh0 = 12
            f = self.symmetry_factor(h, h, 0)
            if f>0:
                self.hkl_list_1d[self.q_hkl_length(h,h,0)] = [h, h, 0, 12, f]

            # m_hhh = 8
            f = self.symmetry_factor(h, h, h)
            if f>0:
                self.hkl_list_1d[self.q_hkl_length(h,h,h)] = [h, h, h, 8, f]

        for h in range(1,max_hkl+1):
            for k in range(1,max_hkl+1):
                if h!=k:

                    # m_hk0 = 24
                    f = self.symmetry_factor(h, k, 0)
                    if f>0:
                        self.hkl_list_1d[self.q_hkl_length(h,k,0)] = [h, k, 0, 24, f]

                    # m_hhk = 24
                    f = self.symmetry_factor(h, h, k)
                    if f>0:
                        self.hkl_list_1d[self.q_hkl_length(h,h,k)] = [h, h, k, 24, f]

        for h in range(1,max_hkl+1):
            for k in range(1,max_hkl+1):
                for l in range(1,max_hkl+1):
                    if h!=k and h!=l and k!=l:
                        # m_hkl = 48
                        f = self.symmetry_factor(h, k, l)
                        if f>0:
                            self.hkl_list_1d[self.q_hkl_length(h,k,l)] = [h, k, l, 48, f]


        # Each element in dictionary is like:
        # q : [ h, k, l, m, f ]
        return self.hkl_list_1d



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3


# BodyCenteredTwoParticleLattice
###################################################################
class BodyCenteredTwoParticleLattice(BCCLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object for corners and center
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object for BCC. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'center']
        self.lattice_positions = ['corner', 'center']

        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.5), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[1], \
                                ]



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3


# BodyCenteredTwoParticleExtendedLattice
###################################################################
class BodyCenteredTwoParticleExtendedLattice(BodyCenteredTwoParticleLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, repeat=3, filling_a=1.0, filling_b=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object for corners and center
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object for BCC. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'center']
        self.lattice_positions = []
        self.lattice_coordinates = []
        self.lattice_objects = []

        #lattice_sub_cell = 1.0/(repeat*1.0)
        lattice_sub_cell = 1.0
        itotal = 0
        for ix in range(repeat):
            xi = ix*lattice_sub_cell
            for iy in range(repeat):
                yi = iy*lattice_sub_cell
                for iz in range(repeat):
                    zi = iz*lattice_sub_cell

                    if( random.uniform(0.0,1.0)<=filling_a ):
                        self.lattice_positions.append( 'corner-of-subcell' )
                        self.lattice_coordinates.append( ( xi , yi , zi ) )
                        self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                    itotal += 1

                    if( random.uniform(0.0,1.0)<=filling_b ):
                        self.lattice_positions.append( 'center-of-subcell' )
                        self.lattice_coordinates.append( ( xi + 0.5*lattice_sub_cell , yi + 0.5*lattice_sub_cell , zi + 0.5*lattice_sub_cell ) )
                        self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                    itotal += 1



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def unit_cell_volume(self):
        return self.lattice_spacing_a**3


    def swap_particles(self, index1, index2):
        """Swaps the position of the two particles. This allows one to test various
        re-arrangements of the lattice."""

        temp = self.lattice_objects[index1]
        self.lattice_objects[index1] = self.lattice_objects[index2]
        self.lattice_objects[index2] = temp



# BodyCenteredTwoParticleExtendedRandomizedLattice
###################################################################
class BodyCenteredTwoParticleExtendedRandomizedLattice(BodyCenteredTwoParticleLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, repeat=6, randomize=0.0, sigma_D=0.01):

        # NOTE: Using this function will require changing the scaling factor c:
        # c_new = 5*c_old/( repeat**3 )

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object for corners and center
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object for BCC. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'center']
        self.lattice_positions = []
        self.lattice_coordinates = []
        self.lattice_objects = []

        #lattice_sub_cell = 1.0/(repeat*1.0)
        lattice_sub_cell = 1.0
        for ix in range(repeat):
            xi = ix*lattice_sub_cell
            for iy in range(repeat):
                yi = iy*lattice_sub_cell
                for iz in range(repeat):
                    zi = iz*lattice_sub_cell

                    swap_rnd = random.uniform(0.0, 2.0)

                    self.lattice_positions.append( 'corner-of-subcell' )
                    self.lattice_coordinates.append( ( xi , yi , zi ) )
                    if randomize<=swap_rnd:
                        self.lattice_objects.append( self.objects[0] )
                    else:
                        self.lattice_objects.append( self.objects[1] )

                    #swap_rnd = random.uniform(0.0, 2.0) # Uncomment this to make the lattice sites random rather that 'swap'-based (overally stoichiometry may deviate from ideality).

                    self.lattice_positions.append( 'center-of-subcell' )
                    self.lattice_coordinates.append( ( xi + 0.5*lattice_sub_cell , yi + 0.5*lattice_sub_cell , zi + 0.5*lattice_sub_cell ) )
                    if randomize<=swap_rnd:
                        self.lattice_objects.append( self.objects[1] )
                    else:
                        self.lattice_objects.append( self.objects[0] )




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def unit_cell_volume(self):
        return self.lattice_spacing_a**3


    def swap_particles(self, index1, index2):
        """Swaps the position of the two particles. This allows one to test various
        re-arrangements of the lattice."""

        temp = self.lattice_objects[index1]
        self.lattice_objects[index1] = self.lattice_objects[index2]
        self.lattice_objects[index2] = temp



# BodyCenteredTwoParticleExtendedJitteredLattice
###################################################################
class BodyCenteredTwoParticleExtendedJitteredLattice(BodyCenteredTwoParticleExtendedLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, repeat=4, pos_jitter=0.1, angle_jitter=10, sigma_D=0.01):
        # angle_jitter (in degrees) causes the corner-to-center bond angle to deviate randomly from
        # a pure BCC version.

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object for corners and center
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object for BCC. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'center']
        self.lattice_positions = []
        self.lattice_coordinates = []
        self.lattice_objects = []

        #lattice_sub_cell = 1.0/(repeat*1.0)
        lattice_sub_cell = 1.0
        for ix in range(repeat):
            xi = ix*lattice_sub_cell
            for iy in range(repeat):
                yi = iy*lattice_sub_cell
                for iz in range(repeat):
                    zi = iz*lattice_sub_cell

                    xdisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell
                    ydisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell
                    zdisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell

                    thdisp = np.radians( random.uniform(-angle_jitter,+angle_jitter) )
                    phdisp = np.radians( random.uniform(-angle_jitter,+angle_jitter) )

                    self.lattice_positions.append( 'corner-of-subcell' )
                    self.lattice_coordinates.append( ( xi+xdisp , yi+ydisp , zi+zdisp ) )
                    self.lattice_objects.append( self.objects[0] )



                    # Center particle is reoriented w.r.t. to the corner particle
                    xu = 0.5
                    yu = 0.5
                    zu = 0.5
                    # Rotate about x
                    xu = xu*1
                    yu = 0 + yu*np.cos(thdisp) - zu*np.sin(thdisp)
                    zu = 0 + yu*np.sin(thdisp) + zu*np.cos(thdisp)
                    # Rotate about z
                    xu = xu*np.cos(thdisp) - yu*np.sin(thdisp) + 0
                    yu = xu*np.sin(thdisp) + yu*np.cos(thdisp) + 0
                    zu = zu*1

                    self.lattice_positions.append( 'center-of-subcell' )
                    self.lattice_coordinates.append( ( xi + xdisp + xu*lattice_sub_cell , yi + ydisp + yu*lattice_sub_cell , zi + zdisp + zu*lattice_sub_cell ) )
                    self.lattice_objects.append( self.objects[1] )

                    if False:
                        xu = 0.5
                        yu = 0.5
                        zu = 0.5
                        # Rotate about x
                        xu = xu*1
                        yu = 0 + yu*np.cos(thdisp) - zu*np.sin(thdisp)
                        zu = 0 + yu*np.sin(thdisp) + zu*np.cos(thdisp)
                        # Rotate about z
                        xu = xu*np.cos(thdisp) - yu*np.sin(thdisp) + 0
                        yu = xu*np.sin(thdisp) + yu*np.cos(thdisp) + 0
                        zu = zu*1

                        self.lattice_positions.append( 'center-of-subcell' )
                        self.lattice_coordinates.append( ( xi + xdisp + xu*lattice_sub_cell , yi + ydisp + yu*lattice_sub_cell , zi + zdisp + zu*lattice_sub_cell ) )
                        self.lattice_objects.append( self.objects[1] )



# BodyCenteredTwoParticleExtendedTwistedLattice
###################################################################
class BodyCenteredTwoParticleExtendedTwistedLattice(BodyCenteredTwoParticleExtendedLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, repeat=6, twist=3, missing=0.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object for corners and center
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object for BCC. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'center']
        self.lattice_positions = []
        self.lattice_coordinates = []
        self.lattice_objects = []

        lattice_sub_cell = 1.0/(repeat*1.0)
        lattice_sub_cell = 1.0
        for ix in range(repeat):
            xi = ix*lattice_sub_cell
            for iy in range(repeat):
                yi = iy*lattice_sub_cell
                for iz in range(repeat):
                    zi = iz*lattice_sub_cell

                    if ix>=0 and ix<(repeat/2):
                        twist_radians = ix*np.radians(twist)
                    else:
                        twist_radians = (repeat-1-ix)*np.radians(twist)
                    xc = xi
                    yc = yi*np.cos(twist_radians) - zi*np.sin(twist_radians)
                    zc = yi*np.sin(twist_radians) + zi*np.cos(twist_radians)

                    if random.uniform(0,1)>missing:
                        self.lattice_positions.append( 'corner-of-subcell' )
                        self.lattice_coordinates.append( ( xc, yc, zc ) )
                        self.lattice_objects.append( self.objects[0] )


                    xu = 0.5
                    yu = 0.5
                    zu = 0.5

                    xc = xi + xu*lattice_sub_cell
                    yc = yi + yu*lattice_sub_cell
                    zc = zi + zu*lattice_sub_cell

                    # Apply twist
                    xc = xc
                    yc = yc*np.cos(twist_radians) - zc*np.sin(twist_radians)
                    zc = yc*np.sin(twist_radians) + zc*np.cos(twist_radians)

                    #if random.uniform(0,1)>missing:
                    if True:
                        self.lattice_positions.append( 'center-of-subcell' )
                        self.lattice_coordinates.append( ( xc, yc, zc ) )
                        self.lattice_objects.append( self.objects[1] )

                    if False:
                        xu = 0.5
                        yu = 0.5
                        zu = 0.5
                        # Rotate about x
                        xu = xu*1
                        yu = 0 + yu*np.cos(thdisp) - zu*np.sin(thdisp)
                        zu = 0 + yu*np.sin(thdisp) + zu*np.cos(thdisp)
                        # Rotate about z
                        xu = xu*np.cos(thdisp) - yu*np.sin(thdisp) + 0
                        yu = xu*np.sin(thdisp) + yu*np.cos(thdisp) + 0
                        zu = zu*1

                        self.lattice_positions.append( 'center-of-subcell' )
                        self.lattice_coordinates.append( ( xi + xdisp + xu*lattice_sub_cell , yi + ydisp + yu*lattice_sub_cell , zi + zdisp + zu*lattice_sub_cell ) )
                        self.lattice_objects.append( self.objects[1] )


# BodyCenteredTwoParticleExtendedOrientationalSpreadLattice
###################################################################
class BodyCenteredTwoParticleExtendedOrientationalSpreadLattice(BodyCenteredTwoParticleExtendedLattice):


    def __init__(self, objects, lattice_spacing_a=1.0, repeat=4, pos_jitter=0.0, angle_jitter=10, sigma_D=0.01):
        # angle_jitter (in degrees) causes each particle to be randomly reoriented a bit

        # NOTE: Using this function will require changing the scaling factor c:
        # c_new = 5*c_old/( repeat**3 )

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object for corners and center
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object for BCC. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'center']
        self.lattice_positions = []
        self.lattice_coordinates = []
        self.lattice_objects = []


        #lattice_sub_cell = 1.0/(repeat*1.0)
        lattice_sub_cell = 1.0
        for ix in range(repeat):
            xi = ix*lattice_sub_cell
            for iy in range(repeat):
                yi = iy*lattice_sub_cell
                for iz in range(repeat):
                    zi = iz*lattice_sub_cell

                    xdisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell
                    ydisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell
                    zdisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell

                    self.lattice_positions.append( 'corner-of-subcell' )
                    self.lattice_coordinates.append( ( xi+xdisp , yi+ydisp , zi+zdisp ) )
                    new_corner = copy.deepcopy(self.objects[0])
                    # Randomize orientation
                    new_corner.set_angles(phi=random.uniform(-angle_jitter,+angle_jitter), theta=random.uniform(-angle_jitter,+angle_jitter))
                    self.lattice_objects.append( new_corner )


                    xdisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell
                    ydisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell
                    zdisp = random.uniform(-pos_jitter,+pos_jitter)*lattice_sub_cell

                    self.lattice_positions.append( 'center-of-subcell' )
                    self.lattice_coordinates.append( ( xi + xdisp + 0.5*lattice_sub_cell , yi + ydisp + 0.5*lattice_sub_cell , zi + zdisp + 0.5*lattice_sub_cell ) )

                    new_center = copy.deepcopy(self.objects[1])
                    # Randomize orientation
                    new_center.set_angles(phi=random.uniform(-angle_jitter,+angle_jitter), theta=random.uniform(-angle_jitter,+angle_jitter))
                    self.lattice_objects.append( new_center )




# BodyCenteredEdgeFaceTwoParticleLattice
###################################################################
class BodyCenteredEdgeFaceTwoParticleLattice(BCCLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two object. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'I'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Im3m'


        self.positions = ['bcc', 'edgeface']
        self.lattice_positions = ['corner', 'center', 'faceXY', 'faceXZ', 'faceYZ', 'edgeX', 'edgeY', 'edgeZ']

        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.5), \
                                        (0.5, 0.5, 0.0), \
                                        (0.5, 0.0, 0.5), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.0), \
                                        (0.0, 0.5, 0.0), \
                                        (0.0, 0.0, 0.5), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                ]




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3



# FCCLattice
###################################################################
class FCCLattice(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        # We only need one object for pure FCC. Ignore everything else.
        self.objects = [ objects[0] ]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'F'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Fm3m'


        self.positions = ['all']
        self.lattice_positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ']

        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                        ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                ]




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        if (h%2==0) and (k%2==0) and (l%2==0):
            # All even
            return 4
        elif (h%2==1) and (k%2==1) and (l%2==1):
            # All odd
            return 4
        else:
            return 0

    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""

        prefactor = (2*np.pi/self.lattice_spacing_a)
        qhkl_vector = ( prefactor*h, \
                        prefactor*k, \
                        prefactor*l )
        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        prefactor = (2*np.pi/self.lattice_spacing_a)
        qhkl = prefactor*np.sqrt( h**2 + k**2 + l**2 )

        return qhkl


    def iterate_over_hkl_1d_alternate(self, max_hkl=6, tolerance=1e-10):
        """Alternative method of computing the hkl peaks.
        (This method is slightly slower, so there is no advantage in using it.)"""

        self.hkl_list_1d = {}
        self.hkl_list_1d[0.0] = [0, 0, 0, 1, 4]

        for h in range(1,max_hkl+1):
            # m_h00 = 6
            f = self.symmetry_factor(h, 0, 0)
            if f>0:
                self.hkl_list_1d[self.q_hkl_length(h,0,0)] = [h, 0, 0, 6, f]

            # m_hh0 = 12
            f = self.symmetry_factor(h, h, 0)
            if f>0:
                self.hkl_list_1d[self.q_hkl_length(h,h,0)] = [h, h, 0, 12, f]

            # m_hhh = 8
            f = self.symmetry_factor(h, h, h)
            if f>0:
                self.hkl_list_1d[self.q_hkl_length(h,h,h)] = [h, h, h, 8, f]

        for h in range(1,max_hkl+1):
            for k in range(1,max_hkl+1):
                if h!=k:

                    # m_hk0 = 24
                    f = self.symmetry_factor(h, k, 0)
                    if f>0:
                        self.hkl_list_1d[self.q_hkl_length(h,k,0)] = [h, k, 0, 24, f]

                    # m_hhk = 24
                    f = self.symmetry_factor(h, h, k)
                    if f>0:
                        self.hkl_list_1d[self.q_hkl_length(h,h,k)] = [h, h, k, 24, f]

        for h in range(1,max_hkl+1):
            for k in range(1,max_hkl+1):
                for l in range(1,max_hkl+1):
                    if h!=k and h!=l and k!=l:
                        # m_hkl = 48
                        f = self.symmetry_factor(h, k, l)
                        if f>0:
                            self.hkl_list_1d[self.q_hkl_length(h,k,l)] = [h, k, l, 48, f]


        # Each element in dictionary is like:
        # q : [ h, k, l, m, f ]
        return self.hkl_list_1d



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3

# FaceCenteredFourParticleLattice
###################################################################
class FaceCenteredFourParticleLattice(FCCLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 4
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0], objects[0], objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ']
        self.lattice_positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[1], \
                                    self.objects[2], \
                                    self.objects[3], \
                                ]




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3




# DiamondTwoParticleLattice
###################################################################
# a.k.a. zincblende
class DiamondTwoParticleLattice(FCCLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 4
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0], objects[0], objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['cornerFCC', 'tetrahedralFCC']
        self.lattice_positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ', 'tetra1', 'tetra2', 'tetra3', 'tetra4']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                        (0.25, 0.25, 0.25), \
                                        (0.25, 0.75, 0.75), \
                                        (0.75, 0.25, 0.75), \
                                        (0.75, 0.75, 0.25), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                ]



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def unit_cell_volume(self):
        return self.lattice_spacing_a**3




# InterpenetratingDiamondTwoParticleLattice
###################################################################
class InterpenetratingDiamondTwoParticleLattice(DiamondTwoParticleLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['cornerFCC', 'tetrahedrals']
        self.lattice_positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ', 'tetra1', 'tetra2', 'tetra3', 'tetra4', 'central', 'edgeX', 'edgeY', 'edgeZ', 'tetra5', 'tetra6', 'tetra7', 'tetra8']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                        (0.25, 0.25, 0.25), \
                                        (0.25, 0.75, 0.75), \
                                        (0.75, 0.25, 0.75), \
                                        (0.75, 0.75, 0.25), \
                                        (0.5, 0.5, 0.5), \
                                        (0.5, 0.0, 0.0), \
                                        (0.0, 0.5, 0.0), \
                                        (0.0, 0.0, 0.5), \
                                        (0.75, 0.75, 0.75), \
                                        (0.75, 0.25, 0.25), \
                                        (0.25, 0.75, 0.25), \
                                        (0.25, 0.25, 0.75), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                ]




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def unit_cell_volume(self):
        return self.lattice_spacing_a**3




# DoubleFilledDiamondTwoParticleLattice
###################################################################
class DoubleFilledDiamondTwoParticleLattice(DiamondTwoParticleLattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['diamond1', 'diamond2']
        self.lattice_positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ', 'tetra1', 'tetra2', 'tetra3', 'tetra4', 'tetra5', 'tetra6', 'tetra7', 'tetra8']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                        (0.25, 0.25, 0.25), \
                                        (0.25, 0.75, 0.75), \
                                        (0.75, 0.25, 0.75), \
                                        (0.75, 0.75, 0.25), \
                                        (0.75, 0.75, 0.75), \
                                        (0.75, 0.25, 0.25), \
                                        (0.25, 0.75, 0.25), \
                                        (0.25, 0.25, 0.75), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                ]



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def unit_cell_volume(self):
        return self.lattice_spacing_a**3




# CristobaliteLattice
###################################################################
class CristobaliteLattice(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need two objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['tetrahedrals', 'bonds']
        self.lattice_positions = ['corner', 'faceXY', 'faceYZ', 'faceXZ', 'tetra1', 'tetra2', 'tetra3', 'tetra4', 'edge1a', 'edge1b', 'edge1c', 'edge1d', 'edge2a', 'edge2b', 'edge2c', 'edge2d', 'edge3a', 'edge3b', 'edge3c', 'edge3d', 'edge4a', 'edge4b', 'edge4c', 'edge4d', ]


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                        (0.25, 0.25, 0.25), \
                                        (0.25, 0.75, 0.75), \
                                        (0.75, 0.25, 0.75), \
                                        (0.75, 0.75, 0.25), \
                                        ( 1*0.125, 1*0.125, 1*0.125 ), \
                                        ( 3*0.125, 3*0.125, 1*0.125 ), \
                                        ( 5*0.125, 5*0.125, 1*0.125 ), \
                                        ( 7*0.125, 7*0.125, 1*0.125 ), \
                                        ( 3*0.125, 1*0.125, 3*0.125 ), \
                                        ( 1*0.125, 3*0.125, 3*0.125 ), \
                                        ( 7*0.125, 5*0.125, 3*0.125 ), \
                                        ( 5*0.125, 7*0.125, 3*0.125 ), \
                                        ( 5*0.125, 1*0.125, 5*0.125 ), \
                                        ( 7*0.125, 3*0.125, 5*0.125 ), \
                                        ( 1*0.125, 5*0.125, 5*0.125 ), \
                                        ( 3*0.125, 7*0.125, 5*0.125 ), \
                                        ( 7*0.125, 1*0.125, 7*0.125 ), \
                                        ( 5*0.125, 3*0.125, 7*0.125 ), \
                                        ( 3*0.125, 5*0.125, 7*0.125 ), \
                                        ( 1*0.125, 7*0.125, 7*0.125 ), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                ]



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3




# HexagonalDiamondLattice
###################################################################
class HexagonalDiamondLattice(HexagonalLattice):


    # Initialization
    ########################################
    def __init__(self, objects, lattice_spacing_a=1.0, lattice_spacing_b=None, lattice_spacing_c=None, alpha=90, beta=90, gamma=60, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        self.objects = objects

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b
        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a*( 4.0/np.sqrt(6) )
        else:
            self.lattice_spacing_c = lattice_spacing_c

        self.alpha = radians(alpha)
        if beta==None:
            self.beta = radians(alpha)
        else:
            self.beta = radians(beta)
        if gamma==None:
            self.gamma = radians(alpha)
        else:
            self.gamma = radians(gamma)


        self.sigma_D = np.array(sigma_D)          # Lattice disorder



        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'triclinic'
        self.symmetry['crystal system'] = 'triclinic'
        self.symmetry['Bravais lattice'] = '?'
        self.symmetry['crystal class'] = '?'
        self.symmetry['point group'] = '?'
        self.symmetry['space group'] = '?'

        self.positions = ['network1', 'network2']
        self.lattice_positions = ['corner bottom layer', \
                                    'strut lower', \
                                    'strut higher', \
                                    'corner midlayer', \
                                ]
        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (1.0/3.0, 1.0/3.0, 1.0/8.0), \
                                        (1.0/3.0, 1.0/3.0, 4.0/8.0), \
                                        (0.0, 0.0, 5.0/8.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                ]





# AlternatingHexagonalDiamondLattice
###################################################################
class AlternatingHexagonalDiamondLattice(HexagonalDiamondLattice):


    # Initialization
    ########################################
    def __init__(self, objects, lattice_spacing_a=1.0, lattice_spacing_b=None, lattice_spacing_c=None, alpha=90, beta=90, gamma=60, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        self.objects = objects

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b
        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a*( 4.0/np.sqrt(6) )
        else:
            self.lattice_spacing_c = lattice_spacing_c

        self.alpha = radians(alpha)
        if beta==None:
            self.beta = radians(alpha)
        else:
            self.beta = radians(beta)
        if gamma==None:
            self.gamma = radians(alpha)
        else:
            self.gamma = radians(gamma)


        self.sigma_D = np.array(sigma_D)          # Lattice disorder



        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'triclinic'
        self.symmetry['crystal system'] = 'triclinic'
        self.symmetry['Bravais lattice'] = '?'
        self.symmetry['crystal class'] = '?'
        self.symmetry['point group'] = '?'
        self.symmetry['space group'] = '?'

        self.positions = ['network1', 'network2']
        self.lattice_positions = ['corner bottom layer', \
                                    'strut lower', \
                                    'strut higher', \
                                    'corner midlayer', \
                                ]
        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (1.0/3.0, 1.0/3.0, 1.0/8.0), \
                                        (1.0/3.0, 1.0/3.0, 4.0/8.0), \
                                        (0.0, 0.0, 5.0/8.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[1], \
                                    self.objects[0], \
                                    self.objects[1], \
                                ]



# AlongBondsHexagonalDiamondLattice
###################################################################
class AlongBondsHexagonalDiamondLattice(HexagonalDiamondLattice):


    # Initialization
    ########################################
    def __init__(self, objects, lattice_spacing_a=1.0, lattice_spacing_b=None, lattice_spacing_c=None, alpha=90, beta=90, gamma=60, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        self.objects = objects

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b
        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a*( 4.0/np.sqrt(6) )
        else:
            self.lattice_spacing_c = lattice_spacing_c

        self.alpha = radians(alpha)
        if beta==None:
            self.beta = radians(alpha)
        else:
            self.beta = radians(beta)
        if gamma==None:
            self.gamma = radians(alpha)
        else:
            self.gamma = radians(gamma)


        self.sigma_D = np.array(sigma_D)          # Lattice disorder



        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'triclinic'
        self.symmetry['crystal system'] = 'triclinic'
        self.symmetry['Bravais lattice'] = '?'
        self.symmetry['crystal class'] = '?'
        self.symmetry['point group'] = '?'
        self.symmetry['space group'] = '?'

        self.positions = ['network1', 'network2']
        self.lattice_positions = ['lower tripod 1', \
                                    'lower tripod 2', \
                                    'lower tripod 3', \
                                    'mid strut', \
                                    'mid tripod 1', \
                                    'mid tripod 2', \
                                    'mid tripod 3', \
                                    'upper connections', \
                                ]
        self.lattice_coordinates = [ (1.0/6.0, 1.0/6.0, 1.0/16.0), \
                                        (4.0/6.0, 1.0/6.0, 1.0/16.0), \
                                        (1.0/6.0, 4.0/6.0, 1.0/16.0), \
                                        (1.0/3.0, 1.0/3.0, 5.0/16.0), \
                                        (1.0/6.0, 1.0/6.0, 9.0/16.0), \
                                        (4.0/6.0, 1.0/6.0, 9.0/16.0), \
                                        (1.0/6.0, 4.0/6.0, 9.0/16.0), \
                                        (0.0, 0.0, 13.0/16.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                ]



# SimpleCubic
###################################################################
class SimpleCubic(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['all']
        self.lattice_positions = ['corner']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                ]



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3


# Orthorhombic
###################################################################
class OrthorhombicLattice(Lattice):
    def __init__(self, objects, lattice_spacing=(1.,1.,1.), sigma_D=0.01):
        ''' Orthorhombic lattice. 90 degrees unit vectors but variable length
            lattice_spacing : three integers, x, y and z spacing
                where x, y and z follow same convention as the units

        '''

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 1
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing[0]
        self.lattice_spacing_b = lattice_spacing[1]
        self.lattice_spacing_c = lattice_spacing[2]
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['all']
        self.lattice_positions = ['corner']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                ]



    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3


# AlternatingSimpleCubic
###################################################################
class AlternatingSimpleCubic(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)

        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a*1.0
        self.lattice_spacing_b = lattice_spacing_a*1.0
        self.lattice_spacing_c = lattice_spacing_a*1.0
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corners', 'edges']
        self.lattice_positions = ['corner', 'edgeX', 'edgeY', 'edgeZ', 'faceXY', 'faceXZ', 'faceYZ', 'center']


        self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                        (0.5, 0.0, 0.0), \
                                        (0.0, 0.5, 0.0), \
                                        (0.0, 0.0, 0.5), \
                                        (0.5, 0.5, 0.0), \
                                        (0.5, 0.0, 0.5), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.5, 0.5), \
                                    ]



        self.lattice_objects = [ self.objects[0], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[1], \
                                ]




    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3


# AlternatingSimpleCubicExtendedLattice
###################################################################
class AlternatingSimpleCubicExtendedLattice(AlternatingSimpleCubic):
    def __init__(self, objects, lattice_spacing_a=1.0, repeat=3, filling_a=1.0, filling_b=1.0, sigma_D=0.01):

        self.init_containers()

        self.min_objects = 8
        self.expected_objects = 8*repeat
        self.object_count(objects)

        # We only need four objects. Ignore everything else.
        self.objects = objects[0:self.expected_objects]

        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)

        self.sigma_D = np.array(sigma_D)          # Lattice disorder

        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Pm3m'

        self.positions = ['corner', 'edgeX', 'edgeY', 'edgeZ', 'faceXY', 'faceXZ', 'faceYZ', 'center']


        if False:
            self.lattice_positions = ['corner', 'edgeX', 'edgeY', 'edgeZ', 'faceXY', 'faceXZ', 'faceYZ', 'center']

            self.lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                            (0.5, 0.0, 0.0), \
                                            (0.0, 0.5, 0.0), \
                                            (0.0, 0.0, 0.5), \
                                            (0.5, 0.5, 0.0), \
                                            (0.5, 0.0, 0.5), \
                                            (0.0, 0.5, 0.5), \
                                            (0.5, 0.5, 0.5), \
                                        ]

            self.lattice_objects = [ self.objects[0], \
                                        self.objects[1], \
                                        self.objects[2], \
                                        self.objects[3], \
                                        self.objects[4], \
                                        self.objects[5], \
                                        self.objects[6], \
                                        self.objects[7], \
                                    ]

        else:

            self.lattice_positions = []
            self.lattice_coordinates = []
            self.lattice_objects = []


            #lattice_sub_cell = 1.0/(repeat*1.0)
            lattice_sub_cell = 1.0
            itotal = 0
            for ix in range(repeat):
                xi = ix*lattice_sub_cell
                for iy in range(repeat):
                    yi = iy*lattice_sub_cell
                    for iz in range(repeat):
                        zi = iz*lattice_sub_cell

                        if( random.uniform(0.0,1.0)<=filling_a ):
                            self.lattice_positions.append( 'corner' )
                            self.lattice_coordinates.append( ( xi + 0.0*lattice_sub_cell , yi + 0.0*lattice_sub_cell , zi + 0.0*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_b ):
                            self.lattice_positions.append( 'edgeX' )
                            self.lattice_coordinates.append( ( xi + 0.5*lattice_sub_cell , yi + 0.0*lattice_sub_cell , zi + 0.0*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_b ):
                            self.lattice_positions.append( 'edgeY' )
                            self.lattice_coordinates.append( ( xi + 0.0*lattice_sub_cell , yi + 0.5*lattice_sub_cell , zi + 0.0*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_b ):
                            self.lattice_positions.append( 'edgeZ' )
                            self.lattice_coordinates.append( ( xi + 0.0*lattice_sub_cell , yi + 0.0*lattice_sub_cell , zi + 0.5*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_a ):
                            self.lattice_positions.append( 'faceXY' )
                            self.lattice_coordinates.append( ( xi + 0.5*lattice_sub_cell , yi + 0.5*lattice_sub_cell , zi + 0.0*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_a ):
                            self.lattice_positions.append( 'faceXZ' )
                            self.lattice_coordinates.append( ( xi + 0.5*lattice_sub_cell , yi + 0.0*lattice_sub_cell , zi + 0.5*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_a ):
                            self.lattice_positions.append( 'faceYZ' )
                            self.lattice_coordinates.append( ( xi + 0.0*lattice_sub_cell , yi + 0.5*lattice_sub_cell , zi + 0.5*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1

                        if( random.uniform(0.0,1.0)<=filling_b ):
                            self.lattice_positions.append( 'center' )
                            self.lattice_coordinates.append( ( xi + 0.5*lattice_sub_cell , yi + 0.5*lattice_sub_cell , zi + 0.5*lattice_sub_cell ) )
                            self.lattice_objects.append( self.objects[itotal%len(self.objects)] )
                        itotal += 1





    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1



    def unit_cell_volume(self):

        return self.lattice_spacing_a**3





# MultiComponentModel
###################################################################
# NOTE: When adding parameters, update here:
idx_c = 0
idx_nu = 1
idx_delta = 2
idx_sigma_D = 3
idx_bc = 4
idx_bp = 5
idx_balpha = 6
idx_bp2 = 7
idx_balpha2 = 8
idx_scale = 9
idx_offset =10
class MultiComponentModel(Model):


    def __init__(self, lattice, peak, background, c=1.0, max_hkl=6, ptype=None, margs={}):
        """Prepares the model object."""
        self.margs = {}
        self.margs['ptype'] = 'intensity'
        self.margs['scale'] = 1.0
        self.margs['offset'] = 0.0
        self.margs['diffuse'] = True
        self.margs['beta_approx'] = False

        self.margs.update(margs)

        self.c=c
        self.max_hkl=max_hkl
        if ptype==None:
            self.ptype = self.margs['ptype']
        else:
            self.ptype = ptype

        self.scale = self.margs['scale']
        self.offset = self.margs['offset']

        self.lattice = lattice
        self.peak = peak
        self.background = background

        self.last_param_array = self.get_non_lattice()

        self.use_experimental_P_of_q = False


    def update_c(self, c=1.0):
        self.c=c

    def update_non_lattice(self, param_array):
        # NOTE: When adding parameters, update here:

        # Always update these:
        self.update_c( c=param_array[idx_c] )
        self.background.update( constant=param_array[idx_bc], prefactor=param_array[idx_bp], alpha=param_array[idx_balpha], prefactor2=param_array[idx_bp2], alpha2=param_array[idx_balpha2] )
        self.scale = param_array[idx_scale]
        self.offset = param_array[idx_offset]

        if param_array[idx_nu]!=self.last_param_array[idx_nu] or param_array[idx_delta]!=self.last_param_array[idx_delta]:
            self.peak.reshape( nu=param_array[idx_nu], delta=param_array[idx_delta] )
            self.lattice.cache_clear( peak=True )

        if param_array[idx_sigma_D]!=self.last_param_array[idx_sigma_D]:
            self.lattice.update_sigma_D( param_array[idx_sigma_D] )
            self.lattice.cache_clear( DW=True )

        # If the lattice is updated:
        #    self.lattice.cache_clear( lattice=True )

        self.last_param_array = param_array



    def get_non_lattice(self):
        # NOTE: When adding parameters, update here:
        param_array = [ self.c , \
                        self.peak.nu , \
                        self.peak.delta , \
                        self.lattice.sigma_D , \
                        self.background.constant , \
                        self.background.prefactor , \
                        self.background.alpha , \
                        self.background.prefactor2 , \
                        self.background.alpha2 , \
                        self.scale , \
                        self.offset , \
                        ]
        return param_array


    def q_value(self, qx, qy=0, qz=0):
        """Returns the intensity for the given point in
        reciprocal-space."""

        q = np.sqrt( qx**2 + qy**2 + qz**2 )

        if self.ptype=='intensity':
            return self.lattice.intensity(q, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)
        elif self.ptype=='form_factor':
            return self.lattice.form_factor_intensity_isotropic( q )
        elif self.ptype=='structure_factor':
            # TODO: update to new style!!
            return self.lattice.structure_factor_isotropic(q, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)
        else:
            print( "Error: Unknown ptype in function 'q_value'." )
            return 0.0


    def q_array(self, q_list):
        """Returns the intensity for the given point in
        reciprocal-space."""

        if self.ptype=='intensity':
            return self.scale*self.lattice.intensity_array(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl) + self.offset
        elif self.ptype=='form_factor':
            return self.scale*self.lattice.form_factor_intensity_isotropic_array( q_list ) + self.offset
        elif self.ptype=='structure_factor':

            if self.use_experimental_P_of_q:
                P = self.get_experimental_P_of_q_array(q_list, normalize=True)

                if self.margs['diffuse']:
                    beta = self.lattice.beta_ratio_array( q_list, approx=self.margs['beta_approx'] )
                    G = np.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = np.ones( (len(q_list)) ) - beta*G
                else:
                    diffuse = np.zeros( (len(q_list)) )

            else:

                if self.margs['diffuse']:
                    P, beta = self.lattice.P_beta_array( q_list, approx=self.margs['beta_approx'] )
                    G = np.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = np.ones( (len(q_list)) ) - beta*G
                else:
                    P = self.lattice.form_factor_intensity_isotropic_array(q_list)
                    diffuse = np.zeros( (len(q_list)) )

            Z_0 = self.lattice.intensity_array(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)
            #S_0 = self.lattice.structure_factor_isotropic_array(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)

            return self.scale*( Z_0/P + diffuse ) + self.offset


        else:
            print( "Error: Unknown ptype in function 'q_array'." )
            return 0.0


    def q_cache(self, q_list):
        """Returns the intensity for the given point in
        reciprocal-space."""

        if self.ptype=='intensity':
            return self.scale*self.lattice.intensity_cache(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl) + self.offset
        elif self.ptype=='form_factor':
            return self.scale*self.lattice.form_factor_intensity_isotropic_cache( q_list ) + self.offset
        elif self.ptype=='structure_factor':

            if self.use_experimental_P_of_q:
                P = self.get_experimental_P_of_q_array(q_list, normalize=True)

                if self.margs['diffuse']:
                    beta = self.lattice.beta_ratio_array( q_list, approx=self.margs['beta_approx'] )
                    G = np.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = np.ones( (len(q_list)) ) - beta*G
                else:
                    diffuse = np.zeros( (len(q_list)) )

            else:

                if self.margs['diffuse']:
                    P, beta = self.lattice.P_beta_array( q_list, approx=self.margs['beta_approx'] )
                    G = np.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = np.ones( (len(q_list)) ) - beta*G
                else:
                    P = self.lattice.form_factor_intensity_isotropic_array(q_list)
                    diffuse = np.zeros( (len(q_list)) )

            #Z_0 = self.lattice.intensity_array(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)
            Z_0 = self.lattice.intensity_cache(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)
            #S_0 = self.lattice.structure_factor_isotropic_array(q_list, self.peak, c=self.c, background=self.background, max_hkl=self.max_hkl)

            return self.scale*( Z_0/P + diffuse ) + self.offset

        else:
            print( "Error: Unknown ptype in function 'q_array'." )
            return 0.0



    def set_experimental_P_of_q(self, q_list, P_list):
        """Forces the model to use experimental data for P(q). (The model still computes F_j(q).)"""
        self.use_experimental_P_of_q = True

        self.exp_P_q_list = q_list
        self.exp_P_int_list = P_list


    def get_experimental_P_of_q_array(self, q_list, normalize=False):

        P_exp = []

        j = 1
        for i, q in enumerate(q_list):

            while self.exp_P_q_list[j]<q and j<len(self.exp_P_q_list)-1:
                j += 1


            P = self.interpolate( [ self.exp_P_q_list[j], self.exp_P_int_list[j] ] , [ self.exp_P_q_list[j-1], self.exp_P_int_list[j-1] ], q )
            P_exp.append( P )


        if normalize:
            P_exp = P_exp/P_exp[-1]
        return P_exp



    def interpolate(self, point1, point2, x_target ):
        x1, y1 = point1
        x2, y2 = point2
        m = (y2-y1)/(x2-x1)
        b = y2-m*x2

        y_target = m*x_target + b

        return y_target




    def to_string(self):
        """Returns a string describing the model."""
        s = "MultiComponentModel"
        return s



# FittingSiman
###################################################################
# This object encapsulates a simulated annealing object, which calculates its
# enegery based on the goodness-of-fit.

# A try loop is used since on many systems GSL will not be available.
# On those systems, this code is skipped, which means that the fitting
# capabilities are missing.
try:

    import pygsl.siman as siman                     # For simulated annealing (data fitting)
    import pygsl.rng as rng                         # Random numbers
    import copy

    class FittingSiman(siman.NumericEnsemble):

        constraint_penalty = 1e100

        def __init__(self, q_list, int_list, model, vary=None, step_sizes=None, constraints=None):

            self.q_list = q_list
            self.int_list = int_list
            self.model = model

            self.vary = vary
            self.step_sizes = step_sizes
            self.constraints = constraints

            self._data = 0.0



        # The current fit is contained within the "position" in parameter-space
        def SetPos(self, positions):
            self.pos = positions


        def GetPos(self):
            return self.pos


        def EFunc(self):
            """Calculates the current energy of the system."""

            # We compute an "energy" by calculating the total error between
            # the dataset (self.data) and the current predictions of the
            # fit variables (self.pos)
            energy = 0


            # 1. Update model
            # NOTE: When adding parameters, update here:
            self.model.update_non_lattice(self.pos)

            # 2. Get fit
            #int_fit = self.model.q_array(self.q_list)
            int_fit = self.model.q_cache(self.q_list)

            # 3. Sum residuals squared
            for fit_val, data_val in zip(int_fit, self.int_list):
                #energy += (data_val - fit_val)**2
                energy += (data_val - fit_val)**2/(data_val**2)
                #energy += abs(data_val - fit_val)/abs(data_val)
                #energy += (data_val - fit_val)**6/(data_val**6)


            # 5. Apply constraints
            for i, p in enumerate(self.pos):
                if self.vary[i] and p<self.constraints[i][0] or p>self.constraints[i][1]:
                    energy += self.constraint_penalty


            return energy


        # This function is part of the simulated annealing specification,
        # but is not used in the present implementation
        def Metric(self, other):
            return numx.absolute(self._data - other.GetData())


        # This function causes the simulation to take a random
        # step. It is these steps that the simulated annealing
        # algorithm samples.
        def Step(self, rng, step_size):

            # Select a random variable (that we're allowed to vary!)
            found_index = False
            while( not found_index ):
                var_index = random.randint(0, len(self.pos)-1 )
                found_index = self.vary[var_index]

            # Determine step size
            step_size = self.step_sizes[var_index]

            # The variables's current position
            var_old = self.pos[var_index]

            # Move to a random new position
            u = rng.uniform();
            var_new = (2*u - 1)*step_size + var_old;

            self.pos[var_index] = var_new


        # Prints out information about the current status of the object.
        def Print(self):
            s = " ["
            for val in self.pos:
                s += "%.3g, " % (val)
            s += "] "
            #print( s )
            sys.stdout.write(s)



        # Make an exact copy of this object
        # This function is necessary because the simulated annealing algorithm
        # generates clones of the current state, and then modifies them, in order
        # to speculatively sample parameter space.
        def Clone(self):
            model = copy.deepcopy(self.model)
            clone = self.__class__(self.q_list, self.int_list, model, vary=self.vary, step_sizes=self.step_sizes, constraints=self.constraints)
            #clone.SetData(copy.copy(self._data))
            clone.SetPos(copy.deepcopy(self.pos))

            return clone


except ImportError:
    # pygsl is not available. Simply ignore the error and move on.
    # (Fitting capabilities will simply not work.)
    pass





# MultiComponentFit
###################################################################
class MultiComponentFit(object):


    def __init__(self, data, model, initial_guess=None, q_start=None, q_end=None, index_start=None, index_end=None, vary=None, step_sizes=None, constraints=None, fargs={}):

        self.fargs = {}
        self.fargs['number_tries'] = 6          # How many points to try before stepping
        self.fargs['iterations_at_T'] = 6       # How many iterations at each temperature?
        self.fargs['step_size'] = 1.0           # Scaling of step sizes
        self.fargs['k'] = 1.0                   # Boltzmann constant
        self.fargs['T_initial'] = 0.008         # Initial temperature
        self.fargs['mu_T'] = 1.02               # Damping factor for temperature
        self.fargs['T_min'] = 1.0e-4            # Final temperature

        self.fargs['chi'] = 'relative'          # Can take values: absolute, relative
                                                # TODO: Add log, etc.

        self.fargs['ptype'] = 'intensity'       # Can be 'intensity', 'form_factor' or 'structure_factor'

        self.fargs.update(fargs)

        self.data = data
        self.model = model


        # The possible parameters for the fit are:
        # symbol, name, min, max, step_size
        # NOTE: When adding parameters, update here:
        self.parameters = [         [ 'c', 'scattering prefactor',                  0.0, 1e50, 0.1e-12 ], \
                                    [ 'nu', 'peak shape',                           0.0, 210.0, 0.05 ] , \
                                    [ 'delta', 'peak width',                        0.0, 2.0, 0.01 ] , \
                                    [ 'sigma_D', 'Debye-Waller',                    0.0001, 10.0, 0.01 ] , \
                                    [ 'bc', 'constant background',                  0.0, 1e10, 0.1 ], \
                                    [ 'bp', 'variable background prefactor',        0.0, 1e10, 0.05 ] , \
                                    [ 'balpha', 'variable background exponent',     -4.0, -1.0, 0.1] , \
                                    [ 'bp2', 'variable background prefactor 2',     0.0, 1e10, 0.05] , \
                                    [ 'balpha2', 'variable background exponent 2',  -3.5, 0.0, 0.1] , \
                                    [ 'scale', 'overall scaling',                   0.0, 1e20, 0.1] , \
                                    [ 'offset', 'overall offset',                   0.0, 1e10, 0.1] , \
                                    ]

        # Determine which parameters we actually care about
        if vary==None:
            self.vary = [ True for i in range(len(self.parameters)) ]
        else:
            self.vary = vary


        # Prepare the arrays that the fitting will use (we only care about parameters to vary)
        self.fit_num_params = 0
        self.fit_param_symbols = []
        self.fit_step_sizes = []
        self.fit_constraints = []
        for i, do_vary in enumerate(self.vary):
            if do_vary:
                self.fit_num_params += 1

            if step_sizes==None:
                # Use default
                self.fit_step_sizes.append( self.fargs['step_size']*self.parameters[i][4] )
            else:
                # user-supplied
                self.fit_step_sizes.append( self.fargs['step_size']*step_sizes[i] )
            if constraints==None:
                self.fit_constraints.append( [self.parameters[i][2],self.parameters[i][3]] )
            else:
                self.fit_constraints.append( constraints[i] )


        # Determine what experimental data to try and fit
        self.set_q_range(q_start=q_start, q_end=q_end, index_start=index_start, index_end=index_end)


        self.fit_initial_guess = []
        if initial_guess==None:
            # Pick some values arbitrarily
            for i, el in enumerate(self.parameters):
                if self.vary[i]:
                    lowerbound = el[2]
                    upperbound = el[3]
                    self.fit_initial_guess.append( (upperbound-lowerbound)/2.0 )
        else:
            self.fit_initial_guess = initial_guess


        # NOTE: When adding parameters, update here:
        self.model.update_non_lattice( self.fit_initial_guess )

        self.fit_final_result = []



    def set_data_indices(self, q_list, q_start=None, q_end=None, index_start=None, index_end=None):

        # Determine which datapoints to consider
        if q_start!=None:
            i = 0
            while i<len(q_list)-1 and q_list[i]<q_start:
                i += 1
            self.index_start = i
        elif index_start!=None:
            self.index_start=index_start
        else:
            self.index_start=0

        if q_end!=None:
            i = len(q_list)-1
            while i>0 and q_list[i]>q_end:
                i -= 1
            self.index_end = i
        elif index_end!=None:
            self.index_end=index_end
        else:
            self.index_end=len(q_list)-1

        return self.index_start, self.index_end


    def set_q_range(self, q_start=None, q_end=None, index_start=None, index_end=None):
        # Determine what experimental data to try and fit
        if self.fargs['ptype']=='intensity':
            self.set_data_indices( self.data.q_vals, q_start=q_start, q_end=q_end, index_start=index_start, index_end=index_end )
            self.q_list = np.array( self.data.q_vals )[self.index_start:self.index_end]
            self.int_list = self.data.intensity_vals[self.index_start:self.index_end]

        elif self.fargs['ptype']=='form_factor':
            self.set_data_indices( self.data.q_ff_vals, q_start=q_start, q_end=q_end, index_start=index_start, index_end=index_end )
            self.q_list = np.array( self.data.q_ff_vals )[self.index_start:self.index_end]
            self.int_list = self.data.ff_vals[self.index_start:self.index_end]

        elif self.fargs['ptype']=='structure_factor':
            S_of_q = self.data.structure_factor()
            self.set_data_indices( S_of_q[:,0], q_start=q_start, q_end=q_end, index_start=index_start, index_end=index_end )
            self.q_list = S_of_q[self.index_start:self.index_end,0]
            self.int_list = S_of_q[self.index_start:self.index_end,1]

        else:
            print( "Error: Unknown ptype in initialization of class 'MultiComponentFit'." )


    def update(self, guess=None, fargs={}):
        self.fargs.update(fargs)

        # NOTE: When adding parameters, update here:
        self.model.update_non_lattice(guess)


    def get_current_guess(self):
        # NOTE: When adding parameters, update here:
        return self.model.get_non_lattice()


    def fit(self, initial_guess=None, verbose=True, fargs={}):

        self.fargs.update(fargs)

        if initial_guess!=None:
            # User-supplied values
            self.fit_initial_guess = initial_guess

        if verbose:
            do_print=1
        else:
            do_print=0



        # Instantiate fit object
        fit_obj = FittingSiman( self.q_list, self.int_list, self.model, vary=self.vary, step_sizes=self.fit_step_sizes, constraints=self.fit_constraints)

        # Initial guess
        fit_obj.SetPos( self.fit_initial_guess )


        # New random-number-generator
        r = rng.rng()

        # Run solver
        result = siman.solve( r, fit_obj, do_print=do_print, n_tries=self.fargs['number_tries'], iters_fixed_T=self.fargs['iterations_at_T'],
                        step_size=self.fargs['step_size'], k=self.fargs['k'], t_initial=self.fargs['T_initial'], mu_t=self.fargs['mu_T'],
                        t_min=self.fargs['T_min'] )

        # Store result
        self.fit_final_result = result.GetPos()
        self.model = result.model


        # Print result
        if verbose:
            name_width = 35
            name = padded_string( 'name', name_width)
            print( "symbol\t%s\tinitial\tmin\tmax\tstep\tfinal" % (name) )


            for i, el in enumerate(self.parameters):

                val = self.fit_final_result[i]
                initial = self.fit_initial_guess[i]
                name = padded_string( el[1], name_width)
                if self.vary[i]:

                    lower = self.fit_constraints[i][0]
                    upper = self.fit_constraints[i][1]
                    step = self.fit_step_sizes[i]
                    print( "%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (el[0], name, initial, lower, upper, step, val) )
                else:
                    print( "%s\t%s\t%.3g\t--\t--\t--\t%.3g" % (el[0], name, initial, val) )


            print( self.params_to_string(self.fit_final_result) )

        return result.model


    def fit_curve(self, q_start=0.05, q_end=0.40, num_q=400):

        q_list = np.linspace( q_start, q_end, num_q )

        int_list = self.model.q_array(q_list)

        return q_list, int_list


    def make_watch_file(self, filename='fit-working.vals' ):

        guess = self.get_current_guess()

        fout = open( filename, 'w' )

        symbol_width = 9
        name_width = 35
        number_width = 11

        symbol = padded_string( 'symbol', symbol_width)
        name = padded_string( 'name', name_width)
        min_c = padded_string( 'min', number_width )
        max_c = padded_string( 'max', number_width )
        step_size = padded_string( 'step', number_width )
        val = 'val'

        fout.write( "i  %s%s:  %s%s%s%s\n" % (symbol, name, min_c, max_c, step_size, val)  )

        for i, els in enumerate(self.parameters):
            symbol, name, min_c, max_c, step_size = els

            symbol = padded_string( symbol, symbol_width )
            name = padded_string( name, name_width)
            min_c, max_c = self.fit_constraints[i]

            min_c = padded_string( "%.3g" % (min_c), number_width )
            max_c = padded_string( "%.3g" % (max_c), number_width )
            step_size = padded_string( "%.2g" % (self.fit_step_sizes[i]), number_width )
            val = "%.4g" % (guess[i])

            fout.write( "%02d %s%s:  %s%s%s%s\n" % (i, symbol, name, min_c, max_c, step_size, val)  )
        fout.close()


    def parse_watch_file(self, filename='fit_working.vals' ):
        guess = []

        fin = open( filename )

        i = 0
        for line in fin.readlines():
            if i>0:
                parts = line.split(':')
                els = parts[1].split()
                guess.append(float(els[-1]))
            i += 1

        fin.close()

        return guess


    def watch_file(self, watch_filename='fit-working.vals', plot_filename='fit-working.png', poling_interval=0.5, title='', scaling=None, xlog=False, ylog=False, show_extended=True, show_indexing=False):

        guess = []

        try:
            last_timestamp = 0
            while True:
                statinfo = os.stat(watch_filename)
                timestamp = statinfo.st_mtime

                if last_timestamp<timestamp:

                    sys.stdout.write("%s updated: " % (watch_filename) )
                    guess = self.parse_watch_file(filename=watch_filename)
                    print( self.params_to_string(guess) )

                    self.update_and_plot( guess, filename=plot_filename, title=title, scaling=scaling, xlog=xlog, ylog=ylog, show_extended=show_extended, show_indexing=show_indexing)
                    last_timestamp = timestamp

                    print('    done.')




                time.sleep(poling_interval)

        except KeyboardInterrupt:
            # End the watching loop
            pass

        return guess


    def update_and_plot(self, guess=None, fargs={}, filename='fit-working.png', title='', scaling=None, xlog=False, ylog=False, show_extended=True, show_indexing=False):
        self.update(guess=guess, fargs=fargs)
        self.plot(filename=filename, title=title, scaling=scaling, xlog=xlog, ylog=ylog, show_extended=show_extended, show_indexing=show_indexing)



    def plot(self, ptype=None, qi=0.0, qf=None, filename='fit.png', title='', scaling=None, xlog=False, ylog=False, show_indexing=True, indexing_simplify=True, show_extended=True, res_fig_height=0.1, dpi=None, interact=False):
        """Outputs a plot of the intensity vs. q data."""


        # Get data
        err_vals = []
        if ptype==None:
            ptype=self.fargs['ptype']

        if ptype=='form_factor':
            pass
        elif ptype=='structure_factor':
            s_of_q = self.data.structure_factor()
            q_list = s_of_q[:,0]
            int_list = s_of_q[:,1]

            # Create a fit to data
            q_fit = q_list[self.index_start:self.index_end]
            #q_fit = np.linspace( q_list[0]*0.01, q_list[-1]*1.5, 2000 ) # Extended q-range
            int_fit = self.model.q_array(q_fit)

            if show_extended:
                # Create extended fit
                q_fit_extended = np.linspace( q_list[0]*0.5, q_list[-1]*1.1, 400 )
                int_fit_extended = self.model.q_array(q_fit_extended)


        else:
            q_list = self.data.q_vals
            int_list = self.data.intensity_vals

            if self.data.error_vals != [] and len(self.data.error_vals)==len(int_list):
                err_vals = self.data.error_vals

            # Create a fit to data
            q_fit = self.q_list
            int_fit = self.model.q_array(q_fit)

            if show_extended:
                # Create extended fit
                q_fit_extended = np.linspace( q_list[0]*0.5, q_list[-1]*1.1, 200 )
                #q_fit_extended = copy.deepcopy(q_fit)
                #q_fit_extended[-1]=0.9
                int_fit_extended = self.model.q_array(q_fit_extended)


        if True:
            # Output fit-data to file
            fout = open( filename + '.dat', 'w' )
            for qcur, intcur in zip(q_fit, int_fit):
                fout.write(str(qcur)+'\t'+str(intcur)+'\n')
            fout.close()
            if show_extended:
                fout = open( filename + '-extended.dat', 'w' )
                for qcur, intcur in zip(q_fit_extended, int_fit_extended):
                    fout.write(str(qcur)+'\t'+str(intcur)+'\n')
                fout.close()


        bigger_axes_fonts = True

        axes_font_size = 30
        plt.rcParams['axes.labelsize'] = axes_font_size # Not used (override)
        plt.rcParams['xtick.labelsize'] = 'x-large'
        plt.rcParams['ytick.labelsize'] = 'x-large'

        if bigger_axes_fonts:
            axes_font_size = 35
            plt.rcParams['xtick.labelsize'] = 25
            plt.rcParams['ytick.labelsize'] = 25



        fig = plt.figure()
        if bigger_axes_fonts:
            ax1 = fig.add_axes( [0.15,0.18,0.94-0.15,0.94-0.15-res_fig_height] )
        else:
            ax1 = fig.add_axes( [0.15,0.15,0.94-0.15,0.94-0.15-res_fig_height] )

        if err_vals != []:
            plt.errorbar( q_list, int_list, yerr=err_vals, fmt='o', color=(0,0,0), ecolor=(0.5,0.5,0.5), markersize=6.0 )
        else:
            plt.plot( q_list, int_list, 'o', color=(0,0,0), markersize=6.0 )

        if xlog and ylog:
            plt.loglog()
        else:
            if xlog:
                plt.semilogx()
            if ylog:
                plt.semilogy()

        plt.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=axes_font_size )
        if ptype=='form_factor':
            plt.ylabel( r'$P(q)$', size=axes_font_size )
        elif ptype=='structure_factor':
            plt.ylabel( r'$S(q)$', size=axes_font_size )
        else:
            plt.ylabel( 'Intensity (a.u.)', size=axes_font_size )

        # Get axis scaling
        xi, xf, yi, yf = plt.axis()


        if show_extended:
            plt.plot( q_fit_extended, int_fit_extended, '-', color=(0.5,0.5,1), linewidth=1.0 )
        plt.plot( q_fit, int_fit, '-', color=(0,0,1), linewidth=2.0 )



        if show_indexing:
            list1d = self.model.lattice.iterate_over_hkl_1d()

            x_peak_indexing = []
            y_peak_indexing = []
            labels_indexing = []

            for q in sorted(list1d.keys()):
                h, k, l, m, f = list1d[q]
                miller = [h, k, l]
                if indexing_simplify:
                    miller.sort()
                    miller.reverse()
                hn, kn, ln = miller

                intensity = m*(f**2)

                # Skip 0,0,0
                if not (hn==0 and kn==0 and ln==0):
                    x_peak_indexing.append( q )
                    y_peak_indexing.append( intensity/(q**2) )
                    s = "%d%d%d" % (hn, kn, ln)
                    labels_indexing.append( s )


            # Rescale the indexing peaks so that they are not larger than the experimental data
            q0, q0_intensity = value_at( x_peak_indexing[0], q_list, int_list )
            rescaling = (q0_intensity/y_peak_indexing[0])*0.5
            y_peak_indexing = np.asarray( y_peak_indexing )*rescaling

            markerline, stemlines, baseline = plt.stem( x_peak_indexing, y_peak_indexing )
            plt.setp(markerline, 'markersize', 4.0)
            for xc, yc, lc in zip(x_peak_indexing, y_peak_indexing, labels_indexing):
                plt.text( xc, yc+q0_intensity*0.015, lc, size=8, verticalalignment='bottom', horizontalalignment='center' )

        # Set axis scaling
        if scaling==None:
            # (We want the experimental data centered, not the fit.)
            xi = qi
            if qf!=None:
                xf = qf
            plt.axis( [xi, xf, yi, yf] )
        else:
            plt.axis( scaling )


        # Get ax1 final scaling
        x1i, x1f, y1i, y1f = plt.axis()
        # Find indices of 'data of interest'
        index_start = 0
        index_finish = 0
        for i, q in enumerate(q_list):
            if q<x1i:
                index_start = i
            if q<x1f:
                index_end = i


        # Residuals plot
        ax2 = fig.add_axes( [0.15,0.94-res_fig_height,0.94-0.15,res_fig_height] )

        if self.fargs['chi']=='absolute':
            residuals = [ y_fit-y_exp for y_fit, y_exp in zip(int_fit, int_list[self.index_start:self.index_end]) ]
        elif self.fargs['chi']=='relative':
            residuals = [ (y_fit-y_exp)/y_exp for y_fit, y_exp in zip(int_fit, int_list[self.index_start:self.index_end]) ]
        else:
            print( "Error: Unknown type of 'chi' in plot()." )

        plt.plot( self.q_list, residuals, '-', color='b', linewidth=2.0 )



        # Error calculations
        int_list_subset = int_list[self.index_start:self.index_end]     # The experimental data that matches the fit region
        degree_freedom = len(q_list)-self.fit_num_params-1
        chi_sq_p = sum( [ (y_fit-y_exp)**2/(y_exp**2) for y_fit, y_exp in zip(int_fit, int_list_subset) ] ) / degree_freedom
        chi_sq_p_str = "%.3g" % (chi_sq_p)
        if err_vals != []:
            chi_sq_red = 0
            for i, y_fit in enumerate(int_fit):
                chi_sq_red += (y_fit-int_list_subset[i])**2/(err_vals[i]**2)
            chi_sq_red = chi_sq_red/degree_freedom
            chi_sq_red_str = "%.3g" % (chi_sq_red)
        else:
            chi_sq_red_str = "NA"

        energy = sum( [ (y_fit-y_exp)**2/(y_exp**2) for y_fit, y_exp in zip(int_fit, int_list_subset) ] )
        energy_str = "%.4g" % (energy)

        chi_str = r"$\chi_{\mathrm{red}}^2 = " + chi_sq_red_str + "$\n$\chi_{\mathrm{P}}^2 = " + chi_sq_p_str + "$\n$E=" + energy_str + "$"
        plt.figtext( 0.94-0.01, 0.94-res_fig_height-0.01, chi_str, horizontalalignment='right', verticalalignment='top', size=20 )


        # Title
        if title!='':
            DW = self.model.lattice.sigma_D
            title = title + ' (%s, DW = %.2f)' % (self.model.lattice.__class__.__name__, DW)
            plt.figtext( 0.06, 0.95, title, size=14, horizontalalignment='left', verticalalignment='bottom' )


        # Figure out scaling
        xi, xf, yi, yf = ax2.axis()
        xi = x1i
        xf = x1f
        yf = max( yf, abs(yi) )
        yi = -yf


        if show_extended:
            # Interpolate data to get residuals
            residuals_extended = []
            j = 0
            for q, y_exp in zip(q_list, int_list):

                # Find closest theoretical values
                while( j<len(q_fit_extended) and q_fit_extended[j] < q ):
                    j += 1
                if j==0:
                    j = 1
                elif j>=len(q_fit_extended):
                    j = len(q_fit_extended)-1

                y_fit = self.interpolate( (q_fit_extended[j], int_fit_extended[j]), (q_fit_extended[j-1],int_fit_extended[j-1]), q )
                if self.fargs['chi']=='absolute':
                    residuals_extended.append( y_fit-y_exp )
                elif self.fargs['chi']=='relative':
                    residuals_extended.append( (y_fit-y_exp)/y_exp )
                else:
                    print( "Error: Unknown type of 'chi' in plot()." )



            plt.plot( q_list, residuals_extended, '-', color='b', linewidth=1.0 )


        if err_vals != []:
            # Create a filled-in region
            q_list_backwards = [ q_list[i] for i in range(len(q_list)-1,-1,-1) ]
            if self.fargs['chi']=='absolute':
                upper2 = [ +2.0*err_vals[i] for i in range(len(err_vals)) ]
                lower2 = [ -2.0*err_vals[i] for i in range(len(err_vals)-1,-1,-1) ]
                upper1 = [ +1.0*err_vals[i] for i in range(len(err_vals)) ]
                lower1 = [ -1.0*err_vals[i] for i in range(len(err_vals)-1,-1,-1) ]
            elif self.fargs['chi']=='relative':
                upper2 = [ +2.0*err_vals[i]/(int_list[i]+0.1) for i in range(len(err_vals)) ]
                lower2 = [ -2.0*err_vals[i]/(int_list[i]+0.1) for i in range(len(err_vals)-1,-1,-1) ]
                upper1 = [ +1.0*err_vals[i]/(int_list[i]+0.1) for i in range(len(err_vals)) ]
                lower1 = [ -1.0*err_vals[i]/(int_list[i]+0.1) for i in range(len(err_vals)-1,-1,-1) ]
            else:
                print( "Error: Unknown type of 'chi' in plot()." )

            ax2.fill( q_list+q_list_backwards, upper2+lower2, edgecolor='0.92', facecolor='0.92' )
            ax2.fill( q_list+q_list_backwards, upper1+lower1, edgecolor='0.82', facecolor='0.82' )


        plt.axhline( 0, color='k' )
        ax2.axis( [xi, xf, yi, yf] )

        ax2.set_xticklabels( [] )
        ax2.set_yticklabels( [] )


        if dpi==None:
            plt.savefig( filename )
        else:
            plt.savefig( filename, dpi=dpi )
        if interact:
            plt.show()


        return int_list

    def interpolate(self, point1, point2, x_target ):
        x1, y1 = point1
        x2, y2 = point2
        m = (y2-y1)/(x2-x1)
        b = y2-m*x2

        y_target = m*x_target + b

        return y_target




    def params_to_string(self, params=None):
        if params==None:
            if self.fit_final_result!= [] and self.fit_final_result!=None:
                params = self.fit_final_result
            else:
                params = self.get_current_guess()

        guess_str = ''
        for el in params:
            guess_str += '%.4g, ' % (el)
        return 'params = [ %s]' % (guess_str)
