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


from BaseClasses import *  
from gamma import *
import os, sys

import random



# NanoObject
####################################################################
class NanoObject(Potential):
    """Defines a nano-object, which can then be placed within a lattice
    for computing scattering data. A nano-object can be anisotropic."""
    
    # TODO: Don't ignore the rotation_matrix/rotation_elements stuff
    
    conversion_factor = 1E-4        # Converts units from 1E-6 A^-2 into nm^-2
    
    def __init__(self, pargs={}, seed=None):
        self.rotation_matrix = numpy.identity(3)
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


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""
        
        return self.conversion_factor*0.0


    def set_angles(self, eta=None, phi=None, theta=None):
        """Update one or multiple orientation angles (degrees)."""
        if eta != None:
            self.pargs['eta'] = eta
        if phi != None:
            self.pargs['phi'] = phi
        if theta != None:
            self.pargs['theta'] = theta
            
        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )
        
        

    def rotation_elements(self, eta, phi, theta):
        """Converts angles into an appropriate rotation matrix."""
        
        # Possible BUG: These rotations might be conceptually wrong...
        
        # Three-axis rotation:
        # 1. Rotate about +z by eta 
        # 2. Tilt by phi with respect to +z (rotation about y-axis) then
        # 3. rotate by theta in-place (rotation about z-axis)

        eta = radians( eta )        # eta is orientation around the z axis (before reorientation)
        phi = radians( phi )        # phi is grain tilt (with respect to +z axis)
        theta = radians( theta )    # grain orientation (around the z axis)
        
        rotation_elements = [[  cos(eta)*cos(phi)*cos(theta)-sin(eta)*sin(theta) ,
                                    -cos(eta)*cos(phi)*sin(theta)-sin(eta)*cos(theta) ,
                                    -cos(eta)*sin(phi)                                   ],
                            [  sin(eta)*cos(phi)*cos(theta)+cos(eta)*sin(theta) ,
                                    -sin(eta)*cos(phi)*sin(theta)+cos(eta)*cos(theta) ,
                                    sin(eta)*sin(phi)                                    ],
                            [ -sin(phi)*cos(theta) ,
                                sin(phi)*sin(theta) ,
                                cos(phi)                                              ]]
        
        return rotation_elements


    def rotate_q(self, qx, qy, qz):
        """Rotates the q-vector in the way given by the internal
        rotation_matrix, which should have been set using "set_angles"
        or the appropriate pargs (eta, phi, theta)."""
        
        q_vector = numpy.array( [[qx],[qy],[qz]] )
        
        q_rotated = numpy.dot( self.rotation_matrix, q_vector )
        qx = q_rotated[0,0]
        qy = q_rotated[1,0]
        qz = q_rotated[2,0]
        
        return qx, qy, qz


    def rotate_q_array__replaced__(self, qx_list, qy_list, qz_list):

        for i in range(len(qx_list)):
            
            q_vector = numpy.array( [[qx_list[i]],[qy_list[i]],[qz_list[i]]] )
        
            q_rotated = numpy.dot( self.rotation_matrix, q_vector )
            qx_list[i] = q_rotated[0,0]
            qy_list[i] = q_rotated[1,0]
            qz_list[i] = q_rotated[2,0]
        
        return qx_list, qy_list, qz_list


    def rotate_q_array(self, qx_list, qy_list, qz_list):
        
        # Using matrix operations is at least 5X faster
        
        Q_r = numpy.dot( self.rotation_matrix, [qx_list, qy_list, qz_list] )
        
        return Q_r[0], Q_r[1], Q_r[2]


    def form_factor_numerical(self, qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None):
        """This is a brute-force calculation of the form-factor, using
        the realspace potential. This is computationally intensive and
        should be avoided in preference to analytical functions which
        are put into the "form_factor(qx,qy,qz)" function."""
        
        qx, qy, qz = self.rotate_q(qx, qy, qz)
        
        q = (qx, qy, qz)
        
        if size_scale==None:
            if 'radius' in self.pargs:
                size_scale = 2.0*self.pargs['radius']
            else:
                size_scale = 2.0

        x_vals, dx = numpy.linspace( -size_scale, size_scale, num_points, endpoint=True, retstep=True)
        y_vals, dy = numpy.linspace( -size_scale, size_scale, num_points, endpoint=True, retstep=True)
        z_vals, dz = numpy.linspace( -size_scale, size_scale, num_points, endpoint=True, retstep=True)

        dVolume = dx*dy*dz
        
        f = 0.0+0.0j
        
        # Triple-integral over 3D space
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    r = (x, y, z)
                    V = self.V(x, y, z, rotation_elements=rotation_elements)
                    
                    #a = numpy.dot(q,r)
                    #b = cexp( 1j*numpy.dot(q,r) )
                    #val = V*cexp( 1j*numpy.dot(q,r) )*dV
                    #print x, y, z, V, a, b, dV, val
                    
                    f += V*cexp( 1j*numpy.dot(q,r) )*dVolume
        
        return self.pargs['delta_rho']*f


    def form_factor_squared_numerical(self, qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None):
        """Returns the square of the form factor."""
        f = self.form_factor_numerical(qx,qy,qz, num_points=num_points, size_scale=size_scale, rotation_elements=rotation_elements)
        g = f*f.conjugate()
        return g.real


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        
        qx, qy, qz = self.rotate_q(qx, qy, qz)

        return self.pargs['delta_rho']*0.0 + self.pargs['delta_rho']*0.0j
        
        
    def form_factor_array(self, qx, qy, qz):
        
        F = numpy.zeros( (len(qx)), dtype=numpy.complex )
        for i, qxi in enumerate(qx):
            F[i] = self.form_factor(qx[i], qy[i], qz[i])
        
        return F

        
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
        if self.pargs['cache_results'] and q in self.form_factor_isotropic_already_computed:
            return self.form_factor_isotropic_already_computed[q]
        
        
        phi_vals, dphi = numpy.linspace( 0, 2*pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
        
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
       
       
    def form_factor_orientation_spread(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over some orientations.
        This function is intended to be used to create a effective form factor
        when particle orientations have some distribution.
        
        The default distribution is uniform. Endpoints are given by pargs['form_factor_orientation_spread']
        """
        
        # Usage:
        # pargs['form_factor_orientation_spread'] = [0, 2*pi, 0, pi] # Isotropic
        
        # WARNING: This function essentially ignores particle orientation, since it remaps a given q_hkl to q, which is then converted back into a spread of qx, qy, qz. Thus, all q_hkl are collapsed unphysically. As such, it does not correctly account for the orientation distribution of a particle. It should only be considered a crude approximation.
        # TODO: Don't ignore particle orientation.
        
        # Check cache
        if self.pargs['cache_results'] and q in self.form_factor_isotropic_already_computed:
            return self.form_factor_isotropic_already_computed[q]
        
        phi_start, phi_end, theta_start, theta_end = self.pargs['form_factor_orientation_spread']
        
        # Phi is orientation around z-axis (in x-y plane)
        phi_vals, dphi = numpy.linspace( phi_start, phi_end, num_phi, endpoint=False, retstep=True )
        # Theta is tilt with respect to +z axis
        theta_vals, dtheta = numpy.linspace( theta_start, theta_end, num_theta, endpoint=False, retstep=True )
        
        
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

        F = numpy.zeros( (len(q_list)), dtype=numpy.complex )
        for i, q in enumerate(q_list):

            F[i] = self.form_factor_isotropic( q, num_phi=num_phi, num_theta=num_theta )
        
        return F


    def form_factor_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        """Returns a 1D array of the isotropic form factor."""
        
        # Using array methods is at least 2X faster
        
        phi_vals, dphi = numpy.linspace( 0, 2*pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
        F = numpy.zeros( (len(q_list)), dtype=numpy.complex )
        
        for theta in theta_vals:
            qz =  q_list*cos(theta)
            dS = sin(theta)*dtheta*dphi
            
            qy_partial = q_list*sin(theta)
            for phi in phi_vals:
                qx = -qy_partial*cos(phi)
                qy =  qy_partial*sin(phi)

                F += self.form_factor_array(qx, qy, qz) * dS
                        
        return F
        
    
    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        
        phi_vals, dphi = numpy.linspace( 0, 2*pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
        
        P = 0.0
        
        for theta in theta_vals:
            qz =  q*cos(theta)
            dS = sin(theta)*dtheta*dphi
            
            for phi in phi_vals:
                qx = -q*sin(theta)*cos(phi)
                qy =  q*sin(theta)*sin(phi)
                
                P += self.form_factor_squared(qx, qy, qz) * dS
                
        
        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        """Returns a 1D array of the form factor intensity (orientation averaged)."""
        
        P = numpy.zeros( (len(q_list)) )
        for i, q in enumerate(q_list):
            P[i] = self.form_factor_intensity_isotropic( q, num_phi=num_phi, num_theta=num_theta )
        
        return P


    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """Returns the numerator of the beta ratio: |<F(q)>|^2"""
        
        # For a monodisperse system, this is simply P(q)
        return self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)


    def beta_numerator_array(self, q_list, num_phi=50, num_theta=50):
        # For a monodisperse system, this is simply P(q)

        return self.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<F(q)>|^2 / <|F(q)|^2>
        This ratio depends on polydispersity: for a monodisperse system, beta = 1 for all q."""
        return 1.0


    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns a 1D array of the beta ratio."""
        beta = numpy.ones( len(q_list) )
        return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""
        
        P = self.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)
        beta = self.beta_ratio(q, num_phi=num_phi, num_theta=num_theta, approx=approx)
        
        return P, beta
        
        
    def P_beta_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns P (isotropic_form_factor_intensity) and beta_ratio.
        This function can be highly optimized in derived classes."""
        
        P = self.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
        beta = self.beta_ratio_array(q_list, num_phi=num_phi, num_theta=num_theta, approx=approx)
        
        return P, beta


    def plot_form_factor_amplitude(self, qtuple, filename='form_factor_amplitude.png', ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values."""
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = []
        for q in q_list:
            int_list.append( self.form_factor(0,0,q).real )
            
        #q_zeros = numpy.zeros(len(q_list))
        #int_list = self.form_factor_array(q_zeros,q_zeros,q_list) 

        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)

        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        pylab.ylabel( r'$F(q)$' )        

        #xi, xf, yi, yf = pylab.axis()
        #yf = 5e5
        #yi = 0
        #pylab.axis( [xi, xf, yi, yf] )

        pylab.savefig( filename )
        
        return int_list
            
    

    def plot_form_factor_intensity(self, qtuple, filename='form_factor_intensity.png', ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q) 
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = []
        for q in q_list:
            int_list.append( self.form_factor_intensity(0,0,q) )


        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)

        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        pylab.ylabel( r'$F(q)$' )        

        #xi, xf, yi, yf = pylab.axis()
        #yf = 5e5
        #pylab.axis( [xi, xf, yi, yf] )

        pylab.savefig( filename )
        
        return int_list
            
    def plot_form_factor_intensity_isotropic(self, qtuple, filename='form_factor_intensity_isotropic.png', ylog=False, num_phi=50, num_theta=50):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = self.form_factor_intensity_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta )


        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.16, bottom=0.15, right=0.94, top=0.94)

        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))

        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        pylab.ylabel( r'$P(q)$' )        


        xi, xf, yi, yf = pylab.axis()
        yi, yf = ( 1e7 , 1e15 )
        pylab.axis( [xi, xf, yi, yf] )
        
        pylab.savefig( filename )
        
        return int_list


    def plot_beta_ratio(self, qtuple, filename='beta_ratio.png', ylog=False, approx=False):
        """Outputs a plot of the beta-ratio vs. q data. Also returns an array
        of the intensity values.
        qtuple = (q_initial, q_final, num_q) 
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = self.beta_ratio_array( q_list, approx=approx )


        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.16, bottom=0.15, right=0.94, top=0.94)

        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))

        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        pylab.ylabel( r'$\beta(q)$' )        


        
        if not ylog:
            xi, xf, yi, yf = pylab.axis()
            yi = 0.0
            yf = 1.05
            pylab.axis( [xi, xf, yi, yf] )
        
        pylab.savefig( filename )
        
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
        
        
        for x in numpy.arange(-full_size, full_size, minibox_size):
            for y in numpy.arange(-full_size, full_size, minibox_size):
                for z in numpy.arange(-full_size, full_size, minibox_size):
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
    size ('radius') of width 'sigma_R'."""
    
    
    def __init__(self, baseNanoObjectClass, pargs={}, seed=None):
        
        NanoObject.__init__(self, pargs=pargs, seed=seed)

        # Set defaults
        if 'sigma_R' not in self.pargs:
            self.pargs['sigma_R'] = 0.01
        if 'distribution_type' not in self.pargs:
            self.pargs['distribution_type'] = 'gaussian'
        if 'distribution_num_points' not in self.pargs:
            self.pargs['distribution_num_points'] = 5
        if 'iso_external' not in self.pargs:
            self.pargs['iso_external'] = False


        self.baseNanoObjectClass = baseNanoObjectClass
        
        self.distribution_list = []
        
        
        self.pargs['cache_results'] = True
        
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
        
        prefactor = 1/( sigma*sqrt(2*pi) )
        
        for i in range(num_points):
            delta = radius-R
            wt = prefactor*exp( - (delta**2)/(2*( sigma**2 ) ) )
            
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
        

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        found, v = self.get_cache_form_factor(qx, qy, qz)
        if found:
            return v
        
        v = 0.0
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor(qx, qy, qz)
            v += wt*v_R*dR

        self.set_cache_form_factor(qx, qy, qz, v)
        
        return v


    def form_factor_array(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""

        v = numpy.zeros(len(qx))
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_array(qx, qy, qz)
            v += wt*v_R*dR

        return v
        
        
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
        
        v = numpy.zeros(len(q_list))
        for R, dR, wt, curNanoObject in self.distribution():
            v_R = curNanoObject.form_factor_intensity_isotropic_array(q_list, num_phi=num_phi, num_theta=num_theta)
            v += wt*v_R*dR

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
            G = G.real/(4*pi)
            
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
            G = G.real/(4*pi)
            
        if self.pargs['cache_results']:
            self.beta_numerator_array_already_computed_qlist = q_list
            self.beta_numerator_array_already_computed = G
        
        return G
            
    
    def beta_numerator_iso_external(self, q, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """

        phi_vals, dphi = numpy.linspace( 0, 2*pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
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
        
        G = numpy.zeros(len(q_list))
        
        for i, q in enumerate(q_list):
            G[i] = self.beta_numerator_iso_external(q, num_phi=num_phi, num_theta=num_theta)
            
        return G


    def beta_numerator_iso_external_array(self, q_list, num_phi=50, num_theta=50):
        """Calculates the beta numerator under the assumption that the orientational
        averaging is done last. That is, instead of calculating |<<F>>_iso|^2, we
        calculate <|<F>|^2>_iso """
        
        phi_vals, dphi = numpy.linspace( 0, 2*pi, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
        G = numpy.zeros(len(q_list))
        
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
            beta = numpy.exp( -( (radius*sigma_R*q)**2 ) )
            return beta
        else:
            P, beta = self.P_beta( q, num_phi=num_phi, num_theta=num_theta )
            return beta
        
        
    def beta_ratio_array(self, q_list, num_phi=50, num_theta=50, approx=False):
        """Returns a 1D array of the beta ratio."""
        
        if approx:
            radius = self.pargs['radius']
            sigma_R = self.pargs['sigma_R']
            beta = numpy.exp( -( (radius*sigma_R*q_list)**2 ) )
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
            beta = numpy.exp( -( (radius*sigma_R*q)**2 ) )
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
            beta = numpy.exp( -( (radius*sigma_R*q_list)**2 ) )
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
        
        #self.rotation_matrix = numpy.identity(3)
        
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
        
        qx, qy, qz = self.rotate_q(qx, qy, qz)
        
        R = self.pargs['radius']
        volume = (2*R)**3
        
        F = self.pargs['delta_rho']*volume*sinc(qx*R)*sinc(qy*R)*sinc(qz*R)+0.0j

        return F


    def form_factor_array(self, qx, qy, qz):
        
        R = self.pargs['radius']
        volume = (2*R)**3
        
        F = numpy.zeros( (len(qx)), dtype=numpy.complex )
        
        qx, qy, qz = self.rotate_q_array(qx, qy, qz)
        F = self.pargs['delta_rho']*volume*numpy.sinc(qx*R/pi)*numpy.sinc(qy*R/pi)*numpy.sinc(qz*R/pi)
        
        return F


    def form_factor_isotropic_unoptimized(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation."""
        
        # TODO: This function is no longer necessary, and can be removed
        
        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi/2, num_theta, endpoint=False, retstep=True )
        
        
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
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi/2, num_theta, endpoint=False, retstep=True )
        
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
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi/2, num_theta, endpoint=False, retstep=True )
        
        R = self.pargs['radius']
        volume = (2*R)**3
        
        if q==0:
            return ( (self.pargs['delta_rho']*volume)**2 )*4*pi

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
        
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi/2, num_theta, endpoint=False, retstep=True )
        
        R = self.pargs['radius']
        volume = (2*R)**3
        
        prefactor = 128*( (self.pargs['delta_rho']*volume)**2 )/( (q_list*R)**6 )
        #prefactor = 16*64*( (self.pargs['delta_rho'])**2 )/( q**6 )

        P = numpy.zeros( len(q_list) )
        
        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:
                
                qz =  q_list*cos(theta)
                theta_part = (numpy.sin(qz*R))/(sin(2*theta))
                theta_part = dtheta*dphi*(theta_part**2)/(sin(theta))

                # Start computing partial values
                qy_partial =  q_list*sin(theta)
                for phi in phi_vals:

                    # When phi==0, the integrand is zero
                    if phi!=0:
                        qx = -qy_partial*cos(phi)
                        qy = qy_partial*sin(phi)
                        
                        phi_part = numpy.sin(qx*R)*numpy.sin(qy*R)/( sin(2*phi) )
                        
                        P += theta_part*( phi_part**2 )

        P *= prefactor
        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*pi
        
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
        
        #self.rotation_matrix = numpy.identity(3)
        
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
        X, Y, Z = numpy.mgrid[ -extent:+extent:size*1j , -extent:+extent:size*1j , -extent:+extent:size*1j ]
        self.realspace_box = numpy.where( ( numpy.power(numpy.abs(X),2.0*p) + numpy.power(numpy.abs(Y),2.0*p) + numpy.power(numpy.abs(Z),2.0*p) )<threshold, 1, 0 )
        
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
        x_vector, dx = numpy.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        exp_iqxx = numpy.exp( 1j*qx*x_vector )
        y_vector, dy = numpy.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        exp_iqyy = numpy.exp( 1j*qy*y_vector )
        z_vector, dz = numpy.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        exp_iqzz = numpy.exp( 1j*qz*z_vector )
        
        F_matrix = self.realspace_box*( exp_iqzz.reshape(size,1,1) )*( exp_iqyy.reshape(1,size,1) )*( exp_iqxx.reshape(1,1,size) )

        F = F_matrix.sum()
        
        #print( 'form_factor_numerical done.' )
        return self.pargs['delta_rho']*F*dx*dy*dz
    
            
    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        
        qx, qy, qz = self.rotate_q(qx, qy, qz)
        
        return self.form_factor_numerical(qx, qy, qz, num_points=100 )
                    

    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        
        if q in self.form_factor_intensity_isotropic_already_computed and self.pargs['cache_results']:
            return self.form_factor_intensity_isotropic_already_computed[q]
        
        # Because of symmetry, we only have to measure 1 of the 8 octants
        phi_vals, dphi = numpy.linspace( 0, pi/2.0, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi/2.0, num_theta, endpoint=False, retstep=True )

        size = self.pargs['num_points_realspace']
        extent = 1.1*self.pargs['radius'] # Size of the box that contains the shape
        x_vector, dx = numpy.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        y_vector, dy = numpy.linspace(-extent, +extent, size, endpoint=True, retstep=True)
        z_vector, dz = numpy.linspace(-extent, +extent, size, endpoint=True, retstep=True)

        prefactor = 8*( (self.pargs['delta_rho']*dx*dy*dz)**2 ) # Factor of eight accounts for only doing one octant

        
        P = 0.0
        
        for theta in theta_vals:
            # When theta==0, there is nothing to contribute to integral
            # (sin(theta)=0, so the whole integrand is zero).
            if theta!=0:
            
                qz =  q*cos(theta)
                exp_iqzz = numpy.exp( 1j*qz*z_vector ).reshape(size,1,1)
                F_matrix_partial = self.realspace_box*( exp_iqzz )
                dS = sin(theta)*dtheta*dphi
                
                for phi in phi_vals:
                    # When phi==0, the integrand is zero
                    if phi!=0:
                        
                        qx = -q*sin(theta)*cos(phi)
                        exp_iqxx = numpy.exp( 1j*qx*x_vector ).reshape(1,1,size)
                        qy =  q*sin(theta)*sin(phi)
                        exp_iqyy = numpy.exp( 1j*qy*y_vector ).reshape(1,size,1)
                        
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
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi/2, num_theta, endpoint=False, retstep=True )
        
        
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
###################################################################
class SphereNanoObject(NanoObject):

    

    def __init__(self, pargs={}, seed=None):
        self.rotation_matrix = numpy.identity(3)
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                    }
        
        
        self.pargs['cache_results'] = True
        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )            
        
        if 'radius' not in self.pargs:
            # Set a default size
            self.pargs['radius'] = 1.0
            
            
        # Store values we compute to avoid laborious re-computation
        self.form_factor_already_computed = {}
        self.form_factor_intensity_isotropic_already_computed = {}


    def V(self, in_x, in_y, in_z, rotation_elements=None):
        """Returns the intensity of the real-space potential at the
        given real-space coordinates."""
        
        R = self.pargs['radius']
        r = sqrt( in_x**2 + in_y**2 + in_z**2 )
        
        if r<R:
            return 1.0
        else:
            return 0.0


    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        
        q = sqrt( qx**2 + qy**2 + qz**2 )
        
        if q in self.form_factor_already_computed and self.pargs['cache_results']:
            return self.form_factor_already_computed[q]
        
        R = self.pargs['radius']
        qR = q*R
        volume = (4.0/3.0)*pi*(R**3)

        if q==0:
            return self.pargs['delta_rho']*volume


        F = 3.0*self.pargs['delta_rho']*volume*( ( sin(qR)-qR*cos(qR) )/( qR**3 ) ) + 0.0j
        
        if self.pargs['cache_results']:
            self.form_factor_already_computed[q] = F
        
        return F


    def form_factor_array(self, qx, qy, qz):

        R = self.pargs['radius']
        volume = (4.0/3.0)*pi*(R**3)
        qR = R*numpy.sqrt( qx**2 + qy**2 + qz**2 )

        F = numpy.zeros( (len(qx)), dtype=numpy.complex )
        #for i, qxi in enumerate(qx):
            #F[i] = self.form_factor(qx[i], qy[i], qz[i])

        F = 3.0*self.pargs['delta_rho']*volume*( numpy.sin(qR) - qR*numpy.cos(qR) )/( qR**3 )

        return F


    def form_factor_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the particle form factor, averaged over every possible orientation."""

        return 4*pi * self.form_factor( q, 0, 0)


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        if q in self.form_factor_intensity_isotropic_already_computed and self.pargs['cache_results']:
            return self.form_factor_intensity_isotropic_already_computed[q]

        R = self.pargs['radius']
        qR = q*R
        volume = (4.0/3.0)*pi*(R**3)

        if q==0:
            return 4*pi*( (self.pargs['delta_rho']*volume)**2 )

        prefactor = 36*pi*( (self.pargs['delta_rho']*volume)**2 )/(qR**6)

        P = prefactor*( (sin(qR)-qR*cos(qR))**2 )
        
        if self.pargs['cache_results']:
            self.form_factor_intensity_isotropic_already_computed[q] = P
        
        return P


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        
        R = self.pargs['radius']
        
        volume = (4.0/3.0)*pi*(R**3)

        prefactor = 36*pi*( (self.pargs['delta_rho']*volume)**2 )


        P = numpy.empty( (len(q_list)) )
        for i, q in enumerate(q_list):
            if q==0:
                P[i] = 4*pi*( (self.pargs['delta_rho']*volume)**2 )
            else:
                P[i] = prefactor*( (sin(q*R)-q*R*cos(q*R))**2 )/((q*R)**6)

        return P


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
        
        texture_num = int(self.pargs['rho1']*10)
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
        
        #self.rotation_matrix = numpy.identity(3)
        
        self.pargs = {   'rho_ambient': 0.0, \
                            'rho1': 15.0, \
                            'height': None, \
                            'pyramid_face_angle': None, \
                    }
        
        self.pargs.update(pargs)
        self.pargs['delta_rho'] = abs( self.pargs['rho_ambient'] - self.pargs['rho1'] )
        
        if self.pargs['height']==None and self.pargs['pyramid_face_angle']==None:
            # Assume user wants a 'regular' pyramid
            self.pargs['height'] = sqrt(2.0)*self.pargs['radius']
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
        
        # TODO: Don't ignore rotation elements
        
        R = self.pargs['radius']
        H = min( self.pargs['height'] , R/tan( radians(self.pargs['pyramid_face_angle']) ) )
        R_z = R - in_z/tan( radians(self.pargs['pyramid_face_angle']) )
        if in_z<H and in_z>0 and abs(in_x)<abs(R_z) and abs(in_y)<abs(R_z):
            return 1.0
        else:
            return 0.0


    def thresh_near_zero(self, value, threshold=1e-7):
        """Forces a near-zero value to be no less than the given threshold.
        This is part of a kludge to avoid numeric errors that occur when
        using values of q that are too small."""
        
        if abs(value)>threshold:
            return value
            
        if value>0:
            return +threshold
        else:
            return -threshold


    def thresh_near_zero_array(self, values, threshold=1e-7):
        
        idx = numpy.nonzero( abs(values)<threshold )
        values[idx] = numpy.sign(values[idx])*threshold
        
        # Catch values that are exactly zero
        idx = numpy.nonzero( values==0.0 )
        values[idx] = +threshold
        

    def form_factor(self, qx, qy, qz):
        """Returns the complex-amplitude of the form factor at the given
        q-coordinates."""
        
        qx, qy, qz = self.rotate_q(qx, qy, qz)
        

        #F = self.form_factor_numerical(qx, qy, qz, num_points=100, size_scale=None, rotation_elements=None)
        
        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = tan(radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )
        
        
        
        # NOTE: (partial kludge) The computation below will hit a divide-by-zero
        # if qx or qy are zero. Because F is smooth near the origin, we will obtain
        # the correct limiting value by using a small, but non-zero, value for qx/qy
        if qx==0 and qy==0 and qz==0:
            # F(0,0,0) = rho*V
            return self.pargs['delta_rho']*volume
        qx = self.thresh_near_zero(qx)
        qy = self.thresh_near_zero(qy)
        qz = self.thresh_near_zero(qz)
        
        q1 = 0.5*( (qx-qy)*amod + qz )
        q2 = 0.5*( (qx-qy)*amod - qz )
        q3 = 0.5*( (qx+qy)*amod + qz )
        q4 = 0.5*( (qx+qy)*amod - qz )

            
        K1 = numpy.sinc(q1*H/numpy.pi)*cexp( +1.0j * q1*H ) + numpy.sinc(q2*H/numpy.pi)*cexp( -1.0j * q2*H )
        K2 = -1.0j*numpy.sinc(q1*H/numpy.pi)*cexp( +1.0j * q1*H ) + 1.0j*numpy.sinc(q2*H/numpy.pi)*cexp( -1.0j * q2*H )
        K3 = numpy.sinc(q3*H/numpy.pi)*cexp( +1.0j * q3*H ) + numpy.sinc(q4*H/numpy.pi)*cexp( -1.0j * q4*H )
        K4 = -1.0j*numpy.sinc(q3*H/numpy.pi)*cexp( +1.0j * q3*H ) + 1.0j*numpy.sinc(q4*H/numpy.pi)*cexp( -1.0j * q4*H )
    
        F = (H/(qx*qy))*( K1*cos((qx-qy)*R) + K2*sin((qx-qy)*R) - K3*cos((qx+qy)*R) - K4*sin((qx+qy)*R) )
        F *= self.pargs['delta_rho']
        
        return F


    def form_factor_array(self, qx, qy, qz):
        
        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = tan(radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )
        
        qx, qy, qz = self.rotate_q_array(qx, qy, qz)

        
        # NOTE: (partial kludge) The computation below will hit a divide-by-zero
        # if qx or qy are zero. Because F is smooth near the origin, we will obtain
        # the correct limiting value by using a small, but non-zero, value for qx/qy
        
        self.thresh_near_zero_array(qx)
        self.thresh_near_zero_array(qy)
        self.thresh_near_zero_array(qz)
        

        q1 = 0.5*( (qx-qy)*amod + qz )
        q2 = 0.5*( (qx-qy)*amod - qz )
        q3 = 0.5*( (qx+qy)*amod + qz )
        q4 = 0.5*( (qx+qy)*amod - qz )
        K1 = numpy.sinc(q1*H/numpy.pi)*numpy.exp( +1.0j * q1*H ) + numpy.sinc(q2*H/numpy.pi)*numpy.exp( -1.0j * q2*H )
        K2 = -1.0j*numpy.sinc(q1*H/numpy.pi)*numpy.exp( +1.0j * q1*H ) + 1.0j*numpy.sinc(q2*H/numpy.pi)*numpy.exp( -1.0j * q2*H )
        K3 = numpy.sinc(q3*H/numpy.pi)*numpy.exp( +1.0j * q3*H ) + numpy.sinc(q4*H/numpy.pi)*numpy.exp( -1.0j * q4*H )
        K4 = -1.0j*numpy.sinc(q3*H/numpy.pi)*numpy.exp( +1.0j * q3*H ) + 1.0j*numpy.sinc(q4*H/numpy.pi)*numpy.exp( -1.0j * q4*H )

        
        F = (H/(qx*qy))*( K1*numpy.cos((qx-qy)*R) + K2*numpy.sin((qx-qy)*R) - K3*numpy.cos((qx+qy)*R) - K4*numpy.sin((qx+qy)*R) )
        F *= self.pargs['delta_rho']
        
        
        return F


    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""

        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = tan(radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )
        
        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
        
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


    def form_factor_intensity_isotropic_array(self, q_list, num_phi=70, num_theta=70):
        """Returns the intensity of the form factor, under the assumption
        of random orientation of the polyhedron. In other words, we
        average over every possible orientation. This value is denoted
        by P(q)"""
        
        R = self.pargs['radius']
        H = self.pargs['height']
        tan_alpha = tan(radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = (4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )
        
        # Note that we only integrate one of the 4 quadrants, since they are all identical
        # (we later multiply by 4 to compensate)
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True ) # Integral from +z-axis to -z-axis
        
        
        P = numpy.zeros( len(q_list) )
        
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
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*pi
            

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
        
        # TODO: This makes assumptions about there being no rotation
        
        return super(OctahedronNanoObject, self).V( in_x, in_y, abs(in_z), rotation_elements=rotation_elements )
        


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


        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True )
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )
        
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
        tan_alpha = tan(radians(self.pargs['pyramid_face_angle']))
        amod = 1.0/tan_alpha
        volume = 2*(4.0/3.0)*tan_alpha*( R**3 - (R - H/tan_alpha)**3 )
        
        
        phi_vals, dphi = numpy.linspace( 0, pi/2, num_phi, endpoint=False, retstep=True ) # In-plane integral
        theta_vals, dtheta = numpy.linspace( 0, pi, num_theta, endpoint=False, retstep=True )  # Integral from +z-axis to -z-axis
        
        
        P = numpy.zeros( len(q_list) )
        
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
                    
                    P += 4 * F*F.conjugate() * theta_part
                
                        
        if q_list[0]==0:
            P[0] = ( (self.pargs['delta_rho']*volume)**2 )*4*pi

        return P



# OctahedronPolydisperseNanoObject
###################################################################
class OctahedronPolydisperseNanoObject(PolydisperseNanoObject):

    def __init__(self, pargs={}, seed=None):
        
        PolydisperseNanoObject.__init__(self, OctahedronNanoObject, pargs=pargs, seed=seed)





    
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
            self.gamma_nu = sqrt(pi)*Gamma( (self.nu+1)/2 )/Gamma( self.nu/2 )
            
        self.already_computed = {}

    def gaussian(self, sigma=None, delta=None, fwhm=None):
        """Sets the peak to be a pure Gaussian (overrides any other setting)."""
        
        self.nu = self.infinity
        self.already_computed = {}
        
        if sigma==None and delta==None and fwhm==None:
            print( "WARNING: No width specified for Gaussian peak. A width has been assumed." )
            self.sigma = 0.1
            self.delta = sqrt(8/pi)*self.sigma
            self.fwhm = 2*sqrt(2*log(2))*self.sigma
        elif sigma!=None:
            # Sigma takes priority
            self.sigma = sigma
            self.delta = sqrt(8/pi)*self.sigma
            self.fwhm = 2*sqrt(2*log(2))*self.sigma
        elif fwhm!=None:
            self.fwhm = fwhm
            self.sigma = self.fwhm/( 2*sqrt(2*log(2)) )
            self.delta = sqrt(8/pi)*self.sigma
        else:
            # Use delta to define peak width
            self.delta = delta
            self.sigma = sqrt(pi/8)*self.delta
            self.fwhm = 2*sqrt(2*log(2))*self.sigma
            
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
        if qs in self.already_computed:
            return self.already_computed[qs]
        
        if self.nu>self.gauss_cutoff:
            # Gaussian
            val = (2/(pi*self.delta))*exp( -(4*(qs**2))/(pi*(self.delta**2)) )
        elif self.nu<self.lorentz_cutoff:
            # Lorentzian
            val = (self.delta/(2*pi))/(qs**2 + ((self.delta/2)**2) )
        else:
            # Brute-force the term 
            val = (2/(pi*self.delta))
            
            if self.gamma_method:
                
                print( "WARNING: The gamma method does not currently work." )
                
                # Use gamma functions
                y = (4*(qs**2))/( (pi**2) * (self.delta**2) ) 
                
                # Note that this equivalence comes from the paper:
                #   Scattering Curves of Ordered Mesoscopic Materials
                #   S. Förster, A. Timmann, M. Konrad, C. Schellbach, A. Meyer, S.S. Funari, P. Mulvaney, R. Knott,
                #   J. Phys. Chem. B, 2005, 109 (4), pp 1347–1360 DOI: 10.1021/jp0467494 
                #   (See equation 27 and last section of Appendix.)
                # However there seems to be a typo in the paper, since it does not match the brute-force product.
                
                numerator = GammaComplex( (self.nu/2) + 1.0j*self.gamma_nu*y )
                #numerator = GammaComplex( (self.nu/2) + 1.0j*self.gamma_nu*(sqrt(y)) )
                denominator = GammaComplex( self.nu/2 )
                term = numerator/denominator
                
                val *= 0.9*term*term.conjugate()
            
            else:
                # Use a brute-force product calculation
                for n in range(0, self.product_terms):
                    #print n, self.nu, self.gamma_nu
                    term1 = (self.gamma_nu**2)/( (n+self.nu/2)**2 )
                    #print "  " + str(term1)
                    term2 = (4*(qs**2))/( (pi**2) * (self.delta**2) )
                    val *= 1/(1+term1*term2)
                
            
        self.already_computed[qs] = val
        
        return val

    def val_array(self, q_list, q_center):
        """Returns the height of the peak for the given array of positions, under the
        assumption of a peak centered about q_center."""

        val = numpy.empty( (len(q_list)) )
        for i, q in enumerate(q_list):
            val[i] = self.val(abs(q-q_center))

        return val
        
        
    def plot(self, plot_width=1.0, num_points=200, filename='peak.png', ylog=False):
        
        q_list = numpy.linspace( -plot_width, plot_width, num_points )
        int_list = self.val_array( q_list, 0.0 )

        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)
        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( 'Intensity (a.u.)', size=30 )        
        
        pylab.savefig( filename )

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
        background = numpy.empty( (len(q_list)) )
        for i, q in enumerate(q_list):
            background[i] = self.val(q)
        return background


# Lattice
###################################################################
class Lattice(object):
    """Defines a lattice type and provides methods for adding objects to the lattice."""
    
    
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
    
    
        self.sigma_D = sigma_D          # Lattice disorder
        
        
        
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
        self.sigma_D = sigma_D


    def object_count(self, objects):
        if len(objects)<self.min_objects:
            print( "WARNING: %s received only %d objects in enumeration, but requires at least %d." % (self.__class__.__name__, len(objects), self.min_objects) )
            exit()
        
    def unit_cell_volume(self):
        V = sqrt( 1 - (cos(self.alpha))**2 - (cos(self.beta))**2 - (cos(self.gamma))**2 + 2*cos(self.alpha)*cos(self.beta)*cos(self.gamma) )
        V *= self.lattice_spacing_a*self.lattice_spacing_b*self.lattice_spacing_c
        
        return V
    
    
    
    # Components of intensity calculation
    ########################################
    
    def iterate_over_objects(self):
        """Returns a sequence of the distinct particle/object
        types in the unit cell. It thus defines the unit cell."""
        
        # r will contain the return value, an array with rows that contain:
        # number, position in unit cell, relative coordinates in unit cell, object
        r = []
        
        for i, pos in enumerate(self.lattice_positions):
            xi, yi, zi = self.lattice_coordinates[i]
            obj = self.lattice_objects[i]
            r.append( [i, pos, xi, yi, zi, obj] )
        
        return r


    def multiplicity_lookup(self, h, k, l):
        """Returns the peak multiplicity for the given reflection."""
        return 1

    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""
        
        # NOTE: This is assuming cubic/rectangular only!
        qhkl_vector = ( 2*pi*h/(self.lattice_spacing_a), \
                        2*pi*k/(self.lattice_spacing_b), \
                        2*pi*l/(self.lattice_spacing_c) ) 
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        qhkl, qhkl_vector = self.q_hkl(h,k,l)
        
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
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
            for q in sorted(hkl_list_1d.iterkeys()):
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
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            e_term = cexp( 2*pi * 1j * ( xi*h + yi*k + zi*l ) )
            
            # TODO: doublecheck rotation
            # (rotate qhkl based on the orientation of the particle in the unit cell; right now
            # this is being handled by setting orientation angles to the particular NanoObjects)
            qx, qy, qz = qhkl_vector

            summation += e_term*obj.form_factor( qx, qy, qz )
            #summation += e_term*obj.form_factor_isotropic( numpy.sqrt(qx**2+qy**2+qz**2) ) # Make the object isotropic
            #summation += e_term*obj.form_factor_orientation_spread( numpy.sqrt(qx**2+qy**2+qz**2) ) # Spread intensity
            
        
        return summation


    def sum_over_hkl(self, q, peak, max_hkl=6):
        
        
        summation = 0
        
        for h, k, l, m, f, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            
            fs = self.sum_over_objects(qhkl_vector, h, k, l)
            term1 = fs*fs.conjugate()
            term2 = exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
            term3 = peak.val( q-qhkl )
            
            summation += (m*(f**2)) * term1.real * term2 * term3
        
        return summation


    def sum_over_hkl_array(self, q_list, peak, max_hkl=6):
        
        
        summation = numpy.zeros( (len(q_list)) )
        
        for h, k, l, m, f, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):
            
            fs = self.sum_over_objects(qhkl_vector, h, k, l)
            term1 = fs*fs.conjugate()
            term2 = exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
            
            summation += (m*(f**2)) * term1.real * term2 * peak.val_array( q_list, qhkl )
        
        return summation


    def sum_over_hkl_cache(self, q_list, peak, max_hkl=6):
        
        
        hkl_list = self.iterate_over_hkl(max_hkl=max_hkl)
        
        if self.lattice_cache==[]:
            # Recalculate the lattice part
            self.lattice_cache = numpy.zeros( (len(hkl_list),len(q_list)) )
            
            i_hkl = 0
            for h, k, l, m, f, qhkl, qhkl_vector in hkl_list:
                fs = self.sum_over_objects(qhkl_vector, h, k, l)
                term1 = fs*fs.conjugate()
                self.lattice_cache[i_hkl] = term1.real
                i_hkl += 1
            
        if self.peak_cache==[]:
            # Recalculate the peak part (DW and peak shape)
            self.peak_cache = numpy.zeros( (len(hkl_list),len(q_list)) )

            i_hkl = 0
            for h, k, l, m, f, qhkl, qhkl_vector in hkl_list:
                term2 = exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
                self.peak_cache[i_hkl] = term2 * peak.val_array( q_list, qhkl )
                i_hkl += 1
        
        summation = numpy.zeros( (len(q_list)) )
        i_hkl = 0
        for h, k, l, m, f, qhkl, qhkl_vector in hkl_list:
            summation += (m*(f**2)) * self.lattice_cache[i_hkl] * self.peak_cache[i_hkl] 
            i_hkl += 1
        
        return summation


    # Form factor computations
    ########################################
    
    def form_factor_intensity_isotropic(self, q, num_phi=50, num_theta=50):
        """Returns the isotropic form factor for the lattice. This is the 
        scattering intensity one would measure if the component particles were
        freely dispersed, and thus randomly oriented in solution."""
        
        # Compute P(q) by summing each object's P(q)
        P = 0
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            P += obj.form_factor_intensity_isotropic(q, num_phi=num_phi, num_theta=num_theta)

        return P

    def form_factor_intensity_isotropic_array(self, q_list, num_phi=50, num_theta=50):
        
        # Compute P(q) by summing each object's P(q) (form_factor_intensity_isotropic)
        P = numpy.zeros( (len(q_list)) )
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
        
        G = numpy.zeros( (len(q_list)) )
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
            beta = numpy.zeros( (len(q_list)) )
            n = 0
            for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
                beta += obj.beta_ratio_array(q_list, num_phi=num_phi, num_theta=num_theta, approx=True)
                n += 1
            beta = beta/n
        else:
            P, beta = self.P_beta_array(q_list, num_phi=num_phi, num_theta=num_theta)
        
        return beta


    def P_beta(self, q, num_phi=50, num_theta=50, approx=False):

        P = self.form_factor_intensity_isotropic_array( q_list, num_phi=num_phi, num_theta=num_theta)
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
        
        if background==None:
            return S
        else:
            return S + background.val_array(q_list)/P


    def intensity(self, q, peak, c=1.0, background=None, max_hkl=6):
        """Returns the predicted scattering intensity."""
        
        
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
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        S_list = self.structure_factor_isotropic_array( q_list, peak, c=c, background=background, max_hkl=max_hkl )


        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)

        
        pylab.plot( q_list, S_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
            fig.subplots_adjust(left=0.15)
        else:
            # Make y-axis scientific notation
            fig.gca().yaxis.major.formatter.set_scientific(True)
            fig.gca().yaxis.major.formatter.set_powerlimits((3,3))
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
        pylab.ylabel( r'$S(q)$' )        

        #xi, xf, yi, yf = pylab.axis()
        #pylab.axis( [xi, xf, yi, yf] )

        pylab.savefig( filename )
        
        return S_list
            
    
    
    def plot_intensity(self, qtuple, peak, filename='intensity.png', c=1.0, background=None, max_hkl=6, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        
        int_list = self.intensity_array( q_list, peak, c=c, background=background, max_hkl=max_hkl )
        

        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)
        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=2.0 )
        
        if ylog:
            pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( 'Intensity (a.u.)', size=30 )        
        
        #xi, xf, yi, yf = pylab.axis()
        #pylab.axis( [xi, xf, yi, yf] )
        
        pylab.savefig( filename )
        
        return int_list
            

    def plot_form_factor_intensity_isotropic(self, qtuple, filename='form_factor_isotropic.png', ylog=False):
        """Outputs a plot of the P(q) vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        int_list = self.form_factor_intensity_isotropic_array( q_list, num_phi=50, num_theta=50)
        

        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.17, bottom=0.15, right=0.94, top=0.94)
        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( r'$P(q)$', size=30 )        
        
        #xi, xf, yi, yf = pylab.axis()
        #pylab.axis( [xi, xf, yi, yf] )
        
        pylab.savefig( filename )
        
        return int_list
         
         
    def plot_beta_ratio(self, qtuple, filename='beta_ratio.png', ylog=False):
        """Outputs a plot of the beta ratio vs. q data. Also returns an array
        of the intensity values.
        qtuple - (q_initial, q_final, num_q)
        """
        (q_initial, q_final, num_q) = qtuple
        # Get data
        q_list = numpy.linspace( q_initial, q_final, num_q, endpoint=True )
        
        int_list = self.beta_ratio_array( q_list )
        

        pylab.rcParams['axes.labelsize'] = 30
        pylab.rcParams['xtick.labelsize'] = 'xx-large'
        pylab.rcParams['ytick.labelsize'] = 'xx-large'
        fig = pylab.figure()
        fig.subplots_adjust(left=0.14, bottom=0.15, right=0.94, top=0.94)
        
        pylab.plot( q_list, int_list, color=(0,0,0), linewidth=3.0 )
        
        if ylog:
            pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( r'$\beta(q)$', size=30 )        
        
        if not ylog:
            xi, xf, yi, yf = pylab.axis()
            yi = 0.0
            yf = 1.05
            pylab.axis( [xi, xf, yi, yf] )
        
        pylab.savefig( filename )
        
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
        s += "                         = (%.2f,%.2f,%.2f) in degrees\n" % (degrees(self.alpha),degrees(self.beta),degrees(self.gamma))
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
    
    
        self.sigma_D = sigma_D          # Lattice disorder
        
        
        
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
        qhkl_vector = ( 2*pi*h/(self.lattice_spacing_a), \
                        2*pi*(h+2*k)/(numpy.sqrt(3)*self.lattice_spacing_b), \
                        2*pi*l/(self.lattice_spacing_c) ) 
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return (qhkl, qhkl_vector)
        
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        prefactor = (2*pi/self.lattice_spacing_a)
        qhkl_vector = ( prefactor*h, \
                        prefactor*k, \
                        prefactor*l ) 
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        prefactor = (2*pi/self.lattice_spacing_a)
        qhkl = prefactor*sqrt( h**2 + k**2 + l**2 )
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
                    
                    thdisp = numpy.radians( random.uniform(-angle_jitter,+angle_jitter) )
                    phdisp = numpy.radians( random.uniform(-angle_jitter,+angle_jitter) )

                    self.lattice_positions.append( 'corner-of-subcell' )
                    self.lattice_coordinates.append( ( xi+xdisp , yi+ydisp , zi+zdisp ) )
                    self.lattice_objects.append( self.objects[0] )

                    
                    
                    # Center particle is reoriented w.r.t. to the corner particle
                    xu = 0.5
                    yu = 0.5
                    zu = 0.5
                    # Rotate about x
                    xu = xu*1
                    yu = 0 + yu*numpy.cos(thdisp) - zu*numpy.sin(thdisp)
                    zu = 0 + yu*numpy.sin(thdisp) + zu*numpy.cos(thdisp)
                    # Rotate about z
                    xu = xu*numpy.cos(thdisp) - yu*numpy.sin(thdisp) + 0
                    yu = xu*numpy.sin(thdisp) + yu*numpy.cos(thdisp) + 0
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
                        yu = 0 + yu*numpy.cos(thdisp) - zu*numpy.sin(thdisp)
                        zu = 0 + yu*numpy.sin(thdisp) + zu*numpy.cos(thdisp)
                        # Rotate about z
                        xu = xu*numpy.cos(thdisp) - yu*numpy.sin(thdisp) + 0
                        yu = xu*numpy.sin(thdisp) + yu*numpy.cos(thdisp) + 0
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
                        twist_radians = ix*numpy.radians(twist)
                    else:
                        twist_radians = (repeat-1-ix)*numpy.radians(twist)
                    xc = xi
                    yc = yi*numpy.cos(twist_radians) - zi*numpy.sin(twist_radians)
                    zc = yi*numpy.sin(twist_radians) + zi*numpy.cos(twist_radians)

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
                    yc = yc*numpy.cos(twist_radians) - zc*numpy.sin(twist_radians)
                    zc = yc*numpy.sin(twist_radians) + zc*numpy.cos(twist_radians)
                    
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
                        yu = 0 + yu*numpy.cos(thdisp) - zu*numpy.sin(thdisp)
                        zu = 0 + yu*numpy.sin(thdisp) + zu*numpy.cos(thdisp)
                        # Rotate about z
                        xu = xu*numpy.cos(thdisp) - yu*numpy.sin(thdisp) + 0
                        yu = xu*numpy.sin(thdisp) + yu*numpy.cos(thdisp) + 0
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        prefactor = (2*pi/self.lattice_spacing_a)
        qhkl_vector = ( prefactor*h, \
                        prefactor*k, \
                        prefactor*l ) 
        qhkl = sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )
        
        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        prefactor = (2*pi/self.lattice_spacing_a)
        qhkl = prefactor*sqrt( h**2 + k**2 + l**2 )
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
            self.lattice_spacing_c = lattice_spacing_a*( 4.0/numpy.sqrt(6) )
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
    
    
        self.sigma_D = sigma_D          # Lattice disorder
        
        
        
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
            self.lattice_spacing_c = lattice_spacing_a*( 4.0/numpy.sqrt(6) )
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
    
    
        self.sigma_D = sigma_D          # Lattice disorder
        
        
        
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
            self.lattice_spacing_c = lattice_spacing_a*( 4.0/numpy.sqrt(6) )
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
    
    
        self.sigma_D = sigma_D          # Lattice disorder
        
        
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        self.sigma_D = sigma_D          # Lattice disorder
        
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
        
        q = sqrt( qx**2 + qy**2 + qz**2 )

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
                    G = numpy.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = numpy.ones( (len(q_list)) ) - beta*G
                else:
                    diffuse = numpy.zeros( (len(q_list)) )
                
            else:

                if self.margs['diffuse']:
                    P, beta = self.lattice.P_beta_array( q_list, approx=self.margs['beta_approx'] )
                    G = numpy.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = numpy.ones( (len(q_list)) ) - beta*G
                else:
                    P = self.lattice.form_factor_intensity_isotropic_array(q_list)
                    diffuse = numpy.zeros( (len(q_list)) )
                
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
                    G = numpy.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = numpy.ones( (len(q_list)) ) - beta*G
                else:
                    diffuse = numpy.zeros( (len(q_list)) )
                
            else:

                if self.margs['diffuse']:
                    P, beta = self.lattice.P_beta_array( q_list, approx=self.margs['beta_approx'] )
                    G = numpy.exp( -( (self.lattice.sigma_D*self.lattice.lattice_spacing_a*q_list)**2 ) )
                    diffuse = numpy.ones( (len(q_list)) ) - beta*G
                else:
                    P = self.lattice.form_factor_intensity_isotropic_array(q_list)
                    diffuse = numpy.zeros( (len(q_list)) )
                
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
            self.q_list = numpy.array( self.data.q_vals )[self.index_start:self.index_end]
            self.int_list = self.data.intensity_vals[self.index_start:self.index_end]
            
        elif self.fargs['ptype']=='form_factor':
            self.set_data_indices( self.data.q_ff_vals, q_start=q_start, q_end=q_end, index_start=index_start, index_end=index_end )
            self.q_list = numpy.array( self.data.q_ff_vals )[self.index_start:self.index_end]
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
        
        q_list = numpy.linspace( q_start, q_end, num_q )
        
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
            #q_fit = numpy.linspace( q_list[0]*0.01, q_list[-1]*1.5, 2000 ) # Extended q-range
            int_fit = self.model.q_array(q_fit)

            if show_extended:
                # Create extended fit
                q_fit_extended = numpy.linspace( q_list[0]*0.5, q_list[-1]*1.1, 400 )
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
                q_fit_extended = numpy.linspace( q_list[0]*0.5, q_list[-1]*1.1, 200 )
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
        pylab.rcParams['axes.labelsize'] = axes_font_size # Not used (override)
        pylab.rcParams['xtick.labelsize'] = 'x-large'
        pylab.rcParams['ytick.labelsize'] = 'x-large'
        
        if bigger_axes_fonts:
            axes_font_size = 35
            pylab.rcParams['xtick.labelsize'] = 25
            pylab.rcParams['ytick.labelsize'] = 25
        
        
        
        fig = pylab.figure()
        if bigger_axes_fonts:
            ax1 = fig.add_axes( [0.15,0.18,0.94-0.15,0.94-0.15-res_fig_height] )
        else:
            ax1 = fig.add_axes( [0.15,0.15,0.94-0.15,0.94-0.15-res_fig_height] )
        
        if err_vals != []:
            pylab.errorbar( q_list, int_list, yerr=err_vals, fmt='o', color=(0,0,0), ecolor=(0.5,0.5,0.5), markersize=6.0 )
        else:
            pylab.plot( q_list, int_list, 'o', color=(0,0,0), markersize=6.0 )
        
        if xlog and ylog:
            pylab.loglog()
        else:
            if xlog:
                pylab.semilogx()
            if ylog:
                pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=axes_font_size )
        if ptype=='form_factor':
            pylab.ylabel( r'$P(q)$', size=axes_font_size )        
        elif ptype=='structure_factor':
            pylab.ylabel( r'$S(q)$', size=axes_font_size )        
        else:
            pylab.ylabel( 'Intensity (a.u.)', size=axes_font_size )        

        # Get axis scaling
        xi, xf, yi, yf = pylab.axis()

        
        if show_extended:
            pylab.plot( q_fit_extended, int_fit_extended, '-', color=(0.5,0.5,1), linewidth=1.0 )
        pylab.plot( q_fit, int_fit, '-', color=(0,0,1), linewidth=2.0 )
        

        
        if show_indexing:
            list1d = self.model.lattice.iterate_over_hkl_1d()
            
            x_peak_indexing = []
            y_peak_indexing = []
            labels_indexing = []
            
            for q in sorted(list1d.iterkeys()):
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
            y_peak_indexing = numpy.asarray( y_peak_indexing )*rescaling
                    
            markerline, stemlines, baseline = pylab.stem( x_peak_indexing, y_peak_indexing )
            pylab.setp(markerline, 'markersize', 4.0)
            for xc, yc, lc in zip(x_peak_indexing, y_peak_indexing, labels_indexing):
                pylab.text( xc, yc+q0_intensity*0.015, lc, size=8, verticalalignment='bottom', horizontalalignment='center' )

        # Set axis scaling
        if scaling==None:
            # (We want the experimental data centered, not the fit.)
            xi = qi
            if qf!=None:
                xf = qf
            pylab.axis( [xi, xf, yi, yf] )
        else:
            pylab.axis( scaling )
        

        # Get ax1 final scaling
        x1i, x1f, y1i, y1f = pylab.axis()
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
        
        pylab.plot( self.q_list, residuals, '-', color='b', linewidth=2.0 )

        
        
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
        pylab.figtext( 0.94-0.01, 0.94-res_fig_height-0.01, chi_str, horizontalalignment='right', verticalalignment='top', size=20 )


        # Title
        if title!='':
            DW = self.model.lattice.sigma_D
            title = title + ' (%s, DW = %.2f)' % (self.model.lattice.__class__.__name__, DW)
            pylab.figtext( 0.06, 0.95, title, size=14, horizontalalignment='left', verticalalignment='bottom' )


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
                    
            
            
            pylab.plot( q_list, residuals_extended, '-', color='b', linewidth=1.0 )        
        

        if err_vals != []:
            # Create a filled-in region
            q_list_backwards = [ q_list[i] for i in range(len(q_list)-1,-1,-1) ]
            if self.fargs['chi']=='absolute':
                upper2 = [ +2.0*err_vals[i] for i in range(len(err_vals)) ]
                lower2 = [ -2.0*err_vals[i] for i in range(len(err_vals)-1,-1,-1) ]
                upper1 = [ +1.0*err_vals[i] for i in range(len(err_vals)) ]
                lower1 = [ -1.0*err_vals[i] for i in range(len(err_vals)-1,-1,-1) ]
            elif self.fargs['chi']=='relative':
                upper2 = [ +2.0*err_vals[i]/int_list[i] for i in range(len(err_vals)) ]
                lower2 = [ -2.0*err_vals[i]/int_list[i] for i in range(len(err_vals)-1,-1,-1) ]
                upper1 = [ +1.0*err_vals[i]/int_list[i] for i in range(len(err_vals)) ]
                lower1 = [ -1.0*err_vals[i]/int_list[i] for i in range(len(err_vals)-1,-1,-1) ]
            else:
                print( "Error: Unknown type of 'chi' in plot()." )
            
            ax2.fill( q_list+q_list_backwards, upper2+lower2, edgecolor='0.92', facecolor='0.92' )
            ax2.fill( q_list+q_list_backwards, upper1+lower1, edgecolor='0.82', facecolor='0.82' )
        
        
        pylab.axhline( 0, color='k' )
        ax2.axis( [xi, xf, yi, yf] )
        
        ax2.set_xticklabels( [] )
        ax2.set_yticklabels( [] )


        if dpi==None:
            pylab.savefig( filename )
        else:
            pylab.savefig( filename, dpi=dpi )
        if interact:
            pylab.show()
        
        
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
    
        
        
        
        
