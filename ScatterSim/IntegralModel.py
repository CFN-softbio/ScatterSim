# -*- coding: utf-8 -*-
###################################################################
# IntegralModel.py
# version 0.0.5
# April 12, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# Description:
#  Implements the "Integral GISAXS" model formalism, wherein the
# z-average film potential profile (e.g. the z density distribution
# deduced from reflectivity) is used as a reference function for
# calculating off-specular scattering (a.k.a. grazing-incdence
# scattering).
###################################################################


from BaseClasses import *  

class IntegralGISAXSModel(Model):
    """Implements the Integral GISAXS formalism for computing
    reciprocal-space intensity based on a particular real-space
    potential."""


    # Model arguments
    ################################
    # reflection_computation        Determines whether or not to perform a 'reflection' calculation.
    #                               The default behavior is to only calculate off-specular intensity,
    #                               with the specular rod thus being essentially zero. When this is
    #                               set to True, the code will instead calculate using the full real-
    #                               space potential, which includes specular reflection-mode effects.
    
    
    # Class variables
    ################################
    # These book-keeping values allow us to re-use pre-computed
    # arrays. The values are set to False when invalidated (e.g.
    # when the incident angle changes), which will force them to
    # be re-computed when needed.
    z_integral_base_uptodate = False
    z_integral_uptodate = False
    stored_theta_incident = 0.0
    stored_theta_scan = 0.0
    
    
    # Setup methods
    ################################
    def __init__(self, Box, Experimental, margs={}):
        """Prepares the model object."""
        self.margs.update(margs)
        
        self.Experimental = Experimental
        
        self.set_Box(Box)
        
        
    def set_Box(self, Box):
        """Associates a particular real-space potential box
        with this model."""
        
        self.Box = Box
        
        self.z_integral_base = numpy.empty( self.Box.size(), numpy.complex )
        self.z_integral = numpy.empty( (self.Box.limits['x']['num'], self.Box.limits['y']['num']), numpy.complex )
        self.working_xy = numpy.empty( (self.Box.limits['x']['num'], self.Box.limits['y']['num']), numpy.complex )
        
        self.clear()

    
    # Computation/update methods
    ################################
    def clear(self):
        """Tells the object that it should recompute internal
        data (e.g. if the input Box has changed)."""
        self.z_integral_base_uptodate = False
        self.z_integral_uptodate = False
    
    def check_z_integral_base_uptodate(self, theta_incident):
        """Checks if the currently-stored data is re-useable.
        Specifically, if the already-computed data is for the
        same theta_incident, it can be reused as-is. If not,
        it is recalculated."""
        
        status_print( "Consider: " + str(self.stored_theta_incident) + "-" + str(theta_incident), depth=10 )
        
        if theta_incident!=self.stored_theta_incident or not self.z_integral_base_uptodate:
            
            status_print( "New theta_incident: " + str(degrees(theta_incident)) + " deg (" + str(theta_incident) + "rad)", depth=4 )
            
            q_i, TR = self.Box.Calculate_TR( theta_incident, k_wavevector=self.Experimental.k_wavevector )


            # Integrand is ( V(x,y,z)-V(z) )*Psi*Psi~*dz
            # (We defer Psi~ until later...)
            
            e_part = [ cexp( 1j*q_i[iz]*z ) for iz, z in self.Box.axis('z') ]
            Psi = TR[:,0,0]*e_part[:] + TR[:,1,0]/e_part[:]

            if self.margs['reflection_computation']:
                self.z_integral_base = (self.Box.V_box)*Psi*self.Box.dz()
            else:
                self.z_integral_base = (self.Box.V_box-self.Box.V_z_average())*Psi*self.Box.dz()
            
            self.stored_theta_incident = theta_incident
            self.z_integral_base_uptodate = True
            self.z_integral_uptodate = False
            

    def check_z_integral_uptodate(self, theta_scan):
        """Checks if the currently-stored data is re-useable."""
        
        if theta_scan!=self.stored_theta_scan or not self.z_integral_uptodate:
            
            status_print( "New theta_scan: " + str(degrees(theta_scan)) + " deg (" + str(theta_scan) + "rad)", depth=5  )

            q_s, TR = self.Box.Calculate_TR( theta_scan, k_wavevector=self.Experimental.k_wavevector )
            
            Psi_inverse = numpy.empty( (self.Box.limits['z']['num']), numpy.complex  )
            for iz, z in self.Box.axis('z'):
                e_part = cexp( 1j*q_s[iz]*z )
                Psi_inverse[iz] = TR[iz,0][0]*e_part + TR[iz,1][0]/e_part


            z_integral_base_temp = numpy.copy(self.z_integral_base) 
            z_integral_base_temp *= Psi_inverse 
            self.z_integral = numpy.sum( z_integral_base_temp, axis=2 )
            
            self.stored_theta_scan = theta_scan
            self.z_integral_uptodate = True


        

    # Reciprocal-space methods
    ################################
    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""

        print( "WARNING: Integral GISAXS Model doesn't support arbitrary points in reciprocal-space." )

        return 0.0

    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""

        # Everything below the sample horizon is "wrong"
        # For now we just suppress the intensity in this region
        if theta_scan < 0:
            #ReflectionIntensity /= 1E20
            return 0.0


        # Convert angle formalism
        # (including conversion from degrees to radians)
        theta_incident = radians(theta_incident)
        theta_scan = radians(theta_scan)
        phi_scan = pi - radians(phi_scan)
        
        
        # Compute sub-components of the integral...
        self.check_z_integral_base_uptodate(theta_incident)
        self.check_z_integral_uptodate(theta_scan)
        self.working_xy = numpy.copy(self.z_integral)

        prefactor = self.Experimental.prefactor(theta_incident, theta_scan)
        
        # Create working vectors that hold the integrand components
        x_component = self.Experimental.k_wavevector*( cos(theta_scan)*cos(phi_scan) + cos(theta_incident) )
        xv = numpy.empty( (self.Box.limits['x']['num'],1), numpy.complex )
        for ix, x in self.Box.axis('x'):
            xv[ix] = cexp( 1j*x_component*x ) * self.Box.limits['x']['step']
            
        y_component = self.Experimental.k_wavevector*cos(theta_scan)*sin(phi_scan)
        yv = numpy.empty( (1,self.Box.limits['y']['num']), numpy.complex )
        for iy, y in self.Box.axis('y'):
            yv[0,iy] = cexp( 1j*y_component*y ) * self.Box.limits['y']['step']


        # Multiply the integrand components through working_xy
        # Thus working_xy contains a 2D array of all the integrand
        # components we need to sum for the integral.
        self.working_xy *= yv              # for ix, x in axis('x'): working_xy[ix,:] *= yv
        self.working_xy *= xv              # for iy, y in axis('y'): working_xy[:,iy] *= xv
        
        ReflectionAmplitude = prefactor*( numpy.sum(self.working_xy) )
        ReflectionIntensity = ReflectionAmplitude*ReflectionAmplitude.conjugate()


        
        return ReflectionIntensity.real

    def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
        
        status_print( "Starting Angular 2D Calc...", depth=2 )
        
        detector_image = numpy.empty( (phi_scan_num, theta_scan_num), numpy.float )

        theta_values = numpy.linspace( theta_scan_min, theta_scan_max, num=theta_scan_num )
        
        phi_values = numpy.linspace( phi_scan_min, phi_scan_max, num=phi_scan_num )

        for itheta, theta_scan in enumerate( theta_values ):
            for iphi, phi_scan in enumerate( phi_values ):
                detector_image[iphi,itheta] = self.angular_value( theta_incident, theta_scan, phi_scan )
        
        return detector_image

    def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max,qz_num)):
        
        status_print( "Starting Ewald 2D Calc...", depth=2 )
        
        detector_image = numpy.empty( (qy_num, qz_num), numpy.float )
        
        qy_values = numpy.linspace( qy_min, qy_max, num=qy_num )
        qz_values = numpy.linspace( qz_min, qz_max, num=qz_num )

        # Convert q-values into angular values
        k = self.Experimental.k_wavevector
        phi_values = numpy.empty( (qy_num) )
        theta_values = numpy.empty( (qz_num) )
        
        for iphi, qy in zip( range(len(phi_values)), qy_values ):
            phi_values[iphi] = degrees( asin(qy/k) )

        for itheta, qz in zip( range(len(theta_values)), qz_values ):
            theta_values[itheta] = degrees( asin( (qz/k) - sin(radians(theta_incident)) ) )

        for itheta, theta_scan in enumerate( theta_values ):
            for iphi, phi_scan in enumerate( phi_values ):
                detector_image[iphi,itheta] = self.angular_value( theta_incident, theta_scan, phi_scan )
        
        return detector_image

    # Query methods
    ################################
    def to_string(self):
        """Returns a string describing the model."""
        s = "Integral GISAXS Model: Computes intensity based on a reflectivity-like layer-formalism."
        return s
           
    
