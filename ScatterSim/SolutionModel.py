# -*- coding: utf-8 -*-
###################################################################
# SolutionModel.py
# version 0.0.5
# September 10, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# Description:
#  Computes solution scattering curves.
###################################################################
# WARNING: This code is in-progress only. It doesn't actually work
#  as of yet.
###################################################################


from ScatterSim.BaseClasses import *  

class SolutionModel(Model):


    # Model arguments
    ################################
    # form                          Can take on values: sphere, cylinder, etc.
    
    
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
    def __init__(self, Realspace, Experimental, margs={}):
        """Prepares the model object."""
        self.margs.update(margs)
        
        self.Realspace = Realspace
        self.Experimental = Experimental
        
    
    def sphere(self, q, scale=1.0, contrast=0.1, background=0.0):
        
        r = self.Realspace.pargs['radius']
        V = (4/3)*numpy.pi*(r**3)
        

        P = (scale/V)*(( 3*V*contrast*(sin(q*r)-q*r*cos(q*r) )/( (q*r)**3 ) )**2) + background
        
        return P
        
        
    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        
        q = sqrt( qx**2 + qy**2 + qz**2 )
        
        if self.margs['form']=='sphere':
            return self.sphere( q )
        else:
            return 0.0

    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""
        return 0.0

    def angular_2d(self, theta_incident, thtuple, phtuple):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples.

        (theta_scan_min, theta_scan_max, theta_scan_num) = thtuple
        (phi_scan_min, phi_scan_max, phi_scan_num) = phtuple
        """
        (theta_scan_min, theta_scan_max, theta_scan_num) = thtuple
        (phi_scan_min, phi_scan_max, phi_scan_num) = phtuple
        return 0.0
        
    def ewald_2d(self, theta_incident, qytuple, qztuple):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range of q-values is specified as tubles.
        (qy_min, qy_max, qy_num) = qytuple
        (qz_min, qz_max,qz_num) = qztuple
        """
        (qy_min, qy_max, qy_num) = qytuple
        (qz_min, qz_max,qz_num) = qztuple
        return 0.0


    def to_string(self):
        """Returns a string describing the model."""
        s = "Base Model (does nothing; just returns zero for all reciprocal-space)."
        return s
               
    
