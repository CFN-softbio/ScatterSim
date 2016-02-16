#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################
# TestSimulation.py
# version 0.0.5
# April 12, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################





# Import
###################################################################
from ScatterSim.Scattering import *



# Define simulation
###################################################################
# The simulation object controls what gets done. A simple simulation
# might just call use a model directly. A more complicated simulation
# might vary a particular parameter (e.g. an angle) and average the
# results.

class TestSimulation(Simulation):
    """This is just a test simulation.
    It doesn't do anything useful."""
    
    def to_string(self):
        s = "This is a Test Simulation.\n"
        s += super(TestSimulation, self).to_string()
        return s
        
    


# Define parameters
###################################################################

        
# Experimental and plotting parameters
theta_incident = 0.2
qy_s = (0.0, 0.5, 60)
qz_s = (0.0, 1.0, 60)
e = Experimental( wavelength=0.124, theta_incident=theta_incident )

# Parameters that control the real-space simulation 'box'.
bargs = { \
        'x': { 'start':0.0, 'end':1000.0, 'num':256 }, \
        'y': { 'start':0.0, 'end':800.0, 'num':200 }, \
        'z': { 'start':0.0, 'end':120.0, 'num':32 }, \
        }
b = SimulationBox(bargs=bargs)

# The potential function (real-space model) parameters
pargs = { 'box_z_span': b.limits['z']['span'], \
        'film_thickness': 80.0, \
        'rho_substrate': 20.1, \
        'rho_ambient': 0.0, \
        'rho_film': 10.0-0.5j, \
        'rho1': 15.0-0.5j, \
        'repeat_spacing': 38.0, \
        'radius': (17.0/2.0), \
        'interfacial_spread': 1.0, \
        'rotation': True, \
        'eta': 0.0, \
        'phi': 90.0, \
        'theta': 0.0, \
        }

p = HexagonalCylindersPotential( pargs=pargs )


# The modeling parameters
margs = { \
        'reflection_computation': False, \
        }
m = IntegralGISAXSModel(b, e, margs=margs)


# Simulation parameters
sargs = { \
        'verbose': True, \
        'smear_theta': True, \
        'smear_theta_range': (0.0, 90.0, 5.0), \
        }





# Run Simulation
###################################################################

def run_simulation():
    """Performs the simulation, returning a scattering
    object that can be further queried."""
    
    # Fill the box with the realspace profile
    b.fill(p)
    b.plot_potential_bisect( filename='potential.png', sargs=sargs, plotmin=0.2, plotmax=0.8 )


    # Create simulation object
    ts = TestSimulation( m, p, b, e, sargs=sargs )
    print( ts.to_string() )

    # Calculate the requested detector image
    scatter_2d = ts.ewald_2d( e.theta_incident, qy_s, qz_s )
    
    return scatter_2d



scatter = run_simulation()
scatter.plotQ(log_scaling=True, plotmin=0.01, plotmax=0.95 )
b.plot_reflectivityQ(0.001,0.1)
