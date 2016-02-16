#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################
# BoxVarySimulation.py
# version 0.0.5
# May 5, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################





# Import
###################################################################
from ScatterSim.Scattering import *





# Define parameters
###################################################################

        
# Experimental and plotting parameters
theta_incident = 0.2
qy_s = (0.0, 0.4, 140)
qz_s = (0.0, 0.8, 160)
e = Experimental( wavelength=0.124, theta_incident=theta_incident )

# Parameters that control the real-space simulation 'box'.
bargs = { \
        'x': { 'start':0.0, 'end':1000.0, 'num':180 }, \
        'y': { 'start':0.0, 'end':800.0, 'num':180 }, \
        'z': { 'start':0.0, 'end':240.0, 'num':100 }, \
        }
b = SimulationBox(bargs=bargs)

# The potential function (real-space model) parameters
pargs = { 'box_z_span': b.limits['z']['span'], \
        'film_thickness': 200.0, \
        'rho_substrate': 20.1, \
        'rho_ambient': 0.0, \
        'rho_film': 10.0, \
        'rho1': 15.0, \
        'repeat_spacing': 38.0, \
        'radius': (17.0/2.0), \
        'interfacial_spread': 1.5, \
        'positional_spread': 1.0, \
        'size_spread': 0.10, \
        'rotation': True, \
        'eta': 0.0, \
        'phi': 0.0, \
        'theta': 0.0, \
        }

#p = HexagonalCylindersPotential( pargs=pargs )
p = HexagonalCylindersJitterPotential( pargs=pargs )



# The modeling parameters
margs = { \
        'reflection_computation': False, \
        }
m = IntegralGISAXSModel(b, e, margs=margs)


# Simulation parameters
sargs = { \
        'verbose': True, \
        'parallelize': True, \
        'num_processes': 7, \
        'b_ranges': { \
                    'x_span_rel': (0.75, 1.25, 0.002), \
                    'y_span_rel': (0.75, 1.25, 0.002), \
                    'z_span_rel': (0.9, 1.15, 0.002), \
                    }, \
        'p_ranges': { \
                    'eta': (0.0, 60.0, 5.0), \
                    'phi': (0.0, 90.0, 5.0), \
                    'theta': (0.0, 180.0, 5.0), \
                    }, \
        'save_samples': False
                
        }





# Run Simulation
###################################################################

def run_simulation():
    """Performs the simulation, returning a scattering
    object that can be further queried."""
    
    # Create simulation object
    sim = StochasticSamplingSimulation( m, p, b, e, sargs=sargs, seed=127 )
    print( sim.to_string() )
    
    sim.set_num_samples( 7*1)
    #print( "Will do values: " )
    #print( sim.get_values() )

    # Calculate
    scatter = sim.ewald_2d( e.theta_incident, qy_s, qz_s )
    #scatter = sim.angular_value( e.theta_incident, 0.2, 0.1 )
    
    return scatter




scatter = run_simulation()
scatter.plot(log_scaling=True, plotmin=0.01, plotmax=0.95, mirror=True )
#scatter.plotQ(filename='scattering2D-log.png', log_scaling=True, plotmin=0.01, plotmax=0.95 )
#print scatter.to_string()


b.fill(p)
b.plot_potential_bisect( filename='potential.png', sargs=sargs, plotmin=0.2, plotmax=0.8 )
#b.plot_reflectivity( 0.005, 1.0, 0.001, filename='reflectivity.png', k_wavevector=e.k_wavevector )
#b.plot_reflectivityQ()




