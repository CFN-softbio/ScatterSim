#from ScatterSim.Scattering import *
from ScatterSim.MultiComponentModel import SpherePolydisperseNanoObject, PeakShape,\
    BodyCenteredTwoParticleLattice, background
#from MultiComponentInteraction import *

# Settings
########################################
ptype = 'structure_factor'
area_of_interest = [0.0,1.2,0,2.5]

plot_data = False

# Candidate model
########################################


# Densities
sld_water = 9.43 # 10^-6 A^-2

sld_Au = 119.16 # 10^-6 A^-2
sld_Fe2O3 = 42.76 # 10^-6 A^-2
sld_strep = 13.809 # 10^-6 A^-2

sld_CdSe = 41.38 # 10^-6 A^-2
sld_CdTe = 41.16 # 10^-6 A^-2
sld_ZnS = 32.90 # 10^-6 A^-2


# parameters
# Lattice
# In nanometers?
# I want a 50nm lattice spacing with one species a 5nm Au and the other 9.7nm Au (radius)
# nanosphere
r1 = 9.5 #nm
r2 = 5.0 #nm
lattice_spacing = 49.5 #nm

# Particles, make radius very small for powder assumption
#pargs={ 'radius': r1, 'sigma_R': 0.11, 'rho_ambient': sld_water, 'rho1': sld_Au, 'iso_external': True }
#Au = SpherePolydisperseNanoObject( pargs=pargs )
#pargs={ 'radius': r2, 'sigma_R': 0.11, 'rho_ambient': sld_water, 'rho1': sld_Au, 'iso_external': True }
#Au2 = SpherePolydisperseNanoObject( pargs=pargs )

pargs={ 'radius': .1, 'sigma_R': 0.01, 'rho_ambient': sld_water, 'rho1': sld_Au*(r1/.1)**3, 'iso_external': True }
Au = SpherePolydisperseNanoObject( pargs=pargs )
pargs={ 'radius': .1, 'sigma_R': 0.01, 'rho_ambient': sld_water, 'rho1': sld_Au*(r2/.1)**3, 'iso_external': True }
Au2 = SpherePolydisperseNanoObject( pargs=pargs )

# Non-lattice parameters
peak1 = PeakShape(nu=0.0001, delta=0.001)
back = background( 5.0, 0.012, -4.0, 5.0, -0.5 )

#nearest_neighbor = lattice_spacing
#lattice_spacing = nearest_neighbor/( sqrt(3.0)/2.0 )
    
l = BodyCenteredTwoParticleLattice( [Au,Au2], lattice_spacing, sigma_D=0.01 )
    
#initial_guess = [0.0104, 1.92, 0.0497, 0.101, 0, 0, -6, 0, -2, 0.843, 0.0246, ] # Manual E = 22.41

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

qs = np.linspace(6e-3, .1,1000)
#Lq = l.form_factor_intensity_isotropic_array(qs*10) #convert q to nm

# Get the total form factor
Pq = l.structure_factor_isotropic_array(qs*10,peak1) #convert q to nm
#get the structure factor for both species
Sq1 = Au.form_factor_intensity_isotropic_array(qs*10)
Sq2 = Au2.form_factor_intensity_isotropic_array(qs*10)

# total structure factor
plt.figure(0);plt.clf();
plt.plot(qs,Pq)

#the form factor of each species
plt.figure(1);plt.clf();
plt.loglog(qs,Sq1)
plt.loglog(qs,Sq2)

# the normalized form factor, should resemble thee supplemental in;
#
#Macfarlane, Robert J., et al. "Nanoparticle superlattice engineering with
# DNA." Science 334.6053 (2011): 204-208.

plt.figure(2);plt.clf();
plt.plot(qs,Pq/(Sq1+Sq2))
