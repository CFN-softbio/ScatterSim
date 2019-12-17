#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################
# Jan 20, 2011
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
 

# Import ScatterSim
###################################################################
import sys, shutil
ScatterSim_PATH='/home/kyager/BNL/ScatterSim/main'
ScatterSim_PATH in sys.path or sys.path.append(ScatterSim_PATH)

from ScatterSim.Scattering import *
from ScatterSim.MultiComponentModel import *
from MultiComponentInteraction import *





# Settings
########################################
ptype = 'structure_factor'
area_of_interest = [0.0,0.7,0,6.0]

plot_data = True




# Load experimental data
########################################

data_dir = './'
data_file = 'B6_t28C2.dat'
ff_data_file = 'B6_t58H1(use as melting).dat'

d = ExperimentalData1D()
d.load_intensity_txt( data_dir+data_file, skiprows=1, subtract_minimum=True )
d.load_form_factor_txt( data_dir+ff_data_file, skiprows=1, subtract_minimum=True )
d.set_structure_factor_asymptote( 0.42, 0.65 )
    

if plot_data:
    d.plot_intensity( scaling=[0.0,0.7,0.5,2e4], ylog=True )
    d.plot_form_factor( scaling=[0.0,0.7,0.5,2e4], ylog=True )
    d.plot_structure_factor( scaling=[0.0,0.7,0,6.0] )
    
    # Save data
    if ptype=='structure_factor':
        s_of_q = d.structure_factor()
        filename = 'fit_dat-data-'+data_file+'.pkl'
        fout = open( filename, 'w' )
        pickle.dump( s_of_q , fout )
        fout.close()





# Candidate model
########################################


# Densities
sld_water = 9.43 # 10^-6 A^-2
sld_Au = 119.16 # 10^-6 A^-2

sld_strep = 13.809 # 10^-6 A^-2

sld_CdSe = 41.38 # 10^-6 A^-2
sld_CdTe = 41.16 # 10^-6 A^-2
sld_ZnS = 32.90 # 10^-6 A^-2


# Particles
# Particles
qd705 = SphereNanoObject( pargs={ 'radius': 3.92, 'rho_ambient': sld_water, 'rho1': sld_CdTe } )
#qd705 = SphereNanoObject( pargs={ 'radius': 6.0, 'rho_ambient': sld_water, 'rho1': sld_CdTe } )
qd605 = SphereNanoObject( pargs={ 'radius': 2.67, 'rho_ambient': sld_water, 'rho1': sld_CdSe } )
qd525 = SphereNanoObject( pargs={ 'radius': 1.26, 'rho_ambient': sld_water, 'rho1': sld_CdSe } )



pargs={ 'radius': 9.1/2.0, 'sigma_R': 0.11, 'rho_ambient': sld_water, 'rho1': sld_Au, 'iso_external': True }
#Au = SphereNanoObject( pargs=pargs )
Au = SpherePolydisperseNanoObject( pargs=pargs )
#Au = CubePolydisperseNanoObject( pargs=pargs )

pargs={ 'radius': 3.92, 'sigma_R': 0.21, 'rho_ambient': sld_water, 'rho1': sld_CdTe, 'iso_external': True }
pargs={ 'radius': 3.92, 'sigma_R': 0.45, 'rho_ambient': sld_water, 'rho1': sld_CdTe, 'iso_external': True }
#qd705 = SphereNanoObject( pargs=pargs )
qd705 = SpherePolydisperseNanoObject( pargs=pargs )


# Non-lattice parameters
peak1 = PeakShape(nu=1, delta=0.05)
#peak1 = PeakShape(nu=1, delta=0.05, q1=0.024, slope=0.004)  #giving q1 and slope for Williamson-Hall analysis to get the lattice strain
back = background( 5.0, 0.012, -4.0, 5.0, -0.5 )

nearest_neighbor = 33.8


# Lattice

if True:
    # BCC
    lattice_spacing = nearest_neighbor/( sqrt(3.0)/(2.0) )
    
    l = BodyCenteredTwoParticleLattice( [Au, qd705], lattice_spacing, sigma_D=0.1 )
    
    initial_guess = [ 0.004, 0, 0.008, 0.08, 0, 0, -6, 0, -2, 1.0, 0, ]
    initial_guess = [ 0.003896, 0.02015, 0.01075, 0.07343, 0, 0, -6, 0, -2, 1, 0, ] # Fit E = 1.017
    initial_guess = [ 0.003567, 0.04388, 0.01072, 0.07085, 0, 0, -6, 0, -2, 1.025, 0.01665, ] # Fit with scale, offset, E = 0.9324
    
    initial_guess = [ 0.003397, 0.1248, 0.01006, 0.07039, 0, 0, -6, 0, -2, 1.046, 0.001812, ] # Fit for ^2 errors
    initial_guess = [ 0.00345, 0.1284, 0.009491, 0.06724, 0, 0, -6, 0, -2, 1.021, 0.02314, ] # Fit for ^4 errors
    initial_guess = [ 0.003533, 0.1083, 0.009496, 0.06311, 0, 0, -6, 0, -2, 1.006, 0.02356, ] # Fit for ^6 errors
    
    initial_guess = [ 0.003867, 0.1761, 0.009097, 0.0725, 0, 0, -6, 0, -2, 1.027, 0.0004611, ] # Tweaked PDI; E = 0.48
    
    
    
print( l.to_string() )




# Fit
########################################
margs = {}
margs['ptype'] = ptype
margs['diffuse'] = True
margs['beta_approx'] = False

fargs = {}
fargs['ptype'] = ptype
fargs['mu_T'] = 1.05

#fargs['T_initial'] = 0.1
#fargs['mu_T'] = 1.15

# parameters are:  [c, nu, delta, sigma_D, bc, bp, balpha, bp2, balpha2, scale, offset ]
step_sizes = [ initial_guess[0]*0.1, 0.05, 0.01, 0.01, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.01]
#step_sizes = [ s*10 for s in step_sizes ]
vary = [ True, False, False, True, False, False, False, False, False, True, False ] # Vary c, sigma_D, and scale
vary = [ True, False, False, False, False, False, False, False, False, True, True ] # Vary scaling
#vary = [ False, True, True, False, False, False, False, False, False, False, False ] # Vary peak shape
#vary = [ True, True, True, False, False, False, False, False, False, True, True ] # Vary peak shape and scaling
#vary = [ False, False, False, True, False, False, False, False, False, False, False ] # Vary sigma_D

vary = [ True, True, True, True, False, False, False, False, False, True, True ]

constraints = [ [0.0, 1e20], # c
                [0.0, 50.4], # nu
                [0.001, 0.05], # delta
                [0.01, 0.101], # sigma_D (DW)
                [0.0, 1e10], # bc
                [0.0, 1e10], #bp
                [-4.0, -1.0], # balpha
                [0.0, 1e10], # bp2
                [-3.5, 0.0], # balpha2
                [0.8, 1.1], # scale
                [0.0, 0.1], # offset
                ]



m = MultiComponentModel( l, peak1, back, c=2.1e-12 , margs=margs )
#m.set_experimental_P_of_q( d.q_ff_vals, d.ff_vals ) # Introduces factor of: ~2e-12*

f = MultiComponentFit( d, m, initial_guess=initial_guess,  q_start=0.1, q_end=0.7, vary=vary, step_sizes=step_sizes, constraints=constraints, fargs=fargs )



# Working
########################################
title = data_file[:-4]
def single():
    filename = 'fit-pretty.png'
    f.plot( filename=filename, title=title, scaling=area_of_interest, ylog=False, show_indexing=False, show_extended=False, interact=False, dpi=300 )
    
def watcher():
    filename = 'fit-working.png'
    f.make_watch_file()
    guess = f.watch_file(plot_filename=filename, title=title, scaling=area_of_interest, ylog=False, show_extended=True )
    
    guess_str = ''
    for el in guess:
        guess_str += '%.4g, ' % (el)
    print( 'initial_guess = [ %s]' % (guess_str) )
    
    return guess

def auto_fit():
    f.plot( filename='fit-before.png', title=title, scaling=area_of_interest )
    f.fit( initial_guess )
    f.plot( filename='fit-after.png', title=title, scaling=area_of_interest )

def copy_working():
    filename = 'fit-' + l.__class__.__name__ + '-manual.png'
    shutil.copy( 'fit-working.png', filename)
    
    filename = 'fit-' + l.__class__.__name__ + '-auto.png'
    shutil.copy( 'fit-after.png', filename)

def save_fit():
    
    filename = 'fit-' + l.__class__.__name__ + '.png'
    f.set_q_range( 0.05, 0.37 )
    f.plot( filename=filename, title=title, scaling=area_of_interest, ylog=False, show_extended=False )
    
    filename = 'fit_dat-' + l.__class__.__name__ + '.pkl'
    fout = open( filename, 'w' )
    pickle.dump( f.fit_curve(q_start=0.06, q_end=0.35) , fout )
    fout.close()


single()
#initial_guess = watcher()
#auto_fit()
#copy_working()
#save_fit()

#overlay_ops(data_file, scaling=[0, 0.4, 0, 6.3], plot=True, output_txt=True)

