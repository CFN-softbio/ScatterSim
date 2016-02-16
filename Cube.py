#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################
# Jan 20, 2011
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
 

# Import
###################################################################
from ScatterSim.Scattering import *
from ScatterSim.MultiComponentModel import *
from ScatterSim.Interaction import *

import pickle



# Settings
########################################
ptype = 'structure_factor'
area_of_interest = [0.0,0.4,0,2.0]

plot_data = True

#lattice_name = 'SC'
#lattice_name = 'BCC'
lattice_name = 'FCC'



# Load experimental data
########################################

data_dir = '../10-ScatterSim_with_MultiComponentModel-Data_to_test/Lu_Cubes/0.15M_PMT/SC_0.15M/'
data_file = 'SC_0.15M_PMT.ave'
ff_data_file = 'SC_0.15M_t71.ave'

d = ExperimentalData1D()
d.load_intensity_txt( data_dir+data_file, skiprows=1, subtract_minimum=True )
d.load_form_factor_txt( data_dir+ff_data_file, skiprows=1, subtract_minimum=True )
d.set_structure_factor_asymptote( 0.42, 0.65 )
    

if plot_data:
    d.plot_intensity( scaling=[0.0,0.7,0.5,2e4], ylog=True )
    d.plot_form_factor( scaling=[0.0,0.7,0.5,2e4], ylog=True )
    d.plot_structure_factor( scaling=[0.0,0.7,0,2.0] )
    
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

# Particles
pargs={ 'radius': 26.0/2.0, 'sigma_R': 0.1, 'rho_ambient': sld_water, 'rho1': sld_Au, 'iso_external': True }
pargs={ 'radius': 30.0/2.0, 'sigma_R': 0.1, 'rho_ambient': sld_water, 'rho1': sld_Au, 'iso_external': True }
#Au = CubeNanoObject( pargs={ 'radius': 26.0/2.0, 'rho_ambient': sld_water, 'rho1': sld_Au } )
Au = CubePolydisperseNanoObject( pargs=pargs )


# Non-lattice parameters
peak1 = PeakShape(nu=1, delta=0.05)
back = background( 5.0, 0.012, -4.0, 5.0, -0.5 )

nearest_neighbor = 55.6


# Lattice
if lattice_name=='SC':
    # Simple Cubic (SC)
    lattice_spacing = nearest_neighbor*1.0
    
    l = SimpleCubic( [Au], lattice_spacing, sigma_D=0.1 )
    #l = AlternatingSimpleCubic( [Au], lattice_spacing, sigma_D=0.1 )
    
    back_c = 0.0
    initial_guess = [600e-5, 0, 0.03, 0.2, back_c*0.0, back_c*0.0, -6.0, back_c*0.30, -2.00, 0.8, 0.0 ]

if lattice_name=='BCC':
    # BCC
    nearest_neighbor = 67.0
    lattice_spacing = nearest_neighbor/( sqrt(3.0)/(2.0) )
    
    l = BCCLattice( [Au], lattice_spacing, sigma_D=0.1 )
    #l = AlternatingSimpleCubic( [Au], lattice_spacing, sigma_D=0.1 )
    
    back_c = 0.0
    initial_guess = [30e-5, 0, 0.03, 0.12, back_c*0.0, back_c*0.0, -6.0, back_c*0.30, -2.00, 0.8, 0.0 ]

if lattice_name=='FCC':
    # FCC
    nearest_neighbor = 69.0
    lattice_spacing = nearest_neighbor/( sqrt(2.0)/(2.0) )
    
    l = FCCLattice( [Au], lattice_spacing, sigma_D=0.1 )
    #l = FaceCenteredFourParticleLattice( [Au, Au, Au, Au], lattice_spacing, sigma_D=0.1 )
    
    back_c = 0
    initial_guess = [4.5e-5, 0, 0.03, 0.1, back_c*0.0, back_c*0.0, -6.0, back_c*0.30, -2.00, 0.8, 0.0 ]




    
print( l.to_string() )


# Fit
########################################
margs = {}
margs['ptype'] = ptype
margs['diffuse'] = True
margs['beta_approx'] = False

fargs = {}
fargs['ptype'] = ptype
fargs['mu_T'] = 1.1

# parameters are:  [c, nu, delta, sigma_D, bc, bp, balpha, bp2, balpha2, scale, offset ]
step_sizes = [0.1e-5, 0.05, 0.01, 0.01, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1]
vary = [ True, True, True, True, False, False, False, False, False, True, True ] # Hold background
vary = [ True, False, False, True, False, False, False, False, False, True, False ] # Vary c and sigma_D, and scale
#vary = [ False, False, False, False, False, False, False, False, False, True, True ] # Vary overall


m = MultiComponentModel( l, peak1, back, c=2.1e-12 , margs=margs )
#m.set_experimental_P_of_q( d.q_ff_vals, d.ff_vals ) # Introduces factor of: ~2e-12*

f = MultiComponentFit( d, m, initial_guess=initial_guess,  q_start=0.06, q_end=0.35, vary=vary, step_sizes=step_sizes, fargs=fargs )


# Working
########################################
def single():
    filename = 'fit-working.png'
    f.plot( filename=filename, scaling=area_of_interest, ylog=False, show_extended=False )
    
def watcher():
    filename = 'fit-working.png'
    f.make_watch_file()
    return f.watch_file(plot_filename=filename, scaling=area_of_interest, ylog=False, show_extended=False )

def auto_fit():
    f.plot( filename='fit-before.png', scaling=area_of_interest )
    f.fit( initial_guess )
    f.plot( filename='fit-after.png', scaling=area_of_interest )

def save_fit():
    filename = 'fit-' + l.__class__.__name__ + '.png'
    f.plot( filename=filename, scaling=area_of_interest, ylog=False, show_extended=False )
    
    filename = 'fit_dat-' + l.__class__.__name__ + '.pkl'
    fout = open( filename, 'w' )
    pickle.dump( f.fit_curve(q_start=0.06, q_end=0.35) , fout )
    fout.close()



single()
#initial_guess = watcher()
#auto_fit()
save_fit()
overlay_ops(data_file, scaling=area_of_interest[:-1]+[6], plot=True, output_txt=True)


