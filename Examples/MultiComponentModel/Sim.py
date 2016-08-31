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
# You need two paths for it to work for now
#ScatterSim_PATH='/home/kyager/BNL/ScatterSim/main'
#ScatterSim_PATH in sys.path or sys.path.append(ScatterSim_PATH)
#ScatterSim_PATH='/home/lhermitt/research/projects/kevin/ScatterSim'
#ScatterSim_PATH in sys.path or sys.path.append(ScatterSim_PATH)
#ScatterSim_PATH = '/home/lhermitt/research/projects/kevin/ScatterSim/ScatterSim'
#ScatterSim_PATH in sys.path or sys.path.append(ScatterSim_PATH)


from ScatterSim.Scattering import *
from ScatterSim.MultiComponentModel import *
from MultiComponentInteraction import *





# Settings
########################################
ptype = 'structure_factor'
area_of_interest = [0.0,1.2,0,2.5]

plot_data = False




# Load experimental data
########################################

data_dir = '../data/'
data_file = 'Iq_FAA15_10120_14Kev.dat'
ff_data_file = 'Free_FAA15_10120_14Kev.dat'

#d = ExperimentalData1D()
#d.load_intensity_txt( data_dir+data_file, skiprows=1, subtract_minimum=True )
#d.load_form_factor_txt( data_dir+ff_data_file, skiprows=1, subtract_minimum=True )
#d.set_structure_factor_asymptote( 0.75, 0.82 )
    

if plot_data:
    d.plot_intensity( scaling=[0.0,1.4,0.5,2e4], ylog=True )
    d.plot_form_factor( scaling=[0.0,1.4,0.5,2e4], ylog=True )
    d.plot_structure_factor( scaling=[0.0,1.2,0,2.5] )
    
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
sld_Fe2O3 = 42.76 # 10^-6 A^-2
sld_strep = 13.809 # 10^-6 A^-2

sld_CdSe = 41.38 # 10^-6 A^-2
sld_CdTe = 41.16 # 10^-6 A^-2
sld_ZnS = 32.90 # 10^-6 A^-2


# Particles
qd705 = SphereNanoObject( pargs={ 'radius': 3.92, 'sigma_R': 0.21, 'rho_ambient': sld_water, 'rho1': sld_CdTe } )
qd605 = SphereNanoObject( pargs={ 'radius': 2.67, 'rho_ambient': sld_water, 'rho1': sld_CdSe } )
qd525 = SphereNanoObject( pargs={ 'radius': 1.26, 'rho_ambient': sld_water, 'rho1': sld_CdSe } )

pargs={ 'radius': 9.1/2.0, 'sigma_R': 0.11, 'rho_ambient': sld_water, 'rho1': sld_Au, 'iso_external': True }
#Au = SphereNanoObject( pargs=pargs )
Au = SpherePolydisperseNanoObject( pargs=pargs )

pargs={ 'radius': 10.2/2.0, 'sigma_R': 0.049, 'rho_ambient': sld_water, 'rho1': sld_Fe2O3, 'iso_external': True }
#Fe2O3 = SphereNanoObject( pargs=pargs )
Fe2O3 = SpherePolydisperseNanoObject( pargs=pargs )


# Non-lattice parameters
peak1 = PeakShape(nu=1, delta=0.05)
back = background( 5.0, 0.012, -4.0, 5.0, -0.5 )


# Lattice

if False:
    # Diamond-like lattice (zincblende)
    nearest_neighbor = 18.7
    lattice_spacing = nearest_neighbor/( sqrt(3.0)/4.0 )
    
    l = DiamondTwoParticleLattice( [Au,Fe2O3], lattice_spacing, sigma_D=0.1 )
    
    initial_guess = [ 0.004349, 2.265, 0.03591, 0.06773, 0, 0, -6, 0, -2, 0.8171, 0.01374, ] # Fit, E = 1.374

if False:
    # Double-filled diamond-like lattice (CaF2)
    nearest_neighbor = 18.5
    lattice_spacing = nearest_neighbor/( sqrt(3.0)/4.0 )
    
    l = DoubleFilledDiamondTwoParticleLattice( [Au,Fe2O3], lattice_spacing, sigma_D=0.1 )

    initial_guess = [ 0.007787, 4.118, 0.03413, 0.0857, 0, 0, -6, 0, -2, 0.7948, 0.0001495, ] # Fit, E = 1.81
    
if False:
    # Two interpenetrated diamond networks (NaTl)
    nearest_neighbor = 18.5
    lattice_spacing = nearest_neighbor/( sqrt(3.0)/4.0 )
    
    l = InterpenetratingDiamondTwoParticleLattice( [Au,Fe2O3], lattice_spacing, sigma_D=0.1 )

    initial_guess = [ 0.02428, 3.377, 0.02977, 0.1082, 0, 0, -6, 0, -2, 0.7583, 0.0004368, ] # Fit, E = 4.59


if True:
    # BCC-like lattice (CsCl)
    nearest_neighbor = 21.0
    lattice_spacing = nearest_neighbor/( sqrt(3.0)/2.0 )
    
    l = BodyCenteredTwoParticleLattice( [Fe2O3,Au], lattice_spacing, sigma_D=0.1 )
    
    initial_guess = [0.0104, 1.92, 0.0497, 0.101, 0, 0, -6, 0, -2, 0.843, 0.0246, ] # Manual E = 22.41
    #initial_guess = [ 0.0004247, 2.224, 0.09982, 0.1694, 0, 0, -6, 0, -2, 0.8669, 0.008418, ] # Fit E = 12.62

if False:
    # Alternating simple cubic (NaCl)
    nearest_neighbor = 21.5
    lattice_spacing = nearest_neighbor*2.0
    
    l = AlternatingSimpleCubic( [Au,Fe2O3], lattice_spacing, sigma_D=0.1 )

    initial_guess = [0.00291, 2.31, 0.0682, 0.0307, 0, 0, -6, 0, -2, 0.842, 0.0448, ] # Fit E = 7.875

if False:
    # Alternating simple cubic (NaCl)
    nearest_neighbor = 24.8
    lattice_spacing = nearest_neighbor*2.0
    
    l = AlternatingSimpleCubic( [Au,Fe2O3], lattice_spacing, sigma_D=0.1 )

    initial_guess = [ 0.001909, 3.149, 0.028, 0.07644, 0, 0, -6, 0, -2, 0.8166, 0.001141, ] # Fit E = 8.75


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
                [0.001, 0.1], # delta
                [0.01, 0.201], # sigma_D (DW)
                [0.0, 1e10], # bc
                [0.0, 1e10], #bp
                [-4.0, -1.0], # balpha
                [0.0, 1e10], # bp2
                [-3.5, 0.0], # balpha2
                [0.7, 1.2], # scale
                [0.0, 0.15], # offset
                ]



m = MultiComponentModel( l, peak1, back, c=2.1e-12 , margs=margs )
#m.set_experimental_P_of_q( d.q_ff_vals, d.ff_vals ) # Introduces factor of: ~2e-12*

#f = MultiComponentFit( d, m, initial_guess=initial_guess,  q_start=0.165, q_end=0.67, vary=vary, step_sizes=step_sizes, constraints=constraints, fargs=fargs )



# Working
########################################
title = data_file[:-4]
def single():
    filename = 'fit-working.png'
    f.plot( filename=filename, title=title, scaling=area_of_interest, ylog=False, show_extended=True )
    
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
    filename = 'saving.png'
    f.set_q_range( 0.05, 0.7 )
    f.plot( filename=filename, title=title, scaling=area_of_interest, ylog=False, show_extended=False )
    
    filename = 'fit_dat-' + l.__class__.__name__ + '.pkl'
    fout = open( filename, 'w' )
    pickle.dump( f.fit_curve(q_start=0.06, q_end=0.67) , fout )
    fout.close()


#single()
#initial_guess = watcher()
#auto_fit()
#copy_working()
#save_fit()

#overlay_ops(data_file, scaling=[0, 0.7, 0, 2.0*7], plot=True, plot_offset=2.0, output_txt=True)

