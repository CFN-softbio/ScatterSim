#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################
# MultiComponentSimulation.py
# version 0.0.5
# November 1, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################





# Import
###################################################################
from ScatterSim.Scattering import *
from ScatterSim.MultiComponentModel import *




# Define parameters
###################################################################

        
# Experimental and plotting parameters
q_s = (0.02, 0.3, 500)
q_vals = numpy.linspace( q_s[0], q_s[1], q_s[2] )
e = Experimental( energy=14.0 )


# The potential function (real-space model) parameters
pargs = { 'radius': 20.0, \
        }


#p = CubeNanoObject( pargs=pargs )
p = SphereNanoObject( pargs=pargs )



# The modeling parameters
margs = { \
        'lattice': 'bcc', \
        'spacing_a': 100.0, \
        'spacing_b': 100.0, \
        'spacing_c': 100.0, \
        }
m = MultiComponentModel(p, e, margs=margs)


# Simulation parameters
sargs = { \
        'verbose': True, \
        'parallelize': True, \
        'num_processes': 7, \
        }



#print m.q_value(0.1,0.0,0.0)



def quick_plot( x, y1, y2=[], y3=[], y4=[], y5=[], y6=[], filename='output.png', ylog=True ):
    
    fig = pylab.figure()
    #fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    ax = pylab.subplot(111)

    color_list = [ (0,0,0), \
                    'r', \
                    'b', \
                    (0.7,0.7,0.7), \
                    'b', \
                    'r', \
                    ]
    color_list = [ (0,0,0), \
                    (0.5,0,0), \
                    (1.0,0,0), \
                    (1.0,0.6,0), \
                    (0,0.5,0), \
                    (0,0,0.9), \
                    ]
    pylab.plot( x, y1, color=color_list[0], linewidth=2.0 )

    if y2!=[]:
        #pylab.plot( x, y2, color=(0.5,0.5,0.5), linewidth=2.0 )
        pylab.plot( x, y2, color=color_list[1], linewidth=2.0 )
    if y3!=[]:
        #pylab.plot( x, y3, color=(0.6,0.6,0.6), linewidth=2.0 )
        pylab.plot( x, y3, color=color_list[2], linewidth=2.0 )
    if y4!=[]:
        pylab.plot( x, y4, color=color_list[3], linewidth=2.0 )
    if y5!=[]:
        pylab.plot( x, y5, color=color_list[4], linewidth=2.0 )
    if y6!=[]:
        pylab.plot( x, y6, color=color_list[5], linewidth=2.0 )

    pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
    pylab.ylabel( 'Intensity (a.u.)' )
    
    xi, xf, yi, yf = ax.axis()
    #yi = 0
    yf *= 1.1
    ax.axis( (xi, xf, yi, yf) )
    
    if ylog:
        pylab.semilogy()
    
    pylab.savefig( filename )
    




def q_test(q_cur):
    print( "For q = %f nm^-1" % q_cur )
    print( "    form_factor =                   " + str(p.form_factor(q_cur, 0, 0)) )
    #print( "    form_factor_numerical =         " + str(p.form_factor_numerical(q_cur, 0, 0)) )
    print( "    form_factor_squared =           " + str(p.form_factor_squared(q_cur, 0, 0)) )
    #print( "    form_factor_squared_numerical = " + str(p.form_factor_squared_numerical(q_cur, 0, 0)) )
    print( "    isotropic =                     " + str(p.form_factor_intensity_isotropic(q_cur)) )
    #print( "    isotropic2 =                    " + str(p.form_factor_intensity_isotropic2(q_cur)) )



# Test form factor, etc.
if False:
    q_test(0)
    q_test(0.001)
    q_test(0.01)
    q_test(0.1)
    q_test(1)

    y1_vals = []
    y2_vals = []
    y3_vals = []
    y4_vals = []
    y5_vals = []
    y6_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        
        y1_vals.append( p.form_factor(q, 0, 0) )
        #y2_vals.append( p.form_factor_numerical(q,0,0, num_points=10) )
        y3_vals.append( p.form_factor_squared(q,0,0) )
        #y4_vals.append( p.form_factor_squared_numerical(q,0,0, num_points=40) )
        y5_vals.append( p.form_factor_intensity_isotropic(q) )
        #y6_vals.append( p.form_factor_intensity_isotropic2(q) )


    quick_plot( q_vals, y1_vals, y2_vals, y3_vals, y4_vals, y5_vals, y6_vals, filename='form_factor.png' )


# Test orientation effect on form factor
if False:
    q_s = (0.012, 1.0, 500)
    q_vals = numpy.linspace( q_s[0], q_s[1], q_s[2] )

    sld1 = 10.0
    sld2 = 10.0
    sld_ambient = 0.0
    particle_size = 20.0
    
    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':0.0, 'theta':0.0 } )
    y1_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        y1_vals.append( p2.form_factor(0, 0, q).real )

    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':9.0, 'theta':0.0 } )
    y2_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        y2_vals.append( p2.form_factor(0, 0, q).real )

    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':18.0, 'theta':0.0 } )
    y3_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        y3_vals.append( p2.form_factor(0, 0, q).real )

    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':27.0, 'theta':0.0 } )
    y4_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        y4_vals.append( p2.form_factor(0, 0, q).real )

    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':36.0, 'theta':0.0 } )
    y5_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        y5_vals.append( p2.form_factor(0, 0, q).real )

    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':45.0, 'theta':0.0 } )
    y6_vals = []
    for i, q in enumerate(q_vals):
        percent = (100.0*i)/len(q_vals)
        print( "%d. Calculating q = %4.5f nm (%2.1f%% done)" % (i+1,q,percent) )
        y6_vals.append( p2.form_factor(0, 0, q).real )



    quick_plot( q_vals, y1_vals, y2_vals, y3_vals, y4_vals, y5_vals, y6_vals, filename='form_factor-orientations.png', ylog=False )
   

# Test real-space profile
if False:
    r_vals = numpy.linspace( -50, 50, 200 )
    y1_vals = []
    for r in r_vals:
        y1_vals.append( p.V(r,0,0) )

    quick_plot( r_vals, y1_vals, filename='potential.png' )

# Test peak shape
if False:
    
    nu_use = 0.01
    delta_use = 0.5
    peak1 = PeakShape( nu=nu_use, delta=delta_use )    # Black
    peak2 = PeakShape( nu=1.00, delta=delta_use )   # Red
    peak3 = PeakShape( nu=1000.0, delta=delta_use )     # Blue
    
    
    q_vals = numpy.linspace(-1, 1, 200)
    y1_vals = []
    y2_vals = []
    y3_vals = []
    for q in q_vals:
        y1_vals.append( peak1.val(q) )
        y2_vals.append( peak2.val(q) )
        y3_vals.append( peak3.val(q) )

    #quick_plot( q_vals, y1_vals, y2_vals, y3_vals, filename='peak.png', ylog=False )
    peak1.plot()



class float_to_string(object):
    
    def __init__(self, mult_cutoff=12, div_cutoff=7, latex=False):
        
        self.mult_cutoff = mult_cutoff
        self.div_cutoff = div_cutoff
        self.latex = latex
        
        self.number_estimates = []
    
    def build_number_estimates(self):
        
        self.number_estimates.append( [0.0, "0", "0"] )
        self.number_estimates.append( [1.0, "1", ""] )
        self.number_estimates.append( [2.0, "2", "2"] )

        # Whole numbers
        for i in range(3,self.mult_cutoff+1):
            self.number_estimates.append( [i, "%d"%i, "%d"%i] )

        # Fractions
        for i in range(2,self.div_cutoff+1):
            self.number_estimates.append( [1.0/i, "1/%d"%i, r"\frac{1}{%d}" % i ] )
        for i in range(3,self.div_cutoff+1):
            self.number_estimates.append( [2.0/i, "2/%d"%i, r"\frac{2}{%d}" % i ] )
        for i in range(2,self.div_cutoff+1):
            self.number_estimates.append( [3.0/i, "3/%d"%i, r"\frac{3}{%d}" % i ] )
        for i in range(3,self.div_cutoff+1):
            self.number_estimates.append( [4.0/i, "4/%d"%i, r"\frac{4}{%d}" % i ] )
        for i in range(2,self.div_cutoff+1):
            self.number_estimates.append( [5.0/i, "5/%d"%i, r"\frac{5}{%d}" % i ] )

        # Square roots
        for i in range(2,self.mult_cutoff+1):
            self.number_estimates.append( [sqrt(i), "sqrt(%d)"%i, r"\sqrt{%d}" % i ] )
            
            # With multipliers
            for j in range(2,self.mult_cutoff+1):
                self.number_estimates.append( [j*sqrt(i), "%dsqrt(%d)"%(j,i), r"%d\sqrt{%d}" % (j,i) ] )

            # With divisors
            for j in range(2,self.div_cutoff+1):
                self.number_estimates.append( [sqrt(i)/j, "sqrt(%d)/%d"%(i,j), r"\frac{\sqrt{%d}}{%d}" % (i,j) ] )

        # Trigonometry
        self.number_estimates.append( [pi, "pi", r"\pi" ] )
        for i in range(2,self.mult_cutoff+1):
            self.number_estimates.append( [i*pi, "%dpi"%i, r"%d\pi" % i ] )


    def analytical_number(self, number, tolerance=1e-6):
        if self.number_estimates==[]:
            # Build up list of numbers
            self.build_number_estimates()

        # Search through estimates for a match
        for value, string, latex_string in self.number_estimates:
            if abs(number-value)<tolerance:
                if self.latex:
                    return latex_string
                else:
                    return string

        # Default, return floating point
        return "%.4f" % number
        
    

def plot_lattice_peaks( peak_dict_1d, title=None, filename='lattice.png', peak_limit=10, print_peaks=True, fundamental_index=1):


    x = []
    y = []
    labels = []

    if print_peaks:
        print( "peak\tq value       \th,k,l\tm\tf\tintensity" )
        
    i = 1
    for q in sorted(list1d.iterkeys()):
        if i<=peak_limit:
            h, k, l, m, f = list1d[q]
            miller = [h, k, l]
            miller.sort()
            miller.reverse()
            hn, kn, ln = miller
            
            intensity = m*(f**2)
            
            # Skip 0,0,0
            if not (hn==0 and kn==0 and ln==0):
                if print_peaks:
                    print( "%d:\t%.12f\t%d,%d,%d\t%d\t%d\t%d" % (i, q, hn, kn, ln, m, f, intensity) )
                    
                x.append( q )
                y.append( intensity )
                
                s = "%d%d%d" % (hn, kn, ln)
                labels.append( s )
                
                i = i + 1
        

    
    scaling = 1.0*y[fundamental_index-1]
    q0 = x[fundamental_index-1]
    y_relative = [ intensity*1.0/scaling for intensity in y ]
    
    
    pylab.rcParams['axes.labelsize'] = 'xx-large'
    pylab.rcParams['xtick.labelsize'] = 'x-large'
    pylab.rcParams['ytick.labelsize'] = 'x-large'

    fig = pylab.figure()
    #fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    ax = pylab.subplot(111)


    pylab.stem( x, y )


    pylab.xlabel( 'q' )
    pylab.ylabel( 'Intensity' )
    
    xi, xf, yi, yf = ax.axis()
    xi = 0
    yi = 0
    yf *= 1.2
    ax.axis( (xi, xf, yi, yf) )

    y_nudge = (yf-yi)*0.012

    # Put labels
    converter = float_to_string(latex=True)
    for q, intensity, int_rel, label in zip(x, y, y_relative, labels): 
        q_rel = q/q0
        q_str = converter.analytical_number(q_rel)
        int_str = converter.analytical_number(int_rel)
        label_final = label + "\n" + r"$" + q_str + r"q_0$" + "\n" + r"$" + int_str + r" I_0$"
        pylab.text(q, intensity+y_nudge, label_final, horizontalalignment='center', verticalalignment='bottom')
    
    
    if title!=None:
        pylab.figtext( 0.12, 0.92, title, verticalalignment="bottom", horizontalalignment='left', size=20 )

    pylab.savefig( filename )

 
    
# Test lattice    
if False:

    sld1 = 10.0
    sld2 = 10.0
    sld_ambient = 0.0

    particle_size = 27.0/2.0
    #p1 = SphereNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld1 } )
    #p2 = SphereNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2 } )
    #p1 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2 } )
    p1 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':0.0, 'theta': 0.0 } )
    p2 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_ambient, 'rho1': sld2, 'eta':0.0, 'phi':0.0, 'theta': 0.0 } )
    
    
    #p3 = SphereNanoObject( pargs={ 'radius': 100.0 } )
    #p3.plot_form_factor_amplitude( q_s )
    #p3.plot_form_factor_intensity( q_s )
    #p3.plot_form_factor_isotropic( q_s, ylog=True )

    #l = BCCLattice( [p1,p2], 100, sigma_D=0.01 )
    l = BodyCenteredTwoParticleLattice( [p1,p2], 55.0, sigma_D=0.1 )
    print l.to_string()
    #print l.sum_over_objects( (0.2,0,0), 1, 0, 0)
    
    # Test structure factor
    if True:
        peak1 = PeakShape()
        peak1.gaussian(delta=0.005)

        if False:
            # Calculate point-by-point
            y1_vals = []
            for i, q in enumerate(q_vals):
                #y1_vals.append( l.structure_factor_isotropic( q, peak1 ) )
                y1_vals.append( l.intensity( q, peak1 ) )
        else:
            # Use array method
            #y1_vals = l.structure_factor_isotropic_array( q_vals, peak1 )
            #y1_vals = l.intensity_array( q_vals, peak1 )
            #quick_plot( q_vals, y1_vals, filename='S_of_q.png', ylog=False )
            
            y1_vals = l.plot_structure_factor_isotropic( q_s, peak1 )
            y1_vals = l.plot_intensity( q_s, peak1, back_pre=1000.0, back_const=0.1, ylog=True )
            

    # Test lattice peak positions
    if False:
        l = BCCLattice( [p1,p2], 1.0, sigma_D=0.1 )
        #l = FCCLattice( [p1,p2], 1.0, sigma_D=0.1 )
        list1d = l.iterate_over_hkl_1d()
        plot_lattice_peaks( list1d, title="BCC", peak_limit=40)

        print l.to_povray_string()


# Simulate cube-packing paper
if False:
    sld_water = 9.43 # 10^-6 A^-2
    sld_Au = 119.16 # 10^-6 A^-2
    particle_size = 27.0/2.0
    
    peak1 = PeakShape()
    peak1.gaussian(delta=0.011)
    back = background( 1e13, 1e12, -2.0 )

    q_s = (0.025, 0.32, 500)
    q_vals = numpy.linspace( q_s[0], q_s[1], q_s[2] )

    p1 = CubeNanoObject( pargs={ 'radius': particle_size, 'rho_ambient': sld_water, 'rho1': sld_Au, 'eta':0.0, 'phi':0.0, 'theta': 0.0 } )
    
    if False:
        # The BCC structure, corner-to-corner connections
        nearest_neighbor = 77.0
        lattice_spacing = nearest_neighbor/( sqrt(3.0)/2.0 )
        
        
        l = BodyCenteredTwoParticleLattice( [p1,p1], lattice_spacing, sigma_D=0.1 )
        print l.to_string()
        
        int_vals = l.plot_structure_factor_isotropic( q_s, peak1, background=back, ylog=True )
        #int_vals = l.plot_structure_factor_isotropic( q_s, peak1, background=None, ylog=False )
        #int_vals = l.plot_intensity( q_s, peak1, background=back, ylog=True )
        
    else:
        # The FCC structure, edge-to-edge connections
        nearest_neighbor = 54.0
        lattice_spacing = nearest_neighbor/( sqrt(2.0)/2.0 )
        
        
        l = FaceCenteredFourParticleLattice( [p1,p1,p1,p1], lattice_spacing, sigma_D=0.1 )
        #l = FCCLattice( [p1], lattice_spacing, sigma_D=0.1 )
        print l.to_string()
        
        int_vals = l.plot_structure_factor_isotropic( q_s, peak1, background=back, ylog=True )
        #int_vals = l.plot_structure_factor_isotropic( q_s, peak1, background=None, ylog=False )
        #int_vals = l.plot_intensity( q_s, peak1, back_pre=1000.0, back_const=0.1, ylog=True )
        
        
        
        
if True:
    # Compare a perfect BCC with a random BCC

    peak1 = PeakShape()
    peak1.gaussian(delta=0.001)
    repeats = 3

    p1 = SphereNanoObject( pargs={ 'radius': 20.0, 'rho_ambient': 0.0, 'rho1': 10.0} )
    p2 = SphereNanoObject( pargs={ 'radius': 12.0, 'rho_ambient': 0.0, 'rho1': 7.0} )
    lattice_spacing = 80.0
    
    # Should be the same (perfect lattice):
    #l = BodyCenteredTwoParticleLattice( [p1,p2], lattice_spacing, sigma_D=0.1)
    l = BodyCenteredTwoParticleExtendedLattice( [p1,p2], lattice_spacing, repeat=repeats, sigma_D=0.1)
    
    # Add disorder
    num_swaps = 25
    num_defects = num_swaps*2
    num_subcells = repeats**3
    random.seed(4987)
    for i in range(num_swaps):
        
        # Even index: a corner
        index1 = random.randint(0,num_subcells-1)*2
        # Odd index, a center
        index2 = random.randint(0,num_subcells-1)*2 + 1
        
        print( "Swapping %d and %d." % (index1,index2) )
        l.swap_particles(index1,index2)
    
    
    print l.to_string()
    print l.to_povray_string()
    filestr = 'S_of_q-%02d.png' % (num_defects)
    int_vals = l.plot_structure_factor_isotropic( q_s, peak1, filename=filestr, background=None, max_hkl=8, ylog=False )
        
        
        
        