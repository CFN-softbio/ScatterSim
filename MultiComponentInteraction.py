#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################
# March 29, 2011
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# This file contains some odds & ends that are useful for quickly
# working with ScatterSim
###################################################################

from math import radians, degrees, pi, sin, cos, asin, sqrt, exp, tan, log, atan, floor  # basic math 
import pickle
import pylab
import glob




def overlay_ops_old(data_file, plot=True, plot_offset=1.4, plot_txt_offset=0.1, output_txt=True, scaling=None):
    
    data_name = data_file[:-4]

    # Load data
    filename = 'fit_dat-data-'+data_file+'.pkl'
    fin = open( filename )
    s_of_q = pickle.load( fin )
    fin.close()
    experimental_q = s_of_q[:,0]
    experimental_int = s_of_q[:,1]


    # Load three fits
    filename_SC = 'fit_dat-SimpleCubic.pkl'
    filename_FCC = 'fit_dat-FCCLattice.pkl'
    filename_BCC = 'fit_dat-BCCLattice.pkl'

    fin = open( filename_SC )
    data_SC = pickle.load(fin)
    fin.close()
    SC_q = data_SC[0]
    SC_int = data_SC[1]

    fin = open( filename_FCC )
    data_FCC = pickle.load(fin)
    fin.close()
    FCC_q = data_FCC[0]
    FCC_int = data_FCC[1]

    fin = open( filename_BCC )
    data_BCC = pickle.load(fin)
    fin.close()
    BCC_q = data_BCC[0]
    BCC_int = data_BCC[1]
    
    if plot:
        # Plot
        overlay_plot( (experimental_q, experimental_int), (SC_q, SC_int), (FCC_q, FCC_int), (BCC_q, BCC_int), data_name=data_name, scaling=scaling, offset=plot_offset, txt_offset=plot_txt_offset )

    if output_txt:
        # Output text
        base_filename = data_file[:-4]
        output_text( data_SC, filename=base_filename+'-simulation_SC.txt' )
        output_text( data_FCC, filename=base_filename+'-simulation_FCC.txt' )
        output_text( data_BCC, filename=base_filename+'-simulation_BCC.txt' )



def overlay_ops(data_file, plot=True, plot_offset=1.4, plot_txt_offset=0.1, output_txt=True, scaling=None):
    
    data_name = data_file[:-4]

    # Load data
    data_filename = 'fit_dat-data-'+data_file+'.pkl'
    fin = open( data_filename )
    s_of_q = pickle.load( fin )
    fin.close()
    experimental_q = s_of_q[:,0]
    experimental_int = s_of_q[:,1]

    curves = [ [data_file, experimental_q, experimental_int] ]


    # Load fits
    
    filenames = []
    for filename in glob.glob('*.pkl'):
        if filename!=data_filename:
            filenames.append(filename)
    filenames.sort()
    
    for filename in filenames:
        
        sim_type = filename[8:-4]
        
        fin = open( filename )
        data = pickle.load(fin)
        fin.close()
        q = data[0]
        intensity = data[1]
        curves.append( [sim_type, q, intensity] )
        
    
    if plot:
        # Plot
        overlay_plot( curves, scaling=scaling, offset=plot_offset, txt_offset=plot_txt_offset )

    if output_txt:
        # Output text
        base_filename = data_file[:-4]
        
        for curve in curves:
            name, q, intensity = curve
            data = [q, intensity]
        
            output_text( data, filename=base_filename+'-'+name+'.txt' )




def overlay_plot_old( (x1, y1), (x2,y2), (x3,y3), (x4,y4), filename='overlay.png', data_name=None, scaling=None, offset=1.4, txt_offset=0.1, xlog=False, ylog=False ):

    pylab.rcParams['axes.labelsize'] = 34
    pylab.rcParams['xtick.labelsize'] = 'x-large'
    pylab.rcParams['ytick.labelsize'] = 'x-large'


    fig = pylab.figure( figsize=(6,9) )
    fig.subplots_adjust(left=0.18, bottom=0.1, right=0.94, top=0.97, wspace=0.2, hspace=0.2)
    ax = pylab.subplot(111)

    y_off = 0.0
    y = [ y_val + y_off for y_val in y1 ]
    pylab.plot( x1, y, 'o', color='k', linewidth=3.0, markersize=7.0 )
    if data_name!=None:
        pylab.text( 0.02, y_off+txt_offset, data_name, verticalalignment='center', size=14 )
    
    y_off += offset
    y = [ y_val + y_off for y_val in y2 ]
    pylab.plot( x2, y, '-', color='b', linewidth=3.0 )
    pylab.text( 0.02, y_off+txt_offset, 'SC', verticalalignment='center', size=14 )
    
    y_off += offset
    y = [ y_val + y_off for y_val in y3 ]
    pylab.plot( x3, y, '-', color='b', linewidth=3.0 )
    pylab.text( 0.02, y_off+txt_offset, 'FCC', verticalalignment='center', size=14 )
    
    y_off += offset
    y = [ y_val + y_off for y_val in y4 ]
    pylab.plot( x4, y, '-', color='b', linewidth=3.0 )
    pylab.text( 0.02, y_off+txt_offset, 'BCC', verticalalignment='center', size=14 )


    pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
    pylab.ylabel( r'$S(q)$' )
    
    xi, xf, yi, yf = ax.axis()
    if scaling!=None:
        xi, xf, yi, yf = scaling
    else:
        xi = 0
        yi = 0
    ax.axis( (xi, xf, yi, yf) )
    
    ax.set_xticks( [0, 0.1, 0.2, 0.3] )
    
    if ylog:
        pylab.semilogy()
    
    pylab.savefig( filename )
    

def overlay_plot( curves, filename='overlay.png', scaling=None, offset=1.4, txt_offset=0.1, xlog=False, ylog=False ):

    pylab.rcParams['axes.labelsize'] = 34
    pylab.rcParams['xtick.labelsize'] = 'x-large'
    pylab.rcParams['ytick.labelsize'] = 'x-large'


    fig = pylab.figure( figsize=(6,9) )
    fig.subplots_adjust(left=0.18, bottom=0.1, right=0.94, top=0.97, wspace=0.2, hspace=0.2)
    ax = pylab.subplot(111)

    y_off = 0.0
    
    for i, curve in enumerate(curves):
        name, x_data, y_data = curve
        y_off = i*offset
    
        y = [ y_val + y_off for y_val in y_data ]
        if i==0:
            pylab.plot( x_data, y, 'o', color='k', linewidth=3.0, markersize=7.0 )
        else:
            pylab.plot( x_data, y, '-', color='b', linewidth=3.0 )
        pylab.text( 0.02, y_off+txt_offset, name, verticalalignment='center', size=14 )
        


    pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
    pylab.ylabel( r'$S(q)$' )
    
    xi, xf, yi, yf = ax.axis()
    if scaling!=None:
        xi, xf, yi, yf = scaling
    else:
        xi = 0
        yi = 0
    ax.axis( (xi, xf, yi, yf) )
    
    ax.set_xticks( [0, 0.1, 0.2, 0.3] )
    
    if ylog:
        pylab.semilogy()
    
    pylab.savefig( filename )
    


def output_text( data, filename='out.txt'):
    print( 'Storing data for: ' + filename )
    
    q_values = data[0]
    int_values = data[1]

    fout = open( filename, 'w' )
    for i, (q, intensity) in enumerate(zip(q_values, int_values)):
        fout.write( str(q) + '\t' + str(intensity) + '\n' )
    fout.close()

    
    
    
def compare_plot( datas, filename='compare.png', scaling=None, offset=0.0, txt_offset=0.1, xlog=False, ylog=False ):

    pylab.rcParams['axes.labelsize'] = 34
    pylab.rcParams['xtick.labelsize'] = 'x-large'
    pylab.rcParams['ytick.labelsize'] = 'x-large'


    fig = pylab.figure( figsize=(8,6) )
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.97, wspace=0.2, hspace=0.2)
    ax = pylab.subplot(111)

    y_off = 0.0

    color_list = [ (0,0,0),
                    (0.4,0.4,0.4),
                    (0.6,0.6,0.6), 
                    (0.7,0.7,0.7), 
                    ]

    for i, data in enumerate(datas):
        x = data[0]
        y = [ y_val + y_off for y_val in data[1] ]
        
        icol = i%len(color_list)
        pylab.plot( x, y, '-', color=color_list[icol], linewidth=3.0, markersize=7.0 )
        
        #pylab.text( 0.02, y_off+txt_offset, data_name, verticalalignment='center', size=14 )
        
        y_off += offset
        


    pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$' )
    pylab.ylabel( r'$S(q)$' )
    
    xi, xf, yi, yf = ax.axis()
    if scaling!=None:
        xi, xf, yi, yf = scaling
    else:
        xi = 0
        yi = 0
    ax.axis( (xi, xf, yi, yf) )
    
    #ax.set_xticks( [0, 0.1, 0.2, 0.3] )
    
    if ylog:
        pylab.semilogy()
    
    pylab.savefig( filename )
    






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
        
    

def plot_lattice_peaks( lattice, title=None, filename='lattice.png', peak_limit=10, print_peaks=True, lorentz_factor=True, fundamental_index=1):

    list1d = lattice.iterate_over_hkl_1d()

    x = []
    y = []
    labels = []

    if print_peaks:
        if lorentz_factor:
            print( "peak\tq value       \th,k,l\tm\tf\tintensity\tintensity_scaled" )
        else:
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
            
            intensity_rescaled = intensity/(q*82)
            
            # Skip 0,0,0
            if not (hn==0 and kn==0 and ln==0):
                if print_peaks:
                    if lorentz_factor:
                        print( "%d:\t%.12f\t%d,%d,%d\t%d\t%d\t%d\t%f" % (i, q, hn, kn, ln, m, f, intensity, intensity_rescaled) )
                    else:
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
