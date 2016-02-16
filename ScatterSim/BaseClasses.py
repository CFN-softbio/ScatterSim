# -*- coding: utf-8 -*-
###################################################################
# BaseClasses.py
# version 0.0.5
# April 12, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# Description:
#  This file includes a variety of global base definitions for
# building scattering scripts. Importantly, it includes the base
# classes, which can then be extended by defining inheriting classes
# that implement more specific features. The classes defined are:
#  - Scattering : Holds scattering data (including corresponding
#                 experimental parameters and q-range). Extensions
#                 that cover some specific cases (1D, 2D, etc.) are
#                 defined.
#  - Simulation : Performs a simulation, which may be as simple as
#                 just invoking a model, or more complex, such as
#                 varying parameters and averaging results.
#  - Model :      Implements a particular modeling formalism for
#                 returning reciprocal-space intensity from certain
#                 real-space model parameters (e.g. a real-space
#                 potential box).
#  - Potential :  Defines a real-space potential.
#  - SimulationBox: An intermediate object that stores a discretized
#                 version of a real-space potential inside a 3D 'box'.
#  - Experimental: Holds some information about an experimental setup.
###################################################################



# Import
###################################################################


from math import radians, degrees, pi, sin, cos, asin, sqrt, exp, tan, log, atan, floor  # basic math
import physcon as pc
keV_to_Joule = pc.e*1000.0
from cmath import exp as cexp, sqrt as csqrt    # complex-number math
import numpy                                    # Numerical Python

import pylab as pylab                           # for matplotlib plotting

import copy                                     # For creating copies of Python objects



import random                                   # For selecting pseudo-random numbers


# Global definitions
###################################################################

#pylab.rcParams['backend'] = 'WXAgg'
pylab.rcParams['axes.labelsize'] = 'x-large'
pylab.rcParams['xtick.labelsize'] = 'large'
pylab.rcParams['ytick.labelsize'] = 'large'


dimensions = ['x', 'y', 'z']

global_verbose = True
verbose_depth = 4




# Performance
###################################################################
import time

global_start_time = time.time()
def timing( reset=False ):
    """Prints out timing information."""
    
    global global_start_time
    
    if reset:
        global_start_time = time.time()
        
    elapsed = (time.time()-global_start_time)
    elapsed_string = '%3.2f'%elapsed    
    status_print( "[[Timing: "+elapsed_string+'s]]', depth=0 )
    
    


# Convenience functions
###################################################################


def status_print( message, args={}, depth=2 ):
    """Print out a status message during execution.
    This only outputs if 'verbose' is set."""
    
    try:
        if global_verbose or args['verbose']:
            if depth <= verbose_depth:
                if depth>0:
                    indent = "    "*depth
                else:
                    indent = ""
                print( indent + message )
    except KeyError:
        pass


def value_at( target, x_array, y_array ):
    # xtarget, ytarget = value_at( xtarget, x_array, y_array )
    
 
    x = numpy.asarray(x_array)
    y = numpy.asarray(y_array)

    # Sort
    indices = numpy.argsort( x )
    x_sorted = x[indices]
    y_sorted = y[indices]

    # Search through x for the target
    idx = numpy.where( x_sorted>target )[0][0]
    xcur = x_sorted[idx]
    ycur = y_sorted[idx]
    
    return xcur, ycur


# Define an iterator that enumerates backwards
def reverse_enumerate( sequence ):
    for iz in range(len(sequence)-1, -1, -1):
        yield iz, sequence[iz]


def log_scale (value, gamma=0.3):
    
    # Note that gamma=1.0 
    c = exp(1/gamma) - 1
    return gamma*log(value*c + 1)

def log_scale_array (data, gamma=0.3):
    c = exp(1/gamma) - 1
    return gamma*numpy.log(data*c + 1)


def randarange(stop, start=0.0, step=1.0):
    """Returns a random float in the specified range.
    The endpoints are inclusive (assuming the step-size
    matches-up properly)."""
    return randarange(start, stop, step)
   
def randarange(start, stop, step=1.0):
    """Returns a random float in the specified range.
    The endpoints are inclusive (assuming the step-size
    matches-up properly)."""
    n = int( (stop-start)/step )
    return start + step*random.randint(0,n)


def compare_arrays( array1, array2, tolerance=1e-8 ):
    if len(array1)!=len(array2):
        return False
    else:
        for v1, v2 in zip(array1, array2):
            if abs( (v1-v2)/v1 )>tolerance:
                return False
        return True
        
def padded_string( string, length=10, pad=" "):
    """Increase the length of a string, using the pad character."""
    num_pads = length-len(string)
    num_pads = max(0, num_pads) # No negative values!
    
    return string+pad*num_pads



def sinc(x):
    if x==0:
        return 1.0
    else:
        return sin(x)/x

def arrays_equal( a, b):
    """Returns true if the two (numpy) arrays are identical (every element matches)."""
    if len(a)==len(b):
        return numpy.all( numpy.equal( a, b ) )
    
    return False

# A new datatype: defaultdict, which automatically
# creates empty dictionaries...
from collections import defaultdict
def make_inner():
   return defaultdict(lambda: defaultdict(make_inner))



# Scattering Objects
###################################################################

# TODO: Scattering classes need cleanup. E.g. the "to_string" for extended
# classes should be implemented. Many methods are missing from inherited classes.

class Scattering(object):
    """Represents scattering data."""
    
    dimension = 0
    
    def __init__(self, q, value, Experimental=None):
        if isinstance(q, (float, int)):
            self.q = q
        else:
            if len(q)==1:
                self.q = q[0]
            elif len(q)==2:
                self.qr = q[0]
                self.qz = q[1]
                self.q = sqrt( self.qr**2 + self.qz**2 )
            elif len(q)==3:
                self.qx = q[0]
                self.qy = q[1]
                self.qz = q[2]
                self.q = sqrt( self.qx**2 + self.qy**2 + self.qz**2 )
            else:
                print( "ERROR: Scattering __init__ got a q with too many values" )
                
        self.data = value
        self.Experimental=Experimental
    
    def get_dimensionality(self):
        """Returns the dimensionality of the scattering data:
        0 for a single data point
        1 for a line or arc
        2 for a 2D grid of scattering data"""
        return self.dimension
    
    def get_data(self):
        """Returns the intensity data for this scattering data point."""
        return self.data
        
    def get_qs(self):
        """Returns the q of this scattering data point."""
        return self.q
        
    def plot(self, filename='scattering.png'):
        print( self.to_string )

    def save(self, filename='data.dat'):
        
        if dimension == 0:
            fout = open( filename, "w" )
            fout.write( self.data )
            fout.close()
        elif dimension == 1:
            numpy.savetxt( filename, self.data )
        elif dimension == 2:
            numpy.savetxt( filename, self.data )
        else:
            print( "ERROR: Scattering save method got an unknown dimension value." )
        

    def to_string(self):
        """Returns a string describing the scattering object."""
        s = "Scattering data point: q = %f, intensity = %f" % (self.q, self.data)
        return s



class Scattering0D(Scattering):
    """Represents a single scattering data point."""

    dimension = 0

    def plot(self, filename='scattering0D.png'):
        print( self.to_string )


class Scattering1D(Scattering):
    """Represents one-dimensional scattering data."""
    
    dimension = 1
    
    def plot(self, filename='scattering1D.png'):
        
        fig = pylab.figure()
        pylab.plot( self.q, self.data )
        pylab.xlabel( 'q (1/nm)' )
        pylab.ylabel( 'Intensity (a.u.)' )
        
        pylab.savefig( filename )


class Scattering2D(Scattering):
    """Represents 2D scattering data."""
    
    dimension = 2
    
    def __init__(self, qr, qz, data, Experimental=None):
        self.qr = qr
        self.qz = qz
        self.data = data
        self.Experimental=Experimental
    
    def x_values(self):
        return numpy.linspace( self.qr[0], self.qr[1], num=self.qr[2] )

    def y_values(self):
        return numpy.linspace( self.qz[0], self.qz[1], num=self.qz[2] )

    def axis_grids(self):
        return numpy.meshgrid( self.x_values(), self.y_values() )

    def plot(self, filename='scattering2D.png'):
        """Plots the scattering data."""
        
        fig = pylab.figure()
        
        pylab.pcolor( self.x_values(), self.y_values(), self.data )
        pylab.xlabel(r'$Q_r \hspace{0.5} (\rm{nm}^{-1})$')
        pylab.ylabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
        pylab.axis('tight')
        
        pylab.savefig( filename )


class Scattering2DAngular(Scattering2D):
    """Represents 2D scattering data obtained by sweeping-out angles."""
    
    dimension = 2
    
    def __init__(self, theta_incident, theta_scan, phi_scan, data, Experimental=None):
        self.theta_incident = theta_incident
        self.theta_scan = theta_scan
        self.phi_scan = phi_scan
        self.data = data
        self.Experimental=Experimental

    def x_values(self):
        return numpy.linspace( self.phi_scan[0], self.phi_scan[1], num=self.phi_scan[2] )
    
    def y_values(self):
        return numpy.linspace( self.theta_scan[0], self.theta_scan[1], num=self.theta_scan[2] )

    def q_inplane_values(self):
        
        k = self.Experimental.k_wavevector
        q_values = numpy.empty( (self.phi_scan[2]), numpy.float )
        
        for iphi, phi in enumerate( self.x_values() ):
            q_values[iphi] = k*sin(radians(phi))
        
        return q_values

    def q_outofplane_values(self):
        
        k = self.Experimental.k_wavevector
        q_values = numpy.empty( (self.theta_scan[2]), numpy.float )
        
        for itheta, theta in enumerate( self.y_values() ):
            q_values[itheta] = k*( sin(radians(theta)) + sin(radians(self.theta_incident)) )
        
        return q_values




    def limits(self):
        """Returns the minimum and maximum potential inside the box."""
        
        vmin=numpy.min( self.data )
        vmax=numpy.max( self.data )

        return vmin, vmax


    def plot(self, filename='scattering2D.png', log_scaling=False, log_gamma=0.3, plotmin=0.0, plotmax=1.0, mirror=False):
        """Plots the scattering data."""
        
        status_print( "Plotting scattering data...", sargs, depth=2 )
        
        fig = pylab.figure()
        
        if log_scaling:
            data = log_scale_array( self.data, gamma=log_gamma )
            vmin = numpy.min( data )
            vmax = numpy.max( data )
        else:
            data = self.data
            vmin, vmax = self.limits()
        
        
        vspan = vmax-vmin
        vmin = vmin + plotmin*vspan
        vmax = vmin + plotmax*vspan

        pylab.pcolor( self.x_values(), self.y_values(), numpy.transpose(data), vmin=vmin, vmax=vmax )
        pylab.xlabel( r'$\phi_s \rm{(\degree)}$' )
        pylab.ylabel( r'$\theta_s \rm{(\degree)}$' )
        pylab.axis('tight')
        
        pylab.savefig( filename )


    
    def plotQ(self, filename='scattering2D.png', log_scaling=False, log_gamma=0.3, plotmin=0.0, plotmax=1.0, mirror=False):
        """Plots the scattering data."""
        
        status_print( "Plotting scattering data...", depth=2 )
        
        fig = pylab.figure()
        
        if log_scaling:
            data = log_scale_array( self.data, gamma=log_gamma )
            vmin = numpy.min( data )
            vmax = numpy.max( data )
        else:
            data = self.data
            vmin, vmax = self.limits()
        
        
        vspan = vmax-vmin
        vmin = vmin + plotmin*vspan
        vmax = vmin + plotmax*vspan

        q_i = self.q_inplane_values()
        q_o = self.q_outofplane_values()
        data = numpy.transpose(data)


        if mirror:
            q_i = numpy.concatenate( ( numpy.multiply(numpy.flipud(q_i[1:]),-1), q_i ) )
            data = numpy.concatenate( (numpy.fliplr(data[:,1:]),data), axis=1 )

        
        pylab.pcolor( q_i, q_o, data, vmin=vmin, vmax=vmax )
        pylab.xlabel( r'$Q_y \hspace{0.5} (\rm{nm}^{-1})$' )
        pylab.ylabel( r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$' )
        pylab.axis('equal')
        pylab.axis('tight')
        
        
        pylab.savefig( filename )



class Scattering2DEwald(Scattering2DAngular):
    """Represents 2D scattering data for a subset of the Ewald sphere.
    (Data is a function of q.)"""
    
    dimension = 2
    
    def __init__(self, theta_incident, qy_s, qz_s, data, Experimental=None):
        
        self.theta_incident = theta_incident
        
        self.qy_s = qy_s
        self.qz_s = qz_s
        
        self.data = data
        self.Experimental=Experimental

    def x_values(self):
        return numpy.linspace( self.qy_s[0], self.qy_s[1], num=self.qy_s[2] )
    
    def y_values(self):
        return numpy.linspace( self.qz_s[0], self.qz_s[1], num=self.qz_s[2] )

    def q_inplane_values(self):
        return self.x_values()

    def q_outofplane_values(self):
        return self.y_values()




    def plot(self, filename='scattering2D.png', log_scaling=False, log_gamma=0.3, plotmin=0.0, plotmax=1.0, mirror=False):
        self.plotQ( filename, log_scaling=log_scaling, log_gamma=log_gamma, plotmin=plotmin, plotmax=plotmax, mirror=mirror )



# Simulation Object
###################################################################

class Simulation(object):
    """A simulation generates intensity data for a series
    of positions in reciprocal-space. During instantiation,
    a model and a potential are defined. The model defines
    what formalism will be used to simulate (Born approximation,
    DWBA, Integral method, etc.); the potential defines the
    real-space density profile."""
    
    
    # Default simulation arguments
    sargs = { \
        'verbose': True, \
        'parallelize': False, \
        }
    

    box_uptodate = False


    def __init__(self, Model, Potential, Box, Experimental, sargs={}, margs={}, pargs={}, bargs={}):
        """Initializes the simulation with a given model, potential,
        and simulation box. Optional arguments should be passed to
        override the defaults and define these objects."""
        
        self.Model = Model
        self.Potential = Potential
        self.Box = Box
        self.Experimental = Experimental
        
        self.sargs.update(sargs)
        
        
        # Prepare for parallel computation
        if 'parallelize' in self.sargs and self.sargs['parallelize']:
            from multiprocessing import Pool
            
            if 'num_processes' in self.sargs:
                self.pool = Pool(processes=self.sargs['num_processes'])
            else:
                self.pool = Pool()
            

        # Do some sanity checks
        if 'z_box_span' in self.Potential.pargs and self.Potential.pargs['z_box_span']<self.Box.limits['z']['span']:
            print( "WARNING: The specified potential function doesn't fit into the specified simulation box." )

        
    def check_uptodate(self):
        """Checks whether simulation box has been filled with the
        target potential. If not, it does it."""
        if not self.box_uptodate:
            self.Box.fill( self.Potential )
            self.box_uptodate = True
    
    def clear(self):
        """Forces the simulation object to recompute internal quantities."""
        self.box_uptodate = False
        

    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        
        self.check_uptodate()

        return Scattering0D( (qx,qy,qz), self.Model.q_value(qx, qy, qz) )

    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""
        
        self.check_uptodate()
        
        # TODO: Return a Scattering object
        return self.Model.angular_value(theta_incident, theta_scan, phi_scan)

    #def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
    def angular_2d(self, theta_incident, thtuple,phituple):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples."""
        theta_scan_min, theta_scan_max, theta_scan_num = thtuple
        phi_scan_min, phi_scan_max, phi_scan_num = phituple

        self.check_uptodate()

        data = self.Model.angular_2d( theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num) )
        scatter = Scattering2DAngular( \
                        theta_incident, \
                        ( theta_scan_min, theta_scan_max, theta_scan_num ), \
                        ( phi_scan_min, phi_scan_max, phi_scan_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter

    #def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
    def ewald_2d(self, theta_incident, qytuple, qztuple):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""
        (qy_min, qy_max, qy_num) = qytuple
        (qz_min, qz_max, qz_num) = qztuple
        self.check_uptodate()

        data = self.Model.ewald_2d( theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num) )
        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter


    def to_string(self):
        """Returns a string describing
        the simulation."""
        
        s = self.Model.to_string() + "\n"
        s += self.Potential.to_string() + "\n"
        s += self.Box.to_string() + "\n"
        
        return s



# This method runs a model calculation and returns the result.
# This method only exists to simplify parallal-processing. The function
# can be called repeatedly because each instance creates separate
# Box and Model objects.
def GenerateSimulation(Model, Potential, arglist, scatter_type='ewald_2d', sargs={}, margs={}, pargs={}, bargs={}):
    """Runs a calculation based on the Potential and Model 
    that are passed."""
    
    if pargs == {}:
        Potential_use = Potential
    else:
        # We can use the Potential object as-is, unless some
        # of its arguments need to be overriden.
        Potential_use = copy.deepcopy( Potential )
        Potential_use.rebuild( pargs=pargs )

    # We need to create a separate Box and Model object on each calling of this
    # function, to prevent parallel threads from trying to modify the same instances.
    
    Box_use = SimulationBox( bargs )
    Box_use.fill( Potential_use )
    
    # TODO: remove this testing code...
    #namestr = 'potential-eta'+str(pargs['eta'])+'phi'+str(pargs['phi'])+'theta'+str(pargs['theta'])
    #Box_use.plot_potential_bisect( filename=namestr+'.png', sargs=sargs, plotmin=0.2, plotmax=0.8 )
    
    Model_use = copy.deepcopy( Model )
    Model_use.set_Box( Box_use )
    
    # TODO: Must handle all the other possible types of scattering requests
    if scatter_type=='ewald_2d':
        return Model_use.ewald_2d( *arglist )





# Model Object
###################################################################

# TODO: This object should probably return arrays, rather than floats,
# for the 2d methods. (Even though they never really get called...)
class Model(object):
    """Implements a particular method for converting real-space
    potential into reciprocal-space intensity."""
    
    margs = { 'reflection_computation': False }
    
    def __init__(self, margs={}):
        """Prepares the model object."""
        self.margs.update(margs)

    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        return 0.0

    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""
        return 0.0

    #def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
    def angular_2d(self, theta_incident, theta_scan_tuple, phi_scan_tuple):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples."""
        (theta_scan_min, theta_scan_max, theta_scan_num) = theta_scan_tuple
        (phi_scan_min, phi_scan_max, phi_scan_num) = phi_scan_tuple
        return 0.0
        
    #def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max,qz_num)):
    def ewald_2d(self, theta_incident, qytuple, qztuple):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range of q-values is specified as tubles."""
        (qy_min, qy_max, qy_num) = qytuple
        (qz_min, qz_max,qz_num) = qztuple
        return 0.0


    def to_string(self):
        """Returns a string describing the model."""
        s = "Base Model (does nothing; just returns zero for all reciprocal-space)."
        return s
        



# Potential Object
###################################################################

class Potential(object):
    """Defines a real-space potential. That is, a method
    of calculating the density for the film to be simulated."""
    
    conversion_factor = 1E-4        # Converts units from 1E-6 A^-2 into nm^-2
    
    pargs = {   'rho_ambient': 0.0, \
                'rho_film': 10.0, \
                'rho1': 15.0, \
                'rho2': 10.0, \
                'rho_substrate': 20.1, \
                'film_thickness': 100.0, \
                }
    
    def __init__(self, pargs={}, seed=None):
        self.pargs.update(pargs)

    def rebuild(self, pargs={}, seed=None):
        """Allows a Potential object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies 
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)
        
        if seed:
            self.seed=seed

    def V(self, in_x, in_y, in_z, rotation_elements=None):
        return self.conversion_factor*0.0


    def to_string(self):
        """Returns a string describing the potential."""
        s = "Base potential (zero potential everywhere)."
        return s









# SimulationBox Object
###################################################################        

class SimulationBox(object):
    """Represents a conceptual real-space 'box' that will contain
    a particular realization of a potential (density profile). In
    other words it is a box containing the sample thin film structure."""

    # Default box parameters
    limits = { \
        'x': { 'start':0.0, 'end':300.0, 'num':128 }, \
        'y': { 'start':0.0, 'end':300.0, 'num':128 }, \
        'z': { 'start':0.0, 'end':100.0, 'num':32 }, \
        }

    V_z = []


    # Setup methods
    ################################
    def __init__(self, bargs={}):
        """Allocates memory for a new box. Starts 'empty'."""
       
        self.limits.update(bargs)
        for dimension in dimensions:
            self.limits[dimension]['span'] = (self.limits[dimension]['end'] - self.limits[dimension]['start'])
            self.limits[dimension]['step'] = self.limits[dimension]['span']/self.limits[dimension]['num']
            
    
        self.V_box = numpy.empty( (self.limits['x']['num'], self.limits['y']['num'], self.limits['z']['num']), numpy.complex )

    def rebuild(self, bargs={}):
        """Forces the object to rebuild internal structures (e.g.
        if you change one of the box limit parameters)."""
        
        self.limits.update(bargs)
        for dimension in dimensions:
            self.limits[dimension]['span'] = (self.limits[dimension]['end'] - self.limits[dimension]['start'])
            self.limits[dimension]['step'] = self.limits[dimension]['span']/self.limits[dimension]['num']
            
    
        self.V_box = numpy.empty( (self.limits['x']['num'], self.limits['y']['num'], self.limits['z']['num']), numpy.complex )


    # Axis methods
    ################################
    def size(self):
        """Returns a tuple with the size of the box."""
        return self.limits['x']['num'], self.limits['y']['num'], self.limits['z']['num']

    def axis(self, dimension):
        """Iterates along a given dimension of the simulation box."""
        for index in range(self.limits[dimension]['num']):
            value = self.limits[dimension]['start'] + index*self.limits[dimension]['step']
            yield index, value

    def axis_values(self, dimension):
        """Returns a vector of the values along a given dimension of the simulation box."""
        return numpy.arange( self.limits[dimension]['start'], self.limits[dimension]['end'], self.limits[dimension]['step'] )

    def axes(self):
        for ix, x in self.axis('x'):
            for iy, y in self.axis('y'):
                for iz, z in self.axis('z'):
                    yield ix, x, iy, y, iz, z

    def index(self, dimension, value):
        """Returns the index value for a given value along a dimension.
        (Nearest integer.)"""
        
        return int( ((value-self.limits[dimension]['start'])/self.limits[dimension]['span'])*self.limits[dimension]['num'] )    

    def dz(self):
        """Convenience function for getting the partition size in z."""
        return self.limits['z']['step']

    def aspect_ratio(self, dimension1, dimension2):
        return (1.0*self.limits[dimension1]['span'])/(1.0*self.limits[dimension2]['span'])


    # Potential lookup methods
    ################################
    def V_limits(self, return_complex=False):
        """Returns the minimum and maximum potential inside the box."""
        
        vmin=numpy.min( self.V_box[:,:,:] )
        vmax=numpy.max( self.V_box[:,:,:] )

        if return_complex:
            return vmin, vmax
        else:
            return vmin.real, vmax.real

    def V_z_average(self):
        """Returns the z-average of the potential in the box."""
        
        if self.V_z==[]:
            # We haven't calculated the average yet...
            self.V_z = numpy.empty( (self.limits['z']['num']), numpy.complex )
            for iz, z in self.axis('z'):
                self.V_z[iz] = numpy.average(self.V_box[:,:,iz])

        return self.V_z

    def get_slices(self, axis_a, axis_b, slice_value, return_complex=False ):
        """Returns a 2d-slice through the box, giving the potentials
        in a 2d-array. Also returns corresponding arrays of the
        axis values (e.g. for plotting)."""

        if (axis_a=='x' and axis_b=='y') or (axis_a=='y' and axis_b=='x'):
            index = self.index('z', slice_value)
            slice_2d = self.V_box[:,:,index]
        elif (axis_a=='x' and axis_b=='z') or (axis_a=='z' and axis_b=='x'):
            index = self.index('y', slice_value)
            slice_2d = self.V_box[:,index,:]
        elif (axis_a=='y' and axis_b=='z') or (axis_a=='z' and axis_b=='y'):
            index = self.index('x', slice_value)
            slice_2d = self.V_box[index,:,:]
        else:
            print( "ERROR: get_slice() got unknown axis/dimension." )
            return [],[],[]

        a_values, b_values = numpy.meshgrid( self.axis_values(axis_a) , self.axis_values(axis_b) )

        if not return_complex:
            slice_2d = slice_2d.real

        return a_values, b_values, slice_2d


    def get_V_slice(self, dimension, value, return_complex=False):
        """Returns the current potential in the box, for
        a particular value along a dimensions (a 2D slice)."""
        
        if return_complex:
            if dimension=='x':
                return self.V_box[ self.index(dimension,value), :, : ]
            elif dimension=='y':
                return self.V_box[ :, self.index(dimension,value) ,: ]
            elif dimension=='z':
                return self.V_box[ :, :, self.index(dimension,value) ]
            else:
                print( "ERROR: get_V_slice() got uknown dimension." )
                return []
        else:
            if dimension=='x':
                return self.V_box[ self.index(dimension,value), :, : ].real
            elif dimension=='y':
                return self.V_box[ :, self.index(dimension,value) ,: ].real
            elif dimension=='z':
                return self.V_box[ :, :, self.index(dimension,value) ].real
            else:
                print( "ERROR: get_V_slice() got uknown dimension." )
                return []

    # Potential set methods
    ################################
    def fill(self, Potential):
        """Fills the box with values based on a particular Potential
        object."""
        
        status_print( "Filling box with potential...", depth=2 )
        
        self.V_z = []
        
        for ix, x, iy, y, iz, z in self.axes():
            self.V_box[ix,iy,iz] = Potential.V( x, y, z )



    # Reflectivity calculation
    ################################
    def Calculate_TR(self, angle, k_wavevector=1.0 ):
        """Returns the wavevectors and optical coefficients
        (transmission, reflection) for each layer in the z-direction.
        These values are those calculated internally to compute
        reflectivity, and can also be used to compute the
        wavefunction in each z-layer."""
        
        
        # Calculate the wavevectors for the incident beam for each z-layer
        q_list = numpy.empty( (self.limits['z']['num']), numpy.complex  )
        q_list[0] = k_wavevector*sin( angle )
        E_0 = q_list[0]**2
        for iq, V in zip(range(len(q_list)), self.V_z_average()):
            if iq>0:
                q_list[iq] = csqrt( E_0 - V/4 )


        # Calculate the transmission and reflection coefficients for each z-layer
        M_accumulator = numpy.identity( 2, complex )
        M = numpy.empty( (self.limits['z']['num'],2,2), numpy.complex )
        Z_z = self.axis_values('z')
        for iz, Z in reverse_enumerate( Z_z ):
            if iz > 0:
                qr = q_list[iz-1]/q_list[iz]
                q_plus = (1+qr)/2.0
                q_minus = (1-qr)/2.0
                e_plus = cexp( 1j*(q_list[iz-1]+q_list[iz])*Z )
                e_minus = cexp( 1j*(q_list[iz-1]-q_list[iz])*Z )
                
                M[iz] = numpy.array( [[ q_plus*e_minus, q_minus/e_plus ],
                                    [ q_minus*e_plus, q_plus/e_minus ]] )

                # Matrix multiplication of this layer into the accumulator
                M_accumulator = numpy.dot( M_accumulator, M[iz] )


        TR = numpy.empty( (self.limits['z']['num'],2,1), numpy.complex )
        TR[0] = numpy.array( [[ 1.0 ],
                            [ -1.0*M_accumulator[1,0]/M_accumulator[1,1] ]] )

        #TR[0] = numpy.array( [[ M_accumulator[0,0]-(M_accumulator[0,1]*M_accumulator[1,0])/M_accumulator[1,1] ],
        #                      [ -1.0*M_accumulator[1,0]/M_accumulator[1,1] ]] )


        for iz in range( 1, len(Z_z) ):
            TR[iz] = numpy.dot( M[iz], TR[iz-1] )

        # R[final_layer] = 0
        TR[-1,1] = 0.0

        return q_list, TR


    def Calculate_Reflectivity(self, theta_min, theta_max, theta_step, k_wavevector=1.0 ):
        """Calculates reflectivity values across a span of reflection angles."""
        
        
        angles = numpy.arange( theta_min, theta_max, theta_step )
        qzs = [ 2*k_wavevector*sin(radians(angle)) for angle in angles ]
        
        # Refl holds the reflectivity in a 5-tuple:
        #  0  angle (degrees)
        #  1  qz (1/nm)
        #  2  q-incident (1/nm)
        #  3  Transmission
        #  4  Reflection
        refl = numpy.empty( (len(angles),5) )

        for ia, angle in enumerate(angles):
            qi, TR = self.Calculate_TR( radians(angle), k_wavevector=k_wavevector )
            refl[ia,0] = angle
            refl[ia,1] = qzs[ia]
            refl[ia,2] = qi[0]
            refl[ia,3] = TR[-1,0][0]*TR[-1,0][0].conjugate()
            refl[ia,4] = TR[0,1][0]*TR[0,1][0].conjugate()

        return refl

    def Calculate_ReflectivityQ(self, qz_min, qz_max, qz_step, k_wavevector=1.0 ):
        """Calculates reflectivity values across a span of qz values."""
        
        qzs = numpy.arange( qz_min, qz_max, qz_step )
        angles = [ degrees( asin( qz/(2*k_wavevector) ) ) for qz in qzs ]

        # Refl holds the reflectivity in a 5-tuple:
        #  0  angle (degrees)
        #  1  qz (1/nm)
        #  2  q-incident (1/nm)
        #  3  Transmission
        #  4  Reflection
        refl = numpy.empty( (len(angles),5) )

        for ia, angle in enumerate(angles):
            qi, TR = self.Calculate_TR( radians(angle), k_wavevector=k_wavevector )
            refl[ia,0] = angle
            refl[ia,1] = qzs[ia]
            refl[ia,2] = qi[0]
            refl[ia,3] = TR[-1,0][0]*TR[-1,0][0].conjugate()
            refl[ia,4] = TR[0,1][0]*TR[0,1][0].conjugate()

        return refl









    # Plotting methods
    ################################

    def plot_reflectivity(self, theta_min=0.001, theta_max=1.0, theta_step=0.001, k_wavevector=1.0, filename='reflectivity.png', plotQ=False ):
        """Plots reflectivity data for the provided range of angles."""

        refl = self.Calculate_Reflectivity(theta_min, theta_max, theta_step, k_wavevector=k_wavevector)
        self.plot_reflectivity_data(refl, k_wavevector=k_wavevector, filename=filename, plotQ=plotQ )


    def plot_reflectivityQ(self, q_min=0.001, q_max=1.5, q_step=0.001, k_wavevector=1.0, filename='reflectivity.png', plotQ=True ):
        """Plots reflectivity data for the provided range of qz values."""

        refl = self.Calculate_ReflectivityQ(q_min, q_max, q_step, k_wavevector=k_wavevector)
        self.plot_reflectivity_data(refl, k_wavevector=k_wavevector, filename=filename, plotQ=plotQ )

    def plot_reflectivity_data(self, refl, k_wavevector=1.0, filename='reflectivity.png', plotQ=True ):
        """Plots reflectivity data from a provided 2D array."""
        
        status_print( "Plotting reflectivity data...", depth=2 )
        
        fig = pylab.figure()
        #fig.subplots_adjust
        
        if plotQ:
            # Plot vs. q
            pylab.plot( refl[:,1], refl[:,4] )
            pylab.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
        else:
            # Plot vs. angle (in degrees)
            pylab.plot( refl[:,0], refl[:,4] )
            pylab.xlabel('angle (degrees)')
        
        pylab.ylabel('R')
        pylab.semilogy()
        pylab.savefig( filename )


    def plot_potential_bisect( self, filename='potential.png', sargs={}, plotmin=0.0, plotmax=1.0, cutting_distance=0.5 ):
        """Outputs a plot of the potential (density) in the box.
        Automatically shows slices that bisect the box in x, y,
        and z."""
        x = self.limits['x']['start'] + cutting_distance*(self.limits['x']['end']+self.limits['x']['start'])
        y = self.limits['y']['start'] + cutting_distance*(self.limits['y']['end']+self.limits['y']['start'])
        z = self.limits['z']['start'] + cutting_distance*(self.limits['z']['end']+self.limits['z']['start'])
        self.plot_potential( x, y, z, filename=filename, sargs={}, plotmin=plotmin, plotmax=plotmax )

    def plot_potential( self, x, y, z, filename='potential.png', sargs={}, plotmin=0.0, plotmax=1.0 ):
        """Outputs a plot of the potential (density) in the box.
        The x, y, and z values define where slices through the
        box are taken."""
        
        status_print( "Plotting potential function...", sargs, depth=2 )

        # Tweak plotting style
        pylab.rcParams['axes.labelsize'] = 'medium'
        pylab.rcParams['xtick.labelsize'] = 'medium'
        pylab.rcParams['ytick.labelsize'] = 'medium'
        
        
        # Get plotting limits from box
        vmin, vmax = self.V_limits()
        vspan = vmax-vmin
        vmin = vmin + plotmin*vspan
        vmax = vmin + plotmax*vspan

        xy_plotsize_h = 1.0
        xy_plotsize_v = xy_plotsize_h/self.aspect_ratio('x','y')

        border_right_top = 0.1
        border_left_bottom = 0.1

        plot_scaling = (xy_plotsize_h-2*border_left_bottom-border_right_top)/ \
                            (1 + 1/(self.aspect_ratio('x','y')*self.aspect_ratio('y','z')))

        if xy_plotsize_v > 1.0:
            plot_scaling /= xy_plotsize_v

        fig = pylab.figure( figsize=(9.5,9.5) )
        
        
        # xy slice (at a constant z)
        ################################
        slice_2d = self.get_V_slice( 'z', z, return_complex=True )
        
        sp3 = fig.add_axes( [ border_left_bottom,
                            border_left_bottom,
                            plot_scaling*xy_plotsize_h,
                            plot_scaling*xy_plotsize_v ]
                            )
        
        sp3.axvline( x, color='white', linestyle=':', linewidth=2.0 )
        sp3.axhline( y, color='white', linestyle=':', linewidth=2.0 )
        
        slice_2d = slice_2d.transpose()
        sp3.pcolor( self.axis_values('x'), self.axis_values('y'), slice_2d, vmin=vmin, vmax=vmax )
        pylab.xlabel('x (nm)')
        pylab.ylabel('y (nm)')
        pylab.axis('tight')


        # xz slice (at a constant y)
        ################################
        slice_2d = self.get_V_slice( 'y', y )

        sp1 = fig.add_axes( [ border_left_bottom,
                            plot_scaling*xy_plotsize_v + 2*border_left_bottom,
                            plot_scaling*xy_plotsize_h,
                            plot_scaling*xy_plotsize_h/self.aspect_ratio('x','z') ],
                            )
        
        sp1.axhline( z, color='white', linestyle=':', linewidth=2.0 )
        
        slice_2d = slice_2d.transpose()
        sp1.pcolor( self.axis_values('x'), self.axis_values('z'), slice_2d, vmin=vmin, vmax=vmax )
        pylab.xlabel('x (nm)')
        pylab.ylabel('z (nm)')
        pylab.axis('tight')


        # yz slice (at a constant x)
        ################################
        slice_2d = self.get_V_slice( 'x', x )
        
        sp4 = fig.add_axes( [ plot_scaling*xy_plotsize_h + 2*border_left_bottom,
                            border_left_bottom,
                            plot_scaling*xy_plotsize_v/self.aspect_ratio('y','z'),
                            plot_scaling*xy_plotsize_v ],
                            )

        sp4.axvline( z, color='white', linestyle=':', linewidth=2.0 )
        
        sp4.pcolor( self.axis_values('z'), self.axis_values('y'), slice_2d, vmin=vmin, vmax=vmax )
        pylab.xlabel('z (nm)')
        pylab.ylabel('y (nm)')
        pylab.axis('tight')
        
        
        # z-average
        ################################
        sp2 = fig.add_axes( [ plot_scaling*xy_plotsize_h + 2*border_left_bottom,
                            plot_scaling*xy_plotsize_v + 2*border_left_bottom,
                            plot_scaling*xy_plotsize_v/self.aspect_ratio('y','z'),
                            plot_scaling*xy_plotsize_h/self.aspect_ratio('x','z') ] )

        V_z = self.V_z_average()

        
        sp2.plot( self.axis_values('z'), 1E4*(V_z/(16*pi)) )
        pylab.xlabel('z (nm)')
        pylab.ylabel('V')
        
        
        pylab.savefig( filename )
        
        
        
        




    # Query methods
    ################################
    def to_string(self):
        """Returns a string describing the box."""
        s = "Simulation Box:\n"
        s += "\tdim\tstart\t\tend\t\tspan\t\tnum\t\tstep\n"
        for dimension in dimensions:
            s += '\t%s\t%f\t%f\t%f\t%f\t%f\n' % (dimension, self.limits[dimension]['start'], \
                                                self.limits[dimension]['end'], \
                                                self.limits[dimension]['span'], \
                                                self.limits[dimension]['num'], \
                                                self.limits[dimension]['step'])
        return s



# Experimental Object
###################################################################

class Experimental(object):
    """Holds the information about the experimental setup."""
    
    def __init__(self, wavelength=None, energy=None, aperture=1.0, detector_distance=5372.0, pixel_size=0.097, beam_size=0.1, sample_size=10, theta_incident=0.1):
        
        # Store values
        if energy!=None and wavelength==None:
            self.energy = energy                        # Units: keV
            # Use energy (in keV) to compute wavelength (in nm).
            wavelength = pc.h*pc.c/(energy*keV_to_Joule)    # Units: m
            self.wavelength = 1e9*wavelength                # Units: nm
        elif wavelength!=None:
            # Use supplied wavelength
            self.wavelength = wavelength                # Units: nm
            self.energy = ( pc.h*pc.c/(self.wavelength*1e-9) )/(keV_to_Joule)    # Units: keV
        else:
            # Use a default value for wavelength
            self.wavelength = 0.1                       # Units: nm
            self.energy = ( pc.h*pc.c/(self.wavelength*1e-9) )/(keV_to_Joule)    # Units: keV


        self.k_wavevector = 2*pi/self.wavelength        # Units 1/nm
        self.a = aperture                               # "Magnitude of illuminated area"
        self.detector_distance = detector_distance      # Units: mm
        self.pixel_size = pixel_size                    # Units: mm

        self.beam_size = beam_size                      # Units: mm
        self.sample_size = sample_size                  # Units: mm

        self.theta_incident = theta_incident            # Units: degrees
        self.theta_incident_rad = radians(theta_incident)   # Units: radians
        
        
        # Compute values
        self.theta_incident_cutoff_angle = asin( self.beam_size/self.sample_size )
        self.theta_scan_cutoff_angle = asin( self.beam_size/self.sample_size )


    def geometric_theta_incident(self, theta_incident):
        """Calculates the geometric factor arising from the intersection
        of the beam with the sample."""
        if theta_incident < self.theta_incident_cutoff_angle:
            return sqrt( self.beam_size/self.sample_size )
        else:
            return sqrt( sin(theta_incident) )
            
    def geometric_theta_scan(self, theta_scan):
        """Calculates the geometric factor arising from the intersection
        of the beam with the sample."""
        if theta_scan < self.theta_scan_cutoff_angle:
            return sqrt( self.beam_size/self.sample_size )
        else:
            return sqrt( sin(theta_scan) )
    
    
    def prefactor(self, theta_incident, theta_scan):
        """Returns the geometric prefactor used in calculating scattering intensity."""
        return 1.0/( 2j * self.k_wavevector * self.a * self.geometric_theta_incident(theta_incident) * self.geometric_theta_scan(theta_scan) )

        
        
# ExperimentalData1D Object        
###################################################################
class ExperimentalData1D(object):
    """Holds an experimental dataset."""
    
    def __init__(self):
        self.q_vals = []
        self.intensity_vals = []
        self.error_vals = []
        
        self.show_fit = True
        self.have_fit_int = False
        self.have_fit_ff = False
        self.have_fit_sf = False
        self.show_residuals = True
        
        self.scale_structure_factor = False
    
    def load_intensity_txt(self, filename, skiprows=0, q_units=10.0, subtract_minimum=False, subtract_constant=None):
        """Loads intensity data from a text file."""
        
        fin = open( filename, 'r' )
        
        self.q_vals = []
        self.intensity_vals = []
        self.error_vals = []
        
        
        for i, line in enumerate(fin.readlines()):
            
            # Skip initial rows
            if i>=skiprows:
                els = line.split()
                try:
                    q = float(els[0])*q_units
                    intensity = float(els[1])
                    
                    # Skip the q=0 rows (other than the 1st)
                    if i!=(0+skiprows) and q!=0:
                        self.q_vals.append( q )
                        self.intensity_vals.append( intensity )
                        if len(els)>2:
                            err = float(els[2])
                            self.error_vals.append( err )
                    
                except ValueError:
                    # Skip this line, since one of the values didn't convert properly
                    pass
                    
        fin.close()

        if subtract_minimum:
            # Subtract the smallest value (this performs a simple background subtraction)
            minimum = min(self.intensity_vals)
            for i, intensity in enumerate(self.intensity_vals):
                self.intensity_vals[i] = intensity-minimum

        if subtract_constant!=None:
            for i, intensity in enumerate(self.intensity_vals):
                self.intensity_vals[i] = intensity-subtract_constant


    def load_form_factor_txt(self, filename, skiprows=0, q_units=10.0, subtract_minimum=False, subtract_constant=None):
        """Loads intensity data from a text file."""
        
        fin = open( filename, 'r' )
        
        self.q_ff_vals = []
        self.ff_vals = []
        self.error_ff_vals = []
        
        for i, line in enumerate(fin.readlines()):
            
            # Skip initial rows
            if i>=skiprows:
                els = line.split()
                try:
                    q = float(els[0])*q_units
                    intensity = float(els[1])
                    
                    # Skip the q=0 rows (other than the 1st)
                    if i!=(0+skiprows) and q!=0:
                        self.q_ff_vals.append( q )
                        self.ff_vals.append( intensity )
                        if len(els)>2:
                            err = float(els[2])
                            self.error_ff_vals.append( err )
                    
                except ValueError:
                    # Skip this line, since one of the values didn't convert properly
                    pass
                    
        fin.close()
        
        if subtract_minimum:
            # Subtract the smallest value (this performs a simple background subtraction)
            minimum = min(self.ff_vals)
            for i, intensity in enumerate(self.ff_vals):
                self.ff_vals[i] = intensity-minimum

        if subtract_constant!=None:
            for i, intensity in enumerate(self.ff_vals):
                self.ff_vals[i] = intensity-subtract_constant

                
    def load_form_factor_data(self, q_vals, P_vals, q_units=1.0, subtract_minimum=False, subtract_constant=None):
        """Loads intensity data from supplied data arrays."""
        
        
        self.q_ff_vals = q_vals
        self.ff_vals = P_vals*q_units
        self.error_ff_vals = []
        
        if subtract_minimum:
            # Subtract the smallest value (this performs a simple background subtraction)
            minimum = min(self.ff_vals)
            for i, intensity in enumerate(self.ff_vals):
                self.ff_vals[i] = intensity-minimum

        if subtract_constant!=None:
            for i, intensity in enumerate(self.ff_vals):
                self.ff_vals[i] = intensity-subtract_constant
                
    
    def set_structure_factor_asymptote(self, q_min, q_max):
        self.scale_structure_factor = True
        self.scale_structure_factor_q_range = (q_min, q_max)
        
        
    def structure_factor(self, tolerance=1e-8):
        """Returns a (q, intensity) array that contains the structure factor (intensity divided by
        form factor)."""
        
        output = []
        
        j = 0
        q_ff = self.q_ff_vals[j]
        for i, q_int in enumerate(self.q_vals):
            
            # Search for the corresponding q in the ff data
            while q_ff<q_int and j<len(self.q_ff_vals):
                j += 1
                if j<len(self.q_ff_vals):
                    q_ff = self.q_ff_vals[j]
                else:
                    q_ff = self.q_ff_vals[-1]
                
            if abs(q_int-q_ff)/q_int<tolerance and j<len(self.q_ff_vals):
                # The two q-values match
                if self.ff_vals[j]==0:
                    intensity = 1
                else:
                    intensity = self.intensity_vals[i]/self.ff_vals[j]
                output.append( [q_int, intensity] )
                
        if self.scale_structure_factor:
            q_min, q_max = self.scale_structure_factor_q_range
            # Use the supplied q-span to determine a scaling constant
            scaling_val = 0.0
            scaling_num = 0
            for q, intensity in output:
                if q>=q_min and q<=q_max:
                    scaling_val += intensity
                    scaling_num += 1
            scaling_val /= scaling_num
            
            # Scale data
            for i in range(len(output)):
                output[i][1] = output[i][1] / scaling_val
        
        return numpy.array(output)
        
        
        
    def plot(self, ptype='intensity', qi=0.0, qf=None, filename='data_intensity.png', scaling=None, xlog=False, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values."""
        
        # Get data
        err_vals = []
        if ptype=='form_factor':
            q_list = self.q_ff_vals
            int_list = self.ff_vals
            q_fit = []
            int_fit = []
            if self.have_fit_ff:
                q_fit = self.q_ff_fit
                int_fit = self.ff_fit
        elif ptype=='structure_factor':
            vals = self.structure_factor()
            q_list = vals[:,0]
            int_list = vals[:,1]
            q_fit = []
            int_fit = []
            if self.have_fit_sf:
                q_fit = self.q_sf_fit
                int_fit = self.sf_fit
        else:
            q_list = self.q_vals
            int_list = self.intensity_vals
            
            if self.error_vals != [] and len(self.error_vals)==len(int_list):
                err_vals = self.error_vals
            
            q_fit = []
            int_fit = []
            if self.have_fit_int:
                q_fit = self.q_fit
                int_fit = self.intensity_fit
        

        pylab.rcParams['axes.labelsize'] = 'xx-large'
        pylab.rcParams['xtick.labelsize'] = 'x-large'
        pylab.rcParams['ytick.labelsize'] = 'x-large'
        
        fig = pylab.figure()
        #fig.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.94)
        ax1 = fig.add_axes( [0.15,0.15,0.94-0.15,0.94-0.15] )
        
        
        if err_vals != []:
            pylab.errorbar( q_list, int_list, yerr=err_vals, fmt='o', color=(0,0,0), ecolor=(0.5,0.5,0.5), linewidth=2.0 )
        else:
            pylab.plot( q_list, int_list, 'o', color=(0,0,0), linewidth=2.0 )
        
        if xlog and ylog:
            pylab.loglog()
        else:
            if xlog:
                pylab.semilogx()
            if ylog:
                pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( 'Intensity (a.u.)', size=30 )        

        # Get axis scaling
        xi, xf, yi, yf = pylab.axis()

        #print ptype
        #print len(q_fit), q_fit[0]
        #print len(int_fit), int_fit[0]
        
        
        # Set axis scaling
        if scaling==None:
            # (We want the experimental data centered, not the fit.)
            xi = qi
            if qf!=None:
                xf = qf
            pylab.axis( [xi, xf, yi, yf] )
        else:
            pylab.axis( scaling )
        
        
        if False:
            # Save data to file
            fout = open( filename + '.dat', 'w')
            for i in range(len(q_list)):
                fout.write( str(q_list[i]) + '\t' + str(int_list[i]) + '\n' )
            fout.close()
                

        pylab.savefig( filename )
        
        return int_list
        
        

    def plot_intensity(self, qi=0.0, qf=None, filename='data_intensity.png', scaling=None, xlog=False, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values."""
        
        return self.plot(ptype='intensity', qi=qi, qf=qf, filename=filename, scaling=scaling, xlog=xlog, ylog=ylog)

    def plot_form_factor(self, qi=0.0, qf=None, filename='data_form_factor.png', scaling=None, xlog=False, ylog=False):
        """Outputs a plot of the form factor vs. q data. Also returns an array
        of the intensity values."""
        
        return self.plot(ptype='form_factor', qi=qi, qf=qf, filename=filename, scaling=scaling, xlog=xlog, ylog=ylog)

    def plot_structure_factor(self, qi=0.0, qf=None, filename='data_structure_factor.png', scaling=None, xlog=False, ylog=False):
        """Outputs a plot of the structure factor vs. q data (intensity divided by form factor). 
        Also returns an array of the intensity values."""
        
        return self.plot(ptype='structure_factor', qi=qi, qf=qf, filename=filename, scaling=scaling, xlog=xlog, ylog=ylog)


    def plot_compare(self, q_vals, compare_vals, ptype='intensity', qi=0.0, qf=None, filename='data_intensity.png', scaling=None, xlog=False, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values."""
        
        # Get data
        err_vals = []
        if ptype=='form_factor':
            q_list = self.q_ff_vals
            int_list = self.ff_vals
            q_fit = []
            int_fit = []
            if self.have_fit_ff:
                q_fit = self.q_ff_fit
                int_fit = self.ff_fit
        elif ptype=='structure_factor':
            vals = self.structure_factor()
            q_list = vals[:,0]
            int_list = vals[:,1]
            q_fit = []
            int_fit = []
            if self.have_fit_sf:
                q_fit = self.q_sf_fit
                int_fit = self.sf_fit
        else:
            q_list = self.q_vals
            int_list = self.intensity_vals
            
            if self.error_vals != [] and len(self.error_vals)==len(int_list):
                err_vals = self.error_vals
            
            q_fit = []
            int_fit = []
            if self.have_fit_int:
                q_fit = self.q_fit
                int_fit = self.intensity_fit
        

        pylab.rcParams['axes.labelsize'] = 'xx-large'
        pylab.rcParams['xtick.labelsize'] = 'x-large'
        pylab.rcParams['ytick.labelsize'] = 'x-large'
        
        fig = pylab.figure()
        #fig.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.94)
        ax1 = fig.add_axes( [0.15,0.15,0.94-0.15,0.94-0.15] )
        
        
        if err_vals != []:
            pylab.errorbar( q_list, int_list, yerr=err_vals, fmt='o', color=(0,0,0), ecolor=(0.5,0.5,0.5), linewidth=2.0 )
        else:
            pylab.plot( q_list, int_list, 'o', color=(0,0,0), linewidth=2.0 )
        
        pylab.plot( q_vals, compare_vals, '-', color='b', linewidth=1.5 )
        
        if xlog and ylog:
            pylab.loglog()
        else:
            if xlog:
                pylab.semilogx()
            if ylog:
                pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( 'Intensity (a.u.)', size=30 )        

        # Get axis scaling
        xi, xf, yi, yf = pylab.axis()

        #print ptype
        #print len(q_fit), q_fit[0]
        #print len(int_fit), int_fit[0]
        
        
        # Set axis scaling
        if scaling==None:
            # (We want the experimental data centered, not the fit.)
            xi = qi
            if qf!=None:
                xf = qf
            pylab.axis( [xi, xf, yi, yf] )
        else:
            pylab.axis( scaling )
        

        pylab.savefig( filename )
        
        return int_list
        



# ExperimentalData1D_compare_fitted Object        
###################################################################
# This object is deprecated! Use a Fit class instead
class ExperimentalData1D_compare_fitted(ExperimentalData1D):
    """Holds an experimental dataset."""
    

        
    def load_intensity_fit(self, q_list, int_list):
        self.have_fit_int = True
        
        self.q_fit = q_list
        self.intensity_fit = int_list
        
    def load_form_factor_fit(self, q_list, int_list):
        self.have_fit_ff = True
        
        self.q_ff_fit = q_list
        self.ff_fit = int_list

    def load_structure_factor_fit(self, q_list, int_list):
        self.have_fit_sf = True
        
        self.q_sf_fit = q_list
        self.sf_fit = int_list

    

    def plot(self, ptype='intensity', qi=0.0, qf=None, filename='data_intensity.png', scaling=None, showresiduals=True, relativeresiduals=True, xlog=False, ylog=False):
        """Outputs a plot of the intensity vs. q data. Also returns an array
        of the intensity values."""
        
        # Get data
        err_vals = []
        if ptype=='form_factor':
            q_list = self.q_ff_vals
            int_list = self.ff_vals
            q_fit = []
            int_fit = []
            if self.have_fit_ff:
                q_fit = self.q_ff_fit
                int_fit = self.ff_fit
        elif ptype=='structure_factor':
            vals = self.structure_factor()
            q_list = vals[:,0]
            int_list = vals[:,1]
            q_fit = []
            int_fit = []
            if self.have_fit_sf:
                q_fit = self.q_sf_fit
                int_fit = self.sf_fit
        else:
            q_list = self.q_vals
            int_list = self.intensity_vals
            
            if self.error_vals != [] and len(self.error_vals)==len(int_list):
                err_vals = self.error_vals
            
            q_fit = []
            int_fit = []
            if self.have_fit_int:
                q_fit = self.q_fit
                int_fit = self.intensity_fit
        

        pylab.rcParams['axes.labelsize'] = 'xx-large'
        pylab.rcParams['xtick.labelsize'] = 'x-large'
        pylab.rcParams['ytick.labelsize'] = 'x-large'
        
        fig = pylab.figure()
        #fig.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.94)
        if showresiduals:
            res_fig_height = 0.1
            ax1 = fig.add_axes( [0.15,0.15,0.94-0.15,0.94-0.15-res_fig_height] )
        else:
            ax1 = fig.add_axes( [0.15,0.15,0.94-0.15,0.94-0.15] )
        
        
        if err_vals != []:
            pylab.errorbar( q_list, int_list, yerr=err_vals, fmt='o', color=(0,0,0), ecolor=(0.5,0.5,0.5), linewidth=2.0 )
        else:
            pylab.plot( q_list, int_list, 'o', color=(0,0,0), linewidth=2.0 )
        
        if xlog and ylog:
            pylab.loglog()
        else:
            if xlog:
                pylab.semilogx()
            if ylog:
                pylab.semilogy()
            
        pylab.xlabel( r'$q \, (\mathrm{nm}^{-1})$', size=30 )
        pylab.ylabel( 'Intensity (a.u.)', size=30 )        

        # Get axis scaling
        xi, xf, yi, yf = pylab.axis()

        #print ptype
        #print len(q_fit), q_fit[0]
        #print len(int_fit), int_fit[0]
        
        if self.show_fit and q_fit!=[]:
            pylab.plot( q_fit, int_fit, '-', color=(0,0,1), linewidth=1.5 )
        
        # Set axis scaling
        if scaling==None:
            # (We want the experimental data centered, not the fit.)
            xi = qi
            if qf!=None:
                xf = qf
            pylab.axis( [xi, xf, yi, yf] )
        else:
            pylab.axis( scaling )
        


        # Residuals figure
        if showresiduals:
            
            # Get ax1 final scaling
            x1i, x1f, y1i, y1f = pylab.axis()
            # Find indices of 'data of interest'
            index_start = 0
            index_finish = 0
            for i, q in enumerate(q_list):
                if q<x1i:
                    index_start = i
                if q<x1f:
                    index_end = i
            
            
            # Interpolate data to get residuals
            residuals = []
            j = 0
            for q, y_exp in zip(q_list, int_list):
                
                # Find closest theoretical values
                while( j<len(q_fit) and q_fit[j] < q ):
                    j += 1
                if j==0:
                    j = 1
                elif j>=len(q_fit):
                    j = len(q_fit)-1
                
                y_fit = self.interpolate( (q_fit[j], int_fit[j]), (q_fit[j-1],int_fit[j-1]), q )
                if relativeresiduals:
                    residuals.append( (y_fit-y_exp)/y_exp )
                else:
                    residuals.append( y_fit-y_exp )
            
            
            ax2 = fig.add_axes( [0.15,0.94-res_fig_height,0.94-0.15,res_fig_height] )
            
            pylab.plot( q_list, residuals )
            pylab.axhline( 0, color='k' )

            if err_vals != []:
                # Create a filled-in region
                q_list_backwards = [ q_list[i] for i in range(len(q_list)-1,-1,-1) ]
                if relativeresiduals:
                    upper2 = [ +2.0*err_vals[i]/int_list[i] for i in range(len(err_vals)) ]
                    lower2 = [ -2.0*err_vals[i]/int_list[i] for i in range(len(err_vals)-1,-1,-1) ]
                    upper1 = [ +1.0*err_vals[i]/int_list[i] for i in range(len(err_vals)) ]
                    lower1 = [ -1.0*err_vals[i]/int_list[i] for i in range(len(err_vals)-1,-1,-1) ]
                else:
                    upper2 = [ +2.0*err_vals[i] for i in range(len(err_vals)) ]
                    lower2 = [ -2.0*err_vals[i] for i in range(len(err_vals)-1,-1,-1) ]
                    upper1 = [ +1.0*err_vals[i] for i in range(len(err_vals)) ]
                    lower1 = [ -1.0*err_vals[i] for i in range(len(err_vals)-1,-1,-1) ]
                
                ax2.fill( q_list+q_list_backwards, upper2+lower2, edgecolor='0.92', facecolor='0.92' )
                ax2.fill( q_list+q_list_backwards, upper1+lower1, edgecolor='0.82', facecolor='0.82' )
            
            xi, xf, yi, yf = ax2.axis()
            xi = x1i
            xf = x1f
            ax2.axis( [xi, xf, yi, yf] )
            
            ax2.set_xticklabels( [] )
            ax2.set_yticklabels( [] )


            

        pylab.savefig( filename )
        
        return int_list

    def interpolate(self, point1, point2, x_target ):
        x1, y1 = point1
        x2, y2 = point2
        m = (y2-y1)/(x2-x1)
        b = y2-m*x2
        
        y_target = m*x_target + b
        
        return y_target


        
