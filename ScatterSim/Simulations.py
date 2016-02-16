# -*- coding: utf-8 -*-
###################################################################
# Simulations.py
# version 0.0.5
# April 12, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# Description:
#  This file defines a series of Simulation objects. A simulation
# object is useful for generating q-space intensity values (e.g.
# detector images) from a given model and real-space potential.
#  A simulation can additionally perform other computations, such
# as averages. For instance, a simulation object can internally average
# a variety of angles, to simulate random in-place grain orientations.
# Or a simulation could internally average a variety of box sizes
# to eliminate artifacts coming from the finite real-space partitioning.
#  A simulation might also apply a resolution function to 'blur' data
# realistically.
###################################################################

from BaseClasses import *
 
 
 
class CircularAverageSimulation(Simulation):
    """This simulation does an in-plane circular average."""
    
    start_angle = 0.0
    end_angle = 360.0
    num_angles = 24
    
    def set_num_angles(self, num_angles):
        self.num_angles = num_angles

    def get_num_angles(self):
        return self.num_angles
    
    def set_angle_range(self, start_angle=0.0, end_angle=360.0, num_angles=24):
        """Sets the angle range for in-plane averaging."""
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.num_angles = num_angles 

    def get_angle_range(self):
        """Returns the angle range elements: starting angle, ending angle, number of angles."""
        return self.start_angle, self.end_angle, self.num_angles

    def get_angle_values(self):
        """Returns an array of the angles that will be used (in degrees)."""
        return numpy.linspace( self.start_angle, self.end_angle, self.num_angles, endpoint=False )



    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        
        status_print( "Starting circular simulation for q-value...", depth=5 )

        intensity = 0

        for theta in self.get_angle_values():
            status_print( "Calculating angle " + str(theta) + "...", depth=5 )

            self.Potential.set_angles( theta=theta )
            self.Box.fill( self.Potential )
            
            self.Model.clear()

            intensity += self.Model.q_value( qx, qy, qz )

        # Normalize by the number of angles
        intensity /= self.num_angles

        return Scattering0D( (qx,qy,qz), intensity )


    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""
        
        status_print( "Starting circular simulation for detector angle value...", depth=5 )

        intensity = 0

        for theta in self.get_angle_values():
            status_print( "Calculating angle " + str(theta) + "...", depth=5 )

            self.Potential.set_angles( theta=theta )
            self.Box.fill( self.Potential )
            
            self.Model.clear()

            intensity += self.Model.angle_value( theta_incident, theta_scan, phi_scan )

        # Normalize by the number of angles
        intensity /= self.num_angles

        # TODO: Return a Scattering object
        return intensity

    def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples."""

        # TODO: Peform simulation

        return 0.0
    
    
    def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""

        status_print( "Starting circular simulation...", depth=0 )

        data = numpy.zeros( (qy_num, qz_num), numpy.float )

        for theta in self.get_angle_values():
            status_print( "Calculating angle " + str(theta) + "...", depth=0 )
            
            self.Potential.set_angles( theta=theta )
            self.Box.fill( self.Potential )
            
            self.Model.clear()
            
            
            data += self.Model.ewald_2d( theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num) )

        # Normalize by the number of angles
        data /= self.num_angles

        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter
    
    
    def to_string(self):
        s = "Circular Average Simulation.\n"
        s += super(CircularAverageSimulation, self).to_string()
        return s



class BoxAverageSimulation(Simulation):
    """This simulation averages different box sizes, to eliminate artifacts."""
    
    start_rel_size = 0.8
    end_rel_size = 1.2
    num_samples = 24
    
    vary_x = True
    vary_y = True
    vary_z = True
    
    limits = {}
    
    
    def __init__(self, Model, Potential, Box, Experimental, sargs={}, margs={}, pargs={}, bargs={}):
        """Initializes the simulation with a given model, potential,
        and simulation box. Optional arguments should be passed to
        override the defaults and define these objects."""
        
        super(BoxAverageSimulation, self).__init__(Model, Potential, Box, Experimental, sargs=sargs, margs=margs, pargs=pargs, bargs=bargs)
        
        # Save a copy of the original box limits
        self.limits.update( self.Box.limits )
        
        # Extract sizing values from simulation arguments, if they exist
        if 'start_rel_size' in self.sargs:
            self.start_rel_size = self.sargs['start_rel_size']
        if 'end_rel_size' in self.sargs:
            self.end_rel_size = self.sargs['end_rel_size']
        if 'num_samples' in self.sargs:
            self.num_samples = self.sargs['num_samples']
    
        if 'vary_x' in self.sargs:
            self.vary_x = self.sargs['vary_x']
        if 'vary_y' in self.sargs:
            self.vary_y = self.sargs['vary_y']
        if 'vary_z' in self.sargs:
            self.vary_z = self.sargs['vary_z']
    
    
    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def get_num_samples(self):
        return self.num_samples

    def set_rel_range(self, relative=0.2, num_samples=24, vary_x=True, vary_y=True, vary_z=True):
        """Set the range (relative to current size) over which the box-size will be varied."""
        start = 1.0-relative
        end = 1.0+relative
        self.set_range(start=start, end=end, num_samples=num_samples )

    def set_range(self, start=0.8, end=1.2, num_samples=24, vary_x=True, vary_y=True, vary_z=True):
        """Sets the range for size averaging."""
        self.start_rel_size = start
        self.end_rel_size = end
        self.num_samples = num_samples
        
        self.vary_x = vary_x
        self.vary_y = vary_y
        self.vary_z = vary_z

    def get_range(self):
        """Returns the range elements: starting, ending, number of samples in average."""
        return self.start_rel_size, self.end_rel_size, self.num_samples

    def get_values(self):
        """Returns an array of the relative box-sizes that will be used (in degrees)."""
        return numpy.linspace( self.start_rel_size, self.end_rel_size, self.num_samples, endpoint=True )


    # TODO: Implement:
    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        return 0.0

    # TODO: Implement:
    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""
        return 0.0
        
    # TODO: Implement:
    def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples."""
        return 0.0
    
        

    def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""

        status_print( "Starting box average simulation...", depth=0 )
        
        if self.sargs['parallelize']:
            return self.ewald_2d_parallel(theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num))

        data = numpy.zeros( (qy_num, qz_num), numpy.float )

        for rel_size in self.get_values():
            status_print( "Calculating box-size " + str(rel_size) + "...", depth=0 )
            
            # Create new box, with altered dimensions
            bargs = {}
            
            if self.vary_x:
                bargs['x'] = { \
                                'start': self.limits['x']['start']*rel_size, \
                                'end': self.limits['x']['end']*rel_size, \
                                'num': self.limits['x']['num'] \
                                }
            if self.vary_y:
                bargs['y'] = { \
                                'start': self.limits['y']['start']*rel_size, \
                                'end': self.limits['y']['end']*rel_size, \
                                'num': self.limits['y']['num'] \
                                }
            if self.vary_z:
                bargs['z'] = { \
                                'start': self.limits['z']['start']*rel_size, \
                                'end': self.limits['z']['end']*rel_size, \
                                'num': self.limits['z']['num'] \
                                }

            self.Box.rebuild( bargs=bargs )
            
            self.Box.fill( self.Potential )
            
            self.Model.clear()
            
            
            data += self.Model.ewald_2d( theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num) )

        # Normalize by the number of angles
        data /= self.num_samples

        # Restore the box object to its original state
        self.Box.rebuild( self.limits )

        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter


    def ewald_2d_parallel(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""

        # Parallel operation is acheived by spawning multiple Python processes
        # using the multiprocessing module, available in Python >=2.6. See:
        # http://docs.python.org/library/multiprocessing.html#module-multiprocessing.pool

        arglist = ( theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num) )

        # Launch parallel jobs
        jobs = []
        for rel_size in self.get_values():
            status_print( "Calculating box-size " + str(rel_size) + "...", depth=0 )
            
            # Create new box, with altered dimensions
            bargs = {}
            
            if self.vary_x:
                bargs['x'] = { \
                                'start': self.limits['x']['start']*rel_size, \
                                'end': self.limits['x']['end']*rel_size, \
                                'num': self.limits['x']['num'] \
                                }
            if self.vary_y:
                bargs['y'] = { \
                                'start': self.limits['y']['start']*rel_size, \
                                'end': self.limits['y']['end']*rel_size, \
                                'num': self.limits['y']['num'] \
                                }
            if self.vary_z:
                bargs['z'] = { \
                                'start': self.limits['z']['start']*rel_size, \
                                'end': self.limits['z']['end']*rel_size, \
                                'num': self.limits['z']['num'] \
                                }

            
            
            # Set each box size as a job to run.
            # Each job is like doing:
            #   self.ewald_2d(theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num))
            # Which is accomplished by invoking:
            #   GenerateSimulation(self.Model, self.Potential, arglist, scatter_type='ewald_2d', bargs=bargs)
            
            jobs.append( (rel_size,
                            self.pool.apply_async( \
                                    func=GenerateSimulation, \
                                    args=(self.Model, self.Potential, arglist,), \
                                    kwds={'scatter_type': 'ewald_2d', 'bargs': bargs}, \
                                    ) ) \
                        )
        
        
        # Recover the answer from each parallel job
        data = numpy.zeros( (qy_num, qz_num), numpy.float )
        for rel_size, job in jobs:
            
            status_print( "Requesting job result from box-size " + str(rel_size) + "...", 1 )
            
            cur_data = job.get()
            data += cur_data
            
            status_print( "Received job result from box-size " + str(rel_size) + ".", 1 )


        # Normalize by the number of angles
        data /= self.num_samples


        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter
        
    
    def to_string(self):
        s = "Box Size Variation Simulation.\n"
        s += super(BoxAverageSimulation, self).to_string()
        return s








# TODO: This class is preliminary. Coding not done.
class StochasticSamplingSimulation(Simulation):
    """The Stochastic Sampling Simulation allows a variety of 
    'to be varied' parameters to be specified. The simulation
    will then randomly select simulations within the set of
    all possiblities, adding them to a running average.
    The intention is to more efficiently simulate systems that
    have many things to be averaged (e.g. different sizes,
    orientations, box sizes). Rather than exhaustively iterating
    through each variable, we keep adding randomnly-selected 
    sims."""

    num_samples = 48
    repeating = False
    
    sargs = {}
    sargs['e_ranges'] = {}
    sargs['b_ranges'] = {}
    sargs['p_ranges'] = {}
    sargs['m_ranges'] = {}
    
    
    # Saved copies of original argument lists
    margs_orig = {}
    pargs_orig = {}
    bargs_orig = {}
    eargs_orig = {}
    
    done = []               # A record of what simulations we've already done

    def __init__(self, Model, Potential, Box, Experimental, sargs={}, margs={}, pargs={}, bargs={}, num_samples=48, repeating=False, seed=2018):
        """Initializes the simulation with a given model, potential,
        and simulation box. Optional arguments should be passed to
        override the defaults and define these objects."""
        
        super(StochasticSamplingSimulation, self).__init__(Model, Potential, Box, Experimental, sargs=sargs, margs=margs, pargs=pargs, bargs=bargs)
        
        # Save a copy of the original arguments
        self.margs_orig.update( self.Model.margs )
        self.pargs_orig.update( self.Potential.pargs )
        self.bargs_orig.update( self.Box.limits )
        
        self.num_samples = num_samples
        self.repeating = repeating
        random.seed( seed )
        
        # TODO: Sanity check that the number of samples is less than the total exploration of the parameter space!
        # TODO: make the 'repeating' argument actually do something

    def set_num_samples(self, num_samples, repeating=False):
        self.num_samples = num_samples
        self.repeating = repeating

    def get_num_samples(self):
        return self.num_samples


    
    def get_values(self):
        r = {}
        r['e_ranges'] = self.sargs['e_ranges']
        r['b_ranges'] = self.sargs['b_ranges']
        r['p_ranges'] = self.sargs['p_ranges']
        r['m_ranges'] = self.sargs['m_ranges']
        
        return r


    # Helper Functions
    
    def dictionary_match( self, dictionary1, dictionary2 ):
        """Compares two dictionaries, and returns True if they are
        identical (the contain the same values for all corresponding
        keys). Will return an error if the dictionaries do not share
        the same keys.
        This function assumes that all dictionaries have one-level
        nested dictionaries within them."""
        
        for key1, sub_dictionary in dictionary1.items():
            for key2, sub_value in sub_dictionary.items():
                
                if sub_value != dictionary2[key1][key2]:
                    return False

        return True
                        


    def is_dictionary_in_list(self, test_dictionary, dictionary_list ):
        
        if dictionary_list == {}:
            return False
        
        for row in dictionary_list:
            if self.dictionary_match( test_dictionary, row ):
                return True
            
        return False

    def generate_random_variable(self):
        variables = make_inner()
        for key, (start, stop, step) in self.sargs['e_ranges'].items():
            variables['e_ranges'][key] = randarange( start, stop, step )
        for key, (start, stop, step) in self.sargs['b_ranges'].items():
            variables['b_ranges'][key] = randarange( start, stop, step )
        for key, (start, stop, step) in self.sargs['p_ranges'].items():
            variables['p_ranges'][key] = randarange( start, stop, step )
        for key, (start, stop, step) in self.sargs['m_ranges'].items():
            variables['m_ranges'][key] = randarange( start, stop, step )
        
        return variables
        
    
    def get_variables(self):

        if self.repeating:
            return self.generate_random_variable()
        else:
            variables_found = False
            while not variables_found:
                variables = self.generate_random_variable()

                if not self.is_dictionary_in_list( variables, self.done ):
                    variables_found = True

            self.done.append( variables )

            return variables


    def generate_pargs(self, variables):
        """Creates a dictionary of potential arguments (pargs), suitable
        for creating/updating Potential objects, based on the provided
        simulation variables."""
        
        pargs = self.pargs_orig.copy()
        pargs.update( variables['p_ranges'] )
        
        return pargs


    def generate_bargs(self, variables):
        """Creates a dictionary of box arguments (bargs), suitable
        for creating new SimulationBox objects, based on the provided
        simulation variables."""
        
        bargs = self.bargs_orig.copy()

        if 'x_span_rel' in variables['b_ranges']:
            rel_size = variables['b_ranges']['x_span_rel']
            bargs['x'] = { \
                            'start': self.bargs_orig['x']['start']*rel_size, \
                            'end': self.bargs_orig['x']['end']*rel_size, \
                            'num': self.bargs_orig['x']['num'] \
                            }
        if 'y_span_rel' in variables['b_ranges']:
            rel_size = variables['b_ranges']['y_span_rel']
            bargs['y'] = { \
                            'start': self.bargs_orig['y']['start']*rel_size, \
                            'end': self.bargs_orig['y']['end']*rel_size, \
                            'num': self.bargs_orig['y']['num'] \
                            }
        if 'z_span_rel' in variables['b_ranges']:
            rel_size = variables['b_ranges']['z_span_rel']
            bargs['z'] = { \
                            'start': self.bargs_orig['z']['start']*rel_size, \
                            'end': self.bargs_orig['z']['end']*rel_size, \
                            'num': self.bargs_orig['z']['num'] \
                            }
            
        return bargs



    # TODO: Implement:
    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        return 0.0

    # TODO: Implement:
    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""

        data = 0.0
  
        # TODO: This isn't returning the right q-value right now
        return Scattering0D( 0.01, data )
        
        
        
    # TODO: Implement:
    def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples."""
        
        # TODO: Return scattering object
        return 0.0


    def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""

        if self.sargs['parallelize']:
            return self.ewald_2d_parallel(theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num))


        data = numpy.zeros( (qy_num, qz_num), numpy.float )

        # Do a bunch of samples...
        for i in range(self.num_samples):
            
            status_print( "Calculating sampling " + str(i) + "...", depth=0 )
            
            # Randomly select a set of parameters
            v = self.get_variables()
            
            # Create new simulation with selected parameters
            bargs = self.generate_bargs( v )
            self.Box.rebuild( bargs=bargs )
            
            pargs = self.generate_pargs( v )
            pargs['box_z_span'] = self.Box.limits['z']['span']
            self.Potential.rebuild( pargs=pargs )
            
            self.Box.fill( self.Potential )
            
            self.Model.clear()
            
            data += self.Model.ewald_2d( theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num) )
            

        # Normalize by the number of angles
        data /= self.num_samples


        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter

    def ewald_2d_parallel(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""

        # Parallel operation is acheived by spawning multiple Python processes
        # using the multiprocessing module, available in Python >=2.6. See:
        # http://docs.python.org/library/multiprocessing.html#module-multiprocessing.pool

        arglist = ( theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num) )
        
        # Launch parallel jobs
        jobs = []
        for i in range(self.num_samples):
            
            status_print( "Starting sampling " + str(i) + "...", depth=0 )
            
            # Randomly select a set of parameters
            v = self.get_variables()
            bargs = self.generate_bargs( v )
            pargs = self.generate_pargs( v )
            pargs['box_z_span'] = (bargs['z']['end']-bargs['z']['start'])
            
            # Set a job to run. We call something like:
            #   GenerateSimulation(self.Model, self.Potential, arglist, scatter_type='ewald_2d', bargs=bargs, pargs=pargs)

            jobs.append( (  i, \
                            v, \
                            self.pool.apply_async( \
                                    func=GenerateSimulation, \
                                    args=(self.Model, self.Potential, arglist,), \
                                    kwds={'scatter_type': 'ewald_2d', 'bargs': bargs, 'pargs': pargs}, \
                                    ) ) \
                        )

        # Recover the answer from each parallel job
        data = numpy.zeros( (qy_num, qz_num), numpy.float )
        for i, v, job in jobs:

            status_print( "Requesting job result from sampling " + str(i) + "...", 1 )

            cur_data = job.get()
            data += cur_data
            
            if self.sargs['save_samples']:
                filename = 'out-'
                for key1, args1 in v.items():
                    for key2, args2 in args1.items():
                        filename += key2 + str(args2)
                numpy.savetxt( filename+'.dat', data )
            
            percent_done = str( (100*(i+1))/self.num_samples )
            status_print( "Received job result from sampling " + str(i) + " ("+percent_done+"% done).", 1 )


        # Normalize by the number of angles
        data /= self.num_samples


        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter


    def to_string(self):
        s = "Stochastic Sampling Simulation.\n"
        s += super(StochasticSamplingSimulation, self).to_string()
        return s

















class DummySimulation(Simulation):
    """This simulation is just a shell. The code can be copied and modified to create new simulations."""

    def __init__(self, Model, Potential, Box, Experimental, sargs={}, margs={}, pargs={}, bargs={}):
        """Initializes the simulation with a given model, potential,
        and simulation box. Optional arguments should be passed to
        override the defaults and define these objects."""
        
        super(DummySimulation, self).__init__(Model, Potential, Box, Experimental, sargs=sargs, margs=margs, pargs=pargs, bargs=bargs)


    # TODO: Implement:
    def q_value(self, qx, qy, qz):
        """Returns the intensity for the given point in
        reciprocal-space."""
        return 0.0

    # TODO: Implement:
    def angular_value(self, theta_incident, theta_scan, phi_scan):
        """Returns the intensity for the given scattering angle."""
        return 0.0
        
    # TODO: Implement:
    def angular_2d(self, theta_incident, (theta_scan_min, theta_scan_max, theta_scan_num), (phi_scan_min, phi_scan_max, phi_scan_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in theta_scan and phi_scan are specified as tuples."""
        return 0.0
    
    # TODO: Implement:
    def ewald_2d(self, theta_incident, (qy_min, qy_max, qy_num), (qz_min, qz_max, qz_num)):
        """Returns the intensity for a 2d subset of the Ewald-sphere.
        The range in qy and qz are specified as tuples."""
        
        data = numpy.zeros( (qy_num, qz_num), numpy.float )
        
        scatter = Scattering2DEwald( \
                        theta_incident, \
                        ( qy_min, qy_max, qy_num ), \
                        ( qz_min, qz_max, qz_num ), \
                        data, \
                        Experimental=self.Experimental )

        return scatter


    def to_string(self):
        s = "Dummy Simulation.\n"
        s += super(BoxAverageSimulation, self).to_string()
        return s
