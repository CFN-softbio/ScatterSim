# -*- coding: utf-8 -*-
###################################################################
# Potentials.py
# version 0.0.5
# April 12, 2010
###################################################################
# Author: Kevin G. Yager
# Affiliation: Brookhaven National Lab, Center for Functional Nanomaterials
###################################################################
# Description:
#  This file stores a variety of potential objects. These objects
# define a real-space density distribution inside a material. They
# can be associated with modeling objects in order to calculate
# scattering patterns from the given real-space potential/density.
#  The FilmPotential object represents a thin film supported on a
# substrate. By extending this object, one can define a variety of
# structured thin films.
###################################################################


from BaseClasses import *



class FilmPotential(Potential):
    """This potential describes a thin film. The density
    above the film (ambient/superstrate) is set via
    'rho_ambient' and the density below the film (substrate)
    is set via 'rho_substrate'. This is intended as a
    base class that can be extended to describe arbitrary
    nanostructure inside the thin film itself. However
    this class can be used directly to simulate a film
    with uniform internal density of 'rho_film'."""
    
    
    # TODO:
    # This should be moved into __init__ so that it isn't global.
    # Also, it should probably be removed since it conflicts with the rotation_elements
    # method
    rotation_elements = numpy.identity(3)
    

    def __init__(self, pargs={}):
        self.pargs.update(pargs)
    
   
        if 'rotation' in self.pargs and self.pargs['rotation']:
            # Compute a rotation matrix based on the supplied rotation angles.
            self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )
        else:
            self.pargs['rotation'] = False

    def rebuild(self, pargs={}):
        """Allows a Potential object to have its potential
        arguments (pargs) updated. Note that this doesn't
        replace the old pargs entirely. It only modifies 
        (or adds) the key/values provided by the new pargs."""
        self.pargs.update(pargs)

        if 'rotation' in self.pargs and self.pargs['rotation']:
            # Compute a rotation matrix based on the supplied rotation angles.
            self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )
        else:
            self.pargs['rotation'] = False


    def set_angles(self, eta=None, phi=None, theta=None):
        """Update one or multiple angles."""
        if eta != None:
            self.pargs['eta'] = eta
        if phi != None:
            self.pargs['phi'] = phi
        if theta != None:
            self.pargs['theta'] = theta
            
        self.rotation_matrix = self.rotation_elements( self.pargs['eta'], self.pargs['phi'], self.pargs['theta'] )
        
        

    def rotation_elements(self, eta, phi, theta):
        """Converts angles into an appropriate rotation matrix."""
        
        
        # Standard rotation:
        # Tilt by phi with respect to +z (rotation about y-axis) then rotate by theta
        # in-place (rotation about z-axis)
        #phi = radians( phi )      # phi is grain tilt (with respect to +z axis)
        #theta = radians( theta )    # grain orientation (around the z axis)
        #rotation_elements = [[  cos(phi)*cos(theta), -cos(phi)*sin(theta), sin(phi) ],
        #                     [  sin(theta),          cos(theta),        0 ],
        #                     [ -sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi) ]]
        
        # Alternate rotation:
        # Rotate by theta around z-axis, then tilt by phi
        #x = cos(theta)*cos(phi)*in_x - sin(theta)*in_y + cos(theta)*sin(phi)*in_z
        #y = sin(theta)*cos(phi)*in_x + cos(theta)*in_y + sin(theta)*sin(phi)*in_z
        #z =           -sin(phi)*in_x +          0*in_y +            cos(phi)*in_z


        # Three-axis rotation:
        # 1. Rotate about +z by eta 
        # 2. Tilt by phi with respect to +z (rotation about y-axis) then
        # 3. rotate by theta in-place (rotation about z-axis)

        eta = radians( eta )        # eta is orientation around the z axis (before reorientation)
        phi = radians( phi )        # phi is grain tilt (with respect to +z axis)
        theta = radians( theta )    # grain orientation (around the z axis)
        rotation_elements = [[  cos(eta)*cos(phi)*cos(theta)-sin(eta)*sin(theta) ,
                                    -cos(eta)*cos(phi)*sin(theta)-sin(eta)*cos(theta) ,
                                    -cos(eta)*sin(phi)                                   ],
                            [  sin(eta)*cos(phi)*cos(theta)+cos(eta)*sin(theta) ,
                                    -sin(eta)*cos(phi)*sin(theta)+cos(eta)*cos(theta) ,
                                    sin(eta)*sin(phi)                                    ],
                            [ -sin(phi)*cos(theta) ,
                                sin(phi)*sin(theta) ,
                                cos(phi)                                              ]]
        
        return rotation_elements

    def rotate_coordinates(self, in_x, in_y, in_z ):
        """Transforms from box coordinates into lattice coordinates, based
        on the pre-set rotation values."""

        if self.pargs['rotation']:
            x = self.rotation_matrix[0][0]*in_x + self.rotation_matrix[0][1]*in_y + self.rotation_matrix[0][2]*in_z
            y = self.rotation_matrix[1][0]*in_x + self.rotation_matrix[1][1]*in_y + self.rotation_matrix[1][2]*in_z
            z = self.rotation_matrix[2][0]*in_x + self.rotation_matrix[2][1]*in_y + self.rotation_matrix[2][2]*in_z
        
            return x, y, z
            
        else:
            return in_x, in_y, in_z


    def V(self, in_x, in_y, in_z):
        """Returns potential."""
        
        box_z = self.pargs['box_z_span']
        film_thick = self.pargs['film_thickness']
        
        if film_thick > box_z:
            status_print( "WARNING: The specified film thickness doesn't fit into the specified box." )

        if in_z > box_z/2.0 + film_thick/2.0:
            # This is below the film (substrate)
            return self.conversion_factor*self.pargs['rho_substrate']
        elif in_z < box_z/2.0 - film_thick/2.0:
            # This is above the film (superstrate/ambient)
            return self.conversion_factor*self.pargs['rho_ambient']
        else:
            # This is inside the thin film
            return self.V_in_film( in_x, in_y, in_z )


    def V_in_film(self, in_x, in_y, in_z):
        """Returns the potential in the thin film region (between
        film surface and substrate interface)."""
        
        return self.conversion_factor*self.pargs['rho_film']
            

    def to_string(self):
        """Returns a string describing the potential."""
        s = "Film potential: describes a generic thin film."
        return s


class TestPotential(FilmPotential):
    """A generic thin-film potential. Useful for testing."""
    
    def V_in_film(self, in_x, in_y, in_z):
        """Returns the potential in the thin film region (between
        film surface and substrate interface)."""
        
        x, y, z = self.rotate_coordinates( in_x, in_y, in_z )
        
        r = sqrt( (x-150.0)**2 + (y-150.0)**2 )
        
        if r < 50.0:
            return self.conversion_factor*self.pargs['rho1']
        else:
            return self.conversion_factor*self.pargs['rho_film']

    def to_string(self):
        """Returns a string describing the potential."""
        s = "Test potential: puts a cylinder somewhere."
        return s


class HexagonalCylindersPotential(FilmPotential):
    """Hexagonally packed cylinders.
    
    pargs should contain:
     rho_ambient        density of ambient
     rho_substrate      density of substrate
     rho1               density of cylinder cores
     rho_film           density of matrix
     radius             radius of cylinder cores
     repeat_spacing     center-to-center distance for hex lattice
     interfacial_spread roughness of the cylinder/matrix boundary
     """


    def V_in_film(self, in_x, in_y, in_z):
        """Returns the potential in the thin film region (between
        film surface and substrate interface)."""
        
        x, y, z = self.rotate_coordinates( in_x, in_y, in_z )


        delta_rho = self.pargs['rho1']-self.pargs['rho_film']
        core_radius = self.pargs['radius']
        interfacial_spread = self.pargs['interfacial_spread']
        
        unit_cell_x = self.pargs['repeat_spacing']*sqrt(3.0)     #unit_cell_x = 2*spacing*cos( radians( 30 ) )
        unit_cell_y = self.pargs['repeat_spacing']               #unit_cell_x = 2*spacing*sqrt(3)/2

        skew_unit_cell_x = unit_cell_x/2.0
        skew_unit_cell_y = unit_cell_y
        skew_unit_cell_slope = 1.0/(sqrt(3.0))      # Equivalent to 1/tan(30 degrees)


        # Calculate the indices of the unit cell we are inside
        # Conceptually this is done by converting the skewed hexagonal coordinate
        # system (where each unit cell is a parallelogram) into a rectangular
        # system. We then divide by the lattice dimensions to get indices.
        x_like_index = floor( x/(skew_unit_cell_x)  )
        y_like_index = floor( (y-x*skew_unit_cell_slope)/(skew_unit_cell_y)  )


        V = self.pargs['rho_film']    # Set to matrix value

        # Determine which quadrant of the unit cell we are in (and only compute that cylinder)
        if (x%skew_unit_cell_x) < (skew_unit_cell_x/2.0):
            if ((y-x*skew_unit_cell_slope)%(skew_unit_cell_y)) < (skew_unit_cell_y/2.0):
                xc = x_like_index*skew_unit_cell_x
                yc = y_like_index*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*x_like_index
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius)/interfacial_spread ) ) )
            else:
                xc = x_like_index*skew_unit_cell_x
                yc = (y_like_index+1)*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*x_like_index
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius)/interfacial_spread ) ) )
        else:
            if ((y-x*skew_unit_cell_slope)%(skew_unit_cell_y)) < (skew_unit_cell_y/2.0):
                xc = (x_like_index+1)*skew_unit_cell_x
                yc = y_like_index*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*(x_like_index+1)
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius)/interfacial_spread ) ) )
            else:
                xc = (x_like_index+1)*skew_unit_cell_x
                yc = (y_like_index+1)*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*(x_like_index+1)
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius)/interfacial_spread ) ) )


        return self.conversion_factor*V





    def to_string(self):
        """Returns a string describing the potential."""
        s = "Hexagonal Cylinder Potential: creates and array of hexagonally-packed cylinders."
        return s



class HexagonalCylindersJitterPotential(HexagonalCylindersPotential):
    """Hexagonally packed cylinders, with jitter (randomness) in
    cylinder sizes and positions.
    
    pargs should contain:
     rho_ambient        density of ambient
     rho_substrate      density of substrate
     rho1               density of cylinder cores
     rho_film           density of matrix
     radius             radius of cylinder cores
     repeat_spacing     center-to-center distance for hex lattice
     interfacial_spread roughness of the cylinder/matrix boundary
     positional_spread  jitter of cylinder centers from ideal lattice
     size_spread        jitter of cylinder sizes from the average radius
     """

    position_jitters = make_inner()

    def __init__(self, pargs={}, seed=2018):
        
        super(HexagonalCylindersJitterPotential, self).__init__(pargs=pargs)
        
        random.seed( seed )
        

    def jitter(self, x, y):
        """Returns the (jittered) position and size of the specified
        cylinder object. This function stores computed jitters so that
        they are not re-calculated, and so that they are consistent."""
        
        if self.position_jitters.has_key( x ):
            if self.position_jitters[x].has_key( y ):
                return self.position_jitters[x][y]
                
        self.position_jitters[x][y] = ( random.uniform(-self.pargs['positional_spread'],+self.pargs['positional_spread']),
                                    random.uniform(-self.pargs['positional_spread'],+self.pargs['positional_spread']),
                                    random.uniform(1.0-self.pargs['size_spread'],1.0+self.pargs['size_spread']) )
        return self.position_jitters[x][y]        
        

    def V_in_film(self, in_x, in_y, in_z):
        """Returns the potential in the thin film region (between
        film surface and substrate interface)."""
        
        x, y, z = self.rotate_coordinates( in_x, in_y, in_z )


        delta_rho = self.pargs['rho1']-self.pargs['rho_film']
        core_radius = self.pargs['radius']
        interfacial_spread = self.pargs['interfacial_spread']
        
        unit_cell_x = self.pargs['repeat_spacing']*sqrt(3.0)     #unit_cell_x = 2*spacing*cos( radians( 30 ) )
        unit_cell_y = self.pargs['repeat_spacing']               #unit_cell_x = 2*spacing*sqrt(3)/2

        skew_unit_cell_x = unit_cell_x/2.0
        skew_unit_cell_y = unit_cell_y
        skew_unit_cell_slope = 1.0/(sqrt(3.0))      # Equivalent to 1/tan(30 degrees)


        # Calculate the indices of the unit cell we are inside
        # Conceptually this is done by converting the skewed hexagonal coordinate
        # system (where each unit cell is a parallelogram) into a rectangular
        # system. We then divide by the lattice dimensions to get indices.
        x_like_index = floor( x/(skew_unit_cell_x)  )
        y_like_index = floor( (y-x*skew_unit_cell_slope)/(skew_unit_cell_y)  )


        V = self.pargs['rho_film']    # Set to matrix value

        # Determine which quadrant of the unit cell we are in (and only compute that cylinder)
        if (x%skew_unit_cell_x) < (skew_unit_cell_x/2.0):
            if ((y-x*skew_unit_cell_slope)%(skew_unit_cell_y)) < (skew_unit_cell_y/2.0):
                xj, yj, rj = self.jitter(x_like_index, y_like_index)
                xc = x_like_index*skew_unit_cell_x + xj
                yc = y_like_index*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*x_like_index + yj
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius*rj)/interfacial_spread ) ) )
            else:
                xj, yj, rj = self.jitter(x_like_index, y_like_index+1)
                xc = x_like_index*skew_unit_cell_x + xj
                yc = (y_like_index+1)*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*x_like_index + yj
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius*rj)/interfacial_spread ) ) )
        else:
            if ((y-x*skew_unit_cell_slope)%(skew_unit_cell_y)) < (skew_unit_cell_y/2.0):
                xj, yj, rj = self.jitter(x_like_index+1, y_like_index)
                xc = (x_like_index+1)*skew_unit_cell_x + xj
                yc = y_like_index*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*(x_like_index+1) + yj
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius*rj)/interfacial_spread ) ) )
            else:
                xj, yj, rj = self.jitter(x_like_index+1, y_like_index+1)
                xc = (x_like_index+1)*skew_unit_cell_x + xj
                yc = (y_like_index+1)*skew_unit_cell_y + skew_unit_cell_slope*skew_unit_cell_x*(x_like_index+1) + yj
                d = sqrt( ( x - xc )**2 + ( y - yc )**2 )
                V += delta_rho*( 1/(1 + exp( (d-core_radius*rj)/interfacial_spread ) ) )



        return self.conversion_factor*V





    def to_string(self):
        """Returns a string describing the potential."""
        s = "Hexagonal Cylinder Jitter Potential: creates and array of hexagonally-packed cylinders, with some randomness in position and size."
        return s



class NoisePotential(FilmPotential):
    """Describes a film that has a random 'noisy' makeup
    with a characteristic length-scale.
    
    pargs should contain:
     rho_ambient        density of ambient
     rho_substrate      density of substrate
     rho1               max density in film
     rho_film           min density in film
     repeat_spacing     characteristic length-scale of noise
     """


    def __init__(self, pargs={}, method='cgkit'):
        
        super(NoisePotential, self).__init__(pargs=pargs)
        
        self.pargs['delta_rho'] = self.pargs['rho1']-self.pargs['rho_film']
        self.spacing = self.pargs['repeat_spacing']

        self.method = method

        if self.method=='cgkit':
            # Uses the noise function from the module "The Python
            # Computer Graphics Kit" (cgkit). For further
            # information, and to download the module, see:
            # http://cgkit.sourceforge.net/
            # This has been tested with cgkit-2.0.0alpha8
            #
            # The noise is a 'Perlin noise function', which is essentially
            # an interpolation of grid of random points.
            
            # Import the cgkit module, which has the noise function.
            from cgkit import noise
            self.noise = noise.noise
            
        elif self.method=='perlin2d':
            # WARNING: This only creates noise in two dimensions.
            
            # Import the perlin noise module. For details, see:
            # http://twistedmatrix.com/users/acapnotic/wares/code/perlin/
            import sys
            sys.path.append( './extras' )
            import perlin
            
            W, L, H = 1000.0, 800.0, 120.0
            width = int( W/self.spacing )
            length = int( L/self.spacing )
            height = int( H/self.spacing )
            
            self.noise = perlin.PerlinNoise( (width,height) )
            
        elif self.method=='perlin3d':
            # WARNING: The perlin3d method currently fails (3d fields not supported).
            
            # Import the perlin noise module. For details, see:
            # http://twistedmatrix.com/users/acapnotic/wares/code/perlin/
            import sys
            sys.path.append( './extras' )
            import perlin
            
            W, L, H = 1000.0, 800.0, 120.0
            width = int( W/self.spacing )
            length = int( L/self.spacing )
            height = int( H/self.spacing )
            
            self.noise = perlin.PerlinNoise( (width,length,height) )
            
            
        elif self.method=='pnoise':
            # WARNING: This method doesn't work if you rotate the structure.
            
            # A direct transcription into Python of Perlin's code. See:
            # http://www.fundza.com/c4serious/noise/perlin/perlin.html
            from noise import pnoise3
            self.noise = pnoise3
            
        else:
            print( "ERROR: NoisePotential received an invalid 'method' specification." )
        


    def V_in_film(self, in_x, in_y, in_z):
        """Returns the potential in the thin film region (between
        film surface and substrate interface)."""
        
        x, y, z = self.rotate_coordinates( in_x, in_y, in_z )
        xs = x/self.spacing
        ys = y/self.spacing
        zs = z/self.spacing

        if self.method=='cgkit':
            # See http://cgkit.sourceforge.net/doc2/noise.html
            # for details about the noise function.
            V = self.pargs['rho_film'] + self.pargs['delta_rho']*(self.noise( xs, ys, zs ))
        elif self.method=='perlin2d':
            # WARNING: This only varies the density in two dimensions.
            V = self.pargs['rho_film'] + self.pargs['delta_rho']*0.5*(self.noise.value_at( (xs, ys) )+1)
        elif self.method=='perlin3d':
            # WARNING: The perlin3d method currently fails (3d fields not supported).
            V = self.pargs['rho_film'] + self.pargs['delta_rho']*0.5*(self.noise.value_at( (xs, ys, zs) )+1)
        elif self.method=='pnoise':
            # WARNING: This method doesn't work if you rotate the structure.
            V = self.pargs['rho_film'] + self.pargs['delta_rho']*self.noise( xs, ys, zs )
        else:
            V = 0.0
        
        
        
        return self.conversion_factor*V


    def to_string(self):
        """Returns a string describing the potential."""
        s = "Noise potential: creates a noise with a given characteristic lenghtscale."
        return s



