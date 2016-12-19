from ScatterSim.CompositeNanoObjects import CompositeNanoObject
from copy import deepcopy
import numpy as np

# Lattice
class Lattice:
    """Defines a lattice type and provides methods for adding objects to the
    lattice. This is the starting point of all the crystalline samples.  It is
    recommended here to set the basis vectors (lattice spacing, and alpha,
    beta, gamma) and then have objects inherit this, such as FCC BCC here.
    Finally, you need to add a list of NanoObjects with form factors for this
    to work.

    sigma_D : the DW factor. Can be one number or a two tuple:
            (unit normal, DW factor) pair
    """
    def __init__(self, objects, lattice_spacing_a=1.0, lattice_spacing_b=None,
                 lattice_spacing_c=None, alpha=90, beta=None, gamma=None,
                 sigma_D=0.01, lattice_positions=None, lattice_coordinates=None, symmetry=None):
        ''' Initialize Lattice object.
                objects : list of NanoObjects
        '''
        # accept just one object too
        if not isinstance(objects, list):
            objects = [objects]

        self.lattice_spacing_a = lattice_spacing_a
        if lattice_spacing_b==None:
            self.lattice_spacing_b = lattice_spacing_a
        else:
            self.lattice_spacing_b = lattice_spacing_b

        if lattice_spacing_c==None:
            self.lattice_spacing_c = lattice_spacing_a
        else:
            self.lattice_spacing_c = lattice_spacing_c

        self.lattice_spacings = np.array([self.lattice_spacing_a,
                                         self.lattice_spacing_b,
                                         self.lattice_spacing_c])

        self.alpha = np.radians(alpha)
        if beta==None:
            self.beta = np.radians(alpha)
        else:
            self.beta = np.radians(beta)
        if gamma==None:
            self.gamma = np.radians(alpha)
        else:
            self.gamma = np.radians(gamma)

        if lattice_positions is None:
            lattice_positions = ['center']
        if lattice_coordinates is None:
            lattice_coordinates = np.array([[0,0,0]])
        if symmetry is None:
            symmetry = {
                    'crystal family' : 'N/A',
                    'crystal system' : 'N/A',
                    'Bravais Lattice' : 'N/A',
                    'crystal class' : 'N/A',
                    'point group' : 'N/A',
                    'space group' : 'N/A',
                    }

        # now set
        self.sigma_D = np.array(sigma_D)          # Lattice disorder
        self.symmetry = symmetry
        self.lattice_positions = lattice_positions
        self.lattice_coordinates = lattice_coordinates
        self.number_objects = len(self.lattice_coordinates)

        # finally make objects
        self.lattice_objects = list()
        if len(objects) == 1:
            # make multiple copies
            obj = objects[0]
            for i in range(self.number_objects):
                self.lattice_objects.append(deepcopy(obj))
        elif len(objects) == self.number_objects:
            # move objects into list
            self.lattice_objects = objects
        else:
            raise ValueError("Can only handle one or {} " +
                             "objects".format(self.number_objects) +
                             ", but received {}".format(len(objects))
                             )

        # now update the positions
        for i in range(len(self.lattice_objects)):
            pos = self.lattice_coordinates[i]*self.lattice_spacings
            self.lattice_objects[i].set_origin(*pos)



    def update_sigma_D(self, sigma_D):
        '''sigma_D : the DW factor. Can be one number or a two tuple:
        (unit normal, DW factor) pair '''
        self.sigma_D = np.array(sigma_D)


    def unit_cell_volume(self):
        V = np.sqrt( 1 - (cos(self.alpha))**2 - (cos(self.beta))**2 - (cos(self.gamma))**2 + 2*cos(self.alpha)*cos(self.beta)*cos(self.gamma) )
        V *= self.lattice_spacing_a*self.lattice_spacing_b*self.lattice_spacing_c

        return V

    def sum_over_objects(self, funcname, shape, dtype, *args, **kwargs):
        ''' Sum the function with name 'funcname' over variable 'vec'.
        Forwards other keyword arguments to function

        Parameters
        ---------
        funcname : the function name
        shape : the shape of the result
        dtype : the data type
        args : arguments to the function
        kwargs : keyword arguments to function

        '''
        res = np.zeros(shape,dtype=dtype)
        cts = 0.

        for obj in self.lattice_objects:
            restmp = getattr(obj, funcname)(*args,**kwargs)
            res += restmp

        return res


    # Components of intensity calculation

    def multiplicity_lookup(self, h, k, l):
        """Returns the peak multiplicity for the given reflection."""
        return 1

    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
        return 1

    def q_hkl(self, h, k, l):
        """Determines the position in reciprocal space for the given reflection."""

        # NOTE: This is assuming cubic/rectangular only!
        qhkl_vector = np.array([ 2*np.pi*h/(self.lattice_spacing_a), \
                        2*np.pi*k/(self.lattice_spacing_b), \
                        2*np.pi*l/(self.lattice_spacing_c) ])
        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return (qhkl, qhkl_vector)

    def q_hkl_length(self, h, k, l):
        qhkl, qhkl_vector = self.q_hkl(h,k,l)

        qhkl = np.sqrt( qhkl_vector[0]**2 + qhkl_vector[1]**2 + qhkl_vector[2]**2 )

        return qhkl

    def iterate_over_hkl_compute(self, max_hkl=6):
        """Returns a sequence of hkl lattice peaks (reflections)."""

        # r will contain the return value, an array with rows that contain:
        # h, k, l, peak multiplicity, symmetry factor, qhkl, qhkl_vector
        r = []

        for h in range(-max_hkl,max_hkl+1):
            for k in range(-max_hkl,max_hkl+1):
                for l in range(-max_hkl,max_hkl+1):

                    # Don't put a reflection at origin
                    if not (h==0 and k==0 and l==0):
                        m = self.multiplicity_lookup(h, k, l)
                        f = self.symmetry_factor(h, k, l)

                        if m!=0 and f!=0:
                            qhkl, qhkl_vector = self.q_hkl(h,k,l)
                            r.append( [ h, k, l, m, f, qhkl, qhkl_vector ] )

        return r


    def iterate_over_hkl(self, max_hkl=6):
        ''' This code would previously check if hkl was already computed.
            I removed it for now. We should eventually add optimizations such
            as this again later.
        '''
        self.hkl_list = self.iterate_over_hkl_compute(max_hkl=max_hkl)
        return self.hkl_list

    def sum_over_hkl(self, q, peak, max_hkl=6):
        summation = 0

        for h, k, l, m, f, qhkl, qhkl_vector in self.iterate_over_hkl(max_hkl=max_hkl):

            fs = self.form_factor(qhkl_vector)
            term1 = (fs*fs.conjugate()).real
            # if 
            #term2 = np.exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
            term2 = self.G_q(qhkl_vector)
            term3 = peak( q-qhkl )

            summation += (m*(f**2)) * term1 * term2 * term3

        return summation

    def sum_over_hkl_array(self, q_list, peak, max_hkl=6):
        ''' Sum the 1D powder curve over the q positions specified by the
        symmetry. Still need to review to see if it can be optimized.
        '''
        summation = np.zeros( (len(q_list)) )
        hkl_info = self.iterate_over_hkl(max_hkl=max_hkl)

        for h, k, l, m, f, qhkl, qhkl_vector in hkl_info:

            fs = self.sum_over_objects(qhkl_vector, h, k, l)
            term1 = fs*fs.conjugate()
            # Debye Waller factor
            term2 = np.exp( -(self.sigma_D**2) * (qhkl**2) * (self.lattice_spacing_a**2) )
            term2 = self.G_q(qhkl_vector)

            summation += (m*(f**2)) * term1.real * term2 * peak.val_array( q_list, qhkl )

        return summation


    # Form factor computations

    def form_factor(self, qvec):
        """Returns the sum of the form factor of particles in the unit cell."""
        return self.sum_over_objects('form_factor', qvec[0].shape, complex, qvec)

    def form_factor_squared(self, qvec):
        """Returns the sum of the form factor of particles in the unit cell."""
        return self.sum_over_objects('form_factor_squared', qvec[0].shape, float, qvec)

    def form_factor_isotropic(self, q):
        """Returns the sum of the form factor of particles in the unit cell."""
        return self.sum_over_objects('form_factor_isotropic', q.shape, complex, q)

    def form_factor_squared_isotropic(self, q):
        """Returns the sum of the form factor of particles in the unit cell."""
        return self.sum_over_objects('form_factor_squared_isotropic', q.shape, float, q)

    def V(self, rvec):
        """Returns the sum of the form factor of particles in the unit cell."""
        return self.sum_over_objects('V', rvec[0].shape, float, rvec)

    def G_q(self,q):
        ''' the G(q) from the Debye-Waller factor.
            If sigma_D is a two tuple, then compute the DW factor per unit
            normal.
            q must be a 3-vector

            Note : normalized to lattice_spacing for now for backwards compatibility
            so sigma = 1/(2sd ld)
        '''
        if self.sigma_D.ndim == 0:
            res = np.exp( -(self.sigma_D**2) * (np.linalg.norm(q)**2 * self.lattice_spacing_a**2) )
        else:
            res = 1.
            for sd, unrmx, unrmy, unrmz in self.sigma_D:
                # need to fix later, should be array form
                u = np.array([unrmx,unrmy, unrmz])
                res *= np.exp( -(sd**2) * (np.dot(q,u)**2) * (self.lattice_spacing_a**2) )
        return res


    def beta_numerator(self, q, num_phi=50, num_theta=50):
        """returns the numerator of the beta ratio: |<u(q)>|^2 = sum_j[ |<f_j(q)>|^2 ] """
        return self.sum_over_objects('beta_numerator', q.shape, float, rvec)


    def beta_ratio(self, q, num_phi=50, num_theta=50, approx=False):
        """Returns the beta ratio: |<U(q)>|^2 / <|U(q)|^2>
        for the lattice."""
        # numerator over denominator
        beta_num = self.sum_over_objects('beta_numerator', q.shape, float, rvec)
        beta_denom = self.sum_over_objects('form_factor_squared', q.shape, float, rvec)
        beta = beta_num/beta_denom

        return beta

    # Final S(q) and Z_0(q)
    ########################################

    def structure_factor_isotropic(self, q, peak, c=1.0, background=None, max_hkl=6):
        """Returns the structure factor S(q) for the specified q-position."""

        P = self.form_factor_intensity_isotropic(q)

        S = (c/(q**2 * P))*self.sum_over_hkl(q, peak, max_hkl=max_hkl)

        if background==None:
            return S
        else:
            return S + background.val(q)/P

    def intensity(self, q, peak, c=1.0, background=None, max_hkl=6):
        """Returns the predicted scattering intensity.
            This is Z0(q) in Kevin's paper
        """

        PS = (c/(q**2))*self.sum_over_hkl(q, peak, max_hkl=max_hkl)

        if background==None:
            return PS
        else:
            return background.val(q) + PS

    # Outputs
    def to_string(self):
        """Returns a string describing the lattice."""
        s = "Lattice of type: " + self.__class__.__name__ + "\n"

        s += "    Family: %s   System: %s   Bravais: %s   Class: %s   Space Group: %s\n" % (self.symmetry['crystal family'], \
                                                                                    self.symmetry['crystal system'], \
                                                                                    self.symmetry['Bravais lattice'], \
                                                                                    self.symmetry['crystal class'], \
                                                                                    self.symmetry['space group'], \
                                                                                    )

        s += "    (a, b, c) = (%.3f,%.3f,%.3f) in nm\n" % (self.lattice_spacing_a,self.lattice_spacing_b,self.lattice_spacing_c)
        s += "    (alpha, beta, gamma) = (%.3f,%.3f,%.3f) in radians\n" % (self.alpha,self.beta,self.gamma)
        s += "                         = (%.2f,%.2f,%.2f) in degrees\n" % (degrees(self.alpha),degrees(self.beta),degrees(self.gamma))
        s += "    volume = %.4f nm^3\n\n" % self.unit_cell_volume()
        s += "    Objects:\n"
        for i, obj in enumerate(self.objects):
            if i<len(self.positions):
                pos = self.positions[i]
            else:
                pos = '---'
            s += "        %d (%s)\t %s (%s)\n" % (i, pos, obj.__class__.__name__, obj.to_short_string() )
        s += "    Unit cell:\n"
        for i, pos, xi, yi, zi, obj in self.iterate_over_objects():
            objstr = obj.__class__.__name__
            s += "        %d (%s)\n" % (i, pos)
            s += "           \t%s (%s)\n" % (objstr, obj.to_short_string())
            s += "           \t  at pos = (%.3f,%.3f,%.3f)\n" % (xi, yi, zi)
        return s

# Different Lattices

# SimpleCubic Lattice
class SimpleCubic(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):
        # prepare variables of lattice
        symmetry = {
            'crystal family' : 'cubic',
            'crystal system' : 'cubic',
            'Bravais lattice' : 'P',
            'crystal class' : 'hexoctahedral',
            'point group' : 'm3m',
            'space group' : 'Pm3m',
        }

        lattice_positions = ['corner']


        lattice_coordinates = [ (0.0, 0.0, 0.0), \
                                    ]

        # now call parent function to initialize
        super(SimpleCubic, self).__init__(objects, lattice_spacing_a=lattice_spacing_a, sigma_D=sigma_D,
                alpha=90, beta=90, gamma=90, symmetry=symmetry, lattice_positions=lattice_positions,
                lattice_coordinates=lattice_coordinates)

    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""

        return 1

    def unit_cell_volume(self):

        return self.lattice_spacing_a**3
