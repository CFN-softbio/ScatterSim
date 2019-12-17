# TODO : Add caching. Basically if q is in a certain range, interpolate
import numpy as np
# the gamma function
from scipy.special import gamma

# PeakShape


class PeakShape(object):
    """Defines an x-ray scattering peak. Once initialized, the object will
    return the height of the peak for any given qs position (distance from peak
    center)."""
    infinity = 1e300

    def __init__(self, nu=0, delta=0.1, gauss_cutoff=200, lorentz_cutoff=0.005,
                 product_terms=None, gamma_method=False, q1=None, slope=None,):
        '''2019/12/12 YG Update init with q1 and slope for Williamson-Hall analysis to get the lattice strain'''

        self.sigma = None
        self.fwhm = None

        self.gauss_cutoff = gauss_cutoff
        self.lorentz_cutoff = lorentz_cutoff
        self.gamma_method = gamma_method

        self.infinity = max(self.infinity, 1.01 * self.gauss_cutoff)

        self.requested_product_terms = product_terms
        self.reshape(nu, delta)
        
        self.q1=q1
        self.slope=slope

    def reshape(self, nu=0, delta=0.1):
        self.nu = 1.0 * nu
        self.delta = 1.0 * delta

        if self.requested_product_terms is None:
            # Figure out how many product terms we need, based on the value of
            # nu
            if self.nu < 1:
                self.product_terms = 20
            elif self.nu > 1000:
                self.product_terms = 2000
            else:
                self.product_terms = int(
                    ((self.nu - 1) / (1000 - 1)) * (2000 - 20) + 20)
        else:
            self.product_terms = self.requested_product_terms

        if self.nu >= self.lorentz_cutoff and self.nu <= self.gauss_cutoff:
            self.gamma_nu = np.sqrt(
                np.pi) * gamma((self.nu + 1) / 2) / gamma(self.nu / 2)

        self.already_computed = {}

    def gaussian(self, sigma=None, delta=None, fwhm=None):
        """Sets the peak to be a pure Gaussian (overrides any other
        setting)."""

        self.nu = self.infinity
        self.already_computed = {}

        if sigma is None and delta is None and fwhm is None:
            print(
                "WARNING: No width specified for Gaussian peak. A width has "
                "been assumed.")
            self.sigma = 0.1
            self.delta = np.sqrt(8 / np.pi) * self.sigma
            self.fwhm = 2 * np.sqrt(2 * np.log(2)) * self.sigma
        elif sigma is not None:
            # Sigma takes priority
            self.sigma = sigma
            self.delta = np.sqrt(8 / np.pi) * self.sigma
            self.fwhm = 2 * np.sqrt(2 * np.log(2)) * self.sigma
        elif fwhm is not None:
            self.fwhm = fwhm
            self.sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
            self.delta = np.sqrt(8 / np.pi) * self.sigma
        else:
            # Use delta to define peak width
            self.delta = delta
            self.sigma = np.sqrt(np.pi / 8) * self.delta
            self.fwhm = 2 * np.sqrt(2 * np.log(2)) * self.sigma

    def lorentzian(self, delta=None, fwhm=None):
        """Sets the peak to be a pure Lorentzian (overrides any other
        setting)."""

        self.nu = 0
        self.already_computed = {}

        if delta is None and fwhm is None:
            print(
                "WARNING: No width specified for "
                "Lorentzian peak. A width has been assumed.")
            self.delta = 0.1
            self.fwhm = self.delta
        elif delta is not None:
            self.delta = delta
            self.fwhm = self.delta
        else:
            self.fwhm = fwhm
            self.delta = self.fwhm

        self.sigma = None

    def __call__(self, qs):
        """Returns the height of the peak at the given qs position.
        The peak is centered about qs = 0. The shape and width of the peak is
        based on the parameters it was instantiated with."""

        qs = np.abs(qs)    # Peak is symmetric
        if self.q1 is not None and self.slope is not None:
            self.delta = self.slope *( qs - self.q1 ) + self.delta 
        if self.nu > self.gauss_cutoff:
            # Gaussian
            val = (2 / (np.pi * self.delta)) * \
                np.exp(-(4 * (qs**2)) / (np.pi * (self.delta**2)))
        elif self.nu < self.lorentz_cutoff:
            # Lorentzian
            val = (self.delta / (2 * np.pi)) / (qs**2 + ((self.delta / 2)**2))
        else:
            # Brute-force the term
            val = (2 / (np.pi * self.delta))

            if self.gamma_method:

                print("WARNING: The gamma method does not currently work.")

                # Use gamma functions
                y = (4 * (qs**2)) / ((np.pi**2) * (self.delta**2))

                # Note that this equivalence comes from the paper:
                #   Scattering Curves of Ordered Mesoscopic Materials
                #   S. Förster, A. Timmann, M. Konrad, C. Schellbach, A. Meyer,
                #   S.S. Funari, P. Mulvaney, R. Knott,
                #   J. Phys. Chem. B, 2005, 109 (4), pp 1347–1360 DOI:
                #   10.1021/jp0467494
                #   (See equation 27 and last section of Appendix.)
                # However there seems to be a typo in the paper, since it does
                # not match the brute-force product.

                numerator = gamma((self.nu / 2) + 1.0j * self.gamma_nu * y)
                # numerator = gamma.GammaComplex( (self.nu/2) +
                # 1.0j*self.gamma_nu*(sqrt(y)) )
                denominator = gamma(self.nu / 2)
                term = numerator / denominator

                val *= 0.9 * term * term.conjugate()

            else:
                # Use a brute-force product calculation
                for n in range(0, self.product_terms):
                    # print n, self.nu, self.gamma_nu
                    term1 = (self.gamma_nu**2) / ((n + self.nu / 2)**2)
                    # print "  " + str(term1)
                    term2 = (4 * (qs**2)) / ((np.pi**2) * (self.delta**2))
                    val *= 1 / (1 + term1 * term2)

        return val
