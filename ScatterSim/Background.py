# Background
class Background(object):
    def __init__(self, constant=0.0, prefactor=0.0, alpha=-2.0, prefactor2=0.0, alpha2=-1.5):
        ''' This is a background of type:
            constant + prefactor*q^alpha + prefactor2*q^alpha2

            Nothing special, but since it is so common an object is left for bookkeeping.
        '''
        self.constant=constant
        self.prefactor=prefactor
        self.alpha=alpha

        self.prefactor2=prefactor2
        self.alpha2=alpha2

    def update(self, constant=0.0, prefactor=0.0, alpha=-2.0, prefactor2=0.0, alpha2=-1.5):
        ''' Routine to update if necessary.
        '''
        self.constant=constant
        self.prefactor=prefactor
        self.alpha=alpha

        self.prefactor2=prefactor2
        self.alpha2=alpha2

    def __call__(self, q):
        ''' Returns the background. '''
        return self.prefactor*( q**(self.alpha) ) + self.prefactor2*( q**(self.alpha2) ) + self.constant
