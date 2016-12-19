I cleaned up ScatterSim into a few files. This file is the first and contains
the NanoObjects.
The files are:
    - NanoObjects.py
    - CompositeNanoObjects.py
    - LatticeObjects.py
    - PeakShape.py

The code used to be in MultiComponentModel.py. I have only moved some code from
there.

There is still a lot to do to get this code up and running Here are some notes
for things that need to be considered later.

Notes :
    1. what is seed? removed for now
    2. Not caching for now (see old code for caching)
        also removed in lattice. I had some troubles. We may need to rethink 
        how to cache anyway to save the most time.
    3. changed from inputs being qx,qy,qz to one qvector
    4. Functions not tested yet. When testing one by one, should
        document somewhere what was tested.
    5. Need scipy version 0.18.1 or higher (for spherical_jn)
        if not, could hard code this...

For documentation:
    1. 'qvector' will signify an array as such : np.array([qx, qy, qz])
        I may forget to document its meaning across all functions
    2. pargs['orientation_spread'] = phi_start, phi_end, theta_start, theta_end
    3. rebuild function meant to reset everything. Run this when need to reset
        (don't have to specify pargs)

Acknowledgements
    Created: March 8th, 2013 (original code)
             December 17th, 2016 (new version)
    Creator: Kevin G. Yager
    Original Contributors: Yugang Zhang
    Contributors to new code: Julien Lhermitte
