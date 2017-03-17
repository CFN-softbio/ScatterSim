Todo for code (not comprehensive):
1. what is seed? removed for now
2. Not caching for now (see old code for caching)
        also removed in lattice. I had some troubles. We may need to rethink 
        how to cache anyway to save the most time.
3. changed from inputs being qx,qy,qz to one qvector
4. Functions not tested yet. When testing one by one, should
        document somewhere what was tested.
5. Need scipy version 0.18.1 or higher (for spherical_jn)
        if not, could hard code this...
