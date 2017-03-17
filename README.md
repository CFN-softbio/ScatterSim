This library is meant to compute the expected scattering from crystalline meso-sctructures.

For documentation:
1. 'qvector' will signify an array as such : np.array([qx, qy, qz])
        I may forget to document its meaning across all functions
2. ``` python
    pargs['orientation_spread'] = phi_start, phi_end, theta_start, theta_end
```
3. rebuild function meant to reset everything. Run this when need to reset
        (don't have to specify pargs)

Acknowledgements

Created: March 8th, 2013 (original code)
    December 17th, 2016 (new version)

Creator: Kevin G. Yager

Original Contributors: Kevin G. Yager and Yugang Zhang

Contributors to new code: Julien Lhermitte


References:
- Theory behind code:
    Yager, Kevin G., et al. "Periodic lattices of arbitrary nano-objects: modeling and applications for self-assembled systems." Journal of Applied Crystallography 47.1 (2014): 118-129.

Link:

    https://doi.org/10.1107/S160057671302832X

- Alternate reading:

    Senesi, Andrew J., and Byeongdu Lee. "Small-angle scattering of particle assemblies." Journal of Applied Crystallography 48.4 (2015): 1172-1182.

Link: https://doi.org/10.1107/S1600576715011474
