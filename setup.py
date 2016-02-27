#!/usr/bin/env python
# to install:
# python setup.py install

import setuptools
from distutils.core import setup, Extension
#from setupext import ext_modules
#import versioneer
import numpy as np
import os

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
#def read(fname):
    #return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='ScatterSim',
    #version=versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    author='Kevin Yager',
    description="Meso scale SAXS Powder Simulations",
    packages=setuptools.find_packages(exclude=['doc']),
    include_dirs=[np.get_include()],
    install_requires=['six', 'numpy'],  # essential deps only
#    ext_modules=ext_modules,
    keywords='Xray Analysis Meso Cluster Powder Diffraction SAXS',
    license='BSD',
    #classifiers=['Development Status :: 3 - Alpha',
                 #"License :: OSI Approved :: BSD License",
                 #"Programming Language :: Python :: 2.7",
                 #"Programming Language :: Python :: 3.4",
                 #"Topic :: Scientific/Engineering :: Physics",
                 #"Topic :: Scientific/Engineering :: Chemistry",
                 #"Topic :: Software Development :: Libraries",
                 #"Intended Audience :: Science/Research",
                 #"Intended Audience :: Developers",
                 #], requires=['numpy']
    )

