#!/usr/bin/env python

from distutils.core import setup

classifiers = [
"Development Status :: 4 - Beta",
"Intended Audience :: Developers",
"Intended Audience :: Science/Research",
"License :: OSI Approved :: BSD License",
"Programming Language :: Python",
"Programming Language :: C",
"Topic :: Software Development :: Code Generators",
"Topic :: Scientific/Engineering",
"Operating System :: OS Independent"
]

setup(name             = "np_inline",
      platforms        = ["any"],
      version          = "0.3",
      description      = "Simple inline C calls using numpy arrays.",
      author           = "J. David Lee",
      author_email     = "johnl@cs.wisc.edu",
      maintainer       = "J. David Lee",
      maintainer_email = "johnl@cs.wisc.edu",
      url              = "http://www.cs.wisc.edu/~johnl/np_inline/",
      py_modules       = ['np_inline'],
      classifiers      = classifiers
     )
