Installation
============

Dependencies
------------

Minimal
*******

- numpy
- scipy

Optional
********

A library of scalable Bayesian generalised linear models with fancy features
  - revrand (v0.6.5)

REST Server interface
  - requests
  - unipath

Visualization and interactive computing
  - matplotlib
  - visvis
  - ipython[notebook]

Development
***********

Documentation and deployment
  - sphinx
  - ghp-import

Testing
  - pytest


Development with Conda
----------------------

Run in repository base dir:

.. code:: console

    conda env create -f environment.yml
    source activate dora
    pip install -e .
