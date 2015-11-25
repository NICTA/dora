Available Regressors
===================

Delaunay Triangulation
----------------------

Using the Scipy implementation.


NICTA/revrand
-------------

The NICTA library for Generalised Bayesian linear regression, and kernel-like
models that scale to large data. See more at:
    http://github.com/nicta/revrand



Gaussian Process Regression
----------------------------

This is a NICTA developed module implementing Gaussian Process regression.
This component is not documented in detail here, but its key components include:

.. currentmodule:: dora.regressors.gp

Building and learning kernels:

.. autosummary::
   :toctree: generated/
    compose
    kernel
    learn
    auto_range


Supervised prediction from a training dataset:
    
.. autosummary::
   :toctree: generated/

    condition
    query
    add_data
    mean
    variance
    predict
   

