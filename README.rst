=======
dora 
=======

------------------------------------------------------------------------------
A library for Bayesian active sampling with non-parametric models
------------------------------------------------------------------------------

Dora simultaneously builds a non-parametric model of an observable
process, and directs new measurements to learn about the process. Inherently, 
it takes actions to improve the model (exploration), and to achieve desired
goals given its knowledge (exploration).

The models utilised are non-parametric models such as bilinear interpolation
of a Delaunay triangulation or Gaussian Process regression. We are building
this tool to be able to plug in other models, so it will shortly be possible
to link in our *revrand* package for fast Bayesian linear regression that
behaves similar to GP regression so that kernel-regression like modeling can
be applied to large sampled datasets. 

The key features of Dora are:

- a Python library to design experimental sampling to simultaneously explore 
  and exploit an underlying process
  
- a RESTful web service that can be called from any other launguages providing
  the core functionality of the Python library

- The ability to use a variety of process models including Gaussian Process
  regression [1]_, Delaunay triangulation, and models from the *Revrand*
  library [2]_.

- a suite of strategies to conduct information, risk or value based sampling
  to learn these models efficiently or target phenomenon of interest.


Quickstart
----------

To install, simply run ``setup.py``:

.. code:: console

   $ python setup.py install

or install with ``pip``:

.. code:: console

   $ pip install git+https://github.com/nicta/dora.git@release

Refer to `docs/installation.rst <docs/installation.rst>`_ for advanced 
installation instructions.

Have a look at some of the `demos <demo/>`_, e.g.: 

.. code:: console

   $ python demo/demo_gp_python_api.py

The demos include examples of different underlying models (GP and Delaunay),
and examples of both Python 3 code calling the api directly, and the general
case of calling a server through a HTTP REST interface.

Here is a very brief example of how to use active sampling with a Gaussian
Process model in Python. We are assuming we can collect some limited training
data with noisy targets ``y_train``, inputs ``X_train`` for the purpose of
selecting initial model hyperparameters, and we now want to use targeted
sampling to efficiently continue the exploration.


.. code:: python
    import dora.active_sampling as dora
    import dora.regressors.gp as gp
    from example_processes import simulate_measurement
    import numpy as np
    
    # Set up a sampling problem:
    n_initial_sample = 50
    lower = [0, 0]
    upper = [1, 1]
    X_train = dora.random_sample(lower, upper, n_initial_sample)
    y_train = np.asarray([simulate_measurement(i) for i in X_train])

    # Set up a sampler using Dora
    sampler = dora.Gaussian_Process(lower, upper, X_train, y_train,
                                    add_train_data=False)

    # Run the active sampling:
    logging.info('Actively sampling new points..')
    target_samples = 501
    for i in range(target_samples):
        # Note - Dora provides a sample X, and a sample Id
        # While we are doing this in a loop, in reality the observations can
        # be asynchronously observed and returned out of order.
        newX, newId = sampler.pick()
        observation = simulate_measurement(newX)
        sampler.update(newId, observation)


Useful Links
------------

Home Page
    http://github.com/nicta/dora

Documentation
    http://nicta.github.io/dora

Issue tracking
    https://github.com/nicta/dora/issues

Bugs & Feedback
---------------

For bugs, questions and discussions, please use 
`Github Issues <https://github.com/NICTA/dora/issues>`_.


References
----------

.. [1] Gaussian Processes for Machine Learning, Carl Edward Rasmussen and 
   Chris Williams, the MIT Press, 2006

.. [2] NICTA 'Revrand <https://github.com/NICTA/revrand>'_ library.

.. [3] Osborne, M. (2010). Bayesian Gaussian Processes for Sequential 
   Prediction, Optimisation and Quadrature (PhD thesis). PhD thesis, 
   University of Oxford.

.. [4] Garnett, R., Osborne, M. A., & Roberts, S. J. (2010). Bayesian 
   optimization for sensor set selection. International Conference on 
   Information Processing in Sensor Networks (pp. 209–219).



Copyright & License
-------------------

Copyright 2015 National ICT Australia.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
