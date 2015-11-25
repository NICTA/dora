Demos
=====

Sampling by calling the Python API
..................................

In this demo, we use the active sampling Python API to learn a
Gaussian Process model. We collect some initial randomly located 
training data with noisy targets ``y_train`` and inputs ``X_train``
for the purpose of selecting initial model hyperparameters, and proceed to use 
targeted sampling to efficiently continue the exploration. The sampling
distribution and the corresponding model are shown at various stages of data
acquisition.

.. plot:: ../demo/demo_gp_python_api.py




References
----------

.. [1] Carl Edward Rasmussen and Christopher KI Williams "Gaussian processes
       for machine learning." the MIT Press 2.3 (2006): 4.
