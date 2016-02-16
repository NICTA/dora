"""
Example processes for demonstration.
"""
import numpy as np


def simulate_measurement(X):
    """
    A binary image of a circle as a test problem for sampling.
    """
    return (np.sum((np.asarray(X) - 0.5)**2, axis=-1) < 0.1).astype(float)


def simulate_measurement_vector(X, uid=None):
    """
    An example of a vector valued observation process.

    The true model is two Gaussian blobs located at centre1 and centre2.

    The model is queried with a 2-D location and returns an vector of length
    output_len with values corresponding to the underlying function's values
    along a z-axis.
    """
    assert X.shape == (2,)
    measurement_range = [0, 1]
    centre1 = np.array([1.5, 1.4, 0.3])
    centre2 = np.array([2.50, 2.0, 0.7])
    l1 = 0.2
    l2 = 0.3
    output_len = 20

    Z = np.arange(measurement_range[0], measurement_range[1],
                  (measurement_range[1] - measurement_range[0]) / output_len)
    dist1 = [np.sqrt((centre1[0] - X[0])**2 + (centre1[1] - X[1])**2 +
             (centre1[2] - z)**2) for z in Z]
    dist2 = [np.sqrt((centre2[0] - X[0])**2 + (centre2[1] - X[1])**2 +
             (centre2[2] - z)**2) for z in Z]
    measurement = np.asarray([np.exp(-dist1[i] / l1) + np.exp(-dist2[i] / l2)
                             for i in range(len(dist1))])
    return uid, measurement
