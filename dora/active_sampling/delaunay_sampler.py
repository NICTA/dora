"""
Delaunay Sampler Module.

Provides the Delaunay Sampler Class which contains the strategies for
active sampling a spatial field
"""
from dora.active_sampling.base_sampler import Sampler, grid_sample

import numpy as np

from scipy.spatial import Delaunay as ScipyDelaunay


class Delaunay(Sampler):
    """
    Delaunay Class.

    Inherits from the Sampler class and augments pick and update with the
    mechanics of the Delanauy triangulation method

    Attributes
    ----------
    triangulation : scipy.spatial.qhull.Delaunay
        The Delaunay triangulation model object
    simplex_cache : dict
        Cached values of simplices for Delaunay triangulation
    explore_priority : float
        The priority of exploration against exploitation

    See Also
    --------
    Sampler : Base Class
    """

    name = 'Delaunay'

    def __init__(self, lower, upper, explore_priority=0.0001):
        """
        Initialise the Delaunay class.

        .. note ::

            Currently only supports rectangular type restrictions on the
            parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)
        self.triangulation = None  # Delaunay model
        self.simplex_cache = {}  # Pre-computed values of simplices
        self.explore_priority = explore_priority

    def update(self, uid, y_true):
        """
        Update a job with its observed value.

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'Delaunay.X' and
            'Delaunay.y' corresponding to the job being updated
        """
        return self._update(uid, y_true)

    def pick(self):
        """
        Pick the next feature location for the next observation to be taken.

        This uses the recursive Delaunay subdivision algorithm.

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job
        """
        n = len(self.X)

        # -- note that we are assuming the points in X are not reordered by
        # the scipy Delaunay implementation

        n_corners = 2 ** self.dims
        if n < n_corners + 1:

            # Bootstrap with a regular sampling strategy to get it started
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = [0.]
        else:

            X = self.X()  # calling returns the value as an array
            y = self.y()
            virtual = self.virtual_flag()

            # Otherwise, recursive subdivide the edges with the Delaunay model
            if not self.triangulation:
                self.triangulation = ScipyDelaunay(X, incremental=True)

            # Weight by hyper-volume
            simplices = [tuple(s) for s in self.triangulation.vertices]
            cache = self.simplex_cache

            def get_value(s):

                # Computes the sample value as:
                #   hyper-volume of simplex * variance of values in simplex
                ind = list(s)
                value = (np.var(y[ind]) + self.explore_priority) * \
                    np.linalg.det((X[ind] - X[ind[0]])[1:])
                if not np.max(virtual[ind]):
                    cache[s] = value
                return value

            # Mostly the simplices won't change from call to call - cache!
            sample_value = [cache[s] if s in cache else get_value(s)
                            for s in simplices]

            # alternatively, a nicely vectorised computation might work here
            # profile and check what the bottleneck is

            # Extract the points in the highest value simplex
            simplex_indices = list(simplices[np.argmax(sample_value)])
            simplex = X[simplex_indices]
            simplex_v = y[simplex_indices]

            # Weight the position in this simplex based on value deviation
            eps = 1e-3
            weight = eps + np.abs(simplex_v - np.mean(simplex_v))
            weight /= np.sum(weight)
            xq = np.sum(weight * simplex, axis=0)  # dot
            yq_exp = np.sum(weight * simplex_v, axis=0)

            self.triangulation.add_points(xq[np.newaxis, :])  # incremental

        uid = Sampler._assign(self, xq, yq_exp)
        return xq, uid
