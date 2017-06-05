"""
https://github.com/GPflow/GPflow/issues/333
"""

from GPflow.gpr import GPR
from GPflow.mean_functions import Zero
from GPflow.param import DataHolder
from GPflow.param import AutoFlow

import tensorflow as tf
import numpy as np


class GPRCached(GPR):
    """
    GPflow.gpr.GPR class that stores cholesky decomposition for efficiency.

    Parameters
    ----------
    x : ndarray
        A 2d array with states to initialize the GP model. Each state is on
        a row.
    y : ndarray
        A 2d array with measurements to initialize the GP model. Each
        measurement is on a row.

    """

    def __init__(self, x, y, kern):
        """Initialize GP and cholesky decomposition."""
        GPR.__init__(self, x, y, kern)

        # Create new dataholders for the cached data
        self.cholesky = DataHolder(np.empty((0, 0), dtype=np.float64),
                                   on_shape_change='pass')
        self.alpha = DataHolder(np.empty((0, 0), dtype=np.float64),
                                on_shape_change='pass')
        self.update_cache()

    @AutoFlow()
    def _compute_cache(self):
        """Compute cache."""
        kernel = (self.kern.K(self.X) +
                  tf.eye(tf.shape(self.X)[0], dtype=np.float64) *
                  self.likelihood.variance)

        cholesky = tf.cholesky(kernel, name='gp_cholesky')

        target = self.Y - self.mean_function(self.X)
        alpha = tf.matrix_triangular_solve(cholesky, target, name='gp_alpha')
        return cholesky, alpha

    def update_cache(self):
        """Update the cache after adding data points."""
        self.cholesky, self.alpha = self._compute_cache()

    def build_predict(self, Xnew, full_cov=False):
        """Predict mean and variance of the GP at locations in Xnew.

        Parameters
        ----------
        Xnew : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        full_cov : bool
            if False returns only the diagonal of the covariance matrix

        Returns
        -------
        mean : ndarray
            The expected function values at the points.
        error_bounds : ndarray
            Diagonal of the covariance matrix (or full matrix).

        """
        Kx = self.kern.K(self.X, Xnew)
        A = tf.matrix_triangular_solve(self.cholesky, Kx, lower=True)
        fmean = (tf.matmul(tf.transpose(A), self.alpha)
                 + self.mean_function(Xnew))
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)),
                           [1, tf.shape(self.Y)[1]])
        return fmean, fvar


class GPRCached2(GPR):
    """
    GPflow.gpr.GPR class that stores cholesky decomposition for efficiency.

    Parameters
    ----------
    x : ndarray
        A 2d array with states to initialize the GP model. Each state is on
        a row.
    y : ndarray
        A 2d array with measurements to initialize the GP model. Each
        measurement is on a row.

    """

    def __init__(self, x, y, kern, mean_function=Zero(), name='name'):
        """Initialize GP and Cholesky decomposition."""
        GPR.__init__(self, x, y, kern=kern, mean_function=mean_function,
                     name=name)

        # Create new dataholders for the cached data
        self.cholesky = DataHolder(np.empty((0, 0), dtype=np.float64),
                                   on_shape_change='pass')
        self.alpha = DataHolder(np.empty((0, 0), dtype=np.float64),
                                on_shape_change='pass')
        self.update_cache()

    def __setattr__(self, key, value):
        """Catch changes to X or Y and invalidate cache."""
        if key in ('X', 'Y') and hasattr(self, key):
            raise ValueError('Changes to X and Y should be made through calls '
                             'to `set_data_points(X, Y)`')

        GPR.__setattr__(self, key, value)

    def optimize(self, method='L-BFGS-B', tol=None, callback=None,
                 maxiter=1000, **kw):
        """Invalidate cache after optimizing."""
        r = GPR.optimize(self, method=method, tol=tol, callback=callback,
                         maxiter=maxiter, **kw)
        self.update_cache()
        return r

    @AutoFlow()
    def _compute_cache(self):
        """Compute cache."""
        kernel = (self.kern.K(self.X)
                  + tf.eye(tf.shape(self.X)[0], dtype=np.float64)
                  * self.likelihood.variance)

        cholesky = tf.cholesky(kernel, name='gp_cholesky')

        target = self.Y - self.mean_function(self.X)
        alpha = tf.matrix_triangular_solve(cholesky, target, name='gp_alpha')
        return cholesky, alpha

    def update_cache(self):
        """Update the cache after adding data points."""
        self.cholesky, self.alpha = self._compute_cache()

    @AutoFlow()
    def _cholesky_update(self):
        """Perform rank-one update of Cholesky decomposition."""
        kernel = (self.kern.K(self.X)
                  + tf.eye(tf.shape(self.X)[0], dtype=np.float64)
                  * self.likelihood.variance)

        r = tf.shape(kernel)[0]
        a = tf.slice(kernel, begin=[0, r - 1], size=[r - 1, 1])

        L = self.cholesky
        c = tf.matmul(tf.matrix_inverse(L), a)
        d = tf.sqrt(kernel[-1, -1] - tf.matmul(tf.transpose(c), c))

        cholesky = tf.concat([
            tf.concat([L, tf.zeros([r - 1, 1], dtype=tf.float64)], axis=1),
            tf.concat([tf.transpose(c), d], axis=1)
        ], axis=0, name='gp_cholesky_update')

        target = self.Y - self.mean_function(self.X)
        alpha = tf.matrix_triangular_solve(cholesky, target,
                                           name='gp_alpha_update')
        return cholesky, alpha

    def set_data_points(self, X, Y):
        """ Reset the data points and update cache. """
        assert X.shape == Y.shape
        GPR.__setattr__(self, 'X', X)
        GPR.__setattr__(self, 'Y', Y)
        self.update_cache()

    def add_data_point(self, x, y):
        """ Add data point to GP and perform rank-one update of Cholesky
            decomposition.
        """
        # add x, y to DataHolders
        self.X.set_data(np.append(self.X.value, np.atleast_2d(x), axis=0))
        self.Y.set_data(np.append(self.Y.value, np.atleast_2d(y), axis=0))

        self.cholesky, self.alpha = self._cholesky_update()

    def build_predict(self, Xnew, full_cov=False):
        """Predict mean and variance of the GP at locations in Xnew.

        Parameters
        ----------
        Xnew : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        full_cov : bool
            if False returns only the diagonal of the covariance matrix

        Returns
        -------
        mean : ndarray
            The expected function values at the points.
        error_bounds : ndarray
            Diagonal of the covariance matrix (or full matrix).

        """
        Kx = self.kern.K(self.X, Xnew)
        A = tf.matrix_triangular_solve(self.cholesky, Kx, lower=True)
        fmean = (tf.matmul(tf.transpose(A), self.alpha)
                 + self.mean_function(Xnew))
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar
