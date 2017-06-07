from GPflow.gpr import GPR
from GPflow.mean_functions import Zero
from GPflow.param import DataHolder
from GPflow.param import AutoFlow

import tensorflow as tf
import numpy as np


class GPRCached(GPR):
    """ GPflow.gpr.GPR class that stores Cholesky decomposition for efficiency
        and performs single row Cholesky update and downdate computations.

        Caching based on https://github.com/GPflow/GPflow/issues/333
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
        """ Disallow setting `X` and `Y` directly, so that cached
            computations remain in sync."""
        if key in ('X', 'Y') and hasattr(self, key):
            raise ValueError('Changes to X and Y should be made through calls '
                             'to `set_data_points(X, Y)`')

        GPR.__setattr__(self, key, value)

    def set_parameter_dict(self, d):
        """ Update cache when parameters are reset. """
        GPR.set_parameter_dict(self, d)
        self.update_cache()

    def set_state(self, x):
        """ Update cache when parameters are reset.

            `set_state` is called during `optimize`.
        """
        GPR.set_state(self, x)
        self.update_cache()

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

    @AutoFlow((tf.float64, [None, None]))
    def _cholesky_update(self, x):
        """ Perform incremental update of Cholesky decomposition by adding
            data point `x`.
        """
        kxn = self.kern.K(self.X, x)
        knn = (self.kern.K(x, x)
              + tf.eye(tf.shape(x)[0], dtype=np.float64)
              * self.likelihood.variance)

        L = self.cholesky
        c = tf.matrix_triangular_solve(L, kxn, lower=True)
        d = tf.cholesky(knn - tf.matmul(tf.transpose(c), c))

        cholesky = tf.concat([
            tf.concat([L, tf.zeros(tf.shape(c), dtype=tf.float64)], axis=1),
            tf.concat([tf.transpose(c), d], axis=1)
        ], axis=0, name='gp_cholesky_update')

        return cholesky

    @AutoFlow((tf.int32,))
    def _cholesky_downdate(self, i):
        """ Perform downdate of Cholesky decomposition by removing a single
            data point at index `i`.
        """
        L = self.cholesky
        n = tf.shape(L)[0]
        m = n - i - 1

        Sa = tf.slice(L, begin=[i+1, i], size=[m, 1])
        Sb = tf.slice(L, begin=[i+1, i+1], size=[m, m])
        R = tf.cholesky(tf.add(
                tf.matmul(Sa, tf.transpose(Sa)),
                tf.matmul(Sb, tf.transpose(Sb))
        ))

        left = tf.concat([
            tf.slice(L, begin=[0, 0], size=[i, i]),
            tf.slice(L, begin=[i+1, 0], size=[m, i]),
            ], axis=0)
        right = tf.concat([tf.zeros([i, m], dtype=tf.float64), R], axis=0)

        cholesky = tf.concat([left, right], axis=1, name='gp_cholesky_downdate')

        return cholesky

    @AutoFlow()
    def _alpha_update(self):
        """ Compute alpha (use after `self.cholesky` has been updated). """
        target = self.Y - self.mean_function(self.X)
        alpha = tf.matrix_triangular_solve(self.cholesky, target,
                                           name='gp_alpha_update')
        return alpha

    def set_data_points(self, X, Y):
        """ Reset the data points to arrays `X` and `Y`. Update cache. """
        assert X.shape == Y.shape
        GPR.__setattr__(self, 'X', X)
        GPR.__setattr__(self, 'Y', Y)
        self.update_cache()

    def add_data_point(self, x, y):
        """ Add data point (`x`, `y`) to GP and perform update of Cholesky
            decomposition.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        self.cholesky = self._cholesky_update(x)
        self.X.set_data(np.append(self.X.value, x, axis=0))
        self.Y.set_data(np.append(self.Y.value, y, axis=0))
        self.alpha = self._alpha_update()

    def remove_data_point(self, index):
        """ Remove point at `index` from both X and Y, and downdate Cholesky
            decomposition.
        """
        if not 0 <= index < self.X.shape[0]:
            raise IndexError('Index {} is out of range.'.format(index))

        self.cholesky = self._cholesky_downdate(index)
        self.X.set_data(np.delete(self.X.value, index, axis=0))
        self.Y.set_data(np.delete(self.Y.value, index, axis=0))
        self.alpha = self._alpha_update()

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
