"""
Active Sampling module

Provides the Active Sampler Classes which contains strategies for
active sampling a spatial field
"""
import numpy as np
from scipy.spatial import Delaunay as ScipyDelaunay
import dora.regressors.gp as gp
import scipy.stats as stats
import hashlib


class Sampler:
    """
    Sampler Class

    Provides a basic template and interface to specific Sampler subclasses

    Attributes
    ----------
    lower : numpy.ndarray
        Lower bounds for each parameter in the parameter space
    upper : numpy.ndarray
        Upper bounds for each parameter in the parameter space
    dims : int
        Dimension of the parameter space (number of parameters)
    X : list
        List of feature vectors representing observed locations in the
        parameter space
    y : list
        List of target outputs or expected (virtual) target outputs
        corresponding to the feature vectors 'X'
    virtual_flag : list
        A list of boolean flags indicating the virtual elements of 'y'
            True: Corresponding target output is virtual
            False: Corresponding target output is observed
    pending_indices : dict
        A dictionary that maps the job ID to the corresponding index in both
        'X' and 'y'
    """

    def __init__(self, lower, upper):
        """
        Initialises the Sampler class

        .. note:: Currently only supports rectangular type restrictions on the
        parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        """
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.dims = self.upper.shape[0]
        assert self.lower.shape[0] == self.dims
        self.X = []
        self.y = []
        self.virtual_flag = []
        self.pending_indices = {}

    def pick(self):
        """
        Picks the next location in parameter space for the next observation
        to be taken

        .. note:: Currently a dummy function whose functionality will be
        filled by subclasses of the Sampler class

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job

        Raises
        ------
        AssertionError
            Under all circumstances. See note above.
        """
        assert False

    def update(self, uid, y_true):
        """
        Updates a job with its observed value

        .. note:: Currently a dummy function whose functionality will be
        filled by subclasses of the Sampler class

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'Sampler.X' and
            'Sampler.y' corresponding to the job being updated

        Raises
        ------
        AssertionError
            Under all circumstances. See note above.
        """
        assert False

    def _assign(self, xq, yq_exp):
        """
        Assigns a pair of picked location in parameter space and virtual
        targets a job ID

        Parameters
        ----------
        xq : numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        yq_exp : float
            The virtual target output at that parameter location

        Returns
        -------
        str
            A random hexadecimal ID to identify the corresponding job
        """
        # Place a virtual observation onto the collected data
        n = len(self.X)
        self.X.append(xq)
        self.y.append(yq_exp)
        self.virtual_flag.append(True)

        # Create an uid for this observation
        m = hashlib.md5()
        m.update(np.array(np.random.random()))
        uid = m.hexdigest()

        # Note the index of corresponding to this picked location
        self.pending_indices[uid] = n

        return uid

    def _update(self, uid, y_true):
        """
        Updates a job with its observed value

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'Sampler.X' and
            'Sampler.y' corresponding to the job being updated
        """
        # Make sure the job uid given is valid
        if uid not in self.pending_indices:
            raise ValueError('Result was not pending!')
        assert uid in self.pending_indices

        # Kill the job and update collected data with true observation
        ind = self.pending_indices.pop(uid)
        self.y[ind] = y_true
        self.virtual_flag[ind] = False

        return ind


class Delaunay(Sampler):
    """
    Delaunay Class

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
    def __init__(self, lower, upper, explore_priority=0.0001):
        """
        Initialises the Delaunay class

        .. note:: Currently only supports rectangular type restrictions on the
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
        Updates a job with its observed value

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
        return self._update(self, uid, y_true)

    def pick(self):
        """
        Picks the next location in parameter space for the next observation
        to be taken, using the recursive Delaunay subdivision algorithm

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job
        """
        n = len(self.X)
        n_corners = 2 ** self.dims
        if n < n_corners + 1:

            # Bootstrap with a regular sampling strategy to get it started
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = 0.
        else:

            # Otherwise, recursive subdivide the edges with the Delaunay model
            if not self.triangulation:
                self.triangulation = ScipyDelaunay(self.X, incremental=True)

            points = self.triangulation.points
            yvals = np.asarray(self.y)
            virtual = np.asarray(self.virtual_flag)

            # Weight by hyper-volume
            simplices = [tuple(s) for s in self.triangulation.vertices]
            cache = self.simplex_cache

            def get_value(s):

                # Computes the sample value as:
                #   hyper-volume of simplex * variance of values in simplex
                ind = list(s)
                value = (np.var(yvals[ind]) + self.explore_priority) * \
                    np.linalg.det((points[ind] - points[ind[0]])[1:])
                if not np.max(virtual[ind]):
                    cache[s] = value
                return value

            # Mostly the simplices won't change from call to call - cache!
            sample_value = [cache[s] if s in cache else get_value(s)
                            for s in simplices]

            # Find the points in the highest value simplex
            simplex_indices = list(simplices[np.argmax(sample_value)])
            simplex = points[simplex_indices]
            simplex_v = yvals[simplex_indices]

            # Weight based on deviation from the mean
            weight = 1e-3 + np.abs(simplex_v - np.mean(simplex_v))
            weight /= np.sum(weight)
            xq = weight.dot(simplex)
            yq_exp = weight.dot(simplex_v)
            self.triangulation.add_points(xq[np.newaxis, :])  # incremental

        uid = Sampler._assign(self, xq, yq_exp)
        return xq, uid


class GaussianProcess(Sampler):
    """
    GaussianProcess Class

    Inherits from the Sampler class and augments pick and update with the
    mechanics of the GP method

    Attributes
    ----------
    hyper_params : numpy.ndarray
        The hyperparameters of the Gaussian Process Inference Model
    regressor : dict
        Cached values of simplices for Delaunay triangulation
    explore_priority : float
        The priority of exploration against exploitation
    kernel : function
        The learned kernel covariance function of the Gaussian process
    print_kernel : function
        A convenient print function for displaying the learned kernel
    explore_priority : float
        The priority of exploration against exploitation

    See Also
    --------
    Sampler : Base Class
    """
    def __init__(self, lower, upper, X, y,
                 kerneldef=None, add_train_data=True, explore_priority=0.01):
        """
        Initialises the GaussianProcess class

        .. note:: Currently only supports rectangular type restrictions on the
        parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        X : numpy.ndarray
            Training features for the Gaussian process model
        y : numpy.ndarray
            Training targets for the Gaussian process model
        kerneldef : function, optional
            Kernel covariance definition
        add_train_data : boolean
            Whether to add training data to the sampler or not
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)
        self.hyper_params = None
        self.regressor = None
        self.kernel = None
        self.print_kernel = None
        self.explore_priority = explore_priority
        self._train(X, y,
                    kerneldef=kerneldef, add_train_data=add_train_data)

    def _train(self, X, y,
               kerneldef=None, add_train_data=True):
        """
        Trains the Gaussian process used for the sampler

        Parameters
        ----------
        X : numpy.ndarray
            Training features for the Gaussian process model
        y : numpy.ndarray
            Training targets for the Gaussian process model
        kerneldef : function, optional
            Kernel covariance definition
        add_train_data : boolean
            Whether to add training data to the sampler or not
        """
        # If 'kerneldef' is not provided, define a default 'kerneldef'
        if kerneldef is None:
            kerneldef = lambda h, k: (h(1e-3, 1e2, 1) *
                                      k('matern3on2', h(1e-2, 1e3, 1)))
        # Set up optimisation
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(kerneldef)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 50.0
        opt_config.global_opt = False

        # Prepare Kernel Covariance
        self.kernel = gp.compose(kerneldef)
        self.print_kernel = gp.describer(kerneldef)

        # Learn the GP
        self.hyper_params = gp.learn(X, y, self.kernel, opt_config)

        # Adds sampled data to the model
        if add_train_data:
            self.X = X.copy()
            self.y = y.copy()
            self.virtual_flag = [False for y in y]
            self.regressor = gp.condition(np.asarray(self.X),
                                          np.asarray(self.y),
                                          self.kernel, self.hyper_params)

    def update(self, uid, y_true):
        """
        Updates a job with its observed value

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'GaussianProcess.X' and
            'GaussianProcess.y' corresponding to the job being updated
        """
        ind = self._update(uid, y_true)
        if self.regressor:
            self.regressor.y[ind] = y_true
            self.regressor.alpha = gp.predict.alpha(self.regressor.y,
                                                    self.regressor.L)
        return ind

    def pick(self, n_test=500, acq_fn='sigmoid'):
        """
        Picks the next location in parameter space for the next observation
        to be taken, with a Gaussian process model

        Parameters
        ----------
        n_test : int, optional
            The number of random query points across the search space to pick
            from
        acq_fn : str, optional
            The type of acquisition function used

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job
        """
        n = len(self.X)
        n_corners = 2 ** self.dims
        if n < n_corners + 1:

            # Bootstrap with a regular sampling strategy to get it started
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = 0.

        else:

            # Randomly sample the volume.
            X_test = random_sample(self.lower, self.upper, n_test)
            query = gp.query(X_test, self.regressor)
            post_mu = gp.mean(self.regressor, query)
            post_var = gp.variance(self.regressor, query)

            acq_func_dict = {
                'maxvar': lambda u, v: np.argmax(v, axis=0),
                'predmax': lambda u, v: np.argmax(u + np.sqrt(v), axis=0),
                'entropyvar': lambda u, v:
                    np.argmax((self.explore_priority + np.sqrt(v)) *
                              u * (1 - u), axis=0),
                'sigmoid': lambda u, v:
                    np.argmax(np.abs(stats.logistic.cdf(u + np.sqrt(v),
                              loc=0.5, scale=self.explore_priority) -
                              stats.logistic.cdf(u - np.sqrt(v),
                              loc=0.5, scale=self.explore_priority)), axis=0)
            }

            iq = acq_func_dict[acq_fn](post_mu, post_var)
            xq = X_test[iq, :]
            yq_exp = post_mu[iq]

        uid = Sampler._assign(self, xq, yq_exp)

        if self.regressor:
            gp.add_data(np.asarray(xq[np.newaxis, :]),
                        np.asarray(yq_exp)[np.newaxis],
                        self.regressor)
        else:
            self.regressor = gp.condition(np.asarray(self.X),
                                          np.asarray(self.y), self.kernel,
                                          self.hyper_params)

        return xq, uid

    def predict(self, Xq):
        """
        Infers the mean and variance of the Gaussian process at given locations
        using the data collected so far

        Parameters
        ----------
        Xq : Query points

        Returns
        -------
        numpy.ndarray
            Expectance of the prediction at the given locations
        numpy.ndarray
            Variance of the prediction at the given locations
        """
        real_flag = ~np.asarray(self.virtual_flag)
        X_real = np.asarray(self.X)[real_flag]
        y_real = np.asarray(self.y)[real_flag]
        y_mean = y_real.mean()

        regressor = gp.condition(X_real, y_real - y_mean, self.kernel,
                                 self.hyper_params)
        predictor = gp.query(Xq, regressor)
        yq_exp = gp.mean(regressor, predictor) + y_mean
        yq_var = gp.variance(regressor, predictor)

        return yq_exp, yq_var


# NOTE: StackedGaussianProcess is to be merged with GaussianProcess!
class StackedGaussianProcess(Sampler):
    """
    GaussianProcess Class

    Inherits from the Sampler class and augments pick and update with the
    mechanics of the GP method

    Attributes
    ----------
    n_stacks : int
        The number of Gaussian process 'stacks', which is also the
        dimensionality of the target output
    hyper_params : numpy.ndarray
        The hyperparameters of the Gaussian Process Inference Model
    regressors : list
        List of regressor objects. See 'gp.types.RegressionParams'
    mean : float
        Mean of the training target outputs
    trained_flag : bool
        Whether the GP model have been trained or not
    acq_func : str
        A string specifying the type of acquisition function used
    explore_priority : float
        The priority of exploration against exploitation
    n_train_threshold : int
        Number of training samples required before sampler can be trained

    See Also
    --------
    Sampler : Base Class
    """
    def __init__(self, lower, upper, X=None, y=None, n_stacks=None,
                 add_train_data=True, hypers=None, n_train_threshold=None,
                 mean=0, acq_func='maxvar', explore_priority=0.3):
        """
        Initialises the GaussianProcess class

        .. note:: Currently only supports rectangular type restrictions on the
        parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        X : numpy.ndarray
            Training features for the Gaussian process model
        y : numpy.ndarray
            Training targets for the Gaussian process model
        n_stacks : int
            The number of Gaussian process 'stacks', which is also the
            dimensionality of the target output
        add_train_data : boolean
            Whether to add training data to the sampler or not
        hypers : tuple
            Hyperparameters of the Gaussian process
        n_train_threshold : int
            Number of training samples required before sampler can be trained
        mean : float
            Mean of the training target outputs
        acq_func : str
            A string specifying the type of acquisition function used
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)
        self.n_stacks = n_stacks
        self.hyper_params = []
        self.regressors = None
        self.mean = mean
        self.trained_flag = False
        self.acq_func = acq_func
        self.explore_priority = explore_priority

        if n_train_threshold is not None:
            self.n_train_threshold = n_train_threshold
        else:
            self.n_train_threshold = 7 ** len(lower)

        # Train the hyperparameters if there are
        # sufficient training points provided
        if X is not None:
            assert y.shape[0] == X.shape[0]
            if X.shape[0] >= self.n_train_threshold:
                self.train_data(X, y, hypers)

        if add_train_data and X is not None:
            assert y.shape[0] == X.shape[0]
            # Convert to lists
            self.X = [x_i for x_i in X]
            self.y = [y_i for y_i in y]
            self.virtual_flag = [False for x in X]

            if self.trained_flag:
                self.regressors = []
                for ind in range(n_stacks):
                    self.regressors.append(
                        gp.condition(np.asarray(self.X),
                                     np.asarray(self.y)[:, ind] - self.mean,
                                     self.kernel, self.hyper_params[ind]))

    def train_data(self, X, y, hypers=None):
        """
        Trains the Gaussian process used for the sampler

        Parameters
        ----------
        X : numpy.ndarray
            Training features for the Gaussian process model
        y : numpy.ndarray
            Training targets for the Gaussian process model
        hypers : tuple
            Hyperparameters of the Gaussian process
        """
        # Set up the GP training and kernel
        self.mean = np.mean(y)
        min_l = 1e-2 * np.ones(2)
        max_l = 1e3 * np.ones(2)
        init_l = np.array([0.5, 0.5])
        kerneldef = lambda h, k: \
            h(1e-3, 1e2, 2.321) * k('matern3on2', h(min_l, max_l, init_l))
        self.kernel = gp.compose(kerneldef)
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(kerneldef)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 50.0
        opt_config.global_opt = False

        # We need to train a regressor for each of the stacks
        # Let's use a common length scale by using folds
        if hypers is None:

            folds = gp.Folds(self.n_stacks, [], [], [])

            for stack in range(self.n_stacks):
                folds.X.append(X)
                folds.flat_y.append(y[:, stack] - self.mean)

            hypers = gp.learn_folds(folds, self.kernel, opt_config)

        for stack in range(self.n_stacks):
            self.hyper_params.append(hypers)

        self.trained_flag = True

    def update(self, uid, y_true):
        """
        Updates a job with its observed value

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
        ind = self._update(uid, y_true)
        if self.trained_flag:
            full_y = np.asarray(self.y)
            for i, regressor in enumerate(self.regressors):
                regressor.y = full_y[:, i] - self.mean
                regressor.alpha = gp.predict.alpha(regressor.y, regressor.L)
        return ind

    def pick(self, n_test=500):
        """
        Picks the next location in parameter space for the next observation
        to be taken, with a Gaussian process model

        Parameters
        ----------
        n_test : int, optional
            The number of random query points across the search space to pick
            from

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job
        """
        n = len(self.X)
        n_corners = 2 ** self.dims

        if not self.trained_flag:
            xq = random_sample(self.lower, self.upper, 1)[0, :]
            yq_exp = 0. * np.ones(self.n_stacks) + self.mean

        elif n < n_corners + 1:
            # Bootstrap with a regular sampling strategy to get it started
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = 0. * np.ones(self.n_stacks) + self.mean

        else:
            # Randomly sample the volume.
            X_test = random_sample(self.lower, self.upper, n_test)
            query_list = [gp.query(X_test, reg) for reg in self.regressors]
            post_mu = np.asarray([gp.mean(reg, qry) for reg, qry in
                                 zip(self.regressors, query_list)]) + self.mean
            post_var = np.asarray([gp.variance(reg, qry) for reg, qry in
                                  zip(self.regressors, query_list)])

            # Aquisition Functions
            explore_priority = 0.3
            acq_func_dict = {
                'maxvar': lambda u, v: np.argmax(np.sum(v, axis=0)),
                'predmax': lambda u, v: np.argmax(np.max(u + 3 * np.sqrt(v),
                                                         axis=0)),
                'prodmax': lambda u, v: np.argmax(np.max((u + (self.mean +
                                                  explore_priority / 3.0)) *
                                                  np.sqrt(v), axis=0)),
                'probGreaterThan':
                    lambda u, v: np.argmax(np.max((1 - stats.norm.cdf(
                                           explore_priority *
                                           np.ones(u.shape), u,
                                           np.sqrt(v))), axis=0))
            }

            iq = acq_func_dict[self.acq_func](post_mu, post_var)
            xq = X_test[iq, :]
            yq_exp = post_mu[:, iq]

        # Place a virtual observation...
        uid = Sampler._assign(self, xq, yq_exp)

        if not self.trained_flag and np.sum([not i for i in self.virtual_flag]) \
                >= self.n_train_threshold:
            real_flag = [not i for i in self.virtual_flag]
            X_real = [x for x, real in zip(self.X, real_flag) if real is True]
            y_real = [y for y, real in zip(self.y, real_flag) if real is True]
            self.train_data(np.asarray(X_real), np.asarray(y_real))

        # if we are still grid sampling and havent initialised the
        # regressors... then create them
        if self.trained_flag:
            if self.regressors is None:
                self.regressors = []  # init a list of regressors
                arr_X = np.asarray(self.X)
                arr_Y = np.asarray(self.y)
                for ind in range(self.n_stacks):
                    self.regressors.append(
                        gp.condition(arr_X, arr_Y[:, ind] - self.mean,
                                     self.kernel, self.hyper_params[ind]))
            else:
                for ind in range(self.n_stacks):
                    gp.add_data(np.asarray(xq[np.newaxis, :]),
                                np.asarray(yq_exp[ind])[np.newaxis] -
                                self.mean, self.regressors[ind])

        return xq, uid


    def predict(self, Xq):
        """
        Infers the mean and variance of the Gaussian process at given locations
        using the data collected so far

        Parameters
        ----------
        Xq : Query points

        Returns
        -------
        numpy.ndarray
            Expectance of the prediction at the given locations
        numpy.ndarray
            Variance of the prediction at the given locations
        """
        # extract only the real observations for conditioning the predictor
        # TODO Consider moving y_real inside of the for loop use regressor.y

        assert self.trained_flag, "Sampler is not trained yet. " \
                                  "Possibly not enough observations provided."

        real_flag = [not i for i in self.virtual_flag]
        X_real = [x for x, real in zip(self.X, real_flag) if real is True]
        y_real = [y for y, real in zip(self.y, real_flag) if real is True]
        X_real = np.asarray(X_real)
        y_real = np.asarray(y_real) - self.mean

        post_mu = []
        post_var = []

        for ind in range(self.n_stacks):
            regressor = gp.condition(X_real, y_real[:, ind], self.kernel,
                                     self.hyper_params[ind])
            query_object = gp.query(Xq, regressor)
            post_mu.append(gp.mean(regressor, query_object))
            post_var.append(gp.variance(regressor, query_object))

        return np.asarray(post_mu).T + self.mean, np.asarray(post_var).T


def grid_sample(lower, upper, n):
    """
    Used to seed an algorithm with a regular pattern of the corners and
    the centre. Provide search parameters and the index.

    Parameters
    ----------
    lower : array_like
        Lower or minimum bounds for the parameter space
    upper : array_like
        Upper or maximum bounds for the parameter space
    n : int
        Index of location

    Returns
    -------
    np.ndarray
        Sampled location in feature space
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    n_dims = lower.shape[0]
    n_corners = 2 ** n_dims
    if n < n_corners:
        xq = lower + (upper - lower) * \
            (n & 2 ** np.arange(n_dims) > 0).astype(float)
    elif n == n_corners:
        xq = lower + 0.5 * (upper - lower)
    else:
        assert(False)
    return xq


def random_sample(lower, upper, n):
    """
    Used to randomly sample the search space.
    Provide search parameters and the number of samples desired.

    Parameters
    ----------
    lower : array_like
        Lower or minimum bounds for the parameter space
    upper : array_like
        Upper or maximum bounds for the parameter space
    n : int
        Number of samples

    Returns
    -------
    np.ndarray
        Sampled location in feature space
    """
    n_dims = len(lower)
    X = np.random.random((n, n_dims))
    volume_range = [upper[i] - lower[i] for i in range(n_dims)]
    X_scaled = X * volume_range
    X_shifted = X_scaled + lower
    return X_shifted
