""" Active Sampling module
Strategies for active sampling a spatial field.
"""
import numpy as np
from scipy.spatial import Delaunay as ScipyDelaunay
import dora.regressors.gp as gp
import scipy.stats as stats
import hashlib


class BaseSampler:
    """
    BaseSampler Class

    Provides a basic template and interface to a Sampler class

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
        Initialises the BaseSampler class

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
        filled by subclasses of the BaseSampler class

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
        filled by subclasses of the BaseSampler class

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data lists 'BaseSampler.X' and
            'BaseSampler.y' corresponding to the job being updated

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
            Index location in the data lists 'BaseSampler.X' and
            'BaseSampler.y' corresponding to the job being updated
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


class DelaunaySampler(BaseSampler):
    """
    DelaunaySampler Class

    Inherits from the BaseSampler class and augments pick and update with the
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
    BaseSampler : Base Class
    """
    def __init__(self, lower, upper, explore_priority=0.0001):
        """
        Initialises the DelaunaySampler class

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
        BaseSampler.__init__(self, lower, upper)
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
            Index location in the data lists 'DelaunaySampler.X' and
            'DelaunaySampler.y' corresponding to the job being updated
        """
        BaseSampler._update(self, uid, y_true)

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

        uid = BaseSampler._assign(self, xq, yq_exp)
        return xq, uid


class GaussianProcessSampler(BaseSampler):
    """
    GaussianProcessSampler Class

    Inherits from the BaseSampler class and augments pick and update with the
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
    BaseSampler : Base Class
    """
    def __init__(self, lower, upper, X_train, y_train,
                 kerneldef=None, add_train_data=True, explore_priority=0.01):
        """
        Initialises the GaussianProcessSampler class

        .. note:: Currently only supports rectangular type restrictions on the
        parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        X_train : numpy.ndarray
            Training features for the Gaussian process model
        y_train : numpy.ndarray
            Training targets for the Gaussian process model
        kerneldef : function, optional
            Kernel covariance definition
        add_train_data : boolean
            Whether to add training data to the sampler or not
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        BaseSampler.__init__(self, lower, upper)
        self.hyper_params = None
        self.regressor = None
        self.kernel = None
        self.print_kernel = None
        self.explore_priority = explore_priority
        self._train(X_train, y_train,
                    kerneldef=kerneldef, add_train_data=add_train_data)

    def _train(self, X_train, y_train,
               kerneldef=None, add_train_data=True):
        """
        Trains the Gaussian process used for the sampler

        Parameters
        ----------
        X_train : numpy.ndarray
            Training features for the Gaussian process model
        y_train : numpy.ndarray
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
        self.hyper_params = gp.learn(X_train, y_train, self.kernel, opt_config)

        # Adds sampled data to the model
        if add_train_data:
            self.X = X_train.copy()
            self.y = y_train.copy()
            self.virtual_flag = [False for y in y_train]
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
            Index location in the data lists 'GaussianProcessSampler.X' and
            'GaussianProcessSampler.y' corresponding to the job being updated
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

        uid = BaseSampler._assign(self, xq, yq_exp)

        if self.regressor:
            gp.add_data(np.asarray(xq[np.newaxis, :]),
                        np.asarray(yq_exp)[np.newaxis],
                        self.regressor)
        else:
            self.regressor = gp.condition(np.asarray(self.X),
                                          np.asarray(self.y), self.kernel,
                                          self.hyper_params)

        return xq, uid

    # def predict(self, Xq):

    #     # extract only the real observations for conditioning the predictor
    #     # TODO Consider moving real_y inside of the for loop use regressor.y
    #     real_id = [not i for i in self.virtual_flag]
    #     real_X = [x for x, real in zip(self.X, real_id) if real is True]
    #     real_y = [y for y, real in zip(self.y, real_id) if real is True]
    #     real_X = np.asarray(real_X)
    #     real_y = np.asarray(real_y) - self.mean # <- where is self.mean?

    #     regressor = gp.condition(real_X, real_y, self.kernel,
    #                              self.hyper_params)
    #     predictor = gp.query(Xq, regressor)
    #     post_mu = gp.mean(regressor, predictor)
    #     post_var = gp.variance(regressor, predictor)

    #     return post_mu + self.mean, post_var


class StackedGaussianProcessSampler(BaseSampler):
    """
    Inherits from the BaseSampler class and augments pick and update with the
    mechanics of the GP method
    """
    def __init__(self, lower, upper, X_train=None, y_train=None, n_stacks=None,
                 add_train_data=True, hypers=None, n_train_threshold=None,
                 mean=0, acq_func='maxvar', explore_factor=0.3):
        """
        Arguments:
        lower (array floats) - min of bounding box
        upper (array floats) - max of bounding box
        """

        BaseSampler.__init__(self, lower, upper)
        self.n_stacks = n_stacks
        self.hyper_params = []
        self.regressors = None
        self.mean = mean
        self.trained_flag = False
        self.acq_func = acq_func
        self.explore_factor=explore_factor

        # sets the number of training samples needed to be observed
        # before the sampler is trained
        if n_train_threshold is not None:
            self.n_train_threshold = n_train_threshold
        else:
            self.n_train_threshold = 7**len(lower)

        # Train the hyperparameters if there are sufficient training points provided
        if X_train is not None:
            assert y_train.shape[0] == X_train.shape[0]
            if X_train.shape[0] >= self.n_train_threshold:
                self.train_data(X_train, y_train, hypers)

        if add_train_data and X_train is not None:
            assert y_train.shape[0] == X_train.shape[0]
            # convert to lists
            self.X = [x for x in X_train]
            self.y = [y for y in y_train]
            self.virtual_flag = [False for x in X_train]

            if self.trained_flag:
                self.regressors = []
                for ind in range(n_stacks):
                    self.regressors.append(
                        gp.condition(np.asarray(self.X), np.asarray(self.y)[:,ind]-self.mean,
                                     self.kernel, self.hyper_params[ind]))

    def train_data(self, X_train, y_train, hypers=None):
        """
        :param X_train: np.array (n_training_data x dimensions of feature space)
        :param y_train: np.array (n_training_data x dimensions of output space)
        :param hypers:
        :return:
        """
         # Set up the GP training and kernel
        self.mean = np.mean(y_train)
        minL = 1e-2*np.ones(2)
        maxL = 1e3*np.ones(2)
        initL = np.array([0.5, 0.5])
        kerneldef = lambda h, k: \
            h(1e-3, 1e2, 2.321) * k('matern3on2', h(minL, maxL, initL))
        self.kernel = gp.compose(kerneldef)
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(kerneldef)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 50.0
        opt_config.global_opt = False

        # We need to train a regressor for each of the stacks
        # Lets use a common length scale by using folds
        if hypers==None:
            folds = gp.Folds(self.n_stacks, [], [], [])
            # import ipdb; ipdb.set_trace()
            for stack in range(self.n_stacks):
                folds.X.append(X_train)
                # folds.Y.append(y_train[:, stack])
                folds.flat_y.append(y_train[:, stack]-self.mean)

            hypers = gp.train.learn_folds(folds, self.kernel, opt_config)

        for stack in range(self.n_stacks):
            self.hyper_params.append(hypers)

        self.trained_flag = True

    def update(self, uid, y_true):
        """ Applies an observation to a Gaussian process active sampling model
        """
        self._update(uid, y_true)
        if self.trained_flag:
            full_y = np.asarray(self.y)
            for i, regressor in enumerate(self.regressors):
                regressor.y = full_y[:,i]-self.mean
                regressor.alpha = gp.predict.alpha(regressor.y, regressor.L)

    def pick(self, n_test=500):
        n = len(self.X)
        n_corners = 2**self.dims

        if not self.trained_flag:
            xq = random_sample(self.lower, self.upper, 1)[0,:]
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
            explore_factor = 0.3
            acq_func_dict = {
                'maxvar': lambda u, v: np.argmax(np.sum(v, axis=0)),
                'predmax': lambda u, v: np.argmax(np.max(u + 3*np.sqrt(v),
                                                         axis=0)),
                'prodmax': lambda u, v: np.argmax(np.max((u + (self.mean+explore_factor/3.0)) * np.sqrt(v),
                                                         axis=0)),
                'probGreaterThan': lambda u, v: np.argmax(np.max((1-stats.norm.cdf
                                        (explore_factor * np.ones(u.shape),u, np.sqrt(v))),
                                                         axis=0))
            }

            iq = acq_func_dict[self.acq_func](post_mu, post_var)
            xq = X_test[iq, :]
            yq_exp = post_mu[:, iq]


        # Place a virtual observation...
        uid = BaseSampler._assign(self, xq, yq_exp)

        if not self.trained_flag and np.sum([not i for i in self.virtual_flag]) \
                >= self.n_train_threshold:
            real_id = [not i for i in self.virtual_flag]
            real_X = [x for x, real in zip(self.X, real_id) if real is True]
            real_y = [y for y, real in zip(self.y, real_id) if real is True]
            self.train_data(np.asarray(real_X), np.asarray(real_y))

        # if we are still grid sampling and havent initialised the
        # regressors... then create them
        if self.trained_flag:
            if self.regressors is None:
                self.regressors = []  # init a list of regressors
                arrX = np.asarray(self.X)
                arrY = np.asarray(self.y)
                for ind in range(self.n_stacks):
                    self.regressors.append(
                        gp.condition(arrX, arrY[:, ind]-self.mean,
                                     self.kernel, self.hyper_params[ind]))
            else:
                for ind in range(self.n_stacks):
                    gp.add_data(np.asarray(xq[np.newaxis, :]),
                                np.asarray(yq_exp[ind])[np.newaxis] - self.mean,
                                self.regressors[ind])

        return xq, uid

    def predict(self, Xq):
        """
        method to query the probabilistic model at locations
        :param Xq: n x d array of points in the region of interest
        :return: post_mu: n x n_stacks array of predicted mean values
                 post_var: n x n_stacks array of predicted variance values
        """

        # extract only the real observations for conditioning the predictor
        #TODO Consider moving real_y inside of the for loop use regressor.y

        assert self.trained_flag, "Sampler is not trained yet. " \
                                  "Possibly not enough observations provided."

        real_id = [not i for i in self.virtual_flag]
        real_X = [x for x, real in zip(self.X, real_id) if real is True]
        real_y = [y for y, real in zip(self.y, real_id) if real is True]
        real_X = np.asarray(real_X)
        real_y = np.asarray(real_y)-self.mean

        post_mu = []
        post_var = []

        for ind in range(self.n_stacks):
            regressor = gp.condition(real_X, real_y[:,ind], self.kernel,
                                     self.hyper_params[ind])
            query_object = gp.query(Xq,regressor)
            post_mu.append(gp.mean(regressor,query_object))
            post_var.append(gp.variance(regressor,query_object))

        return np.asarray(post_mu).T + self.mean, np.asarray(post_var).T


def grid_sample(lower, upper, n):
    """ Used to seed an algorithm with a regular pattern of the corners and
    the centre. Provide search parameters and the index.
    """
    dims = len(lower)
    n_corners = 2 ** dims
    if n < n_corners:  # Sample the corners
        xq = lower + (upper - lower) * \
            (n & 2 ** np.arange(dims) > 0).astype(float)
    elif n == n_corners:  # Then sample the centre
        xq = lower + 0.5 * (upper - lower)
    else:
        assert(False)
    return xq


def random_sample(lower, upper, n):
    """ Used to randomly sample the search space.
    Provide search parameters and the number of samples desired.
    """
    dims = len(lower)
    X = np.random.random((n, dims))
    volume_range = [upper[i] - lower[i] for i in range(dims)]
    X_scaled = X * volume_range
    X_shifted = X_scaled + lower

    return X_shifted


class Candidates:
    def __init__(self, X_test=None, mu=None, var=None):
        self.X_test = X_test
        self.mu = mu
        self.var = var