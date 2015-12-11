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
    virtual_flags : list
        A list of boolean flags i_stackicating the virtual elements of 'y'
            True: Corresponding target output is virtual
            False: Corresponding target output is observed
    pending_indices : dict
        A dictionary that maps the job ID to the corresponding indices in both
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
        self.n_dims = self.upper.shape[0]
        assert self.lower.shape[0] == self.n_dims
        self.X = []
        self.y = []
        self.virtual_flags = []
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
        self.virtual_flags.append(True)

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

        # Kill the job and update collected data with true observation
        i_stack = self.pending_indices.pop(uid)
        self.y[i_stack] = np.asarray(y_true)
        self.virtual_flags[i_stack] = False

        return i_stack


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
    def __init__(self, lower, upper, explore_priority = 0.0001):
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
        n_corners = 2 ** self.n_dims
        if n < n_corners + 1:

            # Bootstrap with a regular sampling strategy to get it started
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = 0.
        else:

            # Otherwise, recursive subdivide the edges with the Delaunay model
            if not self.triangulation:
                self.triangulation = ScipyDelaunay(self.X, incremental = True)

            points = self.triangulation.points
            yvals = np.asarray(self.y)
            virtual = np.asarray(self.virtual_flags)

            # Weight by hyper-volume
            simplices = [tuple(s) for s in self.triangulation.vertices]
            cache = self.simplex_cache

            def get_value(s):

                # Computes the sample value as:
                #   hyper-volume of simplex * variance of values in simplex
                i_stack = list(s)
                value = (np.var(yvals[i_stack]) + self.explore_priority) * \
                    np.linalg.det((points[i_stack] - points[i_stack[0]])[1:])
                if not np.max(virtual[i_stack]):
                    cache[s] = value
                return value

            # Mostly the simplices won't change from call to call - cache!
            sample_value = [cache[s] if s in cache else get_value(s)
                            for s in simplices]

            # Fi_stack the points in the highest value simplex
            simplex_i_stackices = list(simplices[np.argmax(sample_value)])
            simplex = points[simplex_i_stackices]
            simplex_v = yvals[simplex_i_stackices]

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
    hyperparams : numpy.ndarray
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
                 kerneldef = None, add_train = True,
                 explore_priority = 0.01):
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
        add_train : boolean
            Whether to add training data to the sampler or not
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)
        self.hyperparams = None
        self.regressor = None
        self.kernel = None
        self.print_kernel = None
        self.explore_priority = explore_priority
        self._train(X, y,
                    kerneldef = kerneldef, add_train = add_train)

    def _train(self, X, y,
               kerneldef = None, add_train = True):
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
        add_train : boolean
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
        self.hyperparams = gp.learn(X, y, self.kernel, opt_config)

        # Adds sampled data to the model
        if add_train:
            self.X = X.copy()
            self.y = y.copy()
            self.virtual_flags = [False for y_i in y]
            self.regressor = gp.condition(np.asarray(self.X),
                                          np.asarray(self.y),
                                          self.kernel, self.hyperparams)

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
        i_stack = self._update(uid, y_true)
        if self.regressor:
            self.regressor.y[i_stack] = y_true
            self.regressor.alpha = gp.predict.alpha(self.regressor.y,
                                                    self.regressor.L)
        return i_stack

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
            The type of acq_name function used

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job
        """
        n = len(self.X)
        n_corners = 2 ** self.n_dims
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

            acq_name_dict = {
                'var_max': lambda u, v: np.argmax(v, axis=0),
                'pred_max': lambda u, v: np.argmax(u + np.sqrt(v), axis=0),
                'entropy_var': lambda u, v:
                    np.argmax((self.explore_priority + np.sqrt(v)) *
                              u * (1 - u), axis=0),
                'sigmoid': lambda u, v:
                    np.argmax(np.abs(stats.logistic.cdf(u + np.sqrt(v),
                              loc=0.5, scale=self.explore_priority) -
                              stats.logistic.cdf(u - np.sqrt(v),
                              loc=0.5, scale=self.explore_priority)), axis=0)
            }

            iq = acq_name_dict[acq_fn](post_mu, post_var)
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
                                          self.hyperparams)

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
        real_flag = ~np.asarray(self.virtual_flags)
        X_real = np.asarray(self.X)[real_flag]
        y_real = np.asarray(self.y)[real_flag]
        y_mean = y_real.mean()

        regressor = gp.condition(X_real, y_real - y_mean, self.kernel,
                                 self.hyperparams)
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
    hyperparams : numpy.ndarray
        The hyperparameters of the Gaussian Process Inference Model
    regressors : list
        List of regressor objects. See 'gp.types.RegressionParams'
    mean : float
        Mean of the training target outputs
    trained_flag : bool
        Whether the GP model have been trained or not
    acq_name : str
        A string specifying the type of acquisition function used
    explore_priority : float
        The priority of exploration against exploitation
    n_min : int
        Number of training samples required before sampler can be trained

    See Also
    --------
    Sampler : Base Class
    """
    def __init__(self, lower, upper, X = [], y = [],
                 add_train = True, kerneldef = None,
                 hyperparams = None, n_min = None, acq_name = 'var_sum',
                 explore_priority = 0.01):
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
        add_train : boolean
            Whether to add training data to the sampler or not
        hyperparams : tuple
            Hyperparameters of the Gaussian process
        n_min : int
            Number of training samples required before sampler can be trained
        y_mean : float
            Mean of the training target outputs
        acq_name : str
            A string specifying the type of acquisition function used
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)

        if kerneldef is None:
            self.kerneldef = lambda h, k: \
                h(1e-3, 1e+2, 1) * k('matern3on2',
                                     h(1e-2 * np.ones(self.n_dims),
                                       1e+3 * np.ones(self.n_dims),
                                       1e+0 * np.ones(self.n_dims)))
        else:
            self.kerneldef = kerneldef

        self.hyperparams = hyperparams
        self.regressors = None
        self.y_mean = None

        # If the training data is not supplied, the mean of the target output
        # is specified by the keyword argument
        self.acq_name = acq_name
        self.explore_priority = explore_priority

        self.n_stacks = len(y[0]) if y else None
        self.n_min = n_min if n_min is not None else (4 ** self.n_dims)

    def set_kerneldef(self, kerneldef):
        assert callable(kerneldef)
        self.kerneldef = kerneldef

    def get_kerneldef(self):
        return self.kerneldef

    def print_kernel(self, kerneldef):
        # TO DO: Use the printer method to print the current kernel!
        pass

    def set_hyperparams(self, hyperparams):
        if isinstance(hyperparams, list):
            self.hyperparams = hyperparams
        else:
            self.hyperparams = [hyperparams for i in range(self.n_stacks)]

        self.update_regressors()

    def get_hyperparams(self):
        return self.hyperparams

    def set_acq_name(self, acq_name):
        assert type(acq_name) is str
        self.acq_name = acq_name

    def get_acq_func(self):
        return acq_defs(y_mean =
                        self.y_mean,
                        explore_priority =
                        self.explore_priority)[self.acq_name]

    def set_explore_priority(self, explore_priority):
        self.explore_priority = explore_priority

    def get_explore_priority(self):
        return self.explore_priority

    def set_min_training_size(self, n_min):
        self.n_min = n_min

    def get_min_training_size(self):
        return self.n_min

    def refresh(self):

        if self.y:

            self.y = atleast_2d(self.y)
            self.n_stacks = len(self.y[0])
            _, y_real = self.get_real_data()
            self.y_mean = y_real.mean(axis = 0)
            assert self.y_mean.shape[0] == self.n_stacks

        else:

            self.n_stacks = None
            self.y_mean = None

    def get_real_data(self):

        assert self.X != []
        assert self.y != []

        real_flag = ~np.asarray(self.virtual_flags)
        X_real = np.asarray(self.X)[real_flag]
        y_real = np.asarray(self.y)[real_flag]

        return X_real, y_real

    def learn_hyperparams(self, X, y):
        """
        Trains the Gaussian process used for the sampler

        .. note : No properties are modified in this method

        Parameters
        ----------
        X : numpy.ndarray
            Training features for the Gaussian process model
        y : numpy.ndarray
            Training targets for the Gaussian process model
        hyperparams : tuple
            Hyperparameters of the Gaussian process
        """
        # Compose the kernel and setup the optimiser
        kernel = gp.compose(self.kerneldef)
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(self.kerneldef)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 50.0
        opt_config.global_opt = False

        # Make consistent the data format
        if type(X) is not np.ndarray:
            X = np.asarray(X)
        if type(y) is not np.ndarray:
            y = np.asarray(y)

        # Find the mean of the target outputs
        y_mean = y.mean(axis = 0)

        # Make sure the number of stacks recorded is consistent
        if self.n_stacks is None:
            self.n_stacks = y_mean.shape[0]
        else:
            assert self.n_stacks == y_mean.shape[0]

        # We need to train a regressor for each of the stacks
        # Each regressor will use the same hyperparameters!
        # We will use folds to do this
        folds = gp.Folds(self.n_stacks, [], [], [])
        for i_stack in range(self.n_stacks):
            folds.X.append(X)
            folds.flat_y.append(y[:, i_stack] - y_mean[i_stack])
        hyperparams = gp.train.learn_folds(folds, kernel, opt_config)

        # Use the same hyperparameters for each of the stacks
        return [hyperparams for i_stack in range(self.n_stacks)]

    def update_regressors(self):
        """ THE ONLY THING MODIFIED IS self.regressors """
        """ I can probably cache the regressors more efficiently sometime
        atleast right now this is correct """

        if self.hyperparams is None:
            return

        X, y = self.get_real_data()
        # Create the regressors if it hasn't already been
        # if self.regressors is None:
        kernel = gp.compose(self.kerneldef)
        self.regressors = []
        for i_stack in range(self.n_stacks):
            self.regressors.append(
                gp.condition(X, y[:, i_stack] - self.y_mean[i_stack],
                             kernel, self.hyperparams[i_stack]))

        # # Otherwise, simply update the regressors
        # else:
        #     for i_stack, regressor in enumerate(self.regressors):
        #         regressor.y = y[:, i_stack] - self.y_mean[i_stack]
        #         regressor.alpha = gp.predict.alpha(regressor.y, regressor.L)

    def train(self, X, y):
        """
        .. note : Only self.hyperpararms and self.regressors are changed
        """
        # Learn hyperparameters
        self.hyperparams = self.learn_hyperparams(X, y)

        # Update the regressors
        self.update_regressors()

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
        if type(y_true) is not np.ndarray:
            if type(y_true) is list:
                y_true = np.array(y_true)
            else:
                y_true = np.array([y_true])
        ind = self._update(uid, y_true)
        self.refresh()
        self.update_regressors()
        return ind

    def pick(self, n_test = 500, train = False):
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
        n_corners = 2 ** self.n_dims

        self.refresh()

        # If we do not have enough samples yet, randomly sample for more!
        if n < self.n_min:

            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        # Bootstrap with a regular sampling strategy to get it started
        elif n < n_corners + 1:
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        # Active Sample!
        else:

            if train or self.regressors is None:
                self.train(self.X, self.y)

            # Randomly sample the volume for test points
            Xq = random_sample(self.lower, self.upper, n_test)

            # Generate cached predictors for those test points
            predictors = [gp.query(Xq, r) for r in self.regressors]

            # Compute the posterior distributions at those points
            # Note: No covariance information implemented at this stage
            Yq_exp = self.y_mean + np.asarray([gp.mean(r, q) for r, q in
                                               zip(self.regressors,
                                                   predictors)]).T
            Yq_var = np.asarray([gp.variance(r, q) for r, q in
                                 zip(self.regressors, predictors)]).T

            # Aquisition Functions
            acq_defs_current = acq_defs(y_mean =
                                        self.y_mean,
                                        explore_priority =
                                        self.explore_priority)

            # Compute the acquisition levels at those test points
            yq_acq = acq_defs_current[self.acq_name](Yq_exp, Yq_var)

            # Find the test point with the highest acquisition level
            iq_acq = np.argmax(yq_acq)
            xq = Xq[iq_acq, :]
            yq_exp = Yq_exp[iq_acq, :]

        # Place a virtual observation...
        uid = Sampler._assign(self, xq, yq_exp)

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

        real_flag = ~np.asarray(self.virtual_flags)
        X_real = np.asarray(self.X)[real_flag]
        y_real = np.asarray(self.y)[real_flag]

        post_mu = []
        post_var = []

        for i_stack in range(self.n_stacks):
            regressor = gp.condition(X_real, y_real[:, i_stack] - self.y_mean,
                                     self.kernel, self.hyperparams[i_stack])
            predictor = gp.query(Xq, regressor)
            post_mu.append(gp.mean(regressor, predictor))
            post_var.append(gp.variance(regressor, predictor))

        return np.asarray(post_mu).T + self.y_mean, np.asarray(post_var).T


def atleast_2d(y):
    """
    ..note : Assumes homogenous list or arrays
    """
    if isinstance(y, list):
        if type(y[0]) is not np.ndarray:
            return [np.array([y_i]) for y_i in y]
        elif len(y[0].shape) == 1:
            return y
        else:
            raise ValueError("List element already has more than 1 dimension")
    elif isinstance(y, np.ndarray):
        if len(y.shape) == 1:
            return y[:, np.newaxis]
        elif len(y.shape) == 2:
            return y
        else:
            raise ValueError("Object already has more than 2 dimensions")
    else:
        raise ValueError('Object is not a list or an array')


def grid_sample(lower, upper, n):
    """
    Used to seed an algorithm with a regular pattern of the corners and
    the centre. Provide search parameters and the i_stackex.

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


def acq_defs(y_mean = 0, explore_priority = 0.01):

    # Aquisition Functions
    # u: Mean matrix (n x n_stacks)
    # v: Variance matrix (n x n_stacks)
    # Returns an array of n values

    return {
        'var_sum': lambda u, v: np.sum(v, axis = 1),
        'pred_max': lambda u, v: np.max(u + 3 * np.sqrt(v), axis = 1),
        'prod_max': lambda u, v: np.max((u + (y_mean +
                                        explore_priority / 3.0)) *
                                        np.sqrt(v), axis = 1),
        'prob_tail': lambda u, v: np.max((1 - stats.norm.cdf(
                                         explore_priority *
                                         np.ones(u.shape), u,
                                         np.sqrt(v))), axis = 1),
        'sigmoid': lambda u, v: np.abs(stats.logistic.cdf(u + np.sqrt(v),
                                       loc = 0.5,
                                       scale = explore_priority) -
                                       stats.logistic.cdf(u - np.sqrt(v),
                                       loc = 0.5,
                                       scale = explore_priority)).sum(axis = 1)
    }
