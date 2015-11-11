""" Active Sampling module
Strategies for active sampling a spatial field.
"""
import numpy as np
from scipy.spatial import Delaunay as ScipyDelaunay
import dora.regressors.gp as gp
import scipy.stats as stats
import hashlib


class Base_Sampler:
    def __init__(self, lower, upper):
        """
        Arguments:
        lower (array floats) - min of bounding box
        upper (array floats) - max of bounding box
        explore_priority (float) - value of constant-volume
        """
        upper = np.asarray(upper)
        lower = np.asarray(lower)
        self.dims = len(upper)
        assert(len(lower) == self.dims)
        self.lower = lower
        self.upper = upper
        self.X = []
        self.y = []
        self.virtual_flag = []  # true = expected. false = measured
        self.pending_results = {}  # Maps a tuple representation of X
        self.pending_results_uid = {} # tuple that maps job uid to index
        # to an int that points to the index of x and y. It only contains
        # indices of the virtual observations

    def pick(self):
        """
        Picks the next point to be evaluated.
        It also fills in a dummy point at that location.
        """

        assert(False)

    def update(self, X, value):
        """
        Updates the point X with the value y
        """
        assert(False)

    def _update(self, job_uid, value):
        if job_uid not in self.pending_results:
            raise(ValueError('Result wasnt pending!'))
        assert(job_uid in self.pending_results)
        ind = self.pending_results.pop(job_uid)
        self.y[ind] = value
        self.virtual_flag[ind] = False
        return ind


class Delaunay(Base_Sampler):
    """
    Inherits from the Base_Sampler class and augments pick and update with the
    mechanics of the Delanauy triangulation method
    """
    def __init__(self, lower, upper, explore_priority=0.0001):
        """
        Arguments:
        lower (array floats) - min of bounding box
        upper (array floats) - max of bounding box
        explore_priority (float) - value of constant-volume
        """
        Base_Sampler.__init__(self, lower, upper)
        self.triangulation = None  # Delaunay model
        self.simplex_cache = {}  # Pre-computed values of simplices
        self.explore_priority = explore_priority

    def update(self, job_uid, y):
        """ Applies an observation to a Delaunay active sampling model
        """
        Base_Sampler._update(self, job_uid, y)

    def pick(self):
        """
        Picks a new point using the recursive Delaunay subdivision algorithm
        Returns
            X - the new search to sample
        """
        n = len(self.X)
        n_corners = 2**self.dims
        if n < n_corners + 1:
            # Bootstrap with a regular sampling strategy to get it started
            new_X = grid_sample(self.lower, self.upper, n)
            expected_y = 0.
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
                # computes the sample value as:
                # hyper-volume of simplex * variance of values in simplex
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
            new_X = weight.dot(simplex)
            expected_y = weight.dot(simplex_v)
            self.triangulation.add_points(new_X[np.newaxis, :])  # incremental

        # Place a virtual observation...
        self.X.append(new_X)
        self.y.append(expected_y)  # fill with 'expected' value
        self.virtual_flag.append(True)

        # save it for later
        m = hashlib.md5()
        m.update(np.array(np.random.random()))
        uid = m.hexdigest()
        self.pending_results[uid] = n

        return new_X, uid


class Gaussian_Process(Base_Sampler):
    """
    Inherits from the Base_Sampler class and augments pick and update with the
    mechanics of the GP method
    """
    def __init__(self, lower, upper, X_train, y_train, add_train_data=True):
        """
        Arguments:
        lower (array floats) - min of bounding box
        upper (array floats) - max of bounding box
        explore_priority (float) - value of constant-volume
        """
        # import ipdb; ipdb.set_trace()
        Base_Sampler.__init__(self, lower, upper)
        self.hyper_params = None
        self.regressor = None
        self.train_hypers(X_train, y_train, add_train_data)

    def train_hypers(self, X_train, y_train, add_train_data):
        # We will be using the Matern 3/2 Kernel
        minLength = 1e-2*np.ones(1)
        maxLength = 1e3*np.ones(1)
        initLength = np.array([0.5])
        kernel = lambda h, k: (h(1e-3, 1e2, 2.321) *
                               k('matern3on2',
                                 h(minLength, maxLength, initLength)))
        # Set up optimisation
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(kernel)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 50.0
        opt_config.global_opt = False

        self.kernelFn = gp.compose(kernel)
        self.printKernFn = gp.describer(kernel)

        self.hyper_params = gp.learn(X_train, y_train, self.kernelFn, 
                                     opt_config)
        print('Final kernel:', self.printKernFn(self.hyper_params), '+ noise',
              self.hyper_params[1])

        if add_train_data:  # adds this sampled data to the model
            self.X = [i for i in X_train]
            self.y = [i for i in y_train]
            self.virtual_flag = [False for i in X_train]
            self.regressor = gp.condition(np.asarray(self.X),
                                          np.asarray(self.y),
                                          self.kernelFn, self.hyper_params)

    def update(self, job_uid, value):
        """ Applies an observation to a Gaussian process active sampling model
        """
        ind = self._update(job_uid, value)
        self.regressor.y[ind] = value
        self.regressor.alpha = gp.predict.alpha(self.regressor.y,
                                                self.regressor.L)

    def pick(self, n_test=500):
        """

        """
        # import ipdb; ipdb.set_trace()
        n = len(self.X)
        n_corners = 2**self.dims
        if n < n_corners + 1:
            # Bootstrap with a regular sampling strategy to get it started
            new_X = grid_sample(self.lower, self.upper, n)
            expected_y = 0.
        else:

            # Randomly sample the volume.
            X_test = random_sample(self.lower, self.upper, n_test)
            query = gp.query(X_test, self.regressor)
            post_mu = gp.mean(self.regressor, query)
            post_var = gp.variance(self.regressor, query)

            explore = 0.01  # an exploration factor for the acquisition funcs
            acq_func_dict = {
                'maxvar': lambda u, v: np.argmax(v, axis=0),
                'predmax': lambda u, v: np.argmax(u + np.sqrt(v), axis=0),
                'entropyvar': lambda u, v: np.argmax((explore + np.sqrt(v)) *
                                                     u*(1-u), axis=0),
                'sigmoid': lambda u, v: np.argmax(
                    np.abs(stats.logistic.cdf(u+np.sqrt(v), loc=0.5,
                                              scale=explore) -
                           stats.logistic.cdf(u-np.sqrt(v), loc=0.5,
                                              scale=explore)), axis=0)
            }

            new_point_ID = acq_func_dict['sigmoid'](post_mu, post_var)
            new_X = X_test[new_point_ID, :]

            expected_y = post_mu[new_point_ID]

        # Place a virtual observation...
        m = hashlib.md5()
        m.update(np.array(np.random.random()))
        uid = m.hexdigest()
        self.X.append(new_X)
        self.y.append(expected_y)  # fill with 'expected' value
        self.virtual_flag.append(True)
        # save it for later
        self.pending_results[uid] = n

        if not self.regressor:
            self.regressor = gp.condition(np.asarray(self.X),
                                          np.asarray(self.y), self.kernelFn,
                                          self.hyper_params)
        else:
            gp.add_data(np.asarray(new_X[np.newaxis, :]),
                        np.asarray(expected_y)[np.newaxis],
                        self.regressor)

        return new_X, uid

    def predict(self, query_points):
        """
        method to query the probabilistic model at locations
        :param query_points: n x d array of points in the region of interest
        :return: post_mu: n x n_stacks array of predicted mean values
                 post_var: n x n_stacks array of predicted variance values
        """

        # extract only the real observations for conditioning the predictor
        #TODO Consider moving real_y inside of the for loop use regressor.y
        real_id = [not i for i in self.virtual_flag]
        real_X = [x for x, real in zip(self.X, real_id) if real is True]
        real_y = [y for y, real in zip(self.y, real_id) if real is True]
        real_X = np.asarray(real_X)
        real_y = np.asarray(real_y)-self.mean

        regressor = gp.condition(real_X, real_y, self.kernelFn,
                                 self.hyper_params)
        query_object = gp.query(query_points,regressor)
        post_mu = gp.mean(regressor,query_object)
        post_var = gp.variance(regressor,query_object)

        return post_mu + self.mean, post_var


class Stacked_Gaussian_Process(Base_Sampler):
    """
    Inherits from the Base_Sampler class and augments pick and update with the
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
        # import ipdb; ipdb.set_trace()

        Base_Sampler.__init__(self, lower, upper)
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
                                     self.kernelFn, self.hyper_params[ind]))

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
        kernel = lambda h, k: \
            h(1e-3, 1e2, 2.321) * k('matern3on2', h(minL, maxL, initL))
        self.kernelFn = gp.compose(kernel)
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(kernel)
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

            hypers = gp.train.learn_folds(folds, self.kernelFn, opt_config)

        for stack in range(self.n_stacks):
            self.hyper_params.append(hypers)

        self.trained_flag = True

    def update(self, job_uid, value):
        """ Applies an observation to a Gaussian process active sampling model
        """
        self._update(job_uid, value)
        if self.trained_flag:
            full_y = np.asarray(self.y)
            for i, regressor in enumerate(self.regressors):
                regressor.y = full_y[:,i]-self.mean
                regressor.alpha = gp.predict.alpha(regressor.y, regressor.L)

    def pick(self, n_test=500):
        n = len(self.X)
        n_corners = 2**self.dims

        if not self.trained_flag:
            new_X = random_sample(self.lower, self.upper, 1)[0,:]
            expected_y = 0. * np.ones(self.n_stacks) + self.mean

        elif n < n_corners + 1:
            # Bootstrap with a regular sampling strategy to get it started
            new_X = grid_sample(self.lower, self.upper, n)
            expected_y = 0. * np.ones(self.n_stacks) + self.mean

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

            new_point_ID = acq_func_dict[self.acq_func](post_mu, post_var)
            new_X = X_test[new_point_ID, :]
            expected_y = post_mu[:, new_point_ID]


        # Place a virtual observation...
        m = hashlib.md5()
        m.update(np.array(np.random.random()))
        uid = m.hexdigest()
        self.X.append(new_X)
        self.y.append(expected_y)
        self.virtual_flag.append(True)
        self.pending_results[uid] = n

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
                                     self.kernelFn, self.hyper_params[ind]))
            else:
                for ind in range(self.n_stacks):
                    gp.add_data(np.asarray(new_X[np.newaxis, :]),
                                np.asarray(expected_y[ind])[np.newaxis] - self.mean,
                                self.regressors[ind])

        return new_X, uid

    def predict(self, query_points):
        """
        method to query the probabilistic model at locations
        :param query_points: n x d array of points in the region of interest
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
            regressor = gp.condition(real_X, real_y[:,ind], self.kernelFn,
                                     self.hyper_params[ind])
            query_object = gp.query(query_points,regressor)
            post_mu.append(gp.mean(regressor,query_object))
            post_var.append(gp.variance(regressor,query_object))

        return np.asarray(post_mu).T + self.mean, np.asarray(post_var).T



def grid_sample(lower, upper, n):
    """ Used to seed an algorithm with a regular pattern of the corners and
    the centre. Provide search parameters and the index.
    """
    dims = len(lower)
    n_corners = 2**dims
    if n < n_corners:  # Sample the corners
        new_X = lower + (upper - lower) * \
            (n & 2**np.arange(dims) > 0).astype(float)
    elif n == n_corners:  # Then sample the centre
        new_X = lower + 0.5*(upper-lower)
    else:
        assert(False)
    return new_X


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


class candidates:
    def __init__(self, X_test=None, mu=None, var=None):
        self.X_test = X_test
        self.mu = mu
        self.var = var

