
from .base_sampler import Sampler, random_sample
from .acquisition_functions import UpperBound

import GPflow as gp
import numpy as np


class GPflowSampler(Sampler):
    """ Gaussian Process sampler using GPflow's `GPR` Gaussian Process
        regressor.
    """
    name = 'GPflowSampler'

    def __init__(self, lower, upper, n_train=50, kern=None,
                 mean_function=gp.mean_functions.Constant(),
                 acquisition_function=UpperBound(), seed=None):
        """ Initialise the GPflowSampler.
        """
        super().__init__(lower, upper)

        self._n_train = n_train

        self.acquisition_function = acquisition_function
        self.kernel = kern if kern else gp.kernels.RBF(self.dims)
        self.mean_function = mean_function

        self._gpr = None
        self._params = None

        if seed:
            np.random.seed(seed)

    @property
    def min_training_size(self):
        return self._n_train

    @min_training_size.setter
    def min_training_size(self, val):
        self._n_train = val

    @property
    def hyperparams(self):
        return self._params

    @hyperparams.setter
    def hyperparams(self, val):
        self._params = val

    @property
    def gpr(self):
        return self._gpr

    def add_data(self, X, y, train=False):
        """ Add training data, and optionally train hyper parameters.
        """
        [self.X.append(xi) for xi in X]
        [self.y.append(np.atleast_1d(yi)) for yi in y]
        [self.virtual_flag.append(False) for _ in y]

        if self._gpr:
            params = None if train else self._params
            self._gpr = self._create_gpr(self.X(), self.y(), params=params)


    def update(self, uid, y_true):
        """ Update a job id with an observed value. Makes a virtual
            observation real.
        """
        ind = self._update(uid, y_true)
        self.update_y_mean()
        if self._params:
            self._gpr = self._create_gpr(self.X(), self.y(),
                                         params=self._params)

        return ind

    def pick(self, n_test=500):
        """ Pick a feature location for the next observation, which maximises
            the acquisition function.
        """
        n = len(self.X)

        # If we do not have enough samples yet, randomly sample for more!
        if n < self._n_train:
            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        else:
            if self._gpr is None:
                self._gpr = self._create_gpr(self.X(), self.y())
                self._params = self.gpr.get_parameter_dict()

            # Randomly sample the volume for test points
            Xq = random_sample(self.lower, self.upper, n_test)

            # Compute the posterior distributions at those points
            Yq_exp, Yq_var = self.gpr.predict_y(Xq)

            # Acquisition Function
            yq_acq = self.acquisition_function(Yq_exp, Yq_var)

            # Find the test point with the highest acquisition level
            iq_acq = np.argmax(yq_acq)
            xq = Xq[iq_acq, :]
            yq_exp = Yq_exp[iq_acq, :]

        # Place a virtual observation...
        uid = Sampler._assign(self, xq, yq_exp)  # it can be None...

        return xq, uid

    def eval_acq(self, Xq):
        """ Evaluate the acquisition function for a set of query points (Xq).
        """
        if len(Xq.shape) == 1:
            Xq = Xq[:, np.newaxis]

        Yq_exp, Yq_var = self.gpr.predict_y(Xq)

        yq_acq = self.acquisition_function(Yq_exp, Yq_var)

        return yq_acq, np.argmax(yq_acq)

    def predict(self, Xq, real=True):
        """ Return the mean and variance of the GP model at query point.

            Use `real=False` to use both real and virtual observations.
        """
        assert self._params, "Sampler is not trained yet. " \
                             "Possibly not enough observations provided."

        if real:
            X_real, y_real = self.get_real_data()
            m = self._create_gpr(X_real, y_real, params=self._params)
        else:
            m = self.gpr

        Yq_exp, Yq_var = m.predict_y(Xq)

        return Yq_exp, Yq_var


    def _create_gpr(self, X, y, params=None):
        """ Helper function to create (and optimise if necessary) a GPflow
            Gaussian Process Regressor
        """
        m = gp.gpr.GPR(X, y, kern=self.kernel, mean_function=self.mean_function)
        if params is not None:
            m.set_parameter_dict(self._params)
        else:
            m.optimize()

        return m