
from .base_sampler import Sampler, random_sample
from .acquisition_functions import UpperBound

import GPflow as gp

import numpy as np


class GPflowSampler(Sampler):

    name = 'GPflowSampler'

    def __init__(self, lower, upper, n_train=50, kern=None,
                 mean_function=gp.mean_functions.Constant(),
                 acquisition_function=UpperBound(), seed=None):

        super().__init__(lower, upper)

        self.n_min = n_train
        self.acquisition_function = acquisition_function
        self.kernel = kern if kern else gp.kernels.RBF(self.dims)
        self.mean_func = mean_function

        self.gpr = None
        self.params = None
        self.y_mean = None

        if seed:
            np.random.seed(seed)

    def update(self, uid, y_true):

        ind = self._update(uid, y_true)
        self.update_y_mean()
        if self.params:
            self.gpr = self._create_gpr(self.X(), self.y(), params=self.params)
        return ind

    def pick(self, n_test=500):

        n = len(self.X)

        # If we do not have enough samples yet, randomly sample for more!
        if n < self.n_min:
            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        else:
            if self.gpr is None:
                self.gpr = self._create_gpr(self.X(), self.y())
                self.params = self.gpr.get_parameter_dict()

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

        if len(Xq.shape) == 1:
            Xq = Xq[:, np.newaxis]

        Yq_exp, Yq_var = self.gpr.predict_y(Xq)

        yq_acq = self.acquisition_function(Yq_exp, Yq_var)

        return yq_acq, np.argmax(yq_acq)


    def predict(self, Xq, real=True):

        assert self.params, "Sampler is not trained yet. " \
                            "Possibly not enough observations provided."

        if real:
            X_real, y_real = self.get_real_data()
            m = self._create_gpr(X_real, y_real, params=self.params)
        else:
            m = self.gpr

        Yq_exp, Yq_var = m.predict_y(Xq)

        return Yq_exp, Yq_var


    def _create_gpr(self, X, y, params=None):

        m = gp.gpr.GPR(X, y, kern=self.kernel,  mean_function=self.mean_func)
        if params is not None:
            m.set_parameter_dict(self.params)
        else:
            m.optimize()

        return m