
from .base_sampler import Sampler, random_sample
from .acquisition_functions import UpperBound

import GPflow as gp

import numpy as np


class GPflowSampler(Sampler):

    name = 'GPflowSampler'

    def __init__(self, lower, upper, n_train=50, kern=None,
                 mean_function=gp.mean_functions.Zero(),
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
            self.gpr = gp.gpr.GPR(self.X(), self.y(), kern=self.kernel)
            self.gpr.set_parameter_dict(self.params)
        return ind

    def pick(self, n_test=500):

        n = len(self.X)

        # If we do not have enough samples yet, randomly sample for more!
        if n < self.n_min:
            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        else:
            if self.gpr is None:
                self.gpr = gp.gpr.GPR(self.X(), self.y(), kern=self.kernel,
                                      mean_function=self.mean_func)
                print(self.gpr)
                self.gpr.optimize()

                print(self.gpr)
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
        Yq_exp += self.y_mean

        yq_acq = self.acquisition_function(Yq_exp, Yq_var)

        return yq_acq, np.argmax(yq_acq)


    def predict(self, Xq, real=True):

        assert self.params, "Sampler is not trained yet. " \
                            "Possibly not enough observations provided."

        if real:
            X_real, y_real = self.get_real_data()
            m = gp.gpr.GPR(X_real, y_real, kern=self.kernel,
                           mean_function=self.mean_func)
            m.set_parameter_dict(self.params)
        else:
            m = self.gpr

        Yq_exp, Yq_var = m.predict_y(Xq)

        return Yq_exp, Yq_var
