
from dora.active_sampling.base_sampler import Sampler, random_sample
from dora.active_sampling.gp_sampler import acq_defs

import GPflow as gp

import numpy as np
import scipy.stats as stats


class GPflowSampler(Sampler):

    name = 'GPflowSampler'

    def __init__(self, lower, upper, n_train=50, acq_name='var_sum',
                 explore_priority=1.0, seed=None):

        super().__init__(lower, upper)

        self.n_min = n_train
        self.acq_name = acq_name
        self.explore_priority = explore_priority
        self.kernel = None
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
                self.kernel = gp.kernels.RBF(self.dims)
                self.gpr = gp.gpr.GPR(self.X(), self.y() - self.y_mean,
                                      kern=self.kernel)
                self.gpr.optimize()
                self.params = self.gpr.get_parameter_dict()

            # Randomly sample the volume for test points
            Xq = random_sample(self.lower, self.upper, n_test)

            # Compute the posterior distributions at those points
            Yq_exp, Yq_var = self.gpr.predict_y(Xq)
            Yq_exp += self.y_mean

            # Aquisition Functions
            acq_defs_current = acq_defs(y_mean=self.y_mean,
                                        explore_priority=self.explore_priority)

            # Compute the acquisition levels at those test points
            yq_acq = acq_defs_current[self.acq_name](Yq_exp, Yq_var)

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

        # Aquisition Functions
        acq_defs_current = acq_defs(y_mean=self.y_mean,
                                    explore_priority=self.explore_priority)

        # Compute the acquisition levels at those test points
        yq_acq = acq_defs_current[self.acq_name](Yq_exp, Yq_var)

        return yq_acq, np.argmax(yq_acq)

    def predict(self, Xq, real=True):

        assert self.params, "Sampler is not trained yet. " \
                            "Possibly not enough observations provided."

        if real:
            X_real, y_real = self.get_real_data()
            m = gp.gpr.GPR(X_real, y_real - self.y_mean, kern=self.kernel)
            m.set_parameter_dict(self.params)
        else:
            m = self.gpr

        Yq_exp, Yq_var = m.predict_y(Xq)

        return Yq_exp + self.y_mean, Yq_var
