""" Bayesian Linear Sampler Module. """
import logging

import numpy as np
import scipy
import scipy.stats as stats
import scipy.linalg as linalg

from dora.active_sampling.base_sampler import Sampler, random_sample

log = logging.getLogger(__name__)

class BayesianLinear(Sampler):
    """BayesianLinear Class.
    Attributes
    ----------
    basisdef: str
            basis function definition.
    basisparams: list [mu, s]
                basis function parameters.
    feature: int
            number of features.
    n_min: int
        number of training samples required before sampler is trained.
    acq_name: str
            acquisition function definition.
    explore_priority: float, optional
                    the priority of exploration against exploitation.
    hyperparams: list [alpha, beta]
                hyperparameters of the Bayesian Linear model.
    y_mean: float
            mean of the training target outputs.
    regressors: list
                list of regressor objects.
    n_tasks: int
            number of tasks.
    """

    name = 'BayesianLinear'

    def __init__(self, lower, upper, basisdef=None, basisparams=None, feature=5, n_train=50,
                 acq_name="var_sum", explore_priority=1.0, seed=None):

        Sampler.__init__(self, lower, upper)
        self.basisdef = basisdef
        self.basisparams = basisparams
        self.feature =feature
        self.n_min = n_train
        self.acq_name = acq_name
        self.explore_priority = explore_priority
        self.hyperparams = None
        self.regressors = None
        self.y_mean = None
        self.n_tasks = None
        if seed:
            np.random.seed(seed)


    def update_y_mean(self):
        """update the mean of the target outputs."""
        if not self.y:
            return
        self.y_mean = self.y().mean(axis=0) if len(self.y) else None
        if self.n_tasks is None:
            self.n_tasks = self.y_mean.shape[0]
        else:
            assert self.n_tasks == self.y_mean.shape[0]


    def FeatureGen(self, x):
        """Compute the feature matrix.

        Parameters
        ----------
        x: ArrayBuffer
          the input feature.
        Returns
        -------
        theta: numpy.ndarray (#input points, feature)
              the feature matrix.
        """
        mu = self.basisparams[0]
        s = self.basisparams[1]
        basisdef_current = basis_defs(mu=mu, s=s)
        theta = basisdef_current[self.basisdef](x)
        return theta


    def learn_hyperparams(self):
        """Optimize hyperparams of the bayesian linear model.

        Returns
        -------
        param_list: list
                   the list of hyperparams for each task.
        """
        self.update_y_mean()
        logging.info('Training hyperparameters...')
        params0 = [10., 10.]
        def nLMLcriterion(params):
            return self.nMLM(params, self.FeatureGen(self.X))
        newparams = scipy.optimize.minimize(nLMLcriterion,
                                            params0, method="COBYLA")
        logging.info("Done.")
        params = newparams.x
        print("alpha={}, beta={}".format(params[0], params[1]))
        param_list = [tuple(params) for i_task in range(self.n_tasks)]
        return param_list


    def nMLM(self, params, theta):
        """Negative log likelihood.

        Parameters
        ----------
        params: tuple
               hyperparams of the bayesian linear model.
        theta: numpy.ndarray
              the feature matrix.
        Returns
        -------
        -MLM: float
             negative log likelihood.
        """
        alpha = params[0]**2
        beta = params[1]**2
        n, p = theta.shape
        A = beta * np.dot(np.transpose(theta), theta) + alpha * np.eye(p)
        L = linalg.cholesky(A, lower=True)
        mn = beta * linalg.solve(A, np.dot(np.transpose(theta), self.y))
        predY = np.dot(theta, mn)
        MLM = 0.5 * (p * np.log(alpha) + n*np.log(beta) - 2*np.sum(np.log(np.diag(L))) - n*np.log(2*np.pi) - beta * np.sum((self.y-predY)**2) - alpha*np.sum(mn**2))
        return -MLM


    def update_regressors(self):
        """Update regressors.

        Notes
        -----
        Call after self.learn_hyperparams()
        Each regressor object is a tuple (hyperparams, A, mu_w, L)
        """
        if self.hyperparams is None:
            return
        self.regressors = [ ]
        theta = self.FeatureGen(self.X)
        for i_task in range(self.n_tasks):
            #cache A, mu_w and L
            self.regressors.append(self._compute_AmuL(
                self.hyperparams[i_task],theta, self.y()[:, i_task]))


    def train(self):
        """Train the bayesian linear model."""
        assert(self.dims is not None)
        self.hyperparams = self.learn_hyperparams()
        self.update_regressors()


    def update(self, uid, y_true):
        """Update a job with its observed value.

        Parameters
        ----------
        uid: str
            hexadecimal ID uniquely identifies the job to be updated.
        y_true: float
              observed value.

        Returns
        -------
        ind: int
            location in the array buffer self.X and self.y
            corresponding to the job being updated.
        """
        y_true = np.atleast_1d(y_true)
        assert y_true.ndim == 1
        ind = self._update(uid, y_true)
        self.update_y_mean()
        self.update_regressors()
        return ind


    def pick(self, n_test=500):
        """Pick the next observation based on the acquisition value.

        Parameters
        ----------
        n_test: int
               the number of random query points across the search space.
        Returns
        -------
        xq: numpy.ndarray
           the next observation.
        uid: str
            hexademical ID to identify the next job.
        """
        n = len(self.X)
        self.update_y_mean()
        if n < self.n_min:
            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean
        else:
            if self.regressors is None:
                self.train()
            Xq = random_sample(self.lower, self.upper, n_test) #random_sample return Xq as (n_test, 1)
            # compute the posterior distributions at query points
            thetaQuery = self.FeatureGen(Xq)
            Yq_mean = np.array([self._compute_posterior(r, thetaQuery)[0] for r in self.regressors]).T
            Yq_std = np.array([self._compute_posterior(r, thetaQuery)[1] for r in self.regressors]).T
            # evaluate the acquisition function at Xq
            acq_defs_current = acq_defs(y_mean=self.y_mean,
                                        explore_priority=self.explore_priority)
            Yq_acq = acq_defs_current[self.acq_name](Yq_mean, Yq_std**2)
            iq_acq = np.argmax(Yq_acq)
            xq = Xq[iq_acq, :]
            yq_exp = Yq_mean[iq_acq,:]
        uid = Sampler._assign(self, xq, yq_exp)
        return xq, uid


    def _compute_AmuL(self, params, theta, y):
        """Compute A, mu_w and L for prediction.

        Parameters
        ----------
        params: tuple
               params of the bayesian linear model for each task.
        theta: numpy.ndarray
              the feature matrix.
        y: Array Buffer
          the target outputs.
        Returns
        -------
        (A, mu_w, L, params): tuple
        """
        alpha, beta = params[0]**2, params[1]**2
        I = np.eye(theta.shape[1])
        A = beta * np.dot(np.transpose(theta), theta) + alpha * I
        mu_w = beta * linalg.solve(A, np.transpose(theta).dot(y))
        L = linalg.cholesky(A, lower=True)
        return (A, mu_w, L, params)


    def _compute_posterior(self, regressor, thetaQuery):
        """Compute the posterior distribution.

        Parameters
        ----------
        regressor: tuple
                a tuple (A, mu_w, L, hyperparams)
                returned by _compute_AmuL.
        thetaQuery: numpy.ndarray
                  the query feature matrix.
        Returns
        -------
        yq_mean: numpy.ndarray
                mean of the posterior distribution.
        yq_std: numpy.ndarray
                standard deviation of the posterior distribution.
        """
        A, mu_w, L, params = regressor[:]
        beta = params[1]**2
        yq_mean = thetaQuery.dot(mu_w)
        yq_std = np.sqrt(np.sum(linalg.solve(L, np.transpose(thetaQuery))**2,
                                axis=0) + 1./beta)
        return yq_mean, yq_std


    def predict(self, Xq, real=True):
        """Predict the query mean and standard deviation.

        Parameters
        ----------
        Xq: numpy.ndarray
           query points.
        real: bool, optional
        Returns
        -------
        Yq_mean: numpy.ndarray
                mean of the prediction at query points.
        Yq_std: numpy.ndarray
                standard deviation of the prediction at query points.
        """
        assert self.hyperparams, "Sampler is not trained yet."
        if real:
            X_real, y_real = self.get_real_data()
            regressors = [self._compute_AmuL(
                self.hyperparams[i_task], self.FeatureGen(X_real), y_real[:, i_task]) for i_task in range(self.n_tasks)]
        else:
            regressors = self.regressors

        thetaQuery = self.FeatureGen(Xq)
        Yq_mean = np.array([self._compute_posterior(r, thetaQuery)[0] for r in regressors]).T
        Yq_std = np.array([self._compute_posterior(r, thetaQuery)[1] for r in regressors]).T
        return Yq_mean, Yq_std


    def get_real_data(self):
        """Obtain the observed data.

        Returns
        -------
        self.X()[real_flag]: numpy.ndarray
                            the observed feature locations.
        self.y()[real_flag]: numpy.ndarray
                            the observed target outputs.
        """
        assert self.X
        assert self.y
        real_flag = ~self.virtual_flag()
        return self.X()[real_flag], self.y()[real_flag]


    def eval_acq(self, Xq):
        """Evaluate the acquisition function.

        Parameters
        ----------
        Xq: numpy.ndarray
            query points.
        Returns
        -------
        yq_acq: numpy.ndarray
                acquisition function value at Xq.
        np.argmax(yq_acq): int
                          index of the maximum acquisition function value.
        """
        if len(Xq.shape)==1:
            Xq = Xq[:, np.newaxis]
        self.update_y_mean()
        thetaQuery = self.FeatureGen(Xq)
        Yq_mean = np.array([self._compute_posterior(r, thetaQuery)[0] for r in self.regressors]).T
        Yq_std = np.array([self._compute_posterior(r, thetaQuery)[1] for r in self.regressors]).T
        acq_defs_current = acq_defs(y_mean=self.y_mean,
                                    explore_priority=self.explore_priority)
        yq_acq = acq_defs_current[self.acq_name](Yq_mean, Yq_std**2)
        return yq_acq, np.argmax(yq_acq)


def basis_defs(mu, s):
    """Basis function definitions.

    Parameters
    ----------
    mu: numpy.ndarray
        mean of the basis function.
    s: float
       spatial width of the basis function.
    Returns
    -------
    dict
    dictionary of basis functions.
    """
    #TODO: add some other basis functions
    return{
        'radial': lambda x: np.exp(-(((x - mu.transpose())**2)/(2*s**2))) }


def acq_defs(y_mean=0, explore_priority=1.):
    """Acquisition function definitions.

    Parameters
    ----------
    y_mean: float or numpy.ndarray
            the mean of target outputs.
    explore_priority: float
                     exploration priority against exploitation.
    Returns
    -------
    dict
    dictionary of acquisition functions.
    """
    return {
        'var_sum': lambda u, v: np.sum(v, axis=1),
        'pred_upper_bound': lambda u, v: np.max(u + 3 * explore_priority * np.sqrt(v), axis=1),
        'prod_max': lambda u, v: np.max((u + y_mean +
                                        (explore_priority / .1) / 3.0) *
                                        np.sqrt(v), axis=1),
        'prob_tail': lambda u, v: np.max((1 - stats.norm.cdf(
                                        (explore_priority/10000) *
                                        np.ones(u.shape), u,
                                        np.sqrt(v))), axis=1),
        'sigmoid': lambda u, v: np.abs(stats.logistic.cdf(u + np.sqrt(v),
                                        loc=0.5,
                                        scale=explore_priority) -
                                       stats.logistic.cdf(u - np.sqrt(v),
                                        loc=0.5,
                                        scale=explore_priority)).sum(axis=1)
    }
