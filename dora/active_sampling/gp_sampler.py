"""
Gaussian Process Sampler Module.

Provides the Gaussian Process Sampler Class which contains the strategies for
active sampling a spatial field using a non-parametric, Bayesian model
"""
import logging

from dora.active_sampling import Sampler, random_sample

import revrand.legacygp as gp

import numpy as np

import scipy.stats as stats

log = logging.getLogger(__name__)


class GaussianProcess(Sampler):
    """
    GaussianProcess Class.

    Inherits from the Sampler class and augments pick and update with the
    mechanics of the GP method

    Attributes
    ----------
    kerneldef : function
        Kernel function definition. See the 'gp' module.
    n_min : int
        Number of training samples required before sampler can be trained
    acq_name : str
        A string specifying the type of acquisition function used
    explore_priority : float, optional
        The priority of exploration against exploitation
    hyperparams : numpy.ndarray
        The hyperparameters of the Gaussian Process Inference Model
    regressors : list
        List of regressor objects. See 'gp.types.RegressionParams'
    y_mean : float
        Mean of the training target outputs
    n_tasks : int
        Number of tasks or, equivalently, number of target outputs

    See Also
    --------
    Sampler : Base Class
    """

    def __init__(self, lower, upper, kerneldef=None, n_train=50,
                 acq_name='var_sum', explore_priority=0.0001):
        """
        Initialise the GaussianProcess class.

        .. note:: Currently only supports rectangular type restrictions on the
        parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        kerneldef : function
            Kernel function definition. See the 'gp' module.
        n_train : int
            Number of training samples required before sampler can be trained
        acq_name : str
            A string specifying the type of acquisition function used
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)

        self.kerneldef = kerneldef
        self.n_min = n_train
        self.acq_name = acq_name
        self.explore_priority = explore_priority
        self.hyperparams = None
        self.regressors = None
        self.y_mean = None
        self.n_tasks = None

    def update_y_mean(self):
        """
        Update the mean of the target outputs.

        .. note :: [Properties Modified]
                    y_mean,
                    n_tasks

        .. note :: At anytime, 'y_mean' should be the mean of all the output
                   targets including the virtual ones, since that is what
                   we are training upon
        """
        if not self.y:
            return
        self.y_mean = self.y().mean(axis=0) if len(self.y) else None

        # Make sure the number of stacks recorded is consistent
        if self.n_tasks is None:
            self.n_tasks = self.y_mean.shape[0]
        else:
            assert self.n_tasks == self.y_mean.shape[0]

    def learn_hyperparams(self, verbose=False, ftol=1e-15, maxiter=2000):
        """
        Learn the kernel hyperparameters from the data collected so far.

        Equivalent to training the Gaussian process used for the sampler
        The training result is summarised by the hyperparameters of the kernel

        .. note :: Learns common hyperparameters between all tasks

        .. note :: [Properties Modified]
                    (None)

        Returns
        -------
        list
            A list of hyperparameters with each element being the
            hyperparameters of each corresponding task
        """
        self.update_y_mean()

        logging.info('Training hyperparameters...')
        hyperparams = gp.learn(self.X(), self.y(), self.kerneldef,
                               optCriterion=gp.criterions.snlml,
                               verbose=verbose, ftol=ftol, maxiter=maxiter)
        logging.info('Done.')

        return [hyperparams for i_task in range(self.n_tasks)]

    def update_regressors(self):
        """
        Update the regressors of the Gaussian process model.

        Only makes sense to do this after hyperparameters are learned

        .. note :: [Properties Modified]
                    regressors

        .. note :: [Further Work] Use Cholesky Update here correctly to cache
                    regressors and improve efficiency
        """
        if self.hyperparams is None:
            return
            # raise ValueError('Hyperparameters are not learned yet.' +
            #                  'Regressors cannot be updated.')

        # Create the regressors if it hasn't already been
        self.regressors = []
        for i_task in range(self.n_tasks):
            self.regressors.append(
                gp.condition(self.X(), self.y()[:, i_task] -
                             self.y_mean[i_task],
                             self.kerneldef, self.hyperparams[i_task]))

    def train(self):
        """
        Train the Gaussian process model.

        A wrapper function that learns the hyperparameters and updates the
        regressors, which is equivalent to a fully trained model that is
        ready to perform Inference

        .. note :: [Properties Modified]
                    hyperparameters,
                    regressors
        """
        assert(self.dims is not None)

        if self.kerneldef is None:
            onez = np.ones(self.dims)
            self.kerneldef = lambda h, k: \
                h(1e-3, 1e+2, 1) * k('matern3on2',
                                     h(1e-2 * onez, 1e+3 * onez, 1e+0 * onez))
        # Learn hyperparameters
        self.hyperparams = self.learn_hyperparams()

        # Update the regressors
        self.update_regressors()

    def update(self, uid, y_true):
        """
        Update a job with its observed value.

        .. note :: [Properties Modified]
                    y,
                    virtual_flag,
                    y_mean,
                    regressors

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data buffer 'GaussianProcess.X' and
            'GaussianProcess.y' corresponding to the job being updated
        """
        y_true = np.atleast_1d(y_true)
        assert y_true.ndim == 1
        ind = self._update(uid, y_true)
        self.update_y_mean()
        self.update_regressors()
        return ind

    def pick(self, n_test=500):
        """
        Pick the next feature location for the next observation to be taken.

        .. note :: [Properties Modified]
                    X,
                    y,
                    virtual_flag,
                    pending_results,
                    y_mean,
                    hyperparameters,
                    regressors

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

        self.update_y_mean()

        # If we do not have enough samples yet, randomly sample for more!
        if n < self.n_min:

            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        else:

            if self.regressors is None:
                self.train()

            # Randomly sample the volume for test points
            Xq = random_sample(self.lower, self.upper, n_test)

            # Generate cached predictors for those test points
            predictors = [gp.query(r, Xq) for r in self.regressors]

            # Compute the posterior distributions at those points
            # Note: No covariance information implemented at this stage
            Yq_exp = np.asarray([gp.mean(p) for p in predictors]).T + \
                self.y_mean
            Yq_var = np.asarray([gp.variance(p) for p in predictors]).T

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

    def predict(self, Xq, real=True):
        """
        Predict the query mean and variance using the Gaussian process model.

            Infers the mean and variance of the Gaussian process at given
            locations using the data collected so far

        .. note :: [Properties Modified]
                    (None)

        Parameters
        ----------
        Xq : numpy.ndarray
            Query points
        real : bool
            To use only the real observations or also the virtual observations

        Returns
        -------
        numpy.ndarray
            Expectance of the prediction at the given locations
        numpy.ndarray
            Variance of the prediction at the given locations
        """
        assert self.hyperparams, "Sampler is not trained yet. " \
                                 "Possibly not enough observations provided."

        # To use only the real data, extract the real data and compute the
        # regressors using only the real data
        if real:
            X_real, y_real = self.get_real_data()
            regressors = [gp.condition(X_real, y_real[:, i_task] -
                          self.y_mean[i_task],
                          self.kerneldef, self.hyperparams[i_task])
                          for i_task in range(self.n_tasks)]

        # Otherwise, just use the regressors we already have
        else:
            regressors = self.regressors

        # Compute using the standard predictor sequence
        predictors = [gp.query(r, Xq) for r in regressors]
        yq_exp = [gp.mean(p) for p in predictors]
        yq_var = [gp.variance(p) for p in predictors]

        return np.asarray(yq_exp).T + self.y_mean, np.asarray(yq_var).T

    def set_kerneldef(self, kerneldef):
        """
        Set the current kernel definition for the Gaussian process model.

        Parameters
        ----------
        kerneldef : function
            The kernel definition from the gp module
        """
        assert callable(kerneldef)
        self.kerneldef = kerneldef

    def get_kerneldef(self):
        """
        Get the current kernel definition for the Gaussian process model.

        Returns
        -------
        function
            The kernel definition from the gp module
        """
        return self.kerneldef

    def print_kernel(self, kerneldef):
        """
        Print the current kernel for the Gaussian process model.

        .. note :: Not implemented yet
        """
        # TO DO: Use the printer method to print the current kernel!
        pass

    def set_hyperparams(self, hyperparams):
        """
        Set the hyperparameters for the Gaussian process model.

        If only one set of hyperparameters is given, all tasks will receive
        the same set of hyperparameters

        .. note :: update_regressors will be automatically called

        Parameters
        ----------
        hyperparams : list or numpy.ndarray
            The hyperparameter(s) of the Gaussian process model
        """
        if isinstance(hyperparams, list):
            self.hyperparams = hyperparams
        else:
            self.hyperparams = [hyperparams for i in range(self.n_tasks)]

        self.update_regressors()

    def get_hyperparams(self):
        """
        Get the hyperparameters for the Gaussian process model.

        Returns
        -------
        list
            The hyperparameters for each task
        """
        return self.hyperparams

    def set_acq_name(self, acq_name):
        """
        Set the acquisition function through a string.

        Parameters
        ----------
        acq_name : str
            The name of the acquisition function
        """
        assert type(acq_name) is str
        self.acq_name = acq_name

    def get_acq_func(self):
        """
        Get the acquisition function.

        Returns
        -------
        function
            The acquisition function
        """
        return acq_defs(y_mean=self.y_mean,
                        explore_priority=self.explore_priority)[self.acq_name]

    def set_explore_priority(self, explore_priority):
        """
        Set the exploration priority of the active sampler.

        Parameters
        ----------
        explore_priority : float
            The exploration priority of the active sampler
        """
        self.explore_priority = explore_priority

    def get_explore_priority(self):
        """
        Get the exploration priority of the active sampler.

        Returns
        -------
        float
            The exploration priority of the active sampler
        """
        return self.explore_priority

    def set_min_training_size(self, n_min):
        """
        Set the minimum training size of the active sampler.

        Parameters
        ----------
        n_min : int
            The minimum training size of the active sampler
        """
        self.n_min = n_min

    def get_min_training_size(self):
        """
        Get the minimum training size of the active sampler.

        Returns
        -------
        int
            The minimum training size of the active sampler
        """
        return self.n_min

    def get_real_data(self):
        """
        Obtain the observed data.

        This excludes all the virtual data.

        Returns
        -------
        numpy.ndarray
            The observed feature locations
        numpy.ndarray
            The observed target outputs
        """
        assert self.X
        assert self.y

        real_flag = ~self.virtual_flag()
        return self.X()[real_flag], self.y()[real_flag]


def acq_defs(y_mean=0, explore_priority=0.0001):
    """
    Generate a dictionary of acquisition functions.

    Parameters
    ----------
    y_mean : int or np.ndarray
        The mean of the target outputs
    explore_priority : float
        Exploration priority against exploitation

    Returns
    -------
    dict
        A dictionary of acquisition functions to be used for the GP Sampler
    """
    # Aquisition Functions
    # u: Mean matrix (n x n_tasks)
    # v: Variance matrix (n x n_tasks)
    # Returns an array of n values
    return {
        'var_sum': lambda u, v: np.sum(v, axis=1),
        'pred_max': lambda u, v: np.max(u + 3 * np.sqrt(v), axis=1),
        'prod_max': lambda u, v: np.max((u + (y_mean +
                                        explore_priority / 3.0)) *
                                        np.sqrt(v), axis=1),
        'prob_tail': lambda u, v: np.max((1 - stats.norm.cdf(
                                         explore_priority *
                                         np.ones(u.shape), u,
                                         np.sqrt(v))), axis=1),
        'sigmoid': lambda u, v: np.abs(stats.logistic.cdf(u + np.sqrt(v),
                                       loc=0.5,
                                       scale=explore_priority) -
                                       stats.logistic.cdf(u - np.sqrt(v),
                                       loc=0.5,
                                       scale=explore_priority)).sum(axis=1)
    }
