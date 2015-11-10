""" Kernel module
Defines basic kernel functions of the form:
  funcname(x_p, x_q, par) that return a covariance matrix.
  x_p is n1*d, x_q is n2*d, the result should be n1*n2
  par can be a scalar, array, or list of these.

All kernels must allow x_q=None, and efficiently compute the diagonal
of K(x_p, x_p) as a (n1,) shaped vector.

Multi-task kernels must begin with mt_, and such kernels must use the
last dimension of x_p and x_q as an indicator of the task.

This file also contains code for composing these kernels into multi-use
objects.
"""
import numpy as np
from scipy.spatial.distance import cdist
from dora.regressors.gp import predict
from dora.regressors.gp.types import Range
from dora.regressors.gp import train
import logging
log = logging.getLogger(__name__)


def gaussian(x_p, x_q, LS):
    # The 'squared exponential' gaussian radial basis function kernel.
    # This kernel is known to be smooth, differentiable and stationary.
    if x_q is None:
        return np.ones(x_p.shape[0])
    deltasq = cdist(x_p/LS, x_q/LS, 'sqeuclidean')
    value = np.exp(-0.5 * deltasq)
    return value


def laplace(x_p, x_q, LS):
    if x_q is None:
        return np.ones(x_p.shape[0])
    deltasq = cdist(x_p / np.sqrt(LS), x_q / np.sqrt(LS), 'sqeuclidean')
    value = np.exp(- deltasq)
    return value


def sin(x_p, x_q, params):
    # The gaussian-enveloped sinusoidal kernel is good for modeling locally
    # oscillating series.
    if x_q is None:
        return np.ones(x_p.shape[0])
    freq, LS = params
    deltasq = cdist(x_p/LS, x_q/LS, 'sqeuclidean')
    value = np.exp(-0.5 * deltasq)*np.cos(np.sqrt(deltasq))
    return value


def matern3on2(x_p, x_q, LS):
    # The Matern 3/2 kernel is often used as a less smooth alternative to the
    # gaussian kernel for natural data.
    if x_q is None:
        return np.ones(x_p.shape[0])

    r = cdist(x_p/LS, x_q/LS, 'euclidean')
    value = (1.0 + r)*np.exp(-r)
    return value


def chisquare(x_p, x_q, eps=1e-5):

    if x_q is None:
        return np.ones(x_p.shape[0])

    x_pd = x_p[:, np.newaxis, :]
    x_qd = x_q[np.newaxis, :, :]

    return 2 * (x_pd * x_qd / (x_pd + x_qd + eps)).sum(axis=-1)


def non_stationary(x_p, x_q, params):
    """ Implementation of Paciorek's kernel where length scale is defined as
    a continuous function L(x), and computed by operations on L(x1) and L(x2)

    Note - we globally apply ARD scaling, then inside the scaled space apply an
    isotropic non-stationary treatment according to L(x)

    Arguments:
     x_p, x_q : n*d x-values
     params(list): [ (d,) np array of length-scale multipliers scaling the...,
                     function L(x) ]

    """
    assert(x_p.ndim == 2)
    if x_q is None:
        # Just want the magnitude to be evaluated - its always 1 :)
        return np.ones(x_p.shape[0])

    LS_mult, LS_func = params  # unpack parameters
    dims = x_p.shape[1]

    ls_p = LS_func(x_p)
    ls_q = LS_func(x_q)

    if dims > 1:
        assert(LS_mult.shape[0] == dims)
        assert(len(LS_mult.shape) == 1)

    assert(len(ls_p.shape) == 1)

    ls_p = ls_p[:, np.newaxis]
    ls_q = ls_q[:, np.newaxis]

    sig_mult = LS_mult**2
    sig_p = ls_p**2
    sig_q = ls_q**2

    sig_avg = 0.5*(sig_p + sig_q.T)
    ls_avg = sig_avg**0.5

    # In the diagonal [axis aligned] case, the determinants of the length
    # scales are simply their products
    det_sig_mult = np.prod(sig_mult)
    # todo(AL) can probably remove det_sig_mult altogether as it should cancel
    dets_p = det_sig_mult * sig_p**dims
    dets_q = det_sig_mult * sig_q**dims
    dets_avg = det_sig_mult * sig_avg**dims

    # Compute non-stationary kernel
    gain = np.sqrt(np.sqrt(dets_p * dets_q.T) / dets_avg)

    return gain * np.exp(-cdist(x_p/LS_mult, x_q/LS_mult, 'sqeuclidean')
                         / ls_avg**2)


def tree1D(x_p, x_q, params):
    """ Implementation of Paciorek's kernel where length scale is defined as
    a continuous function L(x), and computed by operations on L(x1) and L(x2)

    Note - we globally apply ARD scaling, then inside the scaled space apply an
    isotropic non-stationary treatment according to L(x)

    Arguments:
     x_p, x_q : n*d x-values
     params(list): [ (d,) np array of length-scale multipliers scaling the...,
                     function L(x) ]

    """
    assert(x_p.ndim == 2)
    if x_q is None:
        # Just want the magnitude to be evaluated - its always 1 :)
        return np.ones(x_p.shape[0])

    LS_mult, LS_func = params  # unpack parameters
    dims = x_p.shape[1]

    ls_p = LS_func(x_p)
    ls_q = LS_func(x_q)

    if dims > 1:
        assert(LS_mult.shape[0] == 1)
        assert(len(LS_mult.shape) == 1)

    assert(len(ls_p.shape) == 1)

    ls_p = ls_p[:, np.newaxis]
    ls_q = ls_q[:, np.newaxis]

    sig_mult = LS_mult**2
    sig_p = ls_p**2
    sig_q = ls_q**2

    sig_avg = 0.5*(sig_p + sig_q.T)
    ls_avg = sig_avg**0.5

    # In the diagonal [axis aligned] case, the determinants of the length
    # scales are simply their products
    det_sig_mult = np.prod(sig_mult)
    # todo(AL) can probably remove det_sig_mult altogether as it should cancel
    dets_p = det_sig_mult * sig_p**dims
    dets_q = det_sig_mult * sig_q**dims
    dets_avg = det_sig_mult * sig_avg**dims

    # Compute non-stationary kernel
    gain = np.sqrt(np.sqrt(dets_p * dets_q.T) / dets_avg)

    return gain * np.exp(-cdist(x_p[:, -1][:, np.newaxis]/LS_mult,
        x_q[:, -1][:, np.newaxis]/LS_mult, 'sqeuclidean') / ls_avg**2)  # NOQA


def tree(x_p, x_q, params):
    """ Implementation of Paciorek's kernel where length scale is defined as
    a continuous function L(x), and computed by operations on L(x1) and L(x2)

    Note - we globally apply ARD scaling, then inside the scaled space apply an
    isotropic non-stationary treatment according to L(x)

    Arguments:
     x_p, x_q : n*d x-values
     params(list): [ (d,) np array of length-scale multipliers scaling the...,
                     function L(x) ]

    """
    assert(x_p.ndim==2)
    if x_q is None:
        # Just want the magnitude to be evaluated - its always 1 :)
        return np.ones(x_p.shape[0])

    LS_mult, LS_func = params  # unpack parameters
    dims = x_p.shape[1]

    ls_p = LS_func(x_p)
    ls_q = LS_func(x_q)

    if dims > 1:
        assert(LS_mult.shape[0] == dims)
        assert(len(LS_mult.shape) == 1)

    assert(len(ls_p.shape) == 1)

    ls_p = ls_p[:, np.newaxis]
    ls_q = ls_q[:, np.newaxis]

    sig_mult = LS_mult**2
    sig_p = ls_p**2
    sig_q = ls_q**2

    sig_avg = 0.5*(sig_p + sig_q.T)
    ls_avg = sig_avg**0.5

    # In the diagonal [axis aligned] case, the determinants of the length
    # scales are simply their products
    det_sig_mult = np.prod(sig_mult)
    # todo(AL) can probably remove det_sig_mult altogether as it should cancel
    dets_p = det_sig_mult * sig_p**dims
    dets_q = det_sig_mult * sig_q**dims
    dets_avg = det_sig_mult * sig_avg**dims

    # Compute non-stationary kernel
    gain = np.sqrt(np.sqrt(dets_p * dets_q.T) / dets_avg)

    return gain * np.exp(-cdist(x_p / LS_mult, x_q / LS_mult, 'sqeuclidean')
                         / ls_avg**2)


def nonstat_rr(x_p, x_q, params):
    # A non-stationary implementaiton of Paciorek's kernel
    # where the length scale is computed via a ridge regression:
    # The hyper-parameters are given by:
    # [LS_sigma, LS_mu, LS_noise, LS_x, LS_y0]
    # Where LS_x, LS_y0 are the control points of the latent regression
    # LS_mu is the prior on the length scale
    # LS_sigma is the length scale of the length-scale regression
    # LS_noise is the noise level of the length-scale regression

    min_ls = 1e-3
    assert(x_p.ndim == 2)
    dims = x_p.shape[1]

    if x_q is None:
        # Just want the magnitude to be evaluated - its always 1 :)
        return np.ones(x_p.shape[0])

    # TODO(Al): allow anisotropy
    LS_sigma, LS_mu, LS_noise, LS_x, LS_y0 = params
    LS_y = LS_y0 - LS_mu
    LS_kernel = gaussian  # lambda x1, x2: gaussian(x1, x2, LS_sigma)
    lsgp = train.condition(LS_x, LS_y, LS_kernel, [LS_sigma, [LS_noise]])
    query_p = predict.query(x_p, lsgp)
    query_q = predict.query(x_q, lsgp)
    ls_p = (min_ls + (LS_mu+predict.mean(lsgp, query_p))**2)[:, np.newaxis]
    ls_q = (min_ls + (LS_mu+predict.mean(lsgp, query_q))**2)[:, np.newaxis]

    sig_p = ls_p**2
    sig_q = ls_q**2

    sig_avg = 0.5*(sig_p + sig_q.T)
    # ls_avg = sig_avg**0.5
    dets_p = sig_p**dims
    dets_q = sig_q**dims
    dets_avg = sig_avg**dims

    # Compute non-stationary kernel
    gain = np.sqrt(np.sqrt(dets_p * dets_q.T) / dets_avg)

    return gain * np.exp(-cdist(x_p, x_q, 'sqeuclidean') / sig_avg)


def mt_weights(x_p, x_q, params):
    # This is an example of a multi-task aware kernel.
    # In this case, it is a positive definite per-task weight matrix.
    # TODO(Al): more intuitive weight parametrisation
    task_p = x_p[:, -1].astype(int)
    nTasks = task_p[-1]+1
    tri = np.tri(nTasks)
    tri[tri > 0] = params
    task_weights = np.dot(tri, tri.T)
    if x_q is None:  # Just the diagonal
        return task_weights[task_p, task_p]
    else:  # Normal
        task_q = x_q[:, -1].astype(int)
        return task_weights[task_p][:, task_q]


# The second half of this file deals with the infrastructure for composing
# kernels automatically from functions.
def named_target(covfn, fn_cache):
    # Turns a string into a callable function
    # also, automatically allows use of non multitask functions
    # as multitask functions.
    if covfn in fn_cache:
        return fn_cache[covfn]
    else:
        knowns = globals()
        if covfn in knowns:
            fn = knowns[covfn]
        elif covfn[:3] == 'mt_' and covfn[3:] in knowns:
            target = knowns[covfn[3:]]
            fn = lambda x_p, x_q, par: target(
                x_p[:, :-1], x_q[:, :-1] if x_q is not None else None, par)
        else:
            raise ValueError("No valid target")
        # logging.info('Binding %s to %s.' % (covfn, str(fn)))
        fn_cache[covfn] = fn
        return fn


def compose(user_kernel):
    # user_kernel is fn(h,k)
    # h - Hyperparameter function (min, max, mid)
    # k - Kernel call function (name, hyper, optional_list_of_dimensions)
    fn_cache = {}

    def thekernel(x1, x2, thetas):
        theta_iter = iter(thetas)
        return user_kernel(lambda a, b, c=None: next(theta_iter), 
            lambda k, par, d=None:
            (named_target(k, fn_cache)(x1, x2, par)) if d is None else
             named_target(k, fn_cache)(x1[:, d],
                                    None if x2 is None else x2[:,d], par))
    return thekernel


def auto_range(user_kernel):
    mins = []
    mids = []
    maxs = []
    def range_log(min, max, mid=None):
        if mid is None:
            mid = 0.5*(min+max)
        mins.append(min)
        mids.append(mid)
        maxs.append(max)
        return 0.
    user_kernel(range_log, lambda k,par,d=None:0.)
    return Range(mins, maxs, mids)

def describer(user_kernel):
    def theprinter(thetas):
        theta_iter = iter(thetas[0]) # assuming noise hyperparams included
        return str(user_kernel(
                lambda a, b, c=None: next(theta_iter),
                lambda k, par, d=None: Printer(k+'{'+Printer.txt(par)+'} ')))
    return theprinter
    # TODO(Al): re-implement special case for mt_weights with stub evaluation
    #     if covfn == 'mt_weights':
    #         n_tasks = np.floor(np.sqrt(2.0*len(params)))
    #         placeholder = np.arange(n_tasks)[:,np.newaxis]
    #         weights = globals()[covfn](placeholder, placeholder, params)
    #         return Printer(np.array_str(weights).replace('\n', ','))




# Return object for turning a covariance function call into a string.
# Currently supports addition and subtraction
# TODO(Al): tagging of hyperparameters
class Printer:
    def __init__(self,val='?'):
        self.val = val

    def __mul__(a, b):
        txta = Printer.txt(a)
        txtb = Printer.txt(b)
        if '+' in txta:
            txta = '('+txta+')'
        if '+' in txtb:
            txtb = '('+txtb+')'
        return Printer(txta+'*'+txtb)

    def __str__(self):
        return self.val

    def __add__(a, b):
        return Printer(Printer.txt(a)+'+'+Printer.txt(b))

    def __rmul__(b, a):
        return Printer.__mul__(a,b)

    def __radd__(b, a):
        return Printer.__add__(a,b)

    @staticmethod
    def txt(params):
        if type(params) is list:
            a = '['
            for p in params:
                a += Printer.txt(p)+', '
            a += ']'
            return a
        if type(params) == np.ndarray and params.ndim > 1 and params.shape[1] == 1:
            params = params.T[0]
        if type(params) is float or type(params) is np.float64:
            return "{:.3f}".format(params)
        else:
            return str(params)

