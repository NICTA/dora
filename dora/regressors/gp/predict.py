import numpy as np
import scipy.linalg as la
from computers.gp import linalg
from computers.gp import types
from time import time

def alpha(Y, L):
    result = la.solve_triangular(
            L.T, la.solve_triangular(L,Y, lower=True, check_finite=False),
                check_finite=False)
    return result


# Mux and demux are mainly used for multi-task problems
# They convert lists to tagged vectors and vice versa
def mux(X_list, y_list=None):
    demuxinfo = [x.shape[0] for x in X_list]
    X = np.vstack(X_list)
    label = np.concatenate([d*np.ones(demuxinfo[d])
        for d in range(len(X_list))])
    X = np.hstack((X, label[:,np.newaxis]))

    if y_list is None:
        return X
    else:
        y = np.concatenate(y_list)
        return X,y

def demux(y, X):
    n_tasks = int(X[-1,-1]) +1
    Xv = X[:,:-1]
    label = X[:,-1]
    #X_list = [Xv[label==i] for i in range(n_tasks)]
    y_list = [y[label==i] for i in range(n_tasks)]
    return y_list


def noise_vector(X, noise_params):
    if type(X) is list:
        # multi-task
        result = np.concatenate([noise_params[i]*np.ones(X[i].shape[0]) 
            for i in range(len(X))])
    else:
        # single task
        result = np.ones(X.shape[0])*noise_params[0]
    assert(result.ndim == 1)
    return result

def mean(regressor, query):
    return np.dot(query.K_xxs.T, regressor.alpha)

def covariance(regressor, query):
    K_xs = regressor.kernel(query.Xs, query.Xs)  # matrix
    v = la.solve_triangular(regressor.L, query.K_xxs,
                                lower=True, check_finite=False)
    return K_xs - np.dot(v.T, v)

def variance(regressor, query):
    K_xs = regressor.kernel(query.Xs, None)  # vector
    v = la.solve_triangular(regressor.L, query.K_xxs,
                                lower=True, check_finite=False)
    result = K_xs - np.sum(v**2, axis=0)
    return result

def query(Xs, p):
    K_xxs = p.kernel(p.X, Xs)
    return types.QueryParams(Xs, K_xxs)


# # TODO: These are actually draws from a Gaussian and more general than a GP
# def draws(ndraws, mean, cov):
#     L_s = linalg.jitchol(cov)
#     draws = []
#     for i in range(ndraws):
#         norms = np.random.normal(loc=0.0,scale=1.0,size=(mean.shape[0],1))
#         c = np.dot(L_s, norms).ravel()
#         d = c + mean
#         draws.append(d)
#     return draws

# Proposed change to the above 'draws' function   -Kelvin
def draws(ndraws, exp, cov):
    
    # S: Standard Draw
    # C: Transform to include covariance
    # D: Transform to include expectance
    L = linalg.jitchol(cov)
    S = np.random.normal(loc = 0.0, scale = 1.0, size = (exp.shape[0], ndraws))
    C = np.dot(L, S)
    D = (C.T + exp)
    return D
