import numpy as np
import scipy.linalg as la
import logging
log = logging.getLogger(__name__)

def ndY(Y2d):
    nPoints = Y2d.shape[0]
    nTasks = Y2d.shape[1]
    Ylist = [Y2d[:,i] for i in range(nTasks)]
    Y = np.concatenate(tuple(Ylist),axis=0)
    return Y


def ndX(X, nTasks):
    nPoints = X.shape[0]
    Xlist = [X for i in range(nTasks)]
    Ilist = [np.ones(nPoints) * i for i in range(nTasks)]
    X = np.concatenate(tuple(Xlist),axis=0)
    I = np.concatenate(tuple(Ilist),axis=0)
    X = np.hstack((I[:,np.newaxis],X[:,np.newaxis]))
    return X

def jitchol(X, overwrite_a = False, check_finite = True):

    """Add jitter until a positive definite matrix occurs"""
    n = X.shape[0]
    I = np.eye(n)
    jitter = 1e-8
    max_jitter = 1e10
    L = None
    X_dash = X

    while jitter < max_jitter:
        try:
            L = la.cholesky(X_dash, lower = True, 
                overwrite_a = overwrite_a, check_finite = check_finite)
            break
        except la.LinAlgError:
            X_dash = X + jitter * I
            jitter *= 2.0
            log.warning('Jitter added. Amount: %f!' % jitter)

    if jitter > 1e-2:
        log.warning('Rather large jitchol of %f!' % jitter)

    if L is not None:
        return L
    else:
        raise la.LinAlgError("Max value of jitter reached")

def cholesky(X, kernelfn, sigma_noise):
    K = kernelfn(X, X)
    noise = np.diag(sigma_noise ** 2)
    L = jitchol(K + noise)
    # L = np.eye(X.shape[0])
    # try:
        # L = la.cholesky(K + noise, lower=True)
    # except:
        # print("CHOLESKY ERROR")
    return L 

def choleskyjitter(A, overwrite_a = False, check_finite = True):

    """Add jitter stochastically until a positive definite matrix occurs"""
    # Avoid preparing for jittering if we can already find the cholesky 
    # with no problem
    try:
        return la.cholesky(A, lower = True, overwrite_a = overwrite_a, 
                check_finite = check_finite)
    except Exception:
        pass

    # Prepare for jittering (all the magic numbers here are arbitary...)
    n = A.shape[0]
    maxscale = 1e10
    minscale = 1e-4
    scale = minscale

    # Keep jittering stochastically, increasing the jitter magnitude along 
    # the way, until it's all good
    while scale < maxscale:

        try:
            jitA = scale * np.diag(np.random.rand(n))
            L = la.cholesky(A + jitA, lower = True, overwrite_a = overwrite_a, 
                check_finite = check_finite)
            return L
        except la.LinAlgError:
            scale *= 1.01
            log.warning('Jitter added stochastically. Scale: %f!' % scale)

    raise la.LinAlgError("Max value of jitter reached")