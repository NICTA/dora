import numpy as np
from .types import Folds
from dora.regressors.gp import predict
from dora.regressors.gp import linalg
from dora.regressors.gp import types
import sklearn.cluster as skcluster
import scipy.interpolate as interp
from scipy.spatial import Delaunay
import scipy.stats as stats
from scipy.linalg import solve_triangular
import scipy.linalg as la
from scipy.optimize import minimize
import copy


# Compute the log marginal likelihood
def negative_log_marginal_likelihood(Y, L, alpha):
    n = L.shape[0]
    t1 = np.dot(Y.ravel(), alpha.ravel())
    log_det_k = 2.*np.sum(np.log(np.diag(L)))
    nll = 0.5 * (t1 + log_det_k + n * np.log(2.0 * np.pi))
    return nll


# neCompute the leave one out neg log prob
def negative_log_prob_cross_val(Y, L, alpha):
    n = L.shape[0]
    Kinv = np.linalg.solve(L.T, solve_triangular(L, np.eye(n), lower=True))
    logprob = 0
    for i in range(n):
        Kinvii = Kinv[i][i]
        mu_i = Y[i] - alpha[i]/Kinvii
        sig2i = 1/Kinvii
        logprob += stats.norm.logpdf(Y[i], loc=mu_i, scale=sig2i)
    return -float(logprob)


# The inverse of opt_config_copys_to_vector - gets called a lot
def unpack(theta, unpackinfo):
    return [[theta[tup[0]].reshape(tup[1]) if tup[1]!=() else theta[tup[0]][0] for tup in item] for item in unpackinfo]


def make_folds(X, y, target_size, method='random'):
    n_Y = y.shape[0]
    n_folds = int(n_Y/target_size) + int(target_size>n_Y)

    if method == 'random':
        fold_assignment = np.random.permutation(n_Y)%n_folds
    elif method == 'cluster':
        # Thanks scikit
        print('Clustering [sklearn.cluster] inputs')
        clusterer = skcluster.MiniBatchKMeans(n_clusters=n_folds, batch_size=1000)
        fold_assignment = clusterer.fit_predict(X)
    elif method == 'rcluster':
        print('Clustering [sklearn.cluster] inputs')
        clusters = skcluster.MiniBatchKMeans(n_clusters=n_folds,
                                batch_size=1000, compute_labels=True).fit(X)
        Xcluster = clusters.cluster_centers_
        print('Interpolating probability')
        n_X = X.shape[0]
        assign_prob = np.zeros((n_folds, n_X))
        tris = Delaunay(Xcluster)
        base_labels = clusters.labels_
        for i in range(n_folds):
            indicator = np.zeros(n_folds)
            indicator[i] = 1.
            row = interp.LinearNDInterpolator(tris, indicator,
                                                         fill_value=-1)(X)
            row[row<0] = base_labels[row<0] == i
            assign_prob[i] = row

        # now use these as selection probabilities
        assign_prob = np.cumsum(assign_prob, axis=0)

        rvec = np.random.random(n_X)
        fold_assignment = np.sum(rvec[np.newaxis, :] <assign_prob, axis=0)

        # veryfy fold assignment?
        # pl.scatter(X[:, 0], X[:, 1], c=fold_assignment)
        # pl.show()
        # exit()

    else:
        raise NameError('Unrecognised fold method:'+method)

    fold_inds = np.unique(fold_assignment)
    folds = Folds(n_folds, [], [], [])  # might contain lists in the multitask case
    where = lambda y, v:y[np.where(v)[0]]
    for f in fold_inds:
        folds.X.append(where(X, fold_assignment==f))
        folds.Y.append(where(y, fold_assignment==f))
        folds.flat_y.append(where(y, fold_assignment==f))

    return folds

# Extended to allow lists of arrays of opt_config_copyeters for sigma, signal and noise so we can use multiple kernels etc
# unlike unpack, this doesn't need to be super efficient - its only called once
def pack(theta, noisepar):
    unpackinfo = [[], []]
    aopt_config_copys = []
    count = 0
    for ind, item in enumerate((theta, noisepar)):
        for value in item:
            aitem = np.array(value)
            newval = aitem.ravel()
            aopt_config_copys.append(newval)
            packshape = aitem.shape
            nextcount = count+newval.shape[0]
            unpackinfo[ind].append((list(range(count, nextcount)), packshape))  # had to make these lists for comptibility with python3.4
            count = nextcount
    opt_config_copys = np.concatenate(aopt_config_copys)
    return opt_config_copys, unpackinfo

def condition(X, y, kernelFn, hyper_opt_config_copys):
    assert len(y.shape) == 1    #y must be shapeless (n, )
    h_kernel, noise_std = hyper_opt_config_copys
    kernel = lambda x1, x2: kernelFn(x1, x2, h_kernel)
    noise_vector = predict.noise_vector(X, noise_std)
    L = linalg.cholesky(X, kernel, noise_vector)
    alpha = predict.alpha(y, L)
    return types.RegressionParams(X, L, alpha, kernel, y, noise_std)


def chol_up(L, Sn, Snn, Snn_noise_std_vec):
    # Incremental cholesky update
    Ln = la.solve_triangular(L, Sn, lower=True).T
    On = np.zeros(Ln.shape).T
    noise = np.diag(Snn_noise_std_vec ** 2)
    Lnn = linalg.jitchol(Snn+noise - Ln.dot(Ln.T))
    top = np.concatenate((L, On), axis=1)
    bottom = np.concatenate((Ln, Lnn), axis=1)
    return np.concatenate((top, bottom), axis=0)


def chol_up_insert(L, V12, V23, V22, Snn_noise_std_vec, insertionID):

    R = L.T
    N = R.shape[0]
    n = V22.shape[0]
    noise = np.diag(Snn_noise_std_vec ** 2)
    R11 = R[:insertionID, :insertionID]
    R33 = R[insertionID:, insertionID:]
    S11 = R11
    S12 = la.solve_triangular(R11.T, V12, lower=True)
    S13 = R[:insertionID, insertionID:]
    S22 = linalg.jitchol(V22+noise - S12.T.dot(S12)).T
    if V23.shape[1] != 0:  # The data is being inserted between columns
        S23 = la.solve_triangular(S22.T, (V23-S12.T.dot(S13)), lower=True)
        S33 = linalg.jitchol(R33.T.dot(R33)-S23.T.dot(S23)).T
    else:  #the data is being appended at the end of the matrix
        S23 = np.zeros((n, 0))
        S33 = np.zeros((0, 0))
    On1 = np.zeros((n, insertionID))
    On2 = np.zeros((N-insertionID, insertionID))
    On3 = np.zeros((N-insertionID, n))

    top = np.concatenate((S11, S12, S13), axis=1)
    middle = np.concatenate((On1, S22, S23), axis=1)
    bottom = np.concatenate((On2, On3, S33), axis=1)
    return np.concatenate((top, middle, bottom), axis=0).T

def chol_down(L, remIDList):
    # This works but it might potentially be slower than the naive approach of
    # recomputing the cholesky decomposition from scratch.
    # The jitchol line can apparently be replaces with a chol that exploits the
    # structure of the problem according to Osbourne's Thesis (as
    # cholupdate does).
    remIDList = np.sort(remIDList)
    for i in range(len(remIDList)):
        remID = remIDList[i]
        S = L.T
        n = S.shape[0]
        On = np.zeros((n-(remID+1), remID))
        # Incremental cholesky downdate
        top = np.concatenate((S[:remID, :remID], S[:(remID), (remID+1):]), axis=1)
        S23 = S[remID, (remID+1):][np.newaxis, :]
        S23TS23 = S23.T.dot(S23)
        S33TS33 = S[(remID+1):, (remID+1):].T.dot(S[(remID+1):, (remID+1):])
        R33 = linalg.jitchol(S23TS23+S33TS33).T
        bottom = np.concatenate((On, R33), axis=1)
        L = np.concatenate((top, bottom), axis=0).T
        remIDList -= 1
    return L


def add_data(newX, newY, regressor, query=None, insertionID=None):
    assert(isinstance(regressor, types.RegressionParams))
    assert(not query or isinstance(query, types.QueryParams))
    assert(len(newX.shape) == 2)
    assert(len(newY.shape) == 1)

    if not(insertionID):  #No insterionID provide. Append data to the end.
        # Compute the new rows and columns of the covariance matrix
        Kxn = regressor.kernel(regressor.X, newX)
        Knn = regressor.kernel(newX, newX)
        nn_noise_std = predict.noise_vector(newX, regressor.noise_std)
        # Update the regression opt_config_copys
        regressor.X = np.vstack((regressor.X, newX))
        regressor.y = np.hstack((regressor.y, newY))
        regressor.L = chol_up(regressor.L, Kxn, Knn,
                              nn_noise_std)
        # sadly, this is still expensive. However osborne's thesis appendix B can
        # be used to speed up this step too. Maybe by a factor of 2.
        regressor.alpha = predict.alpha(regressor.y, regressor.L)

        # Optionally update the query
        if query is not None:
            Kxsn = regressor.kernel(newX, query.Xs)
            query.K_xxs = np.vstack((query.K_xxs, Kxsn))
    else:
        # Compute the new rows and columns of the covariance matrix
        Kx1n = regressor.kernel(regressor.X[:insertionID, :], newX)
        Knx2 = regressor.kernel(newX, regressor.X[insertionID:, :])
        Knn = regressor.kernel(newX, newX)
        nn_noise_std = predict.noise_vector(newX, regressor.noise_std)
        regressor.X = np.vstack((regressor.X[:insertionID, :], newX,
                                 regressor.X[insertionID:, :]))
        regressor.y = np.hstack((regressor.y[:insertionID], newY,
                                 regressor.y[insertionID:]))
        regressor.L = chol_up_insert(regressor.L, Kx1n, Knx2, Knn,
                              nn_noise_std, insertionID)
        # sadly, this is still expensive. However osborne's thesis appendix B can
        # be used to speed up this step too. Maybe by a factor of 2.
        regressor.alpha = predict.alpha(regressor.y, regressor.L)

        if query is not None:
            Kxsn = regressor.kernel(newX, query.Xs)
            query.K_xxs = np.vstack((query.K_xxs[:insertionID, :], Kxsn,
                                     query.K_xxs[insertionID:, :]))


def remove_data(regressor, remID, query=None):
    assert(isinstance(regressor, types.RegressionParams))
    assert(not query or isinstance(query, types.QueryParams))


    regressor.X = np.delete(regressor.X, remID, axis=0)
    regressor.y = np.delete(regressor.y, remID, axis=0)
    # regressor.L = chol_down(regressor.L, remID)


    noise_vector = predict.noise_vector(regressor.X, regressor.noise_std)
    regressor.L = linalg.cholesky(regressor.X, regressor.kernel, noise_vector)
    regressor.alpha = predict.alpha(regressor.y, regressor.L)

    # Optionally update the query
    if query is not None:
        query.K_xxs = np.delete(query.K_xxs, remID, axis=0)


def learn(X, Y, cov_fn, optParams, optCrition='logMarg', returnLogMarg=False,
          verbose=False):
    # Normal criterion with all the data
    def criterion(sigma, noise):
        k = lambda x1, x2: cov_fn(x1, x2, sigma)
        X_noise = predict.noise_vector(X, noise)
        L = linalg.cholesky(X, k, X_noise)
        a = predict.alpha(Y, L)
        if optCrition == 'logMarg':
            val = negative_log_marginal_likelihood(Y, L, a)
        elif optCrition == 'crossVal':
            val = negative_log_prob_cross_val(Y, L, a)
        if verbose is True:
            print('['+str(val)+']  ', sigma, noise)
        return val

    sigma, noise, optval = optimise_hypers(criterion, optParams)

    if verbose:
        print('[',optval,']:', sigma, noise)

    if returnLogMarg:
        return sigma, noise, -optval
    else:
        return sigma, noise


def learn_folds(folds, cov_fn, optParams, optCrition='logMarg', verbose=False):
    # Same as learn, but using multiple folds jointly
    # todo: distribute computation!
    def criterion(sigma, noise):
        k = lambda x1, x2: cov_fn(x1, x2, sigma)
        val = 0
        for f in range(folds.n_folds):
            Xf = folds.X[f]
            Yf = folds.flat_y[f]
            Xf_noise = predict.noise_vector(Xf, noise)
            Lf = linalg.cholesky(Xf, k, Xf_noise)
            af = predict.alpha(Yf, Lf)
            if optCrition == 'logMarg':
                val += negative_log_marginal_likelihood(Yf, Lf, af)
            elif optCrition == 'crossVal':
                val += negative_log_prob_cross_val(Yf, Lf, af)
        if verbose is True:
            print('['+str(val)+']  ', sigma, noise)
        return val

    sigma, noise, optval = optimise_hypers(criterion, optParams)

    if verbose:
        print('[', optval, ']:', sigma, noise)
    return sigma, noise


def optimise_hypers(criterion, optParams):
    objective = lambda theta : criterion(*unpack(theta, unpackinfo))
    theta_low, _ = pack(optParams.sigma.lowerBound, optParams.noise.lowerBound)
    theta_0, unpackinfo = pack(optParams.sigma.initialVal, optParams.noise.initialVal)
    theta_high, _ = pack(optParams.sigma.upperBound, optParams.noise.upperBound)

    nParams = theta_0.shape[0]

    assert( (theta_low<=theta_0).all())
    assert( (theta_high>=theta_0).all())

    bounds = [a for a in zip(theta_low, theta_high)]
    theta_opt = minimize(objective, theta_0, method='L-BFGS-B', bounds=bounds)
    sigma, noise_sigma = unpack(theta_opt.x, unpackinfo)
    opt_val = theta_opt.fun
    return sigma, noise_sigma, opt_val


def batch_start(opt_config, initial_values):
    """
    Sets initial values of the optimiser parameters
    Returned as an OptConfig instance or a list of OptConfig instances

    Arguments:
        opt_config      : An instance of OptConfig
        initial_values  : List or np.array of initial parameters values
    Returns:
        batch_config    : A OptConfig instance or a list of OptConfig instances
    """
    if hasattr(initial_values[0], '__iter__'):
        batch_config = []
        for value in initial_values:
            opt_config_copy = copy.deepcopy(opt_config)
            opt_config_copy.sigma.initialVal = value
            batch_config.append(opt_config_copy)
    else:
        batch_config = copy.deepcopy(opt_config)
        batch_config.sigma.initialVal = initial_values
    return batch_config
