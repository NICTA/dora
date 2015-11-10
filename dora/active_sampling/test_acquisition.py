""" This code tests out a GP acquisition function under synthetic conditions
    Alistair Reid 2015
"""

import numpy as np
import computers.gp as gp
import matplotlib.pyplot as pl
import scipy.linalg as la
from mpl_toolkits.mplot3d import Axes3D


def main():
    # 2D input space and grid
    truth = lambda x1, x2: np.maximum(0, np.cos(5.*x1) * np.cos(5.*x2) *
                                      np.exp(-(x1**2 + x2**2)/2.))
    qx, qy = np.mgrid[-1:1:.02, -1:1:.02]
    Xq = np.vstack((qx.ravel(), qy.ravel())).T
    extent = [-1, 1, -1, 1]
    nq = len(Xq)
    values = truth(qx, qy)

    # Acquire samples - gridded in this case
    # X = np.mgrid[-1:1:5j, -1:1:5j].reshape((2, -1)).T
    # X = np.array([[-1, -1]])  # 
    X = 2.*np.random.random(size=(10, 2)) - 1.
    y = truth(X[:, 0], X[:, 1])
    n = X.shape[0]

    # Learn (m r specify) a kernel
    LS = 0.2
    SF = 0.2
    noise = [0.01]  # This is noise STANDARD DEVIATION
    kerneldef = lambda h, k: SF*k('gaussian', LS)
    k = gp.kernel.compose(kerneldef)

    for itr in range(80):
        # Use a GP regressor to estimate the upper envelope function value
        regressor = gp.condition(X, y, k, [[], noise])
        query = gp.query(Xq, regressor)
        mu = gp.mean(regressor, query)
        std = np.sqrt(gp.variance(regressor, query))
        maxvals = mu+2.*std  # upper bound estimate
        muval = (mu - mu.min())/(mu.max() - mu.min())
        sampvals = muval * std

        # which is the best sample location
        indx = np.argmax(sampvals)
        new_samp = Xq[indx]

        # take the sample
        n += 1
        X = np.vstack((X, new_samp))
        y = np.hstack((y, truth(new_samp[0], new_samp[1])))

        if np.mod(itr, 15) == 0:
            maxvals = maxvals.reshape(qx.shape).T[::-1]
            sampvals = sampvals.reshape(qx.shape).T[::-1]
            mu = mu.reshape(qx.shape).T[::-1]

            crange = [0., np.max(maxvals)]
            pl.figure()
            ax = pl.subplot(131)
            pl.imshow(values, extent=extent)
            pl.clim(crange)
            pl.plot(X[:, 0], X[:, 1], 'k.')
            pl.title('Truth')
            pl.subplot(132, sharex=ax, sharey=ax)
            pl.imshow(mu, extent=extent)
            pl.clim(crange)
            pl.plot(X[:, 0], X[:, 1], 'k.')
            pl.title('Mean')
            pl.subplot(133, sharex=ax, sharey=ax)
            pl.imshow(sampvals, extent=extent)
            # pl.colorbar()
            pl.plot(X[:, 0], X[:, 1], 'k.')
            pl.plot(new_samp[0], new_samp[1], 'bo')
            pl.title('Acquisition function %d'%itr)
    pl.show()
    exit()

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    assert(isinstance(ax, Axes3D))
    ax.plot_surface(qx, qy, values)
    pl.show()


def chol_up(L0, Sn, Snn):
    # Incremental cholesky update
    # Special version for scalars
    n = L0.shape[0]
    L = np.zeros((n+1, n+1))
    L[:-1, :-1] = L0
    Ln = la.solve_triangular(L0, Sn[:, np.newaxis], lower=True)[:, 0]
    L[n, :-1] = Ln
    L[n, n] = np.sqrt(np.maximum(0, Snn - Ln.dot(Ln.T)))

    return L


    # sampvals = np.zeros(maxvals.shape)
    # regressor = gp.condition(X, y, k, [[], [0.2]])  # build a regressor
    # current_alph = np.sum(regressor.alpha)
    # L = regressor.L
    # base_variance = regressor.L[0, 0]
    # for i in range(nq):
    #     # use rank-1 updates for each of these...
    #     kn = query.K_xxs[:, i]  # borrow precomputed from query
    #     knn = base_variance  # ONLY FOR STATIONARY
    #     new_y = np.hstack((y, maxvals[i]))
    #     new_L = chol_up(L, kn, knn)
    #     alph_new = gp.alpha(new_y, new_L)
    #     sampvals[i] = np.sum(alph_new) - current_alph
    # maxvals = maxvals.reshape(qx.shape)
    # sampvals = sampvals.reshape(qx.shape)
    # mu = mu.reshape(qx.shape)



if __name__ == '__main__':
    main()


