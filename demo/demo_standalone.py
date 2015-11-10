"""
Active Sampling Demo

This demo uses the Dora active sampling module as a python library that can be
directly called from within the process.

We solve a two dimensional sampling problem using an efficient sampling
scheme. The underlying problem is to sample the boundaries of a circle, given
no prior knowledge of its geometry.

"""

import numpy as np
import logging
import matplotlib.pyplot as pl
import matplotlib as mpl
import dora.active_sampling as sampling
import dora.regressors.gp as gp

# The plotting subpackage is throwing FutureWarnings
import warnings
warnings.simplefilter("ignore", FutureWarning)


def simulate_measurement(X):
    """ A binary image of a circle as a test problem for sampling
    """
    return (np.sum((X-0.5)**2) < 0.1).astype(float)


def main(method_name):

    # Set up a sampling problem:
    target_samples = 501
    lower = [0, 0]
    upper = [1, 1]

    if method_name == 'Delaunay':
        explore_priority = 0.0001  # relative to the *difference* in stdev
        sampler = sampling.Delaunay(lower, upper, explore_priority)

    elif method_name == 'Gaussian_Process':
        n_initial_sample = 50
        X_train = sampling.random_sample(lower, upper, n_initial_sample)
        y_train = np.asarray([simulate_measurement(i) for i in X_train])
        sampler = sampling.Gaussian_Process(lower, upper, X_train, y_train,
                                            add_train_data=False)
    else:
        raise ValueError('Unrecognised method name.')

    # Set up plotting:
    plots = {'fig': pl.figure(),
             'count': 0,
             'shape': (2, 3)}
    plot_triggers = [8, 9, 10, 50, 100, target_samples-1]

    # Run the active sampling:
    for i in range(target_samples):

        newX, newId = sampler.pick()

        observation = simulate_measurement(newX)

        sampler.update(newId, observation)

        if i in plot_triggers:
            plot_progress(plots, sampler)

    pl.show()


def plot_progress(plots, sampler):
    fig = plots['fig']
    cols = pl.cm.jet(np.linspace(0, 1, 64))
    custom = mpl.colors.ListedColormap(cols*0.5+0.5)
    fig.add_subplot(*(plots['shape'] + (1+plots['count'],)))
    plots['count'] += 1
    y = np.asarray(sampler.y)
    w = 4./np.log(1 + len(y))

    if isinstance(sampler, sampling.Delaunay):
        # todo (AL): only show the measured samples!
        X = np.asarray(sampler.X)

        pl.tripcolor(X[:, 0], X[:, 1], y, shading='gouraud', edgecolors='k',
                     linewidth=w, cmap=custom)
        pl.triplot(X[:, 0], X[:, 1], color='k', linewidth=w)

    elif isinstance(sampler, sampling.Gaussian_Process):
        X = sampler.regressor.X
        minv = np.min(X, axis=0)
        maxv = np.max(X, axis=0)
        res = 400
        xi = np.linspace(minv[0], maxv[0], res)
        yi = np.linspace(minv[1], maxv[1], res)
        xg, yg = np.meshgrid(xi, yi)
        x_test = np.array([xg.flatten(), yg.flatten()]).T
        query = gp.query(x_test, sampler.regressor)

        zi = np.reshape(gp.mean(sampler.regressor, query), xg.shape)

        extent = [np.min(X, axis=0)[0], np.max(X, axis=0)[0],
                  np.max(X, axis=0)[0], y.min()]
        pl.imshow(zi, vmin=0, vmax=1, extent=extent)

    else:
        raise(ValueError("Unsupported Sampler!"))

    pl.scatter(X[:, 0], X[:, 1], c=y)
    pl.axis('image')
    pl.title('%d Samples' % len(y))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main('Delaunay')
    main('Gaussian_Process')
