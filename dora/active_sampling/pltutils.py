"""
Utilities for plotting purposes.
"""
import revrand.legacygp as gp

import matplotlib.pyplot as pl

import numpy as np


def split_subplots(n):
    """
    Automatically generate a suitable subplot layout.

    Parameters
    ----------
    n : int
        The number of subplots required

    Returns
    -------
    int
        The number of rows required
    int
        The number of columns required
    """
    n_sqrt = np.sqrt(n)
    c = np.ceil(n_sqrt).astype(int)
    r = np.floor(n_sqrt).astype(int)

    while c * r < n:
        r += 1

    return r, c


def plot_sampler_progress(sampler, ax=None):
    """
    Plot the progress of a particular sampler.

    .. note :: Only works if n_dims = 2 and n_tasks = 1

    Parameters
    ----------
    sampler : Sampler
        An instance of the sampler class or its subclasses
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes or subplot the plotting is to be performed on
    """
    assert sampler.dims == 2
    assert sampler.n_tasks == 1

    X = np.asarray(sampler.X)
    y = np.asarray(sampler.y).flatten()
    virtual_flags = np.asarray(sampler.virtual_flag)

    X_real = X[~virtual_flags]
    y_real = y[~virtual_flags]
    X_virtual = X[virtual_flags]
    y_virtual = y[virtual_flags]

    n_grids = 400
    xi = np.linspace(sampler.lower[0], sampler.upper[0], num=n_grids)
    yi = np.linspace(sampler.lower[1], sampler.upper[1], num=n_grids)

    xg, yg = np.meshgrid(xi, yi)
    X_test = np.array([xg.flatten(), yg.flatten()]).T
    predictor = gp.query(sampler.regressors[0], X_test)
    zg = np.reshape(gp.mean(predictor), xg.shape)

    if ax is None:
        ax = pl.gca()

    extent = [sampler.lower[0], sampler.upper[0],
              sampler.upper[1], sampler.lower[1]]
    ax.imshow(zg, extent=extent)
    ax.scatter(X_real[:, 0], X_real[:, 1], c=y_real)
    ax.scatter(X_virtual[:, 0], X_virtual[:, 1], c=y_virtual, marker='x')
    ax.set_title('%d Samples' % len(y))
    ax.axis('image')
