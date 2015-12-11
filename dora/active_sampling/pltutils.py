import numpy as np
import matplotlib.pyplot as pl
import dora.regressors.gp as gp


def split_subplots(n):
    """
    Automatically generates a suitable subplot layout
    """
    n_sqrt = np.sqrt(n)
    c = np.ceil(n_sqrt).astype(int)
    r = np.floor(n_sqrt).astype(int)

    while c * r < n:
        r += 1

    return r, c


def plot_sampler_progress(sampler, ax = None):
    """
    .. note : Only if n_dims = 2 and n_stack = 1
    """

    assert sampler.n_dims == 2
    assert sampler.n_stacks == 1

    X = np.asarray(sampler.X)
    y = np.asarray(sampler.y).flatten()
    virtual_flags = np.asarray(sampler.virtual_flags)

    X_real = X[~virtual_flags]
    y_real = y[~virtual_flags]
    X_virtual = X[virtual_flags]
    y_virtual = y[virtual_flags]

    n_grids = 400
    xi = np.linspace(sampler.lower[0], sampler.upper[0], num = n_grids)
    yi = np.linspace(sampler.lower[1], sampler.upper[1], num = n_grids)

    xg, yg = np.meshgrid(xi, yi)

    X_test = np.array([xg.flatten(), yg.flatten()]).T
    predictor = gp.query(X_test, sampler.regressors[0])
    zg = np.reshape(gp.mean(sampler.regressors[0], predictor), xg.shape)

    if ax is None:
        ax = pl.gca()

    extent = [sampler.lower[0], sampler.upper[0],
              sampler.upper[1], sampler.lower[1]]
    ax.imshow(zg, extent = extent)

    ax.scatter(X_real[:, 0], X_real[:, 1], c = y_real)
    ax.scatter(X_virtual[:, 0], X_virtual[:, 1], c = y_virtual, marker = 'x')

    ax.set_title('%d Samples' % len(y))

    ax.axis('image')
