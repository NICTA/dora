"""
Utilities for plotting purposes.
"""
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


def plot_sampler_progress(sampler, ax):
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

    X = sampler.X()
    y = sampler.y().flatten()

    if sampler.name in ('GaussianProcess', 'GPflowSampler'):

        n_grids = 400
        xi = np.linspace(sampler.lower[0], sampler.upper[0], num=n_grids)
        yi = np.linspace(sampler.lower[1], sampler.upper[1], num=n_grids)

        xg, yg = np.meshgrid(xi, yi)
        X_test = np.array([xg.flatten(), yg.flatten()]).T
        zg = np.reshape(sampler.predict(X_test)[0], xg.shape)

        extent = [sampler.lower[0], sampler.upper[0],
                  sampler.upper[1], sampler.lower[1]]
        ax.imshow(zg, extent=extent)

    elif sampler.name == 'Delaunay':

        import matplotlib.pyplot as pl
        import matplotlib as mpl

        cols = pl.cm.jet(np.linspace(0, 1, 64))
        custom = mpl.colors.ListedColormap(cols * 0.5 + 0.5)

        w = 4. / np.log(1 + len(y))

        ax.tripcolor(X[:, 0], X[:, 1], y, shading='gouraud',
                     edgecolors='k', linewidth=w, cmap=custom)
        ax.triplot(X[:, 0], X[:, 1], color='k', linewidth=w)
    else:
        raise ValueError('Sampler "%s" not implemented yet'
                         % sampler.name)

    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_title('%d Samples' % len(y))
    ax.axis('image')
