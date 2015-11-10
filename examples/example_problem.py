"""
Active sampling experiment controller
"""
import numpy as np
import logging
import matplotlib.pyplot as pl
import computers.active_sampling as sampling
import matplotlib as mpl
import computers.gp as gp

def property(X):
    """ A binary image of a circle as a test problem for sampling
    """
    return (np.sum((X-0.5)**2) < 0.1).astype(float)


def main(method_name):

    target_samples = 501

    if method_name == 'Delaunay':
        explore_priority = 0.0001  # relative to the *difference* in stdev
        sampler = sampling.Delaunay([0, 0], [1, 1], explore_priority)
    elif method_name == 'Gaussian_Process':
        n_train = 50
        lower = [0, 0]
        upper = [1, 1]
        X_train = sampling.random_sample(lower,upper,n_train)
        y_train = np.asarray([property(i) for i in X_train])
        sampler = sampling.Gaussian_Process(lower, upper, X_train, y_train,
                                            add_train_data=False)
    else:
        raise ValueError('Unrecognised method name.')

    stages = [8, 9, 10, 50, 100, 500]
    
    cols = pl.cm.jet(np.linspace(0,1,64))
    custom = mpl.colors.ListedColormap(cols*0.5+0.5)
    pl.figure()
    count = 0
    for i in range(target_samples):
        newX = sampler.pick()
        observation = property(newX)
        sampler.update(newX, observation)

        # Plot the result:
        if i in stages:
            pl.subplot(231+count)
            count+=1
            X = np.asarray(sampler.X)
            y = np.asarray(sampler.regressor.y)
            w = 4./np.log(1+i)

            if method_name == 'Delaunay':
                pl.tripcolor(X[:, 0], X[:, 1], y, shading='gouraud',
                         edgecolors='k', linewidth=w,
                         cmap=custom)
                pl.triplot(X[:, 0], X[:, 1], color='k', linewidth=w)
            elif method_name == 'Gaussian_Process':
                X = sampler.regressor.X
                xi, yi = np.linspace(np.min(X,axis=0)[0], np.max(X,axis=0)[0], 400), np.linspace(np.min(X,axis=0)[1], np.max(X,axis=0)[1], 400)
                xi, yi = np.meshgrid(xi, yi)
                x_test = np.array([xi.flatten(),yi.flatten()]).T
                query = gp.query(x_test, sampler.regressor)
                zi = np.reshape(gp.mean(sampler.regressor, query),xi.shape)
                pl.imshow(zi, vmin=0, vmax=1,
                    extent=[np.min(X,axis=0)[0], np.max(X,axis=0)[0],  np.max(X,axis=0)[0], y.min()])
            pl.scatter(X[:, 0], X[:, 1],c=y)
            pl.axis('image')
            pl.title('%d Samples' % i)

    pl.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main('Delaunay')
    main('Gaussian_Process')
