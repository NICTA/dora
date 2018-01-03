"""Demo for lin_sampler.py"""
import logging

import numpy as np
import matplotlib.pyplot as pl

import dora.active_sampling as sampling
from dora.active_sampling import pltutils


def ground_truth(X):
    """Compute the ground truth."""
    return np.sin(X-5)+np.sin(X/2-2)+0.4*np.sin(X/5-2)+0.4*np.sin(X-3)+0.2*np.sin(X/0.3-3)


def main():
    #Set up the problem
    x = np.arange(0, 30, 0.1)
    #Set up the bayesian linear sampler
    lower = [0]
    upper = [30]
    n_train = 15
    feature = 20
    #Define the basis function
    basisdef = 'radial'
    mu = np.linspace(0, 30, feature)[:, np.newaxis]
    s = 1.5
    basisparams = [mu, s]
    #Define the acquisition function
    acq_name = 'pred_upper_bound'
    explore_prority = 1.
    #Construct the sampler
    sampler = sampling.BayesianLinear(lower, upper, basisdef=basisdef,
                                      basisparams=basisparams,  feature=feature,
                                      acq_name=acq_name, n_train=n_train, seed=11)
    #Sample the training data for the sampler
    for i in range(n_train):
        xq, uid = sampler.pick()
        yq_true = ground_truth(xq)
        sampler.update(uid, yq_true)
    #Enough training data, train the sampler
    xq, uid = sampler.pick()
    #Query
    for i in range(8):
        yq_true = ground_truth(xq)
        sampler.update(uid, yq_true)
        xquery = x[:, np.newaxis]
        f_mean, f_std = sampler.predict(xquery)
        #Visualize the prediction
        pl.figure(figsize=(15,5))
        pl.subplot(2,1,1)
        pl.plot(x, ground_truth(x), 'k')
        pl.plot(sampler.X[:,-1], sampler.y[:,-1], 'go', markersize=10)
        pl.plot(sampler.X[-1], sampler.y[-1], 'ro', markersize=10)
        pl.plot(xquery, f_mean, 'b--')
        lower = f_mean - f_std*2
        upper = f_mean + f_std*2
        pl.fill_between(xquery[:,0], upper[:,0], lower[:,0], facecolor='lightblue')
        pl.xlabel('x')
        pl.ylabel('f(x)')
        pl.legend(('Ground turth', 'Observations',
               'Most recent observation', 'Predicted mean', 'Predicted 2 standard deviation'))
        pl.title('Observations after update')

        #Visualize the acquisition function
        acq_value, acq_max_ind = sampler.eval_acq(x)
        pl.subplot(2,1,2)
        pl.plot(x, ground_truth(x), 'k')
        pl.plot(sampler.X, sampler.y, 'go', markersize=10)
        pl.plot(x, acq_value, 'r--')
        pl.plot(x[acq_max_ind], acq_value[acq_max_ind], 'rD', markersize=10)
        pl.xlabel('x')
        pl.ylabel('f(x)')
        pl.title('The new acquisition function after update')
        pl.legend(('Ground turth', 'Observations', 'Acquisition function', 'Acquisition function max'))
        pl.show()
        #Pick the next new query point
        xq, uid = sampler.pick()

if __name__ == "__main__":
    main()
