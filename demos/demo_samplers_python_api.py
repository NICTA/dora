"""
Active Sampling Demo.

This demo uses the Dora active sampling module as a python library that can be
directly called from within the process.

We solve a two dimensional sampling problem using an efficient sampling
scheme. The underlying problem is to sample the boundaries of a circle, given
no prior knowledge of its geometry.

The sampler uses a Gaussian process to probabilistically approximate the true
model.

"""
import logging

import dora.active_sampling as sampling

from dora.active_sampling import pltutils

from demos.example_processes import simulate_measurement

import matplotlib.pyplot as pl

from time import sleep

def main(sampler_model='GaussianProcess', plot=False):

    # Set up a sampling problem
    lower = [0, 0]
    upper = [1, 1]
    n_samples = 151

    logging.info('\n \n Sampler Model: %s' % sampler_model)
    sleep(3)


    # Initialise the sampler
    if sampler_model == 'GaussianProcess':
        acq_name = 'sigmoid'
        n_train = 30
        sampler = sampling.GaussianProcess(lower, upper, acq_name=acq_name,
                                           n_train=n_train, seed=100)
    elif sampler_model == 'Delaunay':

        explore_priority = 0.0001
        sampler = sampling.Delaunay(lower, upper,
                                    explore_priority=explore_priority)
    else:
        raise ValueError('Sampling method "%s" not implemented yet'
                         % sampler_model)

    # Set up plotting
    if plot:
        # plot_triggers = [50, 100, 150, 200, 250, 300]
        plot_triggers = [30, 40, 50, 70, 100, 150]
        n_triggers = len(plot_triggers)
        plt_size = pltutils.split_subplots(n_triggers)
        fig = pl.figure()
        pl.title(sampler_model)
        axs = iter([fig.add_subplot(*(plt_size + (i + 1,)))
                    for i in range(n_triggers)])

    # Start active sampling!
    for i in range(n_samples):

        # Log the iteration number
        if divmod(i,100)[1] == 0:
            log_info = True
            logging.info('Iteration: %d' % i)

        # Pick a location to sample
        if log_info:
            logging.info('Picking new observation location..')
        xq, uid = sampler.pick()

        # Sample that location
        if log_info:
            logging.info('Evaluating observation value...')
        yq_true = simulate_measurement(xq)

        # Update the sampler about the new observation
        if log_info:
            logging.info('Updating sampler... \n \n ')
            log_info = False
            sleep(2)
        sampler.update(uid, yq_true)

        # Plot the sampler progress
        if plot and i in plot_triggers:
            pltutils.plot_sampler_progress(sampler, next(axs))



    # Sampler demos must return the sampler itself
    return sampler

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sampler_model='Delaunay', plot=True)
    main(sampler_model='GaussianProcess', plot=True)
    pl.show()
