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


def main(sampling_method='GaussianProcess'):

    # Set up a sampling problem
    lower = [0, 0]
    upper = [1, 1]
    n_samples = 301

    # Initialise the sampler
    if sampling_method == 'GaussianProcess':
        acq_name = 'sigmoid'
        n_train = 49
        sampler = sampling.GaussianProcess(lower, upper, acq_name=acq_name,
                                           n_train=n_train, seed=100)
    elif sampling_method == 'Delaunay':
        explore_priority = 0.0001
        sampler = sampling.Delaunay(lower, upper,
                                    explore_priority=explore_priority)
    else:
        raise ValueError('Sampling method "%s" not implemented yet'
                         % sampling_method)

    # Set up plotting
    plot_triggers = [50, 100, 150, 200, 250, 300]
    n_triggers = len(plot_triggers)
    plt_size = pltutils.split_subplots(n_triggers)
    fig = pl.figure()
    axs = iter([fig.add_subplot(*(plt_size + (i + 1,)))
                for i in range(n_triggers)])

    # Start active sampling!
    for i in range(n_samples):

        # Pick a location to sample
        xq, uid = sampler.pick()

        # Sample that location
        yq_true = simulate_measurement(xq)

        # Update the sampler about the new observation
        sampler.update(uid, yq_true)

        # Plot the sampler progress
        if i in plot_triggers:
            pltutils.plot_sampler_progress(sampler, next(axs))

        # Log the iteration number
        logging.info('Iteration: %d' % i)

    # Sampler demos must return the sampler itself
    return sampler

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sampling_method='Delaunay')
    main(sampling_method='GaussianProcess')
    pl.show()
