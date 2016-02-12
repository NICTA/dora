"""
Active Sampling Demo.

This demo uses the Dora active sampling module as a python library that can be
directly called from within the process.

We solve a two dimensional sampling problem using an efficient sampling
scheme. The underlying problem is to sample the boundaries of a circle, given
no prior knowledge of its geometry.

The sampler uses a Gaussian process to probabilistically approximate the true
model

"""
import logging

import dora.active_sampling as sampling

from dora.active_sampling import pltutils

from example_processes import simulate_measurement

import matplotlib.pyplot as pl


# TEST TRAVIS
def main():

    # Set up a sampling problem
    n_target_samples = 301
    lower = [0, 0]
    upper = [1, 1]

    # Initialise the sampler
    sampler = sampling.GaussianProcess(lower, upper, acq_name='sigmoid',
                                       n_train=49)

    # Set up plotting
    plot_triggers = [50, 100, 150, 200, 250, 300]
    n_triggers = len(plot_triggers)
    plt_size = pltutils.split_subplots(n_triggers)
    fig = pl.figure()
    axs = iter([fig.add_subplot(*(plt_size + (i + 1,)))
                for i in range(n_triggers)])

    retrain = [100, 200]

    # Start active sampling!
    for i in range(n_target_samples):

        # Pick a location to sample
        xq, uid = sampler.pick()

        # Sample that location
        yq_true = simulate_measurement(xq)

        # Update the sampler about the new observation
        sampler.update(uid, yq_true)

        if i in retrain:
            sampler.train()

        # Plot the sampler progress
        if i in plot_triggers:
            # sampler.train()  Be careful - retraining on biased data
            pltutils.plot_sampler_progress(sampler, ax=next(axs))

        logging.info('Iteration: %d' % i)

    pl.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
