"""
Active Sampling Demo

This demo uses the Dora active sampling module as a python library that can be
directly called from within the process.

We solve a two dimensional sampling problem using an efficient sampling
scheme. The underlying problem is to sample the boundaries of a circle, given
no prior knowledge of its geometry.

The sampler uses a Gaussian process to probabilistically approximate the true
model

"""
import logging
import matplotlib.pyplot as pl
from dora.active_sampling import pltutils
import dora.active_sampling as sampling
from example_processes import simulate_measurement

# The plotting subpackage is throwing FutureWarnings
import warnings
warnings.simplefilter("ignore", FutureWarning)


def main():

    # Set up a sampling problem:
    n_target_samples = 301
    lower = [0, 0]
    upper = [1, 1]

    sampler = sampling.StackedGaussianProcess(lower, upper,
                                              acq_name = 'sigmoid')

    # Set up plotting:
    plot_triggers = [20, 50, 100, 150, 200, 300]
    n_triggers = len(plot_triggers)
    plt_size = pltutils.split_subplots(n_triggers)

    fig = pl.figure()
    axs = iter([fig.add_subplot(*(plt_size + (i + 1,)))
                for i in range(n_triggers)])

    train_triggers = [False if i % 25 > 0 else True
                      for i in range(n_target_samples)]

    # Run the active sampling:
    logging.info('Actively sampling new points..')
    for i in range(n_target_samples):

        xq, uid = sampler.pick(train = train_triggers[i])

        yq_true = simulate_measurement(xq)

        sampler.update(uid, yq_true)

        if i in plot_triggers:
            pltutils.plot_sampler_progress(sampler, ax = next(axs))

        logging.info(i)
    pl.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
