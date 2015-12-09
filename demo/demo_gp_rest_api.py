"""
Active Sampling Demo

This demo interacts uses Dora's REST api to actively sample a model that
 returns a vector.

 The acquisition balances sampler uncertainty with the predicted magnitude of
 the underlying process' value


"""

import numpy as np
import logging
import matplotlib.pyplot as pl
import matplotlib as mpl
import dora.active_sampling as sampling
import dora.regressors.gp as gp
import subprocess
import requests
import time
import visvis as vv
from example_processes import simulate_measurement_vector

# The plotting subpackage is throwing FutureWarnings
import warnings
warnings.simplefilter("ignore", FutureWarning)




def main():

    server = subprocess.Popen(['python', '../dora/server/server.py'])
    time.sleep(5)

    # Set up a sampling problem:
    target_samples = 100
    n_train = 12
    lower = [1., 1.]
    upper = [3., 3.]
    explore_factor = 0.3
    n_outputs = 20

    initialiseArgs = {'lower':lower, 'upper':upper,'n_outputs':n_outputs,
                  'n_train_threshold': n_train,
                  'acquisition_func': 'prodmax',
                  'explore_factor': explore_factor}

    # initialise sampler
    sampler_info = requests.post('http://localhost:5000/samplers',
                               json=initialiseArgs).json()
    logging.info("Model Info: " + str(sampler_info))

    # Set up plotting:
    plots = {'fig': vv.figure(),
             'count': 0,
             'shape': (2, 3)}
    plot_triggers = [16, 30, 40, 50, 65, target_samples-1]

    # Run the active sampling:
    for i in range(target_samples):
        logging.info('Samples: %d' % i)
        #post a request to the sampler for a query location
        r = requests.post(sampler_info['obs_uri'])
        r.raise_for_status()

        query_loc = r.json()

        # Evaluate the sampler's query on the forward model
        characteristic = np.array(query_loc['query'])
        uid = query_loc['uid']
        uid, measurement = simulate_measurement_vector(characteristic, uid)
        print(measurement)
        # Update the sampler with the new observation from the forward model
        r = requests.put(query_loc['uri'], json=measurement.tolist())

        if i in plot_triggers:
            plot_progress(plots, sampler_info)

    # Retrieve Training Data:
    training_data = requests.get(sampler_info['training_data_uri']).json()
    logging.info('X:' + str(training_data['X']))
    logging.info('y:' + str(training_data['y']))
    logging.info('Virtual X:' + str(training_data['virtual_X']))
    logging.info('Virtual y:' + str(training_data['virtual_y']))
    server.terminate()
    vv.use().Run()




def plot_progress(plots, sampler_info):

    settings = requests.get(sampler_info['settings']).json()
    lower = settings['lower']
    upper = settings['upper']
    n_outputs = settings['n_stacks']

    fig = plots['fig']
    subplt =vv.subplot(*(plots['shape'] + (1+plots['count'],)))
    plots['count'] += 1

    # Plot predictions and training data
    training_data = requests.get(sampler_info['training_data_uri']).json()

    xres = 30
    yres = 30
    xeva, yeva = np.meshgrid(np.linspace(lower[0], upper[0], xres), np.linspace(
        lower[1], upper[1], yres))
    Xquery = np.array([xeva.flatten(),yeva.flatten()]).T
    r=requests.get(sampler_info['pred_uri'], json=Xquery.tolist())
    r.raise_for_status()
    pred = r.json()
    pred_mean = pred['predictive_mean']
    id_matrix = np.reshape(np.arange(Xquery.shape[0])[:,np.newaxis],xeva.shape)
    vol = np.zeros((n_outputs, xeva.shape[0], xeva.shape[1]))
    for x in range(xres):
        for y in range(yres):
            vol[:,x,y] = pred_mean[id_matrix[x,y]]
    plt = vv.volshow(vol, renderStyle='mip',clim=(-0.5, 1))
    plt.colormap = vv.CM_JET  #  HOT
    subplt.axis.xLabel = 'input 1'
    subplt.axis.yLabel = 'input 2'
    subplt.axis.zLabel = 'model output'
    a = ((np.asarray(training_data['X']) - np.array([np.min(xeva),
            np.min(yeva)])[np.newaxis,:])/ np.array([np.max(xeva) -
            np.min(xeva),np.max(yeva)-np.min(yeva)])[np.newaxis,:])  \
            * np.array(xeva.shape)
    n = a.shape[0]
    a = np.hstack((a, (n_outputs+0.01)*np.ones((n, 1))))
    pp = vv.Pointset(a)
    vv.plot(pp, ms='.', mc='w', mw='9', ls='')




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
