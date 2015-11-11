"""
Active Sampling Demo

This demo interacts uses Dora's REST api to actively sample a model that
 returns a vector.


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

# The plotting subpackage is throwing FutureWarnings
import warnings
warnings.simplefilter("ignore", FutureWarning)

def simulate_measurement(X, uid=None):
    """ The true model is two Gaussian blobs located at centre1 and centre2
    The model is queried with a 2-D location and returns an vector of length
    output_len with values corresponding to the underlying function's values
    along a z-axis
    """
    assert X.shape == (2,)
    measurement_range = [0, 1]
    centre1 = np.array([1.5, 1.4, 0.3])
    centre2 = np.array([2.50, 2.0, 0.7])
    l1 = 0.2
    l2 = 0.3
    output_len = 20

    Z = np.arange(measurement_range[0], measurement_range[1],
                  (measurement_range[1] - measurement_range[0])/output_len)
    dist1 = [np.sqrt((centre1[0]-X[0])**2+(centre1[1]-X[1])**2
                     + (centre1[2]-z)**2) for z in Z]
    dist2 = [np.sqrt((centre2[0]-X[0])**2+(centre2[1]-X[1])**2
                     + (centre2[2]-z)**2) for z in Z]
    measurement = np.asarray([np.exp(-dist1[i]/l1) + np.exp(-dist2[i]/l2)
                          for i in range(len(dist1))])
    return uid, measurement


def main():

    server = subprocess.Popen(['python3', '../dora/server/server.py'])
    time.sleep(5)

    # Set up a sampling problem:
    target_samples = 100
    n_train = 15
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
        #post a request to the sampler for a query location
        query_loc = requests.post(sampler_info['obs_uri']).json()

        # Evaluate the sampler's query on the forward model
        characteristic = np.array(query_loc['query'])
        uid = query_loc['uid']
        uid, measurement = simulate_measurement(characteristic, uid)

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
    vv.use().Run()
    server.terminate()



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
    pred=requests.get(sampler_info['pred_uri'], json=Xquery.tolist()).json()
    pred_mean = pred['predictive_mean']
    id_matrix = np.reshape(np.arange(Xquery.shape[0])[:,np.newaxis],xeva.shape)
    vol = np.zeros((n_outputs, xeva.shape[0], xeva.shape[1]))
    for x in range(xres):
        for y in range(yres):
            vol[:,x,y] = pred_mean[id_matrix[x,y]]
    plt = vv.volshow(vol, renderStyle='mip',clim=(-0.5, 1))
    plt.colormap = vv.CM_JET  #  HOT
    subplt.axis.xLabel = 'input_1'
    subplt.axis.yLabel = 'input_2'
    subplt.axis.zLabel = 'model_output'
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
