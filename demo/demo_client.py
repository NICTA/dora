"""
Demo that creates an empty Dora GP sampler for a forward model with a
20-dimensional vector output. The demo runs for 70 evaluations of the forward
model. n_pre_train indicates how many of these evaluations are randomly
sampled and used as training data for the sampler. Visualisation is done with 
visvis.
"""

import requests
import numpy as np
from worker_jobs import plant_fake
import visvis as vv



def main():

    # Sampler bounds:
    lower_bounds = [1., 1.]
    upper_bounds = [3., 3.]

    n_output = 20  # The length of the model's output vector
    n_pre_train = 40 # The number of evaluations that will be used to train the sampler
    explore_factor = 0.3


    initialiseArgs = {'lower':lower, 'upper':upper,'n_outputs':n_outputs,
                    'n_pre_train': n_pre_train,
                    'acquisition_func': 'prodmax', 'explore_factor': explore_factor}

    # initialise sampler
    sampler_info = requests.post('http://localhost:5000/samplers',
                            json=initialiseArgs).json()
    print("Model Info: " + str(sampler_info))

    # evaluate the sampler 70 times:
    n_runs = 70
    query_loc = []
    for i in range(n_runs):
        print(i)

        #post a request to the sampler for a query location
        query_loc = (requests.post(sampler_info['obs_uri']).json())

        # Evaluate the sampler's query on the forward model
        characteristic = np.array(query_loc['query'])
        uid = query_loc['uid']
        uid, fitness = plant_fake(characteristic, n_outputs, uid)

        # Update the sampler with the new observation from the forward model
        r = requests.put(query_loc['uri'], json=fitness.tolist())


    # Create three queries for the sampler and print the results
    distrib_loc = np.array([[1.1, 1.4], [1.3, 1.8], [1.2, 1.9]])
    pred = requests.get(sampler_info['pred_uri'], json=distrib_loc.tolist()).json()
    print("Predictive Means: ");print(pred['predictive_mean'])
    print("Predictive Variances: ");print(pred['predictive_variance'])


    # Retrieve Training Data:
    training_data = requests.get(sampler_info['training_data_uri']).json()
    print('X:' + str(training_data['X']))
    print('y:' + str(training_data['y']))
    print('Virtual X:' + str(training_data['virtual_X']))
    print('Virtual y:' + str(training_data['virtual_y']))

    # Plot predictions and training data
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
    subplt = vv.subplot(111)
    plt = vv.volshow(vol, renderStyle='mip',clim=(-0.5, 1.))
    plt.colormap = vv.CM_JET  #  HOT
    subplt.axis.xLabel = 'time to disturbance'
    subplt.axis.yLabel = 'B4 Slope'
    subplt.axis.zLabel = 'lma'
    a = ((np.asarray(training_data['X']) - np.array([np.min(xeva),
            np.min(yeva)])[np.newaxis,:])/ np.array([np.max(xeva) -
            np.min(xeva),np.max(yeva)-np.min(yeva)])[np.newaxis,:])  \
            * np.array(xeva.shape)
    n = a.shape[0]
    a = np.hstack((a, (n_outputs+0.01)*np.ones((n, 1))))
    pp = vv.Pointset(a)
    vv.plot(pp, ms='.', mc='w', mw='9', ls='')
    vv.use().Run()


if __name__=='__main__':
    main()
