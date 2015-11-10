import requests
import numpy as np
import computers.active_sampling as sampling
from worker_jobs import plant_fake
import visvis as vv

# Initialising arguments of the sampler
lower = [1., 1.]
upper = [3., 3.]
n_outputs = 20
n_train = 35
X_train = sampling.random_sample(lower, upper, n_train)

y_train = np.zeros((n_train, n_outputs))
for i in range(n_train):
    _, fitness = plant_fake(X_train[i], n_outputs)
    y_train[i,:] = fitness

meany = np.mean(y_train)

initialiseArgs = {'lower':lower, 'upper':upper, 'X_train':X_train.tolist(),
           'y_train':y_train.tolist(), 'mean':meany,
           'n_outputs':n_outputs,'add_train_data': True, 'n_train_threshold': 30}

# initialise sampler
sampler_info = requests.post('http://localhost:5000/samplers', json=initialiseArgs).json()
print("Model Info: " + str(sampler_info))

# Post a request for parameters to query the forward model with:
query_loc_1 = requests.post(sampler_info['obs_uri']).json()
print("Query Point 1: ");print(query_loc_1)
query_loc_2 = requests.post(sampler_info['obs_uri']).json()
print("Query Point 2: ");print(query_loc_2)
query_loc_3 = requests.post(sampler_info['obs_uri']).json()
print("Query Point 3: ");print(query_loc_3)

# Evaluate one of the sampler's queries on the forward model
print("Evaluating Plant at the Query points")
characteristic = np.array(query_loc_1['query'])
uid = query_loc_1['uid']
uid, fitness = plant_fake(characteristic, n_outputs, uid)

# Update the sampler with the new observation from the forward model
r = requests.put(query_loc_1['uri'], json=fitness.tolist())


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
xeva, yeva = np.meshgrid(np.linspace(lower[0], upper[0], xres), np.linspace(lower[1], upper[1], yres))
Xquery = np.array([xeva.flatten(),yeva.flatten()]).T
pred=requests.get(sampler_info['pred_uri'], json=Xquery.tolist()).json()
pred_mean = pred['predictive_mean']
id_matrix = np.reshape(np.arange(Xquery.shape[0])[:,np.newaxis],xeva.shape)
vol = np.zeros((n_outputs, xeva.shape[0], xeva.shape[1]))
for x in range(xres):
    for y in range(yres):
        vol[:,x,y] = pred_mean[id_matrix[x,y]]
subplt = vv.subplot(111)
# import ipdb; ipdb.set_trace()
plt = vv.volshow(vol, renderStyle='mip',clim=(-0.5, 1.))
plt.colormap = vv.CM_JET  #  HOT
subplt.axis.xLabel = 'time to disturbance'
subplt.axis.yLabel = 'B4 Slope'
subplt.axis.zLabel = 'lma'
a = ((np.asarray(training_data['X']) - np.array([np.min(xeva),np.min(yeva)])[np.newaxis,:])/
         np.array([np.max(xeva)-np.min(xeva),np.max(yeva)-np.min(yeva)])[np.newaxis,:])  \
        * np.array(xeva.shape)
n = a.shape[0]
a = np.hstack((a, (n_outputs+0.01)*np.ones((n, 1))))
pp = vv.Pointset(a)
vv.plot(pp, ms='.', mc='w', mw='9', ls='')
vv.use().Run()
