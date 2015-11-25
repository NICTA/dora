import numpy as np
from dora.regressors import gp
from dora.regressors.active_sampling import StackedGaussianProcess as StackedGP
import logging
import pickle
import visvis

log = logging.getLogger(__name__)

path = 'logs/'
filename = 'sampler_2015_09_18__17_25_06.pkl'
data = pickle.load(open(path+filename,"rb"))

# Use the redis workers to evaluate an initial batch of jobs
X_train = np.asarray(data['X'])
y_train = np.asarray(data['y']).T
n_GP_slices = y_train[0].shape[0]
lower = list(np.min(X_train,axis=0))
upper = list(np.max(X_train,axis=0))

meany = np.mean(y_train)
# y_train -= np.mean(y_train)
log.info('Init Sampler')
# import ipdb; ipdb.set_trace()
sampler = StackedGP(lower, upper, X_train, y_train, mean=meany, n_stacks=n_GP_slices,
                    add_train_data=True)
qx, qy = np.mgrid[lower[0]:upper[0]:30j, lower[1]:upper[1]:30j]


def display_results(regressors, qx, qy, grid_qry):
    n_GP_slices = len(regressors)
    shape = qx.shape
    stacks = []
    for i in range(n_GP_slices):
        regressor = regressors[i]
        qz = i*np.ones(shape)
        mu = gp.mean(regressor, grid_qry).reshape(shape).T
        # print(mu)
        stacks.append(mu)
        vol = np.array(stacks)
    plt = visvis.volshow(vol, renderStyle='mip',clim=(-5., 5.))
    plt.colormap = visvis.CM_JET  #  HOT
    a = ((regressors[0].X - np.array([np.min(qx),np.min(qy)])[np.newaxis,:])/
         np.array([np.max(qx)-np.min(qx),np.max(qy)-np.min(qy)])[np.newaxis,:])  \
        * np.array(qx.shape)
    n = a.shape[0]
    a = np.hstack((a, (n_GP_slices+0.01)*np.ones((n, 1))))
    pp = visvis.Pointset(a)
    visvis.plot(pp, ms='.', mc='w', mw='9', ls='')
    visvis.use().Run()