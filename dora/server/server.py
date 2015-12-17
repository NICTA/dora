import flask as fl
from dora.server.response import returns_json
from computers.active_sampling import Stacked_Gaussian_Process as StackedGP
import numpy as np

app = fl.Flask(__name__)


@app.route('/samplers', methods=['POST'])
@returns_json
def initialise_sampler():
    """ Initialise the Sampler Model
    This expects a dict containing:
    lower : a list of the lower bounds of the region of interest
    upper : a list of the upper bounds of the region of interest
    X_train : training data to train the sampler. a list of n elements where is
              element is itself a list with d elements where n
              is the number of
              training points and d is the number of dimension (parameters of
              the forward model)
    y_train : the observed values outputted by the forward model for each set
              of parameters in X_train. another list of lists. n x s where s
              is the number of outputs of the forward model. s also corresponds
              to the number of stacks in the sampler
    Optional keys in the dict:
    add_train_data : a boolean operator that tells the sampler to include the
                     training data to the sampler rather than just using it to
                     train its parameters. Default = True
    n_stacks : the number of stacks in the sampler. this must be equal to the
               number of outputs of the forward model
    mean : the mean of the training data. this is redundant and will be
           removed shortly. Default = 0. TODO(SIMON)
    """
    initDict = fl.request.json

    if 'X_train' in initDict.keys():
        initDict['X_train'] = np.asarray(initDict['X_train'])
        initDict['y_train'] = np.asarray(initDict['y_train'])

    mapping = {'X_train': 'X_train', 'y_train': 'y_train',
               'add_train_data': 'add_train_data',
               'n_outputs': 'n_stacks',
               'mean': 'mean', 'n_train_threshold': 'n_train_threshold',
               'acquisition_func': 'acq_func',
               'explore_factor': 'explore_factor'}

    mapDict = {mapping[k]: v for k, v in initDict.items() if k in mapping}

    if not hasattr(fl.current_app, 'samplers'):
        fl.current_app.samplers = {}

    samplerid = len(fl.current_app.samplers)
    fl.current_app.samplers[samplerid] \
        = StackedGP(initDict['lower'], initDict['upper'], **mapDict)

    obs_uri = fl.url_for('.get_query', samplerid=samplerid, _external=True)
    pred_uri = fl.url_for('.predict', samplerid=samplerid, _external=True)
    training_uri = fl.url_for('.retrieve_trainingdata', samplerid=samplerid,
                              _external=True)
    settings_uri = fl.url_for('.retrieve_settings', samplerid=samplerid,
                              _external=True)

    response_data = {"status": "OK", "samplerid": samplerid, 'obs_uri':
                     obs_uri, 'pred_uri': pred_uri, 'training_data_uri':
                     training_uri, 'settings': settings_uri}

    return response_data, 200


@app.route('/samplers/<string:samplerid>/observations', methods=['POST'])
@returns_json
def get_query(samplerid):

    """ Returns a set of parameters to query the forward model with
        and an associated unique identifier
    """

    newX, uid = fl.current_app.samplers[int(samplerid)].pick()
    obs_uri = fl.url_for('.update_sampler', samplerid=samplerid, uid=uid,
                         _external=True)
    response_data = {"query": newX.tolist(), "uid": uid, 'uri': obs_uri}
    return response_data, 200


@app.route('/samplers/<string:samplerid>/observations/<string:uid>',
           methods=['PUT'])
@returns_json
def update_sampler(samplerid, uid):
    """ updates the sampler with the forward model's observation
    uid : the unique identifier provided with the query parameters

    Expects a list of the forward models outputs (s elements where s also
    equals the number of stacks in the sampler)
    """
    measurement = fl.request.json
    fl.current_app.samplers[int(samplerid)].update(uid,
                                                   np.asarray(measurement))
    response_data = "Model updated with measurement"
    return response_data, 200


@app.route('/samplers/<string:samplerid>/prediction', methods=['GET'])
@returns_json
def predict(samplerid):
    """
    provides a prediction of the forward model at a set of given query
    parameters it expects a list of lists with n elements in the main list
    corrsponding to the number of queries and each element has d elements
    corresponding to the number of parameters in the forward model.  It
    returns a dict with a predictive mean and and predictive variance entry
    Each are n x n_stacks
    """

    query_loc = fl.request.json
    pred_mean, pred_var = \
        fl.current_app.samplers[int(samplerid)].predict(np.asarray(query_loc))
    response_data = {"predictive_mean": pred_mean.tolist(),
                     "predictive_variance": pred_var.tolist()}
    return response_data, 200


@app.route('/samplers/<string:samplerid>/trainingdata', methods=['GET'])
@returns_json
def retrieve_trainingdata(samplerid):
    """
    provides lists of the real and virtual training data used by the sampler.
    """
    X = [x.tolist() for x in fl.current_app.samplers[int(samplerid)].X]
    y = [y.tolist() for y in fl.current_app.samplers[int(samplerid)].y]

    virtualIndices = fl.current_app.samplers[int(samplerid)].virtual_flag
    real_id = [not i for i in virtualIndices]

    real_X = [x_ for x_, real in zip(X, real_id) if real is True]
    real_y = [y_ for y_, real in zip(y, real_id) if real is True]

    virt_X = [x_ for x_, real in zip(X, real_id) if real is False]
    virt_y = [y_ for y_, real in zip(y, real_id) if real is False]
    response_data = {"X": real_X, "y": real_y,
                     "virtual_X": virt_X, "virtual_y": virt_y}
    return response_data, 200


@app.route('/samplers/<string:samplerid>/settings', methods=['GET'])
@returns_json
def retrieve_settings(samplerid):
    """
    provides lists of the settings used by the sampler.
    """
    lower = fl.current_app.samplers[int(samplerid)].lower.tolist()
    upper = fl.current_app.samplers[int(samplerid)].upper.tolist()

    n_stacks = fl.current_app.samplers[int(samplerid)].n_stacks

    mean = fl.current_app.samplers[int(samplerid)].mean
    trained_flag = fl.current_app.samplers[int(samplerid)].trained_flag
    # TODO <SIMON> add ability to retrieve full state
    # hyper_params = fl.current_app.samplers[int(samplerid)].hyper_params
    # regressors = [reg.tolist() for reg in
    # fl.current_app.samplers[int(samplerid)].regressors] acq_func =
    # [y.tolist() for y in fl.current_app.samplers[int(samplerid)].acq_func]
    # explore_factor = [y.tolist() for y in
    # fl.current_app.samplers[int(samplerid)].explore_factor]
    #
    # virtualIndices = fl.current_app.samplers[int(samplerid)].virtual_flag
    # real_id = [not i for i in virtualIndices]
    #
    # real_X = [x for x, real in zip(X, real_id) if real is True]
    # real_y = [y for y, real in zip(y, real_id) if real is True]
    #
    # virt_X = [x for x, real in zip(X, real_id) if real is False]
    # virt_y = [y for y, real in zip(y, real_id) if real is False]

    response_data = {"lower": lower, "upper": upper, 'n_stacks': n_stacks,
                     "mean": mean, "trained_flag": trained_flag}
    return response_data, 200

if __name__ == '__main__':
    app.run(debug=True)
