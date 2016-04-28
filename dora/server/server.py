import flask as fl
from dora.server.response import returns_json
from dora.active_sampling import GaussianProcess as GPsampler
import numpy as np
import logging

app = fl.Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/samplers', methods=['POST'])
@returns_json
def initialise_sampler():
    """ Initialise the Sampler Model
    This expects a dict containing:
        lower : a list of the lower bounds of the region of interest
        upper : a list of the upper bounds of the region of interest

        Optional dict entries for the model initialisation:
            kerneldef : Kernel function definition. See the 'gp' module.
            n_train : int
                Number of training samples required before sampler can be
                trained
            acq_name : str
                A string specifying the type of acquisition function used
            explore_priority : float, optional
                The priority of exploration against exploitation
    """
    initDict = fl.request.json


    if not hasattr(fl.current_app, 'samplers'):
        fl.current_app.samplers = {}

    samplerid = len(fl.current_app.samplers)

    fl.current_app.samplers[samplerid] \
        = GPsampler(initDict['lower'], initDict['upper'], acq_name=initDict['acq_name'],
                    n_train= initDict['n_train'])

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
        and an associated universal resource identifier:

        query: list of the parameters to observe
        uid : unique identifier for the sampler
        uri : universal resource identifier for the observation
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
    parameters

    It expects a list of lists with n elements in the main list
    corresponding to the number of queries and each element has d elements
    corresponding to the number of parameters in the forward model.

    It returns a dict with a predictive mean and and predictive variance entry
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

    # n_stacks = fl.current_app.samplers[int(samplerid)].n_stacks

    mean = list(fl.current_app.samplers[int(samplerid)].y_mean)
    # trained_flag = fl.current_app.samplers[int(samplerid)].trained_flag

    # TODO add ability to retrieve full state
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

    response_data = {"lower": lower, "upper": upper,
                     "mean": mean}
    # , "trained_flag": trained_flag}
    #'n_stacks': n_stacks,
    return response_data, 200

if __name__ == '__main__':
    app.run(debug=True)
