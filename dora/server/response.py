from functools import wraps
import json
import flask as fl


def returns_json(f):
    """ A decorator for Flask route handlers that return JSON.

    Any route handler decorated by this function needs to return a pair
    (dictionary, status code). The dictionary is automatically converted into
    a JSON string. The status code is an integer. The content type of the
    response is automatically set to JSON.
    """
    @wraps(f)
    def decorated_handler(*args, **kwargs):
        r, status = f(*args, **kwargs)
        return fl.Response(json.dumps(r, sort_keys=True), status,
                           content_type='application/json; charset=utf-8')
    return decorated_handler
