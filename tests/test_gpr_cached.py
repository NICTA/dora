import pytest
import numpy as np
import GPflow as gp
from dora.active_sampling.gpr_cached import GPRCached
import time


@pytest.fixture
def data_gen():
    """ 1-D training data generator. """
    def fn(X):
        Y = (np.sin(X - 5) + np.sin(X / 2 - 2)
             + 0.4 * np.sin(X / 5 - 2) + 0.4 * np.sin(X - 3)
             + 0.2 * np.sin(X / 0.3 - 3))
        return Y

    def data(n=1000, low=0, high=100):
        X = np.random.uniform(low, high, n)[:, np.newaxis]
        Y = fn(X)
        return X, Y

    return data


def time_it(fn):
    """ Decorator to time function call. """
    def _wrapper(*args):
        start = time.clock()
        y = fn(*args)
        delta = (time.clock() - start)
        return delta, y
    return _wrapper


def test_cache_sync(data_gen):
    """ Test that cached is updated after optimising, updating the hyper
        parameters, or resetting X, Y
    """
    X, Y = data_gen(1000)
    m = GPRCached(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                  mean_function=gp.mean_functions.Constant(),
                  name='GPRCached')

    L0 = m.cholesky.value
    m.optimize()
    L1 = m.cholesky.value
    assert not np.array_equal(L0, L1)

    new_state = m.get_free_state() + 0.1
    m.set_state(new_state)
    L2 = m.cholesky.value
    assert not np.array_equal(L1, L2)

    new_params = {k: v + 0.1 for k, v in m.get_parameter_dict().items()}
    m.set_parameter_dict(new_params)
    L3 = m.cholesky.value
    assert not np.array_equal(L2, L3)

    new_X = m.X.value + 0.1
    new_Y = m.Y.value + 0.1
    with pytest.raises(ValueError):
        m.X = new_X

    with pytest.raises(ValueError):
        m.Y = new_Y

    m.set_data_points(new_X, new_Y)
    L4 = m.cholesky.value
    assert not np.array_equal(L3, L4)


def test_cholesky_caching(data_gen):
    """ Test that caching the Cholesky decomposition is more efficient
        and produces same result as standard GPR.
    """
    X, Y = data_gen(1000)

    # Create standard GPR and learn hyperparms
    m = gp.gpr.GPR(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                   mean_function=gp.mean_functions.Constant())
    m.optimize()

    # Create GPRCached and set hyperparams to same a GPR above
    m_cached = GPRCached(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                         mean_function=gp.mean_functions.Constant())
    m_cached.set_parameter_dict(m.get_parameter_dict())

    @time_it
    def predict_n_times(m, X):
        """ Call `predict_f` on the provided model for each value of `X` """
        ys = []
        for i in range(len(X)):
            ys.append(m.predict_f(X[i, :, np.newaxis]))
        return np.hstack(ys)

    # Run predict step N times
    Np = 10
    Xq, _ = data_gen(Np)
    m_time, m_pred = predict_n_times(m, Xq)
    m_cached_time, m_cached_pred = predict_n_times(m_cached, Xq)

    # Ensure results are the same and GPRCached runs at least `factor` times
    # faster
    factor = 3
    assert m_time > m_cached_time * factor
    assert np.allclose(m_pred, m_cached_pred)


def test_cholesky_update(data_gen):
    """ Test that incremental update of the Cholesky decomposition is more
        efficient and produces same result as standard GPR.
    """
    Np = 1000
    X, Y = data_gen(Np)

    m1 = GPRCached(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                   mean_function=gp.mean_functions.Constant())
    m1.optimize()
    m2 = GPRCached(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                   mean_function=gp.mean_functions.Constant())
    m2.set_parameter_dict(m1.get_parameter_dict())

    @time_it
    def reset_points_n_times(m, X, Y):
        Xall = np.append(m.X.value, X, axis=0)
        Yall = np.append(m.Y.value, Y, axis=0)
        Ls = []
        start = m.X.value.shape[0]
        for i in range(start, Xall.shape[0]):
            m.set_data_points(Xall[:i + 1, :], Yall[:i + 1, :])
            Ls.append(m.cholesky.value)
        return Ls

    @time_it
    def add_point_n_times(m, X, Y):
        Ls = []
        for i, (x, y) in enumerate(zip(X, Y)):
            m.add_data_point(x, y)
            Ls.append(m.cholesky.value)
        return Ls

    Nn = 50
    Xnew, Ynew = data_gen(Nn)
    m1_time, m1_Ls = reset_points_n_times(m1, Xnew, Ynew)
    m2_time, m2_Ls = add_point_n_times(m2, Xnew, Ynew)

    factor = 2.5
    assert m1_time > m2_time * factor
    for L1, L2 in zip(m1_Ls, m2_Ls):
        assert np.allclose(L1, L2)


def test_cholesky_downdate(data_gen):
    """ Test that incremental downdate of the Cholesky decomposition produces
        same result as standard GPR.
    """
    Np = 1000
    X, Y = data_gen(Np)

    m1 = GPRCached(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                   mean_function=gp.mean_functions.Constant())
    m1.optimize()
    m2 = GPRCached(X, Y, kern=gp.kernels.RBF(X.shape[1]),
                   mean_function=gp.mean_functions.Constant())
    m2.set_parameter_dict(m1.get_parameter_dict())

    @time_it
    def remove_point_n_times(m, indexes):
        Ls = []
        for i in indexes:
            m.remove_data_point(i)
            Ls.append(m.cholesky.value)
        return Ls

    @time_it
    def reset_points_n_times(m, indexes):
        Ls = []
        for i in indexes:
            m.set_data_points(np.delete(m.X.value, i, axis=0),
                              np.delete(m.X.value, i, axis=0))
            Ls.append(m.cholesky.value)
        return Ls

    Nr = 50
    indexes = np.append(np.array([Np - 1, 0]),
                        np.random.randint(1, Np - Nr, Nr - 2))
    m1_time, m1_Ls = reset_points_n_times(m1, indexes)
    m2_time, m2_Ls = remove_point_n_times(m2, indexes)

    #factor = 1
    #assert m1_time > m2_time * factor

    for L1, L2 in zip(m1_Ls, m2_Ls):
        assert np.allclose(L1, L2)
