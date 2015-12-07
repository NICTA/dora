"""
PyTest Modele for Sampler Class Testing
"""
import dora.active_sampling as sampling
import numpy as np


def equality(a, b, tol=1e-4):
    return np.sum((a - b) ** 2) < tol


def ground_truth(x):
    return (np.sum((x - 0.5) ** 2, axis=-1) < 0.1).astype(float)


def test_delaunay():

    np.random.seed(100)

    lower = [0, 0]
    upper = [1, 1]

    n_samples = 10
    explore_priority = 0.0001

    sampler = sampling.Delaunay(lower, upper,
                                explore_priority=explore_priority)

    xq, uid = sampler.pick()

    assert equality(xq, np.array([0, 0]))
    assert uid == 'e99bf8c47ac0cebc187804b541ba1ac6'

    yq = ground_truth(xq)
    assert sampler.update(uid, yq) == 0

    for i in range(n_samples - 1):

        xq, uid = sampler.pick()
        yq = ground_truth(xq)
        sampler.update(uid, yq)

    assert equality(np.array(sampler.X), np.array([
                    [0., 0.],
                    [1., 0.],
                    [0., 1.],
                    [1., 1.],
                    [0.5, 0.5],
                    [0.5, 0.24981292],
                    [0.24981292, 0.5],
                    [0.75018708, 0.5],
                    [0.5, 0.75018708],
                    [0.87518701, 0.5]]))


def test_gp():

    np.random.seed(100)

    lower = [0, 0]
    upper = [1, 1]

    n_samples = 10
    n_initial_sample = 50

    X_train = sampling.random_sample(lower, upper, n_initial_sample)
    y_train = ground_truth(X_train)

    sampler = sampling.GaussianProcess(lower, upper, X_train, y_train,
                                       add_train_data=False)

    xq, uid = sampler.pick()

    assert equality(xq, np.array([0, 0]))
    assert uid == '99970dae22a50c6210333d05b581762e'

    yq = ground_truth(xq)
    assert sampler.update(uid, yq) == 0

    for i in range(n_samples - 1):

        xq, uid = sampler.pick()
        yq = ground_truth(xq)
        sampler.update(uid, yq)

    assert equality(np.array(sampler.X), np.array([
                    [0., 0.],
                    [1., 0.],
                    [0., 1.],
                    [1., 1.],
                    [0.5, 0.5],
                    [0.36052525, 0.54937526],
                    [0.40648024, 0.34281852],
                    [0.2526294, 0.31233231],
                    [0.20180019, 0.56829352],
                    [0.48608311, 0.67256845]]))
