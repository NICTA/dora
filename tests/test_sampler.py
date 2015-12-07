"""
PyTest Modele for Sampler Class Testing
"""
import dora.active_sampling as sampling
import numpy as np


def equality(a, b, tol=1e-4):
    return np.sum((a - b) ** 2) < tol


def ground_truth(x):
    return (np.sum((x - 0.5) ** 2) < 0.1).astype(float)


def test_delaunay(setup_delaunay):

    lower, upper, n_samples, explore_priority = setup_delaunay

    sampler = sampling.Delaunay(lower, upper,
                                explore_priority=explore_priority)

    np.random.seed(100)
    xq, uid = sampler.pick()

    assert equality(xq, np.array([0, 0]))
    assert uid == 'e99bf8c47ac0cebc187804b541ba1ac6'

    yq = ground_truth(xq)
    ind = sampler.update(uid, yq)
    assert ind == 0

    for i in range(n_samples - 1):

        xq, uid = sampler.pick()
        observation = ground_truth(xq)
        sampler.update(uid, observation)

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
