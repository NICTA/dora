"""
PyTest Modele for Sampler Class Testing.
"""
import dora.active_sampling as sampling
import numpy as np
import os

# # Set this to True if a new test reference is to be created
# # Otherwise, keep this as False for new tests
# update_ref_data = False


def ground_truth(x):
    return (np.sum((x - 0.5) ** 2, axis=-1) < 0.1).astype(float)


def test_delaunay(update_ref_data):

    lower = [0, 0]
    upper = [1, 1]

    n_samples = 10
    explore_priority = 0.0001

    sampler = sampling.Delaunay(lower, upper,
                                explore_priority=explore_priority)

    xq, uid = sampler.pick()

    assert np.allclose(xq, np.array([0, 0]))

    yq = ground_truth(xq)
    assert sampler.update(uid, yq) == 0

    for i in range(n_samples - 1):

        xq, uid = sampler.pick()
        yq = ground_truth(xq)
        sampler.update(uid, yq)

    assert np.allclose(sampler.X(), np.array([
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


def test_gp(update_ref_data):

    from demos.demo_gp_python_api import main

    sampler = main()

    cwd = os.path.dirname(__file__)
    if update_ref_data:
        np.savez('%s/data/gp_ref_data.npz' % cwd,
                 X=sampler.X(),
                 y=sampler.y(),
                 v=sampler.virtual_flag())
    else:
        gp_ref_data_final = \
            np.load('%s/data/gp_ref_data.npz' % cwd)
        assert np.allclose(sampler.X(), gp_ref_data_final['X'])
        assert np.allclose(sampler.y(), gp_ref_data_final['y'])
        assert np.allclose(sampler.virtual_flag(), gp_ref_data_final['v'])
