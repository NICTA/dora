"""
PyTest Modele for Sampler Class Testing.
"""
import dora.active_sampling as sampling
import numpy as np
import os

# Set this to True if a new test reference is to be created
# Otherwise, keep this as False for new tests
NEW_TEST_REFERENCE = False


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


def test_gp():

    np.random.seed(100)

    lower = [0, 0]
    upper = [1, 1]
    acq_name = 'sigmoid'
    n_train = 49

    sampler = sampling.GaussianProcess(lower, upper, acq_name=acq_name,
                                       n_train=n_train)

    n_samples = 301
    retrain = [100, 200]

    xq, uid = sampler.pick()
    yq = ground_truth(xq)
    ind = sampler.update(uid, yq)

    cwd = os.path.dirname(__file__)  # os.environ.get('TRAVIS_BUILD_DIR')
    print(cwd)

    if NEW_TEST_REFERENCE:

        np.savez('%s/data/gp_ref_data_0.npz' % cwd, xq=xq, ind=ind)

    else:

        gp_ref_data_0 = np.load('%s/data/gp_ref_data_0.npz' % cwd)
        assert np.allclose(xq, gp_ref_data_0['xq'])
        assert ind == gp_ref_data_0['ind']

    # Start active sampling!
    for i in range(n_samples):

        # Pick a location to sample
        xq, uid = sampler.pick()

        # Sample that location
        yq_true = ground_truth(xq)

        # Update the sampler about the new observation
        ind = sampler.update(uid, yq_true)

        if i in retrain:

            sampler.train()

            if NEW_TEST_REFERENCE:
                np.savez('%s/data/gp_ref_data_%d.npz' % (cwd, i),
                         xq=xq,
                         ind=ind)
            else:
                gp_ref_data_i = np.load('%s/data/gp_ref_data_%d.npz'
                                        % (cwd, i))
                assert np.allclose(xq, gp_ref_data_i['xq'])
                assert ind == gp_ref_data_i['ind']

    if NEW_TEST_REFERENCE:
        np.savez('%s/data/gp_ref_data_final.npz' % cwd,
                 X=sampler.X(),
                 y=sampler.y(),
                 v=sampler.virtual_flag())
    else:
        gp_ref_data_final = \
            np.load('%s/data/gp_ref_data_final.npz' % cwd)
        assert np.allclose(sampler.X(), gp_ref_data_final['X'])
        assert np.allclose(sampler.y(), gp_ref_data_final['y'])
        assert np.allclose(sampler.virtual_flag(), gp_ref_data_final['v'])
