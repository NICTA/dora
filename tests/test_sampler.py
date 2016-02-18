"""
PyTest Modele for Sampler Class Testing.
"""
import numpy as np
import os


def test_gp(update_ref_data):

    verify_common_samplers(update_ref_data=update_ref_data,
                           sampling_method='GaussianProcess')


def test_delaunay(update_ref_data):

    verify_common_samplers(update_ref_data=update_ref_data,
                           sampling_method='Delaunay')


def verify_common_samplers(update_ref_data=False,
                           sampling_method='GaussianProcess'):

    from demos.demo_samplers_python_api import main

    sampler = main(sampling_method=sampling_method)

    cwd = os.path.dirname(__file__)
    filename = '%s/data/ref_data_%s.npz' % (cwd, sampling_method)
    if update_ref_data:
        np.savez(filename,
                 X=sampler.X(),
                 y=sampler.y(),
                 v=sampler.virtual_flag())
    else:
        gp_ref_data_final = \
            np.load(filename)
        assert np.allclose(sampler.X(), gp_ref_data_final['X'])
        assert np.allclose(sampler.y(), gp_ref_data_final['y'])
        assert np.allclose(sampler.virtual_flag(), gp_ref_data_final['v'])
