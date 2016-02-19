"""
PyTest Modele for Sampler Class Testing.
"""
import numpy as np
import os


def test_gp(update_ref_data):
    """
    Test the GaussianProcess sampler.

    This is a wrapper on the general test for the GaussianProcess method

    See Also
    --------
    verify_common_samplers : The general sampler testing framework
    """
    verify_common_samplers(update_ref_data=update_ref_data,
                           sampling_method='GaussianProcess')


def test_delaunay(update_ref_data):
    """
    Test the Delaunay sampler.

    This is a wrapper on the general test for the Delaunay method

    See Also
    --------
    verify_common_samplers : The general sampler testing framework
    """
    verify_common_samplers(update_ref_data=update_ref_data,
                           sampling_method='Delaunay')


def verify_common_samplers(update_ref_data=False,
                           sampling_method='GaussianProcess'):
    """
    Test a general sampler's output.

        For any implemented sampling method, this function tests if the final
        collection of active sampled outputs are the same as before. If the
        reference data is to be updated, it can also do so by setting the
        corresponding fixture through 'py.test --update-ref-data=True'.

    .. note ::

        This will require the corresponding demonstration to return the
        sampler instance to be tested. The demo cannot have a
        'matplotlib.pyplot.show()' call or anything that pauses the script
        (obviously). If the sampling method has a random element, a seed must
        be set in the demo itself.

    Parameters
    ----------
    update_ref_data : bool, optional
        To update the reference data or not
    sampling_method : str, optional
        The sampling method to test and verify
    """
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
        ref_data = np.load(filename)
        np.testing.assert_allclose(sampler.X(), ref_data['X'])
        np.testing.assert_allclose(sampler.y(), ref_data['y'])
        np.testing.assert_allclose(sampler.virtual_flag(), ref_data['v'])
