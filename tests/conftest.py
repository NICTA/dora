import pytest


@pytest.fixture
def setup_delaunay():

    lower = [0, 0]
    upper = [1, 1]

    n_samples = 10
    explore_priority = 0.0001

    return lower, upper, n_samples, explore_priority
