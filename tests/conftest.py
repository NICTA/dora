"""
PyTest Configuration.
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--update-ref-data", action="store", default=False,
                     help="Set to True if reference data is to be updated")


@pytest.fixture()
def update_ref_data(request):
    return request.config.option.update_ref_data
