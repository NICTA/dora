"""
PyTest Configuration.
"""


def pytest_addoption(parser):
    parser.addoption("--update-ref-data", action="store", default=False,
                     help="Set to True if reference data is to be updated")


def pytest_funcarg__update_ref_data(request):
    return request.config.option.update_ref_data
