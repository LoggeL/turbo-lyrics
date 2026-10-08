import pytest


def pytest_addoption(parser):
    parser.addoption("--models", action="store_true", help="run tests that download and run the real models")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--models"):
        return
    skip = pytest.mark.skip(reason="needs --models")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)
