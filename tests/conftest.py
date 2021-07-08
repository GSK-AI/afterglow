import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="whether to run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="only running unit tests by default")
    run_integration = config.getoption("--integration")
    for item in items:
        if not run_integration and "integration" in item.keywords:
            item.add_marker(skip_integration)
