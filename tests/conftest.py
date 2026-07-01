import os

import pytest


@pytest.fixture(scope="module")
def tests_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def dask_client():
    """A single, resource-bounded ``distributed.Client`` shared across the session."""
    distributed = pytest.importorskip("distributed")

    with distributed.Client(
        n_workers=1,
        threads_per_worker=2,
        processes=True,
        dashboard_address=None,
    ) as client:
        yield client
