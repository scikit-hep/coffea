import os

import pytest


@pytest.fixture(scope="module")
def tests_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def _dask_session_client():
    """A single, resource-bounded ``distributed.Client`` shared across the session.

    ``set_as_default=False`` keeps the client from becoming the process-wide
    default scheduler: otherwise every ``.compute()`` in tests that run after
    the first ``dask_client`` test would silently route through the cluster
    (e.g. the FCC nanoevents tests, whose behaviors are not picklable, fail
    with serialization errors in serial runs).
    """
    distributed = pytest.importorskip("distributed")

    with distributed.Client(
        n_workers=1,
        threads_per_worker=2,
        processes=True,
        dashboard_address=None,
        set_as_default=False,
    ) as client:
        yield client


@pytest.fixture
def dask_client(_dask_session_client):
    """Route dask computations through the shared cluster for this test only."""
    import dask

    with dask.config.set(scheduler=_dask_session_client.get):
        with _dask_session_client.as_current():
            yield _dask_session_client
