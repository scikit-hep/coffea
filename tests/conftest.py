import os

import pytest


@pytest.fixture(scope="module")
def tests_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def dask_client():
    """A single, resource-bounded ``distributed.Client`` shared across the session.

    The ``dask_client``-marked tests only need *a* distributed scheduler so that
    ``dask.compute`` dispatches through it; they do not depend on cluster
    isolation. Spinning up a fresh ``LocalCluster`` per test (the previous
    pattern) cost several seconds each and dominated the serial ``dask_client``
    leg. Reusing one bounded cluster exercises the identical distributed
    code path (cross-process task serialization + scheduler execution) while
    paying the startup cost once per worker.

    Bounded to a small, fixed footprint so the leg can run under pytest-xdist
    without oversubscribing CI runners.
    """
    distributed = pytest.importorskip("distributed")

    with distributed.Client(
        n_workers=1,
        threads_per_worker=2,
        processes=True,
        dashboard_address=None,
    ) as client:
        yield client
