"""Regression tests for the shared ``dask_client`` fixture.

The session-scoped cluster must not become the process-wide default
scheduler: if it did, every ``.compute()`` in tests running after the first
``dask_client`` test would silently route through ``distributed`` (breaking
e.g. the FCC nanoevents tests, whose behaviors contain unpicklable lambdas).

Tests within a file run in definition order, so the pair below
deterministically exercises "a dask_client test ran earlier in this process".
"""

import pytest

distributed = pytest.importorskip("distributed")
import dask  # noqa: E402
import dask.base  # noqa: E402
import dask.delayed  # noqa: E402


@pytest.mark.dask_client
def test_computations_route_through_cluster(dask_client):
    # Inside a dask_client test, plain .compute() must use the shared cluster.
    assert dask.base.get_scheduler() == dask_client.get

    assert dask.delayed(lambda: 21)().compute() * 2 == 42


def test_default_scheduler_not_leaked():
    # After (and outside) a dask_client test, the process default scheduler
    # must be untouched by the session cluster.
    scheduler = dask.base.get_scheduler()
    client_gets = {
        client.get
        for client in distributed.Client._instances
        if client.status == "running"
    }
    assert scheduler not in client_gets
