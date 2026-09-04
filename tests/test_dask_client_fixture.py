"""The shared cluster serves ``dask_client`` tests only; the second test relies on
definition order to run after one."""

import pytest

distributed = pytest.importorskip("distributed")
import dask  # noqa: E402
import dask.base  # noqa: E402


@pytest.mark.dask_client
def test_computations_route_through_cluster(dask_client):
    assert dask.base.get_scheduler() == dask_client.get

    assert dask.delayed(lambda: 21)().compute() * 2 == 42


def test_default_scheduler_not_leaked():
    scheduler = dask.base.get_scheduler()
    client_gets = {
        client.get
        for client in distributed.Client._instances
        if client.status == "running"
    }
    assert scheduler not in client_gets
