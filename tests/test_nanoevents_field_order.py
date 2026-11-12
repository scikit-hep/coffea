import os
import subprocess
import sys

import pytest

PROBE = """
import sys
from coffea.nanoevents import NanoEventsFactory, {schema}
events = NanoEventsFactory.from_root(
    {{sys.argv[1]: "{tree}"}}, schemaclass={schema}, mode="virtual"
).events()
print(",".join(events.fields))
"""


@pytest.mark.parametrize(
    "sample, tree, schema",
    [
        ("nano_dy.root", "Events", "NanoAODSchema"),
        ("pfnano.root", "Events", "PFNanoAODSchema"),
        ("scouting_nano.root", "Events", "ScoutingNanoAODSchema"),
        ("delphes.root", "Delphes", "DelphesSchema"),
    ],
)
def test_field_order_is_stable_across_hash_seeds(sample, tree, schema):
    path = os.path.abspath(f"tests/samples/{sample}")
    orders = {
        subprocess.run(
            [sys.executable, "-c", PROBE.format(schema=schema, tree=tree), path],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(orders) == 1
    assert next(iter(orders))
