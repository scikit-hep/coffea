# from coffea.nanoevents import NanoEventsFactory, PDUNESchema, PHYSLITESchema
import os

import awkward as ak
import pytest

from coffea.nanoevents import NanoEventsFactory, PDUNESchema

# import numpy as np


# fpath="tests/samples/DAOD_PHYSLITE_21.2.108.0.art.pool.root"
# tpath="CollectionTree"
# events = NanoEventsFactory.from_root(fpath,treepath=tpath,schemaclass=PHYSLITESchema).events()

# fpath = "samples/pduneana.root"
# tpath = "pduneana/beamana"
# events = NanoEventsFactory.from_root(
#     fpath, treepath=tpath, schemaclass=PDUNESchema
# ).events()
#
# print(events)


@pytest.fixture(scope="module")
def events():
    path = os.path.abspath("tests/samples/pduneana.root")
    factory = NanoEventsFactory.from_root(
        {path: "pduneana/beamana"}, schemaclass=PDUNESchema
    )
    return factory.events()


def test_listify(events):
    assert ak.to_list(events[0])
