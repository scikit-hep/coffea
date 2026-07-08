import os

import awkward as ak
import pytest
import uproot

from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.mapping import BufferCache, SimplePreloadedColumnSource
from coffea.nanoevents.schemas import BaseSchema
from coffea.processor.test_items import NanoEventsProcessor


def test_from_preloaded_honors_buffer_cache(tests_directory):
    rootdir = uproot.open(f"{tests_directory}/samples/nano_dy.root")
    tree = rootdir["Events"]
    arrays = tree.arrays(["nMuon", "Muon_pt"], how=dict)
    src = SimplePreloadedColumnSource(
        arrays, rootdir.file.uuid, tree.num_entries, object_path="/Events"
    )

    cache = BufferCache(cache=None, codec=None)
    factory = NanoEventsFactory.from_preloaded(
        src, buffer_cache=cache, schemaclass=BaseSchema
    )
    assert factory.buffer_cache is cache

    events = factory.events()
    ak.materialize(events.Muon_pt)
    assert len(cache) > 0


def test_preloaded_nanoevents():
    pytest.xfail("preloaded nanoevents doesn't support dask yet")

    columns = [
        "nMuon",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mass",
        "Muon_charge",
        "nJet",
        "Jet_eta",
    ]
    p = NanoEventsProcessor(columns=columns)

    rootdir = uproot.open(os.path.abspath("tests/samples/nano_dy.root"))
    tree = rootdir["Events"]
    arrays = tree.arrays(columns, how=dict)
    src = SimplePreloadedColumnSource(
        arrays, rootdir.file.uuid, tree.num_entries, object_path="/Events"
    )
    print(arrays)

    events = NanoEventsFactory.from_preloaded(
        src, metadata={"dataset": "ZJets"}
    ).events()
    hists = p.process(events)

    print(hists)
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6

    with pytest.raises(AttributeError):
        print(events.Muon.matched_jet)
