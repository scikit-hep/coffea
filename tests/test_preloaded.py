import os

import pytest
import uproot

from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.mapping import SimplePreloadedColumnSource
from coffea.processor.test_items import NanoEventsProcessor


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
