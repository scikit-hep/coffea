import awkward as ak
import numpy as np

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory


def test_base_schema(tests_directory):
    path = f"{tests_directory}/samples/nano_dimuon_rntuple.root"
    factory = NanoEventsFactory.from_root({path: "Events"}, schemaclass=BaseSchema)
    events = factory.events()

    assert all(f in events.fields for f in ["Electron_pt", "Muon_eta", "Tau_phi"])

    array = events.Electron_pt
    assert np.isclose(array[0, 0], 17.627574920654297)

    ak.materialize(events)

    assert "_collection1" not in events.fields


def test_nanoaod_schema(tests_directory):
    path = f"{tests_directory}/samples/nano_dimuon_rntuple.root"
    factory = NanoEventsFactory.from_root({path: "Events"}, schemaclass=NanoAODSchema)
    events = factory.events()

    assert all(
        f in events.Electron.fields for f in ["pt", "eta", "phi", "mass", "charge"]
    )
    assert all(f in events.Muon.fields for f in ["pt", "eta", "phi", "mass", "charge"])
    assert all(f in events.Tau.fields for f in ["pt", "eta", "phi", "mass", "charge"])
    assert all(f in events.Photon.fields for f in ["pt", "eta", "phi"])
    assert all(f in events.Jet.fields for f in ["pt", "eta", "phi", "mass"])

    assert np.isclose(events.Electron.pt[0, 0], 17.627574920654297)

    ak.materialize(events)

    assert "" not in events.fields
