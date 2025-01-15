import os

import awkward as ak
import dask_awkward as dak
import pytest

from coffea.nanoevents import DelphesSchema, NanoEventsFactory


def _events():
    path = os.path.abspath("tests/samples/delphes.root")
    factory = NanoEventsFactory.from_root(
        {path: "Delphes"},
        schemaclass=DelphesSchema,
        delayed=True,
    )
    return factory.events()


@pytest.fixture(scope="module")
def events():
    return _events()


def test_listify(events):
    assert ak.to_list(events.CaloJet02[0].compute())


@pytest.mark.parametrize(
    "collection",
    [
        "CaloJet02",
        "CaloJet04",
        "CaloJet08",
        "CaloJet15",
        "EFlowNeutralHadron",
        "EFlowPhoton",
        "EFlowTrack",
        "Electron",
        "Event",
        "EventLHEF",
        "GenJet",
        "GenJet02",
        "GenJet04",
        "GenJet08",
        "GenJet15",
        "GenMissingET",
        "Jet",
        "MissingET",
        "Muon",
        "Particle",
        "ParticleFlowJet02",
        "ParticleFlowJet04",
        "ParticleFlowJet08",
        "ParticleFlowJet15",
        "Photon",
        "ScalarHT",
        "Tower",
        "Track",
        "TrackJet02",
        "TrackJet04",
        "TrackJet08",
        "TrackJet15",
        "WeightLHEF",
    ],
)
def test_collection_exists(events, collection):
    assert hasattr(events, collection)


@pytest.mark.parametrize(
    "collection",
    [
        "CaloJet02",
        "CaloJet04",
        "CaloJet08",
        "CaloJet15",
        "GenJet",
        "GenJet02",
        "GenJet04",
        "GenJet08",
        "GenJet15",
        "Jet",
        "ParticleFlowJet02",
        "ParticleFlowJet04",
        "ParticleFlowJet08",
        "ParticleFlowJet15",
        "TrackJet02",
        "TrackJet04",
        "TrackJet08",
        "TrackJet15",
    ],
)
def test_lorentz_vectorization(collection, events):
    mask = dak.num(events[collection], axis=1) > 0
    assert (
        ak.parameters(events[collection][mask][0].Area._meta)["__record__"]
        == "LorentzVector"
    )
    assert (
        ak.parameters(events[collection][mask][0].SoftDroppedJet._meta)["__record__"]
        == "LorentzVector"
    )
    assert (
        ak.parameters(events[collection][mask][0].SoftDroppedSubJet1._meta)[
            "__record__"
        ]
        == "LorentzVector"
    )
    assert (
        ak.parameters(events[collection][mask][0].SoftDroppedSubJet2._meta)[
            "__record__"
        ]
        == "LorentzVector"
    )


@pytest.mark.parametrize(
    "collection",
    [
        "CaloJet02",
        "CaloJet04",
        "CaloJet08",
        "CaloJet15",
        "GenJet",
        "GenJet02",
        "GenJet04",
        "GenJet08",
        "GenJet15",
        "Jet",
        "ParticleFlowJet02",
        "ParticleFlowJet04",
        "ParticleFlowJet08",
        "ParticleFlowJet15",
        "TrackJet02",
        "TrackJet04",
        "TrackJet08",
        "TrackJet15",
    ],
)
def test_nested_lorentz_vectorization(collection, events):
    mask = dak.num(events[collection], axis=1) > 0
    assert ak.all(ak.num(events[collection].PrunedP4_5.compute(), axis=2) == 5)
    assert (
        ak.parameters(events[collection][mask].PrunedP4_5[0, 0]._meta.layout.content)[
            "__record__"
        ]
        == "LorentzVector"
    )

    assert ak.all(ak.num(events[collection].SoftDroppedP4_5.compute(), axis=2) == 5)
    assert (
        ak.parameters(
            events[collection][mask].SoftDroppedP4_5[0, 0]._meta.layout.content
        )["__record__"]
        == "LorentzVector"
    )

    assert ak.all(ak.num(events[collection].TrimmedP4_5.compute(), axis=2) == 5)
    assert (
        ak.parameters(events[collection][mask].TrimmedP4_5[0, 0]._meta.layout.content)[
            "__record__"
        ]
        == "LorentzVector"
    )
