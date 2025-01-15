import os

import awkward as ak
import dask_awkward as dak
import pytest
import uproot

from coffea.nanoevents import NanoEventsFactory, TreeMakerSchema


@pytest.fixture(scope="module")
def events():
    path = os.path.abspath("tests/samples/treemaker.root")
    events = NanoEventsFactory.from_root(
        {path: "PreSelection"}, schemaclass=TreeMakerSchema, delayed=True
    ).events()
    return events


def test_listify(events):
    assert ak.to_list(events[0])


@pytest.mark.parametrize(
    "collection",
    ["HT", "MET", "Weight"],
)
def test_collection_exists(events, collection):
    assert hasattr(events, collection)


@pytest.mark.parametrize(
    "collection,arr_type",
    [
        ("Muons", "PtEtaPhiELorentzVector"),
        (
            "Electrons",
            "PtEtaPhiELorentzVector",
        ),
        ("Photons", "PtEtaPhiELorentzVector"),
        ("Jets", "PtEtaPhiELorentzVector"),
        ("JetsAK8", "PtEtaPhiELorentzVector"),
        ("Tracks", "LorentzVector"),
        ("GenParticles", "PtEtaPhiELorentzVector"),
        ("PrimaryVertices", "ThreeVector"),
    ],
)
def test_lorentzvector_behavior(collection, arr_type, events):
    assert dak.type(events[collection])
    assert ak.parameters(events[collection].compute().layout.content)[
        "__record__"
    ].startswith(arr_type)


@pytest.mark.parametrize(
    "collection,subcollection,arr_type,element",
    [
        ("JetsAK8", "subjets", "PtEtaPhiELorentzVector", "pt"),
        ("Tracks", "hitPattern", "int32", None),
    ],
)
def test_nested_collection(collection, subcollection, arr_type, element, events):
    assert dak.type(events[collection][subcollection])
    assert dak.type(events[collection][subcollection + "Counts"])
    if subcollection == "subjets":
        assert ak.parameters(
            events[collection][subcollection].compute().layout.content.content
        )["__record__"].startswith(arr_type)
    if subcollection == "hitPattern":
        assert ak.type(
            events[collection][subcollection].compute().layout.content.content
        ).content.primitive.startswith(arr_type)
    if element is None:
        subcol = events[collection][subcollection].compute()
        assert ak.all(
            events[collection][subcollection + "Counts"].compute()
            == ak.count(subcol, axis=-1)
        )
    else:
        assert ak.all(
            events[collection][subcollection + "Counts"].compute()
            == dak.count(events[collection][subcollection][element], axis=-1).compute()
        )


def test_uproot_write():
    path = os.path.abspath("tests/samples/treemaker.root")
    orig_events = NanoEventsFactory.from_root(
        {path: "PreSelection"}, schemaclass=TreeMakerSchema, delayed=False
    ).events()

    with uproot.recreate("treemaker_write_test.root") as f:
        f["PreSelection"] = TreeMakerSchema.uproot_writeable(orig_events)

    test_events = NanoEventsFactory.from_root(
        {"treemaker_write_test.root": "PreSelection"},
        schemaclass=TreeMakerSchema,
        delayed=False,
    ).events()

    # Checking event structure
    assert len(orig_events) == len(test_events)
    assert ak.all(orig_events.HT == test_events.HT)
    # Checking composite structure and their behavior
    assert ak.all(orig_events.Jets.pt == test_events.Jets.pt)
    assert ak.all(orig_events.JetsAK8.x == test_events.JetsAK8.x)
    # Checking nested composite structure and their behavior
    assert ak.all(orig_events.Tracks.hitPattern == test_events.Tracks.hitPattern)
    assert ak.all(orig_events.JetsAK8.subjets.pt == test_events.JetsAK8.subjets.pt)
    assert ak.all(orig_events.JetsAK8.subjets.x == test_events.JetsAK8.subjets.x)
