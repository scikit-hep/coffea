import awkward as ak

from coffea import processor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.trace import trace


def _make_events(preload):
    return NanoEventsFactory.from_root(
        {"tests/samples/nano_dy.root": "Events"},
        schemaclass=NanoAODSchema,
        mode="virtual",
        preload=preload,
    ).events()


def test_preload_nanoevents():
    # no preload
    events = _make_events(preload=None)
    assert events.attrs["@events_factory"]._mapping.preloaded_arrays is None

    # list of strings
    branches = ["nJet", "Jet_pt"]
    events = _make_events(preload=branches)
    assert set(events.attrs["@events_factory"]._mapping.preloaded_arrays.keys()) == set(
        branches
    )

    # frozenset of strings
    branches = frozenset(["nJet", "Jet_pt", "Jet_eta"])
    events = _make_events(preload=branches)
    assert set(events.attrs["@events_factory"]._mapping.preloaded_arrays.keys()) == set(
        branches
    )

    # callable (filter_branch style)
    branches = {"nJet", "Jet_pt"}
    events = _make_events(preload=lambda b: b.name in branches)
    assert (
        set(events.attrs["@events_factory"]._mapping.preloaded_arrays.keys())
        == branches
    )

    # preloaded data matches non-preloaded
    events_normal = _make_events(preload=None)
    events_preloaded = _make_events(preload=["Jet_pt", "Jet_eta"])
    assert ak.array_equal(events_preloaded.Jet.pt, events_normal.Jet.pt)
    assert ak.array_equal(events_preloaded.Jet.eta, events_normal.Jet.eta)

    # empty list
    events = _make_events(preload=[])
    assert events.attrs["@events_factory"]._mapping.preloaded_arrays == {}

    # nonexistent branch is silently ignored
    events = _make_events(preload=["nonexistent_branch"])
    assert events.attrs["@events_factory"]._mapping.preloaded_arrays == {}


def _preload_check(events):
    dataset = events.metadata["dataset"]
    mapping = events.attrs["@events_factory"]._mapping
    preloaded = mapping.preloaded_arrays
    keys = set(preloaded.keys()) if preloaded is not None else None
    return {
        dataset: {
            "preloaded_keys": keys,
            "jets_pt_sum": float(ak.sum(events.Jet.pt)),
        }
    }


FILESET = {
    "ZJets": {
        "files": {"tests/samples/nano_dy.root": "Events"},
    },
    "Data": {
        "files": {"tests/samples/nano_dimuon.root": "Events"},
    },
}

FILESET_WITH_PRELOAD = {
    "ZJets": {
        "preload": ["nJet", "Jet_pt"],
        "files": {"tests/samples/nano_dy.root": "Events"},
    },
    "Data": {
        "preload": ["nMuon", "Muon_pt"],
        "files": {"tests/samples/nano_dimuon.root": "Events"},
    },
}


def test_preload_executor():
    runner = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=NanoAODSchema,
    )
    proc = _preload_check

    # --- preprocess ---

    # no preload
    chunks = list(runner.preprocess(FILESET))
    assert all(c.preload is None for c in chunks)

    # preload via fileset metadata
    chunks = list(runner.preprocess(FILESET_WITH_PRELOAD))
    for c in chunks:
        assert c.preload is not None
        if c.dataset == "ZJets":
            assert c.preload == frozenset(["nJet", "Jet_pt"])
        elif c.dataset == "Data":
            assert c.preload == frozenset(["nMuon", "Muon_pt"])

    # preload via tracing
    chunks = list(runner.preprocess(FILESET, trace=trace, processor_instance=proc))
    for c in chunks:
        assert c.preload is not None
        assert "Jet_pt" in c.preload
        assert "nJet" in c.preload

    # tracing takes precedence over fileset preload
    chunks = list(
        runner.preprocess(FILESET_WITH_PRELOAD, trace=trace, processor_instance=proc)
    )
    for c in chunks:
        assert c.preload is not None
        assert "Jet_pt" in c.preload
        assert "Muon_pt" not in c.preload

    # --- run ---

    # no preload
    out = runner.run(FILESET, proc)["out"]
    assert out["ZJets"]["preloaded_keys"] is None
    assert out["Data"]["preloaded_keys"] is None

    # preload via fileset metadata
    out = runner.run(FILESET_WITH_PRELOAD, proc)["out"]
    assert out["ZJets"]["preloaded_keys"] == {"nJet", "Jet_pt"}
    assert out["Data"]["preloaded_keys"] == {"nMuon", "Muon_pt"}

    # preload via tracing
    out = runner.run(FILESET, proc, trace=trace)["out"]
    assert "Jet_pt" in out["ZJets"]["preloaded_keys"]
    assert "nJet" in out["ZJets"]["preloaded_keys"]
    assert "Jet_pt" in out["Data"]["preloaded_keys"]
    assert "nJet" in out["Data"]["preloaded_keys"]

    # tracing takes precedence over fileset preload
    out = runner.run(FILESET_WITH_PRELOAD, proc, trace=trace)["out"]
    assert "Jet_pt" in out["ZJets"]["preloaded_keys"]
    assert "Muon_pt" not in out["Data"]["preloaded_keys"]

    # --- __call__ ---

    # no preload
    out = runner(FILESET, proc)
    assert out["ZJets"]["preloaded_keys"] is None
    assert out["Data"]["preloaded_keys"] is None

    # preload via fileset metadata
    out = runner(FILESET_WITH_PRELOAD, proc)
    assert out["ZJets"]["preloaded_keys"] == {"nJet", "Jet_pt"}
    assert out["Data"]["preloaded_keys"] == {"nMuon", "Muon_pt"}

    # preload via tracing
    out = runner(FILESET, proc, trace=trace)
    assert "Jet_pt" in out["ZJets"]["preloaded_keys"]
    assert "nJet" in out["ZJets"]["preloaded_keys"]
    assert "Jet_pt" in out["Data"]["preloaded_keys"]
    assert "nJet" in out["Data"]["preloaded_keys"]

    # tracing takes precedence over fileset preload
    out = runner(FILESET_WITH_PRELOAD, proc, trace=trace)
    assert "Jet_pt" in out["ZJets"]["preloaded_keys"]
    assert "Muon_pt" not in out["Data"]["preloaded_keys"]
