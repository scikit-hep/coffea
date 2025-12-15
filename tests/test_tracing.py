import pytest

from coffea.nanoevents import NanoEventsFactory


def _analysis(events):
    # COPIED FROM AGC: https://github.com/iris-hep/calver-coffea-agc-demo/blob/2025_IRISHEP_Training/agc-coffea-2025-virtual-arrays-and-executors.ipynb
    import awkward as ak
    import numpy as np

    # pT > 30 GeV for leptons, > 25 GeV for jets
    selected_electrons = events.Electron[
        (events.Electron.pt > 30) & (np.abs(events.Electron.eta) < 2.1)
    ]
    selected_muons = events.Muon[
        (events.Muon.pt > 30) & (np.abs(events.Muon.eta) < 2.1)
    ]
    selected_jets = events.Jet[(events.Jet.pt > 25) & (np.abs(events.Jet.eta) < 2.4)]

    # single lepton requirement
    event_filters = (
        ak.count(selected_electrons.pt, axis=1) + ak.count(selected_muons.pt, axis=1)
    ) == 1
    # at least four jets
    event_filters = event_filters & (ak.count(selected_jets.pt, axis=1) >= 4)
    # at least two b-tagged jets ("tag" means score above threshold)
    B_TAG_THRESHOLD = 0.5
    event_filters = event_filters & (
        ak.sum(selected_jets.btagDeepFlavB > B_TAG_THRESHOLD, axis=1) >= 2
    )

    # apply filters
    selected_jets = selected_jets[event_filters]

    trijet = ak.combinations(
        selected_jets, 3, fields=["j1", "j2", "j3"]
    )  # trijet candidate
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # four-momentum of tri-jet system

    trijet["max_btag"] = np.maximum(
        trijet.j1.btagDeepFlavB,
        np.maximum(trijet.j2.btagDeepFlavB, trijet.j3.btagDeepFlavB),
    )
    trijet = trijet[
        trijet.max_btag > B_TAG_THRESHOLD
    ]  # at least one-btag in trijet candidates
    # pick trijet candidate with largest pT and calculate mass of system
    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass

    # ensure we can handle cross-references
    # just touch them so they land in the report
    _ = ak.flatten(events.Electron.matched_jet)

    return ak.flatten(trijet_mass)


def _untypetracable_analysis(events):
    return events.MET.sumEt.to_numpy()


def test_tracing_nanoevents():
    from coffea.nanoevents.trace import (
        trace,
        trace_with_length_one_array,
        trace_with_length_zero_array,
        trace_with_typetracer,
    )

    events = NanoEventsFactory.from_root(
        {"tests/samples/nano_dy.root": "Events"},
        mode="virtual",
        access_log=(access_log := []),
    ).events()

    necessary_columns = trace_with_typetracer(_analysis, events)
    assert sorted(list(necessary_columns)) == [
        "Electron_eta",
        "Electron_jetIdx",
        "Electron_phi",  # actual typetracer "overtouching", see https://github.com/scikit-hep/vector/pull/542
        "Electron_pt",
        "Jet_btagDeepFlavB",
        "Jet_eta",
        "Jet_mass",
        "Jet_phi",
        "Jet_pt",
        "Muon_eta",
        "Muon_phi",  # actual typetracer "overtouching", see https://github.com/scikit-hep/vector/pull/542
        "Muon_pt",
        "nElectron",
        "nJet",
        "nMuon",
    ]

    necessary_columns = trace_with_length_zero_array(_analysis, events)
    assert sorted(list(necessary_columns)) == [
        "Electron_eta",
        "Electron_jetIdx",
        "Electron_pt",
        "Jet_btagDeepFlavB",
        "Jet_eta",
        "Jet_mass",
        "Jet_phi",
        "Jet_pt",
        "Muon_eta",
        "Muon_pt",
        "nElectron",
        "nJet",
        "nMuon",
    ]

    necessary_columns = trace_with_length_one_array(_analysis, events)
    assert sorted(list(necessary_columns)) == [
        "Electron_eta",
        "Electron_jetIdx",
        "Electron_pt",
        "Jet_btagDeepFlavB",
        "Jet_eta",
        "Jet_mass",
        "Jet_phi",
        "Jet_pt",
        "Muon_eta",
        "Muon_pt",
        "nElectron",
        "nJet",
        "nMuon",
    ]

    necessary_columns = trace(_analysis, events)  # this will succeed with typetracer
    assert sorted(list(necessary_columns)) == [
        "Electron_eta",
        "Electron_jetIdx",
        "Electron_phi",  # actual typetracer "overtouchng", see https://github.com/scikit-hep/vector/pull/542
        "Electron_pt",
        "Jet_btagDeepFlavB",
        "Jet_eta",
        "Jet_mass",
        "Jet_phi",
        "Jet_pt",
        "Muon_eta",
        "Muon_phi",  # actual typetracer "overtouching", see https://github.com/scikit-hep/vector/pull/542
        "Muon_pt",
        "nElectron",
        "nJet",
        "nMuon",
    ]

    assert access_log == []
    _ = _analysis(events)
    access_log = [x.branch for x in access_log]
    assert sorted(set(access_log)) == [
        "Electron_eta",
        "Electron_jetIdx",
        "Electron_pt",
        "Jet_btagDeepFlavB",
        "Jet_eta",
        "Jet_mass",
        "Jet_phi",
        "Jet_pt",
        "Muon_eta",
        "Muon_pt",
        "nElectron",
        "nJet",
        "nMuon",
    ]

    events = NanoEventsFactory.from_root(
        {"tests/samples/nano_dy.root": "Events"},
        mode="virtual",
        access_log=(access_log := []),
    ).events()

    with pytest.raises(TypeError, match="from an nplike without known data"):
        necessary_columns = trace_with_typetracer(
            _untypetracable_analysis, events, throw=True
        )
    necessary_columns = trace_with_typetracer(
        _untypetracable_analysis, events, throw=False
    )
    assert sorted(list(necessary_columns)) == []  # nothing found

    necessary_columns = trace_with_length_zero_array(_untypetracable_analysis, events)
    assert sorted(list(necessary_columns)) == ["MET_sumEt"]

    necessary_columns = trace_with_length_one_array(_untypetracable_analysis, events)
    assert sorted(list(necessary_columns)) == ["MET_sumEt"]

    necessary_columns = trace(
        _untypetracable_analysis, events
    )  # this will fall back to length-zero array
    assert sorted(list(necessary_columns)) == ["MET_sumEt"]

    assert access_log == []
    _untypetracable_analysis(events)
    access_log = [x.branch for x in access_log]
    assert sorted(set(access_log)) == ["MET_sumEt"]
