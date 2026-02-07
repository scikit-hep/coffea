import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory


def genroundtrips(genpart):
    # check genpart roundtrip
    assert ak.all(genpart.children.parent.pdgId == genpart.pdgId)
    assert ak.all(
        ak.any(
            genpart.parent.children.pdgId == genpart.pdgId, axis=-1, mask_identity=True
        )
    )
    # distinctParent should be distinct and it should have a relevant child
    assert ak.all(genpart.distinctParent.pdgId != genpart.pdgId)
    assert ak.all(
        ak.any(
            genpart.distinctParent.children.pdgId == genpart.pdgId,
            axis=-1,
            mask_identity=True,
        )
    )

    # distinctChildren should be distinct
    assert ak.all(genpart.distinctChildren.pdgId != genpart.pdgId)
    # their distinctParent's should be the particle itself
    assert ak.all(genpart.distinctChildren.distinctParent.pdgId == genpart.pdgId)

    # parents in decay chains (same pdg id) should never have distinctChildrenDeep
    parents_in_decays = genpart[genpart.parent.pdgId == genpart.pdgId]
    assert ak.all(ak.num(parents_in_decays.distinctChildrenDeep, axis=2) == 0)
    # parents at the top of decay chains that have children should always have distinctChildrenDeep
    real_parents_at_top = genpart[
        (genpart.parent.pdgId != genpart.pdgId) & (ak.num(genpart.children, axis=2) > 0)
    ]
    assert ak.all(ak.num(real_parents_at_top.distinctChildrenDeep, axis=2) > 0)
    # distinctChildrenDeep whose parent pdg id is the same must not have children
    children_in_decays = genpart.distinctChildrenDeep[
        genpart.distinctChildrenDeep.pdgId == genpart.distinctChildrenDeep.parent.pdgId
    ]
    assert ak.all(ak.num(children_in_decays.children, axis=3) == 0)

    # exercise hasFlags
    genpart.hasFlags(["isHardProcess"])
    genpart.hasFlags(["isHardProcess", "isDecayedLeptonHadron"])


def crossref(events):
    # check some cross-ref roundtrips (some may not be true always but they are for the test file)
    assert ak.all(events.Jet.matched_muons.matched_jet.pt == events.Jet.pt)
    assert ak.all(
        events.Electron.matched_photon.matched_electron.r9 == events.Electron.r9
    )
    # exercise LorentzVector.nearest
    assert ak.all(
        events.Muon.matched_jet.delta_r(events.Muon.nearest(events.Jet)) == 0.0
    )


suffixes = ["root", "parquet"]


@pytest.mark.parametrize("suffix", suffixes)
def test_read_nanomc(suffix):
    path = os.path.abspath(f"tests/samples/nano_dy.{suffix}")
    # parquet files were converted from even older nanoaod
    nanoversion = NanoAODSchema.v6 if suffix == "root" else NanoAODSchema.v5
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        path, schemaclass=nanoversion
    )
    events = factory.events()

    # test after views first
    genroundtrips(events.GenPart.mask[events.GenPart.eta > 0])
    genroundtrips(events.mask[ak.any(events.Electron.pt > 50, axis=1)].GenPart)
    genroundtrips(events.GenPart)

    genroundtrips(events.GenPart[events.GenPart.eta > 0])
    genroundtrips(events[ak.any(events.Electron.pt > 50, axis=1)].GenPart)

    # sane gen matching (note for electrons gen match may be photon(22))
    assert ak.all(
        (abs(events.Electron.matched_gen.pdgId) == 11)
        | (events.Electron.matched_gen.pdgId == 22)
    )
    assert ak.all(abs(events.Muon.matched_gen.pdgId) == 13)

    genroundtrips(events.Electron.matched_gen)

    crossref(events[ak.num(events.Jet) > 2])
    crossref(events)

    # test issue 409
    assert ak.to_list(events[[]].Photon.mass) == []

    if suffix == "root":
        assert ak.any(events.Photon.isTight, axis=1).tolist()[:9] == [
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    if suffix == "parquet":
        assert ak.any(events.Photon.isTight, axis=1).tolist()[:9] == [
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]


@pytest.mark.parametrize("suffix", suffixes)
def test_read_from_uri(suffix):
    "Make sure we can properly open the file when a uri is used"
    path = Path(os.path.abspath(f"tests/samples/nano_dy.{suffix}")).as_uri()

    nanoversion = NanoAODSchema.v6 if suffix == "root" else NanoAODSchema.v5
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        path, schemaclass=nanoversion
    )
    events = factory.events()

    assert len(events) == 40 if suffix == "root" else 10


@pytest.mark.parametrize("suffix", suffixes)
def test_read_nanodata(suffix):
    path = os.path.abspath(f"tests/samples/nano_dimuon.{suffix}")
    # parquet files were converted from even older nanoaod
    nanoversion = NanoAODSchema.v6 if suffix == "root" else NanoAODSchema.v5
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        path, schemaclass=nanoversion
    )
    events = factory.events()

    crossref(events)
    crossref(events[ak.num(events.Jet) > 2])


@pytest.mark.parametrize("kind", ["UpDownSystematic", "UpDownMultiSystematic"])
def test_single_field_variation(kind):
    access_log = []
    events = NanoEventsFactory.from_root(
        os.path.abspath("tests/samples/nano_dy.root"),
        access_log=access_log,
    ).events()
    expected_muon_pt = ak.flatten(events.Muon.pt)
    expected_jet_pt = ak.flatten(events.Jet.pt)
    expected_muon_phi = ak.flatten(events.Muon.phi)
    expected_jet_phi = ak.flatten(events.Jet.phi)
    expected_met_pt = events.MET.pt
    expected_met_phi = events.MET.phi

    def some_event_weight(ones):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

    events.add_systematic("RenFactScale", kind, "weight", some_event_weight)
    events.add_systematic("XSectionUncertainty", kind, "weight", some_event_weight)

    muons = events.Muon
    jets = events.Jet
    met = events.MET

    def muon_pt_scale(pt):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * pt[:, None]

    def muon_pt_resolution(pt):
        return np.random.normal(pt[:, None], np.array([0.02, 0.01], dtype=np.float32))

    def muon_eff_weight(ones):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

    muons.add_systematic("PtScale", kind, "pt", muon_pt_scale)
    muons.add_systematic("PtResolution", kind, "pt", muon_pt_resolution)
    muons.add_systematic("EfficiencySF", kind, "weight", muon_eff_weight)

    def jet_pt_scale(pt):
        return (1.0 + np.array([0.10, -0.10], dtype=np.float32)) * pt[:, None]

    def jet_pt_resolution(pt):
        return np.random.normal(pt[:, None], np.array([0.20, 0.10], dtype=np.float32))

    jets.add_systematic("PtScale", kind, "pt", jet_pt_scale)
    jets.add_systematic("PtResolution", kind, "pt", jet_pt_resolution)

    def met_pt_scale(pt):
        return (1.0 + np.array([0.03, -0.03], dtype=np.float32)) * pt[:, None]

    met.add_systematic("PtScale", kind, "pt", met_pt_scale)

    renfact_up = events.systematics.RenFactScale.up.weight_RenFactScale
    renfact_down = events.systematics.RenFactScale.down.weight_RenFactScale
    assert ak.all(ak.isclose(renfact_up, 40 * [1.05]))
    assert ak.all(ak.isclose(renfact_down, 40 * [0.95]))

    muons_PtScale_up_pt = ak.flatten(muons.systematics.PtScale.up.pt)
    muons_PtScale_down_pt = ak.flatten(muons.systematics.PtScale.down.pt)
    assert ak.all(ak.isclose(muons_PtScale_up_pt, expected_muon_pt * 1.05))
    assert ak.all(ak.isclose(muons_PtScale_down_pt, expected_muon_pt * 0.95))

    jets_PtScale_up_pt = ak.flatten(jets.systematics.PtScale.up.pt)
    jets_PtScale_down_pt = ak.flatten(jets.systematics.PtScale.down.pt)
    assert ak.all(ak.isclose(jets_PtScale_up_pt, expected_jet_pt * 1.10))
    assert ak.all(ak.isclose(jets_PtScale_down_pt, expected_jet_pt * 0.90))

    met_PtScale_up_pt = met.systematics.PtScale.up.pt
    met_PtScale_down_pt = met.systematics.PtScale.down.pt
    assert ak.all(ak.isclose(met_PtScale_up_pt, expected_met_pt * 1.03))
    assert ak.all(ak.isclose(met_PtScale_down_pt, expected_met_pt * 0.97))

    assert sorted(access_log) == ["Jet_pt", "MET_pt", "Muon_pt", "nJet", "nMuon"]


def test_multi_field_variation():
    access_log = []
    events = NanoEventsFactory.from_root(
        os.path.abspath("tests/samples/nano_dy.root"),
        access_log=access_log,
    ).events()
    expected_muon_pt = ak.flatten(events.Muon.pt)
    expected_jet_pt = ak.flatten(events.Jet.pt)
    expected_muon_phi = ak.flatten(events.Muon.phi)
    expected_jet_phi = ak.flatten(events.Jet.phi)
    expected_met_pt = events.MET.pt
    expected_met_phi = events.MET.phi

    muons = events.Muon
    jets = events.Jet
    met = events.MET

    def muon_pt_phi_systematic(ptphi):
        pt_var = (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ptphi.pt[:, None]
        phi_var = (1.0 + np.array([0.1, -0.1], dtype=np.float32)) * ptphi.phi[:, None]
        return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)

    muons.add_systematic(
        "PtPhiSystematic",
        "UpDownMultiSystematic",
        ("pt", "phi"),
        muon_pt_phi_systematic,
    )

    def jet_pt_phi_systematic(ptphi):
        pt_var = (1.0 + np.array([0.10, -0.10], dtype=np.float32)) * ptphi.pt[:, None]
        phi_var = (1.0 + np.array([0.2, -0.2], dtype=np.float32)) * ptphi.phi[:, None]
        return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)

    jets.add_systematic(
        "PtPhiSystematic", "UpDownMultiSystematic", ("pt", "phi"), jet_pt_phi_systematic
    )

    def met_pt_phi_systematic(ptphi):
        pt_var = (1.0 + np.array([0.03, -0.03], dtype=np.float32)) * ptphi.pt[:, None]
        phi_var = (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ptphi.phi[:, None]
        return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)

    met.add_systematic(
        "PtPhiSystematic", "UpDownMultiSystematic", ("pt", "phi"), met_pt_phi_systematic
    )

    muons_PtPhiSystematic_up_pt = ak.flatten(muons.systematics.PtPhiSystematic.up.pt)
    muons_PtPhiSystematic_down_pt = ak.flatten(
        muons.systematics.PtPhiSystematic.down.pt
    )
    muons_PtPhiSystematic_up_phi = ak.flatten(muons.systematics.PtPhiSystematic.up.phi)
    muons_PtPhiSystematic_down_phi = ak.flatten(
        muons.systematics.PtPhiSystematic.down.phi
    )
    assert ak.all(ak.isclose(muons_PtPhiSystematic_up_pt, expected_muon_pt * 1.05))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_down_pt, expected_muon_pt * 0.95))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_up_phi, expected_muon_phi * 1.10))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_down_phi, expected_muon_phi * 0.90))

    jets_PtPhiSystematic_up_pt = ak.flatten(jets.systematics.PtPhiSystematic.up.pt)
    jets_PtPhiSystematic_down_pt = ak.flatten(jets.systematics.PtPhiSystematic.down.pt)
    jets_PtPhiSystematic_up_phi = ak.flatten(jets.systematics.PtPhiSystematic.up.phi)
    jets_PtPhiSystematic_down_phi = ak.flatten(
        jets.systematics.PtPhiSystematic.down.phi
    )
    assert ak.all(ak.isclose(jets_PtPhiSystematic_up_pt, expected_jet_pt * 1.10))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_down_pt, expected_jet_pt * 0.90))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_up_phi, expected_jet_phi * 1.20))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_down_phi, expected_jet_phi * 0.80))

    met_PtPhiSystematic_up_pt = met.systematics.PtPhiSystematic.up.pt
    met_PtPhiSystematic_down_pt = met.systematics.PtPhiSystematic.down.pt
    met_PtPhiSystematic_up_phi = met.systematics.PtPhiSystematic.up.phi
    met_PtPhiSystematic_down_phi = met.systematics.PtPhiSystematic.down.phi
    assert ak.all(ak.isclose(met_PtPhiSystematic_up_pt, expected_met_pt * 1.03))
    assert ak.all(ak.isclose(met_PtPhiSystematic_down_pt, expected_met_pt * 0.97))
    assert ak.all(ak.isclose(met_PtPhiSystematic_up_phi, expected_met_phi * 1.05))
    assert ak.all(ak.isclose(met_PtPhiSystematic_down_phi, expected_met_phi * 0.95))

    assert sorted(access_log) == [
        "Jet_phi",
        "Jet_pt",
        "MET_phi",
        "MET_pt",
        "Muon_phi",
        "Muon_pt",
        "nJet",
        "nMuon",
    ]


def test_single_and_multi_field_variation():
    access_log = []
    events = NanoEventsFactory.from_root(
        os.path.abspath("tests/samples/nano_dy.root"),
        access_log=access_log,
    ).events()
    expected_muon_pt = ak.flatten(events.Muon.pt)
    expected_jet_pt = ak.flatten(events.Jet.pt)
    expected_muon_phi = ak.flatten(events.Muon.phi)
    expected_jet_phi = ak.flatten(events.Jet.phi)
    expected_met_pt = events.MET.pt
    expected_met_phi = events.MET.phi

    muons = events.Muon
    jets = events.Jet
    met = events.MET

    def muon_pt_scale(pt):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * pt[:, None]

    def muon_pt_resolution(pt):
        return np.random.normal(pt[:, None], np.array([0.02, 0.01], dtype=np.float32))

    def muon_eff_weight(ones):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

    def muon_pt_phi_systematic(ptphi):
        pt_var = (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ptphi.pt[:, None]
        phi_var = (1.0 + np.array([0.1, -0.1], dtype=np.float32)) * ptphi.phi[:, None]
        return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)

    muons.add_systematic("PtScale", "UpDownMultiSystematic", "pt", muon_pt_scale)
    muons.add_systematic(
        "PtResolution", "UpDownMultiSystematic", "pt", muon_pt_resolution
    )
    muons.add_systematic(
        "PtPhiSystematic",
        "UpDownMultiSystematic",
        ("pt", "phi"),
        muon_pt_phi_systematic,
    )
    muons.add_systematic(
        "EfficiencySF", "UpDownMultiSystematic", "weight", muon_eff_weight
    )

    def jet_pt_scale(pt):
        return (1.0 + np.array([0.10, -0.10], dtype=np.float32)) * pt[:, None]

    def jet_pt_resolution(pt):
        return np.random.normal(pt[:, None], np.array([0.20, 0.10], dtype=np.float32))

    def jet_pt_phi_systematic(ptphi):
        pt_var = (1.0 + np.array([0.10, -0.10], dtype=np.float32)) * ptphi.pt[:, None]
        phi_var = (1.0 + np.array([0.2, -0.2], dtype=np.float32)) * ptphi.phi[:, None]
        return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)

    jets.add_systematic("PtScale", "UpDownMultiSystematic", "pt", jet_pt_scale)
    jets.add_systematic(
        "PtResolution", "UpDownMultiSystematic", "pt", jet_pt_resolution
    )
    jets.add_systematic(
        "PtPhiSystematic", "UpDownMultiSystematic", ("pt", "phi"), jet_pt_phi_systematic
    )

    def met_pt_scale(pt):
        return (1.0 + np.array([0.03, -0.03], dtype=np.float32)) * pt[:, None]

    def met_pt_phi_systematic(ptphi):
        pt_var = (1.0 + np.array([0.03, -0.03], dtype=np.float32)) * ptphi.pt[:, None]
        phi_var = (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ptphi.phi[:, None]
        return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)

    met.add_systematic("PtScale", "UpDownMultiSystematic", "pt", met_pt_scale)
    met.add_systematic(
        "PtPhiSystematic", "UpDownMultiSystematic", ("pt", "phi"), met_pt_phi_systematic
    )

    muons_PtScale_up_pt = ak.flatten(muons.systematics.PtScale.up.pt)
    muons_PtScale_down_pt = ak.flatten(muons.systematics.PtScale.down.pt)
    muons_PtPhiSystematic_up_pt = ak.flatten(muons.systematics.PtPhiSystematic.up.pt)
    muons_PtPhiSystematic_down_pt = ak.flatten(
        muons.systematics.PtPhiSystematic.down.pt
    )
    muons_PtPhiSystematic_up_phi = ak.flatten(muons.systematics.PtPhiSystematic.up.phi)
    muons_PtPhiSystematic_down_phi = ak.flatten(
        muons.systematics.PtPhiSystematic.down.phi
    )
    assert ak.all(ak.isclose(muons_PtScale_up_pt, expected_muon_pt * 1.05))
    assert ak.all(ak.isclose(muons_PtScale_down_pt, expected_muon_pt * 0.95))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_up_pt, expected_muon_pt * 1.05))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_down_pt, expected_muon_pt * 0.95))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_up_phi, expected_muon_phi * 1.10))
    assert ak.all(ak.isclose(muons_PtPhiSystematic_down_phi, expected_muon_phi * 0.90))

    jets_PtScale_up_pt = ak.flatten(jets.systematics.PtScale.up.pt)
    jets_PtScale_down_pt = ak.flatten(jets.systematics.PtScale.down.pt)
    jets_PtPhiSystematic_up_pt = ak.flatten(jets.systematics.PtPhiSystematic.up.pt)
    jets_PtPhiSystematic_down_pt = ak.flatten(jets.systematics.PtPhiSystematic.down.pt)
    jets_PtPhiSystematic_up_phi = ak.flatten(jets.systematics.PtPhiSystematic.up.phi)
    jets_PtPhiSystematic_down_phi = ak.flatten(
        jets.systematics.PtPhiSystematic.down.phi
    )
    assert ak.all(ak.isclose(jets_PtScale_up_pt, expected_jet_pt * 1.10))
    assert ak.all(ak.isclose(jets_PtScale_down_pt, expected_jet_pt * 0.90))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_up_pt, expected_jet_pt * 1.10))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_down_pt, expected_jet_pt * 0.90))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_up_phi, expected_jet_phi * 1.20))
    assert ak.all(ak.isclose(jets_PtPhiSystematic_down_phi, expected_jet_phi * 0.80))

    met_PtScale_up_pt = met.systematics.PtScale.up.pt
    met_PtScale_down_pt = met.systematics.PtScale.down.pt
    met_PtPhiSystematic_up_pt = met.systematics.PtPhiSystematic.up.pt
    met_PtPhiSystematic_down_pt = met.systematics.PtPhiSystematic.down.pt
    met_PtPhiSystematic_up_phi = met.systematics.PtPhiSystematic.up.phi
    met_PtPhiSystematic_down_phi = met.systematics.PtPhiSystematic.down.phi
    assert ak.all(ak.isclose(met_PtScale_up_pt, expected_met_pt * 1.03))
    assert ak.all(ak.isclose(met_PtScale_down_pt, expected_met_pt * 0.97))
    assert ak.all(ak.isclose(met_PtPhiSystematic_up_pt, expected_met_pt * 1.03))
    assert ak.all(ak.isclose(met_PtPhiSystematic_down_pt, expected_met_pt * 0.97))
    assert ak.all(ak.isclose(met_PtPhiSystematic_up_phi, expected_met_phi * 1.05))
    assert ak.all(ak.isclose(met_PtPhiSystematic_down_phi, expected_met_phi * 0.95))

    assert sorted(access_log) == [
        "Jet_phi",
        "Jet_pt",
        "MET_phi",
        "MET_pt",
        "Muon_phi",
        "Muon_pt",
        "nJet",
        "nMuon",
    ]
