import awkward as ak
import dask
import numpy as np
import pytest

from coffea.nanoevents import NanoEventsFactory


@pytest.mark.parametrize("mode", ["eager", "dask", "virtual"])
@pytest.mark.parametrize("kind", ["UpDownSystematic", "UpDownMultiSystematic"])
def test_single_field_variation(tests_directory, mode, kind):
    def get_array(array):
        return array.compute() if mode == "dask" else array

    access_log = []
    events = NanoEventsFactory.from_root(
        {f"{tests_directory}/samples/nano_dy.root": "Events"},
        mode=mode,
        access_log=access_log,
    ).events()
    expected_muon_pt = ak.flatten(events.Muon.pt)
    expected_jet_pt = ak.flatten(events.Jet.pt)
    expected_muon_phi = ak.flatten(events.Muon.phi)
    expected_jet_phi = ak.flatten(events.Jet.phi)
    expected_met_pt = events.MET.pt
    expected_met_phi = events.MET.phi
    if mode == "dask":
        (
            expected_muon_pt,
            expected_jet_pt,
            expected_muon_phi,
            expected_jet_phi,
            expected_met_pt,
            expected_met_phi,
        ) = dask.compute(
            expected_muon_pt,
            expected_jet_pt,
            expected_muon_phi,
            expected_jet_phi,
            expected_met_pt,
            expected_met_phi,
        )

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

    met.add_systematic("PtScale", "UpDownMultiSystematic", "pt", met_pt_scale)

    renfact_up = events.systematics.RenFactScale.up.weight_RenFactScale
    renfact_down = events.systematics.RenFactScale.down.weight_RenFactScale
    assert ak.all(ak.isclose(get_array(renfact_up), 40 * [1.05]))
    assert ak.all(ak.isclose(get_array(renfact_down), 40 * [0.95]))

    muons_PtScale_up_pt = ak.flatten(muons.systematics.PtScale.up.pt)
    muons_PtScale_down_pt = ak.flatten(muons.systematics.PtScale.down.pt)
    assert ak.all(ak.isclose(get_array(muons_PtScale_up_pt), expected_muon_pt * 1.05))
    assert ak.all(ak.isclose(get_array(muons_PtScale_down_pt), expected_muon_pt * 0.95))

    jets_PtScale_up_pt = ak.flatten(jets.systematics.PtScale.up.pt)
    jets_PtScale_down_pt = ak.flatten(jets.systematics.PtScale.down.pt)
    assert ak.all(ak.isclose(get_array(jets_PtScale_up_pt), expected_jet_pt * 1.10))
    assert ak.all(ak.isclose(get_array(jets_PtScale_down_pt), expected_jet_pt * 0.90))

    met_PtScale_up_pt = met.systematics.PtScale.up.pt
    met_PtScale_down_pt = met.systematics.PtScale.down.pt
    assert ak.all(ak.isclose(get_array(met_PtScale_up_pt), expected_met_pt * 1.03))
    assert ak.all(ak.isclose(get_array(met_PtScale_down_pt), expected_met_pt * 0.97))

    if mode == "virtual":
        access_log = [x.branch for x in access_log]
        assert sorted(access_log) == ["Jet_pt", "MET_pt", "Muon_pt", "nJet", "nMuon"]


@pytest.mark.parametrize("mode", ["eager", "dask", "virtual"])
def test_multi_field_variation(tests_directory, mode):
    def get_array(array):
        return array.compute() if mode == "dask" else array

    access_log = []
    events = NanoEventsFactory.from_root(
        {f"{tests_directory}/samples/nano_dy.root": "Events"},
        mode=mode,
        access_log=access_log,
    ).events()
    expected_muon_pt = ak.flatten(events.Muon.pt)
    expected_jet_pt = ak.flatten(events.Jet.pt)
    expected_muon_phi = ak.flatten(events.Muon.phi)
    expected_jet_phi = ak.flatten(events.Jet.phi)
    expected_met_pt = events.MET.pt
    expected_met_phi = events.MET.phi
    if mode == "dask":
        (
            expected_muon_pt,
            expected_jet_pt,
            expected_muon_phi,
            expected_jet_phi,
            expected_met_pt,
            expected_met_phi,
        ) = dask.compute(
            expected_muon_pt,
            expected_jet_pt,
            expected_muon_phi,
            expected_jet_phi,
            expected_met_pt,
            expected_met_phi,
        )

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
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_up_pt), expected_muon_pt * 1.05)
    )
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_down_pt), expected_muon_pt * 0.95)
    )
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_up_phi), expected_muon_phi * 1.10)
    )
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_down_phi), expected_muon_phi * 0.90)
    )

    jets_PtPhiSystematic_up_pt = ak.flatten(jets.systematics.PtPhiSystematic.up.pt)
    jets_PtPhiSystematic_down_pt = ak.flatten(jets.systematics.PtPhiSystematic.down.pt)
    jets_PtPhiSystematic_up_phi = ak.flatten(jets.systematics.PtPhiSystematic.up.phi)
    jets_PtPhiSystematic_down_phi = ak.flatten(
        jets.systematics.PtPhiSystematic.down.phi
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_up_pt), expected_jet_pt * 1.10)
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_down_pt), expected_jet_pt * 0.90)
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_up_phi), expected_jet_phi * 1.20)
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_down_phi), expected_jet_phi * 0.80)
    )

    met_PtPhiSystematic_up_pt = met.systematics.PtPhiSystematic.up.pt
    met_PtPhiSystematic_down_pt = met.systematics.PtPhiSystematic.down.pt
    met_PtPhiSystematic_up_phi = met.systematics.PtPhiSystematic.up.phi
    met_PtPhiSystematic_down_phi = met.systematics.PtPhiSystematic.down.phi
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_up_pt), expected_met_pt * 1.03)
    )
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_down_pt), expected_met_pt * 0.97)
    )
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_up_phi), expected_met_phi * 1.05)
    )
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_down_phi), expected_met_phi * 0.95)
    )

    if mode == "virtual":
        access_log = [x.branch for x in access_log]
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


@pytest.mark.parametrize("mode", ["eager", "dask", "virtual"])
def test_single_and_multi_field_variation(tests_directory, mode):
    def get_array(array):
        return array.compute() if mode == "dask" else array

    access_log = []
    events = NanoEventsFactory.from_root(
        {f"{tests_directory}/samples/nano_dy.root": "Events"},
        mode=mode,
        access_log=access_log,
    ).events()
    expected_muon_pt = ak.flatten(events.Muon.pt)
    expected_jet_pt = ak.flatten(events.Jet.pt)
    expected_muon_phi = ak.flatten(events.Muon.phi)
    expected_jet_phi = ak.flatten(events.Jet.phi)
    expected_met_pt = events.MET.pt
    expected_met_phi = events.MET.phi
    if mode == "dask":
        (
            expected_muon_pt,
            expected_jet_pt,
            expected_muon_phi,
            expected_jet_phi,
            expected_met_pt,
            expected_met_phi,
        ) = dask.compute(
            expected_muon_pt,
            expected_jet_pt,
            expected_muon_phi,
            expected_jet_phi,
            expected_met_pt,
            expected_met_phi,
        )

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
    assert ak.all(ak.isclose(get_array(muons_PtScale_up_pt), expected_muon_pt * 1.05))
    assert ak.all(ak.isclose(get_array(muons_PtScale_down_pt), expected_muon_pt * 0.95))
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_up_pt), expected_muon_pt * 1.05)
    )
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_down_pt), expected_muon_pt * 0.95)
    )
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_up_phi), expected_muon_phi * 1.10)
    )
    assert ak.all(
        ak.isclose(get_array(muons_PtPhiSystematic_down_phi), expected_muon_phi * 0.90)
    )

    jets_PtScale_up_pt = ak.flatten(jets.systematics.PtScale.up.pt)
    jets_PtScale_down_pt = ak.flatten(jets.systematics.PtScale.down.pt)
    jets_PtPhiSystematic_up_pt = ak.flatten(jets.systematics.PtPhiSystematic.up.pt)
    jets_PtPhiSystematic_down_pt = ak.flatten(jets.systematics.PtPhiSystematic.down.pt)
    jets_PtPhiSystematic_up_phi = ak.flatten(jets.systematics.PtPhiSystematic.up.phi)
    jets_PtPhiSystematic_down_phi = ak.flatten(
        jets.systematics.PtPhiSystematic.down.phi
    )
    assert ak.all(ak.isclose(get_array(jets_PtScale_up_pt), expected_jet_pt * 1.10))
    assert ak.all(ak.isclose(get_array(jets_PtScale_down_pt), expected_jet_pt * 0.90))
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_up_pt), expected_jet_pt * 1.10)
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_down_pt), expected_jet_pt * 0.90)
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_up_phi), expected_jet_phi * 1.20)
    )
    assert ak.all(
        ak.isclose(get_array(jets_PtPhiSystematic_down_phi), expected_jet_phi * 0.80)
    )

    met_PtScale_up_pt = met.systematics.PtScale.up.pt
    met_PtScale_down_pt = met.systematics.PtScale.down.pt
    met_PtPhiSystematic_up_pt = met.systematics.PtPhiSystematic.up.pt
    met_PtPhiSystematic_down_pt = met.systematics.PtPhiSystematic.down.pt
    met_PtPhiSystematic_up_phi = met.systematics.PtPhiSystematic.up.phi
    met_PtPhiSystematic_down_phi = met.systematics.PtPhiSystematic.down.phi
    assert ak.all(ak.isclose(get_array(met_PtScale_up_pt), expected_met_pt * 1.03))
    assert ak.all(ak.isclose(get_array(met_PtScale_down_pt), expected_met_pt * 0.97))
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_up_pt), expected_met_pt * 1.03)
    )
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_down_pt), expected_met_pt * 0.97)
    )
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_up_phi), expected_met_phi * 1.05)
    )
    assert ak.all(
        ak.isclose(get_array(met_PtPhiSystematic_down_phi), expected_met_phi * 0.95)
    )

    if mode == "virtual":
        access_log = [x.branch for x in access_log]
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
