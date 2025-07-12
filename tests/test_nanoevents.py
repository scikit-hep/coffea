from pathlib import Path

import awkward as ak
import numpy as np
import pytest
from distributed import Client

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


suffixes = [
    "root",
    #    "parquet",
]


@pytest.mark.parametrize("suffix", suffixes)
def test_read_nanomc(tests_directory, suffix):
    path = f"{tests_directory}/samples/nano_dy.{suffix}"
    # parquet files were converted from even older nanoaod
    nanoversion = NanoAODSchema
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        {path: "Events"},
        schemaclass=nanoversion,
        mode="eager",
    )
    events = factory.events()

    # test after views first
    genroundtrips(ak.mask(events.GenPart, events.GenPart.eta > 0))
    genroundtrips(ak.mask(events, ak.any(events.Electron.pt > 50, axis=1)).GenPart)
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
def test_read_from_uri(tests_directory, suffix):
    """Make sure we can properly open the file when a uri is used"""
    path = Path(f"{tests_directory}/samples/nano_dy.{suffix}").as_uri()

    nanoversion = NanoAODSchema
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        {path: "Events"},
        schemaclass=nanoversion,
        mode="eager",
    )
    events = factory.events()

    assert len(events) == 40 if suffix == "root" else 10


@pytest.mark.parametrize("suffix", suffixes)
def test_read_nanodata(tests_directory, suffix):
    path = f"{tests_directory}/samples/nano_dimuon.{suffix}"
    # parquet files were converted from even older nanoaod
    nanoversion = NanoAODSchema
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        {path: "Events"},
        schemaclass=nanoversion,
        mode="eager",
    )
    events = factory.events()

    crossref(events)
    crossref(events[ak.num(events.Jet) > 2])


def test_missing_eventIds_error(tests_directory):
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    with pytest.raises(RuntimeError):
        factory = NanoEventsFactory.from_root(
            path, schemaclass=NanoAODSchema, mode="eager"
        )
        factory.events()


def test_missing_eventIds_warning(tests_directory):
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    with pytest.warns(
        RuntimeWarning, match=r"Missing event_ids \: \[\'luminosityBlock\'\]"
    ):
        NanoAODSchema.error_missing_event_ids = False
        factory = NanoEventsFactory.from_root(
            path, schemaclass=NanoAODSchema, mode="eager"
        )
        factory.events()


@pytest.mark.dask_client
def test_missing_eventIds_warning_dask(tests_directory):
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    NanoAODSchema.error_missing_event_ids = False
    with Client() as _:
        events = NanoEventsFactory.from_root(
            path,
            schemaclass=NanoAODSchema,
            mode="dask",
        ).events()
        events.Muon.pt.compute()


@pytest.mark.parametrize("mode", ["eager", "dask", "virtual"])
def test_systematics_harness(tests_directory, mode):
    def get_array(array):
        return array.compute() if mode == "dask" else array

    events = NanoEventsFactory.from_root(
        {f"{tests_directory}/samples/nano_dy.root": "Events"}, mode=mode
    ).events()

    expected_muon_pt = ak.Array(
        [
            76.7533187866211,
            20.13140869140625,
            31.03870391845703,
            50.64134216308594,
            14.330102920532227,
            16.72498321533203,
            13.908062934875488,
            46.49888610839844,
            40.66781234741211,
            51.435760498046875,
            39.59278869628906,
            38.89543151855469,
            33.71282196044922,
            17.082792282104492,
            14.526994705200195,
            4.360542297363281,
            10.117709159851074,
            17.949918746948242,
        ]
    )

    expected_jet_pt = ak.Array(
        [
            80.75,
            45.59375,
            29.640625,
            17.5625,
            15.234375,
            105.625,
            38.90625,
            25.953125,
            19.890625,
            18.171875,
            17.46875,
            16.46875,
            15.2421875,
            97.3125,
            64.8125,
            55.46875,
            30.34375,
            22.03125,
            61.03125,
            35.125,
            18.03125,
            90.625,
            84.375,
            29.390625,
            18.0625,
            15.265625,
            38.90625,
            36.6875,
            31.125,
            27.984375,
            22.625,
            16.8125,
            16.625,
            16.609375,
            81.875,
            53.375,
            38.03125,
            25.296875,
            39.25,
            26.140625,
            24.265625,
            18.375,
            20.375,
            91.0,
            45.25,
            37.625,
            31.828125,
            28.46875,
            25.703125,
            18.59375,
            15.40625,
            15.0546875,
            30.015625,
            29.171875,
            67.625,
            41.9375,
            22.546875,
            21.09375,
            20.65625,
            19.515625,
            16.984375,
            15.1484375,
            20.78125,
            15.546875,
            51.5625,
            22.15625,
            20.171875,
            18.328125,
            44.625,
            21.1875,
            21.078125,
            18.828125,
            17.53125,
            57.71875,
            49.59375,
            42.1875,
            23.9375,
            63.09375,
            31.109375,
            20.828125,
            20.375,
            18.25,
            43.90625,
            18.015625,
            15.7421875,
            15.3359375,
            15.2890625,
            18.953125,
            18.84375,
            18.09375,
            17.96875,
            16.578125,
            15.7578125,
            15.0078125,
            28.84375,
            20.8125,
            17.15625,
            15.640625,
            42.1875,
            40.21875,
            15.984375,
            39.53125,
            35.3125,
            27.578125,
            18.171875,
            16.9375,
            159.25,
            119.125,
            65.1875,
            38.03125,
            36.8125,
            19.140625,
            18.375,
            17.203125,
            16.21875,
            65.9375,
            47.84375,
            15.5234375,
            58.46875,
            44.65625,
            24.140625,
            19.09375,
            15.21875,
            29.9375,
            21.5625,
            18.953125,
            17.90625,
            50.21875,
            45.1875,
            27.171875,
            25.421875,
            17.09375,
            17.046875,
            15.171875,
            41.46875,
            33.34375,
            20.203125,
            16.375,
            15.09375,
            34.28125,
            33.28125,
            25.9375,
            24.078125,
            23.421875,
            20.125,
            19.28125,
            18.296875,
            34.0,
            24.828125,
            20.078125,
            20.0,
            18.71875,
            22.203125,
            20.203125,
            18.796875,
            15.21875,
            30.84375,
            26.078125,
            20.015625,
            19.0625,
            18.75,
            16.0625,
            47.9375,
            42.0625,
            29.4375,
            23.234375,
            19.765625,
            17.65625,
            16.84375,
            15.8203125,
            15.65625,
            40.5625,
            27.234375,
            25.828125,
            21.0625,
            17.84375,
            29.265625,
            21.203125,
            15.8359375,
            22.109375,
            56.875,
            53.625,
            24.703125,
            21.59375,
            20.203125,
            15.1875,
            18.828125,
            18.265625,
        ]
    )

    def some_event_weight(ones):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

    events.add_systematic(
        "RenFactScale", "UpDownSystematic", "weight", some_event_weight
    )
    events.add_systematic(
        "XSectionUncertainty", "UpDownSystematic", "weight", some_event_weight
    )

    muons = events.Muon
    jets = events.Jet

    def muon_pt_scale(pt):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * pt[:, None]

    def muon_pt_resolution(pt):
        return np.random.normal(pt[:, None], np.array([0.02, 0.01], dtype=np.float32))

    def muon_eff_weight(ones):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

    muons.add_systematic("PtScale", "UpDownSystematic", "pt", muon_pt_scale)
    muons.add_systematic("PtResolution", "UpDownSystematic", "pt", muon_pt_resolution)
    muons.add_systematic("EfficiencySF", "UpDownSystematic", "weight", muon_eff_weight)

    def jet_pt_scale(pt):
        return (1.0 + np.array([0.10, -0.10], dtype=np.float32)) * pt[:, None]

    def jet_pt_resolution(pt):
        return np.random.normal(pt[:, None], np.array([0.20, 0.10], dtype=np.float32))

    jets.add_systematic("PtScale", "UpDownSystematic", "pt", jet_pt_scale)
    jets.add_systematic("PtResolution", "UpDownSystematic", "pt", jet_pt_resolution)

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
