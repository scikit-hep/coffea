from pathlib import Path

import awkward as ak
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
        delayed=False,
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
def test_read_from_uri(tests_directory, suffix):
    """Make sure we can properly open the file when a uri is used"""
    path = Path(f"{tests_directory}/samples/nano_dy.{suffix}").as_uri()

    nanoversion = NanoAODSchema
    factory = getattr(NanoEventsFactory, f"from_{suffix}")(
        {path: "Events"},
        schemaclass=nanoversion,
        delayed=False,
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
        delayed=False,
    )
    events = factory.events()

    crossref(events)
    crossref(events[ak.num(events.Jet) > 2])


def test_missing_eventIds_error(tests_directory):
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    with pytest.raises(RuntimeError):
        factory = NanoEventsFactory.from_root(
            path, schemaclass=NanoAODSchema, delayed=False
        )
        factory.events()


def test_missing_eventIds_warning(tests_directory):
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    with pytest.warns(
        RuntimeWarning, match=r"Missing event_ids \: \[\'luminosityBlock\'\]"
    ):
        NanoAODSchema.error_missing_event_ids = False
        factory = NanoEventsFactory.from_root(
            path, schemaclass=NanoAODSchema, delayed=False
        )
        factory.events()


def test_missing_eventIds_warning_dask(tests_directory):
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    NanoAODSchema.error_missing_event_ids = False
    with Client() as _:
        events = NanoEventsFactory.from_root(
            path,
            schemaclass=NanoAODSchema,
            delayed=True,
        ).events()
        events.Muon.pt.compute()
