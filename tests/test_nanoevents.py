from pathlib import Path

import awkward as ak
import pytest
import uproot
from distributed import Client

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.schemas import BaseSchema


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
    "parquet",
    "extensionarray.parquet",
]


@pytest.mark.parametrize("suffix", suffixes)
def test_read_nanomc(tests_directory, suffix):
    path = f"{tests_directory}/samples/nano_dy.{suffix}"
    # parquet files were converted from even older nanoaod
    nanoversion = NanoAODSchema
    factory = getattr(
        NanoEventsFactory, f"from_{suffix.removeprefix('extensionarray.')}"
    )(
        {path: "Events"} if suffix == "root" else path,
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
    factory = getattr(
        NanoEventsFactory, f"from_{suffix.removeprefix('extensionarray.')}"
    )(
        {path: "Events"} if suffix == "root" else path,
        schemaclass=nanoversion,
        mode="eager",
    )
    events = factory.events()
    assert len(events) == 40

    # Test storage_options for parquet files
    if suffix == "parquet":
        from unittest.mock import patch

        import fsspec

        path_str = f"{tests_directory}/samples/nano_dy.{suffix}"
        storage_opts = {"some_option": "some_value"}

        original_open = fsspec.open

        def mock_open(file, mode, **kwargs):
            assert kwargs == storage_opts
            return original_open(file, mode)

        with patch("fsspec.open", side_effect=mock_open) as mock_fsspec_open:
            factory = NanoEventsFactory.from_parquet(
                path_str,
                schemaclass=nanoversion,
                storage_options=storage_opts,
                mode="eager",
            )
            events = factory.events()
            assert len(events) == 40
            mock_fsspec_open.assert_called_once()


@pytest.mark.parametrize("suffix", suffixes)
def test_read_nanodata(tests_directory, suffix):
    path = f"{tests_directory}/samples/nano_dimuon.{suffix}"
    # parquet files were converted from even older nanoaod
    nanoversion = NanoAODSchema
    factory = getattr(
        NanoEventsFactory, f"from_{suffix.removeprefix('extensionarray.')}"
    )(
        {path: "Events"} if suffix == "root" else path,
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


@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_access_log(tests_directory, mode):
    """Test that access_log is available on the factory."""
    path = f"{tests_directory}/samples/nano_dy.root:Events"

    # Without passing access_log, it should be None
    factory = NanoEventsFactory.from_root(
        path,
        schemaclass=NanoAODSchema,
        mode=mode,
    )
    assert factory.access_log is None

    # With access_log passed, it should be populated when columns are accessed
    access_log = []
    factory = NanoEventsFactory.from_root(
        path,
        schemaclass=NanoAODSchema,
        mode=mode,
        access_log=access_log,
    )
    events = factory.events()

    assert factory.access_log is access_log
    if mode == "eager":
        assert len(factory.access_log) > 1500

    elif mode == "virtual":
        # In virtual mode, access_log starts empty until columns are accessed
        assert len(factory.access_log) == 0
        # Access a column to trigger lazy loading
        _ = ak.materialize(events.Muon.pt)
        branches = {entry.branch for entry in factory.access_log}
        assert branches == {"nMuon", "Muon_pt"}


@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_file_handle_from_path(tests_directory, mode):
    """Test that file_handle is available when opening from path string."""
    path = f"{tests_directory}/samples/nano_dy.root:Events"

    factory = NanoEventsFactory.from_root(
        path,
        schemaclass=NanoAODSchema,
        mode=mode,
    )

    # file_handle should be ReadOnlyFile when opened from path
    assert factory.file_handle is not None
    assert isinstance(factory.file_handle, uproot.reading.ReadOnlyFile)

    _ = factory.events()

    # file_handle still accessible after events() call
    assert factory.file_handle is not None


@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_file_handle_from_directory(tests_directory, mode):
    """Test that file_handle is available when passing ReadOnlyDirectory."""
    filepath = f"{tests_directory}/samples/nano_dy.root"

    with uproot.open(filepath) as file:
        factory = NanoEventsFactory.from_root(
            file,
            treepath="Events",
            schemaclass=NanoAODSchema,
            mode=mode,
        )

        # file_handle should be ReadOnlyDirectory when passed directly
        assert factory.file_handle is not None
        assert isinstance(factory.file_handle, uproot.ReadOnlyDirectory)

        _ = factory.events()

        # file_handle still accessible after events() call
        assert factory.file_handle is not None


def test_to_flat_columns_base_raises(tests_directory):
    """BaseSchema.to_flat_columns is an explicit stub; subclasses must override."""
    path = f"{tests_directory}/samples/nano_dy.root"
    events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=BaseSchema, mode="eager"
    ).events()
    with pytest.raises(NotImplementedError):
        BaseSchema.to_flat_columns(events)


def test_to_flat_columns_nanoaod_expands_collections(tests_directory):
    """NanoAODSchema.to_flat_columns expands record collections to
    {name}_{subfield} + n{name} and skips cross-reference subfields."""
    path = f"{tests_directory}/samples/nano_dy.root"
    events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()

    out = NanoAODSchema.to_flat_columns(events)

    assert "Muon_pt" in out
    assert "Muon_eta" in out
    assert "nMuon" in out
    # cross-reference subfields (records of records) must not appear
    assert "Muon_matched_jet" not in out
    assert "Muon_matched_gen" not in out
    # values match the source exactly
    assert ak.to_list(out["Muon_pt"]) == ak.to_list(events.Muon.pt)
    assert ak.to_list(out["nMuon"]) == ak.to_list(ak.num(events.Muon))


def test_to_flat_columns_nanoaod_parquet_roundtrip(tests_directory, tmp_path):
    """Round-trip: read ROOT, deconstruct via to_flat_columns, dump to
    parquet, read back with NanoEventsFactory.from_parquet — sum of Muon.pt
    must match the source."""
    path = f"{tests_directory}/samples/nano_dy.root"
    events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    expected_total = float(ak.sum(events.Muon.pt))

    flat = NanoAODSchema.to_flat_columns(events)
    parquet_path = tmp_path / "flat.parquet"
    ak.to_parquet(ak.Array(flat), str(parquet_path))

    reread = NanoEventsFactory.from_parquet(
        str(parquet_path), schemaclass=NanoAODSchema, mode="eager"
    ).events()
    assert float(ak.sum(reread.Muon.pt)) == expected_total


def test_to_flat_columns_nanoaod_root_roundtrip(tests_directory, tmp_path):
    """Round-trip: read ROOT, deconstruct via to_flat_columns, write a new
    ROOT file with uproot, read back with NanoEventsFactory.from_root —
    sum of Muon.pt must match the source."""
    src_path = f"{tests_directory}/samples/nano_dy.root"
    events = NanoEventsFactory.from_root(
        {src_path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    expected_total = float(ak.sum(events.Muon.pt))

    flat = NanoAODSchema.to_flat_columns(events)
    out_path = tmp_path / "flat.root"
    with uproot.recreate(str(out_path)) as fout:
        fout["Events"] = flat

    reread = NanoEventsFactory.from_root(
        {str(out_path): "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    assert float(ak.sum(reread.Muon.pt)) == expected_total
