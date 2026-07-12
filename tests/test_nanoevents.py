import os
from functools import partial
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory


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
def test_missing_eventIds_warning_dask(tests_directory, dask_client):
    pytest.importorskip("dask_awkward")
    path = f"{tests_directory}/samples/missing_luminosityBlock.root:Events"
    NanoAODSchema.error_missing_event_ids = False
    with dask_client.as_current() as _:
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


def test_uproot_write(tmp_path):
    path = os.path.abspath("tests/samples/nano_dy.root")

    # NanoAODSchema round-trip: collection.subfield equality after rewrite.
    orig_events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()

    out_path = str(tmp_path / "nanoaod_write_test.root")
    with uproot.recreate(out_path) as f:
        f.mktree("Events", NanoAODSchema.uproot_writeable(orig_events))

    test_events = NanoEventsFactory.from_root(
        {out_path: "Events"},
        schemaclass=NanoAODSchema,
        mode="eager",
    ).events()

    assert len(orig_events) == len(test_events)
    assert ak.all(orig_events.event == test_events.event)
    assert ak.all(orig_events.Muon.pt == test_events.Muon.pt)
    assert ak.all(orig_events.Muon.eta == test_events.Muon.eta)
    assert ak.all(orig_events.Jet.pt == test_events.Jet.pt)
    assert ak.all(orig_events.MET.pt == test_events.MET.pt)

    # BaseSchema round-trip: flat branch equality after rewrite.
    orig_base = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=BaseSchema, mode="eager"
    ).events()

    base_out_path = str(tmp_path / "baseschema_write_test.root")
    with uproot.recreate(base_out_path) as f:
        f.mktree("Events", BaseSchema.uproot_writeable(orig_base))

    test_base = NanoEventsFactory.from_root(
        {base_out_path: "Events"},
        schemaclass=BaseSchema,
        mode="eager",
    ).events()

    assert len(orig_base) == len(test_base)
    assert ak.all(orig_base.event == test_base.event)
    assert ak.all(orig_base.Muon_pt == test_base.Muon_pt)
    assert ak.all(orig_base.Muon_eta == test_base.Muon_eta)
    assert ak.all(orig_base.Jet_pt == test_base.Jet_pt)
    assert ak.all(orig_base.MET_pt == test_base.MET_pt)


parquet_suffixes = [
    "parquet",
    "extensionarray.parquet",
]


# virtual is the mode the production Runner parquet path uses (executor.py).
@pytest.mark.parametrize("mode", ["eager", "virtual"])
@pytest.mark.parametrize("suffix", parquet_suffixes)
@pytest.mark.parametrize(
    "entry_start,entry_stop", [(5, 15), (1, 40), (0, 10), (37, 40)]
)
def test_parquet_entry_range_matches_full_slice(
    tests_directory, mode, suffix, entry_start, entry_stop
):
    """Regression test for scikit-hep/coffea#1578 (bug 4).

    Reading a parquet file with entry_start > 0 must return exactly the same
    per-event data as reading the whole file and slicing.  The parquet mapping
    used to read jagged offsets from the start of the buffer while extracting
    content with the pyarrow slice offset applied, silently reassigning e.g.
    muon pts to the wrong events (and crashing NanoAOD cross-references).
    """
    path = f"{tests_directory}/samples/nano_dy.{suffix}"
    from_parquet = getattr(
        NanoEventsFactory, f"from_{suffix.removeprefix('extensionarray.')}"
    )

    full = from_parquet(path, schemaclass=NanoAODSchema, mode="eager").events()
    sub = from_parquet(
        path,
        schemaclass=NanoAODSchema,
        mode=mode,
        entry_start=entry_start,
        entry_stop=entry_stop,
    ).events()

    assert len(sub) == entry_stop - entry_start

    # Jagged collections (the buggy path) must match the full-read slice exactly.
    for field in ("Muon", "Jet", "Electron"):
        sub_pt = ak.to_list(getattr(sub, field).pt)
        full_pt = ak.to_list(getattr(full, field).pt[entry_start:entry_stop])
        assert (
            sub_pt == full_pt
        ), f"{field}.pt mismatch for [{entry_start}:{entry_stop}]"

    # A flat (per-event) branch should match as well.
    assert ak.to_list(sub.MET.pt) == ak.to_list(full.MET.pt[entry_start:entry_stop])


@pytest.mark.parametrize(
    "entry_start,entry_stop", [(5, 15), (1, 40), (0, 10), (37, 40)]
)
def test_parquet_int32_list_offsets_entry_range(tmp_path, entry_start, entry_stop):
    """Cover the numpy.int32 offsets branch of the #1578 bug-4 fix.

    The nano_dy sample files all decode to LargeListArray (int64 offsets), so a
    plain pyarrow ``list_`` column (int32 offsets) is needed to exercise the
    other side of the dtype selection in ParquetSourceMapping.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 40
    jagged = [[float(i)] * (i % 3) for i in range(n)]
    table = pa.table(
        {
            "jag": pa.array(jagged, type=pa.list_(pa.float32())),
            "flat": pa.array(np.arange(n, dtype=np.float32)),
        }
    )
    # guard the premise: a non-large list is what yields int32 offsets
    assert pa.types.is_list(table.schema.field("jag").type)
    path = str(tmp_path / "int32list.parquet")
    pq.write_table(table, path)

    full = NanoEventsFactory.from_parquet(
        path, schemaclass=BaseSchema, mode="eager"
    ).events()
    sub = NanoEventsFactory.from_parquet(
        path,
        schemaclass=BaseSchema,
        mode="eager",
        entry_start=entry_start,
        entry_stop=entry_stop,
    ).events()

    assert len(sub) == entry_stop - entry_start
    assert ak.to_list(sub.jag) == ak.to_list(full.jag[entry_start:entry_stop])
    assert ak.to_list(sub.flat) == ak.to_list(full.flat[entry_start:entry_stop])


def test_parquet_column_cache_avoids_repeated_reads(tests_directory, monkeypatch):
    """Regression test for scikit-hep/coffea#1578 (perf bullet).

    A jagged parquet column is materialized through two separate buffer keys
    (offsets and content). Each used to trigger a full re-read of the column
    from the parquet file. The per-source column cache should collapse these
    into a single read while returning identical data.
    """
    import pyarrow.parquet as pq

    from coffea.nanoevents.factory import _key_formatter
    from coffea.nanoevents.mapping.parquet import (
        ParquetSourceMapping,
        TrivialParquetOpener,
    )
    from coffea.nanoevents.util import tuple_to_key

    path = f"{tests_directory}/samples/nano_dy.parquet"

    read_counts = {}
    orig_read = pq.ParquetFile.read

    def counting_read(self, columns=None, use_threads=True, **kwargs):
        for c in columns or []:
            read_counts[c] = read_counts.get(c, 0) + 1
        return orig_read(self, columns=columns, use_threads=use_threads, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read", counting_read)

    parfile = pq.ParquetFile(path)
    n = parfile.metadata.num_rows
    mapping = ParquetSourceMapping(TrivialParquetOpener({"uu": path}), 0, n)
    partition_key = ("uu", "obj", f"0-{n}")
    mapping.preload_column_source(
        partition_key[0],
        partition_key[1],
        TrivialParquetOpener.UprootLikeShim(parfile),
    )

    subform = mapping._extract_base_form(parfile.schema_arrow)
    idx = subform["fields"].index("Muon_pt")
    jagged_form = {
        "class": "RecordArray",
        "fields": ["Muon_pt"],
        "contents": [subform["contents"][idx]],
        "parameters": {"__doc__": "parquetfile"},
        "form_key": "",
    }

    array = ak.from_buffers(
        form=ak.forms.from_dict(jagged_form),
        length=n,
        container=mapping,
        buffer_key=partial(_key_formatter, tuple_to_key(partition_key)),
        highlevel=True,
    )

    # Offsets and content of one jagged column -> a single underlying read.
    assert read_counts["Muon_pt"] == 1

    # And the cache must not corrupt the returned data: compare against the
    # value seen through the normal (un-monkeypatched) reader path.
    reference = NanoEventsFactory.from_parquet(
        path, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    assert ak.to_list(array.Muon_pt) == ak.to_list(reference.Muon.pt)


def test_keys_for_buffer_keys_loadallowmissing():
    """Regression test for scikit-hep/coffea#1578 (bug 5).

    When a saved/union form marks a branch as maybe-missing (an
    ``IndexedOptionArray`` produced by unioning forms across files where some
    files lack the branch), the lazified form key uses the
    ``!loadallowmissing`` token rather than ``!load``. ``keys_for_buffer_keys``
    must still map such buffer keys back to their branch name so that dask mode
    requests the branch from the file. Matching only ``== "!load"`` silently
    drops these branches, so dask mode fabricates them as all-None even in files
    that actually contain the branch.
    """
    from coffea.nanoevents.factory import _map_schema_base
    from coffea.nanoevents.util import quote

    mapper = _map_schema_base()

    # Buffer keys for a maybe-missing branch look like the ones produced by
    # coffea.nanoevents.mapping.uproot._lazify_form for an IndexedOptionArray.
    index_key = f"/index/{quote('flag,!loadallowmissing,!index')}"
    content_key = f"/data/{quote('flag,!loadallowmissing,!content')}"
    # A normal (always-present) branch uses the plain "!load" token.
    load_key = f"/data/{quote('x,!load')}"

    assert mapper.keys_for_buffer_keys({index_key}) == {"flag"}
    assert mapper.keys_for_buffer_keys({content_key}) == {"flag"}
    assert mapper.keys_for_buffer_keys({load_key}) == {"x"}
    assert mapper.keys_for_buffer_keys({index_key, load_key}) == {"flag", "x"}


def _preprocessed_form(files):
    """Preprocess ``files`` ({path: object_path}) and return the saved form."""
    from coffea.dataset_tools import preprocess
    from coffea.dataset_tools.filespec import DatasetSpec

    fileset = {"ds": {"files": files}}
    available, _all = preprocess(fileset, save_form=True, skip_bad_files=False)

    # preprocess preserves the input type: dict in -> dict out.
    ds_entry = available["ds"] if isinstance(available, dict) else available.root["ds"]
    ds = (
        ds_entry
        if isinstance(ds_entry, DatasetSpec)
        else DatasetSpec.model_validate(ds_entry)
    )
    return ds.form


def _write_union_flag_files(tmp_path, n=20):
    """Write two flat ROOT files where only the first has a bool ``flag`` branch."""
    f_has = str(tmp_path / "has_branch.root")
    f_missing = str(tmp_path / "missing_branch.root")

    with uproot.recreate(f_has) as f:
        f.mktree("Events", {"x": np.float32, "flag": np.bool_})
        f["Events"].extend(
            {"x": np.arange(n, dtype=np.float32), "flag": np.ones(n, dtype=bool)}
        )
    with uproot.recreate(f_missing) as f:
        f.mktree("Events", {"x": np.float32})
        f["Events"].extend({"x": np.arange(n, dtype=np.float32) + 100})

    return f_has, f_missing


@pytest.mark.dask_client
def test_union_form_maybe_missing_branch_dask(tmp_path, dask_client):
    """End-to-end regression for scikit-hep/coffea#1578 (bug 5).

    Build two ROOT files where only one contains a boolean ``flag`` branch,
    preprocess them into a union form (which marks ``flag`` as maybe-missing),
    then load the file that DOES contain the branch in dask mode using that
    saved form. Before the fix the branch was never requested, so the column
    came back as all-None instead of its real values.
    """
    pytest.importorskip("dask_awkward")
    import dask_awkward as dak

    f_has, f_missing = _write_union_flag_files(tmp_path)
    union_form = _preprocessed_form({f_has: "Events", f_missing: "Events"})
    # The union form must mark the maybe-missing branch as an IndexedOptionArray.
    assert "IndexedOption" in str(union_form)

    with dask_client.as_current() as _:
        events = NanoEventsFactory.from_root(
            {f_has: "Events"},
            schemaclass=BaseSchema,
            known_base_form=union_form,
            mode="dask",
        ).events()

        # The maybe-missing branch must be requested from the file...
        needed = set().union(*dak.necessary_columns(events["flag"]).values())
        assert "flag" in needed

        # ...and its real values (all True) must be returned, not fabricated None.
        computed = events["flag"].compute()
        assert int(ak.sum(ak.is_none(computed))) == 0
        assert ak.all(computed)


@pytest.mark.dask_client
def test_union_form_genuinely_missing_branch_dask(tmp_path, dask_client):
    """Follow-up regression for scikit-hep/coffea#1578 (bug 5).

    With a saved union form, a branch marked maybe-missing
    (``!loadallowmissing``) that is genuinely absent from a file must be
    backfilled as all-None in dask mode (matching eager/virtual semantics)
    instead of raising ``uproot.KeyInFileError`` when the branch is requested
    from a file that does not contain it.
    """
    pytest.importorskip("dask_awkward")

    n = 20
    f_has, f_missing = _write_union_flag_files(tmp_path, n=n)
    union_form = _preprocessed_form({f_has: "Events", f_missing: "Events"})
    assert "IndexedOption" in str(union_form)

    with dask_client.as_current() as _:
        # (a) The file that LACKS the branch: all-None column, no exception.
        events = NanoEventsFactory.from_root(
            {f_missing: "Events"},
            schemaclass=BaseSchema,
            known_base_form=union_form,
            mode="dask",
        ).events()
        computed = events["flag"].compute()
        assert len(computed) == n
        assert int(ak.sum(ak.is_none(computed))) == n
        # The branches present in the file are unaffected.
        assert ak.all(events["x"].compute() == np.arange(n, dtype=np.float32) + 100)

        # (b) A mixed dataset: real values from the file that has the branch,
        # None backfill from the file that lacks it.
        events = NanoEventsFactory.from_root(
            {f_has: "Events", f_missing: "Events"},
            schemaclass=BaseSchema,
            known_base_form=union_form,
            mode="dask",
        ).events()
        computed = events["flag"].compute()
        assert len(computed) == 2 * n
        assert ak.all(ak.fill_none(computed[:n], False))
        assert int(ak.sum(ak.is_none(computed[:n]))) == 0
        assert int(ak.sum(ak.is_none(computed[n:]))) == n

        # (c) A genuinely-required ("!load") branch that is absent must still
        # error loudly, not be silently backfilled. Preprocessing only the
        # file that has the branch yields a form where "flag" is required.
        required_form = _preprocessed_form({f_has: "Events"})
        assert "IndexedOption" not in str(required_form)
        events = NanoEventsFactory.from_root(
            {f_missing: "Events"},
            schemaclass=BaseSchema,
            known_base_form=required_form,
            mode="dask",
        ).events()
        with pytest.raises(KeyError):
            events["flag"].compute()
