import awkward as ak
import pytest
import uproot

from coffea.nanoevents import (
    BaseSchema,
    NanoAODSchema,
    NanoEventsFactory,
    PFNanoAODSchema,
    TreeMakerSchema,
)


@pytest.mark.parametrize("mode", ["eager", "virtual"])
@pytest.mark.parametrize(
    "file", ["nano_dy", "nano_dimuon", "nano_tree", "pfnano", "treemaker"]
)
def test_base_schema(tests_directory, file, mode):
    key = "PreSelection" if file == "treemaker" else "Events"
    file = f"{tests_directory}/samples/{file}"
    ttree = NanoEventsFactory.from_root(
        {f"{file}.root": key}, schemaclass=BaseSchema, mode=mode
    ).events()
    rntuple = NanoEventsFactory.from_root(
        {f"{file}_rntuple.root": key}, schemaclass=BaseSchema, mode=mode
    ).events()
    if mode == "virtual":
        assert not ttree.layout.is_any_materialized
        assert not rntuple.layout.is_any_materialized
    elif mode == "eager":
        assert ttree.layout.is_all_materialized
        assert rntuple.layout.is_all_materialized
    if key == "PreSelection":
        for field in rntuple.fields:
            subfields = rntuple[field].fields
            if subfields != []:
                assert len(subfields) == 1
                subfield = subfields[0]
                subsubfields = rntuple[field][subfield].fields
                for subsubfield in subsubfields:
                    left = rntuple[field][subfield][subsubfield]
                    right = ttree[field][f"{field}.{subfield}.{subsubfield}"]
                    assert ak.array_equal(
                        left,
                        right,
                        dtype_exact=False,
                        check_parameters=False,
                        equal_nan=True,
                    )
            else:
                left = rntuple[field]
                right = ttree[field]
                assert ak.array_equal(
                    left,
                    right,
                    dtype_exact=False,
                    check_parameters=False,
                    equal_nan=True,
                )
    else:
        assert ak.array_equal(
            rntuple, ttree, dtype_exact=False, check_parameters=False, equal_nan=True
        )


@pytest.mark.parametrize("mode", ["eager", "virtual", "dask"])
@pytest.mark.parametrize("file", ["nano_dy", "nano_dimuon", "nano_tree"])
def test_nanoaod_schema(tests_directory, file, mode):
    if mode == "dask":
        pytest.importorskip("dask_awkward")
    file = f"{tests_directory}/samples/{file}"
    ttree = NanoEventsFactory.from_root(
        {f"{file}.root": "Events"}, schemaclass=NanoAODSchema, mode=mode
    ).events()
    rntuple = NanoEventsFactory.from_root(
        {f"{file}_rntuple.root": "Events"}, schemaclass=NanoAODSchema, mode=mode
    ).events()
    if mode == "virtual":
        assert not ttree.layout.is_any_materialized
        assert not rntuple.layout.is_any_materialized
    elif mode == "eager":
        assert ttree.layout.is_all_materialized
        assert rntuple.layout.is_all_materialized
    elif mode == "dask":
        ttree, rntuple = ttree.compute(), rntuple.compute()
    assert ak.array_equal(
        rntuple, ttree, dtype_exact=False, check_parameters=False, equal_nan=True
    )


@pytest.mark.parametrize("mode", ["eager", "virtual", "dask"])
def test_pfnano_schema(tests_directory, mode):
    if mode == "dask":
        pytest.importorskip("dask_awkward")
    file = f"{tests_directory}/samples/pfnano"
    ttree = NanoEventsFactory.from_root(
        {f"{file}.root": "Events"}, schemaclass=PFNanoAODSchema, mode=mode
    ).events()
    rntuple = NanoEventsFactory.from_root(
        {f"{file}_rntuple.root": "Events"}, schemaclass=PFNanoAODSchema, mode=mode
    ).events()
    if mode == "virtual":
        assert not ttree.layout.is_any_materialized
        assert not rntuple.layout.is_any_materialized
    elif mode == "eager":
        assert ttree.layout.is_all_materialized
        assert rntuple.layout.is_all_materialized
    elif mode == "dask":
        ttree, rntuple = ttree.compute(), rntuple.compute()
    assert ak.array_equal(
        rntuple, ttree, dtype_exact=False, check_parameters=False, equal_nan=True
    )


@pytest.mark.xfail(
    reason="RNTuple version of the treemaker sample has different field structure"
)
@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_treemaker_schema(tests_directory, mode):
    file = f"{tests_directory}/samples/treemaker"
    ttree = NanoEventsFactory.from_root(
        {f"{file}.root": "PreSelection"}, schemaclass=TreeMakerSchema, mode=mode
    ).events()
    rntuple = NanoEventsFactory.from_root(
        {f"{file}_rntuple.root": "PreSelection"}, schemaclass=TreeMakerSchema, mode=mode
    ).events()
    if mode == "virtual":
        assert not ttree.layout.is_any_materialized
        assert not rntuple.layout.is_any_materialized
    elif mode == "eager":
        assert ttree.layout.is_all_materialized
        assert rntuple.layout.is_all_materialized
    assert ak.array_equal(
        rntuple, ttree, dtype_exact=False, check_parameters=False, equal_nan=True
    )


@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_nested_fields_load_leaves(tests_directory, tmp_path, mode):
    ttree = uproot.open(f"{tests_directory}/samples/nano_dy.root:Events")
    fields = {
        "run": ttree["run"].array(),
        "Muon": ak.zip(
            {"pt": ttree["Muon_pt"].array(), "eta": ttree["Muon_eta"].array()}
        ),
        "HLT": ak.zip(
            {
                "IsoMu24": ttree["HLT_IsoMu24"].array(),
                "Mu50": ttree["HLT_Mu50"].array(),
            }
        ),
    }
    filename = tmp_path / "nested_rntuple.root"
    with uproot.recreate(filename) as f:
        f.mkrntuple("Events", fields)

    access_log = []
    events = NanoEventsFactory.from_root(
        {filename: "Events"},
        schemaclass=BaseSchema,
        mode=mode,
        access_log=access_log,
    ).events()
    if mode == "eager":
        # each leaf is read on its own, never a whole record at once
        assert {x.branch for x in access_log} == {
            "run",
            "Muon.pt",
            "Muon.eta",
            "HLT.IsoMu24",
            "HLT.Mu50",
        }
    else:
        assert access_log == []
        ak.materialize(events.HLT.IsoMu24)
        assert {x.branch for x in access_log} == {"HLT.IsoMu24"}
    for field, array in fields.items():
        assert ak.array_equal(events[field], array, check_parameters=False)


@pytest.mark.parametrize(
    "preload, from_file",
    [
        (["pt"], {"Muon.pt", "Muon.eta"}),
        (["Muon.pt"], {"pt", "Muon.eta"}),
        (["Muon"], {"pt"}),
    ],
)
@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_preload_selects_and_finds_fields_by_path(
    tmp_path, monkeypatch, mode, preload, from_file
):
    fields = {
        "pt": ak.Array([[10.0, 20.0], [30.0], [], [40.0]]),
        "Muon": ak.zip(
            {
                "pt": ak.Array([[1.0, 2.0], [3.0], [], [4.0]]),
                "eta": ak.Array([[5.0, 6.0], [7.0], [], [8.0]]),
            }
        ),
    }
    filename = tmp_path / "preload_rntuple.root"
    with uproot.recreate(filename) as f:
        f.mkrntuple("Events", fields)

    read = []
    real_array = uproot.models.RNTuple.RField.array

    def logged_array(field, *args, **kwargs):
        read.append(field.path)
        return real_array(field, *args, **kwargs)

    monkeypatch.setattr(uproot.models.RNTuple.RField, "array", logged_array)
    events = NanoEventsFactory.from_root(
        {filename: "Events"}, schemaclass=BaseSchema, mode=mode, preload=preload
    ).events()
    for field, array in fields.items():
        assert ak.array_equal(
            ak.materialize(events[field]), array, check_parameters=False
        )
    assert set(read) == from_file


@pytest.mark.parametrize("mode", ["eager", "virtual", "dask"])
def test_nested_rntuple_nanoaod_behaviors(tests_directory, tmp_path, mode):
    if mode == "dask":
        pytest.importorskip("dask_awkward")
    orig_events = NanoEventsFactory.from_root(
        {f"{tests_directory}/samples/nano_dy.root": "Events"},
        schemaclass=NanoAODSchema,
        mode="eager",
    ).events()

    rntuple_path = str(tmp_path / "skim_rntuple.root")
    ttree_path = str(tmp_path / "skim_ttree.root")
    writeable = NanoAODSchema.uproot_writeable(orig_events)
    with uproot.recreate(ttree_path) as f:
        f.mktree("Events", writeable)
    with uproot.recreate(rntuple_path) as f:
        f.mkrntuple("Events", writeable)

    access_log = [] if mode == "virtual" else None
    # a second partition reads the RNTuple from past entry 0
    options = {"steps_per_file": 2} if mode == "dask" else {}
    skim_r = NanoEventsFactory.from_root(
        {rntuple_path: "Events"},
        schemaclass=NanoAODSchema,
        mode=mode,
        access_log=access_log,
        **options,
    ).events()
    skim_t = NanoEventsFactory.from_root(
        {ttree_path: "Events"},
        schemaclass=NanoAODSchema,
        mode=mode,
        **options,
    ).events()

    if mode == "virtual":
        assert len(access_log) == 0
        ak.materialize(skim_r.Muon.pt)
        assert {x.branch for x in access_log} == {"Muon.dxy", "Muon.pt"}

    if mode != "dask":
        assert type(skim_r.Muon).__name__ == "MuonArray"
    for column in (lambda e: e.Muon.pt, lambda e: e.Muon.matched_jet.pt):
        r, t = column(skim_r), column(skim_t)
        if mode == "dask":
            r, t = r.compute(), t.compute()
        assert ak.array_equal(r, t, check_parameters=False)
