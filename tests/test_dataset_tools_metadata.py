"""Tests for user-supplied metadata extraction during pydantic preprocessing."""

import awkward
import pytest

from coffea.dataset_tools import DataGroupSpec, preprocess

_DY = "tests/samples/nano_dy.root"


def _nbranches(file_handle):
    """Extractor: number of branches/fields in the Events object of an open ROOT file."""
    return {"nbranches": len(file_handle["Events"].keys())}


def _sumw_style(file_handle):
    """Extractor shaped like the ATLAS CutBookkeepers sum-of-weights pattern."""
    return {"sumw": float(file_handle["Events"].num_entries)}


def _sum_sumw(per_file):
    """Reducer: dataset-level sum of the per-file sumw values."""
    return {"sumw_dataset": sum(meta["sumw"] for meta in per_file.values())}


def _parquet_rows(parquet_metadata):
    return {"rows": int(parquet_metadata["num_rows"])}


def _returns_non_dict(file_handle):
    return 42


def _returns_unserializable(file_handle):
    return {"handle": object()}


def _raises_oserror(file_handle):
    raise OSError("broken metadata source")


@pytest.fixture
def two_files(tmp_path):
    import shutil

    copy = str(tmp_path / "nano_dy_copy.root")
    shutil.copy(_DY, copy)
    return _DY, copy


def test_extractor_fills_file_metadata(two_files):
    file_a, file_b = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events", file_b: "Events"}}})
    available, updated = preprocess(
        dgs, save_form=False, backend="iterative", metadata_extractor=_nbranches
    )
    for out in (available, updated):
        for fs in out["D"].files.values():
            assert fs.metadata is not None
            assert fs.metadata["nbranches"] > 0


def test_reducer_fills_dataset_metadata(two_files):
    file_a, file_b = two_files
    dgs = DataGroupSpec(
        {"D": {"files": {file_a: "Events", file_b: "Events"}, "metadata": {"xs": 1.5}}}
    )
    available, updated = preprocess(
        dgs,
        save_form=False,
        backend="iterative",
        metadata_extractor=_sumw_style,
        metadata_reducer=_sum_sumw,
    )
    for out in (available, updated):
        # both files have 40 entries; existing dataset metadata is preserved
        assert out["D"].metadata["sumw_dataset"] == 80.0
        assert out["D"].metadata["xs"] == 1.5
        for fs in out["D"].files.values():
            assert fs.metadata == {"sumw": 40.0}


def test_extractor_parquet(tmp_path):
    path = str(tmp_path / "d.parquet")
    awkward.to_parquet(awkward.Array([{"x": 1.0}, {"x": 2.0}]), path)
    dgs = DataGroupSpec({"P": {"files": {path: None}}})
    available, _ = preprocess(
        dgs, save_form=False, backend="iterative", metadata_extractor=_parquet_rows
    )
    assert available["P"].files[path].metadata == {"rows": 2}


def test_extractor_backend_equivalence(two_files):
    file_a, file_b = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events", file_b: "Events"}}})
    kwargs = dict(
        save_form=True,
        metadata_extractor=_sumw_style,
        metadata_reducer=_sum_sumw,
    )
    a_iter, u_iter = preprocess(dgs, backend="iterative", **kwargs)
    a_fut, u_fut = preprocess(dgs, backend="futures", **kwargs)
    assert a_iter == a_fut
    assert u_iter == u_fut
    assert a_iter["D"].metadata["sumw_dataset"] == 80.0


def test_extractor_dask_backend(two_files):
    pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    file_a, file_b = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events", file_b: "Events"}}})
    kwargs = dict(
        save_form=True,
        metadata_extractor=_sumw_style,
        metadata_reducer=_sum_sumw,
    )
    a_dask, u_dask = preprocess(dgs, backend="dask", scheduler="synchronous", **kwargs)
    a_iter, u_iter = preprocess(dgs, backend="iterative", **kwargs)
    assert a_dask == a_iter
    assert u_dask == u_iter


def test_extractor_failure_participates_in_skip_bad_files(two_files, tmp_path):
    file_a, file_b = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events", file_b: "Events"}}})
    with pytest.raises(OSError, match="broken metadata source"):
        preprocess(
            dgs,
            save_form=False,
            backend="iterative",
            metadata_extractor=_raises_oserror,
        )
    # a failing extractor is a per-file failure: skip_bad_files drops the files
    available, updated = preprocess(
        dgs,
        save_form=False,
        backend="iterative",
        skip_bad_files=True,
        allow_empty_datasets=True,
        metadata_extractor=_raises_oserror,
    )
    assert "D" not in available or len(available["D"].files) == 0


def test_extractor_non_dict_raises(two_files):
    file_a, _ = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events"}}})
    with pytest.raises(ValueError, match="must return a dict"):
        preprocess(
            dgs,
            save_form=False,
            backend="iterative",
            metadata_extractor=_returns_non_dict,
        )


def test_extractor_unserializable_raises(two_files):
    file_a, _ = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events"}}})
    with pytest.raises(ValueError, match="not JSON-serializable"):
        preprocess(
            dgs,
            save_form=False,
            backend="iterative",
            metadata_extractor=_returns_unserializable,
        )


def test_extractor_rejected_on_legacy_path(two_files):
    file_a, _ = two_files
    with pytest.raises(ValueError, match="not supported"):
        preprocess(
            {"D": {"files": {file_a: "Events"}}},
            preprocess_legacy_root=True,
            metadata_extractor=_nbranches,
        )


def test_file_metadata_survives_json_roundtrip(two_files):
    file_a, file_b = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events", file_b: "Events"}}})
    available, _ = preprocess(
        dgs, save_form=True, backend="iterative", metadata_extractor=_nbranches
    )
    restored = DataGroupSpec.model_validate_json(available.model_dump_json())
    assert restored == available
    for fname, fs in available["D"].files.items():
        assert restored["D"].files[fname].metadata == fs.metadata


def test_file_metadata_excluded_from_legacy_dict_output(two_files):
    file_a, file_b = two_files
    available, _ = preprocess(
        {"D": {"files": {file_a: "Events", file_b: "Events"}}},
        save_form=False,
        backend="iterative",
        metadata_extractor=_sumw_style,
        metadata_reducer=_sum_sumw,
    )
    # dict-in/dict-out: per-file metadata is not part of the legacy format, but the
    # reduced dataset-level metadata is
    for file_info in available["D"]["files"].values():
        assert "metadata" not in file_info
    assert available["D"]["metadata"]["sumw_dataset"] == 80.0


def test_filespec_metadata_add_merges(two_files):
    file_a, _ = two_files
    dgs = DataGroupSpec({"D": {"files": {file_a: "Events"}}})
    available, _ = preprocess(
        dgs,
        save_form=False,
        step_size=20,
        backend="iterative",
        metadata_extractor=_nbranches,
    )
    fs = available["D"].files[file_a]
    assert len(fs.steps) == 2
    merged = fs.limit_steps(slice(0, 1)) + fs.limit_steps(slice(1, 2))
    assert merged.metadata == fs.metadata


def test_extractor_with_rntuple(tmp_path):
    """The extractor receives the open file for RNTuple inputs too."""
    rnt = "tests/samples/nano_dy_rntuple.root"
    dgs = DataGroupSpec({"D": {"files": {rnt: "Events"}}})
    available, _ = preprocess(
        dgs, save_form=False, backend="iterative", metadata_extractor=_sumw_style
    )
    assert available["D"].files[rnt].metadata == {"sumw": 40.0}
