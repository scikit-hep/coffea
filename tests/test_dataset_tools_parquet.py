import contextlib

import awkward
import pytest

from coffea.dataset_tools import preprocess
from coffea.dataset_tools.filespec import DataGroupSpec
from coffea.dataset_tools.preprocess import get_parquet_form_uuid_steps
from coffea.util import decompress_form

pytest.importorskip("dask_awkward")

_parquet_fileset = {"ZParquet": {"files": {"tests/samples/nano_dy.parquet": None}}}
_parquet_empty_fileset = {
    "ZParquetEmpty": {"files": {"tests/samples/nano_dy_empty.parquet": None}}
}
_mixed_fileset = {
    "ZRoot": {"files": {"tests/samples/nano_dy.root": "Events"}},
    "ZParquet": {"files": {"tests/samples/nano_dy.parquet": None}},
    "DataRoot": {"files": {"tests/samples/nano_dimuon.root": "Events"}},
}

_steps_by_size = [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]]
_steps_by_row_group = [[0, 40]]  # nano_dy.parquet is a single 40-row row group


def _normed(file, object_path=None, steps=None, num_entries=None, uuid=None):
    """Build a single-row normalized-file awkward record as get_parquet_form_uuid_steps expects."""
    return awkward.Array(
        [
            {
                "file": file,
                "object_path": object_path,
                "steps": steps,
                "num_entries": num_entries,
                "uuid": uuid,
            }
        ]
    )


# ---------------------------------------------------------------------------
# get_parquet_form_uuid_steps: unit tests (no dask).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "use_row_groups, expected_steps, warns",
    [
        (False, _steps_by_size, False),
        # A single 40-row row group with a 7-entry target yields one oversized, irregular step.
        (True, _steps_by_row_group, True),
    ],
)
def test_get_parquet_form_uuid_steps_branches(use_row_groups, expected_steps, warns):
    ctx = (
        pytest.warns(UserWarning, match="use_row_groups=True")
        if warns
        else contextlib.nullcontext()
    )
    with ctx:
        out = get_parquet_form_uuid_steps(
            _normed("tests/samples/nano_dy.parquet"),
            step_size=7,
            use_row_groups=use_row_groups,
        )
    assert out[0]["num_entries"] == 40
    assert awkward.to_list(out[0]["steps"]) == expected_steps


@pytest.mark.parametrize("use_row_groups", [False, True])
def test_get_parquet_form_uuid_steps_empty_file(use_row_groups):
    # Regression (Blocker 1): an empty parquet file with the default step_size=None used to
    # compute round(0 / 0) and raise ZeroDivisionError; it must now return [[0, 0]].
    out = get_parquet_form_uuid_steps(
        _normed("tests/samples/nano_dy_empty.parquet"),
        step_size=None,
        use_row_groups=use_row_groups,
    )
    assert out[0]["num_entries"] == 0
    assert awkward.to_list(out[0]["steps"]) == [[0, 0]]


def test_get_parquet_form_uuid_steps_save_form():
    out = get_parquet_form_uuid_steps(
        _normed("tests/samples/nano_dy.parquet"), step_size=7, save_form=True
    )
    assert out[0]["compressed_form"] is not None
    assert (
        len(awkward.forms.from_json(decompress_form(out[0]["compressed_form"])).fields)
        > 0
    )


@pytest.mark.parametrize("skip_bad_files", [True, False])
def test_get_parquet_form_uuid_steps_bad_file(skip_bad_files):
    # A missing parquet path raises ValueError; skip_bad_files decides whether it is
    # swallowed (yielding a null entry) or propagated.
    normed = _normed("tests/samples/does_not_exist.parquet")
    if skip_bad_files:
        out = get_parquet_form_uuid_steps(
            normed, step_size=7, skip_bad_files=True, file_exceptions=(ValueError,)
        )
        assert awkward.to_list(out) == [None]
    else:
        with pytest.raises(ValueError):
            get_parquet_form_uuid_steps(
                normed, step_size=7, skip_bad_files=False, file_exceptions=(ValueError,)
            )


# ---------------------------------------------------------------------------
# preprocess: end-to-end parquet path (through dask).
# ---------------------------------------------------------------------------
@pytest.mark.dask_client
@pytest.mark.parametrize(
    "use_row_groups, expected_steps",
    [(False, _steps_by_size), (True, _steps_by_row_group)],
)
def test_preprocess_parquet_steps(dask_client, use_row_groups, expected_steps):
    with dask_client.as_current() as _:
        runnable, _updated = preprocess(
            DataGroupSpec(_parquet_fileset),
            step_size=7,
            use_row_groups=use_row_groups,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=False,
        )
    fspec = runnable["ZParquet"].files["tests/samples/nano_dy.parquet"]
    assert runnable["ZParquet"].format == "parquet"
    assert runnable["ZParquet"].compressed_form is None
    assert fspec.num_entries == 40
    assert fspec.steps == expected_steps


@pytest.mark.dask_client
def test_preprocess_parquet_save_form(dask_client):
    with dask_client.as_current() as _:
        runnable, _updated = preprocess(
            DataGroupSpec(_parquet_fileset),
            step_size=7,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=True,
        )
    assert runnable["ZParquet"].compressed_form is not None
    assert len(runnable["ZParquet"].form.fields) > 0


@pytest.mark.dask_client
@pytest.mark.parametrize("use_row_groups", [False, True])
def test_preprocess_parquet_empty_file(dask_client, use_row_groups):
    # Regression (Blocker 1): a dataset made only of an empty parquet file must not raise
    # and must record a single [0, 0] step.
    with dask_client.as_current() as _:
        _runnable, updated = preprocess(
            DataGroupSpec(_parquet_empty_fileset),
            step_size=7,
            use_row_groups=use_row_groups,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=False,
        )
    fspec = updated["ZParquetEmpty"].files["tests/samples/nano_dy_empty.parquet"]
    assert fspec.num_entries == 0
    assert fspec.steps == [[0, 0]]


# ---------------------------------------------------------------------------
# preprocess: mixed ROOT + parquet dispatcher.
# ---------------------------------------------------------------------------
@pytest.mark.dask_client
def test_preprocess_mixed_root_and_parquet(dask_client):
    with dask_client.as_current() as _:
        runnable, updated = preprocess(
            DataGroupSpec(_mixed_fileset),
            step_size=7,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=False,
        )
    # The dispatcher splits the fileset by format and recombines in the original order.
    for out in (runnable, updated):
        assert list(out.keys()) == ["ZRoot", "ZParquet", "DataRoot"]
    assert [runnable[k].format for k in runnable.keys()] == ["root", "parquet", "root"]
    assert runnable["ZRoot"].files["tests/samples/nano_dy.root"].steps == _steps_by_size
    assert runnable["ZParquet"].files["tests/samples/nano_dy.parquet"].num_entries == 40
    assert (
        runnable["DataRoot"].files["tests/samples/nano_dimuon.root"].num_entries == 40
    )


@pytest.mark.dask_client
def test_preprocess_mixed_format_alignment_warning(dask_client):
    # Mixed root+parquet with align_clusters != use_row_groups warns that the two alignment
    # knobs serve a similar purpose.
    with dask_client.as_current() as _:
        with pytest.warns(UserWarning, match="align_clusters and use_row_groups"):
            preprocess(
                DataGroupSpec(_mixed_fileset),
                step_size=7,
                align_clusters=True,
                use_row_groups=False,
                files_per_batch=10,
                skip_bad_files=True,
                save_form=False,
            )
