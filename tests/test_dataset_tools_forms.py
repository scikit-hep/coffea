"""Tests for union-form generation via DatasetSpec addition and the forms helpers."""

import awkward
import numpy
import pytest
import uproot

from coffea.dataset_tools import DataGroupSpec, preprocess
from coffea.dataset_tools.forms import (
    decode_field_bitset,
    encode_field_bitset,
    prune_form_fields,
    sort_form_fields,
    union_form_jsonstr,
)

_DY = "tests/samples/nano_dy.root"

# Two CMS NanoAOD files carrying disjoint subsets of GenModel_TChiZH_* model-point flags
# (the GenModel case #1478 targets): file A has GenModel_TChiZH_700_1, file B does not.
_GENMODEL_A = "tests/samples/nano_genmodel_with20_700_1_with0_1100_200_with0_950_400"
_GENMODEL_B = "tests/samples/nano_genmodel_without_700_1_with0_1100_200_with20_950_400"


def _record_form(**fields):
    return awkward.Array([fields]).layout.form


@pytest.fixture
def hlt_files(tmp_path):
    """Two small TTree files sharing a flat branch but with disjoint HLT-style bool branches
    (the per-file field variation the union form exists to cover)."""
    file_a = str(tmp_path / "hlt_a.root")
    file_b = str(tmp_path / "hlt_b.root")
    with uproot.recreate(file_a) as f:
        f["Events"] = {
            "x": numpy.arange(10, dtype="f8"),
            "HLT_a": numpy.ones(10, dtype=bool),
        }
    with uproot.recreate(file_b) as f:
        f["Events"] = {
            "x": numpy.arange(10, dtype="f8"),
            "HLT_b": numpy.zeros(10, dtype=bool),
        }
    return file_a, file_b


def _preprocessed(files, name="D"):
    dgs = DataGroupSpec({name: {"files": files}})
    available, _ = preprocess(dgs, save_form=True, backend="iterative")
    return available[name]


# --------------------------------------------------------------------------------------
# forms helpers
# --------------------------------------------------------------------------------------


def test_union_form_jsonstr_first_seen_and_sorted():
    fa = _record_form(x=1.0, HLT_a=True)
    fb = _record_form(x=1.0, HLT_b=False)
    ab = union_form_jsonstr([fa, fb])
    ba = union_form_jsonstr([fb, fa])
    # merge order perturbs the byte-level field order but not form equality
    assert ab != ba
    assert awkward.forms.from_json(ab) == awkward.forms.from_json(ba)
    # sort_fields canonicalizes the bytes independent of merge order
    fa2 = _record_form(x=1.0, HLT_a=True)
    fb2 = _record_form(x=1.0, HLT_b=False)
    ab_sorted = union_form_jsonstr([fa2, fb2], sort_fields=True)
    ba_sorted = union_form_jsonstr([fb2, fa2], sort_fields=True)
    assert ab_sorted == ba_sorted
    assert awkward.forms.from_json(ab_sorted).fields == sorted(
        awkward.forms.from_json(ab).fields
    )


def test_union_form_jsonstr_empty_returns_none():
    assert union_form_jsonstr([]) is None


def test_sort_form_fields_recursive_and_equal():
    form = _record_form(b=1.0, a={"y": 1, "x": 2.0})
    sorted_form = sort_form_fields(form)
    assert sorted_form.fields == ["a", "b"]
    assert sorted_form.contents[0].fields == ["x", "y"]
    assert sorted_form == form


def test_prune_form_fields_top_level():
    form = _record_form(x=1.0, y=2, z=True)
    pruned = prune_form_fields(form, {"x", "z"})
    assert pruned.fields == ["x", "z"]


def test_field_bitset_roundtrip():
    union_fields = ["a", "b", "c", "d"]
    present = {"a", "c"}
    bitset = encode_field_bitset(present, union_fields)
    assert decode_field_bitset(bitset, union_fields) == present
    # fields outside the union are ignored on encode
    assert encode_field_bitset({"a", "c", "zz"}, union_fields) == bitset


# --------------------------------------------------------------------------------------
# DatasetSpec.__add__ / union_with
# --------------------------------------------------------------------------------------


def test_add_unions_forms_matches_joint_preprocess(hlt_files):
    """Adding two separately preprocessed DatasetSpecs yields the same union form as
    preprocessing all files together, without re-opening any file."""
    file_a, file_b = hlt_files
    ds_a = _preprocessed({file_a: "Events"})
    ds_b = _preprocessed({file_b: "Events"})
    joint = _preprocessed({file_a: "Events", file_b: "Events"})

    combined = ds_a + ds_b
    assert combined.compressed_form is not None
    assert combined.form == joint.form
    assert set(combined.form.fields) == {"x", "HLT_a", "HLT_b"}
    # per-file bitsets decode to each file's own field set
    union_fields = list(combined.form.fields)
    assert decode_field_bitset(
        combined.files[file_a].experimental_field_bitset, union_fields
    ) == {"x", "HLT_a"}
    assert decode_field_bitset(
        combined.files[file_b].experimental_field_bitset, union_fields
    ) == {"x", "HLT_b"}


def test_add_equal_forms_short_circuits(tmp_path):
    import shutil

    copy = str(tmp_path / "nano_dy_copy.root")
    shutil.copy(_DY, copy)
    ds_a = _preprocessed({_DY: "Events"})
    ds_b = _preprocessed({copy: "Events"})
    combined = ds_a + ds_b
    assert combined.form == ds_a.form
    assert list(combined.form.fields) == list(ds_a.form.fields)


def test_add_one_sided_form_raises(hlt_files):
    file_a, file_b = hlt_files
    with_form = _preprocessed({file_a: "Events"})
    without_form = DataGroupSpec({"D": {"files": {file_b: "Events"}}})["D"]
    with pytest.raises(ValueError, match="saved form"):
        with_form + without_form
    with pytest.raises(ValueError, match="saved form"):
        without_form + with_form


def test_add_no_forms_stays_none(hlt_files):
    file_a, file_b = hlt_files
    ds_a = DataGroupSpec({"D": {"files": {file_a: "Events"}}})["D"]
    ds_b = DataGroupSpec({"D": {"files": {file_b: "Events"}}})["D"]
    assert (ds_a + ds_b).compressed_form is None


def test_union_with_sort_fields_is_order_independent(hlt_files):
    from coffea.util import decompress_form

    file_a, file_b = hlt_files
    ds_a = _preprocessed({file_a: "Events"})
    ds_b = _preprocessed({file_b: "Events"})
    ab = ds_a.union_with(ds_b, sort_fields=True)
    ba = ds_b.union_with(ds_a, sort_fields=True)
    assert decompress_form(ab.compressed_form) == decompress_form(ba.compressed_form)
    assert list(ab.form.fields) == sorted(ab.form.fields)


def test_datagroupspec_add_unions_same_name_dataset(hlt_files):
    file_a, file_b = hlt_files
    group_a = DataGroupSpec({"D": _preprocessed({file_a: "Events"}).model_dump()})
    group_b = DataGroupSpec({"D": _preprocessed({file_b: "Events"}).model_dump()})
    combined = group_a + group_b
    assert combined["D"].compressed_form is not None
    assert set(combined["D"].form.fields) == {"x", "HLT_a", "HLT_b"}


# --------------------------------------------------------------------------------------
# canonicalize_form / filter pruning
# --------------------------------------------------------------------------------------


def test_canonicalize_form_sorts_and_remaps_bitsets(hlt_files):
    file_a, file_b = hlt_files
    ds = _preprocessed({file_a: "Events", file_b: "Events"})
    canonical = ds.canonicalize_form()
    assert list(canonical.form.fields) == sorted(ds.form.fields)
    assert canonical.form == ds.form
    for fname in ds.files:
        before = decode_field_bitset(
            ds.files[fname].experimental_field_bitset, list(ds.form.fields)
        )
        after = decode_field_bitset(
            canonical.files[fname].experimental_field_bitset,
            list(canonical.form.fields),
        )
        assert before == after


def test_canonicalize_form_without_form_is_copy():
    ds = DataGroupSpec({"D": {"files": {_DY: "Events"}}})["D"]
    assert ds.canonicalize_form() == ds


def test_filter_files_prunes_union_form(hlt_files):
    file_a, file_b = hlt_files
    ds = _preprocessed({file_a: "Events", file_b: "Events"})

    filtered = ds.filter_files(filter_name=".*hlt_a.root")
    assert list(filtered.files) == [file_a]
    # the union form shrinks to exactly the fields the remaining file carries
    assert set(filtered.form.fields) == {"x", "HLT_a"}
    # the remaining file's bitset covers the whole pruned form
    assert decode_field_bitset(
        filtered.files[file_a].experimental_field_bitset, list(filtered.form.fields)
    ) == set(filtered.form.fields)


def test_filter_files_without_bitsets_keeps_superset(hlt_files):
    file_a, file_b = hlt_files
    ds = _preprocessed({file_a: "Events", file_b: "Events"})
    spec = ds.model_dump()
    for file_spec in spec["files"].values():
        file_spec["experimental_field_bitset"] = None
    ds_nobits = type(ds)(**spec)

    filtered = ds_nobits.filter_files(filter_name=".*hlt_a.root")
    assert filtered.form == ds.form


def test_limit_files_prunes_union_form(hlt_files):
    file_a, file_b = hlt_files
    ds = _preprocessed({file_a: "Events", file_b: "Events"})
    limited = ds.limit_files(1)
    remaining = next(iter(limited.files))
    remaining_fields = decode_field_bitset(
        limited.files[remaining].experimental_field_bitset, list(limited.form.fields)
    )
    assert remaining_fields == set(limited.form.fields)
    assert len(limited.form.fields) < len(ds.form.fields)


def test_preprocess_populates_bitsets(hlt_files):
    file_a, file_b = hlt_files
    ds = _preprocessed({file_a: "Events", file_b: "Events"})
    union_fields = list(ds.form.fields)
    all_fields = set()
    for fname, fs in ds.files.items():
        assert fs.experimental_field_bitset is not None
        present = decode_field_bitset(fs.experimental_field_bitset, union_fields)
        assert present <= set(union_fields)
        assert len(present) > 0
        all_fields |= present
    # the union of the per-file field sets is the full union form
    assert all_fields == set(union_fields)


def test_bitset_roundtrips_through_json(hlt_files):
    file_a, file_b = hlt_files
    ds = _preprocessed({file_a: "Events", file_b: "Events"})
    from coffea.dataset_tools.filespec import DatasetSpec

    restored = DatasetSpec.model_validate_json(ds.model_dump_json())
    assert restored == ds
    for fname in ds.files:
        assert (
            restored.files[fname].experimental_field_bitset
            == ds.files[fname].experimental_field_bitset
        )


# --------------------------------------------------------------------------------------
# GenModel: real CMS NanoAOD files with disjoint model-point (GenModel) branch subsets
# --------------------------------------------------------------------------------------

_GM_700 = "GenModel_TChiZH_700_1"


@pytest.mark.parametrize(
    "ext, object_path",
    [(".root", "Events"), (".parquet", None)],
)
def test_genmodel_union_makes_absent_fields_optional(ext, object_path):
    """Adding two DatasetSpecs whose files carry disjoint GenModel model-point flags unions
    their forms: the result is a superset of both, and a flag present in only one file becomes
    an option type so it stays readable (as None) for the file that lacks it -- the GenModel
    behavior #1478 targets, for both ROOT and parquet inputs."""
    files_a = {_GENMODEL_A + ext: object_path}
    files_b = {_GENMODEL_B + ext: object_path}
    da, _ = preprocess(
        DataGroupSpec({"genmodel": {"files": files_a}}),
        save_form=True,
        backend="iterative",
    )
    db, _ = preprocess(
        DataGroupSpec({"genmodel": {"files": files_b}}),
        save_form=True,
        backend="iterative",
    )
    fields_a = set(da["genmodel"].form.fields)
    fields_b = set(db["genmodel"].form.fields)
    # the fixtures differ in their GenModel subset, and only file A has GenModel_TChiZH_700_1
    assert fields_a != fields_b
    assert _GM_700 in fields_a and _GM_700 not in fields_b

    combined = da["genmodel"] + db["genmodel"]
    union = combined.form
    assert set(union.fields) == fields_a | fields_b

    # a flag present in only one file is an IndexedOptionArray(bool) in the union
    only_in_a = sorted(fields_a - fields_b)
    content = union.contents[union.fields.index(only_in_a[0])]
    assert isinstance(content, awkward.forms.IndexedOptionForm)
    assert isinstance(content.content, awkward.forms.NumpyForm)
    assert content.content.primitive == "bool"
    # GenModel_TChiZH_700_1 (present only in A) is optional in the union
    assert isinstance(
        union.contents[union.fields.index(_GM_700)], awkward.forms.IndexedOptionForm
    )


def test_genmodel_matches_joint_preprocess():
    """The union built by adding two separately-preprocessed GenModel DatasetSpecs equals the
    form built by preprocessing both files together in one dataset."""
    da, _ = preprocess(
        DataGroupSpec({"genmodel": {"files": {_GENMODEL_A + ".root": "Events"}}}),
        save_form=True,
        backend="iterative",
    )
    db, _ = preprocess(
        DataGroupSpec({"genmodel": {"files": {_GENMODEL_B + ".root": "Events"}}}),
        save_form=True,
        backend="iterative",
    )
    joint, _ = preprocess(
        DataGroupSpec(
            {
                "genmodel": {
                    "files": {
                        _GENMODEL_A + ".root": "Events",
                        _GENMODEL_B + ".root": "Events",
                    }
                }
            }
        ),
        save_form=True,
        backend="iterative",
    )
    assert (da["genmodel"] + db["genmodel"]).form == joint["genmodel"].form


def test_genmodel_bitsets_and_filter_prune():
    """Each GenModel file's experimental bitset decodes to its own branch set, and filtering the
    combined dataset back to one file prunes the union form to that file's branches."""
    combined, _ = preprocess(
        DataGroupSpec(
            {
                "genmodel": {
                    "files": {
                        _GENMODEL_A + ".root": "Events",
                        _GENMODEL_B + ".root": "Events",
                    }
                }
            }
        ),
        save_form=True,
        backend="iterative",
    )
    ds = combined["genmodel"]
    union_fields = list(ds.form.fields)
    present_a = decode_field_bitset(
        ds.files[_GENMODEL_A + ".root"].experimental_field_bitset, union_fields
    )
    assert _GM_700 in present_a

    only_a = ds.filter_files(filter_name=".*with20_700_1.*")
    assert list(only_a.files) == [_GENMODEL_A + ".root"]
    # the pruned form is exactly file A's fields, and still carries GenModel_TChiZH_700_1
    assert set(only_a.form.fields) == present_a
    assert _GM_700 in only_a.form.fields
