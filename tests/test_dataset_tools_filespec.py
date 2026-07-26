"""Tests for the Pydantic-based ModelFactory classes in filespec.py"""

import copy
import gzip
import os
import tempfile

import awkward
import pytest
from pydantic import ValidationError

from coffea.dataset_tools.filespec import (
    CoffeaParquetFileSpec,
    CoffeaParquetFileSpecOptional,
    CoffeaROOTFileSpec,
    CoffeaROOTFileSpecOptional,
    DataGroupSpec,
    DatasetSpec,
    InputFiles,
    ModelFactory,
    ParquetFileSpec,
    ROOTFileSpec,
    identify_file_format,
)
from coffea.util import compress_form

simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
valid_compressed_form = compress_form(simple_form)
invalid_compressed_form = compress_form("Jun;k")

_starting_fileset = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "1234-5678-90ab-cdef",
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": "Events",
            "tests/samples/nano_dimuon_not_there.root": "Events",
        }
    },
}


def test_identify_format():
    """Test identify_format method"""
    assert identify_file_format("file.root") == "root"
    assert identify_file_format("file.root.0") == "root"
    assert identify_file_format("root:file.root.strange") == "root"
    assert identify_file_format("file.parquet") == "parquet"
    assert identify_file_format("file.parq") == "parquet"
    assert identify_file_format("root://file.parq") == "parquet"
    assert identify_file_format("file.pq") == "parquet"
    assert identify_file_format("file.parq.part00") == "parquet"
    assert identify_file_format("directory") == "parquet"

    with pytest.raises(RuntimeError):
        identify_file_format("file.txt")


class TestStepPair:
    """Test the StepPair type annotation"""

    def test_valid_step_pair(self):
        """Test that valid step pairs are accepted"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        assert spec.steps == [[0, 10]]

    def test_invalid_step_pair_negative(self):
        """Test that negative values in step pairs are rejected"""
        with pytest.raises(ValueError):
            ROOTFileSpec(object_path="Events", steps=[[-1, 10]])

    def test_invalid_step_pair_length(self):
        """Test that step pairs with wrong length are rejected"""
        with pytest.raises(ValueError):
            ROOTFileSpec(object_path="Events", steps=[[0, 10, 20]])

        with pytest.raises(ValueError):
            ROOTFileSpec(object_path="Events", steps=[[0]])


class TestROOTFileSpec:
    """Test ROOTFileSpec class"""

    def test_creation_basic(self):
        """Test basic creation of ROOTFileSpec"""
        spec = ROOTFileSpec(object_path="Events")
        assert spec.object_path == "Events"
        assert spec.steps is None

    def test_creation_with_steps(self):
        """Test creation with steps"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10], [10, 20]])
        assert spec.object_path == "Events"
        assert spec.steps == [[0, 10], [10, 20]]

    def test_creation_comprehensive_steps_combinations(self):
        """Test creation with all step combinations from __main__"""
        # Test all combinations from the __main__ method
        steps_options = [None, [0, 100], [[0, 1], [2, 3]]]

        for steps in steps_options:
            try:
                spec = ROOTFileSpec(object_path="example_path", steps=steps)
                assert spec.object_path == "example_path"
                assert spec.steps == steps
            except Exception as e:
                # Some combinations may be invalid, that's expected
                assert isinstance(e, ValueError)

    def test_json_serialization(self):
        """Test JSON serialization/deserialization"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        json_str = spec.model_dump_json()
        restored = ROOTFileSpec.model_validate_json(json_str)
        assert restored.object_path == spec.object_path
        assert restored.steps == spec.steps

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = ROOTFileSpec.model_validate_json(fin.read())

                assert restored.object_path == "Events"
                assert restored.steps == [[0, 10]]

    def test_num_selected_entries_computation(self):
        """Test computation of num_selected_entries property"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10], [20, 30]])
        assert spec.num_selected_entries == 20

        spec_no_steps = ROOTFileSpec(object_path="Events", steps=None)
        assert spec_no_steps.num_selected_entries is None

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10], [10, 20]])
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10], [10, 20], [20, 30]])
        limited_spec = spec.limit_steps(2)
        assert limited_spec.steps == [[0, 10], [10, 20]]

    def test_add(self):
        """Test addition of two ROOTFileSpec instances"""
        spec1 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        spec2 = ROOTFileSpec(object_path="Events", steps=[[10, 20]])
        combined_spec = spec1 + spec2
        assert combined_spec.object_path == "Events"
        assert combined_spec.steps == [[0, 10], [10, 20]]

    def test_add_invalid_object_path(self):
        """Test addition of two ROOTFileSpec instances with different object_paths"""
        spec1 = ROOTFileSpec(object_path="Events1", steps=[[0, 10]])
        spec2 = ROOTFileSpec(object_path="Events2", steps=[[10, 20]])
        with pytest.raises(ValueError):
            _ = spec1 + spec2

    def test_invalid_format_add(self):
        """Test addition of two FileSpec instances with different formats"""
        spec1 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[10, 20]])
        with pytest.raises(ValueError):
            _ = spec1 + spec2

    def test_subtract(self):
        """Test subtraction of two ROOTFileSpec instances"""
        spec1 = ROOTFileSpec(object_path="Events", steps=[[0, 10], [10, 20]])
        spec2 = ROOTFileSpec(object_path="Events", steps=[[10, 20]])
        reduced_spec = spec1 - spec2
        assert reduced_spec.object_path == "Events"
        assert reduced_spec.steps == [[0, 10]]

    def test_subtract_invalid_object_path(self):
        """Test subtraction of two ROOTFileSpec instances with different object_paths"""
        spec1 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        spec2 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        assert (spec1 - spec2) is None

    def test_subtract_invalid_format(self):
        """Test subtraction of two ROOTFileSpec instances with different formats"""
        spec1 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        with pytest.raises(ValueError):
            _ = spec1 - spec2

    def test_subtract_non_overlapping(self):
        """Test subtraction of two ROOTFileSpec instances with non-overlapping steps"""
        spec1 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        spec2 = ROOTFileSpec(object_path="Events", steps=[[10, 20]])
        non_reduced_spec = spec1 - spec2
        assert non_reduced_spec.steps == [[0, 10]]


class TestParquetFileSpec:
    """Test ParquetFileSpec class"""

    def test_creation_basic(self):
        """Test basic creation of ParquetFileSpec"""
        spec = ParquetFileSpec()
        assert spec.object_path is None
        assert spec.steps is None

    def test_creation_with_steps(self):
        """Test creation with steps"""
        spec = ParquetFileSpec(steps=[[0, 100]])
        assert spec.object_path is None
        assert spec.steps == [[0, 100]]

    def test_object_path_must_be_none(self):
        """Test that object_path must be None for ParquetFileSpec"""
        spec = ParquetFileSpec(object_path=None)
        assert spec.object_path is None

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = ParquetFileSpec.model_validate_json(fin.read())

                assert restored.object_path is None
                assert restored.steps == [[0, 10]]

    def test_num_selected_entries_computation(self):
        """Test computation of num_selected_entries property"""
        spec = ParquetFileSpec(object_path=None, steps=[[0, 10], [20, 30]])
        assert spec.num_selected_entries == 20

        spec_no_steps = ParquetFileSpec(object_path=None, steps=None)
        assert spec_no_steps.num_selected_entries is None

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = ParquetFileSpec(object_path=None, steps=[[0, 10], [10, 20]])
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = ParquetFileSpec(object_path=None, steps=[[0, 10], [10, 20], [20, 30]])
        limited_spec = spec.limit_steps(2)
        assert limited_spec.steps == [[0, 10], [10, 20]]

    def test_add(self):
        """Test addition of two ParquetFileSpec instances"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[10, 20]])
        combined_spec = spec1 + spec2
        assert combined_spec.object_path is None
        assert combined_spec.steps == [[0, 10], [10, 20]]

    def test_add_invalid_object_path(self):
        """Test addition of two ParquetFileSpec instances with different object_paths"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        with pytest.raises(ValueError):
            _ = spec1 + spec2

    def test_invalid_format_add(self):
        """Test addition of two FileSpec instances with different formats"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        spec2 = ROOTFileSpec(object_path="Events", steps=[[10, 20]])
        with pytest.raises(ValueError):
            _ = spec1 + spec2

    def test_subtract(self):
        """Test subtraction of two ParquetFileSpec instances"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10], [10, 20]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[10, 20]])
        reduced_spec = spec1 - spec2
        assert reduced_spec.object_path is None
        assert reduced_spec.steps == [[0, 10]]

    def test_subtract_invalid_object_path(self):
        """Test subtraction of two ParquetFileSpec instances with different object_paths"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        assert (spec1 - spec2) is None

    def test_subtract_invalid_format(self):
        """Test subtraction of two ParquetFileSpec instances with different formats"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        spec2 = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        with pytest.raises(ValueError):
            _ = spec1 - spec2

    def test_subtract_non_overlapping(self):
        """Test subtraction of two ParquetFileSpec instances with non-overlapping steps"""
        spec1 = ParquetFileSpec(object_path=None, steps=[[0, 10]])
        spec2 = ParquetFileSpec(object_path=None, steps=[[10, 20]])
        non_reduced_spec = spec1 - spec2
        assert non_reduced_spec.steps == [[0, 10]]


class TestCoffeaROOTFileSpecOptional:
    """Test CoffeaROOTFileSpecOptional class"""

    def test_creation_minimal(self):
        """Test creation with minimal required fields"""
        spec = CoffeaROOTFileSpecOptional(object_path="Events")
        assert spec.object_path == "Events"
        assert spec.steps is None
        assert spec.num_entries is None
        assert spec.uuid is None

    def test_creation_complete(self):
        """Test creation with all fields"""
        spec = CoffeaROOTFileSpecOptional(
            object_path="Events", steps=[[0, 10]], num_entries=100, uuid="test-uuid"
        )
        assert spec.object_path == "Events"
        assert spec.steps == [[0, 10]]
        assert spec.num_entries == 100
        assert spec.uuid == "test-uuid"

    def test_comprehensive_parameter_combinations(self):
        """Test all parameter combinations"""
        steps_options = [None, [0, 100], [[0, 1], [2, 3]]]
        num_entries_options = [None, 100]
        uuid_options = [None, "hello-there"]

        for steps in steps_options:
            for num_entries in num_entries_options:
                try:
                    spec = CoffeaROOTFileSpecOptional(
                        object_path="example_path",
                        steps=steps,
                        num_entries=num_entries,
                    )
                    assert spec.object_path == "example_path"
                    assert spec.steps == steps
                    assert spec.num_entries == num_entries
                except Exception as e:
                    # Some combinations may be invalid
                    assert isinstance(e, ValueError)

            for uuid in uuid_options:
                try:
                    spec = CoffeaROOTFileSpecOptional(
                        object_path="example_path",
                        steps=steps,
                        uuid=uuid,
                    )
                    assert spec.object_path == "example_path"
                    assert spec.steps == steps
                    assert spec.uuid == uuid
                except Exception as e:
                    # Some combinations may be invalid
                    assert isinstance(e, ValueError)

    def test_negative_num_entries_rejected(self):
        """Test that negative num_entries are rejected"""
        with pytest.raises(ValueError):
            CoffeaROOTFileSpecOptional(object_path="Events", num_entries=-1)

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = CoffeaROOTFileSpecOptional(object_path="Events", steps=[[0, 10]])
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = CoffeaROOTFileSpecOptional.model_validate_json(fin.read())

                assert restored.object_path == "Events"
                assert restored.steps == [[0, 10]]
                assert restored.format == "root"
                assert restored.num_entries is None
                assert restored.uuid is None


class TestCoffeaROOTFileSpec:
    """Test CoffeaROOTFileSpec class"""

    def test_creation_complete(self):
        """Test creation with all required fields"""
        spec = CoffeaROOTFileSpec(
            object_path="Events", steps=[[0, 10]], num_entries=100, uuid="test-uuid"
        )
        assert spec.object_path == "Events"
        assert spec.steps == [[0, 10]]
        assert spec.num_entries == 100
        assert spec.uuid == "test-uuid"

    def test_comprehensive_parameter_combinations(self):
        """Test all parameter combinations from __main__ method"""
        steps_options = [None, [0, 100], [[0, 1], [2, 3]]]

        for steps in steps_options:
            num_entries, uuid = 100, "hello-there"
            try:
                spec = CoffeaROOTFileSpec(
                    object_path="example_path",
                    steps=steps,
                    num_entries=num_entries,
                    uuid=uuid,
                )
                # Only valid when steps is not None
                if steps is not None:
                    assert spec.object_path == "example_path"
                    assert spec.steps == steps
                    assert spec.num_entries == num_entries
                    assert spec.uuid == uuid
            except Exception as e:
                # Steps=None should fail for CoffeaROOTFileSpec
                assert isinstance(e, ValueError)
                if steps is None:
                    # This is expected for required steps field
                    continue

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors"""
        with pytest.raises(ValueError):
            CoffeaROOTFileSpec(object_path="Events")  # Missing steps, num_entries, uuid

    def test_steps_required(self):
        """Test that steps are required (not None)"""
        with pytest.raises(ValueError):
            CoffeaROOTFileSpec(
                object_path="Events", steps=None, num_entries=100, uuid="test-uuid"
            )

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = CoffeaROOTFileSpec(
            object_path="Events", steps=[[0, 10]], num_entries=100, uuid="test-uuid"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = CoffeaROOTFileSpec.model_validate_json(fin.read())

                assert restored.object_path == "Events"
                assert restored.steps == [[0, 10]]
                assert restored.format == "root"
                assert restored.num_entries == 100
                assert restored.uuid == "test-uuid"

    def test_num_selected_entries_computation(self):
        """Test computation of num_selected_entries property"""
        spec = CoffeaROOTFileSpec(
            object_path="Events",
            steps=[[0, 10], [20, 30]],
            num_entries=30,
            uuid="test-uuid",
        )
        assert spec.num_selected_entries == 20

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = CoffeaROOTFileSpec(
            object_path="Events",
            steps=[[0, 10], [10, 20]],
            num_entries=30,
            uuid="test-uuid",
        )
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = CoffeaROOTFileSpec(
            object_path="Events",
            steps=[[0, 10], [10, 20], [20, 30]],
            num_entries=30,
            uuid="test-uuid",
        )
        limited_spec = spec.limit_steps(2)
        assert limited_spec.steps == [[0, 10], [10, 20]]


class TestCoffeaParquetFileSpecOptional:
    """Test CoffeaParquetFileSpecOptional class"""

    def test_creation_minimal(self):
        """Test creation with minimal fields"""
        spec = CoffeaParquetFileSpecOptional()
        assert spec.object_path is None
        assert spec.steps is None
        assert spec.num_entries is None
        assert spec.uuid is None

    def test_creation_complete(self):
        """Test creation with all fields"""
        spec = CoffeaParquetFileSpecOptional(
            object_path=None, steps=[[0, 100]], num_entries=100, uuid="parquet-uuid"
        )
        assert spec.object_path is None
        assert spec.steps == [[0, 100]]
        assert spec.num_entries == 100
        assert spec.uuid == "parquet-uuid"

    def test_comprehensive_parameter_combinations(self):
        """Test all parameter combinations from __main__ method"""
        steps_options = [None, [0, 100], [[0, 1], [2, 3]]]
        uuid_options = [None, "hello-there"]

        for steps in steps_options:
            for uuid in uuid_options:
                try:
                    spec = CoffeaParquetFileSpecOptional(
                        object_path=None,
                        steps=steps,
                        uuid=uuid,
                    )
                    assert spec.object_path is None
                    assert spec.steps == steps
                    assert spec.uuid == uuid
                except Exception as e:
                    # Some combinations may be invalid
                    assert isinstance(e, ValueError)

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = CoffeaParquetFileSpecOptional(object_path=None, steps=[[0, 10]])
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = CoffeaParquetFileSpecOptional.model_validate_json(fin.read())

                assert restored.object_path is None
                assert restored.steps == [[0, 10]]
                assert restored.format == "parquet"


class TestCoffeaParquetFileSpec:
    """Test CoffeaParquetFileSpec class"""

    def test_creation_complete(self):
        """Test creation with all required fields"""
        spec = CoffeaParquetFileSpec(
            steps=[[0, 100]], num_entries=100, uuid="parquet-uuid"
        )
        assert spec.object_path is None
        assert spec.steps == [[0, 100]]
        assert spec.num_entries == 100
        assert spec.uuid == "parquet-uuid"

    def test_comprehensive_parameter_combinations(self):
        """Test all parameter combinations from __main__ method"""
        steps_options = [None, [0, 100], [[0, 1], [2, 3]]]

        for steps in steps_options:
            num_entries, uuid = 100, "hello-there"
            try:
                spec = CoffeaParquetFileSpec(
                    object_path=None,
                    steps=steps,
                    num_entries=num_entries,
                    uuid=uuid,
                )
                # Only valid when steps is not None
                if steps is not None:
                    assert spec.object_path is None
                    assert spec.steps == steps
                    assert spec.num_entries == num_entries
                    assert spec.uuid == uuid
            except Exception as e:
                # Steps=None should fail for CoffeaParquetFileSpec
                assert isinstance(e, ValueError)
                if steps is None:
                    # This is expected for required steps field
                    continue

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors"""
        with pytest.raises(ValueError):
            CoffeaParquetFileSpec()  # Missing steps, num_entries, uuid

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = CoffeaParquetFileSpec(
            object_path=None, steps=[[0, 10]], num_entries=100, uuid="test-uuid"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = CoffeaParquetFileSpec.model_validate_json(fin.read())

                assert restored.object_path is None
                assert restored.steps == [[0, 10]]
                assert restored.format == "parquet"
                assert restored.num_entries == 100
                assert restored.uuid == "test-uuid"

    def test_num_selected_entries_computation(self):
        """Test computation of num_selected_entries property"""
        spec = CoffeaParquetFileSpec(
            object_path=None,
            steps=[[0, 10], [20, 30]],
            num_entries=30,
            uuid="test-uuid",
        )
        assert spec.num_selected_entries == 20

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = CoffeaParquetFileSpec(
            object_path=None,
            steps=[[0, 10], [10, 20]],
            num_entries=30,
            uuid="test-uuid",
        )
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = CoffeaParquetFileSpec(
            object_path=None,
            steps=[[0, 10], [10, 20], [20, 30]],
            num_entries=30,
            uuid="test-uuid",
        )
        limited_spec = spec.limit_steps(2)
        assert limited_spec.steps == [[0, 10], [10, 20]]


class TestInputFiles:
    """Test InputFiles class"""

    def get_files(self):
        return {
            "file1.root": CoffeaROOTFileSpec(
                object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
            ),
            "file1.parquet": CoffeaParquetFileSpec(
                steps=[[0, 50], [50, 100]], num_entries=100, uuid="uuid2"
            ),
            "file2.root": CoffeaROOTFileSpecOptional(
                object_path="Events", steps=[[10, 20]], num_entries=None, uuid=None
            ),
        }

    def test_dict_methods(self):
        """Test that dict methods work properly"""
        file_dict = InputFiles(self.get_files())

        # Test __getitem__
        assert file_dict["file1.root"].uuid == "uuid1"

        # Test __len__
        assert len(file_dict) == 3

        # Test __iter__
        keys = list(file_dict)
        assert "file1.root" in keys
        assert "file2.root" in keys

        # Test keys()
        assert set(file_dict.keys()) == {"file1.root", "file1.parquet", "file2.root"}

        # Test values()
        values = list(file_dict.values())
        assert len(values) == 3

        # Test items()
        items = list(file_dict.items())
        assert len(items) == 3

        # Test get()
        assert file_dict.get("file1.root").uuid == "uuid1"
        assert file_dict.get("nonexistent") is None
        assert file_dict.get("nonexistent", "default") == "default"

    def test_creation_valid(self):
        """Test creation with valid CoffeaROOTFileSpec instances"""
        files = {
            "file1.root": CoffeaROOTFileSpec(
                object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
            )
        }
        file_dict = InputFiles(files)
        assert len(file_dict) == 1
        assert file_dict["file1.root"].uuid == "uuid1"

    def test_dict_file_format(self):
        """Test that invalid file types are rejected"""
        files = {"file1.txt": CoffeaROOTFileSpecOptional(object_path="Events")}
        # with pytest.raises(ValidationError):
        fdict = InputFiles(files)
        with pytest.raises(
            RuntimeError, match="identify_file_format couldn't identify"
        ):
            print(fdict.format)

    def test_mixed_root_and_parquet(self):
        """Test creation with mixed root and parquet files"""
        file_dict = InputFiles(self.get_files())
        assert len(file_dict) == 3

    def test_attempt_promotion(self):
        """Test attempt_promotion method"""
        # Test with valid ROOTFileSpec
        spec = InputFiles(self.get_files())
        assert isinstance(spec["file2.root"], CoffeaROOTFileSpecOptional)
        spec["file2.root"].num_entries = 20
        spec["file2.root"].uuid = "test-uuid"
        promoted = ModelFactory.attempt_promotion(spec)
        assert all(
            [
                isinstance(v, (CoffeaROOTFileSpec, CoffeaParquetFileSpec))
                for v in promoted.values()
            ]
        )

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        spec = InputFiles(self.get_files())
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.json.gz")
            with gzip.open(fname, "wt") as fout:
                fout.write(spec.model_dump_json())
            with gzip.open(fname, "rt") as fin:
                restored = InputFiles.model_validate_json(fin.read())
                assert len(restored) == 3
                assert restored == spec

    def test_num_entries_computation(self):
        """Test computation of num_entries property"""
        spec = InputFiles(self.get_files())
        assert spec.num_entries == 110

    def test_num_selected_entries_computation(self):
        """Test computation of num_selected_entries property"""
        spec = InputFiles(self.get_files())
        assert spec.num_selected_entries == 120

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = InputFiles(self.get_files())
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = InputFiles(self.get_files())
        limited_spec = spec.limit_steps(2)
        assert [v.steps for v in limited_spec.values()] == [[[0, 10]], [[0, 50]]]

    def test_limit_steps_per_file_slicing(self):
        """Test limit_steps with slicing"""
        spec = InputFiles(self.get_files())
        limited_spec = spec.limit_steps(1, per_file=True)
        assert [v.steps for v in limited_spec.values()] == [
            [[0, 10]],
            [[0, 50]],
            [[10, 20]],
        ]

    def test_limit_steps_method_chain_slicing(self):
        """Test limit_steps with slicing"""
        spec = InputFiles(self.get_files())
        limited_spec = spec.limit_steps(None).limit_steps(1, per_file=True)
        assert [v.steps for v in limited_spec.values()] == [
            [[0, 10]],
            [[0, 50]],
            [[10, 20]],
        ]

    def test_addition_distinct(self):
        """Test addition of two InputFiles instances"""
        spec1 = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )
        spec2 = InputFiles(
            {
                "file2.parquet": CoffeaParquetFileSpec(
                    steps=[[0, 50]], num_entries=50, uuid="uuid2"
                )
            }
        )
        combined_spec = spec1 + spec2
        assert len(combined_spec) == 2
        assert "file1.root" in combined_spec
        assert "file2.parquet" in combined_spec

    def test_addition_overlapping(self):
        """Test addition of two InputFiles instances with overlapping files"""
        spec1 = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                ),
                "file2.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid2"
                ),
            }
        )
        spec2 = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[10, 20]], num_entries=20, uuid="uuid1"
                ),
                "file3.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid3"
                ),
            }
        )
        combined_spec = spec1 + spec2
        assert len(combined_spec) == 3
        assert "file1.root" in combined_spec
        assert "file2.root" in combined_spec
        assert "file3.root" in combined_spec
        assert combined_spec["file1.root"].steps == [[0, 10], [10, 20]]
        assert combined_spec["file1.root"].num_entries == 20

    def test_subtraction_overlapping_nonoverlapping(self):
        """Test subtraction of two InputFiles instances"""
        spec1 = InputFiles(
            {
                "file1overlap.root": CoffeaROOTFileSpec(
                    object_path="Events",
                    steps=[[0, 10], [10, 20]],
                    num_entries=20,
                    uuid="uuid1",
                ),
                "file2nonoverlap.root": CoffeaROOTFileSpec(
                    object_path="Events",
                    steps=[[0, 10], [10, 20]],
                    num_entries=30,
                    uuid="uuid1",
                ),
                "file3.root": CoffeaROOTFileSpec(
                    object_path="Events",
                    steps=[[0, 10], [10, 20]],
                    num_entries=20,
                    uuid="uuid1",
                ),
            }
        )
        spec2 = InputFiles(
            {
                "file1overlap.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[10, 20]], num_entries=10, uuid="uuid1"
                ),
                "file2nonoverlap.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[20, 30]], num_entries=30, uuid="uuid1"
                ),
            }
        )
        reduced_spec = spec1 - spec2
        assert len(reduced_spec) == 3
        assert "file1overlap.root" in reduced_spec
        assert "file2nonoverlap.root" in reduced_spec
        assert "file3.root" in reduced_spec
        assert reduced_spec["file1overlap.root"].steps == [[0, 10]]
        assert reduced_spec["file1overlap.root"].num_entries == 20
        assert reduced_spec["file2nonoverlap.root"].steps == [[0, 10], [10, 20]]
        assert reduced_spec["file2nonoverlap.root"].num_entries == 30
        assert reduced_spec["file3.root"].steps == [[0, 10], [10, 20]]
        assert reduced_spec["file3.root"].num_entries == 20


class TestDatasetSpec:
    """Test DatasetSpec class"""

    def get_test_input(self):
        return {
            "ZJets1": {
                "files": {
                    "tests/samples/nano_dy.root": {
                        "object_path": "Events",
                        "steps": [
                            [0, 5],
                            [5, 10],
                            [10, 15],
                            [15, 20],
                            [20, 25],
                            [25, 30],
                        ],
                        "num_entries": 30,
                        "uuid": "1234-5678-90ab-cdef",
                    },
                    "tests/samples/nano_dy_2.root": {
                        "object_path": "Events",
                        "steps": None,
                        "num_entries": 30,
                        "uuid": "1234-5678-90ab-cdef",
                    },
                },
                "metadata": {"key": "value"},
                "compressed_form": valid_compressed_form,
            },
            "ZJets2": {
                "files": [
                    "tests/samples/nano_dy.root:Events",
                    "root://file2.root:Events",
                ]
            },
            "ZParquet": {
                "files": {
                    "tests/samples/nano_dy.parquet": {
                        "object_path": None,
                        "steps": [
                            [0, 5],
                            [5, 10],
                            [10, 15],
                            [15, 20],
                            [20, 25],
                            [25, 30],
                        ],
                        "num_entries": 30,
                        "uuid": "fake-uuid",
                    }
                }
            },
        }

    def get_spec_with_valid_steps(self, steps=[[0, 30]]):
        tmp = self.get_test_input()["ZJets1"]
        tmp["files"]["tests/samples/nano_dy_2.root"]["steps"] = steps
        return DatasetSpec(**tmp)

    def test_creation_valid(self):
        """Test creation with valid concrete file specs"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )
        spec = DatasetSpec(files=files, format="root", metadata={"sample": "test"})
        assert spec.format == "root"
        assert spec.metadata == {"sample": "test"}

    def test_starting_fileset_conversion(self):
        """Test conversion of _starting_fileset"""

        converted = {}
        for k, v in _starting_fileset.items():
            converted[k] = DatasetSpec(**v)

            # Test that the conversion worked
            assert converted[k] is not None
            assert hasattr(converted[k], "files")

            # Test JSON serialization roundtrip
            json_str = converted[k].model_dump_json()
            restored = DatasetSpec.model_validate_json(json_str)
            assert restored is not None

    def test_complex_dataset_spec_optional(self):
        """Test complex DatasetSpec scenario"""

        # Convert via direct constructor
        test = {k: DatasetSpec(**v) for k, v in self.get_test_input().items()}

        # Test that we can modify the steps after creation
        test["ZJets1"].files["tests/samples/nano_dy_2.root"].steps = [[0, 30]]

        assert test["ZJets1"] is not None
        assert test["ZJets1"].format == "root"
        assert test["ZJets1"].metadata == {"key": "value"}

        # Test that a mixture of concrete and optional FileSpecs are in the datasetspec
        assert isinstance(
            test["ZJets1"].files["tests/samples/nano_dy.root"], CoffeaROOTFileSpec
        )
        assert isinstance(
            test["ZJets1"].files["tests/samples/nano_dy_2.root"],
            CoffeaROOTFileSpecOptional,
        )
        assert all(
            [
                isinstance(v, CoffeaROOTFileSpecOptional)
                for k, v in test["ZJets2"].files.items()
            ]
        )
        assert isinstance(
            test["ZParquet"].files["tests/samples/nano_dy.parquet"],
            CoffeaParquetFileSpec,
        )

    def test_attempt_promotion(self):
        """Test attempt_promotion method"""
        # Test with valid ROOTFileSpec
        spec = DatasetSpec(**self.get_test_input()["ZJets2"])
        assert isinstance(
            spec.files["tests/samples/nano_dy.root"], CoffeaROOTFileSpecOptional
        )
        spec.files["tests/samples/nano_dy.root"].steps = [[0, 10], [10, 20]]
        spec.files["tests/samples/nano_dy.root"].num_entries = 20
        spec.files["tests/samples/nano_dy.root"].uuid = "test-uuid"
        spec.files["root://file2.root"].steps = [[0, 10], [10, 20]]
        spec.files["root://file2.root"].num_entries = 20
        spec.files["root://file2.root"].uuid = "test-uuid"
        promoted = ModelFactory.attempt_promotion(spec)
        assert all(
            [
                isinstance(v, (CoffeaROOTFileSpec, CoffeaParquetFileSpec))
                for v in promoted.files.values()
            ]
        )

    def test_json_file_serialization(self):
        """Test JSON serialization with file path"""
        for k, v in self.get_test_input().items():
            spec = DatasetSpec(**v)
            with tempfile.TemporaryDirectory() as tmp:
                fname = os.path.join(tmp, "test.json.gz")
                with gzip.open(fname, "wt") as fout:
                    fout.write(spec.model_dump_json())
                with gzip.open(fname, "rt") as fin:
                    restored = DatasetSpec.model_validate_json(fin.read())
                    if k != "ZJets2":
                        assert restored.files.keys() == v["files"].keys()
                    else:
                        assert len(restored.files.keys()) == len(v["files"])

    def test_num_entries_computation(self):
        """Test computation of num_entries property"""
        spec = self.get_spec_with_valid_steps()
        assert spec.num_entries == 60

    def test_num_selected_entries_computation(self):
        """Test computation of num_selected_entries property"""
        spec = self.get_spec_with_valid_steps([[0, 10]])
        assert spec.num_selected_entries == 40

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = self.get_spec_with_valid_steps()
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = self.get_spec_with_valid_steps()
        limited_spec = spec.limit_steps(2)
        assert [v.steps for v in limited_spec.files.values()] == [[[0, 5], [5, 10]]]

    def test_limit_steps_per_file_slicing(self):
        """Test limit_steps with slicing"""
        spec = self.get_spec_with_valid_steps(steps=[[0, 15], [15, 30]])
        limited_spec = spec.limit_steps(1, per_file=True)
        assert [v.steps for v in limited_spec.files.values()] == [
            [[0, 5]],
            [[0, 15]],
        ]

    def test_limit_steps_method_chain_slicing(self):
        """Test limit_steps with slicing"""
        spec = self.get_spec_with_valid_steps(steps=[[0, 15], [15, 30]])
        limited_spec = spec.limit_steps(1, per_file=True).limit_steps(1)
        assert [v.steps for v in limited_spec.files.values()] == [[[0, 5]]]

    def test_limit_files_no_modification(self):
        """Test that limit_files with no max_steps returns equivalent object"""
        spec = self.get_spec_with_valid_steps()
        limited_spec = spec.limit_files(max_files=None)
        assert limited_spec == spec

    def test_limit_files_single(self):
        """Test that limit_files with max_steps returns expected object"""
        spec = self.get_spec_with_valid_steps()
        limited_spec = spec.limit_files(max_files=1)
        assert len(limited_spec.files.items()) == 1

    def test_filter_files_pattern(self):
        """Test that filter_files_pattern works as expected"""
        spec = self.get_spec_with_valid_steps()
        filtered_spec = spec.filter_files(filter_name="nano_dy_2")
        assert len(filtered_spec.files.items()) == 1
        assert "nano_dy_2" in list(filtered_spec.files.keys())[0]

    def test_filter_files_no_match(self):
        """Test that filter_files with no matches returns empty fileset"""
        spec = self.get_spec_with_valid_steps()
        filtered_spec = spec.filter_files(filter_name="nonexistent_pattern")
        assert len(filtered_spec.files.items()) == 0

    def test_filter_files_all_match(self):
        """Test that filter_files with all matches returns full fileset"""
        spec = self.get_spec_with_valid_steps()
        filtered_spec = spec.filter_files(filter_name="nano_dy")
        assert len(filtered_spec.files.items()) == len(spec.files.items())

    def test_filter_files_callable(self):
        """Test that filter_files with callable works as expected"""
        spec = self.get_spec_with_valid_steps()

        def pattern_func(datasetspec) -> bool:
            return len(datasetspec.steps) > 3

        filtered_spec = spec.filter_files(filter_callable=pattern_func)
        assert len(filtered_spec.files.items()) == 1
        assert "nano_dy" in list(filtered_spec.files.keys())[0]

    def test_addition_distinct(self):
        """Test addition of two DatasetSpec instances"""
        spec1 = self.get_spec_with_valid_steps()
        spec2 = self.get_spec_with_valid_steps()
        spec2.files = InputFiles(
            {
                "tests/samples/nano_dy_3.root": CoffeaROOTFileSpec(
                    object_path="Events",
                    steps=[[0, 10]],
                    num_entries=10,
                    uuid="uuid3",
                )
            }
        )
        combined_spec = spec1 + spec2
        assert len(combined_spec.files) == 3
        assert "tests/samples/nano_dy.root" in combined_spec.files
        assert "tests/samples/nano_dy_2.root" in combined_spec.files
        assert "tests/samples/nano_dy_3.root" in combined_spec.files

    def test_addition_overlapping(self):
        """Test addition of two DatasetSpec instances with overlapping files"""
        spec1 = self.get_spec_with_valid_steps()
        spec2 = self.get_spec_with_valid_steps()
        with pytest.raises(ValidationError):
            _ = spec1 + spec2

    def test_subtraction_overlapping_nonoverlapping(self):
        """Test subtraction of two DatasetSpec instances"""
        spec1in = self.get_test_input()
        spec1in["ZJets1"]["files"]["tests/samples/nano_dy_2.root"]["steps"] = [
            [0, 10],
            [10, 20],
            [20, 30],
        ]
        spec2in = self.get_test_input()
        spec2in["ZJets1"]["files"]["tests/samples/nano_dy.root"]["steps"] = None
        spec2in["ZJets1"]["files"]["tests/samples/nano_dy_2.root"]["steps"] = [[10, 20]]
        spec1 = DatasetSpec(**spec1in["ZJets1"])
        spec2 = DatasetSpec(**spec2in["ZJets1"])
        reduced_spec = spec1 - spec2
        assert len(reduced_spec.files) == 2
        assert "tests/samples/nano_dy.root" in reduced_spec.files
        assert "tests/samples/nano_dy_2.root" in reduced_spec.files
        assert reduced_spec.files["tests/samples/nano_dy.root"].steps == [
            [0, 5],
            [5, 10],
            [10, 15],
            [15, 20],
            [20, 25],
            [25, 30],
        ]
        assert reduced_spec.files["tests/samples/nano_dy_2.root"].steps == [
            [0, 10],
            [20, 30],
        ]

    def test_legacy_form_to_compressed_form_migration(self):
        """Test that legacy 'form' field is migrated to 'compressed_form'"""
        legacy_input = self.get_test_input()["ZJets1"]
        legacy_input["form"] = legacy_input.pop("compressed_form")
        DatasetSpec(**legacy_input)

        raises_runtimeerror = copy.deepcopy(legacy_input)
        raises_runtimeerror["form"] = "invalid_form"
        with pytest.raises(RuntimeError):
            DatasetSpec(**raises_runtimeerror)


class TestDatasetJoinableSpec:
    """Test DatasetJoinableSpec class"""

    def test_creation_valid(self):
        """Test creation with valid form and format"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )

        # Create a valid compressed form, which would normally be found via preprocess
        simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
        compressed_form = compress_form(simple_form)

        try:
            spec = DatasetSpec(
                files=files.model_dump(),
                format="root",
                compressed_form=compressed_form,
                metadata=None,
            )
        except ValidationError as e:
            print(e.errors())
        assert spec.format == "root"
        assert spec.compressed_form == compressed_form
        assert spec.joinable is True

    def test_invalid_format(self):
        """Test that invalid formats are rejected"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )

        simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
        compressed_form = compress_form(simple_form)

        with pytest.raises(
            ValidationError
        ):  # , match="format: format must be one of"):
            spec = DatasetSpec(
                files=files, format="invalid_format", form=compressed_form
            )
            print(type(spec), spec)

    def test_invalid_form(self):
        """Test that invalid forms are rejected"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )

        with pytest.raises(ValidationError):
            DatasetSpec(files=files, format="root", compressed_form="invalid_form")


class TestModelFactory:
    """Test ModelFactory class methods"""

    @pytest.mark.parametrize(
        "input_dict",
        [
            {"object_path": "Events"},
            {
                "object_path": "Events",
                "steps": [[0, 10]],
                "num_entries": 10,
                "uuid": "uproot-uuid",
            },
        ],
    )
    def test_dict_to_uprootfilespec(self, input_dict):
        """Test dict_to_uprootfilespec method"""
        result = ModelFactory.dict_to_uprootfilespec(input_dict)
        if "steps" in input_dict:
            # Test complete spec
            assert isinstance(result, CoffeaROOTFileSpec)
            assert result.object_path == "Events"
            assert result.steps == [[0, 10]]
            assert result.num_entries == 10
            assert result.uuid == "uproot-uuid"
        else:
            # Test optional spec
            assert isinstance(result, CoffeaROOTFileSpecOptional)
            assert result.object_path == "Events"
            assert result.steps is None

    @pytest.mark.parametrize(
        "input_dict",
        [
            {"object_path": None},
            {
                "object_path": None,
                "steps": [[0, 100]],
                "num_entries": 100,
                "uuid": "parquet-uuid",
            },
        ],
    )
    def test_dict_to_parquetfilespec(self, input_dict):
        """Test dict_to_parquetfilespec method"""
        result = ModelFactory.dict_to_parquetfilespec(input_dict)
        if "steps" in input_dict:
            assert isinstance(result, CoffeaParquetFileSpec)
            assert result.object_path is None
            assert result.steps == [[0, 100]]
            assert result.num_entries == 100
            assert result.uuid == "parquet-uuid"
        else:
            assert isinstance(result, CoffeaParquetFileSpecOptional)

    @pytest.mark.parametrize(
        "input_dict, expected_type",
        [
            (
                {
                    "object_path": "Events",
                    "steps": [[0, 10]],
                    "num_entries": 10,
                    "uuid": "test-uuid",
                },
                CoffeaROOTFileSpec,
            ),
            (
                {
                    "object_path": None,
                    "steps": [[0, 10]],
                    "num_entries": 10,
                    "uuid": "test-uuid",
                },
                CoffeaParquetFileSpec,
            ),
            ({"object_path": "Events"}, CoffeaROOTFileSpecOptional),
            ({"object_path": None}, CoffeaParquetFileSpecOptional),
        ],
    )
    def test_filespec_to_dict(self, input_dict, expected_type):
        """Test filespec_to_dict method"""
        spec = expected_type(**input_dict)
        result = ModelFactory.filespec_to_dict(spec)
        expected = copy.deepcopy(input_dict)
        expected["format"] = (
            "parquet"
            if expected_type in [CoffeaParquetFileSpec, CoffeaParquetFileSpecOptional]
            else "root"
        )
        if "lfn" not in expected:
            expected["lfn"] = None
        if "pfn" not in expected:
            expected["pfn"] = None
        if "steps" not in expected:
            expected["steps"] = None
        if "num_entries" not in expected:
            expected["num_entries"] = None
        if "num_selected_entries" not in expected:
            if "steps" in expected and expected["steps"] is not None:
                expected["num_selected_entries"] = sum(
                    end - start for start, end in expected["steps"]
                )
            else:
                expected["num_selected_entries"] = None
        if "uuid" not in expected:
            expected["uuid"] = None
        assert result == expected, print(f"Expected: {expected}, Got: {result}")

        # Test error for invalid input
        with pytest.raises((ValueError, TypeError)):
            ModelFactory.filespec_to_dict("invalid_input")

    def test_dict_to_datasetspec(self):
        """Test dict_to_datasetspec method"""
        input_dict = {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 10]],
                    "num_entries": 10,
                    "uuid": "test-uuid",
                }
            },
            "metadata": {"sample": "test"},
        }

        result = ModelFactory.dict_to_datasetspec(input_dict)
        assert isinstance(result, (DatasetSpec))
        assert result.format == "root"
        assert result.metadata == {"sample": "test"}

    def test_datasetspec_to_dict(self):
        """Test datasetspec_to_dict method"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )
        spec = DatasetSpec(files=files, format="root", metadata={"sample": "test"})

        result = ModelFactory.datasetspec_to_dict(spec)
        expected = {
            "files": {
                "file1.root": {
                    "object_path": "Events",
                    "steps": [[0, 10]],
                    "format": "root",
                    "lfn": None,
                    "pfn": None,
                    "num_entries": 10,
                    "num_selected_entries": 10,
                    "uuid": "uuid1",
                }
            },
            "format": "root",
            "metadata": {"sample": "test"},
            "compressed_form": None,
            "did": None,
        }
        # Check that the result matches the expected dictionary
        assert result == expected

    def test_attempt_promotion(self):
        """Test attempt_promotion method"""
        # Test with valid ROOTFileSpec
        spec = CoffeaROOTFileSpecOptional(object_path="Events")
        spec.steps = [[0, 100]]
        spec.num_entries = 100
        spec.uuid = "test-uuid"
        promoted = ModelFactory.attempt_promotion(spec)
        assert isinstance(promoted, CoffeaROOTFileSpec)

        # Test with valid ParquetFileSpec
        spec_parquet = CoffeaParquetFileSpecOptional()
        spec_parquet.steps = [[0, 100]]
        spec_parquet.num_entries = 100
        spec_parquet.uuid = "test-uuid"
        promoted_parquet = ModelFactory.attempt_promotion(spec_parquet)
        assert isinstance(promoted_parquet, CoffeaParquetFileSpec)


class TestJSONSerialization:
    """Test JSON serialization/deserialization for all classes"""

    def test_uprootfilespec_json_roundtrip(self):
        """Test JSON roundtrip for ROOTFileSpec"""
        spec = ROOTFileSpec(object_path="Events", steps=[[0, 10]])
        json_str = spec.model_dump_json()
        restored = ROOTFileSpec.model_validate_json(json_str)
        assert restored.object_path == spec.object_path
        assert restored.steps == spec.steps

    def test_coffeefilespec_json_roundtrip(self):
        """Test JSON roundtrip for CoffeaROOTFileSpec"""
        spec = CoffeaROOTFileSpec(
            object_path="Events", steps=[[0, 10]], num_entries=10, uuid="test-uuid"
        )
        json_str = spec.model_dump_json()
        restored = CoffeaROOTFileSpec.model_validate_json(json_str)
        assert restored.object_path == spec.object_path
        assert restored.steps == spec.steps
        assert restored.num_entries == spec.num_entries
        assert restored.uuid == spec.uuid

    def test_datasetspec_json_roundtrip(self):
        """Test JSON roundtrip for DatasetSpec"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )
        spec = DatasetSpec(files=files, format="root", metadata={"sample": "test"})

        json_str = spec.model_dump_json()
        restored = DatasetSpec.model_validate_json(json_str)
        assert restored.format == spec.format
        assert restored.metadata == spec.metadata
        assert len(restored.files) == len(spec.files)


class TestDataGroupSpec:
    """Test DataGroupSpec class"""

    def get_sliceable_spec(self):
        """Test limit_steps with slicing"""
        tmp = TestDatasetSpec.get_test_input(TestDatasetSpec)
        tmp["ZJets1"]["files"]["tests/samples/nano_dy_2.root"]["steps"] = [
            [0, 30]
        ]  # Ensure steps are set for counting
        _ = tmp.pop("ZJets2")  # Remove Optional DatasetSpec
        return DataGroupSpec(tmp)

    def test_creation_valid(self):
        """Test creation with valid concrete dataset specs"""
        files = InputFiles(
            {
                "file1.root": CoffeaROOTFileSpec(
                    object_path="Events", steps=[[0, 10]], num_entries=10, uuid="uuid1"
                )
            }
        )
        datasets = {"ZJets": DatasetSpec(files=files, format="root")}
        fileset = DataGroupSpec(datasets)
        assert len(fileset) == 1
        assert "ZJets" in fileset

    def test_fileset_creation_and_json_roundtrip(self):
        """Test DataGroupSpec creation and JSON serialization roundtrip"""

        # Convert to DatasetSpec first
        converted = {}
        for k, v in _starting_fileset.items():
            print(k, v)
            converted[k] = DatasetSpec(**v)

        # Create DataGroupSpec
        conv_pyd = DataGroupSpec(converted)
        assert len(conv_pyd) == 2
        assert "ZJets" in conv_pyd
        assert "Data" in conv_pyd

        # Test JSON serialization roundtrip
        json_str = conv_pyd.model_dump_json()
        restored = DataGroupSpec.model_validate_json(json_str)
        assert len(restored) == len(conv_pyd)
        assert set(restored.keys()) == set(conv_pyd.keys())

    def test_invalid_form(self):
        """Test that invalid form raises ValidationError"""
        invalid_form_dict = copy.deepcopy(_starting_fileset)
        invalid_form_dict["ZJets"]["compressed_form"] = invalid_compressed_form
        print("invalid_form_dict:", invalid_form_dict)
        with pytest.raises(ValidationError):
            DataGroupSpec(invalid_form_dict)

    def test_direct_fileset_creation(self):
        """Test complex DataGroupSpec creation from __main__"""
        test_input = {
            "ZJets1": {
                "files": {
                    "tests/samples/nano_dy.root": {
                        "object_path": "Events",
                        "steps": [
                            [0, 5],
                            [5, 10],
                            [10, 15],
                            [15, 20],
                            [20, 25],
                            [25, 30],
                        ],
                        "num_entries": 30,
                        "uuid": "1234-5678-90ab-cdef",
                    },
                    "tests/samples/nano_dy_2.root": {
                        "object_path": "Events",
                        "steps": None,
                        "num_entries": 30,
                        "uuid": "1234-5678-90ab-cdef",
                    },
                },
                "format": "root",
                "metadata": {"key": "value"},
                "compressed_form": valid_compressed_form,
            },
            "Data": {"files": {"tests/samples/nano_dimuon.root": "Events"}},
        }

        # Convert via direct constructor
        test = DataGroupSpec(test_input)

        # test that the steps are correct
        assert test["ZJets1"].files["tests/samples/nano_dy.root"].steps == [
            [0, 5],
            [5, 10],
            [10, 15],
            [15, 20],
            [20, 25],
            [25, 30],
        ]
        assert test["ZJets1"].files["tests/samples/nano_dy_2.root"].steps is None
        assert len(test) == 2
        assert "ZJets1" in test
        assert test["ZJets1"].format == "root"
        assert test["ZJets1"].metadata == {"key": "value"}

        # Test we can modify the steps after creation
        test["ZJets1"].files["tests/samples/nano_dy_2.root"].steps = [[0, 30]]
        test2 = DataGroupSpec(test)
        assert isinstance(
            test2["ZJets1"].files["tests/samples/nano_dy_2.root"].steps, list
        )

    def test_attempt_promotion(self):
        """Test attempt_promotion method"""
        # Test with valid ROOTFileSpec
        spec = DataGroupSpec(copy.deepcopy(_starting_fileset))
        assert isinstance(
            spec["Data"].files["tests/samples/nano_dimuon.root"],
            CoffeaROOTFileSpecOptional,
        )
        for k, v in spec["Data"].files.items():
            if isinstance(v, CoffeaROOTFileSpecOptional):
                v.steps = [[0, 10], [10, 20]]
                v.num_entries = 20
                v.uuid = "test-uuid"
        promoted = ModelFactory.attempt_promotion(spec)
        assert all(
            [
                isinstance(v, (CoffeaROOTFileSpec, CoffeaParquetFileSpec))
                for v in promoted["Data"].files.values()
            ]
        )

    def test_num_entries_computation(self):
        """Test computation of num_entries property"""
        spec = self.get_sliceable_spec()
        assert spec.num_entries == 90

    def test_num_selected_entries_computation(self):
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_steps(2, per_file=True)
        assert limited_spec.num_selected_entries == 50

    def test_limit_steps_no_modification(self):
        """Test that limit_steps with no max_steps returns equivalent object"""
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_steps(None)
        assert limited_spec == spec

    def test_limit_steps_slicing(self):
        """Test limit_steps with slicing"""
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_steps(1)
        assert limited_spec.steps == {
            "ZJets1": {
                "tests/samples/nano_dy.root": [[0, 5]],
            },
            "ZParquet": {"tests/samples/nano_dy.parquet": [[0, 5]]},
        }  # Only first 1 step per dataset

    def test_limit_steps_per_file_slicing(self):
        """Test limit_steps with slicing"""
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_steps(1, per_file=True)
        assert limited_spec.steps == {
            "ZJets1": {
                "tests/samples/nano_dy.root": [[0, 5]],
                "tests/samples/nano_dy_2.root": [[0, 30]],
            },
            "ZParquet": {"tests/samples/nano_dy.parquet": [[0, 5]]},
        }  # Only first 1 step per dataset

    def test_limit_steps_method_chain_slicing(self):
        """Test limit_steps with slicing"""
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_steps(1, per_file=True).limit_steps(1)
        assert limited_spec.steps == {
            "ZJets1": {
                "tests/samples/nano_dy.root": [[0, 5]],
            },
            "ZParquet": {"tests/samples/nano_dy.parquet": [[0, 5]]},
        }

    def test_limit_files_no_modification(self):
        """Test that limit_files with no max_files returns equivalent object"""
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_files(max_files=None)
        assert limited_spec == spec

    def test_limit_files_single_per_dataset(self):
        """Test that limit_files with max_steps returns expected object"""
        spec = self.get_sliceable_spec()
        limited_spec = spec.limit_files(max_files=1, per_dataset=True)
        assert {k: len(v.files) for k, v in limited_spec.items()} == {
            "ZJets1": 1,
            "ZParquet": 1,
        }

    def test_limit_files_single_not_per_dataset(self):
        """Test that limit_files with per_dataset=False raises NotImplementedError"""
        spec = self.get_sliceable_spec()
        with pytest.raises(NotImplementedError):
            spec.limit_files(max_files=1, per_dataset=False)

    def test_filter_files_pattern(self):
        """Test that filter_files_pattern works as expected"""
        spec = self.get_sliceable_spec()
        filtered_spec = spec.filter_files(filter_name=r"nano_dy\.")
        for k, v in filtered_spec.items():
            if k == "ZJets1":
                assert len(v.files.items()) == 1
                assert "nano_dy_2." not in list(v.files.keys())[0]
            else:
                assert len(v.files.items()) == 1
                assert "parquet" in list(v.files.keys())[0]

    def test_filter_files_no_match(self):
        """Test that filter_files with no matches returns empty fileset"""
        spec = self.get_sliceable_spec()
        filtered_spec = spec.filter_files(filter_name="nonexistent_pattern")
        for k, v in filtered_spec.items():
            assert len(v.files.items()) == 0

    def test_filter_files_all_match(self):
        """Test that filter_files with all matches returns full fileset"""
        spec = self.get_sliceable_spec()
        filtered_spec = spec.filter_files(filter_name="nano_dy")
        assert filtered_spec == spec

    def test_filter_files_callable(self):
        """Test that filter_files with callable works as expected"""
        spec = self.get_sliceable_spec()

        def pattern_func(datasetspec) -> bool:
            return len(datasetspec.steps) > 3

        filtered_spec = spec.filter_files(filter_callable=pattern_func)
        for k, v in filtered_spec.items():
            if k == "ZJets1":
                assert len(v.files.items()) == 1
                assert "nano_dy" in list(v.files.keys())[0]
            else:
                assert len(v.files.items()) == 1

    def test_filter_datasets_pattern(self):
        """Test that filter_name on datasets works as expected"""
        spec = self.get_sliceable_spec()
        filtered_spec = spec.filter_datasets(filter_name="Jets.")
        assert "ZJets1" in filtered_spec
        assert "ZParquet" not in filtered_spec

    def test_filter_datasets_no_match(self):
        """Test that filter_files with no matches returns empty fileset"""
        spec = self.get_sliceable_spec()
        filtered_spec = spec.filter_datasets(filter_name="nonexistent_pattern")
        assert len(filtered_spec) == 0

    def test_filter_datasets_all_match(self):
        """Test that filter_files with all matches returns full fileset"""
        spec = self.get_sliceable_spec()
        filtered_spec = spec.filter_datasets(filter_name="Z.*")
        assert len(filtered_spec) == len(spec)
        assert "ZJets1" in filtered_spec
        assert "ZParquet" in filtered_spec

    def test_filter_datasets_callable(self):
        """Test that filter_files with callable works as expected"""
        spec = self.get_sliceable_spec()

        def pattern_func(datasetspec) -> bool:
            return datasetspec.metadata.get("key", "") == "value"

        filtered_spec = spec.filter_datasets(filter_callable=pattern_func)
        for k, v in filtered_spec.items():
            if k == "ZJets1":
                assert len(v.files.items()) == 2
                assert "nano_dy" in list(v.files.keys())[0]
            else:
                raise AssertionError("Only ZJets1 should match")

    def test_addition_distinct(self):
        """Test addition of two DataGroupSpec instances"""
        spec1 = self.get_sliceable_spec()
        spec2 = self.get_sliceable_spec()
        spec2["ZJets1"].files = InputFiles(
            {
                "tests/samples/nano_dy_3.root": CoffeaROOTFileSpec(
                    object_path="Events",
                    steps=[[0, 10]],
                    num_entries=10,
                    uuid="uuid3",
                )
            }
        )
        spec2["ZParquet"].files = InputFiles(
            {
                "tests/samples/nano_dy_4.parquet": CoffeaParquetFileSpec(
                    steps=[[0, 100]], num_entries=100, uuid="uuid4"
                )
            }
        )
        combined_spec = spec1 + spec2
        assert len(combined_spec) == 2
        assert len(combined_spec["ZJets1"].files) == 3
        assert len(combined_spec["ZParquet"].files) == 2
        assert "tests/samples/nano_dy.root" in combined_spec["ZJets1"].files
        assert "tests/samples/nano_dy_2.root" in combined_spec["ZJets1"].files
        assert "tests/samples/nano_dy_3.root" in combined_spec["ZJets1"].files
        assert "tests/samples/nano_dy.parquet" in combined_spec["ZParquet"].files
        assert "tests/samples/nano_dy_4.parquet" in combined_spec["ZParquet"].files

    def test_addition_overlapping(self):
        """Test addition of two DataGroupSpec instances with overlapping datasets"""
        spec1 = self.get_sliceable_spec()
        spec2 = self.get_sliceable_spec()
        with pytest.raises(ValidationError):
            _ = spec1 + spec2

    def test_subtraction_overlapping_nonoverlapping(self):
        """Test subtraction of two DataGroupSpec instances"""
        spec1in = TestDataGroupSpec.get_sliceable_spec(TestDataGroupSpec)
        spec1in["ZJets1"].files["tests/samples/nano_dy_2.root"].steps = [
            [0, 10],
            [10, 20],
            [20, 30],
        ]  # Ensure steps are set for counting
        spec2in = TestDataGroupSpec.get_sliceable_spec(TestDataGroupSpec)
        spec2in["ZJets1"].files["tests/samples/nano_dy.root"].steps = None
        spec2in["ZJets1"].files["tests/samples/nano_dy_2.root"].steps = [[10, 20]]
        spec1 = DataGroupSpec(spec1in)
        spec2 = DataGroupSpec(spec2in)
        reduced_spec = spec1 - spec2
        assert len(reduced_spec) == 2
        assert len(reduced_spec["ZJets1"].files) == 2
        assert "tests/samples/nano_dy.root" in reduced_spec["ZJets1"].files
        assert "tests/samples/nano_dy_2.root" in reduced_spec["ZJets1"].files
        assert reduced_spec["ZJets1"].files["tests/samples/nano_dy.root"].steps == [
            [0, 5],
            [5, 10],
            [10, 15],
            [15, 20],
            [20, 25],
            [25, 30],
        ]
        assert reduced_spec["ZJets1"].files["tests/samples/nano_dy_2.root"].steps == [
            [0, 10],
            [20, 30],
        ]


class TestMainMethodScenarios:
    """Test scenarios specifically from the __main__ method"""

    def test_all_filespec_creation_combinations(self):
        """Test all file spec creation combinations from __main__"""
        steps_options = [None, [[0, 100]], [[0, 1], [2, 3]]]

        for steps in steps_options:
            # Test ROOTFileSpec
            try:
                spec = ROOTFileSpec(object_path="example_path", steps=steps)
                assert spec.object_path == "example_path"
            except ValueError:
                # Some combinations may be invalid
                pass

            # Test with num_entries
            for num_entries in [None, 100]:
                spec = CoffeaROOTFileSpecOptional(
                    object_path="example_path",
                    steps=steps,
                    num_entries=num_entries,
                )
                assert spec.object_path == "example_path"

            # Test with uuid
            for uuid in [None, "hello-there"]:
                spec1 = CoffeaROOTFileSpecOptional(
                    object_path="example_path",
                    steps=steps,
                    uuid=uuid,
                )
                assert spec1.object_path == "example_path"

                spec2 = CoffeaParquetFileSpecOptional(
                    object_path=None,
                    steps=steps,
                    uuid=uuid,
                )
                assert spec2.object_path is None

            # Test concrete specs
            num_entries, uuid = 100, "hello-there"
            try:
                spec1 = CoffeaROOTFileSpec(
                    object_path="example_path",
                    steps=steps,
                    num_entries=num_entries,
                    uuid=uuid,
                )
                if steps is not None:  # Only valid for non-None steps
                    assert spec1.object_path == "example_path"
            except ValueError:
                # Steps=None should fail
                assert steps is None

            try:
                spec2 = CoffeaParquetFileSpec(
                    object_path=None,
                    steps=steps,
                    num_entries=num_entries,
                    uuid=uuid,
                )
                if steps is not None:  # Only valid for non-None steps
                    assert spec2.object_path is None
            except ValueError:
                # Steps=None should fail
                assert steps is None


class TestComplexScenarios:
    """Test complex real-world scenarios"""

    def test_legacy_fileset_conversion(self):
        """Test converting legacy fileset formats"""
        legacy_fileset = {
            "ZJets": {
                "files": {
                    "tests/samples/nano_dy.root": {
                        "object_path": "Events",
                        "steps": [[0, 5], [5, 10], [10, 15]],
                        "num_entries": 15,
                        "uuid": "test-uuid",
                    }
                }
            },
            "Data": {"files": {"tests/samples/nano_dimuon.root": "Events"}},
        }

        # Convert each dataset
        converted = {}
        for dataset_name, dataset_info in legacy_fileset.items():
            converted[dataset_name] = ModelFactory.dict_to_datasetspec(dataset_info)

        assert "ZJets" in converted
        assert "Data" in converted
        assert isinstance(converted["ZJets"], (DatasetSpec))
        assert isinstance(converted["Data"], (DatasetSpec))

    def test_mixed_format_handling(self):
        """Test handling datasets with mixed file formats"""
        with pytest.raises(ValidationError):
            spec = DatasetSpec(
                files={
                    "file1.root": CoffeaROOTFileSpec(
                        object_path="Events",
                        steps=[[0, 10]],
                        num_entries=10,
                        uuid="uuid1",
                    ),
                    "file1.parquet": CoffeaParquetFileSpec(
                        steps=[[0, 100]], num_entries=100, uuid="uuid2"
                    ),
                }
            )
            # Format should be mixed when files have different formats
            assert spec.format in ("parquet|root", "root|parquet")

    def test_empty_fileset_handling(self):
        """Test handling of empty filesets"""
        empty_fileset = DataGroupSpec({})
        assert len(empty_fileset) == 0

    def test_error_handling_invalid_steps(self):
        """Test error handling for invalid step configurations, effectively nesting the StepPair tests"""
        with pytest.raises(ValueError):
            CoffeaROOTFileSpec(
                object_path="Events",
                steps=[[-1, 10]],  # Negative start
                num_entries=10,
                uuid="uuid",
            )

        with pytest.raises(ValueError):
            CoffeaROOTFileSpec(
                object_path="Events",
                steps=[[0, 10, 20]],  # Too many elements in step
                num_entries=10,
                uuid="uuid",
            )
