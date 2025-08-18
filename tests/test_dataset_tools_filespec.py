"""Tests for the Pydantic-based IOFactory classes in iofactory.py"""

import pytest

from coffea.dataset_tools.filespec import (
    UprootFileSpec,
    ParquetFileSpec,
    CoffeaUprootFileSpec,
    CoffeaUprootFileSpecOptional,
    CoffeaParquetFileSpec,
    CoffeaParquetFileSpecOptional,
    CoffeaFileDict,
    DatasetSpec,
    FilesetSpec,
    IOFactory,
)
from coffea.util import compress_form
from pydantic import ValidationError
import awkward


class TestStepPair:
    """Test the StepPair type annotation"""
    
    def test_valid_step_pair(self):
        """Test that valid step pairs are accepted"""
        spec = UprootFileSpec(object_path="Events", steps=[[0, 10]])
        assert spec.steps == [[0, 10]]
    
    def test_invalid_step_pair_negative(self):
        """Test that negative values in step pairs are rejected"""
        with pytest.raises(ValueError):
            UprootFileSpec(object_path="Events", steps=[[-1, 10]])
    
    def test_invalid_step_pair_length(self):
        """Test that step pairs with wrong length are rejected"""
        with pytest.raises(ValueError):
            UprootFileSpec(object_path="Events", steps=[[0, 10, 20]])
        
        with pytest.raises(ValueError):
            UprootFileSpec(object_path="Events", steps=[[0]])


class TestUprootFileSpec:
    """Test UprootFileSpec class"""
    
    def test_creation_basic(self):
        """Test basic creation of UprootFileSpec"""
        spec = UprootFileSpec(object_path="Events")
        assert spec.object_path == "Events"
        assert spec.steps is None
    
    def test_creation_with_steps(self):
        """Test creation with steps"""
        spec = UprootFileSpec(object_path="Events", steps=[[0, 10], [10, 20]])
        assert spec.object_path == "Events"
        assert spec.steps == [[0, 10], [10, 20]]
    
    def test_creation_with_single_step(self):
        """Test creation with single step pair"""
        spec = UprootFileSpec(object_path="Events", steps=[0, 10])
        assert spec.object_path == "Events"
        assert spec.steps == [0, 10]
    
    def test_json_serialization(self):
        """Test JSON serialization/deserialization"""
        spec = UprootFileSpec(object_path="Events", steps=[[0, 10]])
        json_str = spec.model_dump_json()
        restored = UprootFileSpec.model_validate_json(json_str)
        assert restored.object_path == spec.object_path
        assert restored.steps == spec.steps


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


class TestCoffeaUprootFileSpecOptional:
    """Test CoffeaUprootFileSpecOptional class"""
    
    def test_creation_minimal(self):
        """Test creation with minimal required fields"""
        spec = CoffeaUprootFileSpecOptional(object_path="Events")
        assert spec.object_path == "Events"
        assert spec.steps is None
        assert spec.num_entries is None
        assert spec.uuid is None
    
    def test_creation_complete(self):
        """Test creation with all fields"""
        spec = CoffeaUprootFileSpecOptional(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=100,
            uuid="test-uuid"
        )
        assert spec.object_path == "Events"
        assert spec.steps == [[0, 10]]
        assert spec.num_entries == 100
        assert spec.uuid == "test-uuid"
    
    def test_negative_num_entries_rejected(self):
        """Test that negative num_entries are rejected"""
        with pytest.raises(ValueError):
            CoffeaUprootFileSpecOptional(object_path="Events", num_entries=-1)


class TestCoffeaUprootFileSpec:
    """Test CoffeaUprootFileSpec class"""
    
    def test_creation_complete(self):
        """Test creation with all required fields"""
        spec = CoffeaUprootFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=100,
            uuid="test-uuid"
        )
        assert spec.object_path == "Events"
        assert spec.steps == [[0, 10]]
        assert spec.num_entries == 100
        assert spec.uuid == "test-uuid"
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise errors"""
        with pytest.raises(ValueError):
            CoffeaUprootFileSpec(object_path="Events")  # Missing steps, num_entries, uuid
    
    def test_steps_required(self):
        """Test that steps are required (not None)"""
        with pytest.raises(ValueError):
            CoffeaUprootFileSpec(
                object_path="Events",
                steps=None,
                num_entries=100,
                uuid="test-uuid"
            )


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
            object_path=None,
            steps=[[0, 100]],
            num_entries=100,
            uuid="parquet-uuid"
        )
        assert spec.object_path is None
        assert spec.steps == [[0, 100]]
        assert spec.num_entries == 100
        assert spec.uuid == "parquet-uuid"


class TestCoffeaParquetFileSpec:
    """Test CoffeaParquetFileSpec class"""
    
    def test_creation_complete(self):
        """Test creation with all required fields"""
        spec = CoffeaParquetFileSpec(
            steps=[[0, 100]],
            num_entries=100,
            uuid="parquet-uuid"
        )
        assert spec.object_path is None
        assert spec.steps == [[0, 100]]
        assert spec.num_entries == 100
        assert spec.uuid == "parquet-uuid"
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise errors"""
        with pytest.raises(ValueError):
            CoffeaParquetFileSpec()  # Missing steps, num_entries, uuid


class TestDictMethodsMixin:
    """Test the DictMethodsMixin functionality through CoffeaFileDict"""
    
    def test_dict_methods(self):
        """Test that dict methods work properly"""
        files = {
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            ),
            "file2.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 20]],
                num_entries=20,
                uuid="uuid2"
            )
        }
        file_dict = CoffeaFileDict(files)
        
        # Test __getitem__
        assert file_dict["file1.root"].uuid == "uuid1"
        
        # Test __len__
        assert len(file_dict) == 2
        
        # Test __iter__
        keys = list(file_dict)
        assert "file1.root" in keys
        assert "file2.root" in keys
        
        # Test keys()
        assert set(file_dict.keys()) == {"file1.root", "file2.root"}
        
        # Test values()
        values = list(file_dict.values())
        assert len(values) == 2
        
        # Test items()
        items = list(file_dict.items())
        assert len(items) == 2
        
        # Test get()
        assert file_dict.get("file1.root").uuid == "uuid1"
        assert file_dict.get("nonexistent") is None
        assert file_dict.get("nonexistent", "default") == "default"


class TestCoffeaFileDict:
    """Test CoffeaFileDict class"""
    
    def test_creation_valid(self):
        """Test creation with valid CoffeaUprootFileSpec instances"""
        files = {
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        }
        file_dict = CoffeaFileDict(files)
        assert len(file_dict) == 1
        assert file_dict["file1.root"].uuid == "uuid1"
    
    def test_dict_file_format(self):
        """Test that invalid file types are rejected"""
        files = {
            "file1.txt": CoffeaUprootFileSpecOptional(object_path="Events") 
        }
        #with pytest.raises(ValidationError):
        fdict = CoffeaFileDict(files)
        with pytest.raises(RuntimeError, match="IOFactory couldn't identify"):
            print(fdict.format)
    
    def test_mixed_root_and_parquet(self):
        """Test creation with mixed root and parquet files"""
        files = {
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            ),
            "file1.parquet": CoffeaParquetFileSpec(
                steps=[[0, 100]],
                num_entries=100,
                uuid="uuid2"
            )
        }
        file_dict = CoffeaFileDict(files)
        assert len(file_dict) == 2


class TestDatasetSpec:
    """Test DatasetSpec class"""
    
    def test_creation_valid(self):
        """Test creation with valid concrete file specs"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        spec = DatasetSpec(
            files=files,
            format="root",
            metadata={"sample": "test"}
        )
        assert spec.format == "root"
        assert spec.metadata == {"sample": "test"}


class TestDatasetJoinableSpec:
    """Test DatasetJoinableSpec class"""
    
    def test_creation_valid(self):
        """Test creation with valid form and format"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        
        # Create a valid compressed form
        simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
        compressed_form = compress_form(simple_form)
        
        try:
            spec = DatasetSpec(
                files=files.model_dump(),
                format="root",
                form=compressed_form,
                metadata=None
            )
        except ValidationError as e:
            print(e.errors())
        assert spec.format == "root"
        assert spec.form == compressed_form
    
    def test_invalid_format(self):
        """Test that invalid formats are rejected"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        
        simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
        compressed_form = compress_form(simple_form)
        
        with pytest.raises(ValidationError):#, match="format: format must be one of"):
            spec = DatasetSpec(
                files=files,
                format="invalid_format",
                form=compressed_form
            )
            print(type(spec), spec)
    
    def test_invalid_form(self):
        """Test that invalid forms are rejected"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        
        with pytest.raises(ValidationError):
            DatasetSpec(
                files=files,
                format="root",
                form="invalid_form"
            )

class TestFilesetSpec:
    """Test FilesetSpec class"""
    
    def test_creation_valid(self):
        """Test creation with valid concrete dataset specs"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        datasets = {
            "ZJets": DatasetSpec(files=files, format="root")
        }
        fileset = FilesetSpec(datasets)
        assert len(fileset) == 1
        assert "ZJets" in fileset


class TestIOFactory:
    """Test IOFactory class methods"""
    
    def test_valid_format(self):
        """Test valid_format method"""
        assert IOFactory.valid_format("root") is True
        assert IOFactory.valid_format("parquet") is True
        assert IOFactory.valid_format("invalid") is False
    
    def test_identify_format(self):
        """Test identify_format method"""
        assert IOFactory.identify_format("file.root") == "root"
        assert IOFactory.identify_format("file.parquet") == "parquet"
        assert IOFactory.identify_format("file.parq") == "parquet"
        assert IOFactory.identify_format("directory") == "parquet"
        
        with pytest.raises(RuntimeError):
            IOFactory.identify_format("file.txt")
    
    def test_dict_to_uprootfilespec(self):
        """Test dict_to_uprootfilespec method"""
        # Test complete spec
        input_dict = {
            "object_path": "Events",
            "steps": [[0, 10]],
            "num_entries": 10,
            "uuid": "test-uuid"
        }
        result = IOFactory.dict_to_uprootfilespec(input_dict)
        assert isinstance(result, CoffeaUprootFileSpec)
        assert result.object_path == "Events"
        assert result.steps == [[0, 10]]
        assert result.num_entries == 10
        assert result.uuid == "test-uuid"
        
        # Test optional spec
        input_dict_optional = {
            "object_path": "Events"
        }
        result_optional = IOFactory.dict_to_uprootfilespec(input_dict_optional)
        assert isinstance(result_optional, CoffeaUprootFileSpecOptional)
        assert result_optional.object_path == "Events"
        assert result_optional.steps is None
    
    def test_dict_to_parquetfilespec(self):
        """Test dict_to_parquetfilespec method"""
        input_dict = {
            "object_path": None,
            "steps": [[0, 100]],
            "num_entries": 100,
            "uuid": "parquet-uuid"
        }
        result = IOFactory.dict_to_parquetfilespec(input_dict)
        assert isinstance(result, CoffeaParquetFileSpec)
        assert result.object_path is None
        assert result.steps == [[0, 100]]
        assert result.num_entries == 100
        assert result.uuid == "parquet-uuid"
    
    def test_filespec_to_dict(self):
        """Test filespec_to_dict method"""
        spec = CoffeaUprootFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=10,
            uuid="test-uuid"
        )
        result = IOFactory.filespec_to_dict(spec)
        expected = {
            "object_path": "Events",
            "steps": [[0, 10]],
            "num_entries": 10,
            "uuid": "test-uuid"
        }
        assert result == expected
        
        # Test error for invalid input
        with pytest.raises((ValueError, TypeError)):
            IOFactory.filespec_to_dict("invalid_input")
    
    def test_dict_to_datasetspec(self):
        """Test dict_to_datasetspec method"""
        input_dict = {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 10]],
                    "num_entries": 10,
                    "uuid": "test-uuid"
                }
            },
            "metadata": {"sample": "test"}
        }
        
        result = IOFactory.dict_to_datasetspec(input_dict)
        assert isinstance(result, (DatasetSpec))
        assert result.format == "root"
        assert result.metadata == {"sample": "test"}
    
    def test_datasetspec_to_dict(self):
        """Test datasetspec_to_dict method"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        spec = DatasetSpec(
            files=files,
            format="root",
            metadata={"sample": "test"}
        )
        
        result = IOFactory.datasetspec_to_dict(spec)
        expected = {
            "files": {
                "file1.root": {
                    "object_path": "Events",
                    "steps": [[0, 10]],
                    "num_entries": 10,
                    "uuid": "uuid1"
                }
            },
            "format": "root",
            "metadata": {"sample": "test"},
            "form": None
        }
        assert result == expected


class TestJSONSerialization:
    """Test JSON serialization/deserialization for all classes"""
    
    def test_uprootfilespec_json_roundtrip(self):
        """Test JSON roundtrip for UprootFileSpec"""
        spec = UprootFileSpec(object_path="Events", steps=[[0, 10]])
        json_str = spec.model_dump_json()
        restored = UprootFileSpec.model_validate_json(json_str)
        assert restored.object_path == spec.object_path
        assert restored.steps == spec.steps
    
    def test_coffeefilespec_json_roundtrip(self):
        """Test JSON roundtrip for CoffeaUprootFileSpec"""
        spec = CoffeaUprootFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=10,
            uuid="test-uuid"
        )
        json_str = spec.model_dump_json()
        restored = CoffeaUprootFileSpec.model_validate_json(json_str)
        assert restored.object_path == spec.object_path
        assert restored.steps == spec.steps
        assert restored.num_entries == spec.num_entries
        assert restored.uuid == spec.uuid
    
    def test_datasetspec_json_roundtrip(self):
        """Test JSON roundtrip for DatasetSpec"""
        files = CoffeaFileDict({
            "file1.root": CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10]],
                num_entries=10,
                uuid="uuid1"
            )
        })
        spec = DatasetSpec(
            files=files,
            format="root",
            metadata={"sample": "test"}
        )
        
        json_str = spec.model_dump_json()
        restored = DatasetSpec.model_validate_json(json_str)
        assert restored.format == spec.format
        assert restored.metadata == spec.metadata
        assert len(restored.files) == len(spec.files)


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
                        "uuid": "test-uuid"
                    }
                }
            },
            "Data": {
                "files": {
                    "tests/samples/nano_dimuon.root": "Events"
                }
            }
        }
        
        # Convert each dataset
        converted = {}
        for dataset_name, dataset_info in legacy_fileset.items():
            converted[dataset_name] = IOFactory.dict_to_datasetspec(dataset_info)
        
        assert "ZJets" in converted
        assert "Data" in converted
        assert isinstance(converted["ZJets"], (DatasetSpec))
        assert isinstance(converted["Data"], (DatasetSpec))
    
    def test_mixed_format_handling(self):
        """Test handling datasets with mixed file formats"""
        spec = DatasetSpec(
            files={
                "file1.root": CoffeaUprootFileSpec(
                    object_path="Events",
                    steps=[[0, 10]],
                    num_entries=10,
                    uuid="uuid1"
                ),
                "file1.parquet": CoffeaParquetFileSpec(
                    steps=[[0, 100]],
                    num_entries=100,
                    uuid="uuid2"
                )
            }
        )
        # Format should be mixed when files have different formats
        assert spec.format in ("parquet|root", "root|parquet")
    
    def test_empty_fileset_handling(self):
        """Test handling of empty filesets"""
        empty_fileset = FilesetSpec({})
        assert len(empty_fileset) == 0
    
    def test_error_handling_invalid_steps(self):
        """Test error handling for invalid step configurations"""
        with pytest.raises(ValueError):
            CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[-1, 10]],  # Negative start
                num_entries=10,
                uuid="uuid"
            )
        
        with pytest.raises(ValueError):
            CoffeaUprootFileSpec(
                object_path="Events",
                steps=[[0, 10, 20]],  # Too many elements in step
                num_entries=10,
                uuid="uuid"
            )
