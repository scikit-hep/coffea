import pytest

from coffea.dataset_tools.preprocess import (
    CoffeaFileSpec,
    CoffeaFileSpecOptional,
    CoffeaParquetFileSpec,
    CoffeaParquetFileSpecOptional,
    DatasetSpec,
    DatasetSpecOptional,
    DatasetJoinableSpec,
    IOFactory,
    ParquetFileSpec,
    UprootFileSpec,
)


_starting_fileset_dict = {
    "ZJets": {"tests/samples/nano_dy.root": "Events"},
    "Data": {
        "tests/samples/nano_dimuon.root": "Events",
        "tests/samples/nano_dimuon_not_there.root": "Events",
    },
}

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

_runnable_result = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        },
        "metadata": None,
        "form": None,
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
            }
        },
        "metadata": None,
        "form": None,
    },
}

_updated_result = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        },
        "metadata": None,
        "form": None,
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
            },
            "tests/samples/nano_dimuon_not_there.root": {
                "object_path": "Events",
                "steps": None,
                "num_entries": None,
                "uuid": None,
            },
        },
        "metadata": None,
        "form": None,
    },
}

# Tests for dataclasses and conversions

def test_uprootfilespec_creation():
    """Test UprootFileSpec dataclass creation"""
    spec = UprootFileSpec(
        object_path="Events",
        steps=[[0, 10], [10, 20]]
    )
    assert spec.object_path == "Events"
    assert spec.steps == [[0, 10], [10, 20]]


def test_parquetfilespec_creation():
    """Test ParquetFileSpec dataclass creation"""
    spec = ParquetFileSpec(
        object_path=None,
        steps=[[0, 10], [10, 20]]
    )
    assert spec.object_path is None
    assert spec.steps == [[0, 10], [10, 20]]


def test_coffeefilespec_creation():
    """Test CoffeaFileSpec dataclass creation"""
    spec = CoffeaFileSpec(
        object_path="Events",
        steps=[[0, 10], [10, 20]],
        num_entries=20,
        uuid="test-uuid"
    )
    assert spec.object_path == "Events"
    assert spec.steps == [[0, 10], [10, 20]]
    assert spec.num_entries == 20
    assert spec.uuid == "test-uuid"


def test_coffeefilespec_optional_creation():
    """Test CoffeaFileSpecOptional dataclass creation"""
    spec = CoffeaFileSpecOptional(
        object_path="Events",
        steps=None,
        num_entries=None,
        uuid=None
    )
    assert spec.object_path == "Events"
    assert spec.steps is None
    assert spec.num_entries is None
    assert spec.uuid is None


def test_coffea_parquet_filespec_creation():
    """Test CoffeaParquetFileSpec dataclass creation"""
    spec = CoffeaParquetFileSpec(
        object_path=None,
        steps=[[0, 100]],
        num_entries=100,
        uuid="parquet-uuid"
    )
    assert spec.object_path is None
    assert spec.steps == [[0, 100]]
    assert spec.num_entries == 100
    assert spec.uuid == "parquet-uuid"


def test_coffea_parquet_filespec_optional_creation():
    """Test CoffeaParquetFileSpecOptional dataclass creation"""
    spec = CoffeaParquetFileSpecOptional(
        object_path=None,
        steps=None,
        num_entries=None,
        uuid=None
    )
    assert spec.object_path is None
    assert spec.steps is None
    assert spec.num_entries is None
    assert spec.uuid is None


def test_datasetspec_creation():
    """Test DatasetSpec dataclass creation"""
    files = {
        "file1.root": CoffeaFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=10,
            uuid="uuid1"
        )
    }
    spec = DatasetSpec(
        files=files,
        format="root",
        metadata={"sample": "test"},
        form=None
    )
    assert spec.files == files
    assert spec.format == "root"
    assert spec.metadata == {"sample": "test"}
    assert spec.form is None


def test_datasetspec_optional_creation():
    """Test DatasetSpecOptional dataclass creation"""
    files = {
        "file1.root": "Events"
    }
    spec = DatasetSpecOptional(
        files=files,
        format="root",
        metadata=None,
        form=None
    )
    assert spec.files == files
    assert spec.format == "root"
    assert spec.metadata is None
    assert spec.form is None


def test_dataset_joinable_spec_creation():
    """Test DatasetJoinableSpec dataclass creation with valid form"""
    files = {
        "file1.root": CoffeaFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=10,
            uuid="uuid1"
        )
    }
    # Create a simple valid form string (compressed)
    import awkward
    from coffea.util import compress_form
    simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
    compressed_form = compress_form(simple_form)
    
    spec = DatasetJoinableSpec(
        files=files,
        format="root",
        metadata=None,
        form=compressed_form
    )
    assert spec.files == files
    assert spec.format == "root"
    assert spec.form == compressed_form


@pytest.mark.parametrize("test_against", ["form", "format"])
def test_dataset_joinable_spec_invalid_form(test_against):
    """Test DatasetJoinableSpec validation with invalid form, format"""
    files = {
        "file1.root": CoffeaFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=10,
            uuid="uuid1"
        )
    }
    if test_against == "form":
        with pytest.raises(ValueError, match="form: was not able to decompress_form"):
            DatasetJoinableSpec(
                files=files,
                format="root",
                metadata=None,
                form="invalid_form"
            )
    elif test_against == "format":
        import awkward
        from coffea.util import compress_form
        simple_form = awkward.Array([{"x": 1}]).layout.form.to_json()
        compressed_form = compress_form(simple_form)
        
        with pytest.raises(ValueError, match="format: format must be one of"):
            DatasetJoinableSpec(
                files=files,
                format="invalid_format.txt",
                metadata=None,
                form=compressed_form
            )


def test_iofactory_valid_format():
    """Test IOFactory.valid_format method"""
    assert IOFactory.valid_format("root") is True
    assert IOFactory.valid_format("parquet") is True
    assert IOFactory.valid_format("invalid") is False


def test_iofactory_identify_format():
    """Test IOFactory.identify_format method"""
    assert IOFactory.identify_format("file.root") == "root"
    assert IOFactory.identify_format("file.parquet") == "parquet"
    assert IOFactory.identify_format("file.parq") == "parquet"
    assert IOFactory.identify_format("directory") == "parquet"  # no extension defaults to parquet
    
    with pytest.raises(RuntimeError, match="couldn't identify"):
        IOFactory.identify_format("file.txt")


def test_iofactory_dict_to_uprootfilespec():
    """Test IOFactory.dict_to_uprootfilespec method"""
    # Test CoffeaFileSpec creation
    input_dict = {
        "object_path": "Events",
        "steps": [[0, 10]],
        "num_entries": 10,
        "uuid": "test-uuid"
    }
    result = IOFactory.dict_to_uprootfilespec(input_dict)
    assert type(result) is CoffeaFileSpec
    assert result.object_path == "Events"
    assert result.steps == [[0, 10]]
    assert result.num_entries == 10
    assert result.uuid == "test-uuid"
    
    # Test CoffeaFileSpecOptional creation
    input_dict_optional = {
        "object_path": "Events",
        "steps": None,
        "num_entries": None,
        "uuid": None
    }
    result_optional = IOFactory.dict_to_uprootfilespec(input_dict_optional)
    assert type(result_optional) is CoffeaFileSpecOptional
    assert result_optional.object_path == "Events"
    assert result_optional.steps is None


def test_iofactory_dict_to_parquetfilespec():
    """Test IOFactory.dict_to_parquetfilespec method"""
    # Test CoffeaParquetFileSpec creation
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


    # Test CoffeaParquetFileSpecOptional creation
    input_dict_optional = {
        "object_path": None,
        "steps": None,
        "num_entries": None,
        "uuid": None
    }
    result_optional = IOFactory.dict_to_parquetfilespec(input_dict_optional)
    assert type(result_optional) is CoffeaParquetFileSpecOptional
    assert result_optional.object_path is None
    assert result_optional.steps is None
    assert result_optional.num_entries is None


def test_iofactory_filespec_to_dict():
    """Test IOFactory.filespec_to_dict method"""
    spec = CoffeaFileSpec(
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
    basic_spec = UprootFileSpec(object_path="Events", steps=[[0, 10]])
    with pytest.raises(ValueError, match="expects the fields provided by Coffea"):
        IOFactory.filespec_to_dict(basic_spec)


def test_iofactory_dict_to_datasetspec():
    """Test IOFactory.dict_to_datasetspec method"""
    input_dict = {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 10]],
                "num_entries": 10,
                "uuid": "test-uuid"
            }
        },
        "metadata": {"sample": "test"},
        "form": None
    }
    
    result = IOFactory.dict_to_datasetspec(input_dict)
    assert type(result) in [DatasetSpec, DatasetSpecOptional]
    assert result.format == "root"
    assert type(result.files["tests/samples/nano_dy.root"]) in [CoffeaFileSpec, CoffeaFileSpecOptional]
    assert result.metadata == {"sample": "test"}


def test_iofactory_datasetspec_to_dict():
    """Test IOFactory.datasetspec_to_dict method"""
    files = {
        "file1.root": CoffeaFileSpec(
            object_path="Events",
            steps=[[0, 10]],
            num_entries=10,
            uuid="uuid1"
        )
    }
    spec = DatasetSpec(
        files=files,
        format="root",
        metadata={"sample": "test"},
        form=None
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


@pytest.mark.parametrize(
    "the_fileset", [_starting_fileset_dict, _starting_fileset, _runnable_result, _updated_result]
)
def test_conversion_starting_fileset_to_dataclasses(the_fileset):
    """Test converting _starting_fileset_list to dataclasses"""
    # Convert to DatasetSpecOptional
    converted = {}
    print(the_fileset)
    for dataset_name, dataset_info in the_fileset.items():
        print(dataset_name, dataset_info)
        converted[dataset_name] = IOFactory.dict_to_datasetspec(dataset_info, verbose=True)
    
    assert "ZJets" in converted
    assert "Data" in converted
    assert list(converted["ZJets"].files.keys()) == ["tests/samples/nano_dy.root"]
    assert all([key in ["tests/samples/nano_dimuon.root", "tests/samples/nano_dimuon_not_there.root"] for key in converted["Data"].files.keys()])


def test_dataclass_roundtrip_conversion():
    """Test that dataclass to dict and back preserves data"""
    # Create a DatasetSpec
    original_files = {
        "test.root": CoffeaFileSpec(
            object_path="Events",
            steps=[[0, 10], [10, 20]],
            num_entries=20,
            uuid="test-uuid"
        )
    }
    original_spec = DatasetSpec(
        files=original_files,
        format="root",
        metadata={"test": "value"},
        form=None
    )
    
    # Convert to dict
    dict_form = IOFactory.datasetspec_to_dict(original_spec)
    
    # Convert back to dataclass
    restored_spec = IOFactory.dict_to_datasetspec(dict_form)
    
    # Check equality
    assert type(restored_spec) is DatasetSpec
    assert restored_spec.format == original_spec.format
    assert restored_spec.metadata == original_spec.metadata
    assert restored_spec.form == original_spec.form
    
    # Check file specs
    original_file_spec = original_spec.files["test.root"]
    restored_file_spec = restored_spec.files["test.root"]
    assert isinstance(restored_file_spec, CoffeaFileSpec)
    assert restored_file_spec.object_path == original_file_spec.object_path
    assert restored_file_spec.steps == original_file_spec.steps
    assert restored_file_spec.num_entries == original_file_spec.num_entries
    assert restored_file_spec.uuid == original_file_spec.uuid
