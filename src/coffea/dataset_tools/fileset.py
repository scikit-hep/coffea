from __future__ import annotations

import copy

from collections.abc import Hashable
from coffea.util import _is_interpretable, compress_form, decompress_form
from typing import Any, Callable, Annotated, Literal
from annotated_types import Gt, Len
from pydantic import BaseModel, RootModel, Field, model_validator, computed_field

StepPair = Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)]


class UprootFileSpec(BaseModel):
    object_path: str
    steps: Annotated[list[StepPair], Field(min_length=1)] | StepPair | None = None
    format: Literal["root"] = "root"


class CoffeaUprootFileSpecOptional(UprootFileSpec):
    num_entries: Annotated[int, Field(ge=0)] | None = None
    uuid: str | None = None


class CoffeaUprootFileSpec(CoffeaUprootFileSpecOptional):
    steps: Annotated[list[StepPair], Field(min_length=1)] | StepPair
    num_entries: Annotated[int, Field(ge=0)]
    uuid: str


class ParquetFileSpec(BaseModel):
    object_path: None = None
    steps: Annotated[list[StepPair], Field(min_length=1)] | StepPair | None = None
    format: Literal["parquet"] = "parquet"


class CoffeaParquetFileSpecOptional(ParquetFileSpec):
    num_entries: Annotated[int, Field(ge=0)] | None = None
    uuid: str | None = None



class CoffeaParquetFileSpec(CoffeaParquetFileSpecOptional):
    steps: Annotated[list[StepPair], Field(min_length=1)] | StepPair
    num_entries: Annotated[int, Field(ge=0)]
    uuid: str
    #directory: Literal[True, False] #identify whether it's a directory of parquet files or a single parquet file, may be useful or necessary to distinguish


class DictMethodsMixin:
    def __getitem__(self, key: str) -> str:
        return self.root[key]

    def __setitem__(self, key: str, value: str):
        self.root[key] = value

    def __delitem__(self, key: str):
        del self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[str]:
        print("DictMethodsMixin.__iter__ called")
        return iter(self.root)

    def keys(self) -> Iterator[str]:
        return self.root.keys()

    def values(self) -> Iterator[str]:
        return self.root.values()

    def items(self) -> Iterator[Tuple[str, str]]:
        return self.root.items()

    def get(self, key: str, default=None) -> str | None:
        return self.root.get(key, default)

    def pop(self, key: str, default=...):
        return self.root.pop(key, default)

    def update(self, other=None, **kwargs):
        self.root.update(other, **kwargs)

class CoffeaFileDict(RootModel[dict[str, CoffeaUprootFileSpec | CoffeaParquetFileSpec | CoffeaUprootFileSpecOptional | CoffeaParquetFileSpecOptional]], DictMethodsMixin):

    def __iter__(self) -> Iterator[str]:
        print("CoffeaFileDict.__iter__ called")
        return iter(self.root)
    
    @computed_field
    @property
    def format(self) -> str:
        """Identify the format of the files in the dictionary."""
        union = set()
        formats_by_name = {k: set(IOFactory.identify_format(k), v.format) for k, v in self.root.items()}
        union.update(formats_by_name.values())
        if len(union) == 1:
            return union.pop()
        return "|".join(union)
    
    @model_validator(mode='before')
    def preproc_data(cls, data: Any) -> Any:
        for k, v in data.items():
            if isinstance(v, (str, type(None))):
                data[k] = {"object_path": v}
                v = data[k]
            if isinstance(v, dict):
                fmt = IOFactory.identify_format(k)
                if fmt == "root":
                    if "format" not in v:
                        v["format"] = "root"
                    else:
                        assert v["format"] == "root", f"Expected 'format' to be 'root', got {v['format']} for {k}"
                elif fmt == "parquet":
                    if "format" not in v:
                        v["format"] = "parquet"
                    else:
                        assert v["format"] == "parquet", f"Expected 'format' to be 'parquet', got {v['format']} for {k}"
        return data
    
    @model_validator(mode='after')
    def promote_and_check_files(self) -> Self:
        for k, v in self.root.items():
            try:
                if type(v) in [CoffeaUprootFileSpecOptional]:
                    self.root[k] = CoffeaUprootFileSpec(v)
                if type(v) in [CoffeaParquetFileSpecOptional]:
                    self.root[k] = CoffeaParquetFileSpec(v)
            except Exception:
                pass
        return self


class DatasetSpec(BaseModel):
    files: CoffeaFileDict
    """ files: (
        CoffeaUprootFileDict | CoffeaFileDictOptional
        | dict[
            str,
            str
            | UprootFileSpec
            | ParquetFileSpec
            | CoffeaUprootFileSpecOptional
            | CoffeaParquetFileSpecOptional,
        ]
        | list[str]
    ) """
    metadata: dict[Hashable, Any] | None = None
    format: str | None = None
    form: str | None = None
    
    
    @model_validator(mode='before')
    def preprocess_data(cls, data: Any) -> Any:
        print("preprocess_data - data:", data)
        if isinstance(data, dict):
            files = data.pop("files")
            #promote files list to dict if necessary
            if isinstance(files, list):
                # If files is a list, convert it to a dict and let it pass through the rest of the promotion logic
                tmp = [f.rsplit(":", maxsplit=1) for f in files]
                files = {}
                for fsplit in tmp:
                    # Need a valid split into file name and object path
                    if len(fsplit) > 1:
                        #but ensure we don't catch 'root://' and split that
                        if fsplit[1].startswith("//"):
                            # no object path
                            files[":".join(fsplit)] = None
                        else:
                            # file name and object path
                            files[fsplit[0]] = fsplit[1]
            data["files"] = files
        elif not isinstance(data, DatasetSpec):
            raise ValueError("DatasetSpec expects a dictionary with a 'files' key or a DatasetSpec instance")
        return data
    

    @model_validator(mode='after')
    def check_form(self) -> Self:
        """Check the form can be decompressed, validate the format if mannually specified, and then ."""
        # check_form
        if self.form is not None:
            # If there's a form, validate we can decompress it into an awkward form
            try:
                import awkward

                from coffea.util import decompress_form

                _ = awkward.forms.from_json(decompress_form(self.form))
            except Exception as e:
                raise ValueError(
                    "form: was not able to decompress_form into an awkward form"
                ) from e
        
        if self.format is None:

            # set the format if not already set
            union = set()
            formats_by_name = {k: v.format for k, v in self.files.items()}
            union.update(formats_by_name.values())
            if len(union) == 1:
                self.format = union.pop()
            else:
                self.format = "|".join(union)
        else:
            # validate the format, if present
            if not IOFactory.valid_format(self.format):
                raise ValueError(f"format: format must be one of {IOFactory._formats}")
        
        return self
    
    @model_validator(mode='after')
    def set_format(self) -> Self:
        return self
        """ if isinstance(files, dict):
                for k, v in self.files.items():
                    try:
                        #passthrough or promote to concrete filespec
                        if type(v) in [CoffeaFileSpec, CoffeaParquetFileSpec]:
                            continue
                        elif type(v) in [CoffeaFileSpecOptional]:
                            self.files[k] = CoffeaFileSpec(v)
                        elif type(v) in [CoffeaParquetFileSpecOptional]:
                            self.files[k] = CoffeaParquetFileSpec(v)
                        # handle the basic cases of filename: object_path pairs
                        elif type(v) in [str, type(None)]:
                            # we only have the basic information, identify the format and convert to the appropriate filespecoptional
                            if IOFactory.identify_format(k) in ["root"]:
                                self.files[k] = CoffeaFileSpecOptional(object_path=v)
                            elif IOFactory.identify_format(k) in ["parquet"]:
                                self.files[k] = CoffeaParquetFileSpecOptional(object_path=v)
                            else:
                                raise ValueError(
                                    f"{k}: {v} couldn't be identified as either root or parquet format for conversion"
                                )

                    except Exception:
            # now we have a dictionary or CoffeaFileDict(Optional)
            try:
                self.files = CoffeaFileDict(dict(self.files))
            except Exception as e:
                self.files = CoffeaFileDictOptional(dict(self.files))
            #check if the format can be set
            formats = {
                k: IOFactory.identify_format(k) for k in self.files.keys()
            }
            auto_format = None
            if all(fmt == "root" for fmt in formats.values()):
                auto_format = "root"
            elif all(fmt == "parquet" for fmt in formats.values()):
                auto_format = "parquet"
            elif all(fmt in ["root", "parquet"] for fmt in formats.values()):
                # mixed formats, leave format as None
                auto_format = "root|parquet"
            if self.format is None and auto_format is not None:
                self.format = auto_format
            # check if there's a non-None form. If possible, promote to DatasetJoinableSpec, else DatasetSpec
            if self.format is not None:
                if self.form is not None and self.form is not None:
                    try:
                    return DatasetJoinableSpec(self)
                    except Exception:
                        pass
                else:
                    try:
                        return DatasetSpec(self)
                    except Exception:
                        pass
        return self
        """

class DatasetSpecOptional(BaseModel):
    files: (
        CoffeaFileDict
        | dict[
            str,
            str
            | UprootFileSpec
            | ParquetFileSpec
            | CoffeaUprootFileSpecOptional
            | CoffeaParquetFileSpecOptional,
        ]
        | list[str]
    )
    metadata: dict[Hashable, Any] | None = None
    format: str | None = None
    form: str | None = None

    @model_validator(mode='after')
    def attempt_promotion(self) -> Self:
        if isinstance(self.files, list):
            # If files is a list, convert it to a dict and let it pass through the rest of the promotion logic
            tmp = [f.rsplit(":", maxsplit=1) for f in self.files]
            self.files = {}
            for fsplit in tmp:
                # Need a valid split into file name and object path
                if len(fsplit) > 1:
                    #but ensure we don't catch 'root://' and split that
                    if fsplit[1].startswith("//"):
                        # no object path
                        self.files[":".join(fsplit)] = None
                    else:
                        # file name and object path
                        self.files[fsplit[0]] = fsplit[1]
        if isinstance(self.files, dict):
            for k, v in self.files.items():
                try:
                    #passthrough or promote to concrete filespec
                    if type(v) in [CoffeaUprootFileSpec, CoffeaParquetFileSpec]:
                        continue
                    elif type(v) in [CoffeaUprootFileSpecOptional]:
                        self.files[k] = CoffeaUprootFileSpec(v)
                    elif type(v) in [CoffeaParquetFileSpecOptional]:
                        self.files[k] = CoffeaParquetFileSpec(v)
                    # handle the basic cases of filename: object_path pairs
                    elif type(v) in [str, type(None)]:
                        # we only have the basic information, identify the format and convert to the appropriate filespecoptional
                        if IOFactory.identify_format(k) in ["root"]:
                            self.files[k] = CoffeaUprootFileSpecOptional(object_path=v)
                        elif IOFactory.identify_format(k) in ["parquet"]:
                            self.files[k] = CoffeaParquetFileSpecOptional(object_path=v)
                        else:
                            raise ValueError(
                                f"{k}: {v} couldn't be identified as either root or parquet format for conversion"
                            )

                except Exception:
                    pass
        # now we have a dictionary or CoffeaFileDict(Optional)
        try:
            self.files = CoffeaFileDict(dict(self.files))
        except Exception as e:
            self.files = CoffeaFileDictOptional(dict(self.files))
        #check if the format can be set
        formats = {
            k: IOFactory.identify_format(k) for k in self.files.keys()
        }
        auto_format = None
        if all(fmt == "root" for fmt in formats.values()):
            auto_format = "root"
        elif all(fmt == "parquet" for fmt in formats.values()):
            auto_format = "parquet"
        elif all(fmt in ["root", "parquet"] for fmt in formats.values()):
            # mixed formats, leave format as None
            auto_format = "root|parquet"
        if self.format is None and auto_format is not None:
            self.format = auto_format
        # check if there's a non-None form. If possible, promote to DatasetJoinableSpec, else DatasetSpec
        if self.format is not None:
            if self.form is not None and self.form is not None:
                try:
                   return DatasetJoinableSpec(self)
                except Exception:
                    pass
            else:
                try:
                    return DatasetSpec(self)
                except Exception:
                    pass
        return self


class DatasetJoinableSpec(DatasetSpec):
    files: CoffeaFileDict | CoffeaFileDictOptional
    form: str  # form is required
    format: str


    @model_validator(mode='after')
    def check_form_and_format(self) -> Self:
        if not IOFactory.valid_format(self.format):
            raise ValueError(f"format: format must be one of {IOFactory._formats}")
        try:
            import awkward

            from coffea.util import decompress_form

            _ = awkward.forms.from_json(decompress_form(self.form))
        except Exception as e:
            raise ValueError(
                "form: was not able to decompress_form into an awkward form"
            ) from e
        return self

""" 
class FilesetSpecOptional(RootModel[dict[str, DatasetJoinableSpec | DatasetSpecOptional | DatasetSpec ]], DictMethodsMixin):
    def __iter__(self) -> Iterator[str]:
        return iter(self.root) """


class FilesetSpec(RootModel[dict[str, DatasetSpec]], DictMethodsMixin):
    def __iter__(self) -> Iterator[str]:
        print("FilesetSpec.__iter__ called")
        return iter(self.root)


class IOFactory:
    _formats = ["root", "parquet"]

    def __init__(self):
        pass

    @classmethod
    def valid_format(cls, format: str | DatasetSpecOptional | DatasetSpec | DatasetJoinableSpec) -> bool:
        if type(format) in [DatasetSpecOptional, DatasetSpec, DatasetJoinableSpec]:
            return format.format in cls._formats
        return format in cls._formats

    @classmethod
    def promote_datasetspec(
        cls, input: DatasetSpec | DatasetSpecOptional | DatasetJoinableSpec
    ):
        print("promoting newstyle:", input)
        tmp = CoffeaFileDictOptional({"placeholder": input})["placeholder"]
        print(type(input), "promoted to", type(tmp))
        if type(input) is DatasetJoinableSpec:
            return input
        elif isinstance(input, dict):
            return cls.dict_to_datasetspec(input)
        else:
            # If the input is already a DatasetSpec or DatasetSpecOptional, we can try to promote it to DatasetJoinableSpec
            try:
                return DatasetJoinableSpec(
                    files=input.files,
                    format=input.format,
                    metadata=input.metadata,
                    form=input.form,
                )
            except Exception:
                return input

    @classmethod
    def identify_format(cls, input: Any):
        if type(input) in [DatasetJoinableSpec, DatasetSpec, DatasetSpecOptional]:
            return input.format

        if isinstance(input, str):
            # could check with regular expressions for more compmlicated naming, like atlas .root.N
            if input.endswith(".root"):
                return "root"
            if (
                input.endswith(".parq")
                or input.endswith(".parquet")
                or "." not in input.split("/")[-1]
            ):
                return "parquet"
            else:
                raise RuntimeError(
                    f"{cls.__name__} couldn't identify if the string path is for a root file or parquet file/directory"
                )
        else:
            raise NotImplementedError(
                "identify_format doesn't handle all valid input types, such as fsspec instances"
            )

    @classmethod
    def dict_to_uprootfilespec(cls, input):
        """Convert a dictionary to a CoffeaFileSpec or CoffeaFileSpecOptional."""
        assert isinstance(input, dict), f"{input} is not a dictionary"
        try:
            return CoffeaUprootFileSpec(**input)
        except Exception:
            print("Failed to create CoffeaUprootFileSpec, trying optional")
            add_args = {k: v for k, v in input.items()}
            if "steps" not in input:
                add_args["steps"] = None
            if "num_entries" not in input:
                add_args["num_entries"] = None
            if "uuid" not in input:
                add_args["uuid"] = None
            return CoffeaUprootFileSpecOptional(**add_args)

    @classmethod
    def dict_to_parquetfilespec(cls, input):
        """Convert a dictionary to a CoffeaParquetFileSpec or CoffeaParquetFileSpecOptional."""
        assert isinstance(input, dict), f"{input} is not a dictionary"
        try:
            return CoffeaParquetFileSpec(**input)
        except Exception:
            print("Failed to create CoffeaParquetFileSpec, trying optional")
            add_args = {k: v for k, v in input.items()}
            if "steps" not in input:
                add_args["steps"] = None
            if "num_entries" not in input:
                add_args["num_entries"] = None
            if "uuid" not in input:
                add_args["uuid"] = None
            return CoffeaParquetFileSpecOptional(**add_args)

    @classmethod
    def filespec_to_dict(
        cls,
        input: (
            CoffeaUprootFileSpec
            | CoffeaUprootFileSpecOptional
            | CoffeaParquetFileSpec
            | CoffeaParquetFileSpecOptional
        ),
    ):
        if type(input) in [UprootFileSpec, ParquetFileSpec]:
            raise ValueError(
                f"{cls.__name__}.filespec_to_dict expects the fields provided by Coffea(Parquet)FileSpec(Optional), UprootFileSpec and ParquetFileSpec should be promoted"
            )
        if type(input) not in [
            CoffeaUprootFileSpec,
            CoffeaUprootFileSpecOptional,
            CoffeaParquetFileSpec,
            CoffeaParquetFileSpecOptional,
        ]:
            raise TypeError(
                f"{cls.__name__}.filespec_to_dict expects a Coffea(Parquet)FileSpec(Optional), got {type(input)} instead: {input}"
            )
        output = {}
        output["object_path"] = input.object_path
        output["steps"] = input.steps
        output["num_entries"] = input.num_entries
        output["uuid"] = input.uuid
        return output

    @classmethod
    def dict_to_datasetspec(
        cls, input: dict[str, Any], verbose=False
    ) -> DatasetSpec:
        return DatasetSpec(**input)

    @classmethod
    def old_dict_to_datasetspec(
        cls, input: dict[str, Any], verbose=False
    ) -> DatasetSpec | DatasetSpecOptional | DatasetJoinableSpec:
        input = copy.deepcopy(input)
        output = {}
        # if the input doesn't contain an explicit "files" key, assume the input is a files dictionary
        output["files"] = input.get("files", copy.deepcopy(input))
        if not isinstance(output["files"], dict):
            raise ValueError(
                f"{cls.__name__}.dict_to_datasetspec expects a nested dictionary with files key or interprets the dictionary as filename: object_path pairs, got {output['files']} instead"
            )
        output["format"] = None
        output["metadata"] = input.get("metadata", None)
        output["form"] = input.get("form", None)
        concrete_vs_optional = {}
        formats = {}
        for name, info_raw in output["files"].items():
            format = cls.identify_format(name)
            formats[name] = format
            info_to_convert = copy.deepcopy(info_raw)
            if type(info_to_convert) in [
                CoffeaUprootFileSpec,
                CoffeaParquetFileSpec,
                CoffeaUprootFileSpecOptional,
                CoffeaParquetFileSpecOptional,
            ]:
                # convert to dict to allow promotion potentially
                info_to_convert = cls.filespec_to_dict(info_to_convert)
            elif isinstance(info_to_convert, (str, type(None))):
                # if it's a string, assume it's the object path for root
                # if it's None, assume it's the object path for parquet
                info_to_convert = {"object_path": info_to_convert}
            # Now convert to the appropriate filespec type
            if format in ["root"]:
                output["files"][name] = cls.dict_to_uprootfilespec(info_to_convert)
            elif format in ["parquet"]:
                output["files"][name] = cls.dict_to_parquetfilespec(info_to_convert)
            else:
                raise ValueError(
                    f"{name}: {info_raw} couldn't be identified as either root or parquet format for conversion"
                )

            info = output["files"][name]

            if type(info) in [CoffeaParquetFileSpec, CoffeaUprootFileSpec]:
                concrete_vs_optional[name] = True
            elif type(info) in [CoffeaParquetFileSpecOptional, CoffeaUprootFileSpecOptional]:
                concrete_vs_optional[name] = False
            else:
                concrete_vs_optional[name] = None

        if all(fmt == "root" for fmt in formats.values()):
            output["format"] = "root"
        elif all(fmt == "parquet" for fmt in formats.values()):
            output["format"] = "parquet"

        if all(concrete_vs_optional.values()):
            if output["form"] is not None:
                return DatasetJoinableSpec(**output)
            else:
                return DatasetSpec(**output)
        else:
            if verbose:
                print(f"concrete_vs_optional: {concrete_vs_optional}")
            return DatasetSpecOptional(**output)

    @classmethod
    def datasetspec_to_dict(
        cls,
        input: DatasetSpec | DatasetSpecOptional | DatasetJoinableSpec,
        coerce_filespec_to_dict=True,
    ) -> dict[str, Any]:
        assert type(input) in [
            DatasetSpec,
            DatasetSpecOptional,
            DatasetJoinableSpec,
        ], f"{cls.__name__}.datasetspec_to_dict expects a DatasetSpec, DatasetSpecOptional or DatasetJoinableSpec, got {type(input)} instead: {input}"
        output = {}
        output["files"] = {} if coerce_filespec_to_dict else input.files
        output["format"] = input.format
        output["metadata"] = input.metadata
        output["form"] = input.form
        if coerce_filespec_to_dict:
            for name, info in input.files.items():
                output["files"][name] = cls.filespec_to_dict(info)

        return output

if __name__ == "__main__":
    # This is a placeholder for the main function or test cases
    for steps in [None, [0, 100], [[0, 1], [2, 3]]]:
        print(steps)
        try:
            a = UprootFileSpec(object_path="example_path", steps=steps)
        except Exception as e:
            print(f"Error creating UprootFileSpec with steps={steps}: {e}")
        for num_entries in [None, 100]:
            print("\n\t", num_entries)
            try:
                b = CoffeaUprootFileSpecOptional(
                    object_path="example_path",
                    steps=steps,
                    num_entries=num_entries,
                )
            except Exception as e:
                print(f"\t\tError creating CoffeaUprootFileSpecOptional with steps={steps} and num_entries={num_entries}: {e}")
        for uuid in [None, "hello-there"]:
            print("\n\t", uuid)
            try:
                c1 = CoffeaUprootFileSpecOptional(
                    object_path="example_path",
                    steps=steps,
                    uuid=uuid,
                )
            except Exception as e:
                print(f"\t\tError creating CoffeaUprootFileSpecOptional with steps={steps} and num_entries={num_entries}: {e}")
            try:
                c2 = CoffeaParquetFileSpecOptional(
                    object_path=None,
                    steps=steps,
                    uuid=uuid,
                )
            except Exception as e:
                print(f"\t\tError creating CoffeaParquetFileSpecOptional with steps={steps} and num_entries={num_entries}: {e}")
        for num_entries, uuid in [(100, "hello-there")]:
            print("\n\t", num_entries, uuid)
            try:
                d1 = CoffeaFileSpec(
                    object_path="example_path",
                    steps=steps,
                    num_entries=num_entries,
                    uuid=uuid,
                )
            except Exception as e:
                print(f"\t\tError creating CoffeaFileSpec with steps={steps}, num_entries={num_entries}, and uuid={uuid}: {e}")
            try:
                d2 = CoffeaParquetFileSpec(
                    object_path=None,
                    steps=steps,
                    num_entries=num_entries,
                    uuid=uuid,
                )
            except Exception as e:
                print(f"\t\tError creating CoffeaParquetFileSpec with steps={steps}, num_entries={num_entries}, and uuid={uuid}: {e}")

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

    converted = {}
    for k, v in _starting_fileset.items():
        print("\n\nConverting:", k, v.keys())
        #converted[k] = IOFactory.dict_to_datasetspec(v)
        converted[k] = DatasetSpecOptional(**v)
        try:
            print("\nValidating:", k, "for DatasetSpecOptional")
            DatasetSpecOptional.model_validate_json(converted[k].model_dump_json())
        except Exception as e:
            print("DatasetSpecOptional failed to validate:", k, v, e)
        try:
            print("\nValidating:", k, "for DatasetSpec")
            DatasetSpec.model_validate_json(converted[k].model_dump_json())
        except Exception as e:
            print("DatasetSpec failed to validate:", k, v, e)
        try:
            print("\nValidating:", k, "for FilesetSpecOptional")
            FilesetSpecOptional.model_validate_json(FilesetSpecOptional({k: converted[k]}).model_dump_json())
        except Exception as e:
            print("FilesetSpecOptional failed to validate:", k, v, e)
        print("\n\nTest writing each out to json and loading it back in")
    print("\n\n")
    conv_pyd = FilesetSpecOptional(converted)
    import rich
    rich.print(conv_pyd)
    #print(converted)
    with open("test.json", "w") as f:
        import json
        json.dump(
            conv_pyd.model_dump_json(),
            f,
            indent=2,
            sort_keys=True,
        )
    with open("test.json", "r") as f:
        import json
        data = json.load(f)
        print("Attempting to restore from JSON data")
        restored = FilesetSpecOptional.model_validate_json(data)
        rich.print(restored)

    rich.print("[red]Creating DatasetSpecOptional")
    test_input = {"ZJets1": {
                    "files": {
                        "tests/samples/nano_dy.root": {
                            "object_path": "Events",
                            "steps": [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30]],
                            "num_entries": 30,
                            "uuid": "1234-5678-90ab-cdef",
                        },
                        "tests/samples/nano_dy_2.root": {
                            "object_path": "Events",
                            "steps": None,
                            "num_entries": 30,
                            "uuid": "1234-5678-90ab-cdef",
                        }
                    },
                    "format": "root",
                    "metadata": {"key": "value"},
                    "form": "awkward:0.15.0",
                }}
    # convert test_input via direct constructor:
    test = {k: DatasetSpecOptional(**v) for k, v in test_input.items()}
    test["ZJets1"].files["tests/samples/nano_dy_2.root"].steps = [0, 30]
    rich.print(test)
    rich.print("[blue]Trying to use FilesetSpecOptional")
    test2 = FilesetSpecOptional(test)
    rich.print(test2)
    #rich.print("[green]Trying promote_datasetspec on DatasetSpecOptional")
    #test2 = IOFactory.promote_datasetspec(test["ZJets1"])
    #rich.print(test2)
    """ for k, v in test.items():
        v.object_path = "Events"
        v.steps = [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30]]
        v.num_entries = 30
        v.uuid = "1234-5678-90ab-cdef"
    rich.print(test) """
    print("[red]Trying promote_datasetspec on CoffeaFileDictOptional")
    """ IOFactory.promote_datasetspec(
    DatasetSpecOptional(test """