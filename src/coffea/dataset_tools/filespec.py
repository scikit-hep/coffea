from __future__ import annotations

import copy
import re
from collections.abc import Hashable, Iterable, MutableMapping
from typing import Annotated, Any, Literal, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import BaseModel, Field, RootModel, computed_field, model_validator

StepPair = Annotated[
    list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)
]


class GenericFileSpec(BaseModel):
    object_path: str | None = None
    steps: Annotated[list[StepPair], Field(min_length=1)] | None = None
    num_entries: Annotated[int, Field(ge=0)] | None = None
    format: str | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        if self.steps is None:
            return self
        self.steps.sort(key=lambda x: x[1])
        starts = [step[0] for step in self.steps]
        stops = [step[1] for step in self.steps]

        # check starts and stops are monotonically increasing
        step_indices_to_remove = []
        for i in range(1, len(self.steps)):
            if starts[i] < stops[i - 1]:
                raise ValueError(
                    f"steps: start of step {i} ({starts[i]}) is less than stop of previous step ({stops[i-1]})"
                )
            if stops[i] < starts[i]:
                raise ValueError(
                    f"steps: stop of step {i} ({stops[i]}) is less than the corresponding start ({starts[i]})"
                )
            if self.num_entries is not None and stops[i] > self.num_entries:
                if starts[i] >= self.num_entries:
                    # both start and stop are beyond num_entries, remove this step
                    step_indices_to_remove.append(i)
                else:
                    # only stop is beyond num_entries, cap it to num_entries
                    self.steps[i][1] = self.num_entries
        # remove any steps that are entirely beyond num_entries
        for index in reversed(step_indices_to_remove):
            del self.steps[index]
        return self


    @computed_field
    @property
    def num_entries_in_steps(self) -> int | None:
        """Compute the total number of entries covered by the steps."""
        if self.steps is None:
            return None
        total = 0
        for start, stop in self.steps:
            total += stop - start
        return total

class ROOTFileSpec(GenericFileSpec):
    object_path: str
    format: Literal["root"] = "root"


class CoffeaROOTFileSpecOptional(ROOTFileSpec):
    num_entries: Annotated[int, Field(ge=0)] | None = None
    uuid: str | None = None


class CoffeaROOTFileSpec(CoffeaROOTFileSpecOptional):
    steps: Annotated[list[StepPair], Field(min_length=1)]
    num_entries: Annotated[int, Field(ge=0)]
    uuid: str


class ParquetFileSpec(GenericFileSpec):
    object_path: None = None
    format: Literal["parquet"] = "parquet"


class CoffeaParquetFileSpecOptional(ParquetFileSpec):
    num_entries: Annotated[int, Field(ge=0)] | None = None
    uuid: str | None = None


class CoffeaParquetFileSpec(CoffeaParquetFileSpecOptional):
    steps: Annotated[list[StepPair], Field(min_length=1)]
    num_entries: Annotated[int, Field(ge=0)]
    uuid: str
    # is_directory: Literal[True, False] #identify whether it's a directory of parquet files or a single parquet file, may be useful or necessary to distinguish


# Create union type definition
FileSpecUnion = Union[
    CoffeaROOTFileSpec,
    CoffeaParquetFileSpec,
    CoffeaROOTFileSpecOptional,
    CoffeaParquetFileSpecOptional,
]


ConcreteFileSpecUnion = Union[
    CoffeaROOTFileSpec,
    CoffeaParquetFileSpec,
]


class InputFilesMixin:
    @computed_field
    @property
    def format(self) -> str:
        """Identify the format of the files in the dictionary."""
        union = set()
        identified_formats_by_name = {
            k: identify_file_format(k) for k, v in self.root.items()
        }
        stored_formats_by_name = {
            k: v.format for k, v in self.root.items() if hasattr(v, "format")
        }
        if not all(
            k in identified_formats_by_name and identified_formats_by_name[k] == v
            for k, v in stored_formats_by_name.items()
        ):
            raise ValueError(
                f"identified formats and stored formats do not match: identified formats: {identified_formats_by_name}, stored formats: {stored_formats_by_name}"
            )
        union.update(identified_formats_by_name.values())
        if len(union) == 1:
            return union.pop()
        return "|".join(union)

    @model_validator(mode="before")
    def preproc_data(cls, data: Any) -> Any:
        data = copy.deepcopy(data)
        for k, v in data.items():
            if isinstance(v, (str, type(None))):
                data[k] = {"object_path": v}
                v = data[k]
            if isinstance(v, dict):
                fmt = identify_file_format(k)
                if fmt == "root":
                    if "format" not in v:
                        v["format"] = "root"
                    else:
                        assert (
                            v["format"] == "root"
                        ), f"Expected 'format' to be 'root', got {v['format']} for {k}"
                elif fmt == "parquet":
                    if "format" not in v:
                        v["format"] = "parquet"
                    else:
                        assert (
                            v["format"] == "parquet"
                        ), f"Expected 'format' to be 'parquet', got {v['format']} for {k}"
        return data


class InputFiles(
    RootModel[
        dict[
            str,
            FileSpecUnion,
        ]
    ],
    MutableMapping,
    InputFilesMixin,
):

    def __iter__(self) -> Iterable[str]:
        return iter(self.root)

    def __getitem__(self, key: str) -> FileSpecUnion:
        return self.root[key]

    def __setitem__(self, key: str, value: FileSpecUnion):
        self.root[key] = value

    def __delitem__(self, key: str):
        del self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    @model_validator(mode="after")
    def promote_and_check_files(self) -> Self:
        preprocessed = {k: False for k in self.root}
        for k, v in self.root.items():
            try:
                if isinstance(v, CoffeaROOTFileSpecOptional):
                    self.root[k] = CoffeaROOTFileSpec(v)
                if isinstance(v, CoffeaParquetFileSpecOptional):
                    self.root[k] = CoffeaParquetFileSpec(v)
                preprocessed[k] = True
            except Exception:
                pass
        if all(preprocessed.values()):
            try:
                self.root = PreprocessedFiles(self.root).root
            except Exception:
                pass
        return self


class PreprocessedFiles(
    RootModel[
        dict[
            str,
            ConcreteFileSpecUnion,
        ]
    ],
    MutableMapping,
    InputFilesMixin,
):

    def __iter__(self) -> Iterable[str]:
        return iter(self.root)

    def __getitem__(self, key: str) -> ConcreteFileSpecUnion:
        return self.root[key]

    def __setitem__(self, key: str, value: ConcreteFileSpecUnion):
        self.root[key] = value

    def __delitem__(self, key: str):
        del self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    @model_validator(mode="after")
    def promote_and_check_files(self) -> Self:
        preprocessed = {k: False for k in self.root}
        for k, v in self.root.items():
            try:
                if isinstance(v, CoffeaROOTFileSpecOptional):
                    self.root[k] = CoffeaROOTFileSpec(v)
                if isinstance(v, CoffeaParquetFileSpecOptional):
                    self.root[k] = CoffeaParquetFileSpec(v)
                preprocessed[k] = True
            except Exception:
                pass
        if all(preprocessed.values()):
            return self


class DatasetSpec(BaseModel):
    files: InputFiles
    metadata: dict[Hashable, Any] = {}
    format: str | None = None
    compressed_form: str | None = None

    @model_validator(mode="before")
    def preprocess_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = copy.deepcopy(data)
            files = data.pop("files")
            # promote files list to dict if necessary
            if isinstance(files, list):
                # If files is a list, convert it to a dict and let it pass through the rest of the promotion logic
                tmp = [f.rsplit(":", maxsplit=1) for f in files]
                files = {}
                for fsplit in tmp:
                    # Need a valid split into file name and object path
                    if len(fsplit) > 1:
                        # but ensure we don't catch 'root://' and split that
                        if fsplit[1].startswith("//"):
                            # no object path
                            files[":".join(fsplit)] = None
                        else:
                            # file name and object path
                            files[fsplit[0]] = fsplit[1]
                    else:
                        # no object path
                        files[fsplit[0]] = None
            data["files"] = files
            if "form" in data.keys():
                _form = data.pop("form")
                if "compressed_form" not in data.keys():
                    import awkward

                    from coffea.util import compress_form

                    data["compressed_form"] = compress_form(
                        awkward.forms.from_json(_form)
                    )
            if "metadata" not in data.keys() or data["metadata"] is None:
                data["metadata"] = {}
        elif isinstance(data, DatasetSpec):
            data = data.model_dump()
        elif not isinstance(data, DatasetSpec):
            raise ValueError(
                "DatasetSpec expects a dictionary with a 'files' key DatasetSpec instance"
            )
        return data

    def _check_form(self) -> bool | None:
        """Check the form can be decompressed into an awkward form, if present"""
        if self.compressed_form is not None:
            # If there's a form, validate we can decompress it into an awkward form
            try:
                _ = self.form
                return True
            except Exception:
                return False
        else:
            return None

    def set_check_format(self) -> bool:
        """Set and/or alidate the format if manually specified"""
        if self.format is None:
            # set the format if not already set
            union = set()
            formats_by_name = {k: v.format for k, v in self.files.items()}
            union.update(formats_by_name.values())
            if len(union) == 1:
                self.format = union.pop()
            else:
                self.format = "|".join(union)

        # validate the format, if present
        if not IOFactory.valid_format(self.format):
            return False
        return True

    @model_validator(mode="after")
    def post_validate(self) -> Self:
        # check_form
        if self._check_form() is False:  # None indicates no form to check
            raise ValueError(
                "compressed_form: was not able to decompress_form into an awkward form"
            )
        # set (if necessary) and check the format
        if not self.set_check_format():
            raise ValueError(f"format: format must be one of {IOFactory._formats}")

        return self

    # @computed_field
    # @property
    def joinable(self) -> bool:
        """Identify DatasetSpec criteria to be pre-joined for typetracing (necessary) and column-joining (sufficient)"""
        if self._check_form() and self.set_check_format():
            return True

    @computed_field
    @property
    def form(self) -> str:
        if self.compressed_form is None:
            return None
        else:
            import awkward

            from coffea.util import decompress_form

            return awkward.forms.from_json(decompress_form(self.compressed_form))


class FilesetSpec(RootModel[dict[str, DatasetSpec]], MutableMapping):
    def __iter__(self) -> Iterable[str]:
        return iter(self.root)

    def __getitem__(self, key: str) -> DatasetSpec:
        return self.root[key]

    def __setitem__(self, key: str, value: DatasetSpec):
        self.root[key] = value

    def __delitem__(self, key: str):
        del self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    @model_validator(mode="before")
    def preprocess_data(cls, data: Any) -> Any:
        data = copy.deepcopy(data)
        if isinstance(data, FilesetSpec):
            return data.model_dump()
        return data


def identify_file_format(name_or_directory: str) -> str:
    root_expression = re.compile(r"\.root")
    parquet_expression = re.compile(r"\.(?:parq(?:uet)?|pq)")
    if root_expression.search(name_or_directory):
        return "root"
    elif parquet_expression.search(name_or_directory):
        return "parquet"
    elif "." not in name_or_directory.split("/")[-1]:
        # could be a parquet directory, would require a file opening to determine
        return "parquet"  # maybe "parquet?" to trigger further inspection?
    else:
        raise RuntimeError(
            f"identify_file_format couldn't identify if the string path is for a root file or parquet file/directory for {name_or_directory}"
        )


class IOFactory:
    _formats = {"root", "parquet"}

    def __init__(self):
        pass

    @classmethod
    def valid_format(cls, format: str | DatasetSpec) -> bool:
        if isinstance(format, DatasetSpec):
            test_format = format.format
        else:
            test_format = format
        return test_format in cls._formats or all(
            fmt in cls._formats for fmt in test_format.split("|")
        )

    @classmethod
    def attempt_promotion(
        cls,
        input: (
            CoffeaROOTFileSpec
            | CoffeaROOTFileSpecOptional
            | CoffeaParquetFileSpec
            | CoffeaParquetFileSpecOptional
            | InputFiles
            | DatasetSpec
            | FilesetSpec
        ),
    ):
        try:
            if isinstance(input, (CoffeaROOTFileSpec, CoffeaROOTFileSpecOptional)):
                return CoffeaROOTFileSpec(**input.model_dump())
            elif isinstance(
                input, (CoffeaParquetFileSpec, CoffeaParquetFileSpecOptional)
            ):
                return CoffeaParquetFileSpec(**input.model_dump())
            elif isinstance(input, InputFiles):
                return InputFiles(input.model_dump())
            elif isinstance(input, DatasetSpec):
                return DatasetSpec(**input.model_dump())
            elif isinstance(input, FilesetSpec):
                return FilesetSpec(input.model_dump())
            else:
                raise TypeError(
                    f"IOFactory.attempt_promotion got an unexpected input type {type(input)} for input: {input}"
                )
        except Exception:
            return input

    @classmethod
    def dict_to_uprootfilespec(cls, input):
        """Convert a dictionary to a CoffeaFileSpec or CoffeaFileSpecOptional."""
        assert isinstance(input, dict), f"{input} is not a dictionary"
        try:
            return CoffeaROOTFileSpec(**input)
        except Exception:
            return CoffeaROOTFileSpecOptional(**input)

    @classmethod
    def dict_to_parquetfilespec(cls, input):
        """Convert a dictionary to a CoffeaParquetFileSpec or CoffeaParquetFileSpecOptional."""
        assert isinstance(input, dict), f"{input} is not a dictionary"
        try:
            return CoffeaParquetFileSpec(**input)
        except Exception:
            return CoffeaParquetFileSpecOptional(**input)

    @classmethod
    def filespec_to_dict(
        cls,
        input: (
            CoffeaROOTFileSpec
            | CoffeaROOTFileSpecOptional
            | CoffeaParquetFileSpec
            | CoffeaParquetFileSpecOptional
        ),
    ):
        if type(input) in [ROOTFileSpec, ParquetFileSpec]:
            raise ValueError(
                f"{cls.__name__}.filespec_to_dict expects the fields provided by Coffea(Parquet)FileSpec(Optional), ROOTFileSpec and ParquetFileSpec should be promoted"
            )
        if type(input) not in [
            CoffeaROOTFileSpec,
            CoffeaROOTFileSpecOptional,
            CoffeaParquetFileSpec,
            CoffeaParquetFileSpecOptional,
        ]:
            raise TypeError(
                f"{cls.__name__}.filespec_to_dict expects a Coffea(Parquet)FileSpec(Optional), got {type(input)} instead: {input}"
            )
        return input.model_dump()

    @classmethod
    def dict_to_datasetspec(cls, input: dict[str, Any], verbose=False) -> DatasetSpec:
        return DatasetSpec(**input)

    @classmethod
    def datasetspec_to_dict(
        cls,
        input: DatasetSpec,
        coerce_filespec_to_dict=True,
    ) -> dict[str, Any]:
        assert isinstance(
            input, DatasetSpec
        ), f"{cls.__name__}.datasetspec_to_dict expects a DatasetSpec, got {type(input)} instead: {input}"
        if coerce_filespec_to_dict:
            return input.model_dump()
        else:
            return dict(input)
