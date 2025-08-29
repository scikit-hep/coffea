from __future__ import annotations

import copy
import re
from collections.abc import Hashable, Iterable
from typing import Annotated, Any, Literal, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import BaseModel, Field, RootModel, computed_field, model_validator

StepPair = Annotated[
    list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)
]


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
    # directory: Literal[True, False] #identify whether it's a directory of parquet files or a single parquet file, may be useful or necessary to distinguish


class DictMethodsMixin:
    def __getitem__(self, key: str) -> str:
        return self.root[key]

    def __setitem__(self, key: str, value: str):
        self.root[key] = value

    def __delitem__(self, key: str):
        del self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterable[str]:
        return iter(self.root)

    def keys(self) -> Iterable[str]:
        return self.root.keys()

    def values(self) -> Iterable[str]:
        return self.root.values()

    def items(self) -> Iterable[tuple[str, str]]:
        return self.root.items()

    def get(self, key: str, default=None) -> str | None:
        return self.root.get(key, default)

    def pop(self, key: str, default=...):
        return self.root.pop(key, default)

    def update(self, other=None, **kwargs):
        self.root.update(other, **kwargs)


class CoffeaFileDict(
    RootModel[
        dict[
            str,
            Union[
                CoffeaUprootFileSpec,
                CoffeaParquetFileSpec,
                CoffeaUprootFileSpecOptional,
                CoffeaParquetFileSpecOptional,
            ],
        ]
    ],
    DictMethodsMixin,
):

    def __iter__(self) -> Iterable[str]:
        return iter(self.root)

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
        assert all(
            k in identified_formats_by_name and identified_formats_by_name[k] == v
            for k, v in stored_formats_by_name.items()
        ), f"identified formats and stored formats do not match: identified formats: {identified_formats_by_name}, stored formats: {stored_formats_by_name}"
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

    @model_validator(mode="after")
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
    metadata: dict[Hashable, Any] | None = None
    format: str | None = None
    form: str | None = None

    @model_validator(mode="before")
    def preprocess_data(cls, data: Any) -> Any:
        data = copy.deepcopy(data)
        if isinstance(data, dict):
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
            data["files"] = files
        elif isinstance(data, DatasetSpec):
            data = data.model_dump()
        elif not isinstance(data, DatasetSpec):
            raise ValueError(
                "DatasetSpec expects a dictionary with a 'files' key or a DatasetSpec instance"
            )
        return data

    @model_validator(mode="after")
    def check_form(self) -> Self:
        """Check the form can be decompressed, validate the format if manually specified, and then ."""
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

    # @computed_field
    # @property
    def joinable(self) -> bool:
        """Identify DatasetSpec criteria to be pre-joined for typetracing (necessary) and column-joining (sufficient)"""
        if not IOFactory.valid_format(self.format):
            return False
        try:
            import awkward

            from coffea.util import decompress_form

            _ = awkward.forms.from_json(decompress_form(self.form))
            return True
        except Exception:
            return False


class FilesetSpec(RootModel[dict[str, DatasetSpec]], DictMethodsMixin):
    def __iter__(self) -> Iterable[str]:
        return iter(self.root)

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
            return format.format in cls._formats or all(
                fmt in cls._formats for fmt in format.format.split("|")
            )
        return format in cls._formats

    @classmethod
    def attempt_promotion(
        cls,
        input: (
            CoffeaUprootFileSpec
            | CoffeaUprootFileSpecOptional
            | CoffeaParquetFileSpec
            | CoffeaParquetFileSpecOptional
            | DatasetSpec
            | FilesetSpec
        ),
    ):
        try:
            if isinstance(input, (CoffeaUprootFileSpec, CoffeaUprootFileSpecOptional)):
                return CoffeaUprootFileSpec(**input.model_dump())
            elif isinstance(
                input, (CoffeaParquetFileSpec, CoffeaParquetFileSpecOptional)
            ):
                return CoffeaParquetFileSpec(**input.model_dump())
            elif isinstance(input, CoffeaFileDict):
                return CoffeaFileDict(input.model_dump())
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
            return CoffeaUprootFileSpec(**input)
        except Exception:
            return CoffeaUprootFileSpecOptional(**input)

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
