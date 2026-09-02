from __future__ import annotations

import copy
import pathlib
import re
import sys
from collections.abc import Callable, Hashable, Iterable, MutableMapping
from typing import Annotated, Any, Literal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    computed_field,
    model_validator,
)

StepPair = Annotated[
    list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)
]


class GenericFileSpec(BaseModel):
    object_path: str | None = None
    steps: Annotated[list[StepPair], Field(min_length=1)] | None = None
    num_entries: Annotated[int, Field(ge=0)] | None = None
    format: str | None = None
    lfn: str | None = None
    pfn: str | None = None

    def __add__(self, other: GenericFileSpec) -> GenericFileSpec:
        if not isinstance(other, GenericFileSpec):
            raise TypeError(
                f"Can only add GenericFileSpec to GenericFileSpec, got {type(other)}"
            )
        if self.format != other.format:
            raise ValueError(
                f"Cannot add GenericFileSpec with different formats: {self.format} and {other.format}"
            )
        new_spec = self.model_dump()
        if self.object_path != other.object_path:
            raise ValueError(
                f"Cannot add GenericFileSpec with different object_paths: {self.object_path} and {other.object_path}"
            )
        if self.steps is None:
            new_spec["steps"] = other.steps
        elif other.steps is None:
            new_spec["steps"] = self.steps
        else:
            new_spec["steps"] = sorted(self.steps + other.steps)
        if self.num_entries is None:
            new_spec["num_entries"] = other.num_entries
        elif other.num_entries is None:
            new_spec["num_entries"] = self.num_entries
        else:
            new_spec["num_entries"] = max(self.num_entries, other.num_entries)
        return type(self)(**new_spec)

    def __sub__(self, other: GenericFileSpec) -> GenericFileSpec:
        if not isinstance(other, GenericFileSpec):
            raise TypeError(
                f"Can only subtract GenericFileSpec from GenericFileSpec, got {type(other)}"
            )
        if self.format != other.format:
            raise ValueError(
                f"Cannot subtract GenericFileSpec with different formats: {self.format} and {other.format}"
            )
        new_spec = self.model_dump()
        if self.object_path != other.object_path:
            raise ValueError(
                f"Cannot subtract GenericFileSpec with different object_paths: {self.object_path} and {other.object_path}"
            )
        if self.steps is None:
            raise ValueError("Cannot subtract when left operand's steps is None")
        else:
            if other.steps is not None:
                new_steps = [step for step in self.steps if step not in other.steps]
            else:
                new_steps = self.steps
            new_spec["steps"] = sorted(new_steps)
        new_spec["num_entries"] = (
            self.num_entries
        )  # num_entries remains unchanged, it is a property of the file
        if len(new_spec["steps"]) == 0:
            return None
        return type(self)(**new_spec)

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
    def num_selected_entries(self) -> int | None:
        """Compute the total number of entries covered by the steps."""
        if self.steps is None:
            return None
        total = 0
        for start, stop in self.steps:
            total += stop - start
        return total

    def limit_steps(self, max_steps: int | slice) -> Self:
        """Limit the steps"""
        if self.steps is None:
            # warn user
            from coffea.util import coffea_console

            coffea_console.log("limit_steps called but steps is None, no action taken")
            return self

        if not isinstance(max_steps, slice):
            max_steps = slice(max_steps)

        # if there are valid steps, then create a new Spec with only those steps, permitting method chaining
        spec = self.model_dump()
        spec["steps"] = self.steps[max_steps]
        return type(self)(**spec)


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
FileSpecUnion = (
    CoffeaROOTFileSpec
    | CoffeaParquetFileSpec
    | CoffeaROOTFileSpecOptional
    | CoffeaParquetFileSpecOptional
)


ConcreteFileSpecUnion = CoffeaROOTFileSpec | CoffeaParquetFileSpec


class InputFilesMixin:
    def __add__(self, other: InputFilesMixin) -> InputFilesMixin:
        if not isinstance(other, InputFilesMixin):
            raise TypeError(
                f"Can only add InputFilesMixin to InputFilesMixin, got {type(other)}"
            )
        new_dict = dict(self)
        for k, v in other.items():
            if k in new_dict:
                new_dict[k] = new_dict[k] + v
            else:
                new_dict[k] = v
        return type(self)(new_dict)

    def __sub__(self, other: InputFilesMixin) -> InputFilesMixin:
        if not isinstance(other, InputFilesMixin):
            raise TypeError(
                f"Can only subtract InputFilesMixin from InputFilesMixin, got {type(other)}"
            )
        new_dict = dict(self)
        for k, v in other.items():
            if k in new_dict:
                new_dict[k] = new_dict[k] - v
                if new_dict[k] is None:
                    del new_dict[k]
        return type(self)(new_dict)

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

    @property
    def num_entries(self) -> int | None:
        """Total number of entries across all files, or None if any file's count is unknown.

        An empty collection has 0 entries; a collection in which every file reports an
        entry count returns their sum; if any file's count is None (not yet known) the
        total is genuinely unknown and None is returned rather than an understated sum.
        """
        values = [v.num_entries for v in self.values()]
        if any(n is None for n in values):
            return None
        return sum(values)

    @property
    def num_selected_entries(self) -> int | None:
        """Total selected entries (from steps) across all files, or None if any is unknown."""
        values = [v.num_selected_entries for v in self.values()]
        if any(n is None for n in values):
            return None
        return sum(values)

    def limit_steps(
        self, max_steps: int | slice | None, per_file: bool = False
    ) -> Self:
        """Limit the steps. pass per_file=True to limit steps per file, otherwise limits across all files cumulatively"""

        if max_steps is None:
            return self
        if per_file:
            return type(self)({k: v.limit_steps(max_steps) for k, v in self.items()})
        else:
            from coffea.dataset_tools.manipulations import _concatenated_step_slice

            steps_by_file = _concatenated_step_slice(
                {k: v.steps for k, v in self.items()}, max_steps
            )
            new_dict = {}
            for k, v in self.items():
                if len(steps_by_file[k]) > 0:
                    new_dict[k] = v.model_dump()
                    new_dict[k]["steps"] = steps_by_file[k]
            return type(self)(new_dict)

    def limit_files(self, max_files: int | slice | None) -> Self:
        """Limit the number of files."""
        if max_files is None or (isinstance(max_files, int) and max_files >= len(self)):
            return self
        if isinstance(max_files, slice):
            valid_keys = list(self.keys())[max_files]
            new_dict = {k: v for k, v in self.items() if k in valid_keys}
        else:
            new_dict = {}
            for i, (k, v) in enumerate(self.items()):
                if i < max_files:
                    new_dict[k] = v
                else:
                    break
        return type(self)(new_dict)

    def filter_files(
        self,
        filter_name: str | None = None,
        filter_callable: Callable[[FileSpecUnion], bool] | None = None,
    ) -> Self:
        """Filter files by a regex pattern on the file names(filter_name) or callable applied to Filespecs (filter_callable)."""
        if filter_name is not None:
            regex = re.compile(filter_name)
            new_dict = {k: v for k, v in self.items() if regex.search(k)}
        else:
            new_dict = dict(self)
        if filter_callable is not None:
            new_dict = {k: v for k, v in new_dict.items() if filter_callable(v)}
        return type(self)(new_dict)


class InputFiles(
    RootModel[
        dict[
            str | pathlib.Path,
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
        for k, v in self.root.items():
            try:
                if isinstance(v, CoffeaROOTFileSpecOptional) and not isinstance(
                    v, CoffeaROOTFileSpec
                ):
                    self.root[k] = CoffeaROOTFileSpec(**v.model_dump())
                elif isinstance(v, CoffeaParquetFileSpecOptional) and not isinstance(
                    v, CoffeaParquetFileSpec
                ):
                    self.root[k] = CoffeaParquetFileSpec(**v.model_dump())
            except ValidationError:
                # not enough information to promote this file to its concrete spec; leave it as-is
                pass
        # Note: InputFiles cannot promote *itself* to PreprocessedFiles here, because pydantic
        # ignores a different class returned from an after-validator. The container-level
        # promotion (once all files are concrete) is performed in DatasetSpec.post_validate.
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
        # Values are validated against ConcreteFileSpecUnion, so they are already concrete
        # here; this defensively upgrades any Optional-typed-but-complete spec to its concrete
        # type. A spec that genuinely lacks the required fields cannot reach this point (it
        # fails ConcreteFileSpecUnion validation first); demotion to InputFiles is therefore
        # handled at the DatasetSpec level via the `PreprocessedFiles | InputFiles` union
        # fallback, not here (pydantic ignores a different class returned from an after-validator).
        for k, v in self.root.items():
            if isinstance(v, CoffeaROOTFileSpecOptional) and not isinstance(
                v, CoffeaROOTFileSpec
            ):
                self.root[k] = CoffeaROOTFileSpec(**v.model_dump())
            elif isinstance(v, CoffeaParquetFileSpecOptional) and not isinstance(
                v, CoffeaParquetFileSpec
            ):
                self.root[k] = CoffeaParquetFileSpec(**v.model_dump())
        return self


class DatasetSpec(BaseModel):
    files: PreprocessedFiles | InputFiles
    metadata: dict[Hashable, Any] = {}
    format: str | None = None
    compressed_form: str | None = None
    did: str | None = None

    def __eq__(self, other: Any) -> bool:
        # NOTE: defining __eq__ sets __hash__ to None, so DatasetSpec is unhashable. That
        # is fine only because these pydantic models are not frozen (and hence already
        # unhashable); do not freeze this model later without also restoring a __hash__.
        if not isinstance(other, DatasetSpec):
            return False
        if not all(
            getattr(self, k) == getattr(other, k)
            for k in self.__dict__.keys()
            if k != "compressed_form"
        ):
            return False
        # Compare the compressed form strings first (cheap); the compressed bytes are
        # non-deterministic even for identical decoded forms, so only fall back to
        # decoding and comparing the awkward forms when the compressed strings differ.
        if self.compressed_form == other.compressed_form:
            return True
        return self.form == other.form

    def __add__(self, other: DatasetSpec) -> DatasetSpec:
        if not isinstance(other, DatasetSpec):
            raise TypeError(
                f"Can only add DatasetSpec to DatasetSpec, got {type(other)}"
            )
        if self.did is not None and other.did is not None:
            if self.did != other.did:
                raise ValueError(
                    f"Cannot add DatasetSpec with different dids: {self.did} and {other.did}"
                )
        new_spec = self.model_dump()
        new_spec["files"] = self.files + other.files
        # merge metadata dictionaries, with other taking precedence
        new_metadata = copy.deepcopy(self.metadata)
        new_metadata.update(other.metadata)
        new_spec["metadata"] = new_metadata
        # format will be re-evaluated in post validation
        new_spec["format"] = None
        # compressed_form is not merged, set to None
        new_spec["compressed_form"] = None
        # did is not merged, set to None
        new_spec["did"] = self.did if self.did is not None else other.did
        return type(self)(**new_spec)

    def __sub__(self, other: DatasetSpec) -> DatasetSpec:
        if not isinstance(other, DatasetSpec):
            raise TypeError(
                f"Can only subtract DatasetSpec from DatasetSpec, got {type(other)}"
            )
        if self.did is not None and other.did is not None:
            if self.did != other.did:
                raise ValueError(
                    f"Cannot subtract DatasetSpec with different dids: {self.did} and {other.did}"
                )
        new_spec = self.model_dump()
        new_spec["files"] = self.files - other.files
        return type(self)(**new_spec)

    @model_validator(mode="before")
    def preprocess_data(cls, data: Any) -> Any:
        # Catch a simple case of old style input: a list of files; files will be further handled/converted in dict path if they embed the object_path
        if isinstance(data, list) and all(isinstance(f, str) for f in data):
            data = {"files": copy.deepcopy(data)}
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                if k == "form":
                    if "compressed_form" not in data:
                        # assume this was an uncompressed awkward form, try to compress it
                        if v is None:
                            new_data["compressed_form"] = None
                        else:
                            try:
                                import awkward

                                from coffea.util import compress_form

                                # Validate it's a valid form JSON by parsing it
                                _ = awkward.forms.from_json(data["form"])
                                # Compress the original JSON string
                                new_data["compressed_form"] = compress_form(
                                    data["form"]
                                )
                            except (
                                ValueError,
                                TypeError,
                                KeyError,
                            ):
                                # if we can't compress it, test if it can be decompressed
                                try:
                                    import awkward

                                    from coffea.util import decompress_form

                                    _ = awkward.forms.from_json(
                                        decompress_form(data["form"])
                                    )
                                    new_data["compressed_form"] = data["form"]
                                except (
                                    ValueError,
                                    TypeError,
                                    KeyError,
                                ):
                                    raise RuntimeError(
                                        f"form: provided form could neither be compressed nor decompressed, please provide a valid (compressed_)form, got {data['form']}"
                                    )
                    else:
                        # if there's already a compressed_form, take that and ignore the form
                        new_data["compressed_form"] = copy.deepcopy(
                            data["compressed_form"]
                        )
                elif k == "files":
                    # promote files list to dict if necessary
                    if isinstance(v, list):
                        # If files is a list, convert it to a dict and let it pass through the rest of the promotion logic
                        tmp = [f.rsplit(":", maxsplit=1) for f in v]
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
                        new_data["files"] = files
                    else:
                        new_data["files"] = copy.deepcopy(v)
                else:
                    new_data[k] = copy.deepcopy(v)
            if "files" not in new_data.keys():
                # If the structure is {filename0: object_path0, filename1: object_path1, ...}, embed as "files" key
                new_data = {"files": new_data}
            if "metadata" not in new_data.keys() or new_data["metadata"] is None:
                new_data["metadata"] = {}
        elif isinstance(data, DatasetSpec):
            new_data = data.model_dump()
        elif not isinstance(data, DatasetSpec):
            raise ValueError(
                "DatasetSpec expects a dictionary with a 'files' key DatasetSpec instance"
            )
        return new_data

    def _check_form(self) -> bool | None:
        """Check the form can be decompressed into an awkward form, if present"""
        if self.compressed_form is not None:
            # If there's a form, validate we can decompress it into an awkward form
            try:
                _ = self.form
                return True
            except (ValueError, OSError, EOFError, TypeError, KeyError):
                # base64/gzip/utf-8 decode or awkward.forms.from_json failure
                return False
        else:
            return None

    def _valid_formats(self) -> set[str]:
        _formats = {"root", "parquet"}
        return _formats

    def _valid_format(self) -> bool:
        # in the future, we may loosen the restriction to allow mixed formats in a datasetspec
        return (
            self.format in self._valid_formats()
        )  # or all(fmt in _formats for fmt in self.format.split("|"))

    def set_check_format(self) -> bool:
        """Set and/or validate the format if manually specified"""
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
        return self._valid_format()

    @model_validator(mode="after")
    def post_validate(self) -> Self:
        # check_form
        if self._check_form() is False:  # None indicates no form to check
            raise ValueError(
                "compressed_form: was not able to decompress_form into an awkward form"
            )
        # set (if necessary) and check the format
        if not self.set_check_format():
            raise ValueError(f"format: format must be one of {self._valid_formats()}")

        # Promote InputFiles to PreprocessedFiles if all files are concrete specs
        # This is needed because InputFiles.promote_and_check_files() cannot change the class of Self
        if isinstance(self.files, InputFiles) and not isinstance(
            self.files, PreprocessedFiles
        ):
            all_concrete = all(
                isinstance(v, (CoffeaROOTFileSpec, CoffeaParquetFileSpec))
                for v in self.files.values()
            )
            if all_concrete:
                try:
                    self.files = PreprocessedFiles(dict(self.files.root))
                except ValidationError:
                    # If promotion fails, keep the InputFiles
                    pass
        return self

    @property
    def joinable(self) -> bool:
        """Identify DatasetSpec criteria to be pre-joined for typetracing (necessary) and column-joining (sufficient)"""
        if self._check_form() and self.set_check_format():
            return True
        else:
            return False

    @property
    def form(self) -> str:
        if self.compressed_form is None:
            return None
        else:
            import awkward

            from coffea.util import decompress_form

            return awkward.forms.from_json(decompress_form(self.compressed_form))

    @property
    def num_entries(self) -> int | None:
        """Compute the total number of entries across all files, if available."""
        return self.files.num_entries

    @property
    def num_selected_entries(self) -> int | None:
        """Compute the total number of selected entries across all files (calculated from steps), if available."""
        return self.files.num_selected_entries

    @property
    def steps(self) -> dict[str, list[StepPair]] | None:
        """Get the steps per dataset file, if available."""
        return {k: v.steps for k, v in self.files.items()}

    def limit_steps(self, max_steps: int | slice, per_file: bool = False) -> Self:
        """Limit the steps. pass per_file=True to limit steps per file, otherwise limits across all files cumulatively"""
        spec = self.model_dump()
        spec["files"] = self.files.limit_steps(max_steps, per_file=per_file)
        return type(self)(**spec)

    def limit_files(self, max_files: int | slice | None) -> Self:
        """Limit the number of files."""
        spec = self.model_dump()
        spec["files"] = self.files.limit_files(max_files)
        return type(self)(**spec)

    def filter_files(
        self,
        filter_name: str | None = None,
        filter_callable: Callable[[FileSpecUnion], bool] | None = None,
    ) -> Self:
        """Filter files by a regex pattern on the file names(filter_name) or callable applied to Filespecs (filter_callable)."""
        spec = self.model_dump()
        spec["files"] = self.files.filter_files(
            filter_name=filter_name, filter_callable=filter_callable
        )
        return type(self)(**spec)


class DataGroupSpec(RootModel[dict[str, DatasetSpec]], MutableMapping):
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

    def __add__(self, other: DataGroupSpec) -> DataGroupSpec:
        if not isinstance(other, DataGroupSpec):
            raise TypeError(
                f"Can only add DataGroupSpec to DataGroupSpec, got {type(other)}"
            )
        new_dict = dict(self)
        for k, v in other.items():
            if k in new_dict:
                new_dict[k] = new_dict[k] + v
            else:
                new_dict[k] = v
        return type(self)(new_dict)

    def __sub__(self, other: DataGroupSpec) -> DataGroupSpec:
        if not isinstance(other, DataGroupSpec):
            raise TypeError(
                f"Can only subtract DataGroupSpec from DataGroupSpec, got {type(other)}"
            )
        new_dict = dict(self)
        for k, v in other.items():
            if k in new_dict:
                new_dict[k] = new_dict[k] - v
        return type(self)(new_dict)

    @model_validator(mode="before")
    def preprocess_data(cls, data: Any) -> Any:
        # No deepcopy is needed at this level: the DataGroupSpec branch below produces a
        # fresh dict via model_dump(), and every DatasetSpec value is deep-copied in
        # DatasetSpec.preprocess_data. Caller input is therefore never mutated, while we
        # avoid an extra full-fileset deepcopy pass on the hot preprocessing path.
        if isinstance(data, DataGroupSpec):
            return data.model_dump()
        return data

    @property
    def num_entries(self) -> int | None:
        """Total entries across all datasets, or None if any dataset's count is unknown."""
        values = [v.num_entries for v in self.root.values()]
        if any(n is None for n in values):
            return None
        return sum(values)

    @property
    def num_selected_entries(self) -> int | None:
        """Total selected entries across all datasets, or None if any is unknown."""
        values = [v.num_selected_entries for v in self.root.values()]
        if any(n is None for n in values):
            return None
        return sum(values)

    @property
    def steps(self) -> dict[str, list[StepPair]] | None:
        """Get the steps per dataset file, if available."""
        return {k: v.steps for k, v in self.items()}

    def limit_steps(self, max_steps: int | slice, per_file: bool = False) -> Self:
        """Limit the steps"""
        spec = copy.deepcopy(self)
        # handle both per_file True and False by passthrough
        for k, v in spec.items():
            spec[k] = v.limit_steps(max_steps, per_file=per_file)
        return type(self)(spec)

    def limit_files(
        self, max_files: int | slice | None, per_dataset: bool = True
    ) -> Self:
        """Limit the number of files."""
        spec = copy.deepcopy(self)
        if per_dataset:
            for k, v in spec.items():
                spec[k] = v.limit_files(max_files)
        else:
            raise NotImplementedError(
                "DataGroupSpec.limit_files with per_dataset=False is not implemented"
            )
        return type(self)(spec)

    def filter_files(
        self,
        filter_name: str | None = None,
        filter_callable: Callable[[FileSpecUnion], bool] | None = None,
    ) -> Self:
        """Filter files by a regex pattern on the file names(filter_name) or callable applied to Filespecs (filter_callable)."""
        spec = self.model_dump()
        for k, v in self.items():
            spec[k]["files"] = v.files.filter_files(
                filter_name=filter_name, filter_callable=filter_callable
            )
        return type(self)(**spec)

    def filter_datasets(
        self,
        filter_name: str | None = None,
        filter_callable: Callable[[DatasetSpec], bool] | None = None,
    ) -> Self:
        """Filter files by a regex pattern on the dataset names(filter_name) or callable applied to DatasetSpecs (filter_callable)."""
        if filter_name is not None:
            regex = re.compile(filter_name)
            new_dict = {k: v for k, v in self.items() if regex.search(k)}
        else:
            new_dict = dict(self)
        if filter_callable is not None:
            new_dict = {k: v for k, v in new_dict.items() if filter_callable(v)}
        return type(self)(new_dict)


def identify_file_format(name_or_directory: str) -> str:
    # Inspect only the final path component so directory names can't be mistaken for extensions.
    basename = name_or_directory.split("/")[-1]
    root_matches = list(re.finditer(r"\.root", basename))
    parquet_matches = list(re.finditer(r"\.(?:parq(?:uet)?|pq)", basename))
    root_pos = root_matches[-1].start() if root_matches else -1
    parquet_pos = parquet_matches[-1].start() if parquet_matches else -1
    if root_pos == -1 and parquet_pos == -1:
        if "." not in basename:
            # could be a parquet directory, would require a file opening to determine
            return "parquet"  # maybe "parquet?" to trigger further inspection?
        raise RuntimeError(
            f"identify_file_format couldn't identify if the string path is for a root file or parquet file/directory for {name_or_directory}"
        )
    # Whichever extension appears last in the name wins, so e.g. "data.root.parquet"
    # is identified as parquet rather than root.
    return "root" if root_pos > parquet_pos else "parquet"


class ModelFactory:
    def __init__(self):
        pass

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
            | DataGroupSpec
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
            elif isinstance(input, DataGroupSpec):
                return DataGroupSpec(input.model_dump())
            else:
                raise TypeError(
                    f"ModelFactory.attempt_promotion got an unexpected input type {type(input)} for input: {input}"
                )
        except ValidationError:
            # could not promote to the concrete spec; return the input unchanged
            return input

    @classmethod
    def dict_to_uprootfilespec(cls, input):
        """Convert a dictionary to a CoffeaFileSpec or CoffeaFileSpecOptional."""
        assert isinstance(input, dict), f"{input} is not a dictionary"
        try:
            return CoffeaROOTFileSpec(**input)
        except ValidationError:
            return CoffeaROOTFileSpecOptional(**input)

    @classmethod
    def dict_to_parquetfilespec(cls, input):
        """Convert a dictionary to a CoffeaParquetFileSpec or CoffeaParquetFileSpecOptional."""
        assert isinstance(input, dict), f"{input} is not a dictionary"
        try:
            return CoffeaParquetFileSpec(**input)
        except ValidationError:
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
        coerce_filespec_to_dict: bool = True,
    ) -> dict[str, Any]:
        assert isinstance(
            input, DatasetSpec
        ), f"{cls.__name__}.datasetspec_to_dict expects a DatasetSpec, got {type(input)} instead: {input}"
        if coerce_filespec_to_dict:
            return input.model_dump()
        else:
            return dict(input)
