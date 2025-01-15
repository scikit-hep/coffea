import io
import pathlib
import warnings
import weakref
from functools import partial
from types import FunctionType
from typing import Mapping

import awkward
import dask_awkward
import fsspec
import uproot

from coffea.nanoevents.mapping import (
    CachedMapping,
    ParquetSourceMapping,
    PreloadedOpener,
    PreloadedSourceMapping,
    TrivialParquetOpener,
    TrivialUprootOpener,
    UprootSourceMapping,
)
from coffea.nanoevents.schemas import BaseSchema, NanoAODSchema
from coffea.nanoevents.util import key_to_tuple, quote, tuple_to_key, unquote
from coffea.util import _remove_not_interpretable

_offsets_label = quote(",!offsets")


def _key_formatter(prefix, form_key, form, attribute):
    if attribute == "offsets":
        form_key += _offsets_label
    return prefix + f"/{attribute}/{form_key}"


class _map_schema_base:  # ImplementsFormMapping, ImplementsFormMappingInfo
    def __init__(
        self, schemaclass=BaseSchema, metadata=None, behavior=None, version=None
    ):
        self.schemaclass = schemaclass
        self.behavior = behavior
        self.metadata = metadata
        self.version = version

    def keys_for_buffer_keys(self, buffer_keys):
        base_columns = set()
        for buffer_key in buffer_keys:
            form_key, attribute = self.parse_buffer_key(buffer_key)
            operands = unquote(form_key).split(",")

            it_operands = iter(operands)
            next(it_operands)

            base_columns.update(
                [
                    name
                    for name, maybe_transform in zip(operands, it_operands)
                    if maybe_transform == "!load"
                ]
            )
        return base_columns

    def parse_buffer_key(self, buffer_key):
        prefix, attribute, form_key = buffer_key.rsplit("/", maxsplit=2)
        if attribute == "offsets":
            return (form_key[: -len(_offsets_label)], attribute)
        else:
            return (form_key, attribute)

    @property
    def buffer_key(self):
        return partial(self._key_formatter, "")

    def _key_formatter(self, prefix, form_key, form, attribute):
        if attribute == "offsets":
            form_key += _offsets_label
        return prefix + f"/{attribute}/{form_key}"


class _TranslatedMapping:
    def __init__(self, func, mapping):
        self._func = func
        self._mapping = mapping

    def __getitem__(self, index):
        return self._mapping[self._func(index)]


class _OnlySliceableAs:
    """A workaround for how PreloadedSourceMapping works"""

    def __init__(self, array, expected_slice):
        self._array = array
        self._expected_slice = expected_slice

    def __getitem__(self, s):
        if s != self._expected_slice:
            raise RuntimeError(f"Mismatched slice: {s} vs. {self._expected_slice}")
        return self._array


class _map_schema_uproot(_map_schema_base):
    def __init__(
        self, schemaclass=BaseSchema, metadata=None, behavior=None, version=None
    ):
        super().__init__(
            schemaclass=schemaclass,
            metadata=metadata,
            behavior=behavior,
            version=version,
        )

    def __call__(self, form):
        from coffea.nanoevents.mapping.uproot import _lazify_form

        branch_forms = {}
        for ifield, field in enumerate(form.fields):
            iform = form.contents[ifield].to_dict()
            branch_forms[field] = _lazify_form(
                iform, f"{field},!load", docstr=iform["parameters"]["__doc__"]
            )
        lform = {
            "class": "RecordArray",
            "contents": [item for item in branch_forms.values()],
            "fields": [key for key in branch_forms.keys()],
            "parameters": {
                "__doc__": form.parameters["__doc__"],
                "metadata": self.metadata,
            },
            "form_key": None,
        }
        return (
            awkward.forms.form.from_dict(self.schemaclass(lform, self.version).form),
            self,
        )

    def load_buffers(
        self,
        tree,
        keys,
        start,
        stop,
        decompression_executor,
        interpretation_executor,
        interp_options,
    ):
        from functools import partial

        from coffea.nanoevents.util import tuple_to_key

        partition_key = (
            str(tree.file.uuid),
            tree.object_path,
            f"{start}-{stop}",
        )
        uuidpfn = {partition_key[0]: tree.file.file_path}
        arrays = tree.arrays(
            keys,
            entry_start=start,
            entry_stop=stop,
            ak_add_doc=interp_options["ak_add_doc"],
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
            how=dict,
        )
        source_arrays = {
            k: _OnlySliceableAs(v, slice(start, stop)) for k, v in arrays.items()
        }
        mapping = PreloadedSourceMapping(
            PreloadedOpener(uuidpfn), start, stop, access_log=None
        )
        mapping.preload_column_source(partition_key[0], partition_key[1], source_arrays)

        buffer_key = partial(self._key_formatter, tuple_to_key(partition_key))

        # The buffer-keys that dask-awkward knows about will not include the
        # partition key. Therefore, we must translate the keys here.
        def translate_key(index):
            form_key, attribute = self.parse_buffer_key(index)
            return buffer_key(form_key=form_key, attribute=attribute, form=None)

        return _TranslatedMapping(translate_key, mapping)


class _map_schema_parquet(_map_schema_base):
    def __init__(
        self, schemaclass=BaseSchema, metadata=None, behavior=None, version=None
    ):
        super().__init__(
            schemaclass=schemaclass,
            metadata=metadata,
            behavior=behavior,
            version=version,
        )

    def __call__(self, form):
        # expecting a flat data source in so this is OK
        lza = awkward.Array(
            form.length_zero_array(highlevel=False), behavior=self.behavior
        )
        column_source = {key: lza[key] for key in awkward.fields(lza)}

        lform = PreloadedSourceMapping._extract_base_form(column_source)
        lform["parameters"]["metadata"] = self.metadata

        return awkward.forms.form.from_dict(self.schemaclass(lform, self.version).form)


class NanoEventsFactory:
    """
    A factory class to build NanoEvents objects.

    For most users, it is advisable to construct instances via methods like `from_root` so that
    the constructor args are properly set.
    """

    def __init__(self, schema, mapping, partition_key, cache=None, is_dask=False):
        self._is_dask = is_dask
        self._schema = schema
        self._mapping = mapping
        self._partition_key = partition_key
        self._cache = cache
        self._events = lambda: None

    def __getstate__(self):
        return {
            "schema": self._schema,
            "mapping": self._mapping,
            "partition_key": self._partition_key,
        }

    def __setstate__(self, state):
        self._schema = state["schema"]
        self._mapping = state["mapping"]
        self._partition_key = state["partition_key"]
        self._cache = None
        self._events = lambda: None

    @classmethod
    def from_root(
        cls,
        file,
        treepath=uproot._util.unset,
        entry_start=None,
        entry_stop=None,
        steps_per_file=uproot._util.unset,
        runtime_cache=None,
        persistent_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        uproot_options={},
        access_log=None,
        iteritems_options={},
        use_ak_forth=True,
        delayed=True,
        known_base_form=None,
        decompression_executor=None,
        interpretation_executor=None,
    ):
        """Quickly build NanoEvents from a root file

        Parameters
        ----------
            file : a string or dict input to ``uproot.open()`` or ``uproot.dask()`` or a ``uproot.reading.ReadOnlyDirectory``
                The filename or dict of filenames including the treepath (as it would be passed directly to ``uproot.open()``
                or ``uproot.dask()``) already opened file using e.g. ``uproot.open()``.
            treepath : str, optional
                Name of the tree to read in the file. Used only if ``file`` is a ``uproot.reading.ReadOnlyDirectory``.
            entry_start : int, optional (eager mode only)
                Start at this entry offset in the tree (default 0)
            entry_stop : int, optional (eager mode only)
                Stop at this entry offset in the tree (default end of tree)
            steps_per_file: int, optional
                Partition files into this many steps (previously "chunks")
            runtime_cache : dict, optional
                A dict-like interface to a cache object. This cache is expected to last the
                duration of the program only, and will be used to hold references to materialized
                awkward arrays, etc.
            persistent_cache : dict, optional
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict, optional
                Arbitrary metadata to add to the `base.NanoEvents` object
            uproot_options : dict, optional
                Any options to pass to ``uproot.open`` or ``uproot.dask``
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance
            use_ak_forth:
                Toggle using awkward_forth to interpret branches in root file.
            delayed:
                Nanoevents will use dask as a backend to construct a delayed task graph representing your analysis.
            known_base_form:
                If the base form of the input file is known ahead of time we can skip opening a single file and parsing metadata.
            decompression_executor (None or Executor with a ``submit`` method):
                see: https://github.com/scikit-hep/uproot5/blob/main/src/uproot/_dask.py#L109
            interpretation_executor (None or Executor with a ``submit`` method):
                see: https://github.com/scikit-hep/uproot5/blob/main/src/uproot/_dask.py#L113

        Returns
        -------
            out: NanoEventsFactory
                A NanoEventsFactory instance built from the file at `file`.
        """

        if treepath is not uproot._util.unset and not isinstance(
            file, uproot.reading.ReadOnlyDirectory
        ):
            raise ValueError(
                """Specification of treename by argument to from_root is no longer supported in coffea 2023.
            Please use one of the allowed types for "files" specified by uproot: https://github.com/scikit-hep/uproot5/blob/v5.1.2/src/uproot/_dask.py#L109-L132
            """
            )

        if delayed and steps_per_file is not uproot._util.unset:
            warnings.warn(
                f"""You have set steps_per_file to {steps_per_file}, this should only be used for a
                small number of inputs (e.g. for early-stage/exploratory analysis) since it does not
                inform dask of each chunk lengths at creation time, which can cause unexpected
                slowdowns at scale. If you would like to process larger datasets please specify steps
                using the appropriate uproot "files" specification:
                    https://github.com/scikit-hep/uproot5/blob/v5.1.2/src/uproot/_dask.py#L109-L132.
                """,
                RuntimeWarning,
            )

        if (
            delayed
            and not isinstance(schemaclass, FunctionType)
            and schemaclass.__dask_capable__
        ):
            map_schema = _map_schema_uproot(
                schemaclass=schemaclass,
                behavior=dict(schemaclass.behavior()),
                metadata=metadata,
                version="latest",
            )

            to_open = file
            if isinstance(file, uproot.reading.ReadOnlyDirectory):
                to_open = file[treepath]

            opener = partial(
                uproot.dask,
                to_open,
                full_paths=True,
                open_files=False,
                ak_add_doc=True,
                filter_branch=_remove_not_interpretable,
                steps_per_file=steps_per_file,
                known_base_form=known_base_form,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                **uproot_options,
            )

            return cls(map_schema, opener, None, cache=None, is_dask=True)
        elif delayed and not schemaclass.__dask_capable__:
            warnings.warn(
                f"{schemaclass} is not dask capable despite requesting delayed mode, generating non-dask nanoevents",
                RuntimeWarning,
            )

        if isinstance(file, uproot.reading.ReadOnlyDirectory):
            tree = file[treepath]
        elif "<class 'uproot.rootio.ROOTDirectory'>" == str(type(file)):
            raise RuntimeError(
                "The file instance (%r) is an uproot3 type, but this module is only compatible with uproot5 or higher"
                % file
            )
        else:
            tree = uproot.open(file, **uproot_options)

        if entry_start is None or entry_start < 0:
            entry_start = 0
        if entry_stop is None or entry_stop > tree.num_entries:
            entry_stop = tree.num_entries

        partition_key = (
            str(tree.file.uuid),
            tree.object_path,
            f"{entry_start}-{entry_stop}",
        )
        uuidpfn = {partition_key[0]: tree.file.file_path}
        mapping = UprootSourceMapping(
            TrivialUprootOpener(uuidpfn, uproot_options),
            entry_start,
            entry_stop,
            cache={},
            access_log=access_log,
            use_ak_forth=use_ak_forth,
        )
        mapping.preload_column_source(partition_key[0], partition_key[1], tree)

        base_form = mapping._extract_base_form(
            tree, iteritems_options=iteritems_options
        )

        return cls._from_mapping(
            mapping,
            partition_key,
            base_form,
            runtime_cache,
            persistent_cache,
            schemaclass,
            metadata,
        )

    @classmethod
    def from_parquet(
        cls,
        file,
        treepath=uproot._util.unset,
        entry_start=None,
        entry_stop=None,
        runtime_cache=None,
        persistent_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        parquet_options={},
        skyhook_options={},
        access_log=None,
        delayed=True,
    ):
        """Quickly build NanoEvents from a parquet file

        Parameters
        ----------
            file : str, pathlib.Path, pyarrow.NativeFile, or python file-like
                The filename or already opened file using e.g. ``uproot.open()``
            treepath : str, optional
                Name of the tree to read in the file
            entry_start : int, optional
                Start at this entry offset in the tree (default 0)
            entry_stop : int, optional
                Stop at this entry offset in the tree (default end of tree)
            runtime_cache : dict, optional
                A dict-like interface to a cache object. This cache is expected to last the
                duration of the program only, and will be used to hold references to materialized
                awkward arrays, etc.
            persistent_cache : dict, optional
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict, optional
                Arbitrary metadata to add to the `base.NanoEvents` object
            parquet_options : dict, optional
                Any options to pass to ``pyarrow.parquet.ParquetFile``
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance
            delayed:
                Nanoevents will use dask as a backend to construct a delayed task graph representing your analysis.

        Returns
        -------
            out: NanoEventsFactory
                A NanoEventsFactory instance built from the file at `file`.
        """
        import pyarrow
        import pyarrow.dataset as ds
        import pyarrow.parquet

        ftypes = (
            pathlib.Path,
            pyarrow.NativeFile,
            io.TextIOBase,
            io.BufferedIOBase,
            io.RawIOBase,
            io.IOBase,
        )

        if (
            delayed
            and not isinstance(schemaclass, FunctionType)
            and schemaclass.__dask_capable__
        ):
            map_schema = _map_schema_parquet(
                schemaclass=schemaclass,
                behavior=dict(schemaclass.behavior()),
                metadata=metadata,
                version="latest",
            )

            if isinstance(file, ftypes + (str,)):
                opener = partial(
                    dask_awkward.from_parquet,
                    file,
                )
            else:
                raise TypeError("Invalid file type (%s)" % (str(type(file))))
            return cls(map_schema, opener, None, cache=None, is_dask=True)
        elif delayed and not schemaclass.__dask_capable__:
            warnings.warn(
                f"{schemaclass} is not dask capable despite allowing dask, generating non-dask nanoevents"
            )

        if isinstance(file, ftypes):
            table_file = pyarrow.parquet.ParquetFile(file, **parquet_options)
        elif isinstance(file, str):
            fs_file = fsspec.open(
                file, "rb"
            ).open()  # Call open to materialize the file
            table_file = pyarrow.parquet.ParquetFile(fs_file, **parquet_options)
        elif isinstance(file, pyarrow.parquet.ParquetFile):
            table_file = file
        else:
            raise TypeError("Invalid file type (%s)" % (str(type(file))))

        if entry_start is None or entry_start < 0:
            entry_start = 0
        if entry_stop is None or entry_stop > table_file.metadata.num_rows:
            entry_stop = table_file.metadata.num_rows

        pqmeta = table_file.schema_arrow.metadata
        pquuid = None if pqmeta is None else pqmeta.get(b"uuid", None)
        pqobj_path = None if pqmeta is None else pqmeta.get(b"object_path", None)

        partition_key = (
            str(None) if pquuid is None else pquuid.decode("ascii"),
            str(None) if pqobj_path is None else pqobj_path.decode("ascii"),
            f"{entry_start}-{entry_stop}",
        )
        uuidpfn = {partition_key[0]: pqobj_path}
        mapping = ParquetSourceMapping(
            TrivialParquetOpener(uuidpfn, parquet_options),
            entry_start,
            entry_stop,
            access_log=access_log,
        )

        format_ = "parquet"
        dataset = None
        shim = None
        if len(skyhook_options) > 0:
            format_ = ds.SkyhookFileFormat(
                "parquet",
                skyhook_options["ceph_config_path"].encode(),
                skyhook_options["ceph_data_pool"].encode(),
            )
            dataset = ds.dataset(file, schema=table_file.schema_arrow, format=format_)
            shim = TrivialParquetOpener.UprootLikeShim(file, dataset)
        else:
            shim = TrivialParquetOpener.UprootLikeShim(
                table_file, dataset, openfile=fs_file
            )

        mapping.preload_column_source(partition_key[0], partition_key[1], shim)

        base_form = mapping._extract_base_form(table_file.schema_arrow)

        return cls._from_mapping(
            mapping,
            partition_key,
            base_form,
            runtime_cache,
            persistent_cache,
            schemaclass,
            metadata,
        )

    @classmethod
    def from_preloaded(
        cls,
        array_source,
        entry_start=None,
        entry_stop=None,
        runtime_cache=None,
        persistent_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        access_log=None,
    ):
        """Quickly build NanoEvents from a pre-loaded array source

        Parameters
        ----------
            array_source : Mapping[str, awkward.Array]
                A mapping of names to awkward arrays, it must have a metadata attribute with uuid,
                num_rows, and path sub-items.
            entry_start : int, optional
                Start at this entry offset in the tree (default 0)
            entry_stop : int, optional
                Stop at this entry offset in the tree (default end of tree)
            runtime_cache : dict, optional
                A dict-like interface to a cache object. This cache is expected to last the
                duration of the program only, and will be used to hold references to materialized
                awkward arrays, etc.
            persistent_cache : dict, optional
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict, optional
                Arbitrary metadata to add to the `base.NanoEvents` object
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance

        Returns
        -------
            out: NanoEventsFactory
                A NanoEventsFactory instance built from information in `array_source`.
        """
        if not isinstance(array_source, Mapping):
            raise TypeError(
                "Invalid array source type (%s)" % (str(type(array_source)))
            )
        if not hasattr(array_source, "metadata"):
            raise TypeError(
                "array_source must have 'metadata' with uuid, num_rows, and object_path"
            )

        if entry_start is None or entry_start < 0:
            entry_start = 0
        if entry_stop is None or entry_stop > array_source.metadata["num_rows"]:
            entry_stop = array_source.metadata["num_rows"]

        uuid = array_source.metadata["uuid"]
        obj_path = array_source.metadata["object_path"]

        partition_key = (
            str(uuid),
            obj_path,
            f"{entry_start}-{entry_stop}",
        )
        uuidpfn = {uuid: array_source}
        mapping = PreloadedSourceMapping(
            PreloadedOpener(uuidpfn), entry_start, entry_stop, access_log=access_log
        )
        mapping.preload_column_source(partition_key[0], partition_key[1], array_source)

        base_form = mapping._extract_base_form(array_source)

        return cls._from_mapping(
            mapping,
            partition_key,
            base_form,
            runtime_cache,
            persistent_cache,
            schemaclass,
            metadata,
        )

    @classmethod
    def _from_mapping(
        cls,
        mapping,
        partition_key,
        base_form,
        runtime_cache,
        persistent_cache,
        schemaclass,
        metadata,
    ):
        """Quickly build NanoEvents from a root file

        Parameters
        ----------
            mapping : Mapping
                The mapping of a column_source to columns.
            partition_key : tuple
                Basic information about the column source, uuid, paths.
            base_form : dict
                The awkward form describing the nanoevents interpretation of the mapped file.
            runtime_cache : dict
                A dict-like interface to a cache object. This cache is expected to last the
                duration of the program only, and will be used to hold references to materialized
                awkward arrays, etc.
            persistent_cache : dict
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict
                Arbitrary metadata to add to the `base.NanoEvents` object

        """
        if persistent_cache is not None:
            mapping = CachedMapping(persistent_cache, mapping)
        if metadata is not None:
            base_form["parameters"]["metadata"] = metadata
        if not callable(schemaclass):
            raise ValueError("Invalid schemaclass type")
        schema = schemaclass(base_form)
        if not isinstance(schema, BaseSchema):
            raise RuntimeError("Invalid schema type")
        return cls(
            schema,
            mapping,
            tuple_to_key(partition_key),
            cache=runtime_cache,
            is_dask=False,
        )

    def __len__(self):
        uuid, treepath, entryrange = key_to_tuple(self._partition_key)
        start, stop = (int(x) for x in entryrange.split("-"))
        return stop - start

    def events(self):
        """
        Build events

        Returns
        -------
            out:
                If the NanoEventsFactory is running in delayed mode (Dask), this is
                a Dask awkward array of the events. If the mapping also produces a
                report, the output will be a tuple (events, report).
                If the factory is not running in delayed mode, this is an awkward
                array of the events.
        """
        if self._is_dask:
            events = self._mapping(form_mapping=self._schema)
            report = None
            if isinstance(events, tuple):
                events, report = events
            events._meta.attrs["@original_array"] = events
            if report is not None:
                return events, report
            return events

        events = self._events()
        if events is None:
            events = awkward.from_buffers(
                self._schema.form,
                len(self),
                self._mapping,
                buffer_key=partial(_key_formatter, self._partition_key),
                behavior=self._schema.behavior(),
                attrs={"@events_factory": self},
            )
            self._events = weakref.ref(events)

        return events
