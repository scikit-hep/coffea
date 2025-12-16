import inspect
import io
import pathlib
import warnings
import weakref
from collections.abc import Mapping
from functools import partial
from types import FunctionType

import awkward
import dask_awkward
import fsspec
import uproot

from coffea.nanoevents.mapping import (
    ParquetSourceMapping,
    PreloadedOpener,
    PreloadedSourceMapping,
    TrivialParquetOpener,
    TrivialUprootOpener,
    UprootSourceMapping,
)
from coffea.nanoevents.schemas import BaseSchema, NanoAODSchema
from coffea.nanoevents.util import key_to_tuple, quote, tuple_to_key, unquote
from coffea.util import _is_interpretable

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
                iform,
                f"{field},!load",
                docstr=iform["parameters"]["__doc__"],
                typestr=iform["parameters"]["typename"],
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


_allowed_modes = frozenset(["eager", "virtual", "dask"])


class NanoEventsFactory:
    """
    A factory class to build NanoEvents objects.

    For most users, it is advisable to construct instances via methods like `from_root` so that
    the constructor args are properly set.
    """

    def __init__(self, schema, mapping, partition_key, mode="eager"):
        if mode not in _allowed_modes:
            raise ValueError(f"Invalid mode {mode}, valid modes are {_allowed_modes}")
        self._mode = mode
        self._schema = schema
        self._mapping = mapping
        self._partition_key = partition_key
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
        self._events = lambda: None

    @classmethod
    def from_root(
        cls,
        file,
        *,
        mode="virtual",
        treepath=uproot._util.unset,
        entry_start=None,
        entry_stop=None,
        steps_per_file=uproot._util.unset,
        preload=None,
        buffer_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        uproot_options={},
        iteritems_options={},
        access_log=None,
        use_ak_forth=True,
        known_base_form=None,
        decompression_executor=None,
        interpretation_executor=None,
        delayed=uproot._util.unset,
    ):
        """Quickly build NanoEvents from a root file

        Parameters
        ----------
            file : a string or dict input to ``uproot.open()`` or ``uproot.dask()`` or a ``uproot.reading.ReadOnlyDirectory``
                The filename or dict of filenames including the treepath (as it would be passed directly to ``uproot.open()``
                or ``uproot.dask()``) already opened file using e.g. ``uproot.open()``.
            mode:
                Nanoevents will use "eager", "virtual", or "dask" as a backend.
            treepath : str, optional
                Name of the tree to read in the file. Used only if ``file`` is a ``uproot.reading.ReadOnlyDirectory``.
            entry_start : int, optional (eager and virtual mode only)
                Start at this entry offset in the tree (default 0)
            entry_stop : int, optional (eager and virtual mode only)
                Stop at this entry offset in the tree (default end of tree)
            steps_per_file: int, optional
                Partition files into this many steps (previously "chunks")
            preload (None or Callable):
                A function to call to preload specific branches/columns in bulk. Only works in eager and virtual mode.
                Passed to ``tree.arrays`` as the ``filter_branch`` argument to filter branches to be preloaded.
            buffer_cache : dict, optional
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict, optional
                Arbitrary metadata to add to the `base.NanoEvents` object
            uproot_options : dict, optional
                Any options to pass to ``uproot.open`` or ``uproot.dask``
            iteritems_options : dict, optional (eager and virtual mode only)
                Any options to pass to ``tree.iteritems`` when iterating over the tree's branches to extract the form.
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance
            use_ak_forth : bool, default True
                Toggle using awkward_forth to interpret branches in the ROOT file.
            known_base_form : dict or None, optional
                If the base form of the input file is known ahead of time we can skip opening a single file and parsing metadata.
            decompression_executor : Any, optional
                Executor with a ``submit`` method used for decompression tasks. See
                https://github.com/scikit-hep/uproot5/blob/main/src/uproot/_dask.py#L109.
            interpretation_executor : Any, optional
                Executor with a ``submit`` method used for interpretation tasks. See
                https://github.com/scikit-hep/uproot5/blob/main/src/uproot/_dask.py#L113.

        Returns
        -------
            NanoEventsFactory
                Factory configured from ``file`` that can materialise NanoEvents.
        """
        if delayed is not uproot._util.unset:
            msg = """
            NanoEventsFactory.from_root() behavior has changed.
            The default behavior is that now it reads the input root file using
            the newly developed virtual arrays backend of awkward instead of dask.
            The backend choice is controlled by the ``mode`` argument of the method
            which can be set to "eager", "virtual", or "dask".
            The new default is "virtual" while the `delayed` argument has been removed.
            The old `delayed=True` is now equivalent to `mode="dask"`.
            The old `delayed=False` is now equivalent to `mode="eager"`.
            """
            raise TypeError(inspect.cleandoc(msg))

        if treepath is not uproot._util.unset and not isinstance(
            file, uproot.reading.ReadOnlyDirectory
        ):
            raise ValueError(
                """Specification of treename by argument to from_root is no longer supported in coffea 2023.
            Please use one of the allowed types for "files" specified by uproot: https://github.com/scikit-hep/uproot5/blob/v5.1.2/src/uproot/_dask.py#L109-L132
            """
            )

        if mode not in _allowed_modes:
            raise ValueError(f"Invalid mode {mode}, valid modes are {_allowed_modes}")

        if mode == "dask" and steps_per_file is not uproot._util.unset:
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
            mode == "dask"
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
                ak_add_doc={"__doc__": "title", "typename": "typename"},
                filter_branch=_is_interpretable,
                steps_per_file=steps_per_file,
                known_base_form=known_base_form,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                **uproot_options,
            )

            return cls(map_schema, opener, None, mode="dask")
        elif mode == "dask" and not schemaclass.__dask_capable__:
            warnings.warn(
                f"{schemaclass} is not dask capable despite requesting dask mode, generating non-dask nanoevents",
                RuntimeWarning,
            )
            # fall through to virtual mode
            mode = "virtual"

        if isinstance(file, uproot.reading.ReadOnlyDirectory):
            tree = file[treepath]
            file_handle = file
        elif "<class 'uproot.rootio.ROOTDirectory'>" == str(type(file)):
            raise RuntimeError(
                "The file instance (%r) is an uproot3 type, but this module is only compatible with uproot5 or higher"
                % file
            )
        else:
            tree = uproot.open(file, **uproot_options)
            file_handle = tree.file

        # Get the typenames
        typenames = tree.typenames()

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

        preloaded_arrays = None
        if preload is not None:
            preloaded_arrays = tree.arrays(
                filter_branch=preload,
                entry_start=entry_start,
                entry_stop=entry_stop,
                ak_add_doc=True,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                how=dict,
            )
            # this ensures that the preloaded arrays are only sliced as they are supposed to be
            preloaded_arrays = {
                k: _OnlySliceableAs(v, slice(entry_start, entry_stop))
                for k, v in preloaded_arrays.items()
            }

        mapping = UprootSourceMapping(
            TrivialUprootOpener(uuidpfn, uproot_options),
            entry_start,
            entry_stop,
            cache={},
            access_log=access_log,
            file_handle=file_handle,
            use_ak_forth=use_ak_forth,
            virtual=mode == "virtual",
            preloaded_arrays=preloaded_arrays,
            buffer_cache=buffer_cache,
        )
        mapping.preload_column_source(partition_key[0], partition_key[1], tree)

        base_form = mapping._extract_base_form(
            tree, iteritems_options=iteritems_options
        )
        base_form["typenames"] = typenames

        return cls._from_mapping(
            mapping,
            partition_key,
            base_form,
            buffer_cache,
            schemaclass,
            metadata,
            mode=mode,
        )

    @classmethod
    def from_parquet(
        cls,
        file,
        *,
        mode="virtual",
        entry_start=None,
        entry_stop=None,
        buffer_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        parquet_options={},
        storage_options=None,
        skyhook_options={},
        access_log=None,
    ):
        """Quickly build NanoEvents from a parquet file

        Parameters
        ----------
            file : str or pathlib.Path or pyarrow.NativeFile or io.IOBase
                The filename or already opened file using e.g. ``pyarrow.NativeFile()``.
            mode : {"eager", "virtual", "dask"}, default "virtual"
                Backend to use when interpreting parquet data.
            entry_start : int or None, optional
                Starting entry (only used in eager or virtual mode). Defaults to ``0``.
            entry_stop : int or None, optional
                Stopping entry (only used in eager or virtual mode). Defaults to end of dataset.
            buffer_cache : dict, optional
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict, optional
                Arbitrary metadata to add to the `base.NanoEvents` object
            parquet_options : dict, optional
                Any options to pass to ``pyarrow.parquet.ParquetFile``
            storage_options : dict, optional
                Options to pass to ``fsspec`` when opening the file. Only used when ``file`` is a string path.
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance

        Returns
        -------
            NanoEventsFactory
                Factory configured from ``file`` that can materialise NanoEvents.
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

        if mode not in _allowed_modes:
            raise ValueError(f"Invalid mode {mode}, valid modes are {_allowed_modes}")

        if (
            mode == "dask"
            and not isinstance(schemaclass, FunctionType)
            and schemaclass.__dask_capable__
        ):
            map_schema = _map_schema_parquet(
                schemaclass=schemaclass,
                behavior=dict(schemaclass.behavior()),
                metadata=metadata,
                version="latest",
            )
            if isinstance(file, ftypes + (str,)) or (
                isinstance(file, list)
                and all(isinstance(f, ftypes + (str,)) for f in file)
            ):
                opener = partial(
                    dask_awkward.from_parquet,
                    file,
                )
            else:
                raise TypeError(
                    f"Invalid file type ({str(type(file))}) for file {file}"
                )
            # Form should be applied appropriately, but this requires a hook into dask-awkward or new schema-builder
            raise NotImplementedError(
                "Dask-awkward does not yet support lazy loading of parquet files with a schema"
            )
            return cls(map_schema, opener, None, mode="dask")
        elif mode == "dask" and not schemaclass.__dask_capable__:
            warnings.warn(
                f"{schemaclass} is not dask capable despite allowing dask, generating non-dask nanoevents"
            )
        if isinstance(file, ftypes):
            table_file = pyarrow.parquet.ParquetFile(file, **parquet_options)
        elif isinstance(file, str):
            fs_file = fsspec.open(
                file, "rb", **(storage_options or {})
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
            virtual=mode == "virtual",
            buffer_cache=buffer_cache,
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
            buffer_cache,
            schemaclass,
            metadata,
            mode,
        )

    @classmethod
    def from_preloaded(
        cls,
        array_source,
        *,
        entry_start=None,
        entry_stop=None,
        buffer_cache=None,
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
            entry_start : int or None, optional
                Start index for slicing the array source. Defaults to ``0``.
            entry_stop : int or None, optional
                Stop index for slicing the array source. Defaults to the full length.
            buffer_cache : dict, optional
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
            NanoEventsFactory
                Factory configured from ``array_source`` that can materialise NanoEvents.
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
            buffer_cache,
            schemaclass,
            metadata,
            mode="eager",
        )

    @classmethod
    def _from_mapping(
        cls,
        mapping,
        partition_key,
        base_form,
        buffer_cache,
        schemaclass,
        metadata,
        mode,
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
            buffer_cache : dict
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict
                Arbitrary metadata to add to the `base.NanoEvents` object
            mode:
                Nanoevents will use "eager", "virtual", or "dask" as a backend.

        """
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
            mode=mode,
        )

    def __len__(self):
        uuid, treepath, entryrange = key_to_tuple(self._partition_key)
        start, stop = (int(x) for x in entryrange.split("-"))
        return stop - start

    @property
    def access_log(self):
        """List of accessed branches, populated when columns are lazily loaded."""
        return getattr(self._mapping, "_access_log", None)

    @property
    def file_handle(self):
        """The file handle used to open the source file, if available."""
        return getattr(self._mapping, "_file_handle", None)

    def events(self):
        """
        Build events

        Returns
        -------
            awkward.Array or dask_awkward.Array or tuple
                Events materialised according to the configured backend. In ``\"dask\"``
                mode a ``dask_awkward.Array`` is returned (optionally paired with a
                report). In ``\"virtual\"`` or ``\"eager\"`` mode an ``awkward.Array`` is
                returned.
        """
        if self._mode == "dask":
            dask_awkward.lib.core.dak_cache.clear()
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
            form = self._schema.form
            buffer_key = partial(_key_formatter, self._partition_key)
            events = awkward.from_buffers(
                form=form,
                length=len(self),
                container=self._mapping,
                buffer_key=buffer_key,
                backend="cpu",
                byteorder=awkward._util.native_byteorder,
                allow_noncanonical_form=False,
                highlevel=True,
                behavior=self._schema.behavior(),
                attrs={
                    "@events_factory": self,
                    "@form": form,
                    "@buffer_key": buffer_key,
                },
            )
            self._events = weakref.ref(events)

        return events
