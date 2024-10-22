import json
import warnings

import awkward
import numpy
from fsspec.core import OpenFile

from coffea.nanoevents.mapping.base import BaseSourceMapping, UUIDOpener
from coffea.nanoevents.util import quote, tuple_to_key


# IMPORTANT -> For now the uuid is just the uuid of the pfn.
#              Later we should use the ParquetFile common_metadata to populate.
class TrivialParquetOpener(UUIDOpener):
    class UprootLikeShim:
        def __init__(self, file, dataset=None, openfile: OpenFile = None):
            """
            Shim to allow uproot to read parquet files via pyArrow
            :param file: Open ParquetReader handle
            :param dataset: Optional dataset to support SkyHook
            :param openfile: If the source for the Parquet used fsspec then we may need to
                             explicitly close the OpenFile instance to clean up any
                             materialized copies.
            """
            self.file = file
            self.dataset = dataset
            self.openfile = openfile

        def __del__(self):
            """
            If we used fsspec to open the ParquetReader then there may be a
            materialized view of the file floating around. Make sure we close it to
            remove it from the local drive
            """
            if self.openfile:
                self.openfile.close()

        def read(self, column_name):
            # make sure uproot is single-core since our calling context might not be
            if self.dataset is not None:
                return self.dataset.to_table(use_threads=False, columns=[column_name])
            else:
                return self.file.read([column_name], use_threads=False)

        # for right now spoof the notion of directories in files
        # parquet can do it but we've gotta convince people to
        # use it like that
        def __getitem__(self, name):
            return self.file

    def __init__(self, uuid_pfnmap, parquet_options={}):
        super().__init__(uuid_pfnmap)
        self._parquet_options = parquet_options
        self._schema_map = None

    def open_uuid(self, uuid):
        import pyarrow.parquet as pq

        pfn = self._uuid_pfnmap[uuid]
        parfile = pq.ParquetFile(pfn, **self._parquet_options)
        pqmeta = parfile.schema_arrow.metadata
        pquuid = None if pqmeta is None else pqmeta.get(b"uuid", None)
        pfn = None if pqmeta is None else pqmeta.get(b"url", None)
        if str(pquuid) != uuid:
            raise RuntimeError(
                f"UUID of file {pfn} does not match expected value ({uuid})"
            )
        return TrivialParquetOpener.UprootLikeShim(parfile)


def arrow_schema_to_awkward_form(schema):
    import pyarrow as pa

    if isinstance(schema, (pa.lib.ListType, pa.lib.LargeListType)):
        dtype = schema.value_type.to_pandas_dtype()()
        return awkward.forms.ListOffsetForm(
            offsets="i64",
            content=awkward.forms.NumpyForm(
                inner_shape=[],
                primitive=dtype.dtype.name,
            ),
        )
    elif isinstance(schema, pa.lib.DataType):
        dtype = schema.to_pandas_dtype()()
        return awkward.forms.NumpyForm(
            inner_shape=[],
            primitive=dtype.dtype.name,
        )
    else:
        raise Exception("Unrecognized pyarrow array type")
    return None


class ParquetSourceMapping(BaseSourceMapping):
    _debug = False

    class UprootLikeShim:
        def __init__(self, source, column):
            self.source = source
            self.column = column

        def array(self, entry_start, entry_stop):
            import pyarrow as pa

            aspa = self.source.read(self.column)[entry_start:entry_stop][0].chunk(0)
            out = None
            if isinstance(aspa, (pa.lib.ListArray, pa.lib.LargeListArray)):
                value_type = aspa.type.value_type
                offsets = None
                if isinstance(aspa, pa.lib.LargeListArray):
                    offsets = numpy.frombuffer(aspa.buffers()[1], dtype=numpy.int64)[
                        : len(aspa) + 1
                    ]
                else:
                    offsets = numpy.frombuffer(aspa.buffers()[1], dtype=numpy.int32)[
                        : len(aspa) + 1
                    ]
                    offsets = offsets.astype(numpy.int64)
                offsets = awkward.index.Index64(offsets)

                if not isinstance(value_type, pa.lib.DataType):
                    raise Exception(
                        "arrow only accepts single jagged arrays for now..."
                    )
                flat = aspa.flatten()
                content = awkward.contents.NumpyArray(flat)
                out = awkward.contents.ListOffsetArray(offsets, content)
            elif isinstance(aspa, (pa.lib.NumericArray, pa.lib.BooleanArray)):
                out = awkward.contents.NumpyArray(aspa)
            else:
                raise Exception("array is not flat array or jagged list")
            return awkward.Array(out)

    def __init__(self, fileopener, start, stop, cache=None, access_log=None):
        super().__init__(fileopener, start, stop, cache, access_log)

    @classmethod
    def _extract_base_form(cls, arrow_schema):
        column_forms = {}
        for field in arrow_schema:
            key = field.name
            fmeta = {} if field.metadata is None else field.metadata

            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue

            form = None
            if b"form" in fmeta:
                form = json.loads(fmeta[b"form"])
            else:
                schema = field.type
                form = arrow_schema_to_awkward_form(schema)
                form = json.loads(form.to_json())

            if (
                form["class"].startswith("ListOffset")
                and form["content"]["class"] == "NumpyArray"  # noqa
            ):
                form["form_key"] = quote(f"{key},!load")
                form["content"]["form_key"] = quote(f"{key},!load,!content")
                if b"title" in fmeta:
                    form["content"]["parameters"] = {
                        "__doc__": fmeta[b"title"].decode()
                    }
                elif "__doc__" not in form["content"]["parameters"]:
                    form["content"]["parameters"] = {"__doc__": key}
            elif form["class"] == "NumpyArray":
                form["form_key"] = quote(f"{key},!load")
                if b"title" in fmeta:
                    form["parameters"] = {"__doc__": fmeta[b"title"].decode()}
                elif "__doc__" not in form["parameters"]:
                    form["parameters"] = {"__doc__": key}
            else:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents"
                )
                continue
            column_forms[key] = form
        return {
            "class": "RecordArray",
            "contents": [item for item in column_forms.values()],
            "fields": [key for key in column_forms.keys()],
            "parameters": {"__doc__": "parquetfile"},
            "form_key": "",
        }

    def key_root(self):
        return "ParquetSourceMapping:"

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    def get_column_handle(self, columnsource, name):
        return ParquetSourceMapping.UprootLikeShim(columnsource, name)

    def extract_column(self, columnhandle, start, stop, **kwargs):
        return columnhandle.array(entry_start=start, entry_stop=stop)

    def __len__(self):
        return self._stop - self._start

    def __iter__(self):
        raise NotImplementedError
