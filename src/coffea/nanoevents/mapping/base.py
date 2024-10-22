from abc import abstractmethod
from collections.abc import Mapping

import numpy
from cachetools import LRUCache

from coffea.nanoevents import transforms
from coffea.nanoevents.util import key_to_tuple, tuple_to_key


class UUIDOpener:
    def __init__(self, uuid_pfnmap):
        self._uuid_pfnmap = uuid_pfnmap

    @abstractmethod
    def open_uuid(self, uuid):
        pass


class BaseSourceMapping(Mapping):
    _debug = False

    def __init__(
        self, fileopener, start, stop, cache=None, access_log=None, use_ak_forth=False
    ):
        self._fileopener = fileopener
        self._cache = cache
        self._access_log = access_log
        self._start = start
        self._stop = stop
        self._use_ak_forth = use_ak_forth
        self.setup()

    def setup(self):
        if self._cache is None:
            self._cache = LRUCache(1)

    @classmethod
    @abstractmethod
    def _extract_base_form(cls, source):
        pass

    def _column_source(self, uuid, path_in_source):
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        try:
            return self._cache[key]
        except KeyError:
            pass
        source = self._fileopener.open_uuid(uuid)[path_in_source]
        self._cache[key] = source
        return source

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory._from_mapping"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    @abstractmethod
    def get_column_handle(self, columnsource, name, allow_missing):
        pass

    @abstractmethod
    def extract_column(self, columnhandle, start, stop, allow_missing, **kwargs):
        pass

    @classmethod
    def interpret_key(cls, key):
        uuid, treepath, entryrange, partition, form_key, *layoutattr = key_to_tuple(key)
        start, stop = (int(x) for x in entryrange.split("-"))
        nodes = form_key.split(",")
        if len(layoutattr) == 1:
            nodes.append("!" + layoutattr[0])
        elif len(layoutattr) > 1:
            raise RuntimeError(f"Malformed key: {key}")
        return uuid, treepath, start, stop, nodes

    def __getitem__(self, key):
        uuid, treepath, start, stop, nodes = self.interpret_key(key)
        if self._debug:
            print("Getting (", key, ") :", uuid, treepath, start, stop, nodes)
        stack = []
        skip = False
        for node in nodes:
            if skip:
                skip = False
                continue
            elif node == "!skip":
                skip = True
                continue
            elif node.startswith("!load"):
                handle_name = stack.pop()
                if self._access_log is not None:
                    self._access_log.append(handle_name)
                allow_missing = node == "!loadallowmissing"
                handle = self.get_column_handle(
                    self._column_source(uuid, treepath), handle_name, allow_missing
                )
                stack.append(
                    self.extract_column(
                        handle,
                        start,
                        stop,
                        allow_missing,
                        use_ak_forth=self._use_ak_forth,
                    )
                )
            elif node.startswith("!"):
                tname = node[1:]
                if not hasattr(transforms, tname):
                    raise RuntimeError(
                        f"Syntax error in form_key: no transform named {tname}"
                    )
                getattr(transforms, tname)(stack)
            else:
                stack.append(node)
        if len(stack) != 1:
            raise RuntimeError(f"Syntax error in form key {nodes}")
        out = stack.pop()
        import awkward

        if isinstance(out, awkward.contents.Content):
            out = awkward.to_numpy(out)
        else:
            try:
                out = numpy.array(out)
            except ValueError:
                if self._debug:
                    print(out)
                raise RuntimeError(
                    f"Left with non-bare array after evaluating form key {nodes}"
                )
        return out

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
