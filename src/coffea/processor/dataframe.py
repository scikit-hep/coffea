from collections.abc import MutableMapping

import uproot


class LazyDataFrame(MutableMapping):
    """Simple delayed uproot reader (a la lazyarrays)

    One can access branches either through ``df["bname"]`` or ``df.bname``, although
    the latter is restricted to branches that do not start with a leading underscore.
    Keeps track of values accessed, in the `materialized` attribute.

    Parameters
    ----------
        tree : uproot.TTree
            Tree to read
        entrystart : int, optional
            First entry to read, default: 0
        entrystop : int, optional
            Last entry to read, default None (read to end)
        preload_items : iterable
            Force preloading of a set of columns from the tree
        metadata : Mapping
            Additional metadata for the dataframe
    """

    def __init__(
        self, tree, entrystart=None, entrystop=None, preload_items=None, metadata=None
    ):
        self._tree = tree
        self._branchargs = {
            "decompression_executor": uproot.source.futures.TrivialExecutor(),
            "interpretation_executor": uproot.source.futures.TrivialExecutor(),
        }
        if entrystart is None or entrystart < 0:
            entrystart = 0
        if entrystop is None or entrystop > tree.num_entries:
            entrystop = tree.num_entries
        self._branchargs["entry_start"] = entrystart
        self._branchargs["entry_stop"] = entrystop
        self._available = {k for k in self._tree.keys()}
        self._dict = {}
        self._materialized = set()
        if preload_items:
            self.preload(preload_items)
        self._metadata = metadata

    def __delitem__(self, key):
        del self._dict[key]

    def __getitem__(self, key):
        if key in self._dict:
            return self._dict[key]
        elif key in self._tree:
            self._materialized.add(key)
            array = self._tree[key].array(**self._branchargs)
            self._dict[key] = array
            return self._dict[key]
        else:
            raise KeyError(key)

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __iter__(self):
        yield from self._available

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __contains__(self, key):
        # by default, MutableMapping uses __getitem__ to test, but we want to avoid materialization
        return key in self._dict or key in self._tree

    @property
    def available(self):
        """Set of available columns"""
        return self._available

    @property
    def columns(self):
        """Set of available columns"""
        return self._available

    @property
    def materialized(self):
        """Set of columns read from tree"""
        return self._materialized

    @property
    def size(self):
        """Length of column vector"""
        return self._branchargs["entry_stop"] - self._branchargs["entry_start"]

    @property
    def metadata(self):
        return self._metadata

    def preload(self, columns):
        """Force loading of several columns

        Parameters
        ----------
            columns : iterable
                A list of columns to load
        """
        for name in columns:
            if name in self._tree:
                _ = self[name]
