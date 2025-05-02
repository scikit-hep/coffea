import weakref
from collections.abc import Mapping, MutableMapping

from coffea.nanoevents.util import key_to_tuple


class CachedMapping(Mapping):
    """A cache-wrapped mapping
    Reads will call into ``cache`` first, and if no key exists,
    the read will fall back to ``base``, saving the result into ``cache``.
    """

    def __init__(self, cache, base):
        self.cache = cache
        self.base = base
        self.stats = {"hit": 0, "miss": 0}

    def __getitem__(self, key):
        try:
            value = self.cache[key]
            self.stats["hit"] += 1
            return value
        except KeyError:
            value = self.base[key]
            self.cache[key] = value
            self.stats["miss"] += 1
            return value

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)


class ArrayLifecycleMapping(MutableMapping):
    """A tool to monitor the lifetime of arrays

    Useful for detecting if arrays are getting properly cleaned up
    by garbage collection. To be used with NanoEventsFactory as a "fake"
    ``persistent_cache``

    Example::

        from coffea.nanoevents import NanoEventsFactory
        from coffea.nanoevents.mapping import ArrayLifecycleMapping

        array_log = ArrayLifecycleMapping()

        def run():
            events = NanoEventsFactory.from_root(
                "file.root",
                persistent_cache=array_log,
            ).events()
            # ... access things

        run()
        # may consider gc.collect() here
        print("Accessed:", array_log.accessed)
        print("Finalized:", array_log.finalized)
        print("Possibly leaking arrays:", set(array_log.accessed) - set(array_log.finalized))
    """

    def __init__(self):
        self.accessed = []
        self.finalized = []

    def __getitem__(self, key):
        raise KeyError

    def __setitem__(self, key, value):
        key = key_to_tuple(key)[4]
        key = key.split(",")[0]
        self.accessed.append(key)
        weakref.finalize(value, self.finalized.append, key)

    def __delitem__(self, key):
        pass

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)
