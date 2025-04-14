import os
from collections.abc import MutableMapping
from threading import Lock

import blosc
from distributed import WorkerPlugin, get_worker
from zict import LRU, Buffer, File, Func


class ColumnCache(WorkerPlugin, MutableMapping):
    name = "columncache"

    def __init__(self, maxmem=5e8, maxcompressed=2e9, maxdisk=1e10):
        self._maxmem = maxmem
        self._maxcompressed = maxcompressed
        self._maxdisk = maxdisk

    def setup(self, worker):
        self.cache = Buffer(
            fast={},
            slow=Func(
                dump=blosc.pack_array,
                load=blosc.unpack_array,
                d=Buffer(
                    fast={},
                    slow=LRU(
                        n=self._maxdisk,
                        d=File(os.path.join(worker.local_directory, "cache")),
                        weight=lambda k, v: len(v),
                    ),
                    n=self._maxcompressed,
                    weight=lambda k, v: len(v),
                ),
            ),
            n=self._maxmem,
            weight=lambda k, v: v.nbytes,
        )
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def teardown(self, worker):
        pass

    def __getitem__(self, key):
        with self.lock:
            try:
                out = self.cache[key]
                self.hits += 1
                return out
            except KeyError:
                self.misses += 1
                raise

    def __setitem__(self, key, value):
        with self.lock:
            self.cache[key] = value

    def __delitem__(self, key):
        with self.lock:
            del self.cache[key]

    def __iter__(self):
        with self.lock:
            return iter(self.cache)

    def __len__(self):
        with self.lock:
            return len(self.cache)


def register_columncache(client):
    plugins = set()
    for p in client.run(lambda: set(get_worker().plugins)).values():
        plugins |= p
    if ColumnCache.name not in plugins:
        client.register_worker_plugin(ColumnCache())
