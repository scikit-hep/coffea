from __future__ import annotations

import threading
import typing as tp

import awkward as ak

from coffea.nanoevents.mapping.base import BaseSourceMapping


def _check_attrs_of_array(array: ak.Array) -> None:
    if not isinstance(array, ak.Array):
        raise TypeError("array must be an instance of ak.Array")
    if "@form" not in array.attrs or "@buffer_key" not in array.attrs:
        raise ValueError("array must have '@form' and '@buffer_key' attributes set")


class BufferManager:
    """
    This class helps to manage memory of the `events` object
    from a NanoEventsFactory better. It interacts with the
    underlying buffer_cache of a mode='virtual' instantiated
    events object.

    It provides two primary utility functions:

    1. `clear` which allows to free memory of given set of keys, e.g.:

        >>> manager = BufferManager(events)
        >>> manager.clear(events.Jet["pt"], events.Jet["eta"])

    2. `prefetch` which allows to load/prefetch buffers in background threads, e.g.:

        >>> manager = BufferManager(events)
        >>> with manager.prefetch(events.Jet["pt"], events.Jet["eta"], nthreads=1):
        >>>     ...
    """

    def __init__(self, array: ak.Array):
        _check_attrs_of_array(array)

        self.form = ak.forms.from_dict(array.attrs["@form"])
        self.buffer_key = array.attrs["@buffer_key"]

        self.buffer_cache = array.attrs["@events_factory"].buffer_cache
        if self.buffer_cache is None:
            raise ValueError(
                "Can only manage memory if the `array` was constructed with a buffer_cache, e.g., when using `NanoEventsFactory.from_*(buffer_cache=...).events()`"
            )

        self.buffer_mapping = array.attrs["@events_factory"]._mapping
        if not self.buffer_mapping._virtual:
            raise ValueError(
                "Can only manage memory if the `array` was constructed with virtual mode, e.g., when using `NanoEventsFactory.from_*(mode='virtual').events()`"
            )

    def _yield_buffer_keys(self, *arrays: ak.Array) -> tp.Generator[str]:
        for arr in arrays:
            *_, bufs = ak.to_buffers(arr)
            for buf in bufs.values():
                # virtual arrays track their buffer keys, and also we can only prefetch those
                if isinstance(buf, ak._nplikes.virtual.VirtualNDArray):
                    yield buf.buffer_key

    def clear(self, *arrays: ak.Array) -> None:
        """
        Example
        -------
        >>> manager = BufferManager(events)
        >>> manager.clear(events.Jet["pt"], events.Jet["eta"])
        """
        for bk in self._yield_buffer_keys(*arrays):
            self.buffer_cache.pop(bk, None)

    def prefetch(self, *arrays: ak.Array, nthreads=1):
        """
        Example
        -------
        >>> manager = BufferManager(events)
        >>> with manager.prefetch(events.Jet["pt"], events.Jet["eta"], nthreads=1):
        >>>     ...
        """
        return ThreadedPrefetcher(
            buffer_mapping=self.buffer_mapping,
            keys_to_fetch=set(self._yield_buffer_keys(*arrays)),
            nthreads=nthreads,
        )


class ThreadedPrefetcher:
    def __init__(
        self,
        buffer_mapping: BaseSourceMapping,
        keys_to_fetch: tp.Iterable[str],
        nthreads: int,
    ) -> None:
        self.buffer_mapping = buffer_mapping
        self.keys_to_fetch = keys_to_fetch

        if nthreads <= 0:
            raise ValueError(
                f"Can't start threaded prefetching with non-positive number of threads, got {nthreads=}"
            )
        self.nthreads = nthreads

        # internal to track created threads for proper shutdown
        self._threads = []

    def _prefetch(self):
        while True:
            key = self.buffer_mapping._prefetch_queue.get()
            if key is None:
                break
            try:
                self.buffer_mapping._getitem(key)  # _getitem ensures deduplication
            finally:
                self.buffer_mapping._prefetch_queue.task_done()

    def __enter__(self):
        # start the threads
        for _ in range(self.nthreads):
            prefetch_thread = threading.Thread(target=self._prefetch, daemon=True)
            prefetch_thread.start()
            self._threads.append(prefetch_thread)

        # put keys to fetch in the queue
        for key in self.keys_to_fetch:
            self.buffer_mapping._prefetch_queue.put(key)

    def __exit__(self, exc_type, exc, tb):
        # stop the threads
        for _ in self._threads:
            self.buffer_mapping._prefetch_queue.put(None)
        for w in self._threads:
            w.join()
