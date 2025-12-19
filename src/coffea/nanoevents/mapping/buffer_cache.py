from __future__ import annotations

import abc
import dataclasses
import typing as tp
from collections.abc import MutableMapping
from io import BytesIO
from weakref import finalize

import numpy as np


class NbytesAwareCache(MutableMapping):
    @abc.abstractmethod
    def get_nbytes(self, key: tp.Hashable, value: tp.Any) -> int:
        raise NotImplementedError


class BufferCache(NbytesAwareCache):
    """
    A simple dict-like buffer cache.

    Example
    -------
    >>> buffer_cache=BufferCache()
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),
    """

    __slots__ = ("_cache",)

    def __init__(self):
        self._cache = {}

    @property
    def cache(self) -> dict[tp.Hashable, np.ndarray]:
        return self._cache

    def get_nbytes(self, key: tp.Hashable, value: np.ndarray) -> int:
        return value.nbytes

    def get_current_nbytes(self) -> int:
        return sum(arr.nbytes for arr in self.cache.values())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nbuffers={len(self)}, nbytes={self.get_current_nbytes()})"

    def __setitem__(self, key: tp.Hashable, value: np.ndarray) -> None:
        self.cache[key] = value

    def __getitem__(self, key: tp.Hashable) -> np.ndarray:
        return self.cache[key]

    def __delitem__(self, key: tp.Hashable) -> None:
        del self.cache[key]

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        return iter(self.cache)

    def __len__(self) -> int:
        return len(self.cache)


@dataclasses.dataclass(slots=True, frozen=True)
class ShapeDTypeStruct:
    dtype: np.dtype
    shape: tuple[int]
    strides: tuple[int]


ByteBuffer: tp.TypeAlias = bytes


@tp.runtime_checkable
class Codec(tp.Protocol):
    def encode(self, arr: np.ndarray) -> tuple[ByteBuffer, ShapeDTypeStruct]: ...
    def decode(self, buffer: ByteBuffer, struct: ShapeDTypeStruct) -> np.ndarray: ...


class NumCodecsWrapper:
    __slots__ = ("_codec",)

    def __init__(self, codec: tp.Any) -> None:
        self._codec = codec

    def encode(self, arr: np.ndarray) -> tuple[ByteBuffer, ShapeDTypeStruct]:
        struct = ShapeDTypeStruct(dtype=arr.dtype, shape=arr.shape, strides=arr.strides)
        encoded = self._codec.encode(arr.tobytes())
        return encoded, struct

    def decode(self, buffer: ByteBuffer, struct: ShapeDTypeStruct) -> np.ndarray:
        decoded = self._codec.decode(buffer)
        arr = np.frombuffer(decoded, struct.dtype)
        return np.lib.stride_tricks.as_strided(arr, struct.shape, struct.strides)


CompressedCache: tp.TypeAlias = dict[tp.Hashable, ByteBuffer]
ShapeDTypeStructCache: tp.TypeAlias = dict[tp.Hashable, ShapeDTypeStruct]


class CompressedBufferCache(NbytesAwareCache):
    """
    An in-memory compressed buffer cache. Supports all numcodecs.abc.Codec types.

    Example
    -------
    >>> import numcodecs
    >>> codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    >>> buffer_cache=CompressedBufferCache(codec=codec)
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),
    """

    __slots__ = ("_codec", "_cache", "_meta")

    def __init__(self, codec: tp.Any) -> None:
        import numcodecs

        # auto-wrap for numcodecs.abc.Codec
        if isinstance(codec, numcodecs.abc.Codec):
            codec = NumCodecsWrapper(codec=codec)

        # at this point we expect a proper Codec instance
        if not isinstance(codec, Codec):
            raise TypeError(f"codec must be an instance of Codec, got {type(codec)}")

        self._codec = codec
        self._cache: CompressedCache = {}
        self._meta: ShapeDTypeStructCache = {}

    @property
    def codec(self) -> Codec:
        return self._codec

    @property
    def cache(self) -> CompressedCache:
        return self._cache

    @property
    def meta(self) -> ShapeDTypeStructCache:
        return self._meta

    def get_nbytes(self, key: tp.Hashable, value: ByteBuffer) -> int:
        return len(value)  # respects compression

    def get_current_nbytes(self) -> int:
        return sum(map(len, self.cache.values()), 0)  # respects compression

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nbuffers={len(self)}, nbytes={self.get_current_nbytes()})"

    def __setitem__(self, key: tp.Hashable, value: np.ndarray) -> None:
        buf, struct = self.codec.encode(value)
        self.cache[key] = buf
        self.meta[key] = struct

    def __getitem__(self, key: tp.Hashable) -> np.ndarray:
        return self.codec.decode(buffer=self.cache[key], struct=self.meta[key])

    def __delitem__(self, key: tp.Hashable):
        del self.cache[key]
        del self.meta[key]

    def __iter__(self) -> tp.Iterable:
        return iter(self.cache)

    def __len__(self) -> int:
        return len(self.cache)

    # overwrite .items() because zict.Buffer.weight loops over `.items()` to infer nbytes, see:
    # https://github.com/dask/zict/blob/main/zict/lru.py#L108
    # the default implementation would decompress/load the array,
    # but we can access nbytes from metadata here
    def items(self) -> tp.Iterator[tuple[tp.Hashable, ByteBuffer]]:
        for k in self.cache:
            # avoid decompressing the array
            yield (k, self.cache[k])


HDF5Group: tp.TypeAlias = tp.Any
HDF5Dataset: tp.TypeAlias = tp.Any


def _gracefully_close(
    file_handle: tp.TextIO | tp.BinaryIO,
    group: HDF5Group,
) -> None:
    group.close()
    file_handle.close()
    # can't remove in-memory files
    if not isinstance(file_handle, BytesIO):
        import os

        os.remove(os.path.abspath(file_handle.name))


class HDF5BufferCache(NbytesAwareCache):
    """
    An file-backed hdf5 buffer cache. HDF5BufferCache will gracefully close
    provided file handles automatically through the ``finalize_callback``.

    Example
    -------
    >>> buffer_cache = HDF5BufferCache(file_handle=open("mycache.h5", "wb+"))
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),
    """

    __slots__ = ("_group", "_file_handle", "_create_dataset_opts")

    def __init__(
        self,
        file_handle: tp.TextIO | tp.BinaryIO,
        create_dataset_opts: tp.Any = None,
        finalize_callback: tp.Callable = _gracefully_close,
    ) -> None:
        import h5py

        # a HDF5 group
        # e.g.:
        # in-memory: `group = h5py.File(BytesIO(), "r+")` or `h5py.File.in_memory()` since v3.13 (see: https://docs.h5py.org/en/stable/high/file.html#in-memory-files)
        # on-disk: `group = h5py.File(open("myfile.h5", "wb+"), "w")`
        self._file_handle = file_handle
        gopts = "r+" if isinstance(file_handle, BytesIO) else "w"
        self._group = h5py.File(file_handle, gopts)

        # HDF5 dataset options
        # e.g.:
        # from hdf5plugin import Blosc2
        # {"compression": Blosc2(cname="zstd", clevel=1, filters=Blosc2.BITSHUFFLE)}
        if create_dataset_opts is None:
            create_dataset_opts = {}
        self._create_dataset_opts = create_dataset_opts

        # Zig-like `defer`, handles file obj and h5py group closing before GC
        finalize(self, finalize_callback, self._file_handle, self._group)

    @property
    def file_handle(self) -> tp.TextIO | tp.BinaryIO:
        return self._file_handle

    @property
    def group(self) -> HDF5Group:
        return self._group

    def get_nbytes(self, key: tp.Hashable, value: HDF5Dataset) -> int:
        return value.id.get_storage_size()  # respects compression

    def get_current_nbytes(self) -> int:
        return self.group.id.get_filesize()  # respects compression + metadata

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nbuffers={len(self)}, nbytes={self.get_current_nbytes()})"

    def __setitem__(self, key: tp.Hashable, value: np.ndarray) -> None:
        self.group.create_dataset(
            name=key,
            shape=value.shape,
            dtype=value.dtype,
            chunks=value.shape,
            data=value,
            **self._create_dataset_opts,
        )

    def __getitem__(self, key: tp.Hashable) -> np.ndarray:
        return self.group[key][...]

    def __delitem__(self, key: tp.Hashable):
        del self.group[key]

    def __iter__(self) -> tp.Iterable:
        return iter(self.group)

    def __len__(self) -> int:
        return self.group.id.get_num_objs()

    # overwrite .items() because zict.Buffer.weight loops over `.items()` to infer nbytes, see:
    # https://github.com/dask/zict/blob/main/zict/lru.py#L108
    # the default implementation would decompress/load the array,
    # but we can access nbytes from metadata here
    def items(self) -> tp.Iterator[tuple[tp.Hashable, HDF5Dataset]]:
        for k in self.cache:
            # avoid loading & decompressing the array
            yield (k, self.group[k])


# Optionally TODO (pfackeldey: I don't think they bring anything beneficial on top of the hdf5 buffer cache?):
# - add zarr on-disk cache
# - add blosc2 treestore cache (https://www.blosc.org/python-blosc2/reference/tree_store.html#blosc2.TreeStore)


def lru_cache(
    capacity: int,
    *,
    cache: NbytesAwareCache | None = None,
) -> MutableMapping:
    """
    Wraps a given ``NbytesAwareCache`` into an LRU cache with a ``capacity`` (given in bytes).
    If no ``cache`` is provided, a default ``BufferCache`` is used.

    Example
    -------
    >>> lru_500MB = lru_cache(capacity=500_000_000) # 500 MB
    >>> NanoEventsFactory.from_root(..., buffer_cache=lru_500MB),
    """
    import zict

    if cache is None:
        cache = BufferCache()

    if not isinstance(cache, NbytesAwareCache):
        raise TypeError(
            f"cache must be an instance of NbytesAwareCache, got {type(cache)}"
        )

    return zict.LRU(n=int(capacity), d=cache, weight=cache.get_nbytes)


def hierarchical_cache(
    layers: tp.Iterable[tuple[int, NbytesAwareCache]],
) -> MutableMapping:
    """Compose a stack of caches into a zict.Buffer hierarchy.

    Layers should be ordered from fastest to slowest as ``(limit, cache)`` pairs.
    The final cache in the iterable is treated as the base store, it's limit will
    be ignored; intermediate caches are wrapped with ``Buffer`` instances using
    ``limit`` for the ``n`` parameter. A single layer returns the cache unchanged,
    while an empty iterable raises ``ValueError``.

    Example
    -------
    >>> from numcodecs import Blosc
    >>> inmemory = BufferCache()
    >>> inmemory_compressed = CompressedBufferCache(codec=Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE))
    >>> ondisk = HDF5BufferCache(file_handle=open("mycache.h5", "wb+"))
    >>> cache = hierarchical_cache([
        (100_000_000, inmemory), # 100 MB
        (1_000_000_000, inmemory_compressed), # 1 GB
        (-1, ondisk),
    ])
    """
    import zict

    layers = list(layers)

    nlayers = len(layers)

    match nlayers:
        case 0:
            raise ValueError("layers must be non-empty")
        case 1:
            return layers[0][1]  # returns the cache
        case _:
            # last mapping is the base slow store
            slow = layers[-1][1]
            # wrap remaining layers from bottom to top
            for limit, fast in reversed(layers[:-1]):
                slow = zict.Buffer(
                    fast=fast, slow=slow, n=float(limit), weight=fast.get_nbytes
                )
            return slow
