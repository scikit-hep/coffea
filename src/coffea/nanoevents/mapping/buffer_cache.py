import dataclasses
import typing as tp
from collections.abc import MutableMapping

import numpy as np


@dataclasses.dataclass(slots=True, frozen=True)
class ShapeDTypeStruct:
    dtype: np.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]


ByteBuffer: tp.TypeAlias = bytes


@tp.runtime_checkable
class Codec(tp.Protocol):
    def encode(self, arr: np.ndarray) -> tuple[ByteBuffer, ShapeDTypeStruct]: ...
    def decode(self, buffer: ByteBuffer, struct: ShapeDTypeStruct) -> np.ndarray: ...


class NoCompressionCodec(Codec):
    def encode(self, arr: np.ndarray) -> tuple[ByteBuffer, ShapeDTypeStruct]:
        struct = ShapeDTypeStruct(dtype=arr.dtype, shape=arr.shape, strides=arr.strides)
        return arr.tobytes(), struct

    def decode(self, buffer: ByteBuffer, struct: ShapeDTypeStruct) -> np.ndarray:
        arr = np.frombuffer(buffer, struct.dtype)
        return np.lib.stride_tricks.as_strided(arr, struct.shape, struct.strides)


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


ByteBufferCache: tp.TypeAlias = MutableMapping[tp.Hashable, ByteBuffer]
ShapeDTypeStructCache: tp.TypeAlias = MutableMapping[tp.Hashable, ShapeDTypeStruct]


class CodecAwareCache(MutableMapping):
    __slots__ = ("_cache", "_meta", "_codec")

    def __init__(self, cache: ByteBufferCache, codec: Codec):
        self._cache: ByteBufferCache = cache
        self._meta: ShapeDTypeStructCache = {}

        if not isinstance(codec, Codec):
            raise TypeError(f"codec must be an instance of Codec, got {type(codec)}")
        self._codec = codec

    @property
    def cache(self) -> ByteBufferCache:
        return self._cache

    @property
    def meta(self) -> ShapeDTypeStructCache:
        return self._meta

    @property
    def codec(self) -> Codec:
        return self._codec

    def __setitem__(self, key: tp.Hashable, value: np.ndarray) -> None:
        buf, struct = self.codec.encode(value)
        self.cache[key] = buf
        self.meta[key] = struct

    def __getitem__(self, key: tp.Hashable) -> np.ndarray:
        return self.codec.decode(buffer=self.cache[key], struct=self.meta[key])

    def __delitem__(self, key: tp.Hashable):
        del self.cache[key]
        del self.meta[key]

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        return iter(self.cache)

    def __len__(self) -> int:
        return len(self.cache)


# can't type hint without importing it, so we do this instead
NumCodecsCodec: tp.TypeAlias = tp.Any


def BufferCache(
    cache: ByteBufferCache | None,
    codec: Codec | NumCodecsCodec | None,  # noqa: F821
) -> MutableMapping:
    """
    A compressed buffer cache. Supports all numcodecs.abc.Codec types.


    Example (in-memory no compression)
    -------
    >>> buffer_cache=BufferCache(cache=None, codec=None) # or `NoCompressionCodec()`
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),


    Example (in-memory compressed)
    -------
    >>> import numcodecs
    >>> codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    >>> buffer_cache=BufferCache(cache=None, codec=codec)
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),


    Example (on-disk compressed)
    -------
    >>> import numcodecs, zict
    >>> codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    >>> buffer_cache=BufferCache(cache=zict.File("my_cache"), codec=codec)
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),


    Example (LRU-backed compressed in-memory)
    -------
    >>> import numcodecs, zict
    >>> codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    >>> capacity = 500_000_000 # 500 MB
    >>> # len gives the number of bytes in the bytebuffer
    >>> cache = zict.LRU(n=capacity, d={}, weight=lambda k,v: len(v))
    >>> buffer_cache=BufferCache(cache=cache, codec=codec)
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),


    Example (hierarchical)
    -------
    >>> import zict
    >>> cache = zict.Buffer(
    >>>     fast={},
    >>>     slow=zict.File("mycache"),
    >>>     n=100,
    >>>     weight=lambda k,v: len(v), # len gives the number of bytes in the bytebuffer
    >>> )
    >>> buffer_cache=BufferCache(cache=cache, codec=None)
    >>> NanoEventsFactory.from_root(..., buffer_cache=buffer_cache),
    """
    if cache is None:
        cache = {}

    if not isinstance(cache, MutableMapping):
        raise TypeError(
            f"cache must be an instance of MutableMapping, got {type(cache)}"
        )

    if codec is not None:
        try:
            import numcodecs
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError("""to use BufferCache, you must install numcodecs:

pip install numcodecs

or

conda install -c conda-forge numcodecs""") from err

        # auto-wrap for numcodecs.abc.Codec
        if isinstance(codec, numcodecs.abc.Codec):
            codec = NumCodecsWrapper(codec=codec)

        # at this point we expect a proper Codec instance
        if not isinstance(codec, Codec):
            raise TypeError(f"codec must be an instance of Codec, got {type(codec)}")

        return CodecAwareCache(cache=cache, codec=codec)

    return cache
