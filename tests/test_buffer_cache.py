from collections.abc import MutableMapping

import awkward as ak
import pytest

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.mapping import BufferCache
from coffea.nanoevents.util import unquote


def _make_events_with_cache(path: str, cache: MutableMapping) -> ak.Array:
    factory = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=NanoAODSchema,
        mode="virtual",
        buffer_cache=cache,
    )
    return factory.events()


def _check_cache(events) -> None:
    cache = events.attrs["@events_factory"].buffer_cache
    assert len(cache) == 0

    # materialize something and check that this is now properly populated in the cache
    ak.materialize(events.Jet.pt)

    cache = events.attrs["@events_factory"].buffer_cache
    assert len(cache) == 2

    keys = [*map(unquote, events.attrs["@events_factory"].buffer_cache.keys())]
    assert frozenset(keys) == frozenset(
        [
            # nJet
            "a9490124-3648-11ea-89e9-f5b55c90beef//Events;1/0-40/offsets/nJet,!load,!counts2offsets,!skip,!offsets",
            # Jet_pt
            "a9490124-3648-11ea-89e9-f5b55c90beef//Events;1/0-40/data/Jet_pt,!load,!content",
        ]
    )


def test_buffer_cache(tests_directory):
    pytest.importorskip("zict")

    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(cache=None, codec=None),
    )

    _check_cache(events)


def test_compressed_buffer_cache_in_memory(tests_directory):
    pytest.importorskip("numcodecs")
    pytest.importorskip("zict")

    from numcodecs import Blosc

    codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(cache=None, codec=codec),
    )

    _check_cache(events)


def test_compressed_buffer_cache_on_disk(tests_directory):
    pytest.importorskip("numcodecs")
    pytest.importorskip("zict")

    import zict
    from numcodecs import Blosc

    codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    ondisk = zict.File(f"{tests_directory}/mycache")
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(cache=ondisk, codec=codec),
    )

    _check_cache(events)

    # clean up
    import os

    ondisk.clear()  # rm's all files in mycache
    os.rmdir(f"{tests_directory}/mycache")


def test_buffer_cache_lru(tests_directory):
    zict = pytest.importorskip("zict")

    # large enough to succeed
    cache = zict.LRU(n=100_000_000, d={}, weight=lambda k, v: len(v))
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(cache=cache, codec=None),
    )

    _check_cache(events)

    # small enough to fail (lru cache too small -> keys got evicted -> _check_cache fails)
    cache = zict.LRU(n=100, d={}, weight=lambda k, v: len(v))
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(cache=cache, codec=None),
    )

    with pytest.raises(AssertionError):
        _check_cache(events)


def test_buffer_cache_hierarchical(tests_directory):
    zict = pytest.importorskip("zict")

    hierarchical_cache = zict.Buffer(
        fast={},
        slow=zict.File(f"{tests_directory}/mycache"),
        n=100,
        weight=lambda k, v: len(v),
    )
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(cache=hierarchical_cache, codec=None),
    )

    _check_cache(events)

    # clean up
    import os

    hierarchical_cache.clear()  # rm's all files in mycache
    os.rmdir(f"{tests_directory}/mycache")
