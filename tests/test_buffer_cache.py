import awkward as ak
import pytest

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.mapping import NbytesAwareCache
from coffea.nanoevents.util import unquote


def _make_events_with_cache(path: str, cache: NbytesAwareCache) -> ak.Array:
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
    from coffea.nanoevents.mapping import BufferCache

    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=BufferCache(),
    )

    _check_cache(events)


def test_compressed_buffer_cache(tests_directory):
    pytest.importorskip("numcodecs")

    from numcodecs import Blosc

    from coffea.nanoevents.mapping import CompressedBufferCache

    codec = Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=CompressedBufferCache(codec),
    )

    _check_cache(events)


def test_hdf5_buffer_cache(tests_directory):
    pytest.importorskip("h5py")

    from coffea.nanoevents.mapping import HDF5BufferCache

    file_handle = open(f"{tests_directory}/test_hdf5_buffer_cache.h5", "wb+")
    events = _make_events_with_cache(
        path=f"{tests_directory}/samples/nano_dy.root",
        cache=HDF5BufferCache(file_handle),
    )

    _check_cache(events)

    del events

    # make sure the default finalize callback closes the
    # file handle upon `del events` automatically
    assert file_handle.closed

    # get rid of it
    import os

    os.remove(f"{tests_directory}/test_hdf5_buffer_cache.h5")
