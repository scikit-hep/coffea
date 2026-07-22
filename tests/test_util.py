import os

import pytest

from coffea.processor import defaultdict_accumulator, dict_accumulator
from coffea.processor.test_items import NanoEventsProcessor
from coffea.util import load, save


@pytest.mark.parametrize("compression", [None, "lz4"])
def test_loadsave(compression):
    pytest.importorskip("hist.dask")
    filename = "testprocessor.coffea"
    try:
        aprocessor = NanoEventsProcessor()
        save(aprocessor, filename, compression)
        newprocessor = load(filename, compression)
        assert "pt" in newprocessor.accumulator
        assert newprocessor.accumulator["pt"].axes == aprocessor.accumulator["pt"].axes

        output = {"test": "foo"}
        save(output, filename, compression)
        newoutput = load(filename, compression)
        assert newoutput == output

        output = {}
        output["test"] = output
        save(output, filename, compression)
        newoutput = load(filename, compression)
        assert newoutput["test"] is newoutput

        output = lambda x: x + 1  # noqa E731
        save(output, filename, compression)
        newoutput = load(filename, compression)
        assert newoutput(1) == 2
        assert newoutput(2) == 3

        def output(x):
            return x + 1

        save(output, filename, compression)
        newoutput = load(filename, compression)
        assert newoutput(1) == 2
        assert newoutput(2) == 3

        output = dict_accumulator(
            {
                "cutflow": defaultdict_accumulator(int),
            }
        )
        output["cutflow"]["x"] += 1
        output["cutflow"]["y"] += 2
        save(output, filename, compression)
        newoutput = load(filename, compression)
        assert isinstance(newoutput, dict_accumulator)
        assert isinstance(newoutput["cutflow"], defaultdict_accumulator)
        assert newoutput["cutflow"]["x"] == 1
        assert newoutput["cutflow"]["y"] == 2

        output = defaultdict_accumulator(lambda: defaultdict_accumulator(int))
        output["x"]["y"] += 1
        output["x"]["z"] += 2
        output["a"]["b"] += 3
        save(output, filename, compression)
        newoutput = load(filename, compression)
        assert isinstance(newoutput, defaultdict_accumulator)
        assert isinstance(newoutput["x"], defaultdict_accumulator)
        assert newoutput["x"]["y"] == 1
        assert newoutput["x"]["z"] == 2
        assert newoutput["a"]["b"] == 3
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_dask_property_is_picklable():
    """Behavior classes defined outside an importable module get pickled by
    value (e.g. by cloudpickle, when a dask graph goes to a distributed worker),
    which walks the class dict -- and plain property objects cannot be pickled.
    """
    cloudpickle = pytest.importorskip("cloudpickle")

    from coffea.util import dask_property

    class Thing:
        def __init__(self, x):
            self.x = x

        @dask_property
        def doubled(self):
            """twice x"""
            return 2 * self.x

        @doubled.dask
        def doubled(self, dask_array):
            return 20 * dask_array.x

        @dask_property(no_dispatch=True)
        def tripled(self):
            return 3 * self.x

    # defined in a function body, so this has to go by value
    unpickled = cloudpickle.loads(cloudpickle.dumps(Thing))

    assert unpickled(1).doubled == 2
    assert unpickled(1).tripled == 3

    doubled = unpickled.__dict__["doubled"]
    assert doubled.__doc__ == "twice x"
    assert doubled._dask_get(unpickled(1), unpickled, Thing(3)) == 60
    assert unpickled.__dict__["tripled"]._dask_get(unpickled(2), unpickled, None) == 6
