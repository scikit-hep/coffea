import os

import pytest

from coffea.processor import defaultdict_accumulator, dict_accumulator
from coffea.processor.test_items import NanoEventsProcessor
from coffea.util import load, save


@pytest.mark.parametrize("compression", [None, "lz4"])
def test_loadsave(compression):
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
