import awkward as ak
import cloudpickle
import dask
import dask_awkward as dak
import numpy as np
import pytest
from dask.distributed import Client

from coffea.lumi_tools import LumiData, LumiList, LumiMask
from coffea.nanoevents import NanoEventsFactory


def test_lumidata():
    from numba import types
    from numba.typed import Dict

    lumidata = LumiData("tests/samples/lumi_small.csv")

    # pickle & unpickle
    lumidata_pickle = cloudpickle.loads(cloudpickle.dumps(lumidata))

    # check same internal lumidata
    assert np.all(lumidata._lumidata == lumidata_pickle._lumidata)

    runslumis = np.zeros((10, 2), dtype=np.uint32)
    results = {"lumi": {}, "index": {}}
    for ld in lumidata, lumidata_pickle:
        runslumis[:, 0] = ld._lumidata[0:10, 0]
        runslumis[:, 1] = ld._lumidata[0:10, 1]
        lumi = ld.get_lumi(runslumis)
        results["lumi"][ld] = lumi
        diff = abs(lumi - 1.539941814)
        print("lumi:", lumi, "diff:", diff)
        assert diff < 1e-4

        # test build_lumi_table_kernel
        py_index = Dict.empty(
            key_type=types.Tuple([types.uint32, types.uint32]), value_type=types.float64
        )
        pyruns = ld._lumidata[:, 0].astype("u4")
        pylumis = ld._lumidata[:, 1].astype("u4")
        LumiData._build_lumi_table_kernel.py_func(
            pyruns, pylumis, ld._lumidata, py_index
        )

        assert len(py_index) == len(ld.index)

        # test get_lumi_kernel
        py_tot_lumi = np.zeros((1,), dtype=np.float64)
        LumiData._get_lumi_kernel.py_func(
            runslumis[:, 0], runslumis[:, 1], py_index, py_tot_lumi
        )

        assert abs(py_tot_lumi[0] - lumi) < 1e-4

        # store results:
        results["lumi"][ld] = lumi
        results["index"][ld] = ld.index

    assert np.all(results["lumi"][lumidata] == results["lumi"][lumidata_pickle])
    assert len(results["index"][lumidata]) == len(results["index"][lumidata_pickle])


@pytest.mark.parametrize(
    "jsonfile",
    [
        "tests/samples/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt",
        "https://raw.githubusercontent.com/CoffeaTeam/coffea/master/tests/samples/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt",
    ],
)
def test_lumimask(jsonfile):
    client = Client()

    lumimask = LumiMask(jsonfile)

    # pickle & unpickle
    lumimask_pickle = cloudpickle.loads(cloudpickle.dumps(lumimask))

    # check same mask keys
    keys = lumimask._masks.keys()
    assert keys == lumimask_pickle._masks.keys()
    # check same mask values
    assert all(np.all(lumimask._masks[k] == lumimask_pickle._masks[k]) for k in keys)

    runs = np.array([303825, 123], dtype=np.uint32)
    lumis = np.array([115, 123], dtype=np.uint32)

    for lm in lumimask, lumimask_pickle:
        mask = lm(runs, lumis)
        print("mask:", mask)
        assert mask[0]
        assert not mask[1]

        # test underlying py_func
        py_mask = np.zeros(dtype="bool", shape=runs.shape)
        LumiMask._apply_run_lumi_mask_kernel.py_func(lm._masks, runs, lumis, py_mask)

        assert np.all(mask == py_mask)

    assert np.all(lumimask(runs, lumis) == lumimask_pickle(runs, lumis))

    runs_dak = dak.from_awkward(ak.Array(runs), 1)
    lumis_dak = dak.from_awkward(ak.Array(lumis), 1)
    assert np.all(
        client.compute(lumimask(runs_dak, lumis_dak)).result()
        == lumimask_pickle(runs, lumis)
    )

    client.close()


def test_lumilist():
    lumidata = LumiData("tests/samples/lumi_small.csv")

    runslumis1 = np.zeros((10, 2), dtype=np.uint32)
    runslumis1[:, 0] = lumidata._lumidata[0:10, 0]
    runslumis1[:, 1] = lumidata._lumidata[0:10, 1]

    runslumis2 = np.zeros((10, 2), dtype=np.uint32)
    runslumis2[:, 0] = lumidata._lumidata[10:20, 0]
    runslumis2[:, 1] = lumidata._lumidata[10:20, 1]

    llist1 = LumiList(runs=runslumis1[:, 0], lumis=runslumis1[:, 1])
    llist2 = LumiList(runs=runslumis2[:, 0], lumis=runslumis2[:, 1])
    llist3 = LumiList(delayed=False)

    llist3 += llist1
    llist3 += llist2

    lumi1 = lumidata.get_lumi(llist1)
    lumi2 = lumidata.get_lumi(llist2)
    lumi3 = lumidata.get_lumi(llist3)

    assert abs(lumi3 - (lumi1 + lumi2)) < 1e-4

    llist1.clear()
    assert llist1.array.size == 0


def test_lumilist_dask():
    lumidata = LumiData("tests/samples/lumi_small.csv")

    runslumis1 = np.zeros((10, 2), dtype=np.uint32)
    runslumis1[:, 0] = lumidata._lumidata[0:10, 0]
    runslumis1[:, 1] = lumidata._lumidata[0:10, 1]

    runslumis2 = np.zeros((10, 2), dtype=np.uint32)
    runslumis2[:, 0] = lumidata._lumidata[10:20, 0]
    runslumis2[:, 1] = lumidata._lumidata[10:20, 1]

    drunslumis1 = dak.from_awkward(ak.Array(runslumis1), 3)
    drunslumis2 = dak.from_awkward(ak.Array(runslumis2), 3)

    llist1 = LumiList(runs=drunslumis1[:, 0], lumis=drunslumis1[:, 1])
    llist2 = LumiList(runs=drunslumis2[:, 0], lumis=drunslumis2[:, 1])
    llist3 = llist1 + llist2

    lumi1 = lumidata.get_lumi(llist1)
    lumi2 = lumidata.get_lumi(llist2)
    lumi3 = lumidata.get_lumi(llist3)

    lumi1, lumi2, lumi3 = dask.compute(lumi1, lumi2, lumi3)

    assert abs(lumi3 - (lumi1 + lumi2)) < 1e-4


def test_lumilist_client_fromfile():
    with Client() as _:
        events = NanoEventsFactory.from_root(
            {"tests/samples/nano_dy.root": "Events"},
        ).events()

        lumilist = LumiList(runs=events.run, lumis=events.luminosityBlock)

        (result,) = dask.compute(lumilist.array)

        assert result.to_list() == [[1, 13889]]
