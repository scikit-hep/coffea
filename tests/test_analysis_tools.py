import os

import numpy as np
import pytest
import uproot
from dummy_distributions import dummy_jagged_eta_pt

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

fname = "tests/samples/nano_dy.root"
eagerevents = NanoEventsFactory.from_root(
    {os.path.abspath(fname): "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "DYJets"},
    delayed=False,
).events()
dakevents = NanoEventsFactory.from_root(
    {os.path.abspath(fname): "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "DYJets"},
).events()
uprootevents = uproot.dask({fname: "Events"})


def test_weights():
    from coffea.analysis_tools import Weights

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    scale_central = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    scale_up = scale_central * 1.10
    scale_down = scale_central * 0.95
    scale_up_shift = 0.10 * scale_central
    scale_down_shift = 0.05 * scale_central

    weight = Weights(counts.size)
    weight.add("test", scale_central, weightUp=scale_up, weightDown=scale_down)
    weight.add(
        "testShift",
        scale_central,
        weightUp=scale_up_shift,
        weightDown=scale_down_shift,
        shift=True,
    )

    var_names = weight.variations
    expected_names = ["testShiftUp", "testShiftDown", "testUp", "testDown"]
    for name in expected_names:
        assert name in var_names

    test_central = weight.weight()
    exp_weight = scale_central * scale_central

    assert np.all(np.abs(test_central - (exp_weight)) < 1e-6)

    test_up = weight.weight("testUp")
    exp_up = scale_central * scale_central * 1.10

    assert np.all(np.abs(test_up - (exp_up)) < 1e-6)

    test_down = weight.weight("testDown")
    exp_down = scale_central * scale_central * 0.95

    assert np.all(np.abs(test_down - (exp_down)) < 1e-6)

    test_shift_up = weight.weight("testUp")

    assert np.all(np.abs(test_shift_up - (exp_up)) < 1e-6)

    test_shift_down = weight.weight("testDown")

    assert np.all(np.abs(test_shift_down - (exp_down)) < 1e-6)

    with pytest.raises(ValueError):
        raise weight.partial_weight(include="test")

    with pytest.raises(ValueError):
        raise weight.partial_weight(exclude="test")


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_weights_dak(optimization_enabled):
    import dask
    import dask.array as da
    import dask_awkward as dak

    from coffea.analysis_tools import Weights

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()
        scale_central = dak.from_dask_array(
            da.random.normal(loc=1.0, scale=0.01, size=counts.size)
        )
        scale_up = scale_central * 1.10
        scale_down = scale_central * 0.95
        scale_up_shift = 0.10 * scale_central
        scale_down_shift = 0.05 * scale_central

        weight = Weights(None)
        weight.add("test", scale_central, weightUp=scale_up, weightDown=scale_down)
        weight.add(
            "testShift",
            scale_central,
            weightUp=scale_up_shift,
            weightDown=scale_down_shift,
            shift=True,
        )

        var_names = weight.variations
        expected_names = ["testShiftUp", "testShiftDown", "testUp", "testDown"]
        for name in expected_names:
            assert name in var_names

        test_central = weight.weight()
        exp_weight = scale_central * scale_central

        assert np.all(np.abs(test_central - (exp_weight)).compute() < 1e-6)

        test_up = weight.weight("testUp")
        exp_up = scale_central * scale_central * 1.10

        assert np.all(np.abs(test_up - (exp_up)).compute() < 1e-6)

        test_down = weight.weight("testDown")
        exp_down = scale_central * scale_central * 0.95

        assert np.all(np.abs(test_down - (exp_down)).compute() < 1e-6)

        test_shift_up = weight.weight("testUp")

        assert np.all(np.abs(test_shift_up - (exp_up)).compute() < 1e-6)

        test_shift_down = weight.weight("testDown")

        assert np.all(np.abs(test_shift_down - (exp_down)).compute() < 1e-6)


def test_weights_multivariation():
    from coffea.analysis_tools import Weights

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    scale_central = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    scale_up = scale_central * 1.10
    scale_down = scale_central * 0.95
    scale_up_2 = scale_central * 1.2
    scale_down_2 = scale_central * 0.90

    weight = Weights(counts.size)
    weight.add_multivariation(
        "test",
        scale_central,
        modifierNames=["A", "B"],
        weightsUp=[scale_up, scale_up_2],
        weightsDown=[scale_down, scale_down_2],
    )

    var_names = weight.variations
    expected_names = ["test_AUp", "test_ADown", "test_BUp", "test_BDown"]
    for name in expected_names:
        assert name in var_names

    test_central = weight.weight()
    exp_weight = scale_central

    assert np.all(np.abs(test_central - (exp_weight)) < 1e-6)

    test_up = weight.weight("test_AUp")
    exp_up = scale_central * 1.10

    assert np.all(np.abs(test_up - (exp_up)) < 1e-6)

    test_down = weight.weight("test_ADown")
    exp_down = scale_central * 0.95

    assert np.all(np.abs(test_down - (exp_down)) < 1e-6)

    test_up_2 = weight.weight("test_BUp")
    exp_up = scale_central * 1.2

    assert np.all(np.abs(test_up_2 - (exp_up)) < 1e-6)

    test_down_2 = weight.weight("test_BDown")
    exp_down = scale_central * 0.90

    assert np.all(np.abs(test_down_2 - (exp_down)) < 1e-6)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_weights_multivariation_dak(optimization_enabled):
    import dask
    import dask.array as da
    import dask_awkward as dak

    from coffea.analysis_tools import Weights

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()
        scale_central = dak.from_dask_array(
            da.random.normal(loc=1.0, scale=0.01, size=counts.size)
        )
        scale_up = scale_central * 1.10
        scale_down = scale_central * 0.95
        scale_up_2 = scale_central * 1.2
        scale_down_2 = scale_central * 0.90

        weight = Weights(None)
        weight.add_multivariation(
            "test",
            scale_central,
            modifierNames=["A", "B"],
            weightsUp=[scale_up, scale_up_2],
            weightsDown=[scale_down, scale_down_2],
        )

        var_names = weight.variations
        expected_names = ["test_AUp", "test_ADown", "test_BUp", "test_BDown"]
        for name in expected_names:
            assert name in var_names

        test_central = weight.weight()
        exp_weight = scale_central

        assert np.all(np.abs(test_central - (exp_weight)).compute() < 1e-6)

        test_up = weight.weight("test_AUp")
        exp_up = scale_central * 1.10

        assert np.all(np.abs(test_up - (exp_up)).compute() < 1e-6)

        test_down = weight.weight("test_ADown")
        exp_down = scale_central * 0.95

        assert np.all(np.abs(test_down - (exp_down)).compute() < 1e-6)

        test_up_2 = weight.weight("test_BUp")
        exp_up = scale_central * 1.2

        assert np.all(np.abs(test_up_2 - (exp_up)).compute() < 1e-6)

        test_down_2 = weight.weight("test_BDown")
        exp_down = scale_central * 0.90

        assert np.all(np.abs(test_down_2 - (exp_down)).compute() < 1e-6)


def test_weights_partial():
    from coffea.analysis_tools import Weights

    counts, _, _ = dummy_jagged_eta_pt()
    w1 = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    w2 = np.random.normal(loc=1.3, scale=0.05, size=counts.size)

    weights = Weights(counts.size, storeIndividual=True)
    weights.add("w1", w1)
    weights.add("w2", w2)

    test_exclude_none = weights.weight()
    assert np.all(np.abs(test_exclude_none - w1 * w2) < 1e-6)

    test_exclude1 = weights.partial_weight(exclude=["w1"])
    assert np.all(np.abs(test_exclude1 - w2) < 1e-6)

    test_include1 = weights.partial_weight(include=["w1"])
    assert np.all(np.abs(test_include1 - w1) < 1e-6)

    test_exclude2 = weights.partial_weight(exclude=["w2"])
    assert np.all(np.abs(test_exclude2 - w1) < 1e-6)

    test_include2 = weights.partial_weight(include=["w2"])
    assert np.all(np.abs(test_include2 - w2) < 1e-6)

    test_include_both = weights.partial_weight(include=["w1", "w2"])
    assert np.all(np.abs(test_include_both - w1 * w2) < 1e-6)

    # Check that exception is thrown if arguments are incompatible
    error_raised = False
    try:
        weights.partial_weight(exclude=["w1"], include=["w2"])
    except ValueError:
        error_raised = True
    assert error_raised

    error_raised = False
    try:
        weights.partial_weight()
    except ValueError:
        error_raised = True
    assert error_raised

    # Check that exception is thrown if individual weights
    # are not saved from the start
    weights = Weights(counts.size, storeIndividual=False)
    weights.add("w1", w1)
    weights.add("w2", w2)

    error_raised = False
    try:
        weights.partial_weight(exclude=["test"], include=["test"])
    except ValueError:
        error_raised = True
    assert error_raised


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_weights_partial_dak(optimization_enabled):
    import dask
    import dask.array as da
    import dask_awkward as dak

    from coffea.analysis_tools import Weights

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, _, _ = dummy_jagged_eta_pt()
        w1 = dak.from_dask_array(
            da.random.normal(loc=1.0, scale=0.01, size=counts.size)
        )
        w2 = dak.from_dask_array(
            da.random.normal(loc=1.3, scale=0.05, size=counts.size)
        )

        weights = Weights(None, storeIndividual=True)
        weights.add("w1", w1)
        weights.add("w2", w2)

        test_exclude_none = weights.weight()
        assert np.all(np.abs(test_exclude_none - w1 * w2).compute() < 1e-6)

        test_exclude1 = weights.partial_weight(exclude=["w1"])
        assert np.all(np.abs(test_exclude1 - w2).compute() < 1e-6)

        test_include1 = weights.partial_weight(include=["w1"])
        assert np.all(np.abs(test_include1 - w1).compute() < 1e-6)

        test_exclude2 = weights.partial_weight(exclude=["w2"])
        assert np.all(np.abs(test_exclude2 - w1).compute() < 1e-6)

        test_include2 = weights.partial_weight(include=["w2"])
        assert np.all(np.abs(test_include2 - w2).compute() < 1e-6)

        test_include_both = weights.partial_weight(include=["w1", "w2"])
        assert np.all(np.abs(test_include_both - w1 * w2).compute() < 1e-6)

        # Check that exception is thrown if arguments are incompatible
        error_raised = False
        try:
            weights.partial_weight(exclude=["w1"], include=["w2"])
        except ValueError:
            error_raised = True
        assert error_raised

        error_raised = False
        try:
            weights.partial_weight()
        except ValueError:
            error_raised = True
        assert error_raised

        # Check that exception is thrown if individual weights
        # are not saved from the start
        weights = Weights(None, storeIndividual=False)
        weights.add("w1", w1)
        weights.add("w2", w2)

        error_raised = False
        try:
            weights.partial_weight(exclude=["test"], include=["test"])
        except ValueError:
            error_raised = True
        assert error_raised


@pytest.mark.parametrize("dtype", ["uint16", "uint32", "uint64"])
def test_packed_selection_basic(dtype):
    import awkward as ak
    import dask.array as da
    import dask_awkward as dak

    from coffea.analysis_tools import PackedSelection

    sel = PackedSelection(dtype=dtype)

    shape = (10,)
    all_true = np.full(shape=shape, fill_value=True, dtype=bool)
    all_false = np.full(shape=shape, fill_value=False, dtype=bool)
    fizz = np.arange(shape[0]) % 3 == 0
    buzz = np.arange(shape[0]) % 5 == 0
    ones = np.ones(shape=shape, dtype=np.uint64)
    wrong_shape = np.ones(shape=(shape[0] - 5,), dtype=bool)
    wrong_type = dak.from_awkward(ak.Array(np.arange(shape[0]) % 3 == 0), 1)
    daskarray = da.arange(shape[0]) % 3 == 0

    with pytest.warns(
        UserWarning,
        match="PackedSelection hasn't been initialized with a boolean array yet!",
    ):
        assert sel.delayed_mode is False

    sel.add("fizz", fizz)
    sel.add("buzz", buzz)

    assert np.all(
        sel.all()
        == np.array(
            [True, False, False, False, False, False, False, False, False, False]
        )
    )
    assert np.all(
        sel.allfalse()
        == np.array([False, True, True, False, True, False, False, True, True, False])
    )

    sel.add_multiple({"all_true": all_true, "all_false": all_false})

    assert sel.delayed_mode is False
    with pytest.raises(
        ValueError,
        match="New selection 'wrong_type' is not eager while PackedSelection is!",
    ):
        sel.add("wrong_type", wrong_type)

    assert np.all(sel.require(all_true=True, all_false=False) == all_true)
    # allow truthy values
    assert np.all(sel.require(all_true=1, all_false=0) == all_true)
    assert np.all(sel.all("all_true", "all_false") == all_false)
    assert np.all(sel.any("all_true", "all_false") == all_true)
    assert np.all(
        sel.all("fizz", "buzz")
        == np.array(
            [True, False, False, False, False, False, False, False, False, False]
        )
    )
    assert np.all(
        sel.allfalse("fizz", "buzz")
        == np.array([False, True, True, False, True, False, False, True, True, False])
    )
    assert np.all(
        sel.any("fizz", "buzz")
        == np.array([True, False, False, True, False, True, True, False, False, True])
    )

    with pytest.raises(
        ValueError,
        match=r"New selection 'wrong_shape' has a different shape than existing selections \(\(5,\) vs. \(10,\)\)",
    ):
        sel.add("wrong_shape", wrong_shape)

    with pytest.raises(ValueError, match="Expected a boolean array, received uint64"):
        sel.add("ones", ones)

    with pytest.raises(RuntimeError):
        overpack = PackedSelection(dtype=dtype)
        for i in range(65):
            overpack.add("sel_%d", all_true)

    with pytest.raises(
        ValueError,
        match="Dask arrays are not supported, please convert them to dask_awkward.Array by using dask_awkward.from_dask_array()",
    ):
        sel.add("dask_array", daskarray)


def test_packed_selection_nminusone():
    import awkward as ak

    from coffea.analysis_tools import PackedSelection

    events = eagerevents

    selection = PackedSelection()

    twoelectron = ak.num(events.Electron) == 2
    nomuon = ak.num(events.Muon) == 0
    leadpt20 = ak.any(events.Electron.pt >= 20.0, axis=1) | ak.any(
        events.Muon.pt >= 20.0, axis=1
    )

    selection.add_multiple(
        {
            "twoElectron": twoelectron,
            "noMuon": nomuon,
            "leadPt20": leadpt20,
        }
    )

    assert selection.names == ["twoElectron", "noMuon", "leadPt20"]

    with pytest.raises(
        ValueError,
        match="All arguments must be strings that refer to the names of existing selections",
    ):
        selection.nminusone("twoElectron", "nonexistent")
    nminusone = selection.nminusone("twoElectron", "noMuon", "leadPt20")

    labels, nev, masks = nminusone.result()

    assert labels == ["initial", "N - twoElectron", "N - noMuon", "N - leadPt20", "N"]

    assert nev == [
        len(events),
        len(events[nomuon & leadpt20]),
        len(events[twoelectron & leadpt20]),
        len(events[twoelectron & nomuon]),
        len(events[twoelectron & nomuon & leadpt20]),
    ]

    for mask, truth in zip(
        masks,
        [
            nomuon & leadpt20,
            twoelectron & leadpt20,
            twoelectron & nomuon,
            twoelectron & nomuon & leadpt20,
        ],
    ):
        assert np.all(mask == truth)

    nminusone.to_npz("nminusone.npz", compressed=False).compute()
    with np.load("nminusone.npz") as file:
        assert np.all(file["labels"] == labels)
        assert np.all(file["nev"] == nev)
        assert np.all(file["masks"] == masks)
    os.remove("nminusone.npz")

    nminusone.to_npz("nminusone.npz", compressed=True).compute()
    with np.load("nminusone.npz") as file:
        assert np.all(file["labels"] == labels)
        assert np.all(file["nev"] == nev)
        assert np.all(file["masks"] == masks)
    os.remove("nminusone.npz")

    h, hlabels = nminusone.yieldhist()

    assert hlabels == ["initial", "N - twoElectron", "N - noMuon", "N - leadPt20", "N"]

    assert np.all(h.axes["N-1"].edges == np.arange(0, 6))

    assert np.all(h.counts() == nev)

    with pytest.raises(ValueError):
        nminusone.plot_vars(
            {"Ept": events.Electron.pt, "Ephi": events.Electron.phi[:20]}
        )
    hs, hslabels = nminusone.plot_vars(
        {"Ept": events.Electron.pt, "Ephi": events.Electron.phi}
    )

    assert hslabels == ["initial", "N - twoElectron", "N - noMuon", "N - leadPt20", "N"]

    for h, array in zip(hs, [events.Electron.pt, events.Electron.phi]):
        edges = h.axes[0].edges
        for i, truth in enumerate(
            [
                np.ones(40, dtype=bool),
                nomuon & leadpt20,
                twoelectron & leadpt20,
                twoelectron & nomuon,
                twoelectron & nomuon & leadpt20,
            ]
        ):
            counts = h[:, i].counts(flow=True)
            counts[1] += counts[0]
            counts[-2] += counts[-1]
            c, e = np.histogram(ak.flatten(array[truth]), bins=edges)
            assert np.all(np.isclose(counts[1:-1], c))


def test_packed_selection_cutflow():
    import awkward as ak

    from coffea.analysis_tools import PackedSelection

    events = eagerevents

    selection = PackedSelection()

    twoelectron = ak.num(events.Electron) == 2
    nomuon = ak.num(events.Muon) == 0
    leadpt20 = ak.any(events.Electron.pt >= 20.0, axis=1) | ak.any(
        events.Muon.pt >= 20.0, axis=1
    )

    selection.add_multiple(
        {
            "twoElectron": twoelectron,
            "noMuon": nomuon,
            "leadPt20": leadpt20,
        }
    )

    assert selection.names == ["twoElectron", "noMuon", "leadPt20"]

    with pytest.raises(
        ValueError,
        match="All arguments must be strings that refer to the names of existing selections",
    ):
        selection.cutflow("twoElectron", "nonexistent")
    cutflow = selection.cutflow("noMuon", "twoElectron", "leadPt20")

    labels, nevonecut, nevcutflow, masksonecut, maskscutflow = cutflow.result()

    assert labels == ["initial", "noMuon", "twoElectron", "leadPt20"]

    assert nevonecut == [
        len(events),
        len(events[nomuon]),
        len(events[twoelectron]),
        len(events[leadpt20]),
    ]

    assert nevcutflow == [
        len(events),
        len(events[nomuon]),
        len(events[nomuon & twoelectron]),
        len(events[nomuon & twoelectron & leadpt20]),
    ]

    for mask, truth in zip(masksonecut, [nomuon, twoelectron, leadpt20]):
        assert np.all(mask == truth)

    for mask, truth in zip(
        maskscutflow, [nomuon, nomuon & twoelectron, nomuon & twoelectron & leadpt20]
    ):
        assert np.all(mask == truth)

    cutflow.to_npz("cutflow.npz", compressed=False).compute()
    with np.load("cutflow.npz") as file:
        assert np.all(file["labels"] == labels)
        assert np.all(file["nevonecut"] == nevonecut)
        assert np.all(file["nevcutflow"] == nevcutflow)
        assert np.all(file["masksonecut"] == masksonecut)
        assert np.all(file["maskscutflow"] == maskscutflow)
    os.remove("cutflow.npz")

    cutflow.to_npz("cutflow.npz", compressed=True).compute()
    with np.load("cutflow.npz") as file:
        assert np.all(file["labels"] == labels)
        assert np.all(file["nevonecut"] == nevonecut)
        assert np.all(file["nevcutflow"] == nevcutflow)
        assert np.all(file["masksonecut"] == masksonecut)
        assert np.all(file["maskscutflow"] == maskscutflow)
    os.remove("cutflow.npz")

    honecut, hcutflow, hlabels = cutflow.yieldhist()

    assert hlabels == ["initial", "noMuon", "twoElectron", "leadPt20"]

    assert np.all(honecut.axes["onecut"].edges == np.arange(0, 5))
    assert np.all(hcutflow.axes["cutflow"].edges == np.arange(0, 5))

    assert np.all(honecut.counts() == nevonecut)
    assert np.all(hcutflow.counts() == nevcutflow)

    with pytest.raises(ValueError):
        cutflow.plot_vars({"Ept": events.Electron.pt, "Ephi": events.Electron.phi[:20]})
    honecuts, hcutflows, hslabels = cutflow.plot_vars(
        {"ept": events.Electron.pt, "ephi": events.Electron.phi}
    )

    assert hslabels == ["initial", "noMuon", "twoElectron", "leadPt20"]

    for h, array in zip(honecuts, [events.Electron.pt, events.Electron.phi]):
        edges = h.axes[0].edges
        for i, truth in enumerate(
            [np.ones(40, dtype=bool), nomuon, twoelectron, leadpt20]
        ):
            counts = h[:, i].counts(flow=True)
            counts[1] += counts[0]
            counts[-2] += counts[-1]
            c, e = np.histogram(ak.flatten(array[truth]), bins=edges)
            assert np.all(np.isclose(counts[1:-1], c))

    for h, array in zip(hcutflows, [events.Electron.pt, events.Electron.phi]):
        edges = h.axes[0].edges
        for i, truth in enumerate(
            [
                np.ones(40, dtype=bool),
                nomuon,
                nomuon & twoelectron,
                nomuon & twoelectron & leadpt20,
            ]
        ):
            counts = h[:, i].counts(flow=True)
            counts[1] += counts[0]
            counts[-2] += counts[-1]
            c, e = np.histogram(ak.flatten(array[truth]), bins=edges)
            assert np.all(np.isclose(counts[1:-1], c))


@pytest.mark.parametrize("optimization_enabled", [True, False])
@pytest.mark.parametrize("dtype", ["uint16", "uint32", "uint64"])
def test_packed_selection_basic_dak(optimization_enabled, dtype):
    import awkward as ak
    import dask
    import dask.array as da
    import dask_awkward as dak

    from coffea.analysis_tools import PackedSelection

    sel = PackedSelection(dtype=dtype)

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        shape = (10,)
        all_true = dak.from_awkward(
            ak.Array(np.full(shape=shape, fill_value=True, dtype=bool)), 1
        )
        all_false = dak.from_awkward(
            ak.Array(np.full(shape=shape, fill_value=False, dtype=bool)), 1
        )
        fizz = dak.from_awkward(ak.Array(np.arange(shape[0]) % 3 == 0), 1)
        buzz = dak.from_awkward(ak.Array(np.arange(shape[0]) % 5 == 0), 1)
        ones = dak.from_awkward(ak.Array(np.ones(shape=shape, dtype=np.uint64)), 1)
        wrong_shape = dak.from_awkward(
            ak.Array(np.ones(shape=(shape[0] - 5,), dtype=bool)), 1
        )
        wrong_type = np.arange(shape[0]) % 3 == 0
        daskarray = da.arange(shape[0]) % 3 == 0

    with pytest.warns(
        UserWarning,
        match="PackedSelection hasn't been initialized with a boolean array yet!",
    ):
        assert sel.delayed_mode is False

        sel.add("fizz", fizz)
        sel.add("buzz", buzz)

        assert np.all(
            sel.all().compute()
            == np.array(
                [True, False, False, False, False, False, False, False, False, False]
            )
        )
        assert np.all(
            sel.allfalse().compute()
            == np.array(
                [False, True, True, False, True, False, False, True, True, False]
            )
        )

        sel.add_multiple({"all_true": all_true, "all_false": all_false})

        assert sel.delayed_mode is True
        with pytest.raises(
            ValueError,
            match="New selection 'wrong_type' is not delayed while PackedSelection is!",
        ):
            sel.add("wrong_type", wrong_type)

        assert np.all(
            sel.require(all_true=True, all_false=False).compute() == all_true.compute()
        )
        # allow truthy values
        assert np.all(
            sel.require(all_true=1, all_false=0).compute() == all_true.compute()
        )
        assert np.all(sel.all("all_true", "all_false").compute() == all_false.compute())
        assert np.all(sel.any("all_true", "all_false").compute() == all_true.compute())
        assert np.all(
            sel.all("fizz", "buzz").compute()
            == np.array(
                [True, False, False, False, False, False, False, False, False, False]
            )
        )
        assert np.all(
            sel.allfalse("fizz", "buzz").compute()
            == np.array(
                [False, True, True, False, True, False, False, True, True, False]
            )
        )
        assert np.all(
            sel.any("fizz", "buzz").compute()
            == np.array(
                [True, False, False, True, False, True, True, False, False, True]
            )
        )

        with pytest.raises(
            ValueError,
            match="New selection 'wrong_shape' has a different partition structure than existing selections",
        ):
            sel.add("wrong_shape", wrong_shape)

        with pytest.raises(
            ValueError, match="Expected a boolean array, received uint64"
        ):
            sel.add("ones", ones)

        with pytest.raises(RuntimeError):
            overpack = PackedSelection(dtype=dtype)
            for i in range(65):
                overpack.add("sel_%d", all_true)

        with pytest.raises(
            ValueError,
            match="Dask arrays are not supported, please convert them to dask_awkward.Array by using dask_awkward.from_dask_array()",
        ):
            sel.add("dask_array", daskarray)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_packed_selection_nminusone_dak(optimization_enabled):
    import dask
    import dask_awkward as dak

    from coffea.analysis_tools import PackedSelection

    events = dakevents

    selection = PackedSelection()

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        twoelectron = dak.num(events.Electron) == 2
        nomuon = dak.num(events.Muon) == 0
        leadpt20 = dak.any(events.Electron.pt >= 20.0, axis=1) | dak.any(
            events.Muon.pt >= 20.0, axis=1
        )

        selection.add_multiple(
            {
                "twoElectron": twoelectron,
                "noMuon": nomuon,
                "leadPt20": leadpt20,
            }
        )

        assert selection.names == ["twoElectron", "noMuon", "leadPt20"]

        with pytest.raises(
            ValueError,
            match="All arguments must be strings that refer to the names of existing selections",
        ):
            selection.nminusone("twoElectron", "nonexistent")
        nminusone = selection.nminusone("twoElectron", "noMuon", "leadPt20")

        labels, nev, masks = nminusone.result()

        assert labels == [
            "initial",
            "N - twoElectron",
            "N - noMuon",
            "N - leadPt20",
            "N",
        ]

        assert list(dask.compute(*nev)) == [
            dak.num(events, axis=0).compute(),
            dak.num(events[nomuon & leadpt20], axis=0).compute(),
            dak.num(events[twoelectron & leadpt20], axis=0).compute(),
            dak.num(events[twoelectron & nomuon], axis=0).compute(),
            dak.num(events[twoelectron & nomuon & leadpt20], axis=0).compute(),
        ]

        for mask, truth in zip(
            masks,
            [
                nomuon & leadpt20,
                twoelectron & leadpt20,
                twoelectron & nomuon,
                twoelectron & nomuon & leadpt20,
            ],
        ):
            assert np.all(mask.compute() == truth.compute())

        nminusone.to_npz("nminusone.npz", compressed=False).compute()
        with np.load("nminusone.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nev"] == list(dask.compute(*nev)))
            assert np.all(file["masks"] == list(dask.compute(*masks)))
        os.remove("nminusone.npz")

        nminusone.to_npz("nminusone.npz", compressed=True).compute()
        with np.load("nminusone.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nev"] == list(dask.compute(*nev)))
            assert np.all(file["masks"] == list(dask.compute(*masks)))
        os.remove("nminusone.npz")

        h, hlabels = dask.compute(*nminusone.yieldhist())

        assert hlabels == [
            "initial",
            "N - twoElectron",
            "N - noMuon",
            "N - leadPt20",
            "N",
        ]

        assert np.all(h.axes["N-1"].edges == np.arange(0, 6))

        assert np.all(h.counts() == list(dask.compute(*nev)))

        # with pytest.raises(IncompatiblePartitions):
        #     nminusone.plot_vars(
        #         {"Ept": events.Electron.pt, "Ephi": events[:20].Electron.phi}
        #     )
        hs, hslabels = dask.compute(
            *nminusone.plot_vars(
                {"Ept": events.Electron.pt, "Ephi": events.Electron.phi}
            )
        )

        assert hslabels == [
            "initial",
            "N - twoElectron",
            "N - noMuon",
            "N - leadPt20",
            "N",
        ]

        for h, array in zip(hs, [events.Electron.pt, events.Electron.phi]):
            edges = h.axes[0].edges
            for i, truth in enumerate(
                [
                    np.ones(40, dtype=bool),
                    nomuon.compute() & leadpt20.compute(),
                    twoelectron.compute() & leadpt20.compute(),
                    twoelectron.compute() & nomuon.compute(),
                    twoelectron.compute() & nomuon.compute() & leadpt20.compute(),
                ]
            ):
                counts = h[:, i].counts(flow=True)
                counts[1] += counts[0]
                counts[-2] += counts[-1]
                c, e = np.histogram(dak.flatten(array[truth]).compute(), bins=edges)
                assert np.all(np.isclose(counts[1:-1], c))


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_packed_selection_cutflow_dak(optimization_enabled):
    import dask
    import dask_awkward as dak

    from coffea.analysis_tools import PackedSelection

    events = dakevents

    selection = PackedSelection()

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        twoelectron = dak.num(events.Electron) == 2
        nomuon = dak.num(events.Muon) == 0
        leadpt20 = dak.any(events.Electron.pt >= 20.0, axis=1) | dak.any(
            events.Muon.pt >= 20.0, axis=1
        )

        selection.add_multiple(
            {
                "twoElectron": twoelectron,
                "noMuon": nomuon,
                "leadPt20": leadpt20,
            }
        )

        assert selection.names == ["twoElectron", "noMuon", "leadPt20"]

        with pytest.raises(
            ValueError,
            match="All arguments must be strings that refer to the names of existing selections",
        ):
            selection.cutflow("twoElectron", "nonexistent")
        cutflow = selection.cutflow("noMuon", "twoElectron", "leadPt20")

        labels, nevonecut, nevcutflow, masksonecut, maskscutflow = cutflow.result()

        assert labels == ["initial", "noMuon", "twoElectron", "leadPt20"]

        assert list(dask.compute(*nevonecut)) == [
            dak.num(events, axis=0).compute(),
            dak.num(events[nomuon], axis=0).compute(),
            dak.num(events[twoelectron], axis=0).compute(),
            dak.num(events[leadpt20], axis=0).compute(),
        ]

        assert list(dask.compute(*nevcutflow)) == [
            dak.num(events, axis=0).compute(),
            dak.num(events[nomuon], axis=0).compute(),
            dak.num(events[nomuon & twoelectron], axis=0).compute(),
            dak.num(events[nomuon & twoelectron & leadpt20], axis=0).compute(),
        ]

        for mask, truth in zip(masksonecut, [nomuon, twoelectron, leadpt20]):
            assert np.all(mask.compute() == truth.compute())

        for mask, truth in zip(
            maskscutflow,
            [nomuon, nomuon & twoelectron, nomuon & twoelectron & leadpt20],
        ):
            assert np.all(mask.compute() == truth.compute())

        cutflow.to_npz("cutflow.npz", compressed=False).compute()
        with np.load("cutflow.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nevonecut"] == list(dask.compute(*nevonecut)))
            assert np.all(file["nevcutflow"] == list(dask.compute(*nevcutflow)))
            assert np.all(file["masksonecut"] == list(dask.compute(*masksonecut)))
            assert np.all(file["maskscutflow"] == list(dask.compute(*maskscutflow)))
        os.remove("cutflow.npz")

        cutflow.to_npz("cutflow.npz", compressed=True).compute()
        with np.load("cutflow.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nevonecut"] == list(dask.compute(*nevonecut)))
            assert np.all(file["nevcutflow"] == list(dask.compute(*nevcutflow)))
            assert np.all(file["masksonecut"] == list(dask.compute(*masksonecut)))
            assert np.all(file["maskscutflow"] == list(dask.compute(*maskscutflow)))
        os.remove("cutflow.npz")

        honecut, hcutflow, hlabels = dask.compute(*cutflow.yieldhist())

        assert hlabels == ["initial", "noMuon", "twoElectron", "leadPt20"]

        assert np.all(honecut.axes["onecut"].edges == np.arange(0, 5))
        assert np.all(hcutflow.axes["cutflow"].edges == np.arange(0, 5))

        assert np.all(honecut.counts() == list(dask.compute(*nevonecut)))
        assert np.all(hcutflow.counts() == list(dask.compute(*nevcutflow)))

        # with pytest.raises(IncompatiblePartitions):
        #     cutflow.plot_vars(
        #         {"Ept": events.Electron.pt, "Ephi": events[:20].Electron.phi}
        #     )
        honecuts, hcutflows, hslabels = dask.compute(
            *cutflow.plot_vars({"ept": events.Electron.pt, "ephi": events.Electron.phi})
        )

        assert hslabels == ["initial", "noMuon", "twoElectron", "leadPt20"]

        for h, array in zip(honecuts, [events.Electron.pt, events.Electron.phi]):
            edges = h.axes[0].edges
            for i, truth in enumerate(
                [
                    np.ones(40, dtype=bool),
                    nomuon.compute(),
                    twoelectron.compute(),
                    leadpt20.compute(),
                ]
            ):
                counts = h[:, i].counts(flow=True)
                counts[1] += counts[0]
                counts[-2] += counts[-1]
                c, e = np.histogram(dak.flatten(array[truth]).compute(), bins=edges)
                assert np.all(np.isclose(counts[1:-1], c))

        for h, array in zip(hcutflows, [events.Electron.pt, events.Electron.phi]):
            edges = h.axes[0].edges
            for i, truth in enumerate(
                [
                    np.ones(40, dtype=bool),
                    nomuon.compute(),
                    (nomuon & twoelectron).compute(),
                    (nomuon & twoelectron & leadpt20).compute(),
                ]
            ):
                counts = h[:, i].counts(flow=True)
                counts[1] += counts[0]
                counts[-2] += counts[-1]
                c, e = np.histogram(dak.flatten(array[truth]).compute(), bins=edges)
                assert np.all(np.isclose(counts[1:-1], c))


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_packed_selection_nminusone_dak_uproot_only(optimization_enabled):
    import dask
    import dask_awkward as dak

    from coffea.analysis_tools import PackedSelection

    events = uprootevents

    selection = PackedSelection()

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        twoelectron = dak.num(events.Electron_pt) == 2
        nomuon = dak.num(events.Muon_pt) == 0
        leadpt20 = dak.any(events.Electron_pt >= 20.0, axis=1) | dak.any(
            events.Muon_pt >= 20.0, axis=1
        )

        selection.add_multiple(
            {
                "twoElectron": twoelectron,
                "noMuon": nomuon,
                "leadPt20": leadpt20,
            }
        )

        assert selection.names == ["twoElectron", "noMuon", "leadPt20"]

        with pytest.raises(
            ValueError,
            match="All arguments must be strings that refer to the names of existing selections",
        ):
            selection.nminusone("twoElectron", "nonexistent")
        nminusone = selection.nminusone("twoElectron", "noMuon", "leadPt20")

        labels, nev, masks = nminusone.result()

        assert labels == [
            "initial",
            "N - twoElectron",
            "N - noMuon",
            "N - leadPt20",
            "N",
        ]

        assert list(dask.compute(*nev)) == [
            dak.num(events, axis=0).compute(),
            dak.num(events[nomuon & leadpt20], axis=0).compute(),
            dak.num(events[twoelectron & leadpt20], axis=0).compute(),
            dak.num(events[twoelectron & nomuon], axis=0).compute(),
            dak.num(events[twoelectron & nomuon & leadpt20], axis=0).compute(),
        ]

        for mask, truth in zip(
            masks,
            [
                nomuon & leadpt20,
                twoelectron & leadpt20,
                twoelectron & nomuon,
                twoelectron & nomuon & leadpt20,
            ],
        ):
            assert np.all(mask.compute() == truth.compute())

        nminusone.to_npz("nminusone.npz", compressed=False).compute()
        with np.load("nminusone.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nev"] == list(dask.compute(*nev)))
            assert np.all(file["masks"] == list(dask.compute(*masks)))
        os.remove("nminusone.npz")

        nminusone.to_npz("nminusone.npz", compressed=True).compute()
        with np.load("nminusone.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nev"] == list(dask.compute(*nev)))
            assert np.all(file["masks"] == list(dask.compute(*masks)))
        os.remove("nminusone.npz")

        h, hlabels = dask.compute(*nminusone.yieldhist())

        assert hlabels == [
            "initial",
            "N - twoElectron",
            "N - noMuon",
            "N - leadPt20",
            "N",
        ]

        assert np.all(h.axes["N-1"].edges == np.arange(0, 6))

        assert np.all(h.counts() == list(dask.compute(*nev)))

        # with pytest.raises(IncompatiblePartitions):
        #     nminusone.plot_vars(
        #         {"Ept": events.Electron_pt, "Ephi": events[:20].Electron_phi}
        #     )
        hs, hslabels = dask.compute(
            *nminusone.plot_vars(
                {"Ept": events.Electron_pt, "Ephi": events.Electron_phi}
            )
        )

        assert hslabels == [
            "initial",
            "N - twoElectron",
            "N - noMuon",
            "N - leadPt20",
            "N",
        ]

        for h, array in zip(hs, [events.Electron_pt, events.Electron_phi]):
            edges = h.axes[0].edges
            for i, truth in enumerate(
                [
                    np.ones(40, dtype=bool),
                    nomuon.compute() & leadpt20.compute(),
                    twoelectron.compute() & leadpt20.compute(),
                    twoelectron.compute() & nomuon.compute(),
                    twoelectron.compute() & nomuon.compute() & leadpt20.compute(),
                ]
            ):
                counts = h[:, i].counts(flow=True)
                counts[1] += counts[0]
                counts[-2] += counts[-1]
                c, e = np.histogram(dak.flatten(array[truth]).compute(), bins=edges)
                assert np.all(np.isclose(counts[1:-1], c))


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_packed_selection_cutflow_dak_uproot_only(optimization_enabled):
    import dask
    import dask_awkward as dak

    from coffea.analysis_tools import PackedSelection

    events = uprootevents

    selection = PackedSelection()

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        twoelectron = dak.num(events.Electron_pt) == 2
        nomuon = dak.num(events.Muon_pt) == 0
        leadpt20 = dak.any(events.Electron_pt >= 20.0, axis=1) | dak.any(
            events.Muon_pt >= 20.0, axis=1
        )

        selection.add_multiple(
            {
                "twoElectron": twoelectron,
                "noMuon": nomuon,
                "leadPt20": leadpt20,
            }
        )

        assert selection.names == ["twoElectron", "noMuon", "leadPt20"]

        with pytest.raises(
            ValueError,
            match="All arguments must be strings that refer to the names of existing selections",
        ):
            selection.cutflow("twoElectron", "nonexistent")
        cutflow = selection.cutflow("noMuon", "twoElectron", "leadPt20")

        labels, nevonecut, nevcutflow, masksonecut, maskscutflow = cutflow.result()

        assert labels == ["initial", "noMuon", "twoElectron", "leadPt20"]

        assert list(dask.compute(*nevonecut)) == [
            dak.num(events, axis=0).compute(),
            dak.num(events[nomuon], axis=0).compute(),
            dak.num(events[twoelectron], axis=0).compute(),
            dak.num(events[leadpt20], axis=0).compute(),
        ]

        assert list(dask.compute(*nevcutflow)) == [
            dak.num(events, axis=0).compute(),
            dak.num(events[nomuon], axis=0).compute(),
            dak.num(events[nomuon & twoelectron], axis=0).compute(),
            dak.num(events[nomuon & twoelectron & leadpt20], axis=0).compute(),
        ]

        for mask, truth in zip(masksonecut, [nomuon, twoelectron, leadpt20]):
            assert np.all(mask.compute() == truth.compute())

        for mask, truth in zip(
            maskscutflow,
            [nomuon, nomuon & twoelectron, nomuon & twoelectron & leadpt20],
        ):
            assert np.all(mask.compute() == truth.compute())

        cutflow.to_npz("cutflow.npz", compressed=False).compute()
        with np.load("cutflow.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nevonecut"] == list(dask.compute(*nevonecut)))
            assert np.all(file["nevcutflow"] == list(dask.compute(*nevcutflow)))
            assert np.all(file["masksonecut"] == list(dask.compute(*masksonecut)))
            assert np.all(file["maskscutflow"] == list(dask.compute(*maskscutflow)))
        os.remove("cutflow.npz")

        cutflow.to_npz("cutflow.npz", compressed=True).compute()
        with np.load("cutflow.npz") as file:
            assert np.all(file["labels"] == labels)
            assert np.all(file["nevonecut"] == list(dask.compute(*nevonecut)))
            assert np.all(file["nevcutflow"] == list(dask.compute(*nevcutflow)))
            assert np.all(file["masksonecut"] == list(dask.compute(*masksonecut)))
            assert np.all(file["maskscutflow"] == list(dask.compute(*maskscutflow)))
        os.remove("cutflow.npz")

        honecut, hcutflow, hlabels = dask.compute(*cutflow.yieldhist())

        assert hlabels == ["initial", "noMuon", "twoElectron", "leadPt20"]

        assert np.all(honecut.axes["onecut"].edges == np.arange(0, 5))
        assert np.all(hcutflow.axes["cutflow"].edges == np.arange(0, 5))

        assert np.all(honecut.counts() == list(dask.compute(*nevonecut)))
        assert np.all(hcutflow.counts() == list(dask.compute(*nevcutflow)))

        # with pytest.raises(IncompatiblePartitions):
        #     cutflow.plot_vars(
        #         {"Ept": events.Electron_pt, "Ephi": events[:20].Electron_phi}
        #     )
        honecuts, hcutflows, hslabels = dask.compute(
            *cutflow.plot_vars({"ept": events.Electron_pt, "ephi": events.Electron_phi})
        )

        assert hslabels == ["initial", "noMuon", "twoElectron", "leadPt20"]

        for h, array in zip(honecuts, [events.Electron_pt, events.Electron_phi]):
            edges = h.axes[0].edges
            for i, truth in enumerate(
                [
                    np.ones(40, dtype=bool),
                    nomuon.compute(),
                    twoelectron.compute(),
                    leadpt20.compute(),
                ]
            ):
                counts = h[:, i].counts(flow=True)
                counts[1] += counts[0]
                counts[-2] += counts[-1]
                c, e = np.histogram(dak.flatten(array[truth]).compute(), bins=edges)
                assert np.all(np.isclose(counts[1:-1], c))

        for h, array in zip(hcutflows, [events.Electron_pt, events.Electron_phi]):
            edges = h.axes[0].edges
            for i, truth in enumerate(
                [
                    np.ones(40, dtype=bool),
                    nomuon.compute(),
                    (nomuon & twoelectron).compute(),
                    (nomuon & twoelectron & leadpt20).compute(),
                ]
            ):
                counts = h[:, i].counts(flow=True)
                counts[1] += counts[0]
                counts[-2] += counts[-1]
                c, e = np.histogram(dak.flatten(array[truth]).compute(), bins=edges)
                assert np.all(np.isclose(counts[1:-1], c))
