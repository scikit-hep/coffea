import os

import awkward as ak
import numpy as np
import pytest

# Columns the jet wrappers below are expected to touch.
EXPECTED_JET_COLUMNS = {
    "eta",
    "phi",
    "pfcands.pt",
    "pfcands.phi",
    "pfcands.eta",
    "pfcands.feat1",
    "pfcands.feat2",
}

XGBOOST_FEATURES = [f"feat{i}" for i in range(16)]


def _dask_awkward():
    """dask_awkward, skipping the calling test when it is not installed.

    Imported per test rather than at module scope so the eager tests run in an
    installation without dask_awkward.
    """
    return pytest.importorskip("dask_awkward")


def prepare_jets_array(njets):
    # Creating jagged Jet-with-constituent array
    NFEAT = 100
    jets = ak.zip(
        {
            "pt": ak.from_numpy(np.random.random(size=njets)),
            "eta": ak.from_numpy(np.random.random(size=njets)),
            "phi": ak.from_numpy(np.random.random(size=njets)),
            "ncands": ak.from_numpy(np.random.randint(1, 50, size=njets)),
        }
    )
    pfcands = ak.zip(
        {
            "pt": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "eta": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "phi": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "feat1": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "feat2": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            # Extra features for testing Tensorflow model for PFCandidate classification
            **{
                f"feat{i}": ak.from_regular(np.random.random(size=(njets, NFEAT)))
                for i in range(3, 19)
            },
        }
    )

    idx = ak.local_index(pfcands.pt, axis=-1)
    pfcands = pfcands[idx < jets.ncands]
    jets["pfcands"] = pfcands[:]

    return jets[:]


def to_dask(arr, path):
    """Round-trip an eager array through parquet into a lazy dask_awkward array."""
    dak = _dask_awkward()
    ak.to_parquet(arr, path)
    return dak.from_parquet(path)


def common_prepare_awkward(jets):
    """Common jet parsing routing for pytorch and triton inference"""

    def my_pad(arr):
        return ak.fill_none(ak.pad_none(arr, 100, axis=1, clip=True), 0.0)

    fmap = {
        "points": {
            "deta": my_pad(jets.eta - jets.pfcands.eta),
            "dphi": my_pad(jets.phi - jets.pfcands.phi),
        },
        "features": {
            "dr": my_pad(
                np.sqrt(
                    (jets.eta - jets.pfcands.eta) ** 2
                    + (jets.phi - jets.pfcands.phi) ** 2
                )
            ),
            "lpt": my_pad(np.log(jets.pfcands.pt)),
            "lptf": my_pad(np.log(jets.pfcands.pt / ak.sum(jets.pfcands.pt, axis=-1))),
            "f1": my_pad(np.log(jets.pfcands.feat1 + 1)),
            "f2": my_pad(np.log(jets.pfcands.feat2 + 1)),
        },
        "mask": {
            "mask": my_pad(ak.ones_like(jets.pfcands.pt)),
        },
    }

    return {
        k: ak.concatenate([x[:, np.newaxis, :] for x in fmap[k].values()], axis=1)
        for k in fmap.keys()
    }


def _make_triton_wrapper():
    pytest.importorskip("tritonclient")
    if os.environ.get("TRITON_UNAVAILABLE") == "1":
        # Set by CI when the server image could not be pulled; a server that did
        # start is still expected to answer, so this does not mask real failures.
        pytest.skip("triton inference server image unavailable")

    from coffea.ml_tools.triton_wrapper import triton_wrapper

    # Defining custom wrapper function with awkward padding requirements.
    class triton_wrapper_test(triton_wrapper):
        def prepare_awkward(self, output_list, jets):
            return [], {
                "output_list": output_list,
                "input_dict": common_prepare_awkward(jets),
            }

    return triton_wrapper_test(
        model_url="triton+grpc://localhost:8001/pn_test/1",
        client_args=dict(
            ssl=False
        ),  # Solves SSL version mismatch for local inference server
    )


def test_triton_eager(tmp_path):
    tw = _make_triton_wrapper()
    ak_jets = prepare_jets_array(njets=256)

    ak_res = tw(["output"], ak_jets)
    for k in ak_res.keys():
        assert len(ak_res[k]) == len(ak_jets)

    # Length 0 tests
    ak_res = tw(["output"], ak_jets[ak_jets.eta < 0])
    for k in ak_res.keys():
        assert len(ak_res[k]) == 0


@pytest.mark.dask_client
def test_triton(tmp_path, dask_client):
    dak = _dask_awkward()
    tw = _make_triton_wrapper()

    ak_jets = prepare_jets_array(njets=256)
    dak_jets = to_dask(ak_jets, str(tmp_path / "ml_tools.parquet"))

    # Vanilla awkward arrays
    ak_res = tw(["output"], ak_jets)
    dak_res = tw(["output"], dak_jets)

    for k in ak_res.keys():
        assert ak.all(ak_res[k] == dak_res[k].compute())
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == EXPECTED_JET_COLUMNS

    # Length 0 tests
    ak_res = tw(["output"], ak_jets[ak_jets.eta < 0])
    dak_res = tw(["output"], dak_jets[dak_jets.eta < 0])
    for k in ak_res.keys():
        assert len(ak_res[k]) == 0 and len(dak_res[k].compute()) == 0


def _make_torch_wrapper(**kwargs):
    pytest.importorskip("torch")

    from coffea.ml_tools.torch_wrapper import torch_wrapper

    class torch_wrapper_test(torch_wrapper):
        def prepare_awkward(self, jets):
            default = common_prepare_awkward(jets)
            return [], {
                "points": ak.values_astype(default["points"], np.float32),
                "features": ak.values_astype(default["features"], np.float32),
                "mask": ak.values_astype(default["mask"], np.float16),
            }

    return torch_wrapper_test("tests/samples/pn_demo.pt", **kwargs)


def test_torch_eager(tmp_path):
    tw = _make_torch_wrapper()
    ak_jets = prepare_jets_array(njets=256)
    assert len(tw(ak_jets)) == len(ak_jets)

    # Length-0 testing
    tw = _make_torch_wrapper(expected_output_shape=(None,))
    ak_jets = prepare_jets_array(njets=256)
    ak_jets = ak_jets[ak_jets.eta < -100]  # Mimicking a low efficiency selection
    assert len(ak_jets) == 0
    assert len(tw(ak_jets)) == 0


@pytest.mark.dask_client
def test_torch(tmp_path, dask_client):
    dak = _dask_awkward()
    tw = _make_torch_wrapper()

    ak_jets = prepare_jets_array(njets=256)
    dak_jets = to_dask(ak_jets, str(tmp_path / "ml_tools.parquet"))
    ak_res = tw(ak_jets)
    dak_res = tw(dak_jets)

    assert np.all(np.isclose(ak_res, dak_res.compute()))
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == EXPECTED_JET_COLUMNS

    # Length-0 testing
    tw = _make_torch_wrapper(expected_output_shape=(None,))
    ak_jets = prepare_jets_array(njets=256)
    dak_jets = to_dask(ak_jets, str(tmp_path / "ml_tools_len0.parquet"))
    ak_jets = ak_jets[ak_jets.eta < -100]  # Mimicking a low efficiency selection
    dak_jets = dak_jets[dak_jets.eta < -100]
    ak_res, dak_res = tw(ak_jets), tw(dak_jets)
    assert len(ak_jets) == 0 and len(dak_res.compute()) == 0


def _make_tf_wrapper():
    pytest.importorskip("tensorflow")

    from coffea.ml_tools.tf_wrapper import tf_wrapper

    class tf_wrapper_test(tf_wrapper):
        def prepare_awkward(self, jets):
            # List of PF candidate features used for computation
            features = [f"feat{i}" for i in range(1, 19)]

            cands = ak.concatenate(
                [
                    # Filling pad with dummy value
                    ak.fill_none(
                        ak.pad_none(jets.pfcands[f], 64),
                        0,
                        axis=1,
                    )[..., np.newaxis]
                    for f in features
                ],
                axis=2,
            )
            cands = ak.flatten(cands, axis=None)  # Flatten everything
            cands = ak.unflatten(cands, 18)  # Number of features
            cands = ak.unflatten(cands, 64)  # Number of target entries

            return [cands], {}

        def postprocess_awkward(self, ret, jets):
            # First arguments is the return object of the models method
            ret = ret[:, :, 0]  # Flattening to get the per candidate entry
            ret = ak.from_regular(ret)  # Making this into a jagged array
            ret = ret[ak.local_index(ret) < jets.ncands]
            return ret

    # The tensorflow model here is used to classify jet constitutes
    return tf_wrapper_test("tests/samples/tf_model.keras")


def _make_tf_length0_wrapper():
    pytest.importorskip("tensorflow")

    from coffea.ml_tools.tf_wrapper import tf_wrapper

    # Length 0 testing. we cannot use the unflatten module in this case
    class tf_wrapper_length0_test(tf_wrapper):
        def prepare_awkward(self, arr):
            return [arr], {}

    return tf_wrapper_length0_test(
        "tests/samples/tf_model.keras", skip_length_zero=True
    )


def test_tensorflow_eager(tmp_path):
    tfw = _make_tf_wrapper()
    ak_jets = prepare_jets_array(njets=256)
    assert len(tfw(ak_jets)) == len(ak_jets)

    tfw_length0_tester = _make_tf_length0_wrapper()

    # Making an explicit shape
    arr = ak.from_numpy(np.random.random(size=(10, 64, 18)))
    assert len(tfw_length0_tester(arr)) == 10
    # Reducing the length 0
    arr = ak.from_numpy(np.zeros(shape=(0, 64, 18)))
    tfw_length0_tester(arr)


@pytest.mark.dask_client
def test_tensorflow(tmp_path, dask_client):
    dak = _dask_awkward()
    tfw = _make_tf_wrapper()

    ak_jets = prepare_jets_array(njets=256)
    dak_jets = to_dask(ak_jets, str(tmp_path / "ml_tools.parquet"))

    ak_res = tfw(ak_jets)
    dak_res = tfw(dak_jets)

    assert np.all(np.isclose(ak_res, dak_res.compute()))
    expected_columns = {"ncands"} | {f"pfcands.feat{i}" for i in range(1, 19)}
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == expected_columns

    tfw_length0_tester = _make_tf_length0_wrapper()

    # Making an explicit shape
    arr = ak.from_numpy(np.random.random(size=(10, 64, 18)))
    darr = to_dask(arr, str(tmp_path / "tf_length10.parquet"))
    ak_res = tfw_length0_tester(arr)
    dak_res = tfw_length0_tester(darr)
    assert np.all(np.isclose(ak_res, dak_res.compute()))
    # Reducing the length 0
    arr = ak.from_numpy(np.zeros(shape=(0, 64, 18)))
    darr = to_dask(arr, str(tmp_path / "tf_length0.parquet"))
    ak_res = tfw_length0_tester(arr)
    dak_res = tfw_length0_tester(darr)


def _make_xgboost_wrapper():
    pytest.importorskip("xgboost")

    from coffea.ml_tools.xgboost_wrapper import xgboost_wrapper

    class xgboost_test(xgboost_wrapper):
        def prepare_awkward(self, events):
            ret = ak.concatenate(
                [events[name][:, np.newaxis] for name in XGBOOST_FEATURES], axis=1
            )
            return [], dict(data=ret)

    return xgboost_test("tests/samples/xgboost_example.ubj")


def _xgboost_events(nevents=1_000):
    # Dummy event array with 20 feature branches
    return ak.zip(
        {f"feat{i}": ak.from_numpy(np.random.random(size=nevents)) for i in range(20)}
    )


def test_xgboost_eager(tmp_path):
    xgb_wrap = _make_xgboost_wrapper()
    ak_events = _xgboost_events()

    assert len(xgb_wrap(ak_events)) == len(ak_events)

    # Length 0 testing, xgboost always handles 0-length arrays elegantly
    assert len(xgb_wrap(ak_events[ak_events.feat0 < 0])) == 0


@pytest.mark.dask_client
def test_xgboost(tmp_path, dask_client):
    dak = _dask_awkward()
    xgb_wrap = _make_xgboost_wrapper()

    ak_events = _xgboost_events()
    dak_events = to_dask(ak_events, str(tmp_path / "ml_tools.xgboost.parquet"))

    ak_res = xgb_wrap(ak_events)
    dak_res = xgb_wrap(dak_events)

    # Results should be identical
    assert ak.all(ak_res == dak_res.compute())

    # Should only load required columns
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == set(XGBOOST_FEATURES)

    # Length 0 testing, xgboost always handles 0-length arrays elegantly
    ak_res = xgb_wrap(ak_events[ak_events.feat0 < 0])
    dak_res = xgb_wrap(dak_events[dak_events.feat0 < 0])
