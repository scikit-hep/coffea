import awkward as ak
import dask_awkward as dak
import numpy as np
import pytest
from distributed import Client


def prepare_jets_array(njets):
    # Creating jagged Jet-with-constituent array, returning both awkward and lazy
    # dask_awkward arrays
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

    ak_jets = jets[:]
    ak.to_parquet(jets, "ml_tools.parquet")
    dak_jets = dak.from_parquet("ml_tools.parquet")
    return ak_jets, dak_jets


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


def test_triton():
    _ = pytest.importorskip("tritonclient")

    from coffea.ml_tools.triton_wrapper import triton_wrapper

    client = Client()  # Spawn local cluster

    # Defining custom wrapper function with awkward padding requirements.
    class triton_wrapper_test(triton_wrapper):
        def prepare_awkward(self, output_list, jets):
            return [], {
                "output_list": output_list,
                "input_dict": common_prepare_awkward(jets),
            }

    # Running the evaluation in lazy and non-lazy forms
    tw = triton_wrapper_test(
        model_url="triton+grpc://localhost:8001/pn_test/1",
        client_args=dict(
            ssl=False
        ),  # Solves SSL version mismatch for local inference server
    )

    ak_jets, dak_jets = prepare_jets_array(njets=256)

    # Vanilla awkward arrays
    ak_res = tw(["output"], ak_jets)
    dak_res = tw(["output"], dak_jets)

    for k in ak_res.keys():
        assert ak.all(ak_res[k] == dak_res[k].compute())
    expected_columns = {
        "eta",
        "phi",
        "pfcands.pt",
        "pfcands.phi",
        "pfcands.eta",
        "pfcands.feat1",
        "pfcands.feat2",
    }
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == expected_columns

    client.close()


def test_torch():
    _ = pytest.importorskip("torch")

    from coffea.ml_tools.torch_wrapper import torch_wrapper

    client = Client()  # Spawn local cluster

    class torch_wrapper_test(torch_wrapper):
        def prepare_awkward(self, jets):
            default = common_prepare_awkward(jets)
            return [], {
                "points": ak.values_astype(default["points"], np.float32),
                "features": ak.values_astype(default["features"], np.float32),
                "mask": ak.values_astype(default["mask"], np.float16),
            }

    tw = torch_wrapper_test("tests/samples/pn_demo.pt")
    ak_jets, dak_jets = prepare_jets_array(njets=256)

    ak_res = tw(ak_jets)
    dak_res = tw(dak_jets)

    assert np.all(np.isclose(ak_res, dak_res.compute()))
    expected_columns = {
        "eta",
        "phi",
        "pfcands.pt",
        "pfcands.phi",
        "pfcands.eta",
        "pfcands.feat1",
        "pfcands.feat2",
    }
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == expected_columns
    client.close()


def test_tensorflow():
    _ = pytest.importorskip("tensorflow")

    from coffea.ml_tools.tf_wrapper import tf_wrapper

    client = Client()  # Spawn local cluster

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
    tfw = tf_wrapper_test("tests/samples/tf_model.keras")
    ak_jets, dak_jets = prepare_jets_array(njets=256)

    ak_res = tfw(ak_jets)
    dak_res = tfw(dak_jets)

    assert np.all(np.isclose(ak_res, dak_res.compute()))
    expected_columns = {"ncands"} | {f"pfcands.feat{i}" for i in range(1, 19)}
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == expected_columns
    client.close()


def test_xgboost():
    _ = pytest.importorskip("xgboost")

    from coffea.ml_tools.xgboost_wrapper import xgboost_wrapper

    client = Client()  # Spawn local cluster

    feature_list = [f"feat{i}" for i in range(16)]

    class xgboost_test(xgboost_wrapper):
        def prepare_awkward(self, events):
            ret = ak.concatenate(
                [events[name][:, np.newaxis] for name in feature_list], axis=1
            )
            return [], dict(data=ret)

    xgb_wrap = xgboost_test("tests/samples/xgboost_example.xgb")

    # Dummy 1000 event array with 20 feature branches
    ak_events = ak.zip(
        {f"feat{i}": ak.from_numpy(np.random.random(size=1_000)) for i in range(20)}
    )
    ak.to_parquet(ak_events, "ml_tools.xgboost.parquet")
    dak_events = dak.from_parquet("ml_tools.xgboost.parquet")

    ak_res = xgb_wrap(ak_events)
    dak_res = xgb_wrap(dak_events)

    # Results should be identical
    assert ak.all(ak_res == dak_res.compute())

    # Should only load required columns
    columns = set(list(dak.necessary_columns(dak_res).values())[0])
    assert columns == set(feature_list)
    client.close()
