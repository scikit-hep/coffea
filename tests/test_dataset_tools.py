import copy
import json

import awkward
import pytest
import uproot
from uproot.exceptions import KeyInFileError

from coffea.dataset_tools import (
    apply_to_fileset,
    filter_files,
    get_failed_steps_for_fileset,
    hash_fileset,
    max_chunks,
    max_chunks_per_file,
    max_files,
    preprocess,
    slice_chunks,
    slice_files,
    split_fileset,
)
from coffea.dataset_tools.filespec import (
    DataGroupSpec,
    DatasetSpec,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema
from coffea.processor.test_items import NanoEventsProcessor, NanoTestProcessor
from coffea.util import decompress_form

dask_awkward = pytest.importorskip("dask_awkward")

import dask  # noqa: E402

_starting_fileset_list = {
    "ZJets": ["tests/samples/nano_dy.root:Events"],
    "Data": [
        "tests/samples/nano_dimuon.root:Events",
        "tests/samples/nano_dimuon_not_there.root:Events",
    ],
}

_starting_fileset_dict = {
    "ZJets": {"tests/samples/nano_dy.root": "Events"},
    "Data": {
        "tests/samples/nano_dimuon.root": "Events",
        "tests/samples/nano_dimuon_not_there.root": "Events",
    },
}

_starting_fileset = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": "Events",
            "tests/samples/nano_dimuon_not_there.root": "Events",
        }
    },
}

_starting_fileset_with_steps = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            },
            "tests/samples/nano_dimuon_not_there.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            },
        }
    },
}

_runnable_result = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        },
        "metadata": None,
        "form": None,
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
            }
        },
        "metadata": None,
        "form": None,
    },
}

_updated_result = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        },
        "metadata": None,
        "form": None,
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "num_entries": 40,
                "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
            },
            "tests/samples/nano_dimuon_not_there.root": {
                "object_path": "Events",
                "steps": None,
                "num_entries": None,
                "uuid": None,
            },
        },
        "metadata": None,
        "form": None,
    },
}

_fileset_with_empty_files = {
    "only_empty": {
        "files": {
            "tests/samples/nano_dy_empty.root": "Events",
        },
    },
    "nonempty_and_empty": {
        "files": {
            "tests/samples/nano_dy.root": "Events",
            "tests/samples/nano_dy_empty.root": "Events",
        },
    },
    "empty_and_nonempty": {
        "files": {
            "tests/samples/nano_dy_empty.root": "Events",
            "tests/samples/nano_dy.root": "Events",
        },
    },
    "only_nonempty": {
        "files": {
            "tests/samples/nano_dy.root": "Events",
        },
    },
}

with open("tests/samples/fileset_with_empty_files_compressed_form_base.json") as f:
    _fileset_with_empty_files_compressed_form_base = json.load(f)

_fileset_with_empty_files_preprocessed = {
    "only_empty": {
        "files": {
            "tests/samples/nano_dy_empty.root": {
                "object_path": "Events",
                "steps": [[0, 0]],
                "num_entries": 0,
                "uuid": "f73b274c-da3c-11f0-b00b-2100a8c0beef",
            }
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
    "nonempty_and_empty": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            },
            "tests/samples/nano_dy_empty.root": {
                "object_path": "Events",
                "steps": [[0, 0]],
                "num_entries": 0,
                "uuid": "f73b274c-da3c-11f0-b00b-2100a8c0beef",
            },
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
    "empty_and_nonempty": {
        "files": {
            "tests/samples/nano_dy_empty.root": {
                "object_path": "Events",
                "steps": [[0, 0]],
                "num_entries": 0,
                "uuid": "f73b274c-da3c-11f0-b00b-2100a8c0beef",
            },
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            },
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
    "only_nonempty": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
}

_fileset_with_empty_files_preprocessed_aligned = {
    "only_empty": {
        "files": {
            "tests/samples/nano_dy_empty.root": {
                "object_path": "Events",
                "steps": [[0, 0]],
                "num_entries": 0,
                "uuid": "f73b274c-da3c-11f0-b00b-2100a8c0beef",
            }
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
    "nonempty_and_empty": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 40]],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            },
            "tests/samples/nano_dy_empty.root": {
                "object_path": "Events",
                "steps": [[0, 0]],
                "num_entries": 0,
                "uuid": "f73b274c-da3c-11f0-b00b-2100a8c0beef",
            },
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
    "empty_and_nonempty": {
        "files": {
            "tests/samples/nano_dy_empty.root": {
                "object_path": "Events",
                "steps": [[0, 0]],
                "num_entries": 0,
                "uuid": "f73b274c-da3c-11f0-b00b-2100a8c0beef",
            },
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 40]],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            },
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
    "only_nonempty": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [[0, 40]],
                "num_entries": 40,
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        },
        "compressed_form": _fileset_with_empty_files_compressed_form_base,
        "metadata": None,
    },
}


def _my_analysis_output_2(events):
    return events.Electron.pt, events.Muon.pt


def _my_analysis_output_3(events):
    return events.Electron.pt, events.Muon.pt, events.Tau.pt


@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_tuple_data_manipulation_output(allow_read_errors_with_report):
    out = apply_to_fileset(
        _my_analysis_output_2,
        _runnable_result,
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
    )

    if allow_read_errors_with_report:
        assert isinstance(out, tuple)
        assert len(out) == 2
        out, report = out
        assert isinstance(out, dict)
        assert isinstance(report, dict)
        assert out.keys() == {"ZJets", "Data"}
        assert report.keys() == {"ZJets", "Data"}
        assert isinstance(out["ZJets"], tuple)
        assert isinstance(out["Data"], tuple)
        assert len(out["ZJets"]) == 2
        assert len(out["Data"]) == 2
        for i, j in zip(out["ZJets"], out["Data"]):
            assert isinstance(i, dask_awkward.Array)
            assert isinstance(j, dask_awkward.Array)
        assert isinstance(report["ZJets"], dask_awkward.Array)
        assert isinstance(report["Data"], dask_awkward.Array)
    else:
        assert isinstance(out, dict)
        assert len(out) == 2
        assert out.keys() == {"ZJets", "Data"}
        assert isinstance(out["ZJets"], tuple)
        assert isinstance(out["Data"], tuple)
        assert len(out["ZJets"]) == 2
        assert len(out["Data"]) == 2
        for i, j in zip(out["ZJets"], out["Data"]):
            assert isinstance(i, dask_awkward.Array)
            assert isinstance(j, dask_awkward.Array)

    out = apply_to_fileset(
        _my_analysis_output_3,
        _runnable_result,
        uproot_options={"allow_read_errors_with_report": allow_read_errors_with_report},
    )

    if allow_read_errors_with_report:
        assert isinstance(out, tuple)
        assert len(out) == 2
        out, report = out
        assert isinstance(out, dict)
        assert isinstance(report, dict)
        assert out.keys() == {"ZJets", "Data"}
        assert report.keys() == {"ZJets", "Data"}
        assert isinstance(out["ZJets"], tuple)
        assert isinstance(out["Data"], tuple)
        assert len(out["ZJets"]) == 3
        assert len(out["Data"]) == 3
        for i, j in zip(out["ZJets"], out["Data"]):
            assert isinstance(i, dask_awkward.Array)
            assert isinstance(j, dask_awkward.Array)
        assert isinstance(report["ZJets"], dask_awkward.Array)
        assert isinstance(report["Data"], dask_awkward.Array)
    else:
        assert isinstance(out, dict)
        assert len(out) == 2
        assert out.keys() == {"ZJets", "Data"}
        assert isinstance(out["ZJets"], tuple)
        assert isinstance(out["Data"], tuple)
        assert len(out["ZJets"]) == 3
        assert len(out["Data"]) == 3
        for i, j in zip(out["ZJets"], out["Data"]):
            assert isinstance(i, dask_awkward.Array)
            assert isinstance(j, dask_awkward.Array)


@pytest.mark.dask_client
@pytest.mark.parametrize(
    "proc_and_schema",
    [(NanoTestProcessor, BaseSchema), (NanoEventsProcessor, NanoAODSchema)],
)
def test_apply_to_fileset(proc_and_schema, dask_client):
    proc, schemaclass = proc_and_schema

    with dask_client.as_current() as _:
        to_compute = apply_to_fileset(
            proc(),
            _runnable_result,
            schemaclass=schemaclass,
        )
        out = dask.compute(to_compute)[0]

        assert out["ZJets"]["cutflow"]["ZJets_pt"] == 18
        assert out["ZJets"]["cutflow"]["ZJets_mass"] == 6
        assert out["Data"]["cutflow"]["Data_pt"] == 84
        assert out["Data"]["cutflow"]["Data_mass"] == 66

        to_compute = apply_to_fileset(
            proc(),
            max_chunks(_runnable_result, 1),
            schemaclass=schemaclass,
        )
        out = dask.compute(to_compute)[0]

        assert out["ZJets"]["cutflow"]["ZJets_pt"] == 5
        assert out["ZJets"]["cutflow"]["ZJets_mass"] == 2
        assert out["Data"]["cutflow"]["Data_pt"] == 17
        assert out["Data"]["cutflow"]["Data_mass"] == 14


@pytest.mark.dask_client
@pytest.mark.parametrize(
    "the_fileset",
    [_starting_fileset, DataGroupSpec(_starting_fileset)],
)
def test_apply_to_fileset_hinted_form(the_fileset, dask_client):
    with dask_client.as_current() as _:
        dataset_runnable, dataset_updated = preprocess(
            the_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=True,
        )

        to_compute = apply_to_fileset(
            NanoEventsProcessor(),
            dataset_runnable,
            schemaclass=NanoAODSchema,
        )
        out = dask.compute(to_compute)[0]
        assert out["ZJets"]["cutflow"]["ZJets_pt"] == 18
        assert out["ZJets"]["cutflow"]["ZJets_mass"] == 6
        assert out["Data"]["cutflow"]["Data_pt"] == 84
        assert out["Data"]["cutflow"]["Data_mass"] == 66


@pytest.mark.dask_client
@pytest.mark.parametrize(
    "the_fileset", [_starting_fileset_list, _starting_fileset_dict, _starting_fileset]
)
@pytest.mark.parametrize("preprocess_legacy_root", [True, False])
def test_preprocess(the_fileset, dask_client, preprocess_legacy_root):
    with dask_client.as_current() as _:
        dataset_runnable, dataset_updated = preprocess(
            the_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=False,
            preprocess_legacy_root=preprocess_legacy_root,
        )

        if preprocess_legacy_root:
            assert dataset_runnable == _runnable_result
            assert dataset_updated == _updated_result
        else:
            assert dataset_runnable == DataGroupSpec(_runnable_result)
            assert dataset_updated == DataGroupSpec(_updated_result)


@pytest.mark.dask_client
@pytest.mark.parametrize("the_fileset", [{}, DataGroupSpec({})])
@pytest.mark.parametrize("preprocess_legacy_root", [True, False])
def test_preprocess_empty_fileset(the_fileset, dask_client, preprocess_legacy_root):
    with dask_client.as_current() as _:
        dataset_runnable, dataset_updated = preprocess(
            the_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=False,
            preprocess_legacy_root=preprocess_legacy_root,
        )
    if preprocess_legacy_root:
        # for both pydantic and classical inputs, preprocess_legacy_root returns an empty dictionary
        assert dataset_runnable == {}
        assert dataset_updated == {}
    else:
        # pydantic preprocessing upconverts to DataGroupSpec and so we'll get an empty DataGroupSpec. DatasetSpec doesn't trivially support an empty initialization, so we don't test directly
        assert dataset_runnable == DataGroupSpec({})
        assert dataset_updated == DataGroupSpec({})


@pytest.mark.dask_client
@pytest.mark.parametrize(
    "the_fileset", [_fileset_with_empty_files, DataGroupSpec(_fileset_with_empty_files)]
)
@pytest.mark.parametrize("save_form", [False, True])
@pytest.mark.parametrize("align_clusters", [False, True])
@pytest.mark.parametrize("preprocess_legacy_root", [False, True])
def test_preprocess_empty_files(
        the_fileset, save_form, align_clusters, dask_client, preprocess_legacy_root
):
    with dask_client.as_current() as _:
        dataset_runnable, dataset_updated = preprocess(
            the_fileset,
            step_size=7,
            align_clusters=align_clusters,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=save_form,
            preprocess_legacy_root=preprocess_legacy_root,
        )

    if align_clusters:
        expected_runnable = copy.deepcopy(
            _fileset_with_empty_files_preprocessed_aligned
        )
    else:
        expected_runnable = copy.deepcopy(_fileset_with_empty_files_preprocessed)

    # Handle all the differences between legacy and pydantic preprocessing, starting from json or pydantic input, and save_form True or False
    for k, v in expected_runnable.items():
        new_v = {}
        for kk, vv in v.items():
            key, val = kk, vv
            # if (preprocess_legacy_root and not isinstance(the_fileset, DataGroupSpec)) and kk == "compressed_form": # P F P F P P P P w/o save_form=True variations
            if preprocess_legacy_root and kk == "compressed_form":
                # Expect "form" key instead of "compressed_form" key, maintain dict order for the comparison as well, for json input fileset
                key = "form"
            if not save_form and kk in ["compressed_form", "form"]:
                # If save_form is False, the compressed_form will be None, so set it to None in the expected output for the comparison
                val = None
            new_v[key] = val
        if preprocess_legacy_root and isinstance(the_fileset, DataGroupSpec):
            new_v.update({"format": "root", "did": None, "metadata": {}})
        expected_runnable[k] = new_v
    expected_updated = copy.deepcopy(expected_runnable)

    if isinstance(dataset_runnable, DataGroupSpec):
        expected_runnable = DataGroupSpec(expected_runnable)
        expected_updated = DataGroupSpec(expected_updated)
    elif save_form:
        # There's a non-deterministic component to the compressed_form, so we must manually compare these in the dicts and pop them before asserting the final equality
        for k in expected_runnable.keys():
            dr = (
                dataset_runnable[k].pop("compressed_form")
                if "compressed_form" in dataset_runnable[k]
                else dataset_runnable[k].pop("form")
            )
            er = (
                expected_runnable[k].pop("compressed_form")
                if "compressed_form" in expected_runnable[k]
                else expected_runnable[k].pop("form")
            )
            assert awkward.forms.from_json(
                decompress_form(dr)
            ) == awkward.forms.from_json(decompress_form(er)), (
                f"Difference in compressed_form for dataset_runnable[{k}]",
                decompress_form(dr),
                decompress_form(er),
            )

        for k in expected_updated.keys():
            dr = (
                dataset_updated[k].pop("compressed_form")
                if "compressed_form" in dataset_updated[k]
                else dataset_updated[k].pop("form")
            )
            er = (
                expected_updated[k].pop("compressed_form")
                if "compressed_form" in expected_updated[k]
                else expected_updated[k].pop("form")
            )
            assert awkward.forms.from_json(
                decompress_form(dr)
            ) == awkward.forms.from_json(decompress_form(er)), (
                f"Difference in compressed_form for dataset_runnable[{k}]",
                decompress_form(dr),
                decompress_form(er),
            )
    assert dataset_runnable == expected_runnable
    assert dataset_updated == expected_updated

    def data_manipulation(events):
        return len(events)

    with dask_client.as_current() as _:
        to_compute = apply_to_fileset(
            data_manipulation,
            dataset_runnable,
            schemaclass=NanoAODSchema,
        )
        out = dask.compute(to_compute)[0]

    assert out == {
        "only_empty": 0,
        "nonempty_and_empty": 40,
        "empty_and_nonempty": 40,
        "only_nonempty": 40,
    }


@pytest.mark.dask_client
def test_preprocess_DataGroupSpec_mixed(dask_client):
    fileset = DataGroupSpec(_starting_fileset)
    # Create a mixed DataGroupSpec
    fileset["Data"] = fileset["Data"].model_dump()

    with dask_client.as_current() as _:
        with pytest.raises(
            AttributeError, match="'dict' object has no attribute 'format'"
        ):
            dataset_runnable, dataset_updated = preprocess(
                fileset,
                step_size=7,
                align_clusters=False,
                files_per_batch=10,
                skip_bad_files=True,
                save_form=False,
            )
            # If we update the preprocess function to handle this case again, we can check that it produces the expected output
            assert len(dataset_runnable) == 2
            assert isinstance(dataset_runnable["Data"], DatasetSpec)


@pytest.mark.dask_client
@pytest.mark.parametrize("preprocess_legacy_root", [True, False])
def test_preprocess_calculate_form(dask_client, preprocess_legacy_root):
    with dask_client.as_current() as _:
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=True,
            preprocess_legacy_root=preprocess_legacy_root,
        )

        raw_form_dy = uproot.dask(
            "tests/samples/nano_dy.root:Events",
            open_files=False,
            ak_add_doc={"__doc__": "title", "typename": "typename"},
        ).layout.form.to_json()
        raw_form_data = uproot.dask(
            "tests/samples/nano_dimuon.root:Events",
            open_files=False,
            ak_add_doc={"__doc__": "title", "typename": "typename"},
        ).layout.form.to_json()

        if preprocess_legacy_root:
            assert decompress_form(dataset_runnable["ZJets"]["form"]) == raw_form_dy
            assert decompress_form(dataset_runnable["Data"]["form"]) == raw_form_data
        else:
            assert (
                decompress_form(dataset_runnable["ZJets"].compressed_form)
                == raw_form_dy
            )
            assert (
                decompress_form(dataset_runnable["Data"].compressed_form)
                == raw_form_data
            )


@pytest.mark.dask_client
def test_preprocess_failed_file(dask_client):
    with dask_client.as_current() as _, pytest.raises(FileNotFoundError):
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=False,
            save_form=False,
        )


@pytest.mark.dask_client
@pytest.mark.parametrize("preprocess_legacy_root", [True, False])
def test_preprocess_with_file_exceptions(dask_client, preprocess_legacy_root):
    fileset = {
        "Data": {
            "files": {
                "tests/samples/delphes.root": "Delphes",
                "tests/samples/bad_delphes.root": "Delphes",
            }
        },
    }

    with (
        dask_client.as_current() as _
    ):  # should not throw uproot.exceptions.KeyInFileError
        dataset_runnable, dataset_updated = preprocess(
            fileset,
            step_size=10,
            align_clusters=False,
            files_per_batch=10,
            file_exceptions=KeyInFileError,
            skip_bad_files=True,
            save_form=False,
            preprocess_legacy_root=preprocess_legacy_root,
        )

    if preprocess_legacy_root:
        assert dataset_runnable == {
            "Data": {
                "files": {
                    "tests/samples/delphes.root": {
                        "num_entries": 25,
                        "object_path": "Delphes",
                        "steps": [
                            [
                                0,
                                13,
                            ],
                            [
                                13,
                                25,
                            ],
                        ],
                        "uuid": "ad4cd5ec-123e-11ec-92f6-93e3aac0beef",
                    },
                },
                "form": None,
                "metadata": None,
            },
        }
    else:
        assert dataset_runnable == DataGroupSpec(
            {
                "Data": {
                    "files": {
                        "tests/samples/delphes.root": {
                            "num_entries": 25,
                            "object_path": "Delphes",
                            "steps": [
                                [
                                    0,
                                    13,
                                ],
                                [
                                    13,
                                    25,
                                ],
                            ],
                            "uuid": "ad4cd5ec-123e-11ec-92f6-93e3aac0beef",
                        },
                    },
                    "compressed_form": None,
                    "metadata": None,
                },
            }
        )


@pytest.mark.parametrize(
    "the_fileset", [_updated_result, DataGroupSpec(_updated_result)]
)
def test_filter_files(the_fileset):
    filtered_files = filter_files(the_fileset)

    target = {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "num_entries": 40,
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "num_entries": 40,
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }
    if isinstance(filtered_files, DataGroupSpec):
        assert filtered_files == DataGroupSpec(target)
    else:
        assert filtered_files == target


@pytest.mark.parametrize(
    "the_fileset", [_updated_result, DataGroupSpec(_updated_result)]
)
def test_max_files(the_fileset):
    maxed_files = max_files(the_fileset, 1)

    target = {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "num_entries": 40,
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "num_entries": 40,
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }
    if isinstance(the_fileset, DataGroupSpec):
        assert maxed_files == DataGroupSpec(target)
    else:
        assert maxed_files == target


@pytest.mark.parametrize(
    "the_fileset", [_updated_result, DataGroupSpec(_updated_result)]
)
def test_slice_files(the_fileset):
    sliced_files = slice_files(the_fileset, slice(1, None, 2))

    target = {
        "ZJets": {"files": {}, "metadata": None, "form": None},
        "Data": {
            "files": {
                "tests/samples/nano_dimuon_not_there.root": {
                    "object_path": "Events",
                    "steps": None,
                    "num_entries": None,
                    "uuid": None,
                }
            },
            "metadata": None,
            "form": None,
        },
    }
    if isinstance(the_fileset, DataGroupSpec):
        target["ZJets"]["format"] = "root"
        assert sliced_files == DataGroupSpec(target)
    else:
        assert sliced_files == target


@pytest.mark.parametrize(
    "the_fileset", [_runnable_result, DataGroupSpec(_runnable_result)]
)
def test_max_chunks(the_fileset):
    max_chunked = max_chunks(the_fileset, 3)

    target = {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21]],
                    "num_entries": 40,
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21]],
                    "num_entries": 40,
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }

    if isinstance(the_fileset, DataGroupSpec):
        assert max_chunked == DataGroupSpec(target)
    else:
        assert max_chunked == target

    target2 = {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                        [15, 20],
                        [20, 25],
                        [25, 30],
                        [30, 35],
                        [35, 40],
                    ],
                }
            }
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                        [15, 20],
                        [20, 25],
                        [25, 30],
                        [30, 35],
                        [35, 40],
                    ],
                },
                "tests/samples/nano_dimuon_not_there.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                    ],
                },
            }
        },
    }
    if isinstance(the_fileset, DataGroupSpec):
        max_chunked = max_chunks(DataGroupSpec(_starting_fileset_with_steps), 10)
        assert max_chunked == DataGroupSpec(target2)
    else:
        max_chunked = max_chunks(_starting_fileset_with_steps, 10)
        assert max_chunked == target2

    target3 = {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                    ],
                }
            }
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                    ],
                },
                "tests/samples/nano_dimuon_not_there.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                    ],
                },
            }
        },
    }
    if isinstance(the_fileset, DataGroupSpec):
        max_chunked = max_chunks_per_file(
            DataGroupSpec(_starting_fileset_with_steps), 3
        )
        assert max_chunked == DataGroupSpec(target3)
    else:
        max_chunked = max_chunks_per_file(_starting_fileset_with_steps, 3)
        assert max_chunked == target3


@pytest.mark.parametrize(
    "the_fileset", [_runnable_result, DataGroupSpec(_runnable_result)]
)
def test_slice_chunks(the_fileset):
    slice_chunked = slice_chunks(the_fileset, slice(None, None, 2))

    target = {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [14, 21], [28, 35]],
                    "num_entries": 40,
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [14, 21], [28, 35]],
                    "num_entries": 40,
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }
    if isinstance(the_fileset, DataGroupSpec):
        assert slice_chunked == DataGroupSpec(target)
    else:
        assert slice_chunked == target


@pytest.mark.parametrize(
    "the_fileset",
    [_starting_fileset_with_steps, DataGroupSpec(_starting_fileset_with_steps)],
)
@pytest.mark.dask_client
def test_recover_failed_chunks(the_fileset, dask_client):
    with dask_client.as_current() as _:
        to_compute = apply_to_fileset(
            NanoEventsProcessor(),
            the_fileset,
            schemaclass=NanoAODSchema,
            uproot_options={"allow_read_errors_with_report": True},
        )
        out, reports = dask.compute(*to_compute)

    failed_fset = get_failed_steps_for_fileset(the_fileset, reports)
    target = {
        "Data": {
            "files": {
                "tests/samples/nano_dimuon_not_there.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                        [15, 20],
                        [20, 25],
                        [25, 30],
                        [30, 35],
                        [35, 40],
                    ],
                }
            }
        }
    }
    if isinstance(failed_fset, DataGroupSpec):
        assert failed_fset == DataGroupSpec(target)
    else:
        assert failed_fset == target


_splitting_fs_dict = {
    "ZJets": {
        "files": {
            "/data/zjets/a.root": "Events",
            "/data/zjets/b.root": "Events",
            "/data/zjets/c.root": "Events",
            "/data/zjets/d.root": "Events",
        }
    },
    "Data": {
        "files": {
            "/data/data/a.root": "Events",
            "/data/data/b.root": "Events",
        }
    },
}

_splitting_fs_list_in_dict = {
    "ZJets": {
        "treename": "Events",
        "files": [
            "/data/zjets/a.root",
            "/data/zjets/b.root",
            "/data/zjets/c.root",
            "/data/zjets/d.root",
        ],
    },
    "Data": {
        "treename": "Events",
        "files": [
            "/data/data/a.root",
            "/data/data/b.root",
        ],
    },
}

_splitting_fs_bare_list = {
    "ZJets": [
        "/data/zjets/a.root",
        "/data/zjets/b.root",
        "/data/zjets/c.root",
        "/data/zjets/d.root",
    ],
    "Data": [
        "/data/data/a.root",
        "/data/data/b.root",
    ],
}


def test_split_fileset_strategy_by_dataset():
    chunks = split_fileset(_splitting_fs_dict, strategy="by_dataset")
    assert len(chunks) == 2
    assert {next(iter(c)) for c in chunks} == {"ZJets", "Data"}


def test_split_fileset_no_args_returns_single_group():
    chunks = split_fileset(_splitting_fs_dict)
    assert len(chunks) == 1
    assert set(chunks[0].keys()) == {"ZJets", "Data"}


def test_split_fileset_percentage_mixed():
    chunks = split_fileset(_splitting_fs_dict, percentage=50)
    assert len(chunks) == 2
    for chunk in chunks:
        assert set(chunk.keys()) == {"ZJets", "Data"}
    total_zjets = sum(len(c["ZJets"]["files"]) for c in chunks)
    total_data = sum(len(c["Data"]["files"]) for c in chunks)
    assert total_zjets == 4
    assert total_data == 2


def test_split_fileset_strategy_and_percentage():
    chunks = split_fileset(_splitting_fs_dict, strategy="by_dataset", percentage=50)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == 1


def test_split_fileset_datasets_filter_list():
    chunks = split_fileset(
        _splitting_fs_dict, strategy="by_dataset", datasets=["ZJets"]
    )
    assert len(chunks) == 1
    assert "ZJets" in chunks[0]


def test_split_fileset_datasets_filter_callable():
    chunks = split_fileset(
        _splitting_fs_dict,
        strategy="by_dataset",
        datasets=lambda name: name.startswith("Z"),
    )
    assert len(chunks) == 1
    assert "ZJets" in chunks[0]


def test_split_fileset_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        split_fileset(_splitting_fs_dict, strategy="nope")


@pytest.mark.parametrize("bad", [0, 3, 7, 101, 1.5, "50"])
def test_split_fileset_invalid_percentage(bad):
    with pytest.raises(ValueError, match="percentage"):
        split_fileset(_splitting_fs_dict, percentage=bad)


def test_split_fileset_deterministic_under_dict_reorder():
    """Splitting must not depend on input dict insertion order."""
    fs1 = {
        "Data": {
            "files": {
                "/p/b.root": "Events",
                "/p/a.root": "Events",
                "/p/d.root": "Events",
                "/p/c.root": "Events",
            }
        }
    }
    fs2 = {
        "Data": {
            "files": {
                "/p/a.root": "Events",
                "/p/b.root": "Events",
                "/p/c.root": "Events",
                "/p/d.root": "Events",
            }
        }
    }
    c1 = split_fileset(fs1, percentage=50)
    c2 = split_fileset(fs2, percentage=50)
    for chunk_a, chunk_b in zip(c1, c2):
        assert list(chunk_a["Data"]["files"].keys()) == list(
            chunk_b["Data"]["files"].keys()
        )


def test_split_fileset_supports_list_files_inside_dict():
    chunks = split_fileset(
        _splitting_fs_list_in_dict, strategy="by_dataset", percentage=50
    )
    assert len(chunks) == 4
    for chunk in chunks:
        (data,) = chunk.values()
        assert isinstance(data["files"], list)
        assert data["treename"] == "Events"


def test_split_fileset_promotes_bare_list_with_treename():
    chunks = split_fileset(
        _splitting_fs_bare_list,
        strategy="by_dataset",
        percentage=50,
        treename="Events",
    )
    assert len(chunks) == 4
    for chunk in chunks:
        (data,) = chunk.values()
        assert isinstance(data, dict)
        assert isinstance(data["files"], list)
        assert data["treename"] == "Events"


def test_split_fileset_bare_list_requires_treename():
    with pytest.raises(ValueError, match="treename"):
        split_fileset(_splitting_fs_bare_list, strategy="by_dataset")


def test_split_fileset_list_files_in_dict_requires_treename():
    fs = {"A": {"files": ["/p/a.root", "/p/b.root"]}}
    with pytest.raises(ValueError, match="treename"):
        split_fileset(fs, percentage=50)


def test_split_fileset_list_files_in_dict_promoted_with_treename():
    fs = {"A": {"files": ["/p/a.root", "/p/b.root"]}}
    chunks = split_fileset(fs, percentage=50, treename="Events")
    for chunk in chunks:
        assert chunk["A"]["treename"] == "Events"


def test_split_fileset_preserves_extra_dataset_fields():
    fs = {
        "ZJets": {
            "treename": "Events",
            "preload": ["nMuon", "Muon_pt"],
            "metadata": {"xsec": 1.0},
            "files": {"/p/a.root": "Events", "/p/b.root": "Events"},
        }
    }
    chunks = split_fileset(fs, percentage=50)
    for chunk in chunks:
        assert chunk["ZJets"]["treename"] == "Events"
        assert chunk["ZJets"]["preload"] == ["nMuon", "Muon_pt"]
        assert chunk["ZJets"]["metadata"] == {"xsec": 1.0}


def test_hash_fileset_stable_across_dict_order():
    fs1 = {
        "B": {"files": {"/p/y.root": "Events", "/p/x.root": "Events"}},
        "A": {"files": {"/p/b.root": "Events", "/p/a.root": "Events"}},
    }
    fs2 = {
        "A": {"files": {"/p/a.root": "Events", "/p/b.root": "Events"}},
        "B": {"files": {"/p/x.root": "Events", "/p/y.root": "Events"}},
    }
    assert hash_fileset(fs1) == hash_fileset(fs2)


def test_hash_fileset_changes_with_treename():
    fs1 = {"A": {"files": {"/p/a.root": "Events"}}}
    fs2 = {"A": {"files": {"/p/a.root": "Other"}}}
    assert hash_fileset(fs1) != hash_fileset(fs2)


def test_hash_fileset_changes_with_dataset_level_treename():
    fs1 = {"A": {"treename": "Events", "files": ["/p/a.root"]}}
    fs2 = {"A": {"treename": "Other", "files": ["/p/a.root"]}}
    assert hash_fileset(fs1) != hash_fileset(fs2)


def test_hash_fileset_changes_with_preload():
    fs1 = {"A": {"preload": ["a"], "files": {"/p/a.root": "Events"}}}
    fs2 = {"A": {"preload": ["b"], "files": {"/p/a.root": "Events"}}}
    assert hash_fileset(fs1) != hash_fileset(fs2)


def test_hash_fileset_preload_order_insensitive():
    fs1 = {"A": {"preload": ["a", "b"], "files": {"/p/a.root": "Events"}}}
    fs2 = {"A": {"preload": ["b", "a"], "files": {"/p/a.root": "Events"}}}
    assert hash_fileset(fs1) == hash_fileset(fs2)


def test_hash_fileset_chunks_from_split_are_unique():
    chunks = split_fileset(_splitting_fs_dict, strategy="by_dataset", percentage=50)
    hashes = {hash_fileset(c) for c in chunks}
    assert len(hashes) == len(chunks)


def test_hash_fileset_rejects_bare_list():
    with pytest.raises(TypeError, match="split_fileset"):
        hash_fileset(_splitting_fs_bare_list)


def test_hash_fileset_rejects_list_files_without_treename():
    fs = {"A": {"files": ["/p/a.root"]}}
    with pytest.raises(ValueError, match="treename"):
        hash_fileset(fs)


def test_hash_fileset_distinguishes_treenames_for_promoted_chunks():
    chunks_a = split_fileset(
        _splitting_fs_bare_list, strategy="by_dataset", treename="Events"
    )
    chunks_b = split_fileset(
        _splitting_fs_bare_list, strategy="by_dataset", treename="OtherTree"
    )
    for a, b in zip(chunks_a, chunks_b):
        assert hash_fileset(a) != hash_fileset(b)


def test_hash_fileset_ignores_undocumented_fields():
    """Undocumented dataset-level keys must not affect the hash, and must not
    blow up on non-JSON-serializable values."""
    base = {"A": {"files": {"/p/a.root": "Events"}}}
    extra_unserializable = {
        "A": {
            "files": {"/p/a.root": "Events"},
            "compressed_form": object(),  # not JSON-serializable
            "internal_flag": True,
        }
    }
    assert hash_fileset(base) == hash_fileset(extra_unserializable)


def test_hash_fileset_accepts_frozenset_preload():
    fs1 = {"A": {"preload": frozenset({"a", "b"}), "files": {"/p/a.root": "Events"}}}
    fs2 = {"A": {"preload": ["a", "b"], "files": {"/p/a.root": "Events"}}}
    assert hash_fileset(fs1) == hash_fileset(fs2)
