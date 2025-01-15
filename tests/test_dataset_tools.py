import dask
import pytest
import uproot
from distributed import Client
from uproot.exceptions import KeyInFileError

from coffea.dataset_tools import (
    apply_to_fileset,
    filter_files,
    get_failed_steps_for_fileset,
    max_chunks,
    max_files,
    preprocess,
    slice_chunks,
    slice_files,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema
from coffea.processor.test_items import NanoEventsProcessor, NanoTestProcessor
from coffea.util import decompress_form

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


def _my_analysis_output_2(events):
    return events.Electron.pt, events.Muon.pt


def _my_analysis_output_3(events):
    return events.Electron.pt, events.Muon.pt, events.Tau.pt


@pytest.mark.parametrize("allow_read_errors_with_report", [True, False])
def test_tuple_data_manipulation_output(allow_read_errors_with_report):
    import dask_awkward

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


@pytest.mark.parametrize(
    "proc_and_schema",
    [(NanoTestProcessor, BaseSchema), (NanoEventsProcessor, NanoAODSchema)],
)
def test_apply_to_fileset(proc_and_schema):
    proc, schemaclass = proc_and_schema

    with Client() as _:
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


def test_apply_to_fileset_hinted_form():
    with Client() as _:
        dataset_runnable, dataset_updated = preprocess(
            _starting_fileset,
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


@pytest.mark.parametrize(
    "the_fileset", [_starting_fileset_list, _starting_fileset_dict, _starting_fileset]
)
def test_preprocess(the_fileset):
    with Client() as _:
        dataset_runnable, dataset_updated = preprocess(
            the_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
        )

        assert dataset_runnable == _runnable_result
        assert dataset_updated == _updated_result


def test_preprocess_calculate_form():
    with Client() as _:
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=True,
        )

        raw_form_dy = uproot.dask(
            "tests/samples/nano_dy.root:Events", open_files=False, ak_add_doc=True
        ).layout.form.to_json()
        raw_form_data = uproot.dask(
            "tests/samples/nano_dimuon.root:Events", open_files=False, ak_add_doc=True
        ).layout.form.to_json()

        assert decompress_form(dataset_runnable["ZJets"]["form"]) == raw_form_dy
        assert decompress_form(dataset_runnable["Data"]["form"]) == raw_form_data


def test_preprocess_failed_file():
    with Client() as _, pytest.raises(FileNotFoundError):
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=False,
        )


def test_preprocess_with_file_exceptions():
    fileset = {
        "Data": {
            "files": {
                "tests/samples/delphes.root": "Delphes",
                "tests/samples/bad_delphes.root": "Delphes",
            }
        },
    }

    with Client() as _:  # should not throw uproot.exceptions.KeyInFileError
        dataset_runnable, dataset_updated = preprocess(
            fileset,
            step_size=10,
            align_clusters=False,
            files_per_batch=10,
            file_exceptions=KeyInFileError,
            skip_bad_files=True,
        )

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


def test_filter_files():
    filtered_files = filter_files(_updated_result)

    assert filtered_files == {
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


def test_max_files():
    maxed_files = max_files(_updated_result, 1)

    assert maxed_files == {
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


def test_slice_files():
    sliced_files = slice_files(_updated_result, slice(1, None, 2))

    assert sliced_files == {
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


def test_max_chunks():
    max_chunked = max_chunks(_runnable_result, 3)

    assert max_chunked == {
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


def test_slice_chunks():
    slice_chunked = slice_chunks(_runnable_result, slice(None, None, 2))

    assert slice_chunked == {
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


def test_recover_failed_chunks():
    with Client() as _:
        to_compute = apply_to_fileset(
            NanoEventsProcessor(),
            _starting_fileset_with_steps,
            schemaclass=NanoAODSchema,
            uproot_options={"allow_read_errors_with_report": True},
        )
        out, reports = dask.compute(*to_compute)

    failed_fset = get_failed_steps_for_fileset(_starting_fileset_with_steps, reports)
    assert failed_fset == {
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
