import json
import os
import sys

import awkward as ak
import dask
import pytest

from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema


def _events(filter=None):
    path = os.path.abspath("tests/samples/PHYSLITE_example.root")
    factory = NanoEventsFactory.from_root(
        {path: "CollectionTree"},
        schemaclass=PHYSLITESchema,
        mode="dask",
        uproot_options=dict(filter_name=filter),
    )
    return factory.events()


@pytest.fixture(scope="module")
def events():
    return _events()


def test_load_single_field_of_linked(events):
    with dask.config.set({"awkward.raise-failed-meta": True}):
        events.Electrons.caloClusters.calE.compute()


@pytest.mark.skip(
    reason="temporarily disabled because of uproot issue #1267 https://github.com/scikit-hep/uproot5/issues/1267"
)
@pytest.mark.parametrize("do_slice", [False, True])
def test_electron_track_links(events, do_slice):
    if do_slice:
        events = events[::2]
    trackParticles = events.Electrons.trackParticles.compute()
    for i, event in enumerate(events[["Electrons", "GSFTrackParticles"]].compute()):
        for j, electron in enumerate(event.Electrons):
            for link_index, link in enumerate(electron.trackParticleLinks):
                track_index = link.m_persIndex
                assert (
                    event.GSFTrackParticles[track_index].z0
                    == trackParticles[i][j][link_index].z0
                )


def mock_empty(form, behavior={}):
    return ak.Array(
        form.length_zero_array(),
        behavior=behavior,
    )


def test_electron_forms():
    def filter_name(name):
        return name in [
            "AnalysisElectronsAuxDyn.pt",
            "AnalysisElectronsAuxDyn.eta",
            "AnalysisElectronsAuxDyn.phi",
            "AnalysisElectronsAuxDyn.m",
        ]

    events = _events(filter_name)

    mocked, _, _ = ak.to_buffers(mock_empty(events.form))

    expected_json = {
        "class": "RecordArray",
        "fields": ["Electrons"],
        "contents": [
            {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "RecordArray",
                    "fields": ["pt", "_eventindex", "eta", "phi", "m"],
                    "contents": [
                        {
                            "class": "NumpyArray",
                            "primitive": "float32",
                            "inner_shape": [],
                            "parameters": {
                                "__doc__": "AnalysisElectronsAuxDyn.pt",
                                "typename": "std::vector<float>",
                            },
                            "form_key": "node3",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "int64",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": "node4",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "float32",
                            "inner_shape": [],
                            "parameters": {
                                "__doc__": "AnalysisElectronsAuxDyn.eta",
                                "typename": "std::vector<float>",
                            },
                            "form_key": "node5",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "float32",
                            "inner_shape": [],
                            "parameters": {
                                "__doc__": "AnalysisElectronsAuxDyn.phi",
                                "typename": "std::vector<float>",
                            },
                            "form_key": "node6",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "float32",
                            "inner_shape": [],
                            "parameters": {
                                "__doc__": "AnalysisElectronsAuxDyn.m",
                                "typename": "std::vector<float>",
                            },
                            "form_key": "node7",
                        },
                    ],
                    "parameters": {
                        "__record__": "Electron",
                        "collection_name": "Electrons",
                    },
                    "form_key": "node2",
                },
                "parameters": {},
                "form_key": "node1",
            }
        ],
        "parameters": {
            "__doc__": "CollectionTree",
            "__record__": "NanoEvents",
            "metadata": {},
        },
        "form_key": "node0",
    }
    assert json.dumps(expected_json) == mocked.to_json()


def test_entry_start_and_entry_stop():
    is_windows = sys.platform.startswith("win")

    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_start=31,
        iteritems_options=dict(
            filter_name=lambda name: name
            in [
                "AnalysisElectronsAuxDyn.pt",
                "AnalysisElectronsAuxDyn.trackParticleLinks",
            ]
        ),
    ).events()

    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_stop=31,
        iteritems_options=dict(
            filter_name=lambda name: name
            in [
                "AnalysisElectronsAuxDyn.pt",
                "AnalysisElectronsAuxDyn.trackParticleLinks",
            ]
        ),
    ).events()

    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_start=31,
        entry_stop=62,
        iteritems_options=dict(
            filter_name=lambda name: name
            in [
                "AnalysisElectronsAuxDyn.pt",
                "AnalysisElectronsAuxDyn.trackParticleLinks",
            ]
        ),
    ).events()

    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_start=31,
    ).events()

    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_stop=31,
    ).events()

    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_start=31,
        entry_stop=62,
    ).events()

    access_log = []
    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="virtual",
        schemaclass=PHYSLITESchema,
        entry_start=31,
        access_log=access_log,
    ).events()
    if not is_windows:
        assert access_log == []

    access_log = []
    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="virtual",
        schemaclass=PHYSLITESchema,
        entry_stop=31,
        access_log=access_log,
    ).events()
    if not is_windows:
        assert access_log == []

    access_log = []
    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="virtual",
        schemaclass=PHYSLITESchema,
        entry_start=31,
        entry_stop=62,
        access_log=access_log,
    ).events()
    if not is_windows:
        assert access_log == []

    events = NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="eager",
        schemaclass=PHYSLITESchema,
        entry_start=31,
        entry_stop=40,
    ).events()

    assert events.Electrons.trackParticleLinks.tolist() == [
        [
            [
                {"m_persIndex": 0, "m_persKey": 776133387},
                {"m_persIndex": 0, "m_persKey": 0},
                {"m_persIndex": 0, "m_persKey": 0},
                {"m_persIndex": 0, "m_persKey": 0},
            ],
            [{"m_persIndex": 2, "m_persKey": 776133387}],
            [{"m_persIndex": 1, "m_persKey": 776133387}],
        ],
        [
            [{"m_persIndex": 0, "m_persKey": 776133387}],
            [{"m_persIndex": 1, "m_persKey": 776133387}],
        ],
        [
            [
                {"m_persIndex": 0, "m_persKey": 776133387},
                {"m_persIndex": 0, "m_persKey": 0},
                {"m_persIndex": 0, "m_persKey": 0},
            ],
            [{"m_persIndex": 1, "m_persKey": 776133387}],
        ],
        [
            [{"m_persIndex": 0, "m_persKey": 776133387}],
            [{"m_persIndex": 1, "m_persKey": 776133387}],
        ],
        [
            [{"m_persIndex": 1, "m_persKey": 776133387}],
            [{"m_persIndex": 0, "m_persKey": 776133387}],
        ],
        [],
        [[{"m_persIndex": 0, "m_persKey": 776133387}]],
        [[{"m_persIndex": 1, "m_persKey": 776133387}]],
        [
            [
                {"m_persIndex": 0, "m_persKey": 776133387},
                {"m_persIndex": 1, "m_persKey": 776133387},
                {"m_persIndex": 0, "m_persKey": 0},
            ],
            [{"m_persIndex": 2, "m_persKey": 776133387}],
        ],
    ]
    assert events.Electrons.pt.tolist() == [
        [58211.04296875, 36531.87890625, 6576.1328125],
        [136858.140625, 85670.0390625],
        [52149.78125, 31514.447265625],
        [72111.9375, 27898.1015625],
        [71988.453125, 22240.546875],
        [],
        [16296.65625],
        [49905.984375],
        [98232.9921875, 29483.591796875],
    ]
