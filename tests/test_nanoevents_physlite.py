import json
import os

import awkward as ak
import dask
import pytest

from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema


def _events(filter=None):
    path = os.path.abspath("tests/samples/PHYSLITE_example.root")
    factory = NanoEventsFactory.from_root(
        {path: "CollectionTree"},
        schemaclass=PHYSLITESchema,
        delayed=True,
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
                            "parameters": {"__doc__": "AnalysisElectronsAuxDyn.pt"},
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
                            "parameters": {"__doc__": "AnalysisElectronsAuxDyn.eta"},
                            "form_key": "node5",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "float32",
                            "inner_shape": [],
                            "parameters": {"__doc__": "AnalysisElectronsAuxDyn.phi"},
                            "form_key": "node6",
                        },
                        {
                            "class": "NumpyArray",
                            "primitive": "float32",
                            "inner_shape": [],
                            "parameters": {"__doc__": "AnalysisElectronsAuxDyn.m"},
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
