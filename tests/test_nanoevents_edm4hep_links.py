import awkward as ak
import numpy as np
import pytest
import uproot

from coffea.nanoevents import EDM4HEPSchema, NanoEventsFactory
from coffea.nanoevents.schemas import edm4hep

# ILD reconstruction, EDM4hep 1.0.0 written by podio 1.7: links are generic
# podio::LinkCollection<From,To> and several collection names contain underscores
SAMPLE = "tests/samples/edm4hep_ILD_mumuH_v01-00_podio_v01-07_10ev.root"
SCHEMAS = [EDM4HEPSchema, EDM4HEPSchema.version("01.00")]
MODES = ["eager", "virtual", "dask"]


def _events(mode, schemaclass=EDM4HEPSchema):
    if mode == "dask":
        pytest.importorskip("dask_awkward")
    return NanoEventsFactory.from_root(
        {SAMPLE: "events"}, schemaclass=schemaclass, mode=mode
    ).events()


def _get(array):
    return array.compute() if hasattr(array, "compute") else array


def _assert_global_index(index, index_global, target):
    """index_Global is the local index shifted by the per-event offsets of the target."""
    index, index_global, target = (_get(x) for x in (index, index_global, target))
    offsets = np.concatenate([[0], np.cumsum(ak.num(target))[:-1]])
    valid = index >= 0
    assert ak.sum(valid) > 0
    assert ak.all((index_global == index + offsets)[valid])


@pytest.mark.parametrize("schemaclass", SCHEMAS, ids=lambda s: s.edm4hep_version)
@pytest.mark.parametrize("mode", MODES)
def test_generic_links(mode, schemaclass):
    events = _events(mode, schemaclass)
    link = events.RecoMCTruthLink
    assert {"weight", "Link_from_PandoraPFOs", "Link_to_MCParticles"} <= set(
        link.fields
    )
    to = link.Link_to_MCParticles
    _assert_global_index(to.index, to.index_Global, events.MCParticles)

    # link whose target collection is underscore-named
    src = events.SiTracksMCTruthLink.Link_from_SiTracks_Refitted
    _assert_global_index(src.index, src.index_Global, events.SiTracks_Refitted)

    # underscore-named link collection (its "to" side is unset in this sample)
    vertex_link = events.BuildUpVertices_associatedParticles
    assert {"Link_from_BuildUpVertices", "Link_to_PandoraPFOs"} <= set(
        vertex_link.fields
    )
    src = vertex_link.Link_from_BuildUpVertices
    _assert_global_index(src.index, src.index_Global, events.BuildUpVertices)


@pytest.mark.parametrize("mode", MODES)
def test_underscore_named_collections(mode):
    events = _events(mode)

    # vector member of an underscore-named collection keeps every leaf
    track_states = _get(events.SiTracks_Refitted.trackStates)
    assert track_states.fields == _get(events.SiTracks.trackStates).fields
    raw = uproot.open(SAMPLE)["events"][
        "_SiTracks_Refitted_trackStates/_SiTracks_Refitted_trackStates.D0"
    ].array()
    assert ak.all(ak.flatten(track_states.D0, axis=None) == ak.flatten(raw, axis=None))

    # one-to-one relation whose branch name is prefixed by two other collection names
    dqdx = events.SiTracks_Refitted_dQdx
    _assert_global_index(
        dqdx.track_idx_SiTracks_Refitted_index,
        dqdx.track_idx_SiTracks_Refitted_index_Global,
        events.SiTracks_Refitted,
    )


def test_relation_branches_exact_match():
    forms = {
        "_X_Y_m/_X_Y_m.index": "xy_m",
        "_X_Y_members/_X_Y_members.index": "xy_members",
        "_X_Y_m": "xy_m_flat",
        "_X_m/_X_m.index": "x_m",
    }
    assert edm4hep._relation_branches(forms, "X_Y", "m") == {"m.index": "xy_m"}
    assert edm4hep._relation_branches(forms, "X", "m") == {"m.index": "x_m"}
    assert set(forms) == {"_X_Y_members/_X_Y_members.index", "_X_Y_m"}


def test_unresolved_links_error_and_override(monkeypatch):
    from coffea.nanoevents import factory

    monkeypatch.setattr(factory, "podio_collection_types", lambda tree: None)
    with pytest.raises(
        RuntimeError, match="BuildUpVertices_associatedParticles.*extra_mixins"
    ):
        _events("virtual")

    tree = uproot.open(SAMPLE)["events"]
    link_types = {
        (link["From"], link["To"]): name.split("::")[-1]
        for name, link in edm4hep.load_edm4hep(EDM4HEPSchema.edm4hep_version)[0][
            "links"
        ].items()
    }
    overrides = {}
    for name, datatype in edm4hep.podio_collection_types(tree).items():
        endpoints = edm4hep._link_collection.match(datatype)
        if endpoints and name in tree:
            overrides[name] = link_types[endpoints.groups()]

    class TypedLinks(EDM4HEPSchema):
        extra_mixins = {**EDM4HEPSchema.extra_mixins, **overrides}

    events = _events("virtual", TypedLinks)
    assert "Link_to_MCParticles" in events.RecoMCTruthLink.fields
