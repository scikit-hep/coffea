import os

import awkward
import pytest

from coffea.nanoevents import EDM4HEPSchema, NanoEventsFactory


def _events(**kwargs):
    # Original sample generated from key4hep workflow
    path = os.path.abspath("tests/samples/edm4hep.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=EDM4HEPSchema, **kwargs
    )
    return factory.events()


@pytest.fixture(scope="module")
def eager_events():
    return _events(delayed=False)


@pytest.fixture(scope="module")
def delayed_events():
    return _events(delayed=True)


@pytest.mark.parametrize(
    "field",
    [
        "CaloHitContributionCollection",
        "CaloHitMCParticleLinkCollection",
        "CaloHitSimCaloHitLinkCollection",
        "CalorimeterHitCollection",
        "ClusterCollection",
        "ClusterMCParticleLinkCollection",
        "EventHeader",
        "GPDoubleKeys",
        "GPDoubleValues",
        "GPFloatKeys",
        "GPFloatValues",
        "GPIntKeys",
        "GPIntValues",
        "GPStringKeys",
        "GPStringValues",
        "GeneratorEventParametersCollection",
        "GeneratorPdfInfoCollection",
        "MCParticleCollection",
        "ParticleIDCollection",
        "RawCalorimeterHitCollection",
        "RawTimeSeriesCollection",
        "RecDqdxCollection",
        "RecoMCParticleLinkCollection",
        "ReconstructedParticleCollection",
        "SimCalorimeterHitCollection",
        "SimTrackerHitCollection",
        "TimeSeriesCollection",
        "TrackCollection",
        "TrackMCParticleLinkCollection",
        "TrackerHit3DCollection",
        "TrackerHitPlaneCollection",
        "TrackerHitSimTrackerHitLinkCollection",
        "VertexCollection",
        "VertexRecoParticleLinkCollection",
    ],
)
def test_field_is_present(eager_events, delayed_events, field):
    eager_fields = eager_events.fields
    delayed_fields = delayed_events.fields
    assert field in eager_fields
    assert field in delayed_fields


def test_MC_daughters(delayed_events):
    d = delayed_events.MCParticleCollection.Map_Relation(
        "daughters", "MCParticleCollection"
    ).compute()
    assert isinstance(d, awkward.highlevel.Array)
    assert d.layout.branch_depth[1] == 3


def test_MC_parents(delayed_events):
    p = delayed_events.MCParticleCollection.Map_Relation(
        "parents", "MCParticleCollection"
    ).compute()
    assert isinstance(p, awkward.highlevel.Array)
    assert p.layout.branch_depth[1] == 3
