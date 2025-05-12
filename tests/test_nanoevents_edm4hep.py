import os

import pytest

from coffea.nanoevents import EDM4HEPSchema, NanoEventsFactory

# Basic Tests


def _events(**kwargs):
    # Original sample generated from key4hep workflow
    path = os.path.abspath("tests/samples/edm4hep.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=EDM4HEPSchema, **kwargs
    )
    return factory.events()


@pytest.fixture(scope="module")
def eager_events():
    return _events(
        mode="eager", uproot_options={"filter_name": lambda x: "PARAMETERS" not in x}
    )


@pytest.fixture(scope="module")
def delayed_events():
    return _events(
        mode="dask", uproot_options={"filter_name": lambda x: "PARAMETERS" not in x}
    )


branches = {
    "CaloHitContributionCollection": [
        "E",
        "PDG",
        "particle_idx_MCParticleCollection_collectionID",
        "particle_idx_MCParticleCollection_index",
        "particle_idx_MCParticleCollection_index_Global",
        "stepPosition",
        "time",
    ],
    "CaloHitMCParticleLinkCollection": [
        "Link_from_CalorimeterHitCollection",
        "Link_to_MCParticleCollection",
        "weight",
    ],
    "CaloHitSimCaloHitLinkCollection": [
        "Link_from_CalorimeterHitCollection",
        "Link_to_SimCalorimeterHitCollection",
        "weight",
    ],
    "CalorimeterHitCollection": [
        "E",
        "cellID",
        "energyError",
        "position",
        "time",
        "type",
    ],
    "ClusterCollection": [
        "E",
        "clusters_idx_ClusterCollection_collectionID",
        "clusters_idx_ClusterCollection_index",
        "clusters_idx_ClusterCollection_index_Global",
        "directionError",
        "energyError",
        "hits_idx_CalorimeterHitCollection_collectionID",
        "hits_idx_CalorimeterHitCollection_index",
        "hits_idx_CalorimeterHitCollection_index_Global",
        "iTheta",
        "phi",
        "position",
        "positionError",
        "shapeParameters",
        "subdetectorEnergies",
        "type",
    ],
    "ClusterMCParticleLinkCollection": [
        "Link_from_ClusterCollection",
        "Link_to_MCParticleCollection",
        "weight",
    ],
    "EventHeader": ["eventNumber", "runNumber", "timeStamp", "weight", "weights"],
    "GPDoubleKeys": [],
    "GPDoubleValues": [],
    "GPFloatKeys": [],
    "GPFloatValues": [],
    "GPIntKeys": [],
    "GPIntValues": [],
    "GPStringKeys": [],
    "GPStringValues": [],
    "GeneratorEventParametersCollection": [
        "alphaQCD",
        "alphaQED",
        "crossSectionErrors",
        "crossSections",
        "eventScale",
        "signalProcessId",
        "signalVertex_idx_MCParticleCollection_collectionID",
        "signalVertex_idx_MCParticleCollection_index",
        "signalVertex_idx_MCParticleCollection_index_Global",
        "sqrts",
    ],
    "GeneratorPdfInfoCollection": [
        "lhapdfId[2]",
        "partonId[2]",
        "scale",
        "x[2]",
        "xf[2]",
    ],
    "MCParticleCollection": [
        "PDG",
        "charge",
        "colorFlow",
        "daughters_idx_MCParticleCollection_collectionID",
        "daughters_idx_MCParticleCollection_index",
        "daughters_idx_MCParticleCollection_index_Global",
        "endpoint",
        "generatorStatus",
        "mass",
        "momentumAtEndpoint",
        "parents_idx_MCParticleCollection_collectionID",
        "parents_idx_MCParticleCollection_index",
        "parents_idx_MCParticleCollection_index_Global",
        "px",
        "py",
        "pz",
        "simulatorStatus",
        "spin",
        "time",
        "vertex",
    ],
    "ParticleIDCollection": [
        "PDG",
        "algorithmType",
        "likelihood",
        "parameters",
        "particle_idx_ReconstructedParticleCollection_collectionID",
        "particle_idx_ReconstructedParticleCollection_index",
        "particle_idx_ReconstructedParticleCollection_index_Global",
        "type",
    ],
    "RawCalorimeterHitCollection": ["amplitude", "cellID", "timeStamp"],
    "RawTimeSeriesCollection": [
        "adcCounts",
        "cellID",
        "charge",
        "interval",
        "quality",
        "time",
    ],
    "RecDqdxCollection": [
        "dQdx",
        "track_idx_TrackCollection_collectionID",
        "track_idx_TrackCollection_index",
        "track_idx_TrackCollection_index_Global",
    ],
    "RecoMCParticleLinkCollection": [
        "Link_from_ReconstructedParticleCollection",
        "Link_to_MCParticleCollection",
        "weight",
    ],
    "ReconstructedParticleCollection": [
        "E",
        "PDG",
        "charge",
        "clusters_idx_ClusterCollection_collectionID",
        "clusters_idx_ClusterCollection_index",
        "clusters_idx_ClusterCollection_index_Global",
        "covMatrix",
        "decayVertex_idx_VertexCollection_collectionID",
        "decayVertex_idx_VertexCollection_index",
        "decayVertex_idx_VertexCollection_index_Global",
        "goodnessOfPID",
        "mass",
        "particles_idx_ReconstructedParticleCollection_collectionID",
        "particles_idx_ReconstructedParticleCollection_index",
        "particles_idx_ReconstructedParticleCollection_index_Global",
        "px",
        "py",
        "pz",
        "referencePoint",
        "tracks_idx_TrackCollection_collectionID",
        "tracks_idx_TrackCollection_index",
        "tracks_idx_TrackCollection_index_Global",
    ],
    "SimCalorimeterHitCollection": [
        "E",
        "cellID",
        "contributions_idx_CaloHitContributionCollection_collectionID",
        "contributions_idx_CaloHitContributionCollection_index",
        "contributions_idx_CaloHitContributionCollection_index_Global",
        "position",
    ],
    "SimTrackerHitCollection": [
        "cellID",
        "eDep",
        "particle_idx_MCParticleCollection_collectionID",
        "particle_idx_MCParticleCollection_index",
        "particle_idx_MCParticleCollection_index_Global",
        "pathLength",
        "position",
        "px",
        "py",
        "pz",
        "quality",
        "time",
    ],
    "TimeSeriesCollection": ["amplitude", "cellID", "interval", "time"],
    "TrackCollection": [
        "Nholes",
        "chi2",
        "ndf",
        "subdetectorHitNumbers",
        "subdetectorHoleNumbers",
        "trackStates",
        "tracks_idx_TrackCollection_collectionID",
        "tracks_idx_TrackCollection_index",
        "tracks_idx_TrackCollection_index_Global",
        "type",
    ],
    "TrackMCParticleLinkCollection": [
        "Link_from_TrackCollection",
        "Link_to_MCParticleCollection",
        "weight",
    ],
    "TrackerHit3DCollection": [
        "cellID",
        "covMatrix",
        "eDep",
        "eDepError",
        "position",
        "quality",
        "time",
        "type",
    ],
    "TrackerHitPlaneCollection": [
        "cellID",
        "covMatrix",
        "du",
        "dv",
        "eDep",
        "eDepError",
        "position",
        "quality",
        "time",
        "type",
        "u",
        "v",
    ],
    "TrackerHitSimTrackerHitLinkCollection": [
        "Link_from_TrackerHit3DCollection",
        "Link_from_TrackerHitPlaneCollection",
        "Link_to_SimTrackerHitCollection",
        "weight",
    ],
    "VertexCollection": [
        "algorithmType",
        "chi2",
        "covMatrix",
        "ndf",
        "parameters",
        "particles_idx_ReconstructedParticleCollection_collectionID",
        "particles_idx_ReconstructedParticleCollection_index",
        "particles_idx_ReconstructedParticleCollection_index_Global",
        "position",
        "type",
    ],
    "VertexRecoParticleLinkCollection": [
        "Link_from_VertexCollection",
        "Link_to_ReconstructedParticleCollection",
        "weight",
    ],
}


# Test 1: Are the required branches and sub-branches present?
@pytest.mark.parametrize(
    "field",
    branches.keys(),
)
def test_field_is_present(eager_events, delayed_events, field):
    eager_fields = eager_events.fields
    delayed_fields = delayed_events.fields
    subfields = branches[field]
    assert field in eager_fields
    assert eager_events[field].fields == subfields
    assert field in delayed_fields
    assert delayed_events[field].fields == subfields


# Test 2: Do all the relations work?
@pytest.mark.parametrize(
    "field",
    branches.keys(),
)
def test_Relations(eager_events, delayed_events, field):
    skip_list = ["GeneratorEventParametersCollection", "GeneratorPdfInfoCollection"]
    if (
        not field.endswith("LinkCollection")
        and not field.startswith("GP")
        and field not in skip_list
    ):
        eager_relation_branches = eager_events[field].List_Relations

        relations = {
            name.split("_idx_")[0]: name.split("_idx_")[1].split("_")[0]
            for name in eager_relation_branches
        }

        for generic_name, target_name in relations.items():
            e_fin = eager_events[field].Map_Relation(generic_name, target_name)

            if e_fin.layout.branch_depth[1] == 2:
                mixin = e_fin.layout.content.content.parameter("__record__")
                assert target_name.startswith(mixin)
            elif e_fin.layout.branch_depth[1] == 3:
                mixin = e_fin.layout.content.content.content.parameter("__record__")
                assert target_name.startswith(mixin)

            d_fin = delayed_events[field].Map_Relation(generic_name, target_name)

            if d_fin.layout.branch_depth[1] == 2:
                mixin = d_fin.layout.content.content.parameter("__record__")
                assert target_name.startswith(mixin)
            elif d_fin.layout.branch_depth[1] == 3:
                mixin = d_fin.layout.content.content.content.parameter("__record__")
                assert target_name.startswith(mixin)
