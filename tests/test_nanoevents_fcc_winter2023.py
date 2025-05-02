import os

import awkward
import pytest

from coffea.nanoevents import FCC, NanoEventsFactory


# Basic Tests
def _events(**kwargs):
    # Path to original sample: /eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_mumuH_Hbb_ecm240/events_159112833.root
    path = os.path.abspath("tests/samples/test_FCC_Winter2023.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=FCC.get_schema(version="pre-edm4hep1"), **kwargs
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
    "CalorimeterHits": ["E", "cellID", "energyError", "position", "time", "type"],
    "EFlowNeutralHadron": [
        "E",
        "clusters",
        "directionError",
        "energyError",
        "hits",
        "iTheta",
        "particleIDs",
        "phi",
        "position",
        "positionError_6_",
        "shapeParameters",
        "subdetectorEnergies",
        "type",
    ],
    "EFlowNeutralHadron_0": [],
    "EFlowNeutralHadron_1": [],
    "EFlowNeutralHadronidx0": ["collectionID", "index"],
    "EFlowNeutralHadronidx1": ["collectionID", "index"],
    "EFlowNeutralHadronidx2": ["collectionID", "index"],
    "EFlowPhoton": [
        "E",
        "clusters",
        "directionError",
        "energyError",
        "hits",
        "iTheta",
        "particleIDs",
        "phi",
        "position",
        "positionError_6_",
        "shapeParameters",
        "subdetectorEnergies",
        "type",
    ],
    "EFlowPhoton_0": [],
    "EFlowPhoton_1": [],
    "EFlowPhotonidx0": ["collectionID", "index"],
    "EFlowPhotonidx1": ["collectionID", "index"],
    "EFlowPhotonidx2": ["collectionID", "index"],
    "EFlowTrack": [
        "chi2",
        "dEdx",
        "dEdxError",
        "dxQuantities",
        "ndf",
        "radiusOfInnermostHit",
        "subDetectorHitNumbers",
        "trackStates",
        "trackerHits",
        "tracks",
        "type",
    ],
    "EFlowTrack_0": [],
    "EFlowTrack_1": [
        "D0",
        "Z0",
        "covMatrix_21_",
        "location",
        "omega",
        "phi",
        "referencePoint",
        "tanLambda",
        "time",
    ],
    "EFlowTrack_2": ["error", "type", "value"],
    "EFlowTrack_L": [],
    "EFlowTrackidx0": ["collectionID", "index"],
    "EFlowTrackidx1": ["collectionID", "index"],
    "Electronidx0": ["collectionID", "index"],
    "Jet": [
        "E",
        "charge",
        "clusters",
        "covMatrix_10_",
        "goodnessOfPID",
        "mass",
        "particleIDs",
        "particles",
        "px",
        "py",
        "pz",
        "referencePoint",
        "tracks",
        "type",
    ],
    "Jetidx0": ["collectionID", "index"],
    "Jetidx1": ["collectionID", "index"],
    "Jetidx2": ["collectionID", "index"],
    "Jetidx3": ["collectionID", "index"],
    "Jetidx4": ["collectionID", "index"],
    "Jetidx5": ["collectionID", "index"],
    "MCRecoAssociations": ["mc", "reco", "weight"],
    "MissingET": [
        "E",
        "charge",
        "clusters",
        "covMatrix_10_",
        "goodnessOfPID",
        "mass",
        "particleIDs",
        "particles",
        "px",
        "py",
        "pz",
        "referencePoint",
        "tracks",
        "type",
    ],
    "MissingETidx0": ["collectionID", "index"],
    "MissingETidx1": ["collectionID", "index"],
    "MissingETidx2": ["collectionID", "index"],
    "MissingETidx3": ["collectionID", "index"],
    "MissingETidx4": ["collectionID", "index"],
    "MissingETidx5": ["collectionID", "index"],
    "Muonidx0": ["collectionID", "index"],
    "Particle": [
        "MCRecoAssociationsidx1_indexGlobal",
        "PDG",
        "charge",
        "colorFlow",
        "daughters",
        "endpoint",
        "generatorStatus",
        "mass",
        "momentumAtEndpoint",
        "parents",
        "px",
        "py",
        "pz",
        "simulatorStatus",
        "spin",
        "time",
        "vertex",
    ],
    "ParticleIDs": ["PDG", "algorithmType", "likelihood", "parameters", "type"],
    "ParticleIDs_0": [],
    "Particleidx0": ["collectionID", "index"],
    "Particleidx1": ["collectionID", "index"],
    "Photonidx0": ["collectionID", "index"],
    "ReconstructedParticles": [
        "E",
        "Electronidx0_indexGlobal",
        "MCRecoAssociationsidx0_indexGlobal",
        "Muonidx0_indexGlobal",
        "charge",
        "clusters",
        "covMatrix_10_",
        "goodnessOfPID",
        "mass",
        "particleIDs",
        "particles",
        "px",
        "py",
        "pz",
        "referencePoint",
        "tracks",
        "type",
    ],
    "ReconstructedParticlesidx0": ["collectionID", "index"],
    "ReconstructedParticlesidx1": ["collectionID", "index"],
    "ReconstructedParticlesidx2": ["collectionID", "index"],
    "ReconstructedParticlesidx3": ["collectionID", "index"],
    "ReconstructedParticlesidx4": ["collectionID", "index"],
    "ReconstructedParticlesidx5": ["collectionID", "index"],
    "TrackerHits": [
        "cellID",
        "covMatrix_6_",
        "eDep",
        "eDepError",
        "position",
        "quality",
        "rawHits",
        "time",
        "type",
    ],
    "TrackerHits_0": ["collectionID", "index"],
    "magFieldBz": [],
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


# Test 2: Do all the relations and links work?
def test_MC_daughters(delayed_events):
    d = delayed_events.Particle.get_daughters.compute()
    assert isinstance(d, awkward.highlevel.Array)
    assert d.layout.branch_depth[1] == 3
    assert d.fields == delayed_events.Particle.fields


def test_MC_parents(delayed_events):
    p = delayed_events.Particle.get_parents.compute()
    assert isinstance(p, awkward.highlevel.Array)
    assert p.layout.branch_depth[1] == 3
    assert p.fields == delayed_events.Particle.fields


def test_MCRecoAssociations(delayed_events):
    mr = delayed_events.MCRecoAssociations.reco_mc.compute()
    assert isinstance(mr, awkward.highlevel.Array)
    assert mr.layout.branch_depth[1] == 3


# Validity Tests
def test_RecoMC_mass_diference(delayed_events):
    """For this particular sample, the difference between
    the mc mass and reco mass in all the events is less than 1 GeV"""
    mr = delayed_events.MCRecoAssociations.reco_mc.compute()
    r, m = mr[:, :, 0], mr[:, :, 1]
    diff = awkward.flatten(m.mass - r.mass)
    assert awkward.all(diff < 1.0)


def test_KaonParent_to_PionDaughters_Loop(eager_events):
    """Test to thoroughly check get_parents and get_daughters
    - We look at the decay of Kaon $K_S^0 \\rightarrow pions $
    - Two decay modes:
        $$ K_S^0 \\rightarrow \\pi^0 + \\pi^0 $$
        $$ K_S^0 \\rightarrow \\pi^+ + \\pi^- $$
    """
    PDG_IDs = {"K(S)0": 310, "pi+": 211, "pi-": -211, "pi0": 111}
    mc = eager_events.Particle

    # Find Single K(S)0
    K_S0_cut = mc.PDG == PDG_IDs["K(S)0"]
    K_S0 = mc[K_S0_cut]
    single_K_S0_cut = awkward.num(K_S0, axis=1) == 1
    single_K_S0 = K_S0[single_K_S0_cut]

    # Daughter Test
    # The Kaon K(S)0 must have only pions as the daughters

    # Find the daughters of Single K(S)0
    daughters_of_K_S0 = single_K_S0.get_daughters

    # Some K_S0 can go undetected (I think)
    # Ensure that at least one daughter is available per event
    bool_non_empty_daughter_list = awkward.num(daughters_of_K_S0, axis=2) > 0
    daughters_of_K_S0 = daughters_of_K_S0[
        awkward.flatten(bool_non_empty_daughter_list, axis=1)
    ]

    # Are these valid daughter particles (pi+ or pi- or pi0)?
    flat_PDG = awkward.ravel(daughters_of_K_S0.PDG)
    is_pi_0 = flat_PDG == PDG_IDs["pi0"]
    is_pi_plus = flat_PDG == PDG_IDs["pi+"]
    is_pi_minus = flat_PDG == PDG_IDs["pi-"]
    names_valid = awkward.all(is_pi_0 | is_pi_plus | is_pi_minus)
    assert names_valid

    # Do the daughters have valid charges (-1 or 0)?
    nested_bool = awkward.prod(daughters_of_K_S0.charge, axis=2) <= 0
    charge_valid = awkward.all(awkward.ravel(nested_bool))
    assert charge_valid

    # Parent Test
    # These pion daughters, just generated, must point back to the single parent K(S)0

    p = daughters_of_K_S0.get_parents

    # Do the daughters have a single parent?
    nested_bool_daughter = awkward.num(p, axis=3) == 1
    daughters_have_single_parent = awkward.all(awkward.ravel(nested_bool_daughter))
    assert daughters_have_single_parent

    # Is that parent K(S)0 ?
    nested_bool_parent = p.PDG == PDG_IDs["K(S)0"]
    daughters_have_K_S0_parent = awkward.all(awkward.ravel(nested_bool_parent))
    assert daughters_have_K_S0_parent
