import os

import awkward
import numpy
import pytest

from coffea.nanoevents import FCC, NanoEventsFactory


# Basic Tests
def _events(**kwargs):
    # Path to original sample: /eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_mumuH_Hbb_ecm240/events_159112833.root
    path = os.path.abspath("tests/samples/p8_ee_WW_ecm240_edm4hep.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=FCC.get_schema(version="latest"), **kwargs
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
        "clusters_idx_EFlowNeutralHadron_collectionID",
        "clusters_idx_EFlowNeutralHadron_index",
        "clusters_idx_EFlowNeutralHadron_index_Global",
        "clusters_idx_EFlowPhoton_collectionID",
        "clusters_idx_EFlowPhoton_index",
        "clusters_idx_EFlowPhoton_index_Global",
        "directionError",
        "energyError",
        "hits_idx_CalorimeterHits_collectionID",
        "hits_idx_CalorimeterHits_index",
        "hits_idx_CalorimeterHits_index_Global",
        "iTheta",
        "phi",
        "position",
        "positionError",
        "shapeParameters",
        "subdetectorEnergies",
        "type",
    ],
    "EFlowPhoton": [
        "E",
        "clusters_idx_EFlowNeutralHadron_collectionID",
        "clusters_idx_EFlowNeutralHadron_index",
        "clusters_idx_EFlowNeutralHadron_index_Global",
        "clusters_idx_EFlowPhoton_collectionID",
        "clusters_idx_EFlowPhoton_index",
        "clusters_idx_EFlowPhoton_index_Global",
        "directionError",
        "energyError",
        "hits_idx_CalorimeterHits_collectionID",
        "hits_idx_CalorimeterHits_index",
        "hits_idx_CalorimeterHits_index_Global",
        "iTheta",
        "phi",
        "position",
        "positionError",
        "shapeParameters",
        "subdetectorEnergies",
        "type",
    ],
    "EFlowTrack": [
        "Nholes",
        "chi2",
        "ndf",
        "subdetectorHitNumbers",
        "subdetectorHoleNumbers",
        "trackStates",
        "tracks_idx_EFlowTrack_collectionID",
        "tracks_idx_EFlowTrack_index",
        "tracks_idx_EFlowTrack_index_Global",
        "type",
    ],
    "EFlowTrack_L": [],
    "EFlowTrack_dNdx": [
        "dQdx",
        "track_idx_EFlowTrack_collectionID",
        "track_idx_EFlowTrack_index",
        "track_idx_EFlowTrack_index_Global",
    ],
    "Electron_IsolationVar": [],
    "Electron_objIdx": ["collectionID", "index"],
    "EventHeader": ["eventNumber", "runNumber", "timeStamp", "weight", "weights"],
    "GPDoubleKeys": [],
    "GPDoubleValues": [],
    "GPFloatKeys": [],
    "GPFloatValues": [],
    "GPIntKeys": [],
    "GPIntValues": [],
    "GPStringKeys": [],
    "GPStringValues": [],
    "Jet": [
        "E",
        "PDG",
        "charge",
        "clusters_idx_EFlowNeutralHadron_collectionID",
        "clusters_idx_EFlowNeutralHadron_index",
        "clusters_idx_EFlowNeutralHadron_index_Global",
        "clusters_idx_EFlowPhoton_collectionID",
        "clusters_idx_EFlowPhoton_index",
        "clusters_idx_EFlowPhoton_index_Global",
        "covMatrix",
        "goodnessOfPID",
        "mass",
        "particles_idx_Jet_collectionID",
        "particles_idx_Jet_index",
        "particles_idx_Jet_index_Global",
        "particles_idx_ReconstructedParticles_collectionID",
        "particles_idx_ReconstructedParticles_index",
        "particles_idx_ReconstructedParticles_index_Global",
        "px",
        "py",
        "pz",
        "referencePoint",
        "tracks_idx_EFlowTrack_collectionID",
        "tracks_idx_EFlowTrack_index",
        "tracks_idx_EFlowTrack_index_Global",
    ],
    "MCRecoAssociations": [
        "Link_from_Jet",
        "Link_from_ReconstructedParticles",
        "Link_to_Particle",
        "weight",
    ],
    "Muon_IsolationVar": [],
    "Muon_objIdx": ["collectionID", "index"],
    "Particle": [
        "PDG",
        "charge",
        "colorFlow",
        "daughters_idx_Particle_collectionID",
        "daughters_idx_Particle_index",
        "daughters_idx_Particle_index_Global",
        "endpoint",
        "generatorStatus",
        "mass",
        "momentumAtEndpoint",
        "parents_idx_Particle_collectionID",
        "parents_idx_Particle_index",
        "parents_idx_Particle_index_Global",
        "px",
        "py",
        "pz",
        "simulatorStatus",
        "spin",
        "time",
        "vertex",
    ],
    "ParticleIDs": [
        "PDG",
        "algorithmType",
        "likelihood",
        "parameters",
        "particle_idx_Jet_collectionID",
        "particle_idx_Jet_index",
        "particle_idx_Jet_index_Global",
        "particle_idx_ReconstructedParticles_collectionID",
        "particle_idx_ReconstructedParticles_index",
        "particle_idx_ReconstructedParticles_index_Global",
        "type",
    ],
    "Photon_IsolationVar": [],
    "Photon_objIdx": ["collectionID", "index"],
    "ReconstructedParticles": [
        "E",
        "Link_from_ReconstructedParticles",
        "Link_to_Particle",
        "PDG",
        "charge",
        "clusters_idx_EFlowNeutralHadron_collectionID",
        "clusters_idx_EFlowNeutralHadron_index",
        "clusters_idx_EFlowNeutralHadron_index_Global",
        "clusters_idx_EFlowPhoton_collectionID",
        "clusters_idx_EFlowPhoton_index",
        "clusters_idx_EFlowPhoton_index_Global",
        "covMatrix",
        "goodnessOfPID",
        "mass",
        "particles_idx_Jet_collectionID",
        "particles_idx_Jet_index",
        "particles_idx_Jet_index_Global",
        "particles_idx_ReconstructedParticles_collectionID",
        "particles_idx_ReconstructedParticles_index",
        "particles_idx_ReconstructedParticles_index_Global",
        "px",
        "py",
        "pz",
        "referencePoint",
        "tracks_idx_EFlowTrack_collectionID",
        "tracks_idx_EFlowTrack_index",
        "tracks_idx_EFlowTrack_index_Global",
    ],
    "TrackerHits": [
        "cellID",
        "covMatrix",
        "eDep",
        "eDepError",
        "position",
        "quality",
        "time",
        "type",
    ],
    "magFieldBz": [],
}

all_methods = {
    "Particle": {"get_daughters": ("PDG", [-24, 24]), "get_parents": ("PDG", [-11])},
    "ReconstructedParticles": {
        "match_gen": ("PDG", numpy.int32(211)),
        "get_cluster_photons": (None, [[]]),
        "get_reconstructedparticles": (None, [[]]),
        "get_tracks": (
            "trackStates",
            [
                {
                    "location": [0.0],
                    "D0": [0.0003872188972309232],
                    "phi": [-1.4047781229019165],
                    "omega": [-0.00011777195322792977],
                    "Z0": [-0.5684034824371338],
                    "tanLambda": [-0.13527415692806244],
                    "time": [0.0],
                    "referencePoint": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                    "covMatrix": [
                        [
                            1.7015145203913562e-05,
                            -9.489050398769905e-07,
                            7.143341917981161e-08,
                            1.5243341528262122e-12,
                            -1.8112620507170635e-13,
                            2.9262629340877973e-15,
                            -2.0860079885665073e-08,
                            1.1119254406111168e-09,
                            1.4423419056330822e-11,
                            5.319845513440669e-06,
                            1.4412778748251753e-09,
                            -7.614676250655705e-11,
                            -1.1035737211215202e-12,
                            -6.439618971398886e-08,
                            4.636626194098881e-09,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ],
                }
            ],
        ),
    },
    "ParticleIDs": {"get_reconstructedparticles": ("E", numpy.float32(5.1393394))},
    "EFlowNeutralHadron": {
        "get_cluster_photons": (None, [[]]),
        "get_hits": ("time", [1.01947987829476e-08]),
    },
    "EFlowPhoton": {
        "get_cluster_photons": (None, [[]]),
        "get_hits": ("time", [7.57331264367167e-09]),
    },
    "EFlowTrack": {"get_tracks": (None, [[]])},
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


# Test 2: Do all the methods work?
@pytest.mark.parametrize(
    "collection",
    all_methods.keys(),
)
def test_methods(eager_events, delayed_events, collection):
    """Test all the methods and also check their validity by comparing with known result"""
    ev, item = 4, 3
    numeric_types = (float, int, numpy.float32, numpy.float64, numpy.int32, numpy.int64)
    for method, attr_value in all_methods[collection].items():
        attr, value = attr_value
        eager_value = getattr(eager_events[collection], method)[ev][item]
        delayed_value = getattr(delayed_events[collection], method).compute()[ev][item]
        if value is not None:
            final_eager_value = eager_value[attr]
            if not isinstance(final_eager_value, numeric_types):
                final_eager_value = final_eager_value.to_list()
            final_delayed_value = delayed_value[attr]
            if not isinstance(final_delayed_value, numeric_types):
                final_delayed_value = final_delayed_value.to_list()
            assert final_eager_value == value
            assert final_delayed_value == value


# Test 3: Other Validity Tests
def test_RecoMC_mass_diference(delayed_events):
    """For this particular sample, the difference between
    the mc mass and reco mass in all the events is less than 1 GeV"""
    r = delayed_events.ReconstructedParticles
    m = r.match_gen
    diff = awkward.flatten(m.mass - r.mass)
    assert awkward.all(diff.compute() < 1.0)


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
