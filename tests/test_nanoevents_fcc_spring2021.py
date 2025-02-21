import os

import awkward
import pytest

from coffea.nanoevents import FCC, NanoEventsFactory


def _events(**kwargs):
    # Path to original sample : /eos/experiment/fcc/ee/generation/DelphesEvents/spring2021/IDEA/p8_ee_ZH_ecm240/events_082532938.root
    path = os.path.abspath("tests/samples/test_FCC_Spring2021.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=FCC.get_schema(version="pre-edm4hep1"), **kwargs
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
        "AllMuonidx0",
        "EFlowNeutralHadron",
        "EFlowNeutralHadron_0",
        "EFlowNeutralHadron_1",
        "EFlowNeutralHadron_2",
        "EFlowNeutralHadronidx0",
        "EFlowNeutralHadronidx1",
        "EFlowNeutralHadronidx2",
        "EFlowPhoton",
        "EFlowPhoton_0",
        "EFlowPhoton_1",
        "EFlowPhoton_2",
        "EFlowPhotonidx0",
        "EFlowPhotonidx1",
        "EFlowPhotonidx2",
        "EFlowTrack",
        "EFlowTrack_0",
        "EFlowTrack_1",
        "EFlowTrackidx0",
        "EFlowTrackidx1",
        "Electronidx0",
        "Jet",
        "Jetidx0",
        "Jetidx1",
        "Jetidx2",
        "Jetidx3",
        "Jetidx4",
        "Jetidx5",
        "MCRecoAssociations",
        "MissingET",
        "MissingETidx0",
        "MissingETidx1",
        "MissingETidx2",
        "MissingETidx3",
        "MissingETidx4",
        "MissingETidx5",
        "Muonidx0",
        "Particle",
        "ParticleIDs",
        "ParticleIDs_0",
        "Particleidx0",
        "Particleidx1",
        "Photonidx0",
        "ReconstructedParticles",
        "ReconstructedParticlesidx0",
        "ReconstructedParticlesidx1",
        "ReconstructedParticlesidx2",
        "ReconstructedParticlesidx3",
        "ReconstructedParticlesidx4",
        "ReconstructedParticlesidx5",
    ],
)
def test_field_is_present(eager_events, delayed_events, field):
    eager_fields = eager_events.fields
    delayed_fields = delayed_events.fields
    assert field in eager_fields
    assert field in delayed_fields


# Faulty Test: Returns Passed even with no events
# For now, commenting it out
# def test_lorentz_behavior(delayed_events):
#     assert delayed_events.Particle.behavior["LorentzVector"] == LorentzVectorRecord
#     assert (
#         delayed_events.ReconstructedParticles.behavior["LorentzVector"]
#         == LorentzVectorRecord
#     )
#     assert isinstance(delayed_events.Particle.eta.compute(), awkward.highlevel.Array)
#     assert isinstance(
#         delayed_events.ReconstructedParticles.eta.compute(), awkward.highlevel.Array
#     )


def test_MC_daughters(delayed_events):
    d = delayed_events.Particle.get_daughters.compute()
    assert isinstance(d, awkward.highlevel.Array)
    assert d.layout.branch_depth[1] == 3


def test_MC_parents(delayed_events):
    p = delayed_events.Particle.get_parents.compute()
    assert isinstance(p, awkward.highlevel.Array)
    assert p.layout.branch_depth[1] == 3


def test_MCRecoAssociations(delayed_events):
    mr = delayed_events.MCRecoAssociations.reco_mc.compute()
    assert isinstance(mr, awkward.highlevel.Array)
    assert mr.layout.branch_depth[1] == 3


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
