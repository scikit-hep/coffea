import os

import awkward
import pytest

from coffea.nanoevents import FCC, NanoEventsFactory


def _events(**kwargs):
    path = os.path.abspath("tests/samples/p8_ee_WW_ecm240_edm4hep.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=FCC.get_schema(version="edm4hep1"), **kwargs
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
        "CalorimeterHits",
        "EFlowNeutralHadron",
        "EFlowPhoton",
        "EFlowTrack",
        "EFlowTrack_L",
        "EFlowTrack_dNdx",
        "Electron_IsolationVar",
        "Electron_objIdx",
        "EventHeader",
        "GPDoubleKeys",
        "GPDoubleValues",
        "GPFloatKeys",
        "GPFloatValues",
        "GPIntKeys",
        "GPIntValues",
        "GPStringKeys",
        "GPStringValues",
        "Jet",
        "MCRecoAssociations",
        "Muon_IsolationVar",
        "Muon_objIdx",
        "Particle",
        "ParticleIDs",
        "Photon_IsolationVar",
        "Photon_objIdx",
        "ReconstructedParticles",
        "TrackerHits",
        "magFieldBz",
    ],
)
def test_field_is_present(eager_events, delayed_events, field):
    eager_fields = eager_events.fields
    delayed_fields = delayed_events.fields
    assert field in eager_fields
    assert field in delayed_fields


def test_MC_daughters(delayed_events):
    d = delayed_events.Particle.Map_Relation("daughters", "Particle").compute()
    assert isinstance(d, awkward.highlevel.Array)
    assert d.layout.branch_depth[1] == 3


def test_MC_parents(delayed_events):
    p = delayed_events.Particle.Map_Relation("parents", "Particle").compute()
    assert isinstance(p, awkward.highlevel.Array)
    assert p.layout.branch_depth[1] == 3


# Todo: Add Link tests
#


def test_KaonParent_to_PionDaughters_Loop(eager_events):
    """Test to thoroughly check parents and daughters
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
    daughters_of_K_S0 = single_K_S0.Map_Relation("daughters", "Particle")

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

    p = daughters_of_K_S0.Map_Relation("parents", "Particle")

    # Do the daughters have a single parent?
    nested_bool_daughter = awkward.num(p, axis=3) == 1
    daughters_have_single_parent = awkward.all(awkward.ravel(nested_bool_daughter))
    assert daughters_have_single_parent

    # Is that parent K(S)0 ?
    nested_bool_parent = p.PDG == PDG_IDs["K(S)0"]
    daughters_have_K_S0_parent = awkward.all(awkward.ravel(nested_bool_parent))
    assert daughters_have_K_S0_parent
