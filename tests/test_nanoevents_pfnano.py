import os

import pytest

from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema


@pytest.fixture(scope="module")
def events(tests_directory):
    path = os.path.join(tests_directory, "samples/pfnano.root")
    events = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=PFNanoAODSchema,
        delayed=True,
    ).events()
    return events


@pytest.mark.parametrize(
    "field",
    [
        # Jet Sanity check
        "Jet",
        # associated PF candidate
        "Jet.constituents",
        "Jet.constituents.pt",
        "Jet.constituents.pf",
        "Jet.constituents.pf.pt",
        "Jet.constituents.pf.eta",
        # FatJet sanity check
        "FatJet",
        # Subjet collection (secondary sanity check)
        "FatJet.subjets",
        "FatJet.subjets.pt",
        # TODO: Example file does not have constituents for fat jets
        # "FatJet.constituents.pt",
        # "FatJet.constituents.pf",
        # "FatJet.constituents.pf.eta",
    ],
)
def test_nested_collections(events, field):
    def check_fields_recursive(coll, field):
        if "." not in field:
            assert hasattr(coll, field)
        else:
            split = field.split(".")
            return check_fields_recursive(getattr(coll, split[0]), ".".join(split[1:]))

    check_fields_recursive(events, field)
