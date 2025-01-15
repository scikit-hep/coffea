import os

import pytest

from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema


@pytest.fixture(scope="module")
def events(tests_directory):
    path = os.path.join(tests_directory, "samples/scouting_nano.root")
    ScoutingNanoAODSchema.warn_missing_crossrefs = False
    events = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=ScoutingNanoAODSchema,
        delayed=True,
    ).events()
    return events


@pytest.mark.parametrize(
    "field",
    [
        # Jet Sanity check
        "ScoutingJet",
        # associated PF candidate
        "ScoutingJet.pt",
        # FatJet sanity check
        "ScoutingFatJet",
        # Subjet collection (secondary sanity check)
        "ScoutingFatJet.pt",
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
