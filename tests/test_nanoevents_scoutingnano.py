import os

import awkward as ak
import pytest
import uproot

from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema


@pytest.fixture(scope="module")
def events(tests_directory):
    pytest.importorskip("dask_awkward")
    path = os.path.join(tests_directory, "samples/scouting_nano.root")
    ScoutingNanoAODSchema.warn_missing_crossrefs = False
    events = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=ScoutingNanoAODSchema,
        mode="dask",
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


def test_uproot_write(tmp_path):
    path = os.path.abspath("tests/samples/scouting_nano.root")
    ScoutingNanoAODSchema.warn_missing_crossrefs = False
    orig_events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=ScoutingNanoAODSchema, mode="eager"
    ).events()

    out_path = str(tmp_path / "scouting_nano_write_test.root")
    with uproot.recreate(out_path) as f:
        f.mktree("Events", ScoutingNanoAODSchema.uproot_writeable(orig_events))

    test_events = NanoEventsFactory.from_root(
        {out_path: "Events"},
        schemaclass=ScoutingNanoAODSchema,
        mode="eager",
    ).events()

    assert len(orig_events) == len(test_events)
    assert ak.all(orig_events.ScoutingJet.pt == test_events.ScoutingJet.pt)
    assert ak.all(orig_events.ScoutingFatJet.pt == test_events.ScoutingFatJet.pt)
