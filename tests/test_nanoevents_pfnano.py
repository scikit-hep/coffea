import os

import awkward as ak
import pytest
import uproot

from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema


@pytest.fixture(scope="module")
def events(tests_directory):
    pytest.importorskip("dask_awkward")
    path = os.path.join(tests_directory, "samples/pfnano.root")
    events = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=PFNanoAODSchema,
        mode="dask",
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


def test_uproot_write(tmp_path):
    path = os.path.abspath("tests/samples/pfnano.root")
    orig_events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=PFNanoAODSchema, mode="eager"
    ).events()

    out_path = str(tmp_path / "pfnano_write_test.root")
    with uproot.recreate(out_path) as f:
        f.mktree("Events", PFNanoAODSchema.uproot_writeable(orig_events))

    test_events = NanoEventsFactory.from_root(
        {out_path: "Events"},
        schemaclass=PFNanoAODSchema,
        mode="eager",
    ).events()

    assert len(orig_events) == len(test_events)
    assert ak.all(orig_events.event == test_events.event)
    assert ak.all(orig_events.Muon.pt == test_events.Muon.pt)
    assert ak.all(orig_events.Jet.pt == test_events.Jet.pt)
    assert ak.all(orig_events.FatJet.pt == test_events.FatJet.pt)
