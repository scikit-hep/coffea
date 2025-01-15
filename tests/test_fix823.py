from pathlib import Path

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory


def test_explicit_delete_after_assign():
    data_dir = Path(__file__).parent / "samples"
    testfile = data_dir / "nano_dy.root"

    events = NanoEventsFactory.from_root(
        {testfile: "Events"},
        metadata={"dataset": "nano_dy"},
        schemaclass=NanoAODSchema,
    ).events()

    genpart = events["GenPart"]
    del events
    _ = genpart.parent
