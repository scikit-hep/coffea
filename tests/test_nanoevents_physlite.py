from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema


def test_entry_start_and_entry_stop():
    access_log = []
    NanoEventsFactory.from_root(
        {"tests/samples/PHYSLITE_example.root": "CollectionTree"},
        mode="virtual",
        schemaclass=PHYSLITESchema,
        access_log=access_log,
    ).events()
    assert access_log == []
