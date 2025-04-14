#!/usr/bin/env python3

from coffea.processor.executor import WorkItem


def test_work_item():
    item1 = WorkItem("TestDataSet", "/a/b/c.root", "Events", 500, 670, "abc", {})
    item2 = WorkItem(
        "TestDataSet", "/a/b/c.root", "Events", 500, 670, "abc", {"meta": "data"}
    )
    item3 = WorkItem("TestDataSet", "/a/b/c.root", "Events", 500, 760, "abc", {})

    assert item1 == item1
    assert item1 == item2
    assert item1 != item3
    assert item1.dataset == "TestDataSet"
    assert item1.filename == "/a/b/c.root"
    assert item1.treename == "Events"
    assert item1.entrystart == 500
    assert item1.entrystop == 670
    assert item1.fileuuid == "abc"
    assert len(item1) == 670 - 500
    assert len(item3) == 760 - 500

    # Test if hashable
    hash(item2)

    # Test if usermeta is mutable
    item1.usermeta["user"] = "meta"
