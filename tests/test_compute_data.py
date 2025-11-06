from coffea.compute.data import DataGroup, Dataset, File, Step


def test_make_stepiterables():
    group = DataGroup(
        {
            "singlemuon": Dataset(
                files=[
                    File(path="file1.root", steps=[(0, 100), (100, 200), (200, 300)]),
                    File(path="file2.root", steps=[(0, 100), (100, 200), (200, 300)]),
                ]
            ),
            "drellyan": Dataset(
                files=[
                    File(path="file3.root", steps=[(0, 150), (150, 300)]),
                ]
            ),
        }
    )

    assert list(group.iter_steps()) == [
        Step("file1.root", (0, 100)),
        Step("file1.root", (100, 200)),
        Step("file1.root", (200, 300)),
        Step("file2.root", (0, 100)),
        Step("file2.root", (100, 200)),
        Step("file2.root", (200, 300)),
        Step("file3.root", (0, 150)),
        Step("file3.root", (150, 300)),
    ]

    group.traversal = "breadth"
    assert list(group.iter_steps()) == [
        Step("file1.root", (0, 100)),
        Step("file3.root", (0, 150)),
        Step("file1.root", (100, 200)),
        Step("file3.root", (150, 300)),
        Step("file1.root", (200, 300)),
        Step("file2.root", (0, 100)),
        Step("file2.root", (100, 200)),
        Step("file2.root", (200, 300)),
    ]

    for dataset in group.datasets.values():
        dataset.traversal = "breadth"

    assert list(group.iter_steps()) == [
        Step("file1.root", (0, 100)),
        Step("file3.root", (0, 150)),
        Step("file2.root", (0, 100)),
        Step("file3.root", (150, 300)),
        Step("file1.root", (100, 200)),
        Step("file2.root", (100, 200)),
        Step("file1.root", (200, 300)),
        Step("file2.root", (200, 300)),
    ]


def test_make_fileiterables():
    group = DataGroup(
        {
            "singlemuon": Dataset(
                files=[
                    File(path="file1.root", steps=[(0, 100), (100, 200), (200, 300)]),
                    File(path="file2.root", steps=[(0, 100), (100, 200), (200, 300)]),
                ]
            ),
            "drellyan": Dataset(
                files=[
                    File(path="file3.root", steps=[(0, 150), (150, 300)]),
                ]
            ),
        }
    )

    assert list(group.iter_files()) == [
        File(path="file1.root", steps=[(0, 100), (100, 200), (200, 300)]),
        File(path="file2.root", steps=[(0, 100), (100, 200), (200, 300)]),
        File(path="file3.root", steps=[(0, 150), (150, 300)]),
    ]
