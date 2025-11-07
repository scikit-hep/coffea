from uproot import ReadOnlyDirectory

from coffea.compute.backends.threaded import SingleThreadedBackend
from coffea.compute.data import (
    DataGroup,
    Dataset,
    File,
    FileElement,
    InputDataGroup,
    InputDataset,
    StepElement,
)
from coffea.compute.func import EventsArray


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
        StepElement("file1.root", (0, 100)),
        StepElement("file1.root", (100, 200)),
        StepElement("file1.root", (200, 300)),
        StepElement("file2.root", (0, 100)),
        StepElement("file2.root", (100, 200)),
        StepElement("file2.root", (200, 300)),
        StepElement("file3.root", (0, 150)),
        StepElement("file3.root", (150, 300)),
    ]

    group.traversal = "breadth"
    assert list(group.iter_steps()) == [
        StepElement("file1.root", (0, 100)),
        StepElement("file3.root", (0, 150)),
        StepElement("file1.root", (100, 200)),
        StepElement("file3.root", (150, 300)),
        StepElement("file1.root", (200, 300)),
        StepElement("file2.root", (0, 100)),
        StepElement("file2.root", (100, 200)),
        StepElement("file2.root", (200, 300)),
    ]

    for dataset in group.datasets.values():
        dataset.traversal = "breadth"

    assert list(group.iter_steps()) == [
        StepElement("file1.root", (0, 100)),
        StepElement("file3.root", (0, 150)),
        StepElement("file2.root", (0, 100)),
        StepElement("file3.root", (150, 300)),
        StepElement("file1.root", (100, 200)),
        StepElement("file2.root", (100, 200)),
        StepElement("file1.root", (200, 300)),
        StepElement("file2.root", (200, 300)),
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
        FileElement(path="file1.root"),
        FileElement(path="file2.root"),
        FileElement(path="file3.root"),
    ]

    input_group = InputDataGroup(
        {
            "singlemuon": InputDataset(files=["file1.root", "file2.root"]),
            "drellyan": InputDataset(files=["file3.root"]),
        }
    )

    assert list(input_group.iter_files()) == [
        FileElement(path="file1.root"),
        FileElement(path="file2.root"),
        FileElement(path="file3.root"),
    ]


def fake_uproot_open(path: str, **kwargs):
    class FakeFile:
        def __init__(self, path: str):
            self.path = path

    class FakeReadOnlyDirectory:
        def __init__(self, path: str):
            self.file = FakeFile(path)

    return FakeReadOnlyDirectory(path)


def make_up_steps(root_dir: ReadOnlyDirectory) -> Dataset:
    return Dataset(
        files=[
            File(
                # TODO: clear that we need metadata coming in with root_dir
                path=root_dir.file.path,
                steps=[
                    (0, 100),
                    (100, 200),
                ],
            )
        ],
    )


def test_prepare(mocker):
    mocker.patch("uproot.open", new=fake_uproot_open)

    # TODO: how does this look with InputDataGroup?
    input_data = InputDataset(files=["file1.root", "file2.root"])
    prepare = input_data.map_files(make_up_steps)

    with SingleThreadedBackend() as backend:
        prepared = backend.compute(prepare).result()

    assert isinstance(prepared, Dataset)
    assert prepared == Dataset(
        files=[
            File(path="file1.root", steps=[(0, 100), (100, 200)]),
            File(path="file2.root", steps=[(0, 100), (100, 200)]),
        ]
    )

    # Now we could process the dataset
    class CountStepsProcessor:
        def process(self, events: EventsArray) -> int:
            return len(events)

    computable = prepared.map_steps(CountStepsProcessor())
    with SingleThreadedBackend() as backend:
        result = backend.compute(computable).result()
    assert result == 400  # 2 files * 2 steps * 100 events each
