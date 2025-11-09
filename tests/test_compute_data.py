from pytest_mock import MockerFixture

from coffea.compute.data import (
    Chunk,
    DataGroup,
    Dataset,
    File,
    FileContextDataGroup,
    FileContextDataset,
    FileElement,
    InputDataGroup,
    InputDataset,
    OpenFile,
    StepContextDataGroup,
    StepContextDataset,
    StepContextFile,
    StepElement,
)
from coffea.compute.func import EventsArray
from coffea.compute.protocol import EmptyResult


def _make_test_group() -> DataGroup:
    return DataGroup(
        datasets=[
            Dataset(
                files=[
                    File(path="file1.root", steps=[(0, 100), (100, 200), (200, 300)]),
                    File(path="file2.root", steps=[(0, 100), (100, 200), (200, 300)]),
                ],
                name="singlemuon",
            ),
            Dataset(
                files=[
                    File(path="file3.root", steps=[(0, 150), (150, 300)]),
                ],
                name="drellyan",
            ),
        ],
        name="my_analysis",
    )


def test_make_stepiterables():
    group = _make_test_group()

    expected_ez = {
        "file1:0-100": StepElement(
            (0, 100),
            StepContextDataGroup("file1.root", "", "singlemuon", 1.0, "my_analysis"),
        ),
        "file1:100-200": StepElement(
            (100, 200),
            StepContextDataGroup("file1.root", "", "singlemuon", 1.0, "my_analysis"),
        ),
        "file1:200-300": StepElement(
            (200, 300),
            StepContextDataGroup("file1.root", "", "singlemuon", 1.0, "my_analysis"),
        ),
        "file2:0-100": StepElement(
            (0, 100),
            StepContextDataGroup("file2.root", "", "singlemuon", 1.0, "my_analysis"),
        ),
        "file2:100-200": StepElement(
            (100, 200),
            StepContextDataGroup("file2.root", "", "singlemuon", 1.0, "my_analysis"),
        ),
        "file2:200-300": StepElement(
            (200, 300),
            StepContextDataGroup("file2.root", "", "singlemuon", 1.0, "my_analysis"),
        ),
        "file3:0-150": StepElement(
            (0, 150),
            StepContextDataGroup("file3.root", "", "drellyan", 1.0, "my_analysis"),
        ),
        "file3:150-300": StepElement(
            (150, 300),
            StepContextDataGroup("file3.root", "", "drellyan", 1.0, "my_analysis"),
        ),
    }

    expected = [
        expected_ez[val]
        for val in [
            "file1:0-100",
            "file1:100-200",
            "file1:200-300",
            "file2:0-100",
            "file2:100-200",
            "file2:200-300",
            "file3:0-150",
            "file3:150-300",
        ]
    ]
    assert list(group.iter_steps()) == expected

    group.traversal = "breadth"
    expected = [
        expected_ez[val]
        for val in [
            "file1:0-100",
            "file3:0-150",
            "file1:100-200",
            "file3:150-300",
            "file1:200-300",
            "file2:0-100",
            "file2:100-200",
            "file2:200-300",
        ]
    ]
    assert list(group.iter_steps()) == expected

    for dataset in group.datasets:
        dataset.traversal = "breadth"

    expected = [
        expected_ez[val]
        for val in [
            "file1:0-100",
            "file3:0-150",
            "file2:0-100",
            "file3:150-300",
            "file1:100-200",
            "file2:100-200",
            "file1:200-300",
            "file2:200-300",
        ]
    ]
    assert list(group.iter_steps()) == expected

    # Check also that iterating at a lower leven produces the correct contexts
    dataset = group.datasets[1]
    dataset.traversal = "depth"
    expected = [
        StepElement(
            (0, 150),
            StepContextDataset("file3.root", "", "drellyan", 1.0),
        ),
        StepElement(
            (150, 300),
            StepContextDataset("file3.root", "", "drellyan", 1.0),
        ),
    ]
    assert list(dataset.iter_steps()) == expected

    file = dataset.files[0]
    expected = [
        StepElement(
            (0, 150),
            StepContextFile("file3.root", ""),
        ),
        StepElement(
            (150, 300),
            StepContextFile("file3.root", ""),
        ),
    ]
    assert list(file.iter_steps()) == expected


def test_make_fileiterables():
    group = _make_test_group()

    expected = [
        FileElement(
            path="file1.root",
            context=FileContextDataGroup("singlemuon", 1.0, "my_analysis"),
        ),
        FileElement(
            path="file2.root",
            context=FileContextDataGroup("singlemuon", 1.0, "my_analysis"),
        ),
        FileElement(
            path="file3.root",
            context=FileContextDataGroup("drellyan", 1.0, "my_analysis"),
        ),
    ]
    assert list(group.iter_files()) == expected

    input_group = InputDataGroup(
        [
            InputDataset(files=["file1.root", "file2.root"], name="singlemuon"),
            InputDataset(files=["file3.root"], name="drellyan"),
        ],
        name="my_analysis",
    )

    assert list(input_group.iter_files()) == expected


def fake_uproot_open(path: str, **kwargs):
    class FakeFile:
        def __init__(self, path: str):
            self.path = path

    class FakeReadOnlyDirectory:
        def __init__(self, path: str):
            self.file = FakeFile(path)

    return FakeReadOnlyDirectory(path)


def make_up_steps(file: OpenFile[FileContextDataset]) -> Dataset:
    return Dataset(
        files=[
            File(
                # TODO: clear that we need metadata coming in with root_dir
                path=file.file_path,
                steps=[
                    (0, 100),
                    (100, 200),
                ],
            )
        ],
        name=file.context.dataset_name,
    )


def test_prepare(mocker: MockerFixture) -> None:
    mocker.patch("uproot.open", new=fake_uproot_open)

    # TODO: how does this look with InputDataGroup?
    input_data = InputDataset(files=["file1.root", "file2.root"], name="singlemuon")
    prepare = input_data.map_files(make_up_steps)

    prepared = sum((f() for f in prepare), EmptyResult())
    assert isinstance(prepared, Dataset)
    assert prepared == Dataset(
        files=[
            File(path="file1.root", steps=[(0, 100), (100, 200)]),
            File(path="file2.root", steps=[(0, 100), (100, 200)]),
        ],
        name="singlemuon",
    )

    # Now we could process the dataset
    class CountStepsProcessor:
        def process(self, events: EventsArray) -> int:
            return len(events)

    computable = prepared.map_steps(CountStepsProcessor())
    result = sum((f() for f in computable), 0)
    assert result == 400  # 2 files * 2 steps * 100 events each

    def count_steps_func(chunk: Chunk[StepContextDataset]) -> int:
        chunk.context.uuid
        return len(chunk.events)

    computable = prepared.map_steps(count_steps_func)
    result = sum((f() for f in computable), 0)
    assert result == 400  # 2 files * 2 steps * 100 events
