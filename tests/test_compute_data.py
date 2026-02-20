from typing import Any

from pytest_mock import MockerFixture

from coffea.compute.context import ContextDataElement, ContextInput
from coffea.compute.data import (
    ContextDataGroup,
    ContextDataset,
    ContextFile,
    DataGroup,
    Dataset,
    File,
    FileContextDataGroup,
    InputDataGroup,
    InputDataset,
    OpenROOTFile,
    ROOTFileElement,
    StepContextDataGroup,
    StepContextDataset,
    StepElement,
)
from coffea.compute.func import EventsArray
from coffea.compute.group import GroupedResult
from coffea.compute.protocol import EmptyResult


def _make_test_group() -> DataGroup:
    return DataGroup(
        datasets=[
            Dataset(
                files=[
                    File(path="file1.root", steps=[(0, 100), (100, 200), (200, 300)]),
                    File(path="file2.root", steps=[(0, 100), (100, 200), (200, 300)]),
                ],
                metadata=ContextDataset(dataset_name="singlemuon", cross_section=None),
            ),
            Dataset(
                files=[
                    File(path="file3.root", steps=[(0, 150), (150, 300)]),
                ],
                metadata=ContextDataset(dataset_name="drellyan", cross_section=1.0),
            ),
        ],
        metadata=ContextDataGroup(group_name="my_analysis"),
    )


def test_make_stepiterables():
    group = _make_test_group()

    expected_ez = {
        "file1:0-100": ContextDataElement(
            StepElement((0, 100), "file1.root"),
            StepContextDataGroup("file1.root", "", "singlemuon", None, "my_analysis"),
        ),
        "file1:100-200": ContextDataElement(
            StepElement((100, 200), "file1.root"),
            StepContextDataGroup("file1.root", "", "singlemuon", None, "my_analysis"),
        ),
        "file1:200-300": ContextDataElement(
            StepElement((200, 300), "file1.root"),
            StepContextDataGroup("file1.root", "", "singlemuon", None, "my_analysis"),
        ),
        "file2:0-100": ContextDataElement(
            StepElement((0, 100), "file2.root"),
            StepContextDataGroup("file2.root", "", "singlemuon", None, "my_analysis"),
        ),
        "file2:100-200": ContextDataElement(
            StepElement((100, 200), "file2.root"),
            StepContextDataGroup("file2.root", "", "singlemuon", None, "my_analysis"),
        ),
        "file2:200-300": ContextDataElement(
            StepElement((200, 300), "file2.root"),
            StepContextDataGroup("file2.root", "", "singlemuon", None, "my_analysis"),
        ),
        "file3:0-150": ContextDataElement(
            StepElement((0, 150), "file3.root"),
            StepContextDataGroup("file3.root", "", "drellyan", 1.0, "my_analysis"),
        ),
        "file3:150-300": ContextDataElement(
            StepElement((150, 300), "file3.root"),
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
        ContextDataElement(
            StepElement((0, 150), "file3.root"),
            StepContextDataset("file3.root", "", "drellyan", 1.0),
        ),
        ContextDataElement(
            StepElement((150, 300), "file3.root"),
            StepContextDataset("file3.root", "", "drellyan", 1.0),
        ),
    ]
    assert list(dataset.iter_steps()) == expected

    file = dataset.files[0]
    expected = [
        ContextDataElement(
            StepElement((0, 150), "file3.root"),
            ContextFile("file3.root", ""),
        ),
        ContextDataElement(
            StepElement((150, 300), "file3.root"),
            ContextFile("file3.root", ""),
        ),
    ]
    assert list(file.iter_steps()) == expected


def test_make_fileiterables():
    group = _make_test_group()

    expected = [
        ContextDataElement(
            ROOTFileElement(path="file1.root"),
            FileContextDataGroup("singlemuon", None, "my_analysis"),
        ),
        ContextDataElement(
            ROOTFileElement(path="file2.root"),
            FileContextDataGroup("singlemuon", None, "my_analysis"),
        ),
        ContextDataElement(
            ROOTFileElement(path="file3.root"),
            FileContextDataGroup("drellyan", 1.0, "my_analysis"),
        ),
    ]
    assert list(group.iter_files()) == expected

    input_group = InputDataGroup(
        [
            InputDataset(
                files=["file1.root", "file2.root"],
                metadata=ContextDataset(dataset_name="singlemuon", cross_section=None),
            ),
            InputDataset(
                files=["file3.root"],
                metadata=ContextDataset(dataset_name="drellyan", cross_section=1.0),
            ),
        ],
        metadata=ContextDataGroup(group_name="my_analysis"),
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


def make_up_steps(item: ContextInput[OpenROOTFile, Any]) -> list[File]:
    return [
        File(
            path=item.data.file_path,
            steps=[
                (0, 100),
                (100, 200),
            ],
        )
    ]


def test_prepare(mocker: MockerFixture) -> None:
    mocker.patch("uproot.open", new=fake_uproot_open)

    input_group = InputDataGroup(
        [
            InputDataset(
                files=["file1.root", "file2.root"],
                metadata=ContextDataset(dataset_name="singlemuon", cross_section=None),
            ),
            InputDataset(
                files=["file3.root"],
                metadata=ContextDataset(dataset_name="drellyan", cross_section=1.0),
            ),
        ],
        metadata=ContextDataGroup(group_name="my_analysis"),
    )
    prepare = input_group.map_files_by(
        make_up_steps, grouper=lambda ctx: ctx.dataset_name
    )

    prepared = sum((f() for f in prepare.gen_steps()), EmptyResult())
    assert isinstance(prepared, GroupedResult)

    as_datasets = [
        Dataset(
            files=agg.result,
            metadata=ContextDataset(
                dataset_name=name,
                cross_section=None if name == "singlemuon" else 1.0,  # TODO: better way
            ),
        )
        for name, agg in prepared.children.items()
    ]
    assert as_datasets == [
        Dataset(
            files=[
                File(path="file1.root", steps=[(0, 100), (100, 200)]),
                File(path="file2.root", steps=[(0, 100), (100, 200)]),
            ],
            metadata=ContextDataset(dataset_name="singlemuon", cross_section=None),
        ),
        Dataset(
            files=[
                File(path="file3.root", steps=[(0, 100), (100, 200)]),
            ],
            metadata=ContextDataset(dataset_name="drellyan", cross_section=1.0),
        ),
    ]

    # Now we could process the dataset
    class CountStepsProcessor:
        def process(self, events: EventsArray) -> int:
            return len(events)

    computable = as_datasets[0].map_steps(CountStepsProcessor())
    result = sum((f() for f in computable.gen_steps()), 0)
    assert result == 400  # 2 files * 2 steps * 100 events each

    def count_steps_func(chunk: ContextInput[EventsArray, StepContextDataset]) -> int:
        chunk.context.uuid
        return len(chunk.data)

    computable = as_datasets[0].map_steps(count_steps_func)
    result = sum((f() for f in computable.gen_steps()), 0)
    assert result == 400  # 2 files * 2 steps * 100 events

    def count_steps_func2(chunk: ContextInput[EventsArray, StepContextDataset]) -> int:
        chunk.context.uuid
        return len(chunk.data)

    computable = as_datasets[0].map_steps(count_steps_func2)
    result = sum((f() for f in computable.gen_steps()), 0)
    assert result == 400  # 2 files * 2 steps * 100 events
