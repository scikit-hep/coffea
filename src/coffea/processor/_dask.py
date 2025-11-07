from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.live import Live
from rich.progress import Progress

from coffea.util import coffea_console, rich_bar

_processing_sentinel = object()
_final_merge_sentinel = object()


# group of progress bars for dask executor
def pbar_group(datasets: list[str]) -> tuple[Live, dict[Any, Progress]]:
    pbars = {_processing_sentinel: rich_bar()}
    pbars.update({ds: rich_bar() for ds in datasets})
    pbars[_final_merge_sentinel] = rich_bar()
    return Live(Group(*pbars.values()), console=coffea_console), pbars
