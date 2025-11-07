from collections.abc import Sized
from typing import Callable, Generic, Protocol

from uproot import ReadOnlyDirectory

from coffea.compute.protocol import ResultT


class EventsArray(Sized, Protocol):
    "Awkward array of events or similar"

    # metadata: dict[str, Any]


EventsFunc = Callable[[EventsArray], ResultT]
"Function that processes an EventsArray and returns a ResultType"


class Processor(Protocol, Generic[ResultT]):
    def process(self, events: EventsArray) -> ResultT:
        """Process a chunk of events and return a result."""
        ...


DirectoryFunc = Callable[[ReadOnlyDirectory], ResultT]
"Function that processes a uproot directory"
