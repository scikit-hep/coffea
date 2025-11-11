from collections.abc import Sized
from typing import Callable, Protocol

from coffea.compute.protocol import ResultT


class EventsArray(Sized, Protocol):
    """Awkward array of events as constructed by NanoEventsFactory.events()

    TODO: define fully later
    """

    # metadata: dict[str, Any]


EventsFunc = Callable[[EventsArray], ResultT]
"Function that processes an EventsArray and returns a ResultType"


class Processor(Protocol[ResultT]):
    """The Processor protocol is used to represent serializable processing units.

    This should not be subclassed directly, but rather implemented by user-defined
    processor classes.
    """

    def process(self, events: EventsArray) -> ResultT:
        """Process a chunk of events and return a result."""
        ...
