from collections.abc import Callable, Sized
from typing import Protocol

from coffea.compute.context import ContextInput, Ctx_co
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


def process_chunk(
    func: Processor[ResultT], chunk: ContextInput[EventsArray, Ctx_co]
) -> ResultT:
    """Toss away the context and just call the processor on the events.

    TODO: this is until we define how Processors handle context properly.
    """

    return func.process(chunk.data)
