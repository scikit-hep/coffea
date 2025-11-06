from abc import ABC, abstractmethod
from typing import Protocol
from collections.abc import Sized

from coffea.compute.protocol import ResultType


class EventsArray(Sized, Protocol):
    "Awkward array of events or similar"

    # metadata: dict[str, Any]


class ProcessorABC(ABC):
    @abstractmethod
    def process(self, events: EventsArray) -> ResultType:
        """Process a chunk of events and return a result."""
        raise NotImplementedError
