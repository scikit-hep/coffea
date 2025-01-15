from abc import ABCMeta, abstractmethod


class ProcessorABC(metaclass=ABCMeta):
    """ABC for a generalized processor

    The various data delivery mechanisms (spark, striped, uproot, uproot+futures, condor, ...)
    receive such an object and the appropriate metadata to deliver NanoEvents to it.
    It is expected that the entire processor object can be serializable (check with `coffea.util.save`)
    No attempt should be made to track state inside an instance of ProcessorABC, it is to be
    treated simply as a bundle of methods.

    Examples
    --------

    A skeleton processor::

        from coffea import hist, processor

        class MyProcessor(processor.ProcessorABC):
            def __init__(self, flag=False):
                self._flag = flag

            def process(self, events):
                out = {"sumw": ak.num(events, axis=0)}

                # ...

                return {events.metadata.dataset: out}

            def postprocess(self, accumulator):
                pass

        p = MyProcessor()

    """

    @abstractmethod
    def process(self, events):
        """Processes a single NanoEvents chunk

        Returns a filled accumulator object
        """
        pass

    @abstractmethod
    def postprocess(self, accumulator):
        """Final processing on aggregated accumulator

        Do any final processing on the resulting accumulator object, it should be modified in-place
        """
        pass
