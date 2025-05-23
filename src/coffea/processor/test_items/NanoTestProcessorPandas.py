import awkward as ak

from coffea import processor


class NanoTestProcessorPandas(processor.ProcessorABC):
    def __init__(self, columns=[]):
        self._columns = columns

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = events[ak.any(events.Muon.pt > 20, axis=1)]

        muons = events.Muon[events.Muon.pt > 20]

        muon_pts = ak.pad_none(muons.pt, 2, clip=True)

        output = ak.zip(
            {
                "run": events.run,
                "lumi": events.luminosityBlock,
                "event": events.event,
                "mu1_pt": muon_pts[:, 0],
                "mu2_pt": muon_pts[:, 1],
            }
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
