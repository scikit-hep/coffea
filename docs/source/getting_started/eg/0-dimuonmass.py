from coffea import processor


class DiMuonMassCalculator(processor.ProcessorABC):
    def process(self, events):
        # not doing anything right now,
        # just want to make sure run-local.py works
        return {}
