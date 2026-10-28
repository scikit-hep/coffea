import awkward as ak
import hist

from coffea import processor


class DiMuonMassCalculator(processor.ProcessorABC):
    def process(self, events):
        # the "dataset" metadata is the key in the original fileset
        # we are using ("DYJets" or "Data" in this example)
        dataset = events.metadata["dataset"]

        # number of reconstructed muons in each event
        h_nmuons = hist.Hist.new.Integer(0, 10, name="nmuons", label="N Muons").Double()

        h_nmuons.fill(nmuons=ak.num(events.Muon))

        return {
            dataset: {
                "h_nmuons": h_nmuons,
                "events": len(events),
            }
        }
