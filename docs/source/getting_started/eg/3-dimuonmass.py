import awkward as ak
import hist

from coffea import processor


class DiMuonMassCalculator(processor.ProcessorABC):
    def process(self, events):
        dataset = events.metadata["dataset"]

        h_nmuons = hist.Hist.new.Integer(0, 10, name="nmuons", label="N Muons").Double()
        h_mass = hist.Hist.new.Reg(60, 60, 120, name="mass", label="mμμ [GeV]").Double()

        muons = events.Muon
        h_nmuons.fill(nmuons=ak.num(muons))
        # 1. get all possible pairs of muons
        dimuons = ak.combinations(muons, 2, fields=["lead", "trail"])
        # 2. only use pairs where the muons are opposite-charges
        dimuons = dimuons[dimuons.lead.charge != dimuons.trail.charge]
        # 3. calculate the mass by combining the 4-momenta
        mass = (dimuons.lead + dimuons.trail).mass
        # 4. flatten calculated mass, allowing for any number of pairs per event
        h_mass.fill(mass=ak.flatten(mass))

        return {
            dataset: {
                "h_nmuons": h_nmuons,
                "h_mass": h_mass,
                "events": len(events),
            }
        }
