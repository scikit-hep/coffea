import dask_awkward as dak
import hist
import hist.dask as dah

from coffea import processor
from coffea.nanoevents.methods import vector


class NanoTestProcessor(processor.ProcessorABC):
    def __init__(self, columns=[]):
        self._columns = columns
        self.expected_usermeta = {
            "ZJets": ("someusermeta", "hello"),
            "Data": ("someusermeta2", "world"),
        }

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        dataset_axis = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        mass_axis = hist.axis.Regular(
            30000, 0.25, 300, name="mass", label=r"$m_{\mu\mu}$ [GeV]"
        )
        pt_axis = hist.axis.Regular(30000, 0.24, 300, name="pt", label=r"$p_{T}$ [GeV]")

        accumulator = {
            # replace when py3.6 is dropped
            # "mass": hist.Hist(dataset_axis, mass_axis, name="Counts"),
            # "pt": hist.Hist(dataset_axis, pt_axis, name="Counts"),
            "mass": dah.Hist(dataset_axis, mass_axis),
            "pt": dah.Hist(dataset_axis, pt_axis),
            "cutflow": {},
        }

        return accumulator

    def process(self, df):
        output = self.accumulator

        dataset = df.metadata["dataset"]
        if "checkusermeta" in df.metadata:
            metaname, metavalue = self.expected_usermeta[dataset]
            assert metavalue == df.metadata[metaname]

        muon = dak.zip(
            {
                "pt": df.Muon_pt,
                "eta": df.Muon_eta,
                "phi": df.Muon_phi,
                "mass": df.Muon_mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        dimuon = dak.combinations(muon, 2)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=dak.flatten(muon.pt))
        output["mass"].fill(dataset=dataset, mass=dak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] = dak.sum(dak.num(muon, axis=1))
        output["cutflow"]["%s_mass" % dataset] = dak.sum(dak.num(dimuon, axis=1))

        return output

    def postprocess(self, accumulator):
        return accumulator
