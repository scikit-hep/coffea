import awkward as ak
import dask_awkward as dak
import hist
import hist.dask as dah

from coffea import processor
from coffea.nanoevents.methods import vector


class NanoDaskProcessor(processor.ProcessorABC):
    def __init__(self, columns=[], mode="dask"):
        self._columns = columns
        self.mode = mode
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

        if self.mode == "dask":
            mass_hist = dah.Hist(dataset_axis, mass_axis)
            pt_hist = dah.Hist(dataset_axis, pt_axis)
        elif self.mode in ["eager", "virtual"]:
            mass_hist = hist.Hist(dataset_axis, mass_axis)
            pt_hist = hist.Hist(dataset_axis, pt_axis)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        accumulator = {
            "mass": mass_hist,
            "pt": pt_hist,
            "cutflow": {},
            "worker": set(),
        }

        return accumulator

    def process(self, df):
        ak.behavior.update(vector.behavior)

        metadata = df.layout.parameter("metadata")
        dataset = metadata["dataset"]
        output = self.accumulator
        if "checkusermeta" in metadata:
            metaname, metavalue = self.expected_usermeta[dataset]
            assert metavalue == metadata[metaname]

        muon = ak.zip(
            {
                "pt": df.Muon_pt,
                "eta": df.Muon_eta,
                "phi": df.Muon_phi,
                "mass": df.Muon_mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        dimuon = ak.combinations(muon, 2)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=ak.flatten(muon.pt))
        output["mass"].fill(dataset=dataset, mass=ak.flatten(dimuon.mass))

        output["cutflow"]["%s_pt" % dataset] = ak.sum(ak.num(muon, axis=1))
        output["cutflow"]["%s_mass" % dataset] = ak.sum(ak.num(dimuon, axis=1))

        if self.mode == "dask":
            output["skim"][dataset] = dak.to_parquet(
                dimuon, f"test_skim/{dataset}", compute=False
            )
        elif self.mode in ["eager", "virtual"]:
            output["skim"][dataset] = ak.to_parquet(dimuon, f"test_skim/{dataset}")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return output

    def postprocess(self, accumulator):
        return accumulator
