import awkward as ak
import hist
import hist.dask as dah

from coffea import processor


class NanoEventsProcessor(processor.ProcessorABC):
    def __init__(self, columns=[], canaries=[], mode="dask"):
        self._columns = columns
        self._canaries = canaries
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

    def process(self, events):
        output = self.accumulator

        dataset = events.metadata["dataset"]
        # print(events.metadata)
        if "checkusermeta" in events.metadata:
            metaname, metavalue = self.expected_usermeta[dataset]
            assert metavalue == events.metadata[metaname]

        # mapping = events.behavior["__events_factory__"]._mapping
        muon_pt = events.Muon.pt
        # if isinstance(mapping, nanoevents.mapping.CachedMapping):
        #    keys_in_cache = list(mapping.cache.cache.keys())
        #    has_canaries = [canary in keys_in_cache for canary in self._canaries]
        #    if has_canaries:
        #        try:
        #            from distributed import get_worker
        #
        #            worker = get_worker()
        #            output["worker"].add(worker.name)
        #        except ValueError:
        #            pass

        dimuon = ak.combinations(events.Muon, 2)
        # print(events.Muon.behavior)
        # print(dimuon["0"].behavior)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=ak.flatten(muon_pt))
        output["mass"].fill(dataset=dataset, mass=ak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] = ak.sum(ak.num(events.Muon, axis=1))
        output["cutflow"]["%s_mass" % dataset] = ak.sum(ak.num(dimuon, axis=1))

        return output

    def postprocess(self, accumulator):
        return accumulator
