import dask_awkward as dak
import hist
import hist.dask as dah

from coffea import processor


class NanoEventsProcessor(processor.ProcessorABC):
    def __init__(self, columns=[], canaries=[]):
        self._columns = columns
        self._canaries = canaries

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

        dimuon = dak.combinations(events.Muon, 2)
        # print(events.Muon.behavior)
        # print(dimuon["0"].behavior)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=dak.flatten(muon_pt))
        output["mass"].fill(dataset=dataset, mass=dak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] = dak.sum(dak.num(events.Muon, axis=1))
        output["cutflow"]["%s_mass" % dataset] = dak.sum(dak.num(dimuon, axis=1))

        return output

    def postprocess(self, accumulator):
        return accumulator
