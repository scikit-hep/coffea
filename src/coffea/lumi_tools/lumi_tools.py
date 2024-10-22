import json
from functools import partial

import awkward
import dask.delayed
import dask_awkward
import fsspec
import numba
import numpy
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask_awkward.layers import AwkwardTreeReductionLayer
from dask_awkward.lib.core import new_array_object
from numba import types
from numba.typed import Dict


def wrap_get_lumi(runlumis, lumi_index):
    runlumis_or_lz = awkward.typetracer.length_zero_if_typetracer(runlumis).to_numpy()
    wrap_tot_lumi = numpy.zeros((1,))
    LumiData._get_lumi_kernel(
        runlumis_or_lz[:, 0], runlumis_or_lz[:, 1], lumi_index, wrap_tot_lumi
    )
    out = awkward.Array(wrap_tot_lumi)
    if awkward.backend(runlumis) == "typetracer":
        out = awkward.Array(out.layout.to_typetracer(forget_length=True))
    return out


class LumiData:
    r"""Holds per-lumiSection integrated lumi values

    Parameters
    ----------
        lumi_csv : str
            The path to the luminosity csv file to read from. Generally, this is the output file from brilcalc.
        is_inst_lumi: bool, default False
            If True, treats the values read in from `lumi_csv` as average instantaneous luminosities, instead of integrated luminosities.

    The values are extracted from the csv output as returned by brilcalc_, e.g. with a command such as::

        brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
                 -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
                 -u /pb --byls --output-style csv -i Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt > lumi2017.csv

    .. _brilcalc: https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html

    Note that some brilcalc files may be in different units than inverse picobarns, including possibly average instantaneous luminosity.
    You should make sure that you understand the units of the LumiData file you are using before calculating luminosity with this tool.
    If you are using a LumiData file containing avg. inst. luminosity, make sure to set is_inst_lumi=True in the constructor of this class.
    """

    # 2^18 orbits / 40 MHz machine clock / 3564 bunch positions
    seconds_per_lumi_LHC = 2**18 / (40079000 / 3564)

    def __init__(self, lumi_csv, is_inst_lumi=False):
        self._is_inst_lumi = is_inst_lumi
        self._lumidata = numpy.loadtxt(
            lumi_csv,
            delimiter=",",
            usecols=(0, 1, 6, 7),
            converters={
                0: lambda s: s.split(":")[0],
                1: lambda s: s.split(":")[
                    0
                ],  # not sure what lumi:0 means, appears to be always zero (DAQ off before beam dump?)
            },
            encoding="utf-8",
        )
        self.index = None
        self.index_delayed = None

    def get_lumi(self, runlumis):
        """Calculate integrated lumi

        Parameters
        ----------
            runlumis : numpy.ndarray or LumiList
                A 2d numpy array of ``[[run,lumi], [run,lumi], ...]`` or `LumiList` object
                of the lumiSections to integrate over, where `run` is a run number and `lumi` is a
                lumisection number.

        Returns
        -------
            (float) The total integrated luminosity of the runs and lumisections indicated in `runlumis`.
        """
        if self.index is None:
            self.index = Dict.empty(
                key_type=types.Tuple([types.uint32, types.uint32]),
                value_type=types.float64,
            )
            runs = self._lumidata[:, 0].astype("u4")
            lumis = self._lumidata[:, 1].astype("u4")
            # fill self.index
            LumiData._build_lumi_table_kernel(runs, lumis, self._lumidata, self.index)
            # delayed object cache
            self.index_delayed = dask.delayed(self.index)

        if isinstance(runlumis, LumiList):
            runlumis = runlumis.array
        tot_lumi = numpy.zeros((1,), dtype=numpy.dtype("float64"))
        if isinstance(runlumis, dask_awkward.Array):
            lumi_meta = wrap_get_lumi(runlumis._meta, self.index)
            lumi_per_partition = dask_awkward.map_partitions(
                wrap_get_lumi,
                runlumis,
                self.index_delayed,
                label="get_lumi",
                meta=lumi_meta,
            )
            tot_lumi = awkward.sum(lumi_per_partition, keepdims=True)
        else:
            LumiData._get_lumi_kernel(
                runlumis[:, 0], runlumis[:, 1], self.index, tot_lumi
            )
        return (
            tot_lumi[0] * self.seconds_per_lumi_LHC
            if self._is_inst_lumi
            else tot_lumi[0]
        )

    @staticmethod
    @numba.njit(parallel=False, fastmath=False)
    def _build_lumi_table_kernel(runs, lumis, lumidata, index):
        for i in range(len(runs)):
            run = runs[i]
            lumi = lumis[i]
            index[(run, lumi)] = float(lumidata[i, 2])

    @staticmethod
    @numba.njit(parallel=False, fastmath=False)
    def _get_lumi_kernel(runs, lumis, index, tot_lumi):
        ks_done = set()
        for iev in range(len(runs)):
            run = numpy.uint32(runs[iev])
            lumi = numpy.uint32(lumis[iev])
            k = (run, lumi)
            if k not in ks_done:
                ks_done.add(k)
                tot_lumi[0] += index.get(k, 0)


class LumiMask:
    """
    Holds a luminosity mask index, and provides vectorized lookup, retaining only valid (run,lumisection) pairs.

    Parameters
    ----------
        jsonfile : str
            Path the the 'golden json' file or other valid lumiSection database in json format.

    This class parses a CMS lumi json into an efficient valid lumiSection lookup table.
    """

    def __init__(self, jsonfile):
        with fsspec.open(jsonfile) as fin:
            goldenjson = json.load(fin)

        self._masks = {}

        for run, lumilist in goldenjson.items():
            mask = numpy.array(lumilist, dtype=numpy.uint32).flatten()
            mask[::2] -= 1
            self._masks[numpy.uint32(run)] = mask

    def __call__(self, runs, lumis):
        """
        Check pairs of runs and lumis for validity, and produce a mask retaining the valid pairs.

        Parameters
        ----------
            runs : numpy.ndarray or awkward.highlevel.Array or dask_awkward.Array
                Vectorized list of run numbers
            lumis : numpy.ndarray or awkward.highlevel.Array or dask_awkward.Array
                Vectorized list of lumiSection numbers

        Returns
        -------
            mask_out : numpy.ndarray
                An array of dtype `bool` where valid (run, lumi) tuples
                will have their corresponding entry set ``True``.
        """

        def apply(runs, lumis):
            # fill numba typed dict
            _masks = Dict.empty(key_type=types.uint32, value_type=types.uint32[:])
            for k, v in self._masks.items():
                _masks[k] = v

            runs_orig = runs
            if isinstance(runs, awkward.highlevel.Array):
                runs = awkward.to_numpy(
                    awkward.typetracer.length_zero_if_typetracer(runs)
                )
            if isinstance(lumis, awkward.highlevel.Array):
                lumis = awkward.to_numpy(
                    awkward.typetracer.length_zero_if_typetracer(lumis)
                )
            mask_out = numpy.zeros(dtype="bool", shape=runs.shape)
            LumiMask._apply_run_lumi_mask_kernel(_masks, runs, lumis, mask_out)
            if isinstance(runs_orig, awkward.Array):
                mask_out = awkward.Array(mask_out)
            if awkward.backend(runs_orig) == "typetracer":
                mask_out = awkward.Array(
                    mask_out.layout.to_typetracer(forget_length=True)
                )
            return mask_out

        if isinstance(runs, dask_awkward.Array):
            return dask_awkward.map_partitions(apply, runs, lumis)
        else:
            return apply(runs, lumis)

    # This could be run in parallel, but windows does not support it
    @staticmethod
    @numba.njit(parallel=False, fastmath=True)
    def _apply_run_lumi_mask_kernel(masks, runs, lumis, mask_out):
        for iev in numba.prange(len(runs)):
            run = numpy.uint32(runs[iev])
            lumi = numpy.uint32(lumis[iev])

            if run in masks:
                lumimask = masks[run]
                ind = numpy.searchsorted(lumimask, lumi)
                if numpy.mod(ind, 2) == 1:
                    mask_out[iev] = 1


def _wrap_unique(array):
    out = numpy.unique(awkward.typetracer.length_one_if_typetracer(array), axis=0)

    if awkward.backend(array) == "typetracer":
        out = awkward.Array(
            out.layout.to_typetracer(forget_length=True),
            behavior=out.behavior,
            attrs=out.attrs,
        )
    return out


def _lumilist_dak_unique(runs_and_lumis, split_every=8):
    concat_fn = partial(awkward.concatenate, axis=0)

    tree_node_fn = _wrap_unique
    finalize_fn = _wrap_unique

    label = "lumilist-unique"

    token = tokenize(
        runs_and_lumis,
        numpy.unique,
        label,
        numpy.uint64,
        split_every,
    )

    name_tree_node = f"{label}-tree-node-{token}"
    name_finalize = f"{label}-finalize-{token}"

    chunked = dask_awkward.map_partitions(
        _wrap_unique, runs_and_lumis, label="lumilist-unique-chunked"
    )

    trl = AwkwardTreeReductionLayer(
        name=name_finalize,
        name_input=chunked.name,
        npartitions_input=chunked.npartitions,
        concat_func=concat_fn,
        tree_node_func=tree_node_fn,
        finalize_func=finalize_fn,
        split_every=split_every,
        tree_node_name=name_tree_node,
    )

    graph = HighLevelGraph.from_collections(name_finalize, trl, dependencies=(chunked,))

    meta = _wrap_unique(runs_and_lumis._meta)

    return new_array_object(graph, name_finalize, meta=meta, npartitions=1)


class LumiList:
    """Mergeable list of unique (run, lumiSection) values

    This list can be merged with another via ``+=``.

    Parameters
    ----------
        runs : numpy.ndarray, dask_awkward.Array
            Vectorized list of run numbers
        lumis : numpy.ndarray, dask_awkward.Array
            Vectorized list of lumiSection values
        delayed: bool
            Is this LumiList in delayed mode or not.
    """

    def __init__(self, runs=None, lumis=None, delayed=True):
        if (runs is None) != (lumis is None):
            raise ValueError(
                "Both runs and lumis must be provided when given to the constructor of LumiList."
            )

        if delayed and runs is None:
            raise ValueError(
                "You must supply runs and lumis when using LumiList is delayed mode."
            )

        self.array = None
        if not delayed:
            self.array = numpy.zeros(shape=(0, 2))

        if isinstance(runs, dask_awkward.Array) and isinstance(
            lumis, dask_awkward.Array
        ):
            self.array = _lumilist_dak_unique(
                awkward.concatenate([runs[:, None], lumis[:, None]], axis=1)
            )
        else:
            if runs is not None:
                self.array = numpy.unique(numpy.c_[runs, lumis], axis=0)

    def __iadd__(self, other):
        # TODO: re-apply unique? Or wait until end
        if isinstance(other, LumiList):
            if isinstance(self.array, dask_awkward.Array):
                self.array = _lumilist_dak_unique(
                    awkward.concatenate([self.array, other.array], axis=0)
                )
            else:
                self.array = numpy.r_[self.array, other.array]
        else:
            raise ValueError("Expected LumiList object, got %r" % other)
        return self

    def __add__(self, other):
        temp = LumiList(delayed=False)
        temp.array = other.array
        temp += self
        return temp

    def clear(self):
        """Clear current lumi list"""
        if isinstance(self.array, dask_awkward.Array):
            raise RuntimeError("Delayed-mode LumiList cannot be cleared!")
        self.array = numpy.zeros(shape=(0, 2))
