import re
from functools import reduce

import awkward
import dask_awkward
import numpy

from coffea.lookup_tools.jme_standard_function import jme_standard_function


def _checkConsistency(against, tocheck):
    if against is None:
        against = tocheck
    else:
        if against != tocheck:
            raise Exception(
                "Corrector for {} is mixed"
                "with correctors for {}!".format(tocheck, against)
            )
    return tocheck


_levelre = re.compile("[L1-7]+")


def _getLevel(levelName):
    matches = _levelre.findall(levelName)
    if len(matches) > 1:
        raise Exception(f"Malformed JEC level name: {levelName}")
    return matches[0]


_level_order = ["L1", "L2", "L3", "L2L3", "L4", "L5", "L6", "L7"]


def _sorting_key(name_and_func):
    this_level = _getLevel(name_and_func[0])
    return _level_order.index(this_level)


class _getCorrectionFn:
    def __init__(self, jec, **kwargs):
        self.jec = jec
        self.kwarg_keys = list(kwargs.keys())

    def __call__(self, *args):
        kwargs = {k: v for k, v in zip(self.kwarg_keys, args)}
        corrs = self.jec.getSubCorrections(**kwargs)
        return reduce(lambda x, y: y * x, corrs, 1.0)


class FactorizedJetCorrector:
    """
    This class is a columnar implementation of the FactorizedJetCorrector tool in
    CMSSW and FWLite. It applies a series of JECs in ascending order as defined by
    '_level_order', and checks for the consistency of input corrections.

    It implements the jet energy correction definition specified in the JEC TWiki_.

    .. _TWiki: https://twiki.cern.ch/twiki/bin/view/CMS/JetEnergyScale

    You can use this class as follows::

        fjc = FactorizedJetCorrector(name1=corrL1,...)
        jetCorrs = fjc.getCorrection(JetParameter1=jet.parameter1,...)

    in which `jetCorrs` are the corrected jet scaled factors, with the same shape as
    the input parameters. In order to see what parameters must be passed to
    `getCorrection()`, one can do `fjc.signature`.

    You construct a FactorizedJetCorrector by passing in a dict of names and functions.
    Names must be formatted as '<campaign>_<dataera>_<datatype>_<level>_<jettype>'. You
    can use coffea.lookup_tools' `extractor` and `evaluator` to get the functions from
    some input files.
    """

    def __init__(self, **kwargs):
        jettype = None
        levels = []
        funcs = []
        datatype = None
        campaign = None
        dataera = None
        for name, func in kwargs.items():
            if not isinstance(func, jme_standard_function):
                raise Exception(
                    "{} is a {} and not a jme_standard_function!".format(
                        name, type(func)
                    )
                )
            info = name.split("_")
            if len(info) > 6 or len(info) < 5:
                raise Exception("Corrector name is not properly formatted!")
            offset = len(info) - 5

            campaign = _checkConsistency(campaign, info[0])
            dataera = _checkConsistency(dataera, info[1])
            datatype = _checkConsistency(datatype, info[2 + offset])
            levels.append(info[3 + offset])
            funcs.append(func)
            jettype = _checkConsistency(jettype, info[4 + offset])

        if campaign is None:
            raise Exception("Unable to determine production campaign of JECs!")
        else:
            self._campaign = campaign

        if dataera is None:
            raise Exception("Unable to determine data era of JECs!")
        else:
            self._dataera = dataera

        if datatype is None:
            raise Exception("Unable to determine if JECs are for MC or Data!")
        else:
            self._datatype = datatype

        if len(levels) == 0:
            raise Exception("No levels provided?")
        else:
            self._levels = levels
            self._funcs = funcs

        if jettype is None:
            raise Exception("Unable to determine type of jet to correct!")
        else:
            self._jettype = jettype

        temp = list(zip(*sorted(zip(self._levels, self._funcs), key=_sorting_key)))
        self._levels = list(temp[0])
        self._funcs = list(temp[1])

        # now we setup the call signature for this factorized JEC
        self._signature = []
        for func in self._funcs:
            sig = func.signature
            for input in sig:
                if input not in self._signature:
                    self._signature.append(input)

    @property
    def signature(self):
        """list the necessary jet properties that must be input to this function"""
        return self._signature

    def __repr__(self):
        out = "campaign   : %s\n" % (self._campaign)
        out += "data era   : %s\n" % (self._dataera)
        out += "data type  : %s\n" % (self._datatype)
        out += "jet type   : %s\n" % (self._jettype)
        out += "levels     : %s\n" % (",".join(self._levels))
        out += "signature  : (%s)\n" % (",".join(self._signature))
        return out

    def getCorrection(self, **kwargs):
        """
        Returns the set of corrections for all input jets at the highest available level

        Use it like::

            jecs = corrector.getCorrection(JetProperty1=jet.property1,...)

        """
        first_kwarg = kwargs[list(kwargs.keys())[0]]
        if type(first_kwarg) is dask_awkward.Array:
            levels = "/".join(self._levels)
            func = _getCorrectionFn(self, **kwargs)
            zl_out = func(
                *tuple(
                    awkward.Array(
                        arg._meta.layout.form.length_zero_array(highlevel=False),
                        behavior=arg.behavior,
                    )
                    for arg in kwargs.values()
                )
            )
            meta = awkward.Array(
                zl_out.layout.to_typetracer(forget_length=True),
                behavior=zl_out.behavior,
            )

            return dask_awkward.map_partitions(
                func,
                *tuple(kwargs.values()),
                label=f"{self._campaign}-{self._dataera}-{self._datatype}-{levels}-{self._jettype}",
                meta=meta,
            )
        else:
            corrs = self.getSubCorrections(**kwargs)
            return reduce(lambda x, y: y * x, corrs, 1.0)

    def getSubCorrections(self, **kwargs):
        """
        Returns the set of corrections for all input jets broken down by level

        Use it like::

            jecs = corrector.getSubCorrections(JetProperty1=jet.property1,...)
            #'jecs' will be formatted like [[jec_jet1 jec_jet2 ...] ...]

        """
        # cache = kwargs.pop("lazy_cache", None)
        # form = kwargs.pop("form", None)
        corrVars = {}
        if "JetPt" in kwargs.keys():
            corrVars["JetPt"] = kwargs["JetPt"]
            kwargs.pop("JetPt")
        if "JetE" in kwargs.keys():
            corrVars["JetE"] = kwargs["JetE"]
            kwargs.pop("JetE")
        if len(corrVars) == 0:
            raise Exception("No variable to correct, need JetPt or JetE in inputs!")

        corrections = []
        for i, func in enumerate(self._funcs):
            sig = func.signature
            cumCorr = reduce(lambda x, y: y * x, corrections, 1.0)

            fargs = tuple(
                (cumCorr * corrVars[arg]) if arg in corrVars.keys() else kwargs[arg]
                for arg in sig
            )

            # lookup_base handles dask/awkward/numpy
            if isinstance(
                fargs[0], (dask_awkward.Array, awkward.highlevel.Array, numpy.ndarray)
            ):
                corrections.append(
                    func(
                        *fargs,
                        dask_label=f"{self._campaign}_{self._dataera}_{self._datatype}_{self._levels[i]}_{self._jettype}",
                    )
                )
            else:
                raise Exception("Unknown array library for inputs.")

        return corrections
