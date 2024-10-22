import re

import awkward
import dask_awkward
import numpy

from coffea.lookup_tools.jec_uncertainty_lookup import jec_uncertainty_lookup


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


def split_jec_name(name):
    info = name.split("_")

    # Check for the case of regrouped jes uncertainties
    if "Regrouped" in info[0]:
        info.pop(0)
        if "UncertaintySources" in info:
            subinfo = info[info.index("UncertaintySources") :]
            if len(subinfo) == 4:
                info[-2:] = [subinfo[-2] + "_" + subinfo[-1]]

    # Check for the case when the dataera name contains a _ like "17Nov2017_V6"
    if re.match(r"V[0-9]+", info[2]):
        dataera = info[1] + info[2]
        info.pop(2)
        info[1] = dataera

    if "UncertaintySources" in info:
        lvl = info.pop()
        info[3] = lvl

    if len(info) != 5:
        raise Exception(f"Corrector name {name} is not properly formatted!")

    campaign = _checkConsistency(None, info[0])
    dataera = _checkConsistency(None, info[1])
    datatype = _checkConsistency(None, info[2])
    level = info[3].replace(
        "Uncertainty", "jes"
    )  # use a generic 'jes' for normal uncertainty
    jettype = _checkConsistency(None, info[4])

    return campaign, dataera, datatype, level, jettype


class JetCorrectionUncertainty:
    """
    This class is a columnar implementation of the JetCorrectionUncertainty tool in
    CMSSW and FWLite. It calculates the jet energy scale uncertainty for a corrected jet
    in a given binning.

    It implements the jet energy correction definition specified in the JES Uncertainty TWiki_.

    .. _TWiki: https://twiki.cern.ch/twiki/bin/view/CMS/JECUncertaintySources

    You can use this class as follows::

        jcu = JetCorrectionUncertainty(name1=corrL1,...)
        jetUncs = jcu.getUncertainty(JetParameter1=jet.parameter1,...)

    in which `jetUncs` are the uncertainties, with the same shape as the input parameters.
    In order to see which parameters must be passed to `getUncertainty`, one can do
    `jcu.signature`.

    You construct a JetCorrectionUncertainty by passing in a dict of names and functions.
    Names must be formatted as '<campaign>_<dataera>_<datatype>_<level>_<jettype>'. You
    can use coffea.lookup_tools' `extractor` and `evaluator` to get the functions from
    some input files.
    """

    def __init__(self, **kwargs):
        """
        You construct a JetCorrectionUncertainty by passing in a dict of names and functions.
        Names must be formatted as '<campaign>_<dataera>_<datatype>_<level>_<jettype>'.
        """
        jettype = None
        levels = []
        funcs = []
        datatype = None
        campaign = None
        dataera = None
        for name, func in kwargs.items():
            if not isinstance(func, jec_uncertainty_lookup):
                raise Exception(
                    "{} is a {} and not a jec_uncertainty_lookup!".format(
                        name, type(func)
                    )
                )
            campaign, dataera, datatype, level, jettype = split_jec_name(name)

            levels.append(level)
            funcs.append(func)

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

    @property
    def levels(self):
        """list the different sources of uncertainty"""
        return self._levels

    def __repr__(self):
        out = "campaign   : %s\n" % (self._campaign)
        out += "data era   : %s\n" % (self._dataera)
        out += "data type  : %s\n" % (self._datatype)
        out += "jet type   : %s\n" % (self._jettype)
        out += "levels     : %s\n" % (",".join(self._levels))
        out += "signature  : (%s)\n" % (",".join(self._signature))
        return out

    def getUncertainty(self, **kwargs):
        """
        Returns the set of uncertainties for all input jets for all the levels (== sources)

        Use it like::

            juncs = uncertainty.getUncertainty(JetProperty1=jet.property1,...)
            #'juncs' will be formatted like [('SourceName', [[up_val down_val]_jet1 ... ]), ...]
            #in a zip iterator

        """
        uncs = []
        for i, func in enumerate(self._funcs):
            sig = func.signature
            args = tuple(kwargs[inp] for inp in sig)

            if isinstance(
                args[0], (dask_awkward.Array, awkward.highlevel.Array, numpy.ndarray)
            ):
                uncs.append(
                    func(
                        *args,
                        dask_label=f"{self._campaign}_{self._dataera}_{self._datatype}_{self._levels[i]}_{self._jettype}",
                    )
                )
            else:
                raise Exception("Unknown array library for inputs.")

        return zip(self._levels, uncs)
