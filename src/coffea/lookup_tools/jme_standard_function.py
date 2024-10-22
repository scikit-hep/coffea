from copy import deepcopy

import awkward
import numpy
from numpy import sqrt  # noqa: F401
from numpy import abs, exp, log, log10  # noqa: F401
from numpy import maximum as max  # noqa: F401
from numpy import minimum as min  # noqa: F401
from numpy import power as pow  # noqa: F401

# we need to import these ahead of time for function parsing
from scipy.special import erf  # noqa: F401

from coffea.lookup_tools.lookup_base import lookup_base


def wrap_formula(fstr, varlist):
    """
    Convert function string to python function
    Supports only simple math for now
    """
    val = fstr
    try:
        val = float(fstr)
        fstr = f"numpy.full_like({varlist[0]},{val:f})"
    except ValueError:
        val = fstr
    lstr = "lambda {}: {}".format(",".join(varlist), fstr)
    func = eval(lstr)
    return func


def masked_bin_eval(dim1_indices, dimN_bins, dimN_vals):
    dimN_indices = numpy.empty_like(dim1_indices)
    dimN_overflows = numpy.empty_like(dim1_indices, dtype=bool)
    for i in numpy.unique(dim1_indices):
        idx = numpy.where(dim1_indices == i)
        dimN_indices[idx] = numpy.clip(
            numpy.searchsorted(dimN_bins[i], dimN_vals[idx], side="right") - 1,
            0,
            len(dimN_bins[i]) - 2,
        )
        dimN_overflows[idx] = (dimN_vals[idx] > numpy.amax(dimN_bins[i])) | (
            dimN_vals[idx] < numpy.amin(dimN_bins[i])
        )
    return dimN_indices, dimN_overflows


# idx_in is a tuple of indices in increasing jaggedness
# idx_out is a list of flat indices
def flatten_idxs(idx_in, jaggedarray):
    """
    This provides a faster way to convert between tuples of
    jagged indices and flat indices in a jagged array's contents
    """
    if len(idx_in) == 0:
        return numpy.array([], dtype=numpy.int)
    idx_out = jaggedarray.starts[idx_in[0]]
    if len(idx_in) == 1:
        pass
    elif len(idx_in) == 2:
        idx_out += idx_in[1]
    else:
        raise Exception("jme_standard_function only works for two binning dimensions!")

    flattened = awkward.flatten(jaggedarray)
    good_idx = idx_out < len(flattened)
    if (~good_idx).any():
        input_idxs = tuple(
            [idx_out[~good_idx]] + [idx_in[i][~good_idx] for i in range(len(idx_in))]
        )
        raise Exception(
            "Calculated invalid index {} for"
            " array with length {}".format(numpy.vstack(input_idxs), len(flattened))
        )

    return idx_out


class jme_standard_function(lookup_base):
    """
    This class defines a lookup table for jet energy corrections and resolutions.
    The JEC and JER values can be looked up with a call as follows:
    jerc_lut = jme_standard_function()
    jercs = jerc_lut(JetProperty1=jet.property1,...)
    "jercs" will be of the same shape as the input jet properties.
    The list of required jet properties are given in jersf_lut.signature
    """

    def __init__(self, formula, bins_and_orders, clamps_and_vars, params_and_orders):
        """
        The constructor takes the output of the "convert_jec(jr)_txt_file"
        text file converter, which returns a formula, bins, and parameter values.
        """
        super().__init__()
        self._dim_order = bins_and_orders[1]
        self._bins = bins_and_orders[0]
        self._eval_vars = clamps_and_vars[2]
        self._eval_clamp_mins = clamps_and_vars[0]
        self._eval_clamp_maxs = clamps_and_vars[1]
        self._param_order = params_and_orders[1]
        self._params = params_and_orders[0]
        self._formula_str = formula
        self._formula = wrap_formula(formula, self._param_order + self._eval_vars)

        for binname in self._dim_order[1:]:
            binsaslists = self._bins[binname].tolist()
            self._bins[binname] = [numpy.array(bins) for bins in binsaslists]

        # get the jit to compile if we've got more than one bin dim
        if len(self._dim_order) > 1:
            masked_bin_eval(
                numpy.array([0, 0]),
                self._bins[self._dim_order[1]],
                numpy.array([0.0, 0.0]),
            )

        # compile the formula
        argsize = len(self._param_order) + len(self._eval_vars)
        some_ones = tuple([50 * numpy.ones(argsize) for i in range(argsize)])
        _ = self._formula(*some_ones)

        self._signature = deepcopy(self._dim_order)
        for eval in self._eval_vars:
            if eval not in self._signature:
                self._signature.append(eval)
        self._dim_args = {self._dim_order[i]: i for i in range(len(self._dim_order))}
        self._eval_args = {}
        for i, argname in enumerate(self._eval_vars):
            self._eval_args[argname] = i + len(self._dim_order)
            if argname in self._dim_args.keys():
                self._eval_args[argname] = self._dim_args[argname]

    def _evaluate(self, *args, **kwargs):
        """jec/jer = f(args)"""
        bin_vals = {
            argname: args[self._dim_args[argname]] for argname in self._dim_order
        }
        eval_vals = {
            argname: args[self._eval_args[argname]] for argname in self._eval_vars
        }

        # lookup the bins that we care about
        dim1_name = self._dim_order[0]
        dim1_indices = numpy.clip(
            numpy.searchsorted(self._bins[dim1_name], bin_vals[dim1_name], side="right")
            - 1,
            0,
            self._bins[dim1_name].size - 2,
        )
        overflows = (bin_vals[dim1_name] > numpy.amax(self._bins[dim1_name])) | (
            bin_vals[dim1_name] < numpy.amin(self._bins[dim1_name])
        )
        bin_indices = [dim1_indices]
        for binname in self._dim_order[1:]:
            dimN_indices, dimN_overflows = masked_bin_eval(
                bin_indices[0], self._bins[binname], bin_vals[binname]
            )
            bin_indices.append(dimN_indices)
            overflows |= dimN_overflows

        bin_tuple = tuple(bin_indices)

        # get clamp values and clip the inputs
        eval_values = []
        for eval_name in self._eval_vars:
            clamp_mins = None
            if len(awkward.flatten(self._eval_clamp_mins[eval_name])) == 1:
                clamp_mins = awkward.flatten(self._eval_clamp_mins[eval_name])[0]
            else:
                clamp_mins = numpy.array(
                    self._eval_clamp_mins[eval_name][bin_tuple]
                ).flatten()

            clamp_maxs = None
            if len(awkward.flatten(self._eval_clamp_maxs[eval_name])) == 1:
                clamp_maxs = awkward.flatten(self._eval_clamp_maxs[eval_name])[0]
            else:
                clamp_maxs = numpy.array(
                    self._eval_clamp_maxs[eval_name][bin_tuple]
                ).flatten()

            eval_values.append(numpy.clip(eval_vals[eval_name], clamp_mins, clamp_maxs))

        # get parameter values
        param_values = []
        if len(self._params) > 0:
            param_values = [
                numpy.array(param[bin_tuple]).flatten() for param in self._params
            ]

        raw_eval = self._formula(*tuple(param_values + eval_values))
        return numpy.where(overflows, numpy.ones_like(raw_eval), raw_eval)

    @property
    def signature(self):
        """the list of all needed jet properties to be passed as kwargs to this lookup"""
        return self._signature

    def __repr__(self):
        out = object.__repr__(self) + "\n"
        out += "binned dims: %s\n" % (self._dim_order)
        out += "eval vars  : %s\n" % (self._eval_vars)
        out += "parameters : %s\n" % (self._param_order)
        out += "formula    : %s\n" % (self._formula_str)
        out += "signature  : (%s)\n" % (",".join(self._signature))
        return out
