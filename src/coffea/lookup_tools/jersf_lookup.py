from copy import deepcopy

import numpy

from coffea.lookup_tools.lookup_base import lookup_base


def masked_bin_eval(dim1_indices, dimN_bins, dimN_vals):
    dimN_indices = numpy.empty_like(dim1_indices)
    for i in numpy.unique(dim1_indices):
        idx = numpy.where(dim1_indices == i)
        dimN_indices[idx] = numpy.clip(
            numpy.searchsorted(dimN_bins[i], dimN_vals[idx], side="right") - 1,
            0,
            len(dimN_bins[i]) - 2,
        )
    return dimN_indices


class jersf_lookup(lookup_base):
    """
    This class defines a lookup table for jet energy resolution scale factors.
    The uncertainty values can be looked up with a call as follows:
    jersf_lut = jersf_lookup()
    SFs = jersf_lut(JetProperty1=jet.property1,...)
    "SFs" will be of the same shape as the input jet properties.
    The list of required jet properties are given in jersf_lut.signature
    """

    def __init__(self, formula, bins_and_orders, clamps_and_vars, params_and_orders):
        """
        The constructor takes the output of the "convert_jersf_txt_file"
        text file converter, which returns a formula, bins, and values.
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
        self._formula = None
        if formula != "None":
            raise Exception("jet energy resolution scale factors have no formula!")

        for binname in self._dim_order[1:]:
            binsaslists = self._bins[binname].tolist()
            self._bins[binname] = [numpy.array(bins) for bins in binsaslists]

        # get the jit to compile if we've got more than one bin dim
        if len(self._dim_order) > 1:
            masked_bin_eval(
                numpy.array([0]), self._bins[self._dim_order[1]], numpy.array([0.0])
            )

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
        """SFs = f(args)"""
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
        bin_indices = [dim1_indices]
        for binname in self._dim_order[1:]:
            bin_indices.append(
                masked_bin_eval(bin_indices[0], self._bins[binname], bin_vals[binname])
            )
        bin_tuple = tuple(bin_indices)

        # get clamp values and clip the inputs
        eval_values = []
        for eval_name in self._eval_vars:
            clamp_mins = self._eval_clamp_mins[eval_name][bin_tuple]
            clamp_maxs = self._eval_clamp_maxs[eval_name][bin_tuple]
            eval_values.append(numpy.clip(eval_vals[eval_name], clamp_mins, clamp_maxs))

        # get parameter values
        param_values = numpy.stack(
            [
                numpy.atleast_1d(numpy.array(param[bin_tuple]).squeeze())
                for param in self._params
            ],
            axis=1,
        )
        if param_values.shape[2:] == (0,):
            param_values.shape = param_values.shape[:2]
        return param_values

    @property
    def signature(self):
        """the list of all needed jet properties to be passed as kwargs to this lookup"""
        return self._signature

    def __repr__(self):
        out = object.__repr__(self) + "\n"
        out += "binned dims   : %s\n" % (self._dim_order)
        out += "eval vars     : %s\n" % (self._eval_vars)
        out += "return format : %s\n" % (self._param_order)
        out += "formula       : %s\n" % (self._formula_str)
        out += "signature     : (%s)\n" % (",".join(self._signature))
        return out
