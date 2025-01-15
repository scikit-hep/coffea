from copy import deepcopy

import numba
import numpy

from coffea.lookup_tools.lookup_base import lookup_base


# methods for dealing with b-tag SFs
@numba.jit(forceobj=True)
def numba_apply_1d(functions, variables):
    out = numpy.empty(variables.shape)
    for i in range(functions.size):
        out[i] = functions[i](variables[i])
    return out


def numbaize(fstr, varlist):
    """
    Convert function string to numba function
    Supports only simple math for now
    """

    lstr = "lambda {}: {}".format(",".join(varlist), fstr)
    func = eval(lstr, {"log": numpy.log, "sqrt": numpy.sqrt})
    nfunc = numba.njit(func)
    return nfunc


# methods for dealing with b-tag SFs
class dense_evaluated_lookup(lookup_base):
    def __init__(self, values, dims, feval_dim=None):
        super().__init__()
        self._dimension = 0
        whattype = type(dims)
        if whattype == numpy.ndarray:
            self._dimension = 1
        else:
            self._dimension = len(dims)
        if self._dimension == 0:
            raise Exception(f"Could not define dimension for {whattype}")
        self._axes = deepcopy(dims)
        self._feval_dim = None
        vals_are_strings = (
            "string" in values.dtype.name
            or "str" in values.dtype.name
            or "unicode" in values.dtype.name
            or "bytes" in values.dtype.name
        )  # ....
        if not isinstance(values, numpy.ndarray):
            raise TypeError("values is not a numpy array, but %r" % type(values))
        if not vals_are_strings:
            raise Exception("Non-string values passed to dense_evaluated_lookup!")
        if feval_dim is None:
            raise Exception(
                "Evaluation dimensions not specified in dense_evaluated_lookup"
            )
        funcs = numpy.zeros(shape=values.shape, dtype="O")
        for i in range(values.size):
            idx = numpy.unravel_index(i, shape=values.shape)
            funcs[idx] = numbaize(values[idx], ["x"])
        self._values = deepcopy(funcs)
        # TODO: support for multidimensional functions and functions with variables other than 'x'
        if len(feval_dim) > 1:
            raise Exception(
                "lookup_tools.evaluator only accepts 1D functions right now!"
            )
        self._feval_dim = feval_dim[0]

    def _evaluate(self, *args, **kwargs):
        indices = []
        if self._dimension == 1:
            indices.append(
                numpy.clip(
                    numpy.searchsorted(self._axes, args[0], side="right") - 1,
                    0,
                    self._values.shape[0] - 1,
                )
            )
        else:
            for dim in range(self._dimension):
                indices.append(
                    numpy.clip(
                        numpy.searchsorted(self._axes[dim], args[dim], side="right")
                        - 1,
                        0,
                        self._values.shape[len(self._axes) - dim - 1] - 1,
                    )
                )
        indices.reverse()
        return numba_apply_1d(self._values[tuple(indices)], args[self._feval_dim])

    def __repr__(self):
        myrepr = object.__repr__(self) + "\n"
        myrepr += f"{self._dimension} dimensional histogram with axes:\n"
        temp = ""
        if self._dimension == 1:
            temp = f"\t1: {self._axes}\n"
        else:
            temp = f"\t1: {self._axes[0]}\n"
        for idim in range(1, self._dimension):
            temp += f"\t{idim + 1}: {self._axes[idim]}\n"
        myrepr += temp
        return myrepr
